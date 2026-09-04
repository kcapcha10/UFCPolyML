"""Final-holdout evaluation with a one-time post-holdout lock.

Scores the reserved 2026 holdout exactly once using the locked candidate config
and the development-selected calibrator method, assembling the full required
metric set (Brier, Brier skill, log loss, ECE, reliability, MDE, paired
permutation p-value, and event-bootstrap CIs) plus history-depth stratification.

On completion it stamps a persistent ``holdout_evaluated_at`` timestamp; any
attempt to re-score the holdout or to change a modeling choice afterward raises
``PostHoldoutLockError``, preserving the integrity of the one-time evaluation.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import numpy as np
from mlflow.tracking import MlflowClient

from ufc_edge.eval.market_relative import MarketFightRow, compute_brier_skill
from ufc_edge.eval.metrics import (
    FightPrediction,
    StratifiedMetrics,
    brier_score,
    event_bootstrap_ci,
    expected_calibration_error,
    log_loss_score,
    stratify_by_history_depth,
)
from ufc_edge.eval.power import minimum_detectable_effect, paired_permutation_test
from ufc_edge.eval.reliability import stratified_reliability
from ufc_edge.eval.schemas import EvaluationReport, ReliabilityBucket
from ufc_edge.model.schemas import CandidateConfig

_LOGGER = logging.getLogger(__name__)

# Experiment tag persisting the one-time holdout evaluation timestamp.
_HOLDOUT_LOCK_TAG = "holdout_evaluated_at"

_DEFAULT_BOOTSTRAP_N = 5000
_DEFAULT_BOOTSTRAP_ALPHA = 0.05
_DEFAULT_PERMUTATION_N = 10_000
_DEFAULT_SEED = 42
_DEFAULT_SPARSE_HISTORY_THRESHOLD = 3


class PostHoldoutLockError(RuntimeError):
    """Raised when the holdout is re-scored or a modeling choice changes post-lock.

    The 2026 holdout is evaluated exactly once. After the lock is stamped, any
    further scoring or model/calibrator/hyperparameter change would compromise
    the integrity of that single final measurement.
    """


class HoldoutLock(Protocol):
    """Persistent one-time lock recording when the holdout was evaluated."""

    def read(self) -> datetime | None:
        """Return the stored evaluation timestamp, or None if unlocked."""
        ...

    def write(self, evaluated_at: datetime) -> None:
        """Persist the evaluation timestamp, engaging the lock."""
        ...


@dataclass(frozen=True)
class MlflowHoldoutLock:
    """Holdout lock backed by an MLflow experiment tag.

    The timestamp is stored on the experiment (not a run) so the lock is a single
    shared fact across every run in that experiment.
    """

    experiment_name: str

    def read(self) -> datetime | None:
        """Read the lock timestamp from the experiment tag, if present."""
        client = MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return None
        raw = experiment.tags.get(_HOLDOUT_LOCK_TAG)
        return datetime.fromisoformat(raw) if raw is not None else None

    def write(self, evaluated_at: datetime) -> None:
        """Stamp the lock timestamp onto the experiment tag."""
        client = MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else client.create_experiment(self.experiment_name)
        )
        client.set_experiment_tag(experiment_id, _HOLDOUT_LOCK_TAG, evaluated_at.isoformat())


@dataclass(frozen=True)
class HoldoutEvaluation:
    """The complete, locked holdout result bundled for reporting and provenance."""

    report: EvaluationReport
    reliability: list[ReliabilityBucket]
    history_depth: StratifiedMetrics
    locked_config: CandidateConfig
    calibrator_method: str
    evaluated_at: datetime


# Behavior/params/return: score the holdout once and engage the lock; return the bundle.
# Assumes: market_rows (when given) align 1:1 with predictions by fight; priors align too.
# Errors: PostHoldoutLockError if the holdout was already evaluated.
def evaluate_holdout(
    *,
    run_id: str,
    predictions: list[FightPrediction],
    market_rows: list[MarketFightRow] | None,
    fighter_a_prior_ufc_fights: list[int],
    fighter_b_prior_ufc_fights: list[int],
    locked_config: CandidateConfig,
    calibrator_method: str,
    lock: HoldoutLock,
    now: datetime | None = None,
    sparse_history_threshold: int = _DEFAULT_SPARSE_HISTORY_THRESHOLD,
    bootstrap_n: int = _DEFAULT_BOOTSTRAP_N,
    bootstrap_alpha: float = _DEFAULT_BOOTSTRAP_ALPHA,
    permutation_n: int = _DEFAULT_PERMUTATION_N,
    seed: int = _DEFAULT_SEED,
) -> HoldoutEvaluation:
    """Evaluate the final holdout exactly once and lock further modeling changes.

    Parameters:
        run_id: MLflow run ID of the holdout run, recorded in the report.
        predictions: Calibrated per-fight model predictions on the holdout set.
        market_rows: Per-fight model/market/outcome rows aligned 1:1 with
            ``predictions``; None or empty skips market-relative metrics.
        fighter_a_prior_ufc_fights: Per-fight prior UFC bout count for fighter A.
        fighter_b_prior_ufc_fights: Per-fight prior UFC bout count for fighter B.
        locked_config: The development-selected candidate configuration.
        calibrator_method: The development-selected calibrator method.
        lock: Persistent one-time holdout lock.
        now: Evaluation timestamp; defaults to the current UTC time.
        sparse_history_threshold: Max prior bouts for the sparse-history stratum.
        bootstrap_n: Event-bootstrap replicate count for confidence intervals.
        bootstrap_alpha: Two-sided significance level for confidence intervals.
        permutation_n: Permutation count for the paired model-vs-market test.
        seed: Seed making the bootstrap and permutation test reproducible.

    Returns:
        HoldoutEvaluation bundling the report, reliability, and stratified metrics.

    Raises:
        PostHoldoutLockError: If the holdout has already been evaluated.
    """
    # Refuse re-scoring before doing any work: the holdout is measured once.
    if lock.read() is not None:
        raise PostHoldoutLockError(
            "The final holdout has already been evaluated; re-scoring is refused."
        )

    evaluated_at = now if now is not None else datetime.now(tz=UTC)

    model_probs = np.array([p.prob for p in predictions], dtype=np.float64)
    labels = np.array([p.label for p in predictions], dtype=np.float64)

    pooled_brier = brier_score(model_probs, labels)
    pooled_log_loss = log_loss_score(model_probs, labels)
    pooled_ece = expected_calibration_error(model_probs, labels)
    brier_ci = event_bootstrap_ci(
        predictions, brier_score, n_bootstrap=bootstrap_n, alpha=bootstrap_alpha, seed=seed
    )
    log_loss_ci = event_bootstrap_ci(
        predictions, log_loss_score, n_bootstrap=bootstrap_n, alpha=bootstrap_alpha, seed=seed
    )

    reliability = stratified_reliability(predictions)
    history_depth = stratify_by_history_depth(
        predictions,
        fighter_a_prior_ufc_fights,
        fighter_b_prior_ufc_fights,
        threshold=sparse_history_threshold,
    )

    market = _market_relative_metrics(
        predictions=predictions,
        market_rows=market_rows,
        model_probs=model_probs,
        labels=labels,
        bootstrap_n=bootstrap_n,
        bootstrap_alpha=bootstrap_alpha,
        permutation_n=permutation_n,
        seed=seed,
    )

    report = EvaluationReport(
        run_id=run_id,
        fold_metrics=[],
        pooled_brier=pooled_brier,
        pooled_log_loss=pooled_log_loss,
        pooled_ece=pooled_ece,
        brier_ci=brier_ci,
        log_loss_ci=log_loss_ci,
        brier_skill=market.brier_skill,
        brier_skill_ci=market.brier_skill_ci,
        mde=market.mde,
        permutation_p=market.permutation_p,
        sparse_history_brier=history_depth.sparse.brier,
        n_fights=len(predictions),
        n_excluded=market.n_excluded,
        holdout=True,
        evaluated_at=evaluated_at,
    )

    # Engage the lock only after the report is fully assembled, so a failure mid
    # computation leaves the holdout re-runnable rather than permanently locked.
    lock.write(evaluated_at)
    _LOGGER.info("Holdout evaluated at %s; modeling choices are now locked.", evaluated_at)

    return HoldoutEvaluation(
        report=report,
        reliability=reliability,
        history_depth=history_depth,
        locked_config=locked_config,
        calibrator_method=calibrator_method,
        evaluated_at=evaluated_at,
    )


# Behavior/params: refuse a post-holdout modeling change once the lock is engaged.
# Errors: PostHoldoutLockError, naming the attempted action, when locked.
def assert_unlocked(lock: HoldoutLock, *, action: str) -> None:
    """Raise if a modeling change is attempted after the holdout is locked."""
    evaluated_at = lock.read()
    if evaluated_at is not None:
        raise PostHoldoutLockError(
            f"'{action}' is refused: the holdout was evaluated at "
            f"{evaluated_at.isoformat()} and all modeling choices are locked."
        )


@dataclass(frozen=True)
class _MarketRelative:
    """Internal bundle of market-relative outputs for report assembly."""

    brier_skill: float | None
    brier_skill_ci: tuple[float, float] | None
    permutation_p: float | None
    mde: float
    n_excluded: int


# Behavior/params/return: compute skill, its CI, the paired test, and the MDE.
# Assumes: market_rows align 1:1 with predictions; empty/None means no market data.
def _market_relative_metrics(
    *,
    predictions: list[FightPrediction],
    market_rows: list[MarketFightRow] | None,
    model_probs: np.ndarray,
    labels: np.ndarray,
    bootstrap_n: int,
    bootstrap_alpha: float,
    permutation_n: int,
    seed: int,
) -> _MarketRelative:
    """Assemble the market-relative metrics, or MDE-only when market data is absent."""
    if not market_rows:
        # No market benchmark: report the model's own detectable-effect floor only.
        sigma = float(np.std((model_probs - labels) ** 2, ddof=1)) if len(labels) > 1 else 0.0
        mde = minimum_detectable_effect(len(predictions), sigma).mde
        return _MarketRelative(
            brier_skill=None,
            brier_skill_ci=None,
            permutation_p=None,
            mde=mde,
            n_excluded=0,
        )

    if len(market_rows) != len(predictions):
        msg = "market_rows must align 1:1 with predictions when provided"
        raise ValueError(msg)

    skill = compute_brier_skill(market_rows)

    matched = [
        (prediction.event_id, row.model_prob, row.market_prob, row.outcome)
        for prediction, row in zip(predictions, market_rows, strict=True)
        if row.market_prob is not None
    ]

    if not matched:
        sigma = float(np.std((model_probs - labels) ** 2, ddof=1)) if len(labels) > 1 else 0.0
        mde = minimum_detectable_effect(len(predictions), sigma).mde
        return _MarketRelative(
            brier_skill=None,
            brier_skill_ci=None,
            permutation_p=None,
            mde=mde,
            n_excluded=skill.n_excluded,
        )

    model_brier_per_fight = np.array([(m - o) ** 2 for _, m, _, o in matched], dtype=np.float64)
    market_brier_per_fight = np.array([(k - o) ** 2 for _, _, k, o in matched], dtype=np.float64)

    diffs = model_brier_per_fight - market_brier_per_fight
    sigma = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    mde = minimum_detectable_effect(len(matched), sigma).mde

    permutation = paired_permutation_test(
        model_brier_per_fight,
        market_brier_per_fight,
        n_permutations=permutation_n,
        seed=seed,
    )

    brier_skill_ci = _brier_skill_bootstrap_ci(
        matched,
        n_bootstrap=bootstrap_n,
        alpha=bootstrap_alpha,
        seed=seed,
    )

    return _MarketRelative(
        brier_skill=skill.brier_skill,
        brier_skill_ci=brier_skill_ci,
        permutation_p=permutation.p_value,
        mde=mde,
        n_excluded=skill.n_excluded,
    )


# Behavior/params/return: event-clustered bootstrap CI on Brier skill.
# Assumes: rows carry event_id, model_prob, market_prob, outcome for matched fights.
def _brier_skill_bootstrap_ci(
    matched: list[tuple[str, float, float, int]],
    *,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    """Resample events with replacement and take the skill percentile interval."""
    events: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    for event_id, model_prob, market_prob, outcome in matched:
        events[event_id].append((model_prob, market_prob, outcome))

    event_ids = list(events.keys())
    n_events = len(event_ids)
    rng = np.random.default_rng(seed)

    skills: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, n_events, size=n_events)
        model_sq: list[float] = []
        market_sq: list[float] = []
        for idx in sampled:
            for model_prob, market_prob, outcome in events[event_ids[idx]]:
                model_sq.append((model_prob - outcome) ** 2)
                market_sq.append((market_prob - outcome) ** 2)
        brier_market = float(np.mean(market_sq))
        # A degenerate (perfect) market replicate has no defined skill ratio; skip it.
        if brier_market == 0.0:
            continue
        skills.append(1.0 - float(np.mean(model_sq)) / brier_market)

    # Every replicate can be skipped when the market is a perfect predictor
    # (brier_market == 0 in all samples); the skill interval is then undefined.
    if not skills:
        return (float("nan"), float("nan"))

    lower_pct = 100.0 * (alpha / 2)
    upper_pct = 100.0 * (1 - alpha / 2)
    return (float(np.percentile(skills, lower_pct)), float(np.percentile(skills, upper_pct)))
