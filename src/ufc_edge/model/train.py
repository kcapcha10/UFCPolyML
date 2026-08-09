"""XGBoost trainer and candidate selection.

Trains individual XGBoost candidates with fixed boosting rounds and selects
the best candidate from cross-validated fold results. Boosting rounds are
fixed per candidate rather than early-stopped, so the same held-out data
isn't used both to stop training and to evaluate the result.

This module owns model construction and candidate comparison. It does not
own fold generation, calibration, or evaluation metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import xgboost as xgb

from ufc_edge.model.schemas import CandidateConfig, TrainResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_CANDIDATES = 8
MAX_CANDIDATES = 20

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CandidateCountError(ValueError):
    """Raised when the candidate set size is outside the valid 8–20 range.

    The candidate set must be large enough to cover meaningful hyperparameter
    diversity, but bounded to keep total training time manageable.
    """


# ---------------------------------------------------------------------------
# Fold-level candidate result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldCandidateResult:
    """Metrics for one candidate evaluated on one fold's test set after calibration.

    Captures the three metrics used for candidate ranking: calibrated Brier score,
    log loss, and expected calibration error.
    """

    candidate_config: CandidateConfig
    fold_id: int
    calibrated_brier: float
    log_loss: float
    ece: float


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_candidate(
    X_train: np.ndarray,  # noqa: N803
    y_train: np.ndarray,
    config: CandidateConfig,
    seed: int,
) -> tuple[xgb.Booster, TrainResult]:
    """Train one XGBoost model with fixed boosting rounds.

    Constructs a booster using exactly config.n_estimators rounds with no
    early stopping. The same held-out data is never used both to stop training
    and to evaluate the result — rounds are predetermined by the candidate
    configuration.

    Parameters:
        X_train: Feature matrix (n_samples, n_features).
        y_train: Binary labels array (n_samples,).
        config: Fixed hyperparameter specification for this candidate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (trained Booster, TrainResult metadata).
    """
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "min_child_weight": config.min_child_weight,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "reg_alpha": config.reg_alpha,
        "reg_lambda": config.reg_lambda,
        "seed": seed,
        "verbosity": 0,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, nthread=1)

    start = time.perf_counter()
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=config.n_estimators,
    )
    elapsed = time.perf_counter() - start

    result = TrainResult(
        candidate_config=config,
        fold_id=0,
        random_seed=seed,
        feature_version="",
        data_revision="",
        elapsed_seconds=elapsed,
    )

    return booster, result


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


def select_best_candidate(
    fold_results: list[FoldCandidateResult],
    candidates: list[CandidateConfig],
) -> CandidateConfig:
    """Select the best candidate by mean calibrated Brier across folds.

    Ranks candidates by mean calibrated Brier score (lower is better).
    Ties are broken first by mean log loss, then by mean ECE.

    Parameters:
        fold_results: Per-fold evaluation results for all candidates.
        candidates: The full candidate set; validated for count in [8, 20].

    Returns:
        The CandidateConfig with the best aggregate score.

    Raises:
        CandidateCountError: If candidate count is outside the 8–20 range.
    """
    n_candidates = len(candidates)
    if n_candidates < MIN_CANDIDATES or n_candidates > MAX_CANDIDATES:
        raise CandidateCountError(
            f"Candidate count must be between {MIN_CANDIDATES} and {MAX_CANDIDATES}, "
            f"got {n_candidates}."
        )

    # Group fold results by candidate config.
    candidate_metrics: dict[int, list[FoldCandidateResult]] = {}
    for result in fold_results:
        key = _candidate_key(result.candidate_config)
        candidate_metrics.setdefault(key, []).append(result)

    # Compute mean metrics per candidate and rank.
    scored: list[tuple[float, float, float, CandidateConfig]] = []
    for config in candidates:
        key = _candidate_key(config)
        results = candidate_metrics.get(key, [])
        if not results:
            continue

        mean_brier = np.mean([r.calibrated_brier for r in results])
        mean_log_loss = np.mean([r.log_loss for r in results])
        mean_ece = np.mean([r.ece for r in results])
        scored.append((mean_brier, mean_log_loss, mean_ece, config))

    # Sort ascending by (brier, log_loss, ece) — lower is better for all three.
    scored.sort(key=lambda x: (x[0], x[1], x[2]))

    return scored[0][3]


def _candidate_key(config: CandidateConfig) -> int:
    """Produce a hashable identity for a candidate config.

    Uses a tuple of all hyperparameter values so configs with identical
    settings map to the same key regardless of object identity.
    """
    return hash(
        (
            config.n_estimators,
            config.learning_rate,
            config.max_depth,
            config.min_child_weight,
            config.subsample,
            config.colsample_bytree,
            config.reg_alpha,
            config.reg_lambda,
        )
    )
