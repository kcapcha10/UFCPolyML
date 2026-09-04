"""Fixture tests for the locked final-holdout evaluation.

Covers the one-time holdout contract: recording the locked candidate config and
calibrator method, producing the full required metric set (Brier, Brier skill,
log loss, ECE, reliability, MDE, paired permutation p, event-bootstrap CIs,
history-depth stratification), stamping ``holdout_evaluated_at`` on completion,
and refusing both re-scoring and any post-holdout modeling change via
``PostHoldoutLockError``.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from ufc_edge.eval.holdout import (
    HoldoutEvaluation,
    MlflowHoldoutLock,
    PostHoldoutLockError,
    assert_unlocked,
    evaluate_holdout,
)
from ufc_edge.eval.market_relative import MarketFightRow
from ufc_edge.eval.metrics import FightPrediction
from ufc_edge.model.schemas import CandidateConfig

_NOW = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
_SPARSE_EVENTS = 2
_TOTAL_EVENTS = 8
_FIGHTS_PER_EVENT = 5


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class _MemoryLock:
    """Deterministic in-memory holdout lock for fast unit tests."""

    def __init__(self, evaluated_at: datetime | None = None) -> None:
        self._evaluated_at = evaluated_at

    def read(self) -> datetime | None:
        return self._evaluated_at

    def write(self, evaluated_at: datetime) -> None:
        self._evaluated_at = evaluated_at


def _locked_config() -> CandidateConfig:
    """A representative locked configuration recorded by the holdout run."""
    return CandidateConfig(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
    )


def _holdout_inputs() -> dict[str, object]:
    """Build aligned predictions, market rows, and history-depth counts."""
    rng = random.Random(0)
    predictions: list[FightPrediction] = []
    market_rows: list[MarketFightRow] = []
    fighter_a_prior: list[int] = []
    fighter_b_prior: list[int] = []

    for event_idx in range(_TOTAL_EVENTS):
        event_id = f"http://evt/{event_idx:03d}"
        for fight_idx in range(_FIGHTS_PER_EVENT):
            label = fight_idx % 2
            # Model leans toward the correct outcome with mild per-fight variation.
            edge = rng.uniform(0.02, 0.15)
            prob = 0.5 + edge if label == 1 else 0.5 - edge
            predictions.append(FightPrediction(event_id=event_id, prob=prob, label=label))
            # Market sits near a coin flip so the model shows positive skill.
            market_rows.append(MarketFightRow(model_prob=prob, market_prob=0.5, outcome=label))
            # The first events involve sparse-history fighters (min prior <= 3).
            depth = 1 if event_idx < _SPARSE_EVENTS else 10
            fighter_a_prior.append(depth)
            fighter_b_prior.append(depth + 2)

    return {
        "predictions": predictions,
        "market_rows": market_rows,
        "fighter_a_prior_ufc_fights": fighter_a_prior,
        "fighter_b_prior_ufc_fights": fighter_b_prior,
    }


def _evaluate(lock: _MemoryLock, **overrides: object) -> HoldoutEvaluation:
    """Invoke evaluate_holdout with fast, deterministic defaults."""
    inputs = _holdout_inputs()
    kwargs: dict[str, object] = {
        "run_id": "holdout-run-1",
        "predictions": inputs["predictions"],
        "market_rows": inputs["market_rows"],
        "fighter_a_prior_ufc_fights": inputs["fighter_a_prior_ufc_fights"],
        "fighter_b_prior_ufc_fights": inputs["fighter_b_prior_ufc_fights"],
        "locked_config": _locked_config(),
        "calibrator_method": "isotonic",
        "lock": lock,
        "now": _NOW,
        "bootstrap_n": 200,
        "permutation_n": 200,
        "seed": 0,
    }
    kwargs.update(overrides)
    return evaluate_holdout(**kwargs)


@pytest.fixture
def tracking_store(tmp_path: Path) -> Iterator[MlflowClient]:
    """Isolated local sqlite MLflow store — never a live tracking server."""
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    yield MlflowClient(tracking_uri=tracking_uri)
    mlflow.end_run()


# ---------------------------------------------------------------------------
# Required metric outputs
# ---------------------------------------------------------------------------


class TestRequiredMetrics:
    """The holdout report exposes every required metric."""

    def test_absolute_metrics_present(self) -> None:
        """Pooled Brier, log loss, and ECE are finite floats."""
        report = _evaluate(_MemoryLock()).report
        assert report.pooled_brier > 0.0
        assert report.pooled_log_loss > 0.0
        assert report.pooled_ece >= 0.0

    def test_event_bootstrap_cis_present(self) -> None:
        """Event-bootstrap CIs bracket Brier and log loss as ordered tuples."""
        report = _evaluate(_MemoryLock()).report
        assert report.brier_ci[0] <= report.brier_ci[1]
        assert report.log_loss_ci[0] <= report.log_loss_ci[1]

    def test_mde_is_reported(self) -> None:
        """The minimum detectable effect is reported as a finite float."""
        report = _evaluate(_MemoryLock()).report
        assert report.mde >= 0.0

    def test_marks_report_as_holdout(self) -> None:
        """The report flags itself as the final holdout evaluation."""
        report = _evaluate(_MemoryLock()).report
        assert report.holdout is True

    def test_n_fights_matches_prediction_count(self) -> None:
        """The report counts every holdout fight."""
        report = _evaluate(_MemoryLock()).report
        assert report.n_fights == _TOTAL_EVENTS * _FIGHTS_PER_EVENT


# ---------------------------------------------------------------------------
# Market-relative metrics
# ---------------------------------------------------------------------------


class TestMarketRelativeMetrics:
    """Market comparison populates skill and the paired permutation test."""

    def test_brier_skill_present_with_market_data(self) -> None:
        """Positive Brier skill is reported when the model beats a coin-flip market."""
        report = _evaluate(_MemoryLock()).report
        assert report.brier_skill is not None
        assert report.brier_skill > 0.0

    def test_brier_skill_ci_present_with_market_data(self) -> None:
        """A Brier-skill confidence interval accompanies the point estimate."""
        report = _evaluate(_MemoryLock()).report
        assert report.brier_skill_ci is not None
        assert report.brier_skill_ci[0] <= report.brier_skill_ci[1]

    def test_permutation_p_value_present_with_market_data(self) -> None:
        """The paired permutation test yields a p-value in the unit interval."""
        report = _evaluate(_MemoryLock()).report
        assert report.permutation_p is not None
        assert 0.0 <= report.permutation_p <= 1.0

    def test_no_market_data_skips_market_metrics(self) -> None:
        """Without market rows, skill and the paired test are None but MDE stands."""
        report = _evaluate(_MemoryLock(), market_rows=None).report
        assert report.brier_skill is None
        assert report.permutation_p is None
        assert report.mde >= 0.0


# ---------------------------------------------------------------------------
# Reliability and history-depth outputs
# ---------------------------------------------------------------------------


class TestReliabilityAndStratification:
    """Reliability buckets and history-depth strata are included."""

    def test_reliability_has_four_fixed_buckets(self) -> None:
        """The four fixed probability buckets are always produced."""
        evaluation = _evaluate(_MemoryLock())
        assert len(evaluation.reliability) == 4

    def test_reliability_buckets_carry_binomial_cis(self) -> None:
        """Every bucket exposes an ordered binomial confidence interval."""
        evaluation = _evaluate(_MemoryLock())
        assert all(b.ci_lower <= b.ci_upper for b in evaluation.reliability)

    def test_history_depth_strata_present(self) -> None:
        """Sparse and non-sparse strata are both populated from the inputs."""
        evaluation = _evaluate(_MemoryLock())
        assert evaluation.history_depth.sparse.n_fights == _SPARSE_EVENTS * _FIGHTS_PER_EVENT
        assert evaluation.history_depth.non_sparse.n_fights == (
            (_TOTAL_EVENTS - _SPARSE_EVENTS) * _FIGHTS_PER_EVENT
        )

    def test_sparse_history_brier_matches_stratum(self) -> None:
        """The report's sparse Brier equals the sparse stratum's Brier."""
        evaluation = _evaluate(_MemoryLock())
        assert evaluation.report.sparse_history_brier == evaluation.history_depth.sparse.brier


# ---------------------------------------------------------------------------
# Locked config and calibrator provenance
# ---------------------------------------------------------------------------


class TestLockedInputsRecorded:
    """The holdout records the locked config and calibrator method it used."""

    def test_records_locked_config(self) -> None:
        """The locked candidate configuration is preserved on the result."""
        evaluation = _evaluate(_MemoryLock())
        assert evaluation.locked_config == _locked_config()

    def test_records_calibrator_method(self) -> None:
        """The development-selected calibrator method is preserved on the result."""
        evaluation = _evaluate(_MemoryLock())
        assert evaluation.calibrator_method == "isotonic"


# ---------------------------------------------------------------------------
# Locking behavior
# ---------------------------------------------------------------------------


class TestHoldoutLock:
    """Completion stamps the lock and blocks any later modeling change."""

    def test_lock_timestamp_written_on_completion(self) -> None:
        """A successful holdout evaluation stamps the evaluation timestamp."""
        lock = _MemoryLock()
        evaluation = _evaluate(lock)
        assert lock.read() == _NOW
        assert evaluation.evaluated_at == _NOW

    def test_rescore_after_lock_raises(self) -> None:
        """Re-scoring the holdout after the lock is set is refused."""
        lock = _MemoryLock()
        _evaluate(lock)
        with pytest.raises(PostHoldoutLockError, match="already been evaluated"):
            _evaluate(lock)

    def test_assert_unlocked_raises_after_lock(self) -> None:
        """A post-holdout modeling change is refused once the lock is set."""
        lock = _MemoryLock(evaluated_at=_NOW)
        with pytest.raises(PostHoldoutLockError, match="calibrator change"):
            assert_unlocked(lock, action="calibrator change")

    def test_assert_unlocked_passes_before_lock(self) -> None:
        """Modeling changes are permitted while the holdout is unlocked."""
        assert assert_unlocked(_MemoryLock(), action="candidate reselection") is None


# ---------------------------------------------------------------------------
# MLflow-backed lock
# ---------------------------------------------------------------------------


class TestMlflowHoldoutLock:
    """The production lock persists the timestamp as an experiment tag."""

    def test_write_then_read_roundtrips(self, tracking_store: MlflowClient) -> None:
        """A written timestamp reads back identically from the experiment tag."""
        lock = MlflowHoldoutLock(experiment_name="holdout-lock-tests")
        lock.write(_NOW)
        assert lock.read() == _NOW

    def test_read_none_when_unset(self, tracking_store: MlflowClient) -> None:
        """An experiment with no lock tag reads as unlocked."""
        lock = MlflowHoldoutLock(experiment_name="never-evaluated")
        assert lock.read() is None

    def test_evaluate_with_mlflow_lock_blocks_rescore(self, tracking_store: MlflowClient) -> None:
        """Evaluating with the MLflow lock persists the stamp and blocks re-scoring."""
        lock = MlflowHoldoutLock(experiment_name="holdout-lock-tests")
        _evaluate(lock)
        assert lock.read() == _NOW
        with pytest.raises(PostHoldoutLockError):
            _evaluate(lock)
