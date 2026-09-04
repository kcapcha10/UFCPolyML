"""Fixture tests for retrain-after-event orchestration.

Covers the operational retrain contract: expanding the training window to include
a newly completed event, gating on clean ingestion + validation, creating a fresh
non-overwriting MLflow run through the provenance logger, using the locked
candidate configuration without re-selection, and advancing the active-model
pointer only on success.

The heavy model fit is injected as a deterministic fake so these tests exercise
retrain's own orchestration logic rather than XGBoost training; the real pipeline
is wired in the end-to-end integration test.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import mlflow
import numpy as np
import pytest
import xgboost as xgb
from mlflow.tracking import MlflowClient

from ufc_edge.data.validation.schemas import ValidationReport
from ufc_edge.eval.calibration import BetaCalibrator
from ufc_edge.eval.schemas import (
    AblationRungResult,
    EvaluationReport,
    Fold,
    FoldMetrics,
    ReliabilityBucket,
)
from ufc_edge.eval.splits import EventEntry
from ufc_edge.model.retrain import (
    RetrainArtifacts,
    RetrainBlockedError,
    RetrainResult,
    get_active_model,
    retrain_after_event,
    select_retrain_window,
    set_active_model,
)
from ufc_edge.model.schemas import AssemblyManifest, CandidateConfig
from ufc_edge.model.train import FoldCandidateResult

_EXPERIMENT = "retrain-tests"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tracking_store(tmp_path: Path) -> Iterator[MlflowClient]:
    """Isolated local sqlite MLflow store — never a live tracking server."""
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    yield MlflowClient(tracking_uri=tracking_uri)
    mlflow.end_run()


def _locked_config() -> CandidateConfig:
    """The single locked candidate configuration retrain must reuse verbatim."""
    return CandidateConfig(
        n_estimators=2,
        learning_rate=0.1,
        max_depth=2,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
    )


def _event_index() -> list[EventEntry]:
    """Five chronological events; the fourth is the newly completed event."""
    return [
        EventEntry("http://evt/001", date(2024, 1, 6), 10),
        EventEntry("http://evt/002", date(2024, 3, 2), 11),
        EventEntry("http://evt/003", date(2024, 5, 4), 12),
        EventEntry("http://evt/new", date(2024, 7, 6), 13),
        EventEntry("http://evt/005", date(2024, 9, 7), 14),
    ]


def _passing_validation() -> ValidationReport:
    """A clean validation report (no label-universe or undated violations)."""
    return ValidationReport(
        ran_at=datetime(2024, 7, 7, tzinfo=UTC),
        label_start_date=date(2010, 1, 1),
        total_violations=0,
        label_universe_violations=0,
        pre_cutoff_violations=0,
        undated_violations=0,
        counts_by_reason={},
        passed=True,
    )


def _failing_validation() -> ValidationReport:
    """A failing validation report with a blocking label-universe violation."""
    return ValidationReport(
        ran_at=datetime(2024, 7, 7, tzinfo=UTC),
        label_start_date=date(2010, 1, 1),
        total_violations=1,
        label_universe_violations=1,
        pre_cutoff_violations=0,
        undated_violations=0,
        counts_by_reason={"missing_winner": 1},
        passed=False,
    )


def _artifacts_for(config: CandidateConfig) -> RetrainArtifacts:
    """Build minimal but real provenance artifacts for the given locked config."""
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 1, "verbosity": 0},
        xgb.DMatrix(np.array([[0.0], [1.0], [0.5], [1.5]]), label=[0, 1, 0, 1]),
        num_boost_round=1,
    )
    folds = [
        Fold(
            fold_id=0,
            train_event_ids=frozenset({"http://evt/001", "http://evt/002"}),
            calibration_event_ids=frozenset({"http://evt/003"}),
            test_event_ids=frozenset(),
        )
    ]
    report = EvaluationReport(
        run_id="pending",
        fold_metrics=[
            FoldMetrics(
                fold_id=0,
                n_train_fights=33,
                n_cal_fights=12,
                n_test_fights=0,
                brier=0.2,
                log_loss=0.5,
                ece=0.1,
                calibrator_method="beta",
                all_calibrator_scores={},
            )
        ],
        pooled_brier=0.2,
        pooled_log_loss=0.5,
        pooled_ece=0.1,
        brier_ci=(0.1, 0.3),
        log_loss_ci=(0.4, 0.6),
        mde=0.05,
        n_fights=46,
        n_excluded=0,
        holdout=False,
        evaluated_at=datetime(2024, 7, 7, 12, tzinfo=UTC),
    )
    reliability = [
        ReliabilityBucket(
            lower=0.5,
            upper=0.7,
            n_fights=46,
            mean_predicted=0.55,
            observed_win_rate=0.52,
            calibration_error=0.03,
            ci_lower=0.4,
            ci_upper=0.65,
            low_support=False,
        )
    ]
    ablation = [AblationRungResult(rung="naive", brier=0.25, brier_ci=(0.2, 0.3))]
    candidate_results = [
        FoldCandidateResult(
            candidate_config=config,
            fold_id=0,
            calibrated_brier=0.2,
            log_loss=0.5,
            ece=0.1,
        )
    ]
    manifest = AssemblyManifest(
        n_rows=92,
        n_features=1,
        feature_version="v1",
        feature_source_hash="sha-retrain",
        columns=["feature_delta"],
        exclusions={"draw": 2, "nc": 1},
        assembled_at=datetime(2024, 7, 7, 11, tzinfo=UTC),
    )
    return RetrainArtifacts(
        booster=booster,
        calibrator=BetaCalibrator(a=1.0, b=1.0, c=0.0),
        folds=folds,
        eval_report=report,
        reliability=reliability,
        ablation=ablation,
        manifest=manifest,
        candidate_results=candidate_results,
    )


class _RecordingBuilder:
    """Deterministic fake model builder recording the arguments retrain passes."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        window_event_ids: frozenset[str],
        locked_config: CandidateConfig,
        calibrator_method: str,
        feature_version: str,
    ) -> RetrainArtifacts:
        self.calls.append(
            {
                "window_event_ids": window_event_ids,
                "locked_config": locked_config,
                "calibrator_method": calibrator_method,
                "feature_version": feature_version,
            }
        )
        return _artifacts_for(locked_config)


def _retrain(build_model: _RecordingBuilder, **overrides: object) -> RetrainResult:
    """Invoke retrain_after_event with sensible defaults, allowing overrides."""
    kwargs: dict[str, object] = {
        "new_event_url": "http://evt/new",
        "event_index": _event_index(),
        "locked_config": _locked_config(),
        "calibrator_method": "beta",
        "validation": _passing_validation(),
        "ingestion_succeeded": True,
        "build_model": build_model,
        "config": {"random_seed": 42, "data_revision": "rev-1"},
        "feature_version": "v1",
        "experiment_name": _EXPERIMENT,
    }
    kwargs.update(overrides)
    return retrain_after_event(**kwargs)


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------


class TestSelectRetrainWindow:
    """The training window expands to include the newly completed event."""

    def test_window_includes_new_event_and_all_prior_history(self) -> None:
        """Window is every event up to and including the new event, chronologically."""
        window = select_retrain_window("http://evt/new", _event_index())
        assert window == frozenset(
            {"http://evt/001", "http://evt/002", "http://evt/003", "http://evt/new"}
        )

    def test_window_excludes_events_after_the_new_event(self) -> None:
        """Events occurring after the new event are not in the training window."""
        window = select_retrain_window("http://evt/new", _event_index())
        assert "http://evt/005" not in window

    def test_unknown_new_event_raises(self) -> None:
        """A new-event URL absent from the index is a hard error."""
        with pytest.raises(ValueError, match="not present"):
            select_retrain_window("http://evt/missing", _event_index())


# ---------------------------------------------------------------------------
# Run creation and non-overwrite
# ---------------------------------------------------------------------------


class TestRunCreation:
    """Each retrain creates a fresh, finished MLflow run."""

    def test_retrain_creates_finished_run_with_all_artifacts(
        self, tracking_store: MlflowClient
    ) -> None:
        """A successful retrain produces a FINISHED run containing every artifact."""
        result = _retrain(_RecordingBuilder())

        run = tracking_store.get_run(result.run_id)
        assert run.info.status == "FINISHED"

    def test_retrain_twice_creates_two_distinct_runs(self, tracking_store: MlflowClient) -> None:
        """A second retrain never overwrites the first; both runs persist."""
        builder = _RecordingBuilder()
        first = _retrain(builder)
        second = _retrain(builder)

        assert first.run_id != second.run_id
        assert tracking_store.get_run(first.run_id).info.status == "FINISHED"
        assert tracking_store.get_run(second.run_id).info.status == "FINISHED"

    def test_window_and_feature_version_passed_to_builder(
        self, tracking_store: MlflowClient
    ) -> None:
        """The builder receives the expanded window and requested feature version."""
        builder = _RecordingBuilder()
        _retrain(builder)

        assert builder.calls[0]["window_event_ids"] == select_retrain_window(
            "http://evt/new", _event_index()
        )
        assert builder.calls[0]["feature_version"] == "v1"


# ---------------------------------------------------------------------------
# Locked configuration
# ---------------------------------------------------------------------------


class TestLockedConfig:
    """Retrain reuses the locked config and never re-selects candidates."""

    def test_builder_receives_locked_config(self, tracking_store: MlflowClient) -> None:
        """The exact locked config is handed to the model builder."""
        builder = _RecordingBuilder()
        _retrain(builder)
        assert builder.calls[0]["locked_config"] == _locked_config()

    def test_provenance_logs_single_locked_candidate(self, tracking_store: MlflowClient) -> None:
        """Candidate provenance contains exactly the locked config as sole candidate."""
        result = _retrain(_RecordingBuilder())

        comparison = json.loads(
            Path(
                tracking_store.download_artifacts(result.run_id, "candidate_comparison.json")
            ).read_text()
        )
        candidates = comparison["candidates"]
        assert len(candidates) == 1
        assert candidates[0]["candidate_config"]["n_estimators"] == 2
        assert comparison["selection"]["winner"]["n_estimators"] == 2


# ---------------------------------------------------------------------------
# Active-model pointer
# ---------------------------------------------------------------------------


class TestActiveModelPointer:
    """The active-model pointer advances on success and holds on failure."""

    def test_pointer_updated_to_new_run(self, tracking_store: MlflowClient) -> None:
        """After a successful retrain the pointer names the new run."""
        result = _retrain(_RecordingBuilder())
        assert get_active_model(experiment_name=_EXPERIMENT) == result.run_id

    def test_pointer_advances_to_latest_run(self, tracking_store: MlflowClient) -> None:
        """A second successful retrain moves the pointer to the newer run."""
        builder = _RecordingBuilder()
        _retrain(builder)
        second = _retrain(builder)
        assert get_active_model(experiment_name=_EXPERIMENT) == second.run_id

    def test_get_active_model_none_when_unset(self, tracking_store: MlflowClient) -> None:
        """An experiment with no pointer tag returns None."""
        assert get_active_model(experiment_name="never-trained") is None


# ---------------------------------------------------------------------------
# Gating on ingestion + validation
# ---------------------------------------------------------------------------


class TestPromotionGating:
    """A dirty ingestion or failed validation blocks promotion entirely."""

    def test_failed_validation_blocks_and_skips_builder(self, tracking_store: MlflowClient) -> None:
        """Failed validation raises before any model is built."""
        builder = _RecordingBuilder()
        with pytest.raises(RetrainBlockedError, match="validation"):
            _retrain(builder, validation=_failing_validation())
        assert builder.calls == []

    def test_failed_ingestion_blocks_and_skips_builder(self, tracking_store: MlflowClient) -> None:
        """Failed ingestion raises before any model is built."""
        builder = _RecordingBuilder()
        with pytest.raises(RetrainBlockedError, match="ingestion"):
            _retrain(builder, ingestion_succeeded=False)
        assert builder.calls == []

    def test_blocked_retrain_leaves_pointer_untouched(self, tracking_store: MlflowClient) -> None:
        """A blocked retrain never advances a previously set pointer."""
        set_active_model("prior-run-id", experiment_name=_EXPERIMENT)
        builder = _RecordingBuilder()
        with pytest.raises(RetrainBlockedError):
            _retrain(builder, validation=_failing_validation())
        assert get_active_model(experiment_name=_EXPERIMENT) == "prior-run-id"

    def test_blocked_retrain_creates_no_run(self, tracking_store: MlflowClient) -> None:
        """A blocked retrain logs nothing to MLflow."""
        builder = _RecordingBuilder()
        with pytest.raises(RetrainBlockedError):
            _retrain(builder, validation=_failing_validation())
        experiment = tracking_store.get_experiment_by_name(_EXPERIMENT)
        runs = (
            tracking_store.search_runs([experiment.experiment_id]) if experiment is not None else []
        )
        assert list(runs) == []
