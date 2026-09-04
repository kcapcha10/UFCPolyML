"""Fixture tests for MLflow provenance logging."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import mlflow
import numpy as np
import pytest
import xgboost as xgb
from mlflow.tracking import MlflowClient

from ufc_edge.eval.calibration import BetaCalibrator
from ufc_edge.eval.provenance import ProvenanceLoggingError, log_training_run
from ufc_edge.eval.schemas import (
    AblationRungResult,
    EvaluationReport,
    Fold,
    FoldMetrics,
    ReliabilityBucket,
)
from ufc_edge.model.schemas import AssemblyManifest, CandidateConfig
from ufc_edge.model.train import FoldCandidateResult


@pytest.fixture
def tracking_store(tmp_path: Path) -> Iterator[MlflowClient]:
    """Use an isolated local MLflow store for each provenance test."""
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment("provenance-tests")
    yield MlflowClient(tracking_uri=tracking_uri)
    mlflow.set_experiment(experiment.name)
    mlflow.end_run()


@pytest.fixture
def provenance_inputs() -> dict[str, object]:
    """Build small typed inputs covering every provenance artifact."""
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 1, "verbosity": 0},
        xgb.DMatrix(np.array([[0.0], [1.0], [0.5], [1.5]]), label=[0, 1, 0, 1]),
        num_boost_round=1,
    )
    folds = [
        Fold(
            fold_id=0,
            train_event_ids=frozenset({"event-001"}),
            calibration_event_ids=frozenset({"event-002"}),
            test_event_ids=frozenset({"event-003"}),
        )
    ]
    report = EvaluationReport(
        run_id="temporary-evaluation-id",
        fold_metrics=[
            FoldMetrics(
                fold_id=0,
                n_train_fights=2,
                n_cal_fights=2,
                n_test_fights=2,
                brier=0.2,
                log_loss=0.5,
                ece=0.1,
                calibrator_method="beta",
                all_calibrator_scores={
                    "platt": {"ece": 0.2, "brier": 0.25, "log_loss": 0.6},
                    "isotonic": {"ece": 0.15, "brier": 0.22, "log_loss": 0.55},
                    "beta": {"ece": 0.1, "brier": 0.2, "log_loss": 0.5},
                },
            )
        ],
        pooled_brier=0.2,
        pooled_log_loss=0.5,
        pooled_ece=0.1,
        brier_ci=(0.1, 0.3),
        log_loss_ci=(0.4, 0.6),
        brier_skill=0.2,
        brier_skill_ci=(0.0, 0.4),
        mde=0.1,
        permutation_p=0.2,
        sparse_history_brier=0.25,
        n_fights=2,
        n_excluded=0,
        holdout=False,
        evaluated_at="2026-08-30T12:00:00Z",
    )
    reliability = [
        ReliabilityBucket(
            lower=0.1,
            upper=0.3,
            n_fights=2,
            mean_predicted=0.2,
            observed_win_rate=0.5,
            calibration_error=0.3,
            ci_lower=0.05,
            ci_upper=0.95,
            low_support=True,
        )
    ]
    manifest = AssemblyManifest(
        n_rows=4,
        n_features=1,
        feature_version="v1",
        feature_source_hash="feature-sha256",
        columns=["feature_delta"],
        exclusions={},
        assembled_at="2026-08-30T11:00:00Z",
    )
    ablation = [
        AblationRungResult(
            rung="naive",
            brier=0.25,
            brier_ci=(0.2, 0.3),
        )
    ]
    candidate_configs = [
        CandidateConfig(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=1,
            min_child_weight=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
        )
        for n_estimators in (1, 2, 3)
    ]
    candidate_results = [
        FoldCandidateResult(
            candidate_config=candidate_configs[0],
            fold_id=0,
            calibrated_brier=0.20,
            log_loss=0.60,
            ece=0.20,
        ),
        FoldCandidateResult(
            candidate_config=candidate_configs[0],
            fold_id=1,
            calibrated_brier=0.20,
            log_loss=0.58,
            ece=0.18,
        ),
        FoldCandidateResult(
            candidate_config=candidate_configs[1],
            fold_id=0,
            calibrated_brier=0.20,
            log_loss=0.50,
            ece=0.30,
        ),
        FoldCandidateResult(
            candidate_config=candidate_configs[1],
            fold_id=1,
            calibrated_brier=0.20,
            log_loss=0.52,
            ece=0.28,
        ),
        FoldCandidateResult(
            candidate_config=candidate_configs[2],
            fold_id=0,
            calibrated_brier=0.25,
            log_loss=0.40,
            ece=0.10,
        ),
        FoldCandidateResult(
            candidate_config=candidate_configs[2],
            fold_id=1,
            calibrated_brier=0.24,
            log_loss=0.42,
            ece=0.12,
        ),
    ]

    return {
        "config": {
            "random_seed": 42,
            "feature_version": "v1",
            "data_revision": "data-revision-1",
            "package_version": "0.1.0",
            "elapsed_seconds": 12.5,
            "candidates": [{"n_estimators": 1, "max_depth": 1}],
            "nested": {"calibration_seed": 7},
        },
        "booster": booster,
        "calibrator": BetaCalibrator(a=1.0, b=1.0, c=0.0),
        "folds": folds,
        "eval_report": report,
        "reliability": reliability,
        "ablation": ablation,
        "manifest": manifest,
        "candidate_results": candidate_results,
        "selected_candidate": candidate_configs[1],
    }


def test_successful_run_logs_all_expected_artifacts(
    tracking_store: MlflowClient,
    provenance_inputs: dict[str, object],
) -> None:
    """A successful run is finished and contains every required artifact family."""
    run_id = log_training_run(**provenance_inputs)

    run = tracking_store.get_run(run_id)
    artifact_names = {artifact.path for artifact in tracking_store.list_artifacts(run_id)}

    assert run.info.status == "FINISHED"
    assert artifact_names == {
        "assembly_manifest.json",
        "config.json",
        "fold_assignments.json",
        "evaluation_report.json",
        "reliability.json",
        "candidate_comparison.json",
        "ablation.json",
        "calibrator.pkl",
        "run_manifest.json",
        "model",
    }


def test_run_manifest_lists_artifact_paths_and_types(
    tracking_store: MlflowClient,
    provenance_inputs: dict[str, object],
) -> None:
    """The manifest records path and type for every logged artifact."""
    run_id = log_training_run(**provenance_inputs)

    manifest_path = tracking_store.download_artifacts(run_id, "run_manifest.json")
    manifest = json.loads(Path(manifest_path).read_text())

    assert manifest["run_id"] == run_id
    entries = {entry["path"]: entry["type"] for entry in manifest["artifacts"]}
    assert entries == {
        "config.json": "resolved_configuration",
        "assembly_manifest.json": "assembly_manifest",
        "fold_assignments.json": "fold_assignments",
        "evaluation_report.json": "evaluation_report",
        "reliability.json": "reliability_artifact",
        "candidate_comparison.json": "candidate_comparison",
        "ablation.json": "ablation_artifact",
        "model": "xgboost_model",
        "calibrator.pkl": "calibrator",
        "run_manifest.json": "run_manifest",
    }


def test_typed_artifacts_preserve_evaluation_reliability_and_ablation(
    tracking_store: MlflowClient,
    provenance_inputs: dict[str, object],
) -> None:
    """Typed outputs remain independently readable in their JSON artifacts."""
    run_id = log_training_run(**provenance_inputs)

    evaluation = json.loads(
        Path(tracking_store.download_artifacts(run_id, "evaluation_report.json")).read_text()
    )
    reliability = json.loads(
        Path(tracking_store.download_artifacts(run_id, "reliability.json")).read_text()
    )
    ablation = json.loads(
        Path(tracking_store.download_artifacts(run_id, "ablation.json")).read_text()
    )

    assert evaluation["fold_metrics"][0]["calibrator_method"] == "beta"
    assert reliability["buckets"][0]["low_support"] is True
    assert ablation["rungs"][0]["rung"] == "naive"


def test_candidate_comparison_contains_scores_and_winning_rationale(
    tracking_store: MlflowClient,
    provenance_inputs: dict[str, object],
) -> None:
    """Candidate provenance preserves all scores and the deterministic selection reason."""
    run_id = log_training_run(**provenance_inputs)

    comparison = json.loads(
        Path(tracking_store.download_artifacts(run_id, "candidate_comparison.json")).read_text()
    )

    candidates = comparison["candidates"]
    assert len(candidates) == 3
    assert {entry["candidate_config"]["n_estimators"] for entry in candidates} == {1, 2, 3}

    winner = next(entry for entry in candidates if entry["candidate_config"]["n_estimators"] == 2)
    assert winner["candidate_config"]["n_estimators"] == 2
    assert winner["fold_scores"] == [
        {"fold_id": 0, "calibrated_brier": 0.2, "log_loss": 0.5, "ece": 0.3},
        {"fold_id": 1, "calibrated_brier": 0.2, "log_loss": 0.52, "ece": 0.28},
    ]
    assert winner["mean_scores"] == {
        "mean_calibrated_brier": 0.2,
        "mean_log_loss": 0.51,
        "mean_ece": 0.29000000000000004,
    }

    selection = comparison["selection"]
    assert selection["primary_metric"] == "mean_calibrated_brier"
    assert selection["tie_breakers"] == ["mean_log_loss", "mean_ece"]
    assert selection["winner"]["n_estimators"] == 2
    assert selection["winner_rationale"]["criterion_order"] == [
        "mean_calibrated_brier",
        "mean_log_loss",
        "mean_ece",
    ]
    assert "ties are resolved" in selection["winner_rationale"]["explanation"]


def test_artifact_failure_marks_run_failed_and_not_reportable(
    monkeypatch: pytest.MonkeyPatch,
    tracking_store: MlflowClient,
    provenance_inputs: dict[str, object],
) -> None:
    """A write failure closes the run as failed and never marks it complete."""
    original_log_dict = mlflow.log_dict

    def fail_evaluation_report(
        dictionary: dict[str, object], artifact_file: str, run_id: str | None = None
    ) -> None:
        if artifact_file == "evaluation_report.json":
            raise OSError("fixture artifact-store failure")
        original_log_dict(dictionary, artifact_file, run_id=run_id)

    monkeypatch.setattr(mlflow, "log_dict", fail_evaluation_report)

    with pytest.raises(ProvenanceLoggingError, match="artifact-store failure"):
        log_training_run(**provenance_inputs)

    assert mlflow.active_run() is None
    failed_runs = tracking_store.search_runs(
        experiment_ids=[tracking_store.get_experiment_by_name("provenance-tests").experiment_id],
        filter_string="attributes.status = 'FAILED'",
    )
    assert len(failed_runs) == 1
    assert failed_runs[0].data.tags["provenance_status"] == "failed"
    assert "run_manifest.json" not in {
        artifact.path for artifact in tracking_store.list_artifacts(failed_runs[0].info.run_id)
    }
    assert (
        tracking_store.search_runs(
            experiment_ids=[failed_runs[0].info.experiment_id],
            filter_string="tags.provenance_status = 'complete'",
        )
        == []
    )


def test_finalization_failure_does_not_publish_complete_run(
    monkeypatch: pytest.MonkeyPatch,
    tracking_store: MlflowClient,
    provenance_inputs: dict[str, object],
) -> None:
    """A failed finalization is converted to a failed, non-reportable run."""
    original_set_terminated = MlflowClient.set_terminated
    termination_attempts: list[tuple[str, str | None]] = []

    def fail_success_finalization(
        client: MlflowClient,
        run_id: str,
        status: str | None = None,
        end_time: int | None = None,
    ) -> None:
        termination_attempts.append((run_id, status))
        if status == "FINISHED":
            raise OSError("forced finalization failure")
        original_set_terminated(client, run_id, status=status, end_time=end_time)

    monkeypatch.setattr(MlflowClient, "set_terminated", fail_success_finalization)

    with pytest.raises(ProvenanceLoggingError, match="forced finalization failure"):
        log_training_run(**provenance_inputs)

    assert len({run_id for run_id, _ in termination_attempts}) == 1
    experiment_id = tracking_store.get_experiment_by_name("provenance-tests").experiment_id
    failed_runs = tracking_store.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FAILED'",
    )
    assert len(failed_runs) == 1
    assert failed_runs[0].data.tags["provenance_status"] == "failed"
    assert (
        tracking_store.search_runs(
            experiment_ids=[experiment_id],
            filter_string="tags.provenance_status = 'complete'",
        )
        == []
    )


def test_config_seed_and_feature_version_are_logged_as_parameters(
    tracking_store: MlflowClient,
    provenance_inputs: dict[str, object],
) -> None:
    """Resolved config leaves, seeds, and feature provenance are queryable params."""
    run_id = log_training_run(**provenance_inputs)
    params = tracking_store.get_run(run_id).data.params

    assert params["random_seed"] == "42"
    assert params["calibration_seed"] == "7"
    assert params["feature_version"] == "v1"
    assert params["feature_source_hash"] == "feature-sha256"
    assert params["data_revision"] == "data-revision-1"
    assert params["package_version"] == "0.1.0"
    assert params["elapsed_seconds"] == "12.5"
