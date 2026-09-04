"""MLflow logging orchestration for reproducible model and evaluation runs.

This module receives typed outputs from the model/evaluation pipeline and records
one auditable MLflow run. It does not compute metrics or make model-selection
decisions; its responsibility is serialization, artifact logging, and run status.
"""

from __future__ import annotations

import importlib.metadata
import logging
import tempfile
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path

import joblib
import mlflow
import xgboost as xgb
from pydantic import BaseModel

from ufc_edge.eval.calibration import CalibratorProtocol
from ufc_edge.eval.schemas import (
    AblationRungResult,
    EvaluationReport,
    Fold,
    ReliabilityBucket,
)
from ufc_edge.model.schemas import AssemblyManifest

_LOGGER = logging.getLogger(__name__)


class ProvenanceLoggingError(RuntimeError):
    """Raised when an MLflow run cannot be logged as a complete provenance unit."""


_ARTIFACT_TYPES = {
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


# Behavior/params/return: convert supported config and Pydantic values into JSON-safe values.
# Assumes: input contains only normal Hydra/Python scalar, mapping, and sequence values.
def _to_jsonable(value: object) -> object:
    """Return a JSON-serializable representation of a provenance value."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "item"):
        return value.item()
    return value


# Behavior/params/return: flatten scalar config leaves for MLflow parameters.
# Assumes: sequence-valued config entries belong in config.json rather than params.
def _scalar_parameters(
    value: Mapping[str, object],
    prefix: str = "",
) -> dict[str, str]:
    """Flatten scalar configuration leaves into MLflow-compatible parameters."""
    parameters: dict[str, str] = {}
    for key, item in value.items():
        parameter_name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            parameters.update(_scalar_parameters(item, parameter_name))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            continue
        else:
            parameters[parameter_name] = str(_to_jsonable(item))
    return parameters


# Behavior/params/return: create the explicit provenance fields required by reporting.
# Assumes: feature identity comes from the typed assembly manifest.
def _provenance_parameters(
    config: Mapping[str, object],
    manifest: AssemblyManifest,
) -> dict[str, str]:
    """Build explicit parameters for package, data, feature, seed, and timing provenance."""
    parameters = _scalar_parameters(config)
    for key, value in list(parameters.items()):
        if "seed" in key.lower():
            parameters.setdefault(key.rsplit(".", maxsplit=1)[-1], value)
    parameters.update(
        {
            "package_version": parameters.get("package_version", _package_version()),
            "data_revision": parameters.get("data_revision", "unknown"),
            "feature_version": manifest.feature_version,
            "feature_source_hash": manifest.feature_source_hash,
            "elapsed_seconds": parameters.get("elapsed_seconds", "unknown"),
        }
    )
    return parameters


# Behavior/params/return: look up the installed package version without requiring
# repository metadata.
# Assumes/errors: an uninstalled editable package is represented as "unknown".
def _package_version() -> str:
    """Return the installed package version used by the run."""
    try:
        return importlib.metadata.version("ufc-edge")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


# Behavior/params/return: serialize temporal fold assignments for artifact logging.
# Assumes: fold IDs and event sets are already validated by the Fold schema.
def _fold_assignments(folds: list[Fold]) -> list[dict[str, object]]:
    """Serialize fold partitions with deterministic event ordering."""
    return [
        {
            "fold_id": fold.fold_id,
            "train_event_ids": sorted(fold.train_event_ids),
            "calibration_event_ids": sorted(fold.calibration_event_ids),
            "test_event_ids": sorted(fold.test_event_ids),
        }
        for fold in folds
    ]


# Behavior/params/return: log the fitted XGBoost booster as a stable artifact directory.
# Assumes/errors: the caller owns the active MLflow run; serialization errors propagate.
def _log_booster(booster: xgb.Booster) -> None:
    """Serialize and log the fitted XGBoost booster."""
    with tempfile.TemporaryDirectory(prefix="ufc-edge-booster-") as directory:
        path = Path(directory) / "booster.json"
        booster.save_model(str(path))
        mlflow.log_artifact(str(path), artifact_path="model")


# Behavior/params/return: log a fitted calibrator as a local binary artifact.
# Assumes/errors: the caller owns the active MLflow run; serialization errors propagate.
def _log_calibrator(calibrator: CalibratorProtocol) -> None:
    """Serialize and log the fitted calibrator without adding a runtime dependency."""
    with tempfile.TemporaryDirectory(prefix="ufc-edge-calibrator-") as directory:
        path = Path(directory) / "calibrator.pkl"
        joblib.dump(calibrator, path)
        mlflow.log_artifact(str(path))


# Behavior/params/return: close a partially logged run as failed without hiding cleanup errors.
# Assumes/errors: MLflow may reject either the failure tag or termination independently.
def _mark_run_failed(run_id: str, error: Exception) -> None:
    """Mark a partially logged run failed and always attempt to terminate it."""
    try:
        mlflow.set_tags(
            {
                "provenance_status": "failed",
                "provenance_error": str(error)[:500],
            },
            synchronous=True,
        )
    except Exception:
        _LOGGER.exception("Failed to tag MLflow run %s as failed", run_id)

    # These operations are deliberately separate: a tag-write failure must not
    # leave the run RUNNING and accidentally eligible for downstream queries.
    try:
        mlflow.end_run(status="FAILED")
    except Exception:
        _LOGGER.exception("Failed to terminate failed MLflow run %s", run_id)


# Behavior/params/return: return a completed MLflow run ID.
# Assumes/errors: config is a resolved Hydra mapping or equivalent mapping; all
# typed outputs are valid.
def log_training_run(
    config: Mapping[str, object],
    booster: xgb.Booster,
    calibrator: CalibratorProtocol,
    folds: list[Fold],
    eval_report: EvaluationReport,
    reliability: list[ReliabilityBucket],
    ablation: list[AblationRungResult] | None,
    manifest: AssemblyManifest,
) -> str:
    """Log a complete model/evaluation run and return its MLflow run ID.

    Parameters:
        config: Resolved Hydra configuration represented as a mapping.
        booster: Fitted XGBoost booster.
        calibrator: Fitted calibrator implementing ``transform`` and ``method``.
        folds: Event-grouped temporal fold assignments.
        eval_report: Typed pooled and per-fold evaluation output.
        reliability: Typed probability-bucket reliability output.
        ablation: Optional typed feature-family ablation output.
        manifest: Typed matrix assembly and feature provenance metadata.

    Raises:
        ProvenanceLoggingError: If any logging operation fails. The MLflow run is
            closed with FAILED status and is not tagged as reportable.
    """
    active_run = mlflow.start_run()
    run_id = active_run.info.run_id

    try:
        parameters = _provenance_parameters(config, manifest)
        mlflow.log_params(parameters, synchronous=True)
        mlflow.set_tags(
            {
                "provenance_status": "in_progress",
                "feature_version": manifest.feature_version,
                "feature_source_hash": manifest.feature_source_hash,
            },
            synchronous=True,
        )

        # Keep each typed output in a separate artifact so consumers can read only
        # the contract they need without reconstructing the full evaluation.
        mlflow.log_dict(_to_jsonable(config), "config.json")
        mlflow.log_dict(manifest.model_dump(mode="json"), "assembly_manifest.json")
        mlflow.log_dict({"folds": _fold_assignments(folds)}, "fold_assignments.json")
        mlflow.log_dict(eval_report.model_dump(mode="json"), "evaluation_report.json")
        mlflow.log_dict(
            {"buckets": [bucket.model_dump(mode="json") for bucket in reliability]},
            "reliability.json",
        )
        mlflow.log_dict(
            {
                "candidates": _to_jsonable(config.get("candidates", [])),
                "fold_metrics": [
                    metric.model_dump(mode="json") for metric in eval_report.fold_metrics
                ],
            },
            "candidate_comparison.json",
        )
        if ablation is not None:
            mlflow.log_dict(
                {"rungs": [rung.model_dump(mode="json") for rung in ablation]},
                "ablation.json",
            )

        _log_booster(booster)
        _log_calibrator(calibrator)

        manifest_entries = [
            {"path": path, "type": artifact_type}
            for path, artifact_type in _ARTIFACT_TYPES.items()
            if path != "ablation.json" or ablation is not None
        ]
        run_manifest = {
            "run_id": run_id,
            "artifacts": manifest_entries,
            "provenance": {
                "package_version": parameters["package_version"],
                "data_revision": parameters["data_revision"],
                "feature_version": parameters["feature_version"],
                "feature_source_hash": parameters["feature_source_hash"],
                "seeds": {key: value for key, value in parameters.items() if "seed" in key.lower()},
            },
        }
        mlflow.log_dict(run_manifest, "run_manifest.json")
        mlflow.set_tag("provenance_status", "complete", synchronous=True)
        mlflow.end_run(status="FINISHED")
        return run_id
    except Exception as error:
        # Do not let a partially logged run look successful to reporting queries.
        # Explicitly closing here also works when an artifact API fails before a
        # context manager gets a chance to translate the exception into FAILED.
        _mark_run_failed(run_id, error)
        raise ProvenanceLoggingError(
            f"MLflow provenance logging failed for run {run_id}: {error}"
        ) from error
