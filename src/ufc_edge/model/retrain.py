"""Retrain-after-event operational orchestration.

Runs one production retrain cycle after a UFC event is ingested and validated:
it expands the training window to include the newly completed event, refuses to
proceed unless ingestion and validation are clean, logs a fresh non-overwriting
MLflow run through the provenance logger using the locked candidate configuration
(never re-selecting), and advances the active-model pointer only on success.

The model fit itself is injected (``RetrainModelBuilder``) so this module owns
operational orchestration — gating, windowing, provenance, and pointer movement —
rather than XGBoost training, which the trainer and calibration modules already own.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import mlflow
import xgboost as xgb
from mlflow.tracking import MlflowClient

from ufc_edge.data.validation.schemas import ValidationReport
from ufc_edge.eval.calibration import CalibratorProtocol
from ufc_edge.eval.provenance import log_training_run
from ufc_edge.eval.schemas import (
    AblationRungResult,
    EvaluationReport,
    Fold,
    ReliabilityBucket,
)
from ufc_edge.eval.splits import EventEntry
from ufc_edge.model.schemas import AssemblyManifest, CandidateConfig
from ufc_edge.model.train import FoldCandidateResult

_LOGGER = logging.getLogger(__name__)

# Experiment tag naming the latest successful run for downstream report consumption.
_ACTIVE_MODEL_TAG = "active_model_run_id"


class RetrainBlockedError(RuntimeError):
    """Raised when a retrain is blocked by a dirty ingestion or failed validation.

    Retraining on data that failed ingestion or validation would promote a model
    built on quarantined or incomplete inputs, so the cycle refuses to proceed.
    """


@dataclass(frozen=True)
class RetrainArtifacts:
    """Typed outputs of one locked-config fit, ready for provenance logging.

    Groups exactly the artifacts ``log_training_run`` consumes so the injected
    builder returns a single typed object instead of a long positional tuple.
    """

    booster: xgb.Booster
    calibrator: CalibratorProtocol
    folds: list[Fold]
    eval_report: EvaluationReport
    reliability: list[ReliabilityBucket]
    ablation: list[AblationRungResult] | None
    manifest: AssemblyManifest
    candidate_results: Sequence[FoldCandidateResult]


@dataclass(frozen=True)
class RetrainResult:
    """Outcome of a successful retrain: the new run and the window it trained on."""

    run_id: str
    window_event_ids: frozenset[str]


class RetrainModelBuilder(Protocol):
    """Injected fit step producing provenance artifacts for a training window.

    Implementations train the locked configuration on the window and calibrate
    with the locked method; retrain remains agnostic to how that is done.
    """

    def __call__(
        self,
        *,
        window_event_ids: frozenset[str],
        locked_config: CandidateConfig,
        calibrator_method: str,
        feature_version: str,
    ) -> RetrainArtifacts:
        """Return typed artifacts for the locked config fit on the window."""
        ...


# Behavior/params/return: run one gated retrain cycle; return the new run and window.
# Assumes: build_model trains the locked config; config is a resolved mapping.
# Errors: RetrainBlockedError if ingestion/validation are not clean.
def retrain_after_event(
    *,
    new_event_url: str,
    event_index: list[EventEntry],
    locked_config: CandidateConfig,
    calibrator_method: str,
    validation: ValidationReport,
    ingestion_succeeded: bool,
    build_model: RetrainModelBuilder,
    config: Mapping[str, object],
    feature_version: str,
    experiment_name: str,
) -> RetrainResult:
    """Orchestrate a retrain after a newly completed event.

    Parameters:
        new_event_url: URL of the freshly completed event to fold into training.
        event_index: All known events with dates and fight counts.
        locked_config: The candidate configuration selected during development;
            reused verbatim so retrain never re-runs candidate selection.
        calibrator_method: The calibrator method locked during development.
        validation: The post-ingestion validation report for the new event.
        ingestion_succeeded: Whether the event's ingestion completed cleanly.
        build_model: Injected fit step returning provenance-ready artifacts.
        config: Resolved run configuration logged as provenance parameters.
        feature_version: Feature-table version the builder must assemble against.
        experiment_name: MLflow experiment holding runs and the active-model pointer.

    Returns:
        RetrainResult with the new MLflow run ID and the training window.

    Raises:
        RetrainBlockedError: If ingestion or validation did not complete cleanly.
    """
    # Gate first so a blocked cycle never touches MLflow or trains a model.
    _guard_preconditions(ingestion_succeeded=ingestion_succeeded, validation=validation)

    window_event_ids = select_retrain_window(new_event_url, event_index)

    # Bind runs and the pointer to a single experiment.
    mlflow.set_experiment(experiment_name)

    artifacts = build_model(
        window_event_ids=window_event_ids,
        locked_config=locked_config,
        calibrator_method=calibrator_method,
        feature_version=feature_version,
    )

    # A new MLflow run is started inside log_training_run, so existing runs are
    # never mutated or overwritten by a retrain.
    run_id = log_training_run(
        config=config,
        booster=artifacts.booster,
        calibrator=artifacts.calibrator,
        folds=artifacts.folds,
        eval_report=artifacts.eval_report,
        reliability=artifacts.reliability,
        ablation=artifacts.ablation,
        manifest=artifacts.manifest,
        candidate_results=artifacts.candidate_results,
        selected_candidate=locked_config,
    )

    # Advance the pointer only after a fully logged, successful run so a failed or
    # partial run can never become the active model.
    set_active_model(run_id, experiment_name=experiment_name)

    return RetrainResult(run_id=run_id, window_event_ids=window_event_ids)


# Behavior/params/return: chronological window of events up to and including the new one.
# Assumes: dates order events; event_url breaks same-date ties deterministically.
# Errors: ValueError if new_event_url is absent from event_index.
def select_retrain_window(
    new_event_url: str,
    event_index: list[EventEntry],
) -> frozenset[str]:
    """Return the event URLs up to and including the newly completed event."""
    ordered = sorted(event_index, key=lambda entry: (entry.event_date, entry.event_url))
    for position, entry in enumerate(ordered):
        if entry.event_url == new_event_url:
            return frozenset(item.event_url for item in ordered[: position + 1])
    msg = f"new_event_url '{new_event_url}' is not present in the event index"
    raise ValueError(msg)


# Behavior/params: publish the active-model pointer as an experiment tag.
# Assumes: the ambient MLflow tracking URI identifies the target store.
def set_active_model(run_id: str, *, experiment_name: str) -> None:
    """Point the experiment's active-model tag at the given run."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment(experiment_name)
    )
    client.set_experiment_tag(experiment_id, _ACTIVE_MODEL_TAG, run_id)


# Behavior/return: read the active-model run ID, or None when unset.
# Assumes: the ambient MLflow tracking URI identifies the target store.
def get_active_model(*, experiment_name: str) -> str | None:
    """Return the run ID the experiment's active-model pointer names, if any."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    return experiment.tags.get(_ACTIVE_MODEL_TAG)


# Behavior: enforce clean ingestion and validation before promotion.
# Errors: RetrainBlockedError, logged, when either precondition is unmet.
def _guard_preconditions(
    *,
    ingestion_succeeded: bool,
    validation: ValidationReport,
) -> None:
    """Block and log unless ingestion and validation both completed cleanly."""
    if not ingestion_succeeded:
        msg = "Retrain blocked: ingestion did not complete cleanly for the new event."
        _LOGGER.warning(msg)
        raise RetrainBlockedError(msg)
    if not validation.passed:
        msg = (
            "Retrain blocked: validation failed with "
            f"{validation.label_universe_violations} label-universe and "
            f"{validation.undated_violations} undated violation(s)."
        )
        _LOGGER.warning(msg)
        raise RetrainBlockedError(msg)
