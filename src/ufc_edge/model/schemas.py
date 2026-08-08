"""Typed data contracts for the model training layer.

Defines the assembly manifest, candidate hyperparameter configuration, training
result, and the ablation-ladder rung enum. All Pydantic models are frozen to
enforce immutability at external boundaries.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AblationRung(StrEnum):
    """Feature-family ablation ladder rungs in nesting order.

    Each successive rung is a strict superset of the prior rung's columns.
    The naive rung uses no features (constant 0.5 floor).
    """

    naive = "naive"
    record = "record"
    physical = "physical"
    schedule_strength = "schedule_strength"
    domain_interactions = "domain_interactions"


class CandidateConfig(BaseModel):
    """One fixed-hyperparameter XGBoost specification from the candidate set.

    The 8-axis surface covers tree count, learning rate, depth, regularization,
    and column/row sampling. No early stopping is used; n_estimators is the
    exact number of boosting rounds.
    """

    model_config = ConfigDict(frozen=True)

    n_estimators: int = Field(..., gt=0, description="Fixed number of boosting rounds")
    learning_rate: float = Field(..., gt=0.0, le=1.0, description="Step size shrinkage")
    max_depth: int = Field(..., gt=0, le=20, description="Maximum tree depth")
    min_child_weight: float = Field(
        ..., ge=0.0, description="Minimum sum of instance weight in a child"
    )
    subsample: float = Field(
        ..., gt=0.0, le=1.0, description="Row subsample ratio per tree"
    )
    colsample_bytree: float = Field(
        ..., gt=0.0, le=1.0, description="Column subsample ratio per tree"
    )
    reg_alpha: float = Field(..., ge=0.0, description="L1 regularization on weights")
    reg_lambda: float = Field(..., ge=0.0, description="L2 regularization on weights")


class AssemblyManifest(BaseModel):
    """Metadata emitted after matrix assembly completes.

    Records the shape, provenance, and exclusion counts of the assembled
    training matrix so downstream consumers can verify compatibility.
    """

    model_config = ConfigDict(frozen=True)

    n_rows: int = Field(..., ge=0, description="Total training examples (2x fights)")
    n_features: int = Field(..., ge=0, description="Column count in the feature matrix")
    feature_version: str = Field(..., description="Feature table version, e.g. 'v1'")
    feature_source_hash: str = Field(
        ..., description="SHA-256 of the feature package source"
    )
    columns: list[str] = Field(..., description="Ordered column names in the matrix")
    exclusions: dict[str, int] = Field(
        default_factory=dict, description="Reason code -> excluded fight count"
    )
    ablation_rung: str | None = Field(
        None, description="Ablation rung name if subset was applied"
    )
    assembled_at: datetime = Field(
        ..., description="UTC timestamp of assembly completion"
    )


class TrainResult(BaseModel):
    """Result of training a single candidate on one fold.

    Captures the configuration, fold identity, seed, and timing so that
    provenance logging can reconstruct the exact training conditions.
    """

    model_config = ConfigDict(frozen=True)

    candidate_config: CandidateConfig = Field(
        ..., description="Hyperparameter configuration used"
    )
    fold_id: int = Field(..., ge=0, description="Fold identifier")
    random_seed: int = Field(..., description="Random seed used for training")
    feature_version: str = Field(..., description="Feature table version")
    data_revision: str = Field(..., description="Data revision identifier")
    elapsed_seconds: float = Field(
        ..., ge=0.0, description="Wall-clock training time in seconds"
    )
