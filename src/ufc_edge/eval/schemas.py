"""Pydantic models for evaluation results and artifacts.

Each model is frozen (immutable) and fully typed. They represent the outputs of
the evaluation pipeline: fold definitions, per-fold and pooled metrics, reliability
buckets, ablation results, power analysis, permutation tests, and cumulative
evidence tracking.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import Field

from ufc_edge.data.schemas import _FrozenModel


class Fold(_FrozenModel):
    """One expanding-window temporal fold with event-grouped partitions.

    All fights from a single event stay in the same partition. Train precedes
    calibration precedes test in chronological order.
    """

    fold_id: int = Field(description="Zero-indexed fold identifier")
    train_event_ids: frozenset[str] = Field(
        description="Event URLs assigned to the training partition"
    )
    calibration_event_ids: frozenset[str] = Field(
        description="Event URLs for the trailing calibration slice"
    )
    test_event_ids: frozenset[str] = Field(
        description="Event URLs for the held-out test partition"
    )


class FoldMetrics(_FrozenModel):
    """Per-fold evaluation scores and calibrator selection outcome."""

    fold_id: int = Field(description="Matches the corresponding Fold.fold_id")
    n_train_fights: int
    n_cal_fights: int
    n_test_fights: int
    brier: float
    log_loss: float
    ece: float
    calibrator_method: str = Field(
        description="Winning calibrator for this fold (platt, isotonic, or beta)"
    )
    all_calibrator_scores: dict[str, dict[str, float]] = Field(
        description="Method name → {ece, brier, log_loss} on evaluation slice"
    )


class ReliabilityBucket(_FrozenModel):
    """One probability bucket from the stratified reliability artifact.

    Four fixed buckets are used: [0.1–0.3], [0.3–0.5], [0.5–0.7], [0.7–0.9].
    Binomial CIs use the Clopper-Pearson exact interval at 95% confidence.
    """

    lower: float
    upper: float
    n_fights: int
    mean_predicted: float
    observed_win_rate: float
    calibration_error: float
    ci_lower: float = Field(description="Clopper-Pearson 95% CI lower bound")
    ci_upper: float = Field(description="Clopper-Pearson 95% CI upper bound")
    low_support: bool = Field(
        description="True if n_fights < 10, flagging unreliable estimates"
    )


class AblationRungResult(_FrozenModel):
    """Metrics for one rung of the ablation ladder.

    The ladder progresses: naive → record → +physical → +schedule-strength →
    +domain-interactions. Each rung trains on identical folds; incremental
    improvement is measured vs. the preceding rung.
    """

    rung: str = Field(description="Ablation rung name (e.g. naive, record, physical)")
    brier: float
    brier_ci: tuple[float, float] = Field(
        description="Event-bootstrap 95% CI on Brier score"
    )
    delta_brier: float | None = Field(
        default=None,
        description="Brier improvement vs. prior rung; None for naive (first rung)",
    )
    delta_ci: tuple[float, float] | None = Field(
        default=None,
        description="Event-bootstrap 95% CI on delta; None for naive",
    )
    significant: bool | None = Field(
        default=None,
        description="True if delta CI excludes zero; None for naive",
    )


class PowerResult(_FrozenModel):
    """Pre-evaluation power analysis: minimum detectable effect at given α/power."""

    n_fights: int
    estimated_sigma: float = Field(
        description="Estimated SD of per-fight Brier differences"
    )
    mde: float = Field(
        description="Minimum detectable Brier-skill difference at the given α and power"
    )
    alpha: float
    power: float


class PermutationResult(_FrozenModel):
    """Outcome of a paired permutation test on per-fight Brier differences."""

    observed_diff: float = Field(
        description="Mean of (model_brier_i - market_brier_i) across fights"
    )
    p_value: float
    n_permutations: int
    ci_lower: float = Field(description="Bootstrap CI lower bound on observed diff")
    ci_upper: float = Field(description="Bootstrap CI upper bound on observed diff")


class CumulativePoint(_FrozenModel):
    """One point in the running Brier-skill-vs-market time series.

    Maintained after the holdout period as new events resolve.
    """

    as_of_event_url: str = Field(description="Event URL through which evidence accrues")
    as_of_date: date
    cumulative_n_fights: int
    brier_skill: float
    ci_lower: float = Field(description="Running confidence band lower bound")
    ci_upper: float = Field(description="Running confidence band upper bound")


class EvaluationReport(_FrozenModel):
    """Complete evaluation output: pooled metrics, fold detail, and statistical tests.

    Produced once per train/evaluate cycle and logged to MLflow.
    """

    run_id: str = Field(description="MLflow run ID for provenance tracing")
    fold_metrics: list[FoldMetrics]
    pooled_brier: float
    pooled_log_loss: float
    pooled_ece: float
    brier_ci: tuple[float, float] = Field(
        description="Event-bootstrap 95% CI on pooled Brier"
    )
    log_loss_ci: tuple[float, float] = Field(
        description="Event-bootstrap 95% CI on pooled log loss"
    )
    brier_skill: float | None = Field(
        default=None,
        description="1 - (Brier_model / Brier_market); None if no market data",
    )
    brier_skill_ci: tuple[float, float] | None = Field(
        default=None,
        description="Event-bootstrap CI on Brier skill; None if no market data",
    )
    mde: float = Field(description="Minimum detectable effect at α=0.05, power=0.8")
    permutation_p: float | None = Field(
        default=None,
        description="Paired permutation test p-value; None if no market data",
    )
    sparse_history_brier: float | None = Field(
        default=None,
        description="Brier for fights where min_prior_ufc_fights ≤ 3",
    )
    n_fights: int
    n_excluded: int = Field(
        description="Fights excluded from market-relative metrics (no market match)"
    )
    holdout: bool = Field(description="True if this is the final holdout evaluation")
    evaluated_at: datetime
