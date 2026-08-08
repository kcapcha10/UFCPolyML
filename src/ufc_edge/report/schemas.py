"""Frozen Pydantic data models for the mismatch report and paper-signal log.

Defines entity-resolution records, paper signals, gate results,
due-diligence verdicts, post-signal snapshots, and scoreboard entries.
All models inherit from _FrozenModel (immutable by default).
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import field_validator

from ufc_edge.data.schemas import _FrozenModel

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class MatchStatus(StrEnum):
    """Entity-resolution outcome for a fight-market pair."""

    MATCHED = "MATCHED"
    NO_CANDIDATE = "NO_CANDIDATE"
    MULTIPLE_CANDIDATES = "MULTIPLE_CANDIDATES"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_SNAPSHOT = "MISSING_SNAPSHOT"


class MatchMethod(StrEnum):
    """How a MATCHED link was established."""

    AUTO_NAME = "AUTO_NAME"
    HUMAN_CONFIRMED = "HUMAN_CONFIRMED"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"


class GateVerdict(StrEnum):
    """Magnitude-gate outcome for a mismatch row."""

    FLAGGED = "FLAGGED"
    WITHIN_NOISE = "WITHIN_NOISE"
    NO_BUCKET_DATA = "NO_BUCKET_DATA"


class DueDiligenceVerdictType(StrEnum):
    """LLM due-diligence conclusion."""

    CONFIRM = "CONFIRM"
    QUALIFY = "QUALIFY"
    VETO = "VETO"


class SnapshotOffset(StrEnum):
    """Scheduled post-signal capture offset."""

    ONE_HOUR = "1H"
    FOUR_HOURS = "4H"
    TWENTY_FOUR_HOURS = "24H"
    FIGHT_TIME = "FIGHT_TIME"


class SnapshotStatus(StrEnum):
    """Outcome of a scheduled post-signal capture."""

    CAPTURED = "CAPTURED"
    MISSED = "MISSED"
    SKIPPED = "SKIPPED"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class MarketFightLink(_FrozenModel):
    """Persisted entity-resolution record joining a Polymarket token to a UFC fight."""

    fight_url: str
    token_id: str
    match_status: MatchStatus
    match_method: MatchMethod | None = None
    candidate_count: int | None = None
    matched_at: datetime
    reviewed_by: str | None = None


class GateResult(_FrozenModel):
    """Magnitude-gate evaluation result with bucket transparency columns."""

    verdict: GateVerdict
    bucket_id: str
    bucket_n: int
    bucket_calibration_error: float
    ci_lower: float
    ci_upper: float


class SparseHistoryResult(_FrozenModel):
    """Sparse-history annotation for a fight."""

    min_prior_ufc_fights: int
    sparse_history: bool


class Finding(_FrozenModel):
    """Single checklist finding from due-diligence research."""

    present: bool
    detail: str
    source_url: str


class ChecklistFindings(_FrozenModel):
    """Structured sub-object of due-diligence checklist categories."""

    injury_news: Finding | None = None
    weight_cut_concern: Finding | None = None
    short_notice_replacement: Finding | None = None
    camp_change: Finding | None = None
    other_material_news: Finding | None = None


class DueDiligenceVerdict(_FrozenModel):
    """Frozen LLM due-diligence verdict with evidence and structured findings."""

    fight_url: str
    report_run_id: str
    verdict: DueDiligenceVerdictType
    confidence: float
    evidence_urls: list[str]
    summary: str
    checklist_findings: ChecklistFindings
    prompt_version: str
    model_name: str
    model_version: str
    invoked_at: datetime

    @field_validator("evidence_urls")
    @classmethod
    def _at_least_one_evidence_url(cls, v: list[str]) -> list[str]:
        """Require at least one evidence URL per verdict."""
        if len(v) < 1:
            msg = "evidence_urls must contain at least one URL"
            raise ValueError(msg)
        return v


class PostSignalSnapshot(_FrozenModel):
    """Scheduled market-state re-capture following a flagged signal."""

    snapshot_id: str
    signal_id: str
    token_id: str
    scheduled_offset: SnapshotOffset
    scheduled_at: datetime
    actual_captured_at: datetime | None = None
    status: SnapshotStatus
    failure_reason: str | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    best_bid_size: float | None = None
    best_ask_size: float | None = None
    mid_price: float | None = None
    captured_at: datetime | None = None


class PaperSignal(_FrozenModel):
    """Immutable record of one bout's model-vs-market comparison at a point in time."""

    signal_id: str
    report_run_id: str
    fight_url: str
    event_date: date | None = None
    fighter_a_url: str
    fighter_b_url: str
    fighter_a_name: str
    fighter_b_name: str
    weight_class: str | None = None
    market_id: str | None = None
    token_id: str | None = None
    snapshot_timestamp: datetime | None = None
    tick_id: str | None = None
    p_model: float | None = None
    p_market_mid: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    best_bid_size: float | None = None
    best_ask_size: float | None = None
    mismatch: float | None = None
    gate_verdict: GateVerdict | None = None
    bucket_id: str | None = None
    bucket_n: int | None = None
    bucket_calibration_error: float | None = None
    bucket_ci_lower: float | None = None
    bucket_ci_upper: float | None = None
    min_prior_ufc_fights: int | None = None
    sparse_history: bool = False
    match_status: MatchStatus
    mlflow_run_id: str
    data_revision: str
    feature_version: str
    config_hash: str
    created_at: datetime


class ReportRun(_FrozenModel):
    """Append-only metadata record identifying one report pipeline execution."""

    report_run_id: str
    as_of_timestamp: datetime
    mlflow_run_id: str
    data_revision: str
    feature_version: str
    config_hash: str
    bout_count: int
    flagged_count: int
    created_at: datetime


class ScoreboardEntry(_FrozenModel):
    """Running tally entry comparing fight outcomes against due-diligence verdicts."""

    fight_url: str
    report_run_id: str
    verdict: DueDiligenceVerdictType
    mismatch_at_signal: float | None = None
    fight_resolved: bool = False
    outcome_correct: bool | None = None
    resolved_at: datetime | None = None
