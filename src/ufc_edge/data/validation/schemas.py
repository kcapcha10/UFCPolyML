"""Pydantic models for validation findings and the run report."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import Field

from ufc_edge.data.schemas import _FrozenModel


class Violation(_FrozenModel):
    """One invariant violation on one row.

    `event_date` drives era scoping (D12): None means the row could not be
    dated (e.g. an orphan row with no event join) and is treated as loudly as
    a label-universe violation — we cannot prove it is old.
    """

    table_name: str
    row_key: str = Field(description="Human-readable composite key, e.g. 'fight_url|fighter_url'")
    reason_code: str
    detail: str
    event_date: date | None = None


class ValidationReport(_FrozenModel):
    """Aggregate result of one validation run.

    `passed` is False iff any violation is in the label universe
    (event_date >= label_start_date) or undated. Pre-cutoff violations are
    report-only: visible, quarantined, but never blocking (D12).
    """

    ran_at: datetime
    label_start_date: date
    total_violations: int
    label_universe_violations: int
    pre_cutoff_violations: int
    undated_violations: int
    counts_by_reason: dict[str, int]
    passed: bool
