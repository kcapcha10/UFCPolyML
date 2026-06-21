"""Pydantic models for ufcstats.com entities.

Every model corresponds to a DuckDB table. Validate at ingest; fail loudly on
schema drift so silent data corruption cannot accumulate across crawl runs.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import Field

from ufc_edge.data.schemas import _FrozenModel


class Event(_FrozenModel):
    """A UFC event (card)."""

    event_url: str = Field(description="Canonical ufcstats URL — used as primary key")
    name: str
    date: date
    location: str | None = None
    scraped_at: datetime


class Fighter(_FrozenModel):
    """A UFC fighter's biographical data."""

    fighter_url: str = Field(description="Canonical ufcstats URL — used as primary key")
    name: str
    height_cm: float | None = None
    weight_kg: float | None = None
    reach_cm: float | None = None
    stance: str | None = None
    date_of_birth: date | None = None
    scraped_at: datetime


class Fight(_FrozenModel):
    """A single UFC bout."""

    fight_url: str = Field(description="Canonical ufcstats URL — used as primary key")
    event_url: str
    fighter_a_url: str
    fighter_b_url: str
    winner_url: str | None = Field(default=None, description="None indicates a draw or no-contest")
    method: str
    ending_round: int
    ending_time: str = Field(description="MM:SS string e.g. '4:23'")
    time_format: str = Field(description="e.g. '3 Rnd (5-5-5)'")
    referee: str | None = None
    weight_class: str | None = None
    scraped_at: datetime


class FightTotals(_FrozenModel):
    """Aggregate (full-fight) stats for one fighter in one bout."""

    fight_url: str
    fighter_url: str
    knockdowns: int = 0
    significant_strikes_landed: int = 0
    significant_strikes_attempted: int = 0
    total_strikes_landed: int = 0
    total_strikes_attempted: int = 0
    takedowns_landed: int = 0
    takedowns_attempted: int = 0
    submission_attempts: int = 0
    reversals: int = 0
    control_time_seconds: int | None = None
    scraped_at: datetime


class RoundStats(_FrozenModel):
    """Per-round stats for one fighter in one bout."""

    fight_url: str
    fighter_url: str
    round: int
    knockdowns: int = 0
    significant_strikes_landed: int = 0
    significant_strikes_attempted: int = 0
    total_strikes_landed: int = 0
    total_strikes_attempted: int = 0
    takedowns_landed: int = 0
    takedowns_attempted: int = 0
    submission_attempts: int = 0
    reversals: int = 0
    control_time_seconds: int | None = None
    scraped_at: datetime


class SigStrikeBreakdown(_FrozenModel):
    """Significant-strike breakdown by target and distance.

    round=0 is the full-fight aggregate row; round=1..5 is per-round. (0 is used
    rather than NULL because `round` is part of the table's primary key.)
    """

    fight_url: str
    fighter_url: str
    round: int = 0
    head_landed: int = 0
    head_attempted: int = 0
    body_landed: int = 0
    body_attempted: int = 0
    leg_landed: int = 0
    leg_attempted: int = 0
    distance_landed: int = 0
    distance_attempted: int = 0
    clinch_landed: int = 0
    clinch_attempted: int = 0
    ground_landed: int = 0
    ground_attempted: int = 0
    scraped_at: datetime
