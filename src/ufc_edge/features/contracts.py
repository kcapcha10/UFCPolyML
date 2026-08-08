"""Feature engine protocol contracts and data models.

Defines the structural contracts that enforce temporal isolation between state
accumulation (StateComponent) and feature emission (FeatureEmitter). EmitContext
is the frozen read-only bridge: it exposes fighter profiles, event metadata, and
component snapshots but never labels, market data, or mutable references.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from datetime import date, datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Frozen Pydantic base (mirrors project convention from data/schemas.py)
# ---------------------------------------------------------------------------


class _FrozenModel(BaseModel):
    """Immutable Pydantic base for all feature-engine domain models."""

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# FrozenState — base class for component snapshot objects
# ---------------------------------------------------------------------------


class FrozenState:
    """Base class for read-only state snapshots returned by StateComponent.freeze().

    Subclasses should use __slots__ or frozen dataclass semantics to prevent
    accidental mutation. This base uses __slots__ to block attribute assignment.
    """

    __slots__: tuple[str, ...] = ()

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            f"Cannot set '{name}' on frozen state object — "
            f"state snapshots are immutable after freeze()"
        )


# ---------------------------------------------------------------------------
# Supporting value objects
# ---------------------------------------------------------------------------


class FighterProfile(_FrozenModel):
    """Static fighter attributes available at fight time.

    Height, reach, stance, and date of birth. All optional since early records
    may lack physical measurements.
    """

    fighter_url: str
    height_cm: float | None = None
    reach_cm: float | None = None
    stance: str | None = None
    dob: date | None = None


class FightTotals(_FrozenModel):
    """Aggregated per-fighter statistics for a single fight.

    Captures striking, grappling, and control-time totals. Fields are optional
    because early UFC events lack detailed statistics.
    """

    knockdowns: int | None = None
    sig_strikes_landed: int | None = None
    sig_strikes_attempted: int | None = None
    total_strikes_landed: int | None = None
    total_strikes_attempted: int | None = None
    takedowns_landed: int | None = None
    takedowns_attempted: int | None = None
    submissions_attempted: int | None = None
    reversals: int | None = None
    control_time_seconds: int | None = None


# ---------------------------------------------------------------------------
# FightOutcomeView — the input to StateComponent.update()
# ---------------------------------------------------------------------------


class FightOutcomeView(_FrozenModel):
    """Immutable view of a fight outcome passed to StateComponent.update().

    Contains only the fight result fields needed for state accumulation.
    Excludes market data, order book data, and any other non-outcome fields.
    """

    fight_url: str
    event_url: str
    event_date: date
    fighter_a_url: str
    fighter_b_url: str
    winner_url: str | None
    method: str
    ending_round: int
    ending_time: str
    weight_class: str
    bout_order: int | None


# ---------------------------------------------------------------------------
# HistoricalFight — the full fight record for replay
# ---------------------------------------------------------------------------


class HistoricalFight(_FrozenModel):
    """Complete fight record loaded for replay.

    Carries both fighter profiles and per-fighter statistics alongside the outcome.
    Used by the HistoricalFightLoader and EventTicker; consumed by both the
    emitter runner (for building EmitContext) and the state updater (via
    FightOutcomeView projection).
    """

    fight_url: str
    event_url: str
    event_date: date
    fighter_a_url: str
    fighter_b_url: str
    winner_url: str | None
    method: str
    ending_round: int
    ending_time: str
    time_format: str
    weight_class: str
    bout_order: int | None
    fighter_a_profile: FighterProfile
    fighter_b_profile: FighterProfile
    fighter_a_totals: FightTotals | None
    fighter_b_totals: FightTotals | None


# ---------------------------------------------------------------------------
# EventTick — one atomic replay step
# ---------------------------------------------------------------------------


class EventTick(_FrozenModel):
    """One atomic replay step: all fights on a single UFC event.

    No fight within a tick may observe another fight's outcome from that same tick.
    Fights are sorted by fight_url for deterministic ordering.
    """

    event_url: str
    event_date: date
    fights: list[HistoricalFight]


# ---------------------------------------------------------------------------
# EmitContext — frozen read-only view passed to emitters
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class EmitContext:
    """Frozen context passed to FeatureEmitter.emit().

    Exposes the focal fighter, opponent, event metadata, and a read-only mapping
    of component names to their frozen snapshots. Structurally cannot expose labels,
    market data, DuckDB connections, or mutable component internals.
    """

    fighter_url: str
    fighter_profile: FighterProfile
    opponent_url: str
    opponent_profile: FighterProfile
    event_date: date
    event_url: str
    weight_class: str
    fight_url: str
    bout_order: int | None
    components: Mapping[str, FrozenState]


# ---------------------------------------------------------------------------
# FeatureRow — one emitted feature record
# ---------------------------------------------------------------------------


class FeatureRow(_FrozenModel):
    """One row in the features_v{N} table.

    Metadata fields are typed strictly; the feature columns live in a flexible
    dict keyed by column name, allowing the registry to define the schema
    without requiring model changes per feature addition.
    """

    fight_url: str
    fighter_url: str
    event_url: str
    event_date: date
    opponent_url: str
    weight_class: str
    feature_version: str
    generated_at: datetime
    features: dict[str, float | str | None]


# ---------------------------------------------------------------------------
# Protocol contracts
# ---------------------------------------------------------------------------


@runtime_checkable
class StateComponent(Protocol):
    """Protocol for state-accumulating components.

    Each implementation tracks one concern (e.g. Elo, career record).
    update() applies a fight outcome; freeze() returns an immutable snapshot.
    """

    def update(self, fight: FightOutcomeView) -> None:
        """Apply one fight outcome to internal state."""
        ...

    def freeze(self) -> FrozenState:
        """Return a deeply-frozen snapshot of current state."""
        ...


@runtime_checkable
class FeatureEmitter(Protocol):
    """Protocol for stateless feature emitters.

    Each emitter reads frozen state from EmitContext and returns a flat dict
    of column_name → scalar value. Emitters must not mutate any state.
    """

    name: str

    def emit(self, context: EmitContext) -> dict[str, float | str | None]:
        """Emit feature values for the focal fighter in the given context."""
        ...
