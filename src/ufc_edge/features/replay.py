"""Replay engine orchestration — event-atomic feature generation.

Implements the EventTicker (groups fights by event) and the main replay loop
that walks UFC history event-by-event with strict emit-before-update semantics.
For every event tick: (1) freeze all state components, (2) emit features for
all fights on the card from frozen state, (3) update accumulators with outcomes.
This guarantees no fight sees its own event's results.

The replay engine is the critical-path component that turns 7 state components
and 12 emitters into the features_v{N} DuckDB table. It owns temporal isolation,
deterministic ordering, emitter output validation, and memory discipline.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import groupby

from ufc_edge.features.contracts import (
    EmitContext,
    EventTick,
    FeatureEmitter,
    FeatureRow,
    FightOutcomeView,
    FrozenState,
    HistoricalFight,
    StateComponent,
)
from ufc_edge.features.registry import FeatureRegistry
from ufc_edge.features.versioning import FEATURE_VERSION

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclass — injected from Hydra/OmegaConf at call site
# ---------------------------------------------------------------------------


class ReplayConfig:
    """Configuration for the replay engine. Injected from Hydra at the call site.

    Attributes:
        log_every_n_events: How often to log progress (every N events).
    """

    __slots__ = ("log_every_n_events",)

    def __init__(self, *, log_every_n_events: int = 50) -> None:
        self.log_every_n_events = log_every_n_events


# ---------------------------------------------------------------------------
# EventTicker — groups fights by event into atomic ticks
# ---------------------------------------------------------------------------


def group_into_ticks(fights: Sequence[HistoricalFight]) -> list[EventTick]:
    """Group chronologically-sorted fights into event-atomic ticks.

    Each tick contains all fights for a single event, sorted by fight_url for
    deterministic ordering within the event. Input MUST already be sorted by
    (event_date, event_url, fight_url) as guaranteed by HistoricalFightLoader.
    """
    ticks: list[EventTick] = []

    for (event_date, event_url), group in groupby(
        fights, key=lambda f: (f.event_date, f.event_url)
    ):
        event_fights = sorted(group, key=lambda f: f.fight_url)
        ticks.append(
            EventTick(
                event_url=event_url,
                event_date=event_date,
                fights=event_fights,
            )
        )

    return ticks


# ---------------------------------------------------------------------------
# Component registry — maps string keys to StateComponent instances
# ---------------------------------------------------------------------------


class ComponentRegistry:
    """Maps component names to their StateComponent instances.

    The replay engine uses these names as keys in EmitContext.components so
    emitters can locate the frozen snapshot they need (e.g. "elo", "career").
    """

    __slots__ = ("_components",)

    def __init__(self, components: dict[str, StateComponent]) -> None:
        self._components = components

    def freeze_all(self) -> dict[str, FrozenState]:
        """Call freeze() on every registered component; return name→snapshot map."""
        return {name: comp.freeze() for name, comp in self._components.items()}

    def update_all(self, outcome: FightOutcomeView) -> None:
        """Call update() on every registered component with the fight outcome."""
        for comp in self._components.values():
            comp.update(outcome)

    def get(self, name: str) -> StateComponent | None:
        """Look up a component by name."""
        return self._components.get(name)

    @property
    def names(self) -> list[str]:
        """Return all registered component names."""
        return list(self._components.keys())


# ---------------------------------------------------------------------------
# Emitter output validation
# ---------------------------------------------------------------------------


class EmitterValidationError(Exception):
    """Raised when an emitter returns columns not declared in the registry."""


def _validate_emitter_output(
    emitter_name: str,
    output: dict[str, float | str | None],
    registry: FeatureRegistry,
) -> None:
    """Validate that all keys in emitter output are declared in the registry.

    Raises EmitterValidationError if an emitter produces a key that doesn't
    exist in the registry schema. This prevents silent schema drift.
    """
    schema = registry.schema()
    invalid_keys = set(output.keys()) - set(schema.keys())
    if invalid_keys:
        raise EmitterValidationError(
            f"Emitter '{emitter_name}' produced undeclared columns: "
            f"{sorted(invalid_keys)}. All columns must be registered."
        )


# ---------------------------------------------------------------------------
# Core replay loop
# ---------------------------------------------------------------------------


def _build_outcome_view(fight: HistoricalFight) -> FightOutcomeView:
    """Project a HistoricalFight into the FightOutcomeView consumed by components."""
    return FightOutcomeView(
        fight_url=fight.fight_url,
        event_url=fight.event_url,
        event_date=fight.event_date,
        fighter_a_url=fight.fighter_a_url,
        fighter_b_url=fight.fighter_b_url,
        winner_url=fight.winner_url,
        method=fight.method,
        ending_round=fight.ending_round,
        ending_time=fight.ending_time,
        weight_class=fight.weight_class,
        bout_order=fight.bout_order,
    )


def _build_emit_context(
    fight: HistoricalFight,
    fighter_url: str,
    opponent_url: str,
    fighter_profile: object,
    opponent_profile: object,
    frozen_components: dict[str, FrozenState],
) -> EmitContext:
    """Construct the frozen EmitContext for one fighter orientation."""
    return EmitContext(
        fighter_url=fighter_url,
        fighter_profile=fighter_profile,  # type: ignore[arg-type]
        opponent_url=opponent_url,
        opponent_profile=opponent_profile,  # type: ignore[arg-type]
        event_date=fight.event_date,
        event_url=fight.event_url,
        weight_class=fight.weight_class,
        fight_url=fight.fight_url,
        bout_order=fight.bout_order,
        components=frozen_components,
    )


def _emit_for_fight(
    fight: HistoricalFight,
    frozen_components: dict[str, FrozenState],
    emitters: Sequence[FeatureEmitter],
    registry: FeatureRegistry,
    generated_at: datetime,
) -> list[FeatureRow]:
    """Emit feature rows for both orientations of a single fight.

    Each fight produces exactly 2 FeatureRow objects: one with fighter_a as
    focal and one with fighter_b as focal. Emitter outputs are merged into a
    single feature dict per orientation and validated against the registry.
    """
    orientations: list[tuple[str, str, object, object]] = [
        (
            fight.fighter_a_url,
            fight.fighter_b_url,
            fight.fighter_a_profile,
            fight.fighter_b_profile,
        ),
        (
            fight.fighter_b_url,
            fight.fighter_a_url,
            fight.fighter_b_profile,
            fight.fighter_a_profile,
        ),
    ]

    rows: list[FeatureRow] = []
    for fighter_url, opponent_url, fighter_profile, opponent_profile in orientations:
        context = _build_emit_context(
            fight,
            fighter_url,
            opponent_url,
            fighter_profile,
            opponent_profile,
            frozen_components,
        )

        features: dict[str, float | str | None] = {}
        for emitter in emitters:
            output = emitter.emit(context)
            _validate_emitter_output(emitter.name, output, registry)
            features.update(output)

        rows.append(
            FeatureRow(
                fight_url=fight.fight_url,
                fighter_url=fighter_url,
                event_url=fight.event_url,
                event_date=fight.event_date,
                opponent_url=opponent_url,
                weight_class=fight.weight_class,
                feature_version=FEATURE_VERSION,
                generated_at=generated_at,
                features=features,
            )
        )

    return rows


def replay(
    fights: Sequence[HistoricalFight],
    component_registry: ComponentRegistry,
    emitters: Sequence[FeatureEmitter],
    feature_registry: FeatureRegistry,
    *,
    config: ReplayConfig | None = None,
    experience_accumulator: object | None = None,
) -> list[FeatureRow]:
    """Execute the full replay loop: tick → freeze → emit → update.

    This is the main entry point for the feature engine. It processes all fights
    event-by-event with strict emit-before-update ordering:

    1. Group fights into event-atomic ticks.
    2. For each tick: freeze all components (immutable snapshots).
    3. Emit features for every fight × orientation from frozen state.
    4. Update all components with fight outcomes (mutations happen AFTER emission).

    The generated_at timestamp is captured once at replay start and shared by all
    rows for reproducibility. The same input always produces the same output
    regardless of wall-clock time during execution.

    Args:
        fights: Chronologically sorted HistoricalFight list from loader.
        component_registry: Named components whose state accumulates.
        emitters: Stateless emitters that read frozen state.
        feature_registry: Schema owner for output validation.
        config: Optional replay configuration (log frequency, etc.).
        experience_accumulator: The ExperienceAccumulator instance if registered,
            for calling update_with_context() with time_format. Pass None if not
            using experience features.

    Returns:
        Complete list of FeatureRow objects (2 per fight: one per orientation).
    """
    cfg = config or ReplayConfig()

    # Single timestamp for all rows — determinism regardless of execution duration
    generated_at = datetime.now(UTC)

    ticks = group_into_ticks(fights)
    total_events = len(ticks)
    total_fights = sum(len(t.fights) for t in ticks)

    logger.info(
        "Replay starting: %d events, %d fights, %d components, %d emitters",
        total_events,
        total_fights,
        len(component_registry.names),
        len(emitters),
    )

    all_rows: list[FeatureRow] = []

    for tick_idx, tick in enumerate(ticks):
        # === PHASE 1: FREEZE (snapshot all state BEFORE any updates) ===
        frozen_components = component_registry.freeze_all()

        # === PHASE 2: EMIT (all features from frozen state) ===
        # Emit features for every fight on this card using frozen snapshots.
        # No fight on this card can see any other fight on this card's outcome.
        for fight in tick.fights:
            fight_rows = _emit_for_fight(
                fight, frozen_components, emitters, feature_registry, generated_at
            )
            all_rows.extend(fight_rows)

        # === MEMORY DISCIPLINE: Release frozen snapshots before update phase ===
        # Frozen snapshots from CommonOpponentIndex, CareerAccumulator, and
        # RematchEmitter deep-copy their entire accumulated history on freeze().
        # Retaining references across ticks would accumulate gigabytes. Snapshots
        # are consumed during emission above and MUST NOT survive this tick.
        # Deleting the reference allows GC to reclaim the deep-copied data.
        del frozen_components

        # === PHASE 3: UPDATE (mutate state with fight outcomes) ===
        # Only now — after all emission is complete — do accumulators see outcomes.
        for fight in tick.fights:
            outcome = _build_outcome_view(fight)
            component_registry.update_all(outcome)

            # ExperienceAccumulator needs time_format which is not in FightOutcomeView
            if experience_accumulator is not None:
                experience_accumulator.update_with_context(  # type: ignore[attr-defined]
                    outcome, time_format=fight.time_format
                )

        # Progress logging
        if (tick_idx + 1) % cfg.log_every_n_events == 0 or tick_idx == total_events - 1:
            logger.info(
                "Replay progress: %d/%d events processed (%d rows emitted)",
                tick_idx + 1,
                total_events,
                len(all_rows),
            )

    logger.info(
        "Replay complete: %d events, %d fights, %d feature rows emitted",
        total_events,
        total_fights,
        len(all_rows),
    )

    return all_rows
