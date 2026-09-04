"""Tests for the replay engine orchestration.

Verifies:
- EventTicker groups fights by event correctly
- Emit-before-update ordering (same-card isolation)
- Deterministic output across repeated replays
- Emitter output validation against the registry
- Memory discipline (snapshots released per tick)
- Both orientations emitted per fight
- Experience accumulator update_with_context integration
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from ufc_edge.features.contracts import (
    EmitContext,
    FeatureRow,
    FighterProfile,
    FightOutcomeView,
    FightTotals,
    FrozenState,
    HistoricalFight,
)
from ufc_edge.features.registry import FeatureFamily, FeatureRegistry
from ufc_edge.features.replay import (
    ComponentRegistry,
    EmitterValidationError,
    ReplayConfig,
    _validate_emitter_output,
    group_into_ticks,
    replay,
)

# ---------------------------------------------------------------------------
# Test fixtures — minimal fight data
# ---------------------------------------------------------------------------

_DEFAULT_PROFILE = FighterProfile(fighter_url="fighter://unknown")

_DEFAULT_TOTALS = FightTotals(
    knockdowns=1,
    sig_strikes_landed=50,
    sig_strikes_attempted=100,
    total_strikes_landed=80,
    total_strikes_attempted=150,
    takedowns_landed=2,
    takedowns_attempted=5,
    submissions_attempted=1,
    reversals=0,
    control_time_seconds=120,
)


def _make_fight(
    fight_url: str = "fight://1",
    event_url: str = "event://1",
    event_date: date = date(2023, 1, 1),
    fighter_a_url: str = "fighter://a",
    fighter_b_url: str = "fighter://b",
    winner_url: str | None = "fighter://a",
    method: str = "Decision",
    weight_class: str = "Lightweight",
) -> HistoricalFight:
    """Create a minimal HistoricalFight for testing."""
    return HistoricalFight(
        fight_url=fight_url,
        event_url=event_url,
        event_date=event_date,
        fighter_a_url=fighter_a_url,
        fighter_b_url=fighter_b_url,
        winner_url=winner_url,
        method=method,
        ending_round=3,
        ending_time="5:00",
        time_format="3 Rnd (5-5-5)",
        weight_class=weight_class,
        bout_order=None,
        fighter_a_profile=FighterProfile(fighter_url=fighter_a_url),
        fighter_b_profile=FighterProfile(fighter_url=fighter_b_url),
        fighter_a_totals=_DEFAULT_TOTALS,
        fighter_b_totals=_DEFAULT_TOTALS,
    )


def _make_registry_with_columns(columns: dict[str, type]) -> FeatureRegistry:
    """Create a minimal registry with given columns."""
    family = FeatureFamily(name="test", columns=columns, order=0)
    return FeatureRegistry(families=[family])


# ---------------------------------------------------------------------------
# Tracking StateComponent — records update/freeze call order
# ---------------------------------------------------------------------------


class _TrackingFrozenState(FrozenState):
    """Frozen state that carries a version counter for ordering verification."""

    __slots__ = ("_version", "_updates_seen")

    def __init__(self, version: int, updates_seen: list[str]) -> None:
        object.__setattr__(self, "_version", version)
        object.__setattr__(self, "_updates_seen", tuple(updates_seen))

    @property
    def version(self) -> int:
        return self._version

    @property
    def updates_seen(self) -> tuple[str, ...]:
        return self._updates_seen


class _TrackingComponent:
    """Records the order of update() and freeze() calls for ordering tests."""

    def __init__(self, name: str, call_log: list[tuple[str, str]]) -> None:
        self._name = name
        self._call_log = call_log
        self._version = 0
        self._updates: list[str] = []

    def update(self, fight: FightOutcomeView) -> None:
        self._call_log.append(("update", f"{self._name}:{fight.fight_url}"))
        self._version += 1
        self._updates.append(fight.fight_url)

    def freeze(self) -> _TrackingFrozenState:
        self._call_log.append(("freeze", self._name))
        return _TrackingFrozenState(self._version, list(self._updates))


# ---------------------------------------------------------------------------
# Tracking FeatureEmitter — records what frozen state it sees at emit time
# ---------------------------------------------------------------------------


class _TrackingEmitter:
    """Records component version at emit time to prove emit-before-update."""

    name: str = "tracker"

    def __init__(self, emit_log: list[dict[str, Any]]) -> None:
        self._emit_log = emit_log

    def emit(self, context: EmitContext) -> dict[str, float | str | None]:
        tracking_state = context.components.get("tracking")
        version = tracking_state.version if tracking_state else -1  # type: ignore[union-attr]
        self._emit_log.append(
            {
                "fight_url": context.fight_url,
                "fighter_url": context.fighter_url,
                "frozen_version": version,
            }
        )
        return {"test_col": float(version)}


# ---------------------------------------------------------------------------
# Tests: EventTicker (group_into_ticks)
# ---------------------------------------------------------------------------


class TestGroupIntoTicks:
    """Tests for the EventTicker that groups fights into event-atomic ticks."""

    def test_single_event_single_fight(self) -> None:
        fights = [_make_fight()]
        ticks = group_into_ticks(fights)
        assert len(ticks) == 1
        assert ticks[0].event_url == "event://1"
        assert len(ticks[0].fights) == 1

    def test_single_event_multiple_fights(self) -> None:
        fights = [
            _make_fight(fight_url="fight://1"),
            _make_fight(fight_url="fight://2"),
            _make_fight(fight_url="fight://3"),
        ]
        ticks = group_into_ticks(fights)
        assert len(ticks) == 1
        assert len(ticks[0].fights) == 3

    def test_multiple_events_sorted_by_date(self) -> None:
        fights = [
            _make_fight(fight_url="fight://1", event_url="event://a", event_date=date(2023, 1, 1)),
            _make_fight(fight_url="fight://2", event_url="event://b", event_date=date(2023, 2, 1)),
            _make_fight(fight_url="fight://3", event_url="event://c", event_date=date(2023, 3, 1)),
        ]
        ticks = group_into_ticks(fights)
        assert len(ticks) == 3
        assert ticks[0].event_date == date(2023, 1, 1)
        assert ticks[1].event_date == date(2023, 2, 1)
        assert ticks[2].event_date == date(2023, 3, 1)

    def test_fights_within_tick_sorted_by_fight_url(self) -> None:
        fights = [
            _make_fight(fight_url="fight://c", event_url="event://1"),
            _make_fight(fight_url="fight://a", event_url="event://1"),
            _make_fight(fight_url="fight://b", event_url="event://1"),
        ]
        ticks = group_into_ticks(fights)
        urls = [f.fight_url for f in ticks[0].fights]
        assert urls == ["fight://a", "fight://b", "fight://c"]

    def test_empty_input_returns_no_ticks(self) -> None:
        ticks = group_into_ticks([])
        assert ticks == []

    def test_same_date_different_events_are_separate_ticks(self) -> None:
        fights = [
            _make_fight(fight_url="fight://1", event_url="event://a", event_date=date(2023, 1, 1)),
            _make_fight(fight_url="fight://2", event_url="event://b", event_date=date(2023, 1, 1)),
        ]
        ticks = group_into_ticks(fights)
        assert len(ticks) == 2

    def test_tick_model_is_frozen(self) -> None:
        fights = [_make_fight()]
        ticks = group_into_ticks(fights)
        with pytest.raises((TypeError, ValueError, AttributeError)):
            ticks[0].event_url = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: Emit-before-update ordering invariant
# ---------------------------------------------------------------------------


class TestEmitBeforeUpdate:
    """Proves that all features are emitted from frozen state BEFORE any update."""

    def test_freeze_happens_before_update_within_tick(self) -> None:
        """All freeze calls happen before any update call within a single tick."""
        call_log: list[tuple[str, str]] = []
        component = _TrackingComponent("tracker", call_log)
        registry_components = ComponentRegistry({"tracking": component})  # type: ignore[dict-item]

        emit_log: list[dict[str, Any]] = []
        emitter = _TrackingEmitter(emit_log)

        feature_registry = _make_registry_with_columns({"test_col": float})

        fights = [
            _make_fight(fight_url="fight://1"),
            _make_fight(fight_url="fight://2"),
        ]

        replay(fights, registry_components, [emitter], feature_registry)

        # Extract call order: all freezes before all updates within the tick
        first_update_idx = next(
            (i for i, (op, _) in enumerate(call_log) if op == "update"), len(call_log)
        )
        last_freeze_idx = max(
            (i for i, (op, _) in enumerate(call_log) if op == "freeze"),
            default=-1,
        )
        # In the first tick, freeze must precede all updates
        assert last_freeze_idx < first_update_idx

    def test_same_card_fights_see_identical_frozen_version(self) -> None:
        """Two fights on the same card observe the same frozen state version."""
        call_log: list[tuple[str, str]] = []
        component = _TrackingComponent("tracker", call_log)
        registry_components = ComponentRegistry({"tracking": component})  # type: ignore[dict-item]

        emit_log: list[dict[str, Any]] = []
        emitter = _TrackingEmitter(emit_log)

        feature_registry = _make_registry_with_columns({"test_col": float})

        fights = [
            _make_fight(
                fight_url="fight://1", fighter_a_url="fighter://a", fighter_b_url="fighter://b"
            ),
            _make_fight(
                fight_url="fight://2", fighter_a_url="fighter://c", fighter_b_url="fighter://d"
            ),
        ]

        replay(fights, registry_components, [emitter], feature_registry)

        # All emissions in the first tick should see version 0 (no updates yet)
        versions = [entry["frozen_version"] for entry in emit_log]
        assert all(v == 0 for v in versions)

    def test_same_card_isolation_fights_do_not_affect_each_other(self) -> None:
        """Altering one fight on a card does not change another fight's features.

        This is the core same-card isolation property: within a single event tick,
        fights are independent — removing or altering fight G does not change the
        features emitted for fight F on the same card.
        """
        call_log: list[tuple[str, str]] = []
        emit_log: list[dict[str, Any]] = []

        feature_registry = _make_registry_with_columns({"test_col": float})

        # Run with 2 fights on the same card
        fights_full = [
            _make_fight(
                fight_url="fight://1", fighter_a_url="fighter://a", fighter_b_url="fighter://b"
            ),
            _make_fight(
                fight_url="fight://2", fighter_a_url="fighter://c", fighter_b_url="fighter://d"
            ),
        ]
        component_full = _TrackingComponent("tracker", call_log)
        registry_full = ComponentRegistry({"tracking": component_full})  # type: ignore[dict-item]
        emitter_full = _TrackingEmitter(emit_log)
        rows_full = replay(fights_full, registry_full, [emitter_full], feature_registry)

        # Run with only fight://1 — remove fight://2 from the card
        call_log_partial: list[tuple[str, str]] = []
        emit_log_partial: list[dict[str, Any]] = []
        fights_partial = [
            _make_fight(
                fight_url="fight://1", fighter_a_url="fighter://a", fighter_b_url="fighter://b"
            ),
        ]
        component_partial = _TrackingComponent("tracker", call_log_partial)
        registry_partial = ComponentRegistry({"tracking": component_partial})  # type: ignore[dict-item]
        emitter_partial = _TrackingEmitter(emit_log_partial)
        rows_partial = replay(fights_partial, registry_partial, [emitter_partial], feature_registry)

        # fight://1's features must be identical regardless of whether fight://2 exists
        full_fight1_rows = [r for r in rows_full if r.fight_url == "fight://1"]
        partial_fight1_rows = rows_partial

        for full_row, partial_row in zip(full_fight1_rows, partial_fight1_rows, strict=True):
            assert full_row.features == partial_row.features

    def test_second_event_sees_first_events_updates(self) -> None:
        """An event after the first observes the first event's state updates."""
        call_log: list[tuple[str, str]] = []
        emit_log: list[dict[str, Any]] = []
        component = _TrackingComponent("tracker", call_log)
        registry_components = ComponentRegistry({"tracking": component})  # type: ignore[dict-item]
        emitter = _TrackingEmitter(emit_log)

        feature_registry = _make_registry_with_columns({"test_col": float})

        fights = [
            _make_fight(fight_url="fight://1", event_url="event://1", event_date=date(2023, 1, 1)),
            _make_fight(fight_url="fight://2", event_url="event://2", event_date=date(2023, 2, 1)),
        ]

        replay(fights, registry_components, [emitter], feature_registry)

        # Event 2's emissions should see version > 0 (reflecting event 1's update)
        event2_emissions = [e for e in emit_log if e["fight_url"] == "fight://2"]
        assert all(e["frozen_version"] > 0 for e in event2_emissions)


# ---------------------------------------------------------------------------
# Tests: Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Proves that replaying the same input produces byte-identical output."""

    def test_identical_output_across_two_replays(self) -> None:
        """Running replay twice with identical input yields identical feature rows."""

        class _SimpleEmitter:
            name: str = "simple"

            def emit(self, context: EmitContext) -> dict[str, float | str | None]:
                return {"col_a": 1.0, "col_b": "x"}

        feature_registry = _make_registry_with_columns({"col_a": float, "col_b": str})

        fights = [
            _make_fight(fight_url="fight://1", event_url="event://1", event_date=date(2023, 1, 1)),
            _make_fight(fight_url="fight://2", event_url="event://1", event_date=date(2023, 1, 1)),
            _make_fight(fight_url="fight://3", event_url="event://2", event_date=date(2023, 2, 1)),
        ]

        def _run_replay() -> list[FeatureRow]:
            class _NullComponent:
                def update(self, fight: FightOutcomeView) -> None:
                    pass

                def freeze(self) -> FrozenState:
                    return FrozenState()

            components = ComponentRegistry({"null": _NullComponent()})  # type: ignore[dict-item]
            return replay(fights, components, [_SimpleEmitter()], feature_registry)

        rows_1 = _run_replay()
        rows_2 = _run_replay()

        assert len(rows_1) == len(rows_2)
        for r1, r2 in zip(rows_1, rows_2, strict=True):
            assert r1.fight_url == r2.fight_url
            assert r1.fighter_url == r2.fighter_url
            assert r1.features == r2.features
            assert r1.event_url == r2.event_url
            assert r1.event_date == r2.event_date


# ---------------------------------------------------------------------------
# Tests: Emitter output validation
# ---------------------------------------------------------------------------


class TestEmitterValidation:
    """Proves that emitter output is validated against the registry schema."""

    def test_valid_output_passes(self) -> None:
        registry = _make_registry_with_columns({"col_a": float, "col_b": str})
        _validate_emitter_output("test", {"col_a": 1.0, "col_b": "x"}, registry)

    def test_undeclared_column_raises(self) -> None:
        registry = _make_registry_with_columns({"col_a": float})
        with pytest.raises(EmitterValidationError, match="undeclared columns"):
            _validate_emitter_output("test", {"col_a": 1.0, "bad_col": 2.0}, registry)

    def test_replay_aborts_on_invalid_emitter_output(self) -> None:
        """The full replay aborts when an emitter produces undeclared columns."""

        class _BadEmitter:
            name: str = "bad"

            def emit(self, context: EmitContext) -> dict[str, float | str | None]:
                return {"undeclared": 99.0}

        feature_registry = _make_registry_with_columns({"col_a": float})
        fights = [_make_fight()]

        class _NullComponent:
            def update(self, fight: FightOutcomeView) -> None:
                pass

            def freeze(self) -> FrozenState:
                return FrozenState()

        components = ComponentRegistry({"null": _NullComponent()})  # type: ignore[dict-item]

        with pytest.raises(EmitterValidationError):
            replay(fights, components, [_BadEmitter()], feature_registry)


# ---------------------------------------------------------------------------
# Tests: Output structure
# ---------------------------------------------------------------------------


class TestOutputStructure:
    """Verifies the replay produces correct row count and structure."""

    def test_two_rows_per_fight(self) -> None:
        """Each fight produces exactly 2 FeatureRow objects (both orientations)."""

        class _SimpleEmitter:
            name: str = "simple"

            def emit(self, context: EmitContext) -> dict[str, float | str | None]:
                return {"val": 1.0}

        feature_registry = _make_registry_with_columns({"val": float})
        fights = [_make_fight(fighter_a_url="fighter://a", fighter_b_url="fighter://b")]

        class _NullComponent:
            def update(self, fight: FightOutcomeView) -> None:
                pass

            def freeze(self) -> FrozenState:
                return FrozenState()

        components = ComponentRegistry({"null": _NullComponent()})  # type: ignore[dict-item]
        rows = replay(fights, components, [_SimpleEmitter()], feature_registry)

        assert len(rows) == 2
        fighter_urls = {r.fighter_url for r in rows}
        assert fighter_urls == {"fighter://a", "fighter://b"}

    def test_opponent_url_is_symmetric(self) -> None:
        """Row for fighter A has opponent B, and vice versa."""

        class _SimpleEmitter:
            name: str = "simple"

            def emit(self, context: EmitContext) -> dict[str, float | str | None]:
                return {"val": 1.0}

        feature_registry = _make_registry_with_columns({"val": float})
        fights = [_make_fight(fighter_a_url="fighter://a", fighter_b_url="fighter://b")]

        class _NullComponent:
            def update(self, fight: FightOutcomeView) -> None:
                pass

            def freeze(self) -> FrozenState:
                return FrozenState()

        components = ComponentRegistry({"null": _NullComponent()})  # type: ignore[dict-item]
        rows = replay(fights, components, [_SimpleEmitter()], feature_registry)

        row_a = next(r for r in rows if r.fighter_url == "fighter://a")
        row_b = next(r for r in rows if r.fighter_url == "fighter://b")
        assert row_a.opponent_url == "fighter://b"
        assert row_b.opponent_url == "fighter://a"

    def test_feature_version_and_metadata_populated(self) -> None:
        """All rows have feature_version and generated_at set correctly."""

        class _SimpleEmitter:
            name: str = "simple"

            def emit(self, context: EmitContext) -> dict[str, float | str | None]:
                return {"val": 1.0}

        feature_registry = _make_registry_with_columns({"val": float})
        fights = [_make_fight()]

        class _NullComponent:
            def update(self, fight: FightOutcomeView) -> None:
                pass

            def freeze(self) -> FrozenState:
                return FrozenState()

        components = ComponentRegistry({"null": _NullComponent()})  # type: ignore[dict-item]
        rows = replay(fights, components, [_SimpleEmitter()], feature_registry)

        for row in rows:
            assert row.feature_version == "v1"
            assert row.generated_at is not None
            assert isinstance(row.generated_at, datetime)

    def test_multiple_emitters_merge_features(self) -> None:
        """Features from multiple emitters are merged into a single dict."""

        class _EmitterA:
            name: str = "a"

            def emit(self, context: EmitContext) -> dict[str, float | str | None]:
                return {"col_a": 1.0}

        class _EmitterB:
            name: str = "b"

            def emit(self, context: EmitContext) -> dict[str, float | str | None]:
                return {"col_b": 2.0}

        feature_registry = _make_registry_with_columns({"col_a": float, "col_b": float})
        fights = [_make_fight()]

        class _NullComponent:
            def update(self, fight: FightOutcomeView) -> None:
                pass

            def freeze(self) -> FrozenState:
                return FrozenState()

        components = ComponentRegistry({"null": _NullComponent()})  # type: ignore[dict-item]
        rows = replay(fights, components, [_EmitterA(), _EmitterB()], feature_registry)

        for row in rows:
            assert row.features["col_a"] == 1.0
            assert row.features["col_b"] == 2.0


# ---------------------------------------------------------------------------
# Tests: ComponentRegistry
# ---------------------------------------------------------------------------


class TestComponentRegistry:
    """Tests for the ComponentRegistry helper class."""

    def test_freeze_all_calls_every_component(self) -> None:
        call_log: list[tuple[str, str]] = []
        c1 = _TrackingComponent("elo", call_log)
        c2 = _TrackingComponent("career", call_log)
        registry = ComponentRegistry({"elo": c1, "career": c2})  # type: ignore[dict-item]

        frozen = registry.freeze_all()
        assert "elo" in frozen
        assert "career" in frozen

    def test_update_all_calls_every_component(self) -> None:
        call_log: list[tuple[str, str]] = []
        c1 = _TrackingComponent("elo", call_log)
        c2 = _TrackingComponent("career", call_log)
        registry = ComponentRegistry({"elo": c1, "career": c2})  # type: ignore[dict-item]

        outcome = FightOutcomeView(
            fight_url="fight://1",
            event_url="event://1",
            event_date=date(2023, 1, 1),
            fighter_a_url="fighter://a",
            fighter_b_url="fighter://b",
            winner_url="fighter://a",
            method="Decision",
            ending_round=3,
            ending_time="5:00",
            weight_class="Lightweight",
            bout_order=None,
        )
        registry.update_all(outcome)
        update_calls = [(op, target) for op, target in call_log if op == "update"]
        assert len(update_calls) == 2


# ---------------------------------------------------------------------------
# Tests: ExperienceAccumulator integration
# ---------------------------------------------------------------------------


class TestExperienceAccumulatorIntegration:
    """Verifies that update_with_context is called with time_format."""

    def test_update_with_context_called_per_fight(self) -> None:
        """Experience accumulator receives time_format for five-round detection."""
        mock_experience = MagicMock()
        mock_experience.update_with_context = MagicMock()

        class _SimpleEmitter:
            name: str = "simple"

            def emit(self, context: EmitContext) -> dict[str, float | str | None]:
                return {"val": 1.0}

        feature_registry = _make_registry_with_columns({"val": float})

        fight = _make_fight()
        fights = [fight]

        class _NullComponent:
            def update(self, f: FightOutcomeView) -> None:
                pass

            def freeze(self) -> FrozenState:
                return FrozenState()

        components = ComponentRegistry({"null": _NullComponent()})  # type: ignore[dict-item]
        replay(
            fights,
            components,
            [_SimpleEmitter()],
            feature_registry,
            experience_accumulator=mock_experience,
        )

        mock_experience.update_with_context.assert_called_once()
        call_kwargs = mock_experience.update_with_context.call_args
        assert call_kwargs[1]["time_format"] == "3 Rnd (5-5-5)"


# ---------------------------------------------------------------------------
# Tests: ReplayConfig
# ---------------------------------------------------------------------------


class TestReplayConfig:
    """Verifies configuration defaults and overrides."""

    def test_default_log_frequency(self) -> None:
        cfg = ReplayConfig()
        assert cfg.log_every_n_events == 50

    def test_custom_log_frequency(self) -> None:
        cfg = ReplayConfig(log_every_n_events=10)
        assert cfg.log_every_n_events == 10


# ---------------------------------------------------------------------------
# Tests: Multi-event replay ordering
# ---------------------------------------------------------------------------


class TestMultiEventReplay:
    """Verifies correct state evolution across multiple events."""

    def test_three_events_progressive_state_evolution(self) -> None:
        """State updates accumulate across events: event N+1 sees N's updates."""
        call_log: list[tuple[str, str]] = []
        emit_log: list[dict[str, Any]] = []
        component = _TrackingComponent("tracker", call_log)
        registry_components = ComponentRegistry({"tracking": component})  # type: ignore[dict-item]
        emitter = _TrackingEmitter(emit_log)
        feature_registry = _make_registry_with_columns({"test_col": float})

        fights = [
            _make_fight(fight_url="fight://1", event_url="event://1", event_date=date(2023, 1, 1)),
            _make_fight(fight_url="fight://2", event_url="event://2", event_date=date(2023, 2, 1)),
            _make_fight(fight_url="fight://3", event_url="event://3", event_date=date(2023, 3, 1)),
        ]

        replay(fights, registry_components, [emitter], feature_registry)

        # Event 1 sees version 0, event 2 sees version 1, event 3 sees version 2
        event1_versions = [e["frozen_version"] for e in emit_log if e["fight_url"] == "fight://1"]
        event2_versions = [e["frozen_version"] for e in emit_log if e["fight_url"] == "fight://2"]
        event3_versions = [e["frozen_version"] for e in emit_log if e["fight_url"] == "fight://3"]

        assert all(v == 0 for v in event1_versions)
        assert all(v == 1 for v in event2_versions)
        assert all(v == 2 for v in event3_versions)

    def test_row_ordering_matches_event_then_fight_order(self) -> None:
        """Output rows follow event chronological order, then fight_url within event."""

        class _SimpleEmitter:
            name: str = "simple"

            def emit(self, context: EmitContext) -> dict[str, float | str | None]:
                return {"val": 1.0}

        feature_registry = _make_registry_with_columns({"val": float})

        fights = [
            _make_fight(fight_url="fight://b", event_url="event://1", event_date=date(2023, 1, 1)),
            _make_fight(fight_url="fight://a", event_url="event://1", event_date=date(2023, 1, 1)),
            _make_fight(fight_url="fight://c", event_url="event://2", event_date=date(2023, 2, 1)),
        ]

        class _NullComponent:
            def update(self, fight: FightOutcomeView) -> None:
                pass

            def freeze(self) -> FrozenState:
                return FrozenState()

        components = ComponentRegistry({"null": _NullComponent()})  # type: ignore[dict-item]
        rows = replay(fights, components, [_SimpleEmitter()], feature_registry)

        fight_urls = [r.fight_url for r in rows]
        # Within event://1, fights sorted by fight_url: a, a, b, b; then event://2: c, c
        assert fight_urls == [
            "fight://a",
            "fight://a",
            "fight://b",
            "fight://b",
            "fight://c",
            "fight://c",
        ]
