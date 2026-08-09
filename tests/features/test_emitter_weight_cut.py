"""Tests for WeightCutEmitter.

Validates moving-down-in-weight detection from weight class migration history,
and confirms that missed-weight and short-notice fields correctly return None
(no structured data source in ufcstats schema).
"""

from __future__ import annotations

from datetime import date

from ufc_edge.features.components.weight_class import (
    FighterWeightSnapshot,
    WeightClassFrozenState,
    WeightClassTracker,
)
from ufc_edge.features.contracts import EmitContext, FighterProfile, FightOutcomeView
from ufc_edge.features.emitters.weight_cut import WeightCutEmitter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIGHTER_A = "http://fighter/a"
FIGHTER_B = "http://fighter/b"


def _make_weight_snapshot(
    fighters: dict[str, FighterWeightSnapshot],
) -> WeightClassFrozenState:
    """Build a frozen weight class state from explicit fighter snapshots."""
    return WeightClassFrozenState(fighters=fighters, thresholds={})


def _make_context(
    weight_state: WeightClassFrozenState,
    *,
    fighter_url: str = FIGHTER_A,
    opponent_url: str = FIGHTER_B,
    event_date: date = date(2024, 6, 1),
    weight_class: str = "Lightweight",
) -> EmitContext:
    """Build a minimal EmitContext with weight class state."""
    return EmitContext(
        fighter_url=fighter_url,
        fighter_profile=FighterProfile(fighter_url=fighter_url),
        opponent_url=opponent_url,
        opponent_profile=FighterProfile(fighter_url=opponent_url),
        event_date=event_date,
        event_url="http://event/test",
        weight_class=weight_class,
        fight_url="http://fight/test",
        bout_order=None,
        components={"weight_class": weight_state},
    )


def _outcome(
    *,
    fight_url: str = "http://fight/1",
    event_url: str = "http://event/1",
    event_date: date = date(2024, 6, 1),
    fighter_a_url: str = FIGHTER_A,
    fighter_b_url: str = FIGHTER_B,
    winner_url: str | None = FIGHTER_A,
    method: str = "Decision - Unanimous",
    ending_round: int = 3,
    ending_time: str = "5:00",
    weight_class: str = "Lightweight",
    bout_order: int | None = None,
) -> FightOutcomeView:
    return FightOutcomeView(
        fight_url=fight_url,
        event_url=event_url,
        event_date=event_date,
        fighter_a_url=fighter_a_url,
        fighter_b_url=fighter_b_url,
        winner_url=winner_url,
        method=method,
        ending_round=ending_round,
        ending_time=ending_time,
        weight_class=weight_class,
        bout_order=bout_order,
    )


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestWeightCutEmitterProtocol:
    """WeightCutEmitter satisfies the FeatureEmitter protocol."""

    def test_has_name_attribute(self) -> None:
        emitter = WeightCutEmitter()
        assert isinstance(emitter.name, str)
        assert emitter.name == "weight_cut"

    def test_emit_returns_dict(self) -> None:
        emitter = WeightCutEmitter()
        state = _make_weight_snapshot({})
        ctx = _make_context(state)
        result = emitter.emit(ctx)
        assert isinstance(result, dict)

    def test_emits_expected_keys(self) -> None:
        emitter = WeightCutEmitter()
        state = _make_weight_snapshot({})
        ctx = _make_context(state)
        result = emitter.emit(ctx)
        expected_keys = {
            "missed_weight_last_3",
            "missed_weight_career",
            "moving_down_in_weight",
            "short_notice",
            "full_camp",
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Missed weight fields — always None (no ufcstats structured source)
# ---------------------------------------------------------------------------


class TestMissedWeightFieldsAlwaysNone:
    """Missed-weight fields return None regardless of state.

    UFCStats does not track whether a fighter missed weight at weigh-ins.
    These fields remain None until a structured data source is available.
    """

    def test_missed_weight_last_3_is_none_unknown_fighter(self) -> None:
        emitter = WeightCutEmitter()
        state = _make_weight_snapshot({})
        ctx = _make_context(state)
        result = emitter.emit(ctx)
        assert result["missed_weight_last_3"] is None

    def test_missed_weight_career_is_none_unknown_fighter(self) -> None:
        emitter = WeightCutEmitter()
        state = _make_weight_snapshot({})
        ctx = _make_context(state)
        result = emitter.emit(ctx)
        assert result["missed_weight_career"] is None

    def test_missed_weight_last_3_is_none_known_fighter(self) -> None:
        emitter = WeightCutEmitter()
        state = _make_weight_snapshot(
            {
                FIGHTER_A: FighterWeightSnapshot(
                    first_class="Welterweight",
                    current_class="Lightweight",
                    migration_count=1,
                    fights_at_class={"Welterweight": 3, "Lightweight": 2},
                    wins_at_class={"Welterweight": 2, "Lightweight": 1},
                )
            }
        )
        ctx = _make_context(state)
        result = emitter.emit(ctx)
        assert result["missed_weight_last_3"] is None

    def test_missed_weight_career_is_none_known_fighter(self) -> None:
        emitter = WeightCutEmitter()
        state = _make_weight_snapshot(
            {
                FIGHTER_A: FighterWeightSnapshot(
                    first_class="Lightweight",
                    current_class="Lightweight",
                    migration_count=0,
                    fights_at_class={"Lightweight": 5},
                    wins_at_class={"Lightweight": 3},
                )
            }
        )
        ctx = _make_context(state)
        result = emitter.emit(ctx)
        assert result["missed_weight_career"] is None


# ---------------------------------------------------------------------------
# Short-notice / full-camp — always None (no structured data source)
# ---------------------------------------------------------------------------


class TestShortNoticeFieldsAlwaysNone:
    """Short-notice and full-camp fields return None.

    No structured announcement/booking date data exists in the ufcstats schema.
    """

    def test_short_notice_is_none(self) -> None:
        emitter = WeightCutEmitter()
        state = _make_weight_snapshot({})
        ctx = _make_context(state)
        result = emitter.emit(ctx)
        assert result["short_notice"] is None

    def test_full_camp_is_none(self) -> None:
        emitter = WeightCutEmitter()
        state = _make_weight_snapshot({})
        ctx = _make_context(state)
        result = emitter.emit(ctx)
        assert result["full_camp"] is None


# ---------------------------------------------------------------------------
# Moving down in weight — derivable from weight class migration history
# ---------------------------------------------------------------------------


class TestMovingDownInWeight:
    """Tests for the moving_down_in_weight feature.

    Uses weight class hierarchy to determine if a fighter moved to a lighter
    division. Returns 1.0 when current class is lighter than first class,
    0.0 when same or heavier, None when fighter is unknown or classes
    are not in the known hierarchy.
    """

    def test_unknown_fighter_returns_none(self) -> None:
        emitter = WeightCutEmitter()
        state = _make_weight_snapshot({})
        ctx = _make_context(state)
        result = emitter.emit(ctx)
        assert result["moving_down_in_weight"] is None

    def test_same_class_returns_zero(self) -> None:
        emitter = WeightCutEmitter()
        state = _make_weight_snapshot(
            {
                FIGHTER_A: FighterWeightSnapshot(
                    first_class="Lightweight",
                    current_class="Lightweight",
                    migration_count=0,
                    fights_at_class={"Lightweight": 5},
                    wins_at_class={"Lightweight": 3},
                )
            }
        )
        ctx = _make_context(state)
        result = emitter.emit(ctx)
        assert result["moving_down_in_weight"] == 0.0

    def test_welterweight_to_lightweight_returns_one(self) -> None:
        emitter = WeightCutEmitter()
        state = _make_weight_snapshot(
            {
                FIGHTER_A: FighterWeightSnapshot(
                    first_class="Welterweight",
                    current_class="Lightweight",
                    migration_count=1,
                    fights_at_class={"Welterweight": 3, "Lightweight": 2},
                    wins_at_class={"Welterweight": 2, "Lightweight": 1},
                )
            }
        )
        ctx = _make_context(state)
        result = emitter.emit(ctx)
        assert result["moving_down_in_weight"] == 1.0

    def test_lightweight_to_welterweight_returns_zero(self) -> None:
        """Moving up in weight is not 'moving down'."""
        emitter = WeightCutEmitter()
        state = _make_weight_snapshot(
            {
                FIGHTER_A: FighterWeightSnapshot(
                    first_class="Lightweight",
                    current_class="Welterweight",
                    migration_count=1,
                    fights_at_class={"Lightweight": 3, "Welterweight": 2},
                    wins_at_class={"Lightweight": 2, "Welterweight": 1},
                )
            }
        )
        ctx = _make_context(state)
        result = emitter.emit(ctx)
        assert result["moving_down_in_weight"] == 0.0

    def test_heavyweight_to_light_heavyweight_returns_one(self) -> None:
        emitter = WeightCutEmitter()
        state = _make_weight_snapshot(
            {
                FIGHTER_A: FighterWeightSnapshot(
                    first_class="Heavyweight",
                    current_class="Light Heavyweight",
                    migration_count=1,
                    fights_at_class={"Heavyweight": 2, "Light Heavyweight": 3},
                    wins_at_class={"Heavyweight": 1, "Light Heavyweight": 2},
                )
            }
        )
        ctx = _make_context(state)
        result = emitter.emit(ctx)
        assert result["moving_down_in_weight"] == 1.0

    def test_flyweight_to_bantamweight_returns_zero(self) -> None:
        """Moving from Flyweight to Bantamweight is moving UP."""
        emitter = WeightCutEmitter()
        state = _make_weight_snapshot(
            {
                FIGHTER_A: FighterWeightSnapshot(
                    first_class="Flyweight",
                    current_class="Bantamweight",
                    migration_count=1,
                    fights_at_class={"Flyweight": 3, "Bantamweight": 1},
                    wins_at_class={"Flyweight": 2, "Bantamweight": 1},
                )
            }
        )
        ctx = _make_context(state)
        result = emitter.emit(ctx)
        assert result["moving_down_in_weight"] == 0.0

    def test_unknown_class_returns_none(self) -> None:
        """Weight classes not in the known hierarchy produce None."""
        emitter = WeightCutEmitter()
        state = _make_weight_snapshot(
            {
                FIGHTER_A: FighterWeightSnapshot(
                    first_class="Super Heavyweight",
                    current_class="Cruiserweight",
                    migration_count=1,
                    fights_at_class={"Super Heavyweight": 2, "Cruiserweight": 1},
                    wins_at_class={"Super Heavyweight": 1, "Cruiserweight": 0},
                )
            }
        )
        ctx = _make_context(state)
        result = emitter.emit(ctx)
        assert result["moving_down_in_weight"] is None

    def test_women_strawweight_to_flyweight_returns_zero(self) -> None:
        """Women's divisions: Strawweight to Flyweight is moving UP."""
        emitter = WeightCutEmitter()
        state = _make_weight_snapshot(
            {
                FIGHTER_A: FighterWeightSnapshot(
                    first_class="Women's Strawweight",
                    current_class="Women's Flyweight",
                    migration_count=1,
                    fights_at_class={"Women's Strawweight": 4, "Women's Flyweight": 1},
                    wins_at_class={"Women's Strawweight": 3, "Women's Flyweight": 1},
                )
            }
        )
        ctx = _make_context(state)
        result = emitter.emit(ctx)
        assert result["moving_down_in_weight"] == 0.0

    def test_women_bantamweight_to_flyweight_returns_one(self) -> None:
        """Women's divisions: Bantamweight to Flyweight is moving DOWN."""
        emitter = WeightCutEmitter()
        state = _make_weight_snapshot(
            {
                FIGHTER_A: FighterWeightSnapshot(
                    first_class="Women's Bantamweight",
                    current_class="Women's Flyweight",
                    migration_count=1,
                    fights_at_class={
                        "Women's Bantamweight": 3,
                        "Women's Flyweight": 2,
                    },
                    wins_at_class={"Women's Bantamweight": 2, "Women's Flyweight": 1},
                )
            }
        )
        ctx = _make_context(state)
        result = emitter.emit(ctx)
        assert result["moving_down_in_weight"] == 1.0

    def test_only_first_class_unknown_returns_none(self) -> None:
        """If only one of first/current class is unrecognized, return None."""
        emitter = WeightCutEmitter()
        state = _make_weight_snapshot(
            {
                FIGHTER_A: FighterWeightSnapshot(
                    first_class="Catch Weight",
                    current_class="Lightweight",
                    migration_count=1,
                    fights_at_class={"Catch Weight": 1, "Lightweight": 2},
                    wins_at_class={"Catch Weight": 0, "Lightweight": 1},
                )
            }
        )
        ctx = _make_context(state)
        result = emitter.emit(ctx)
        assert result["moving_down_in_weight"] is None


# ---------------------------------------------------------------------------
# Integration with WeightClassTracker component
# ---------------------------------------------------------------------------


class TestMovingDownWithTracker:
    """Integration test using the real WeightClassTracker to produce state."""

    def test_tracker_detects_downward_migration(self) -> None:
        """Fighter starts at Welterweight, migrates to Lightweight."""
        tracker = WeightClassTracker()
        tracker.update(
            _outcome(
                fight_url="http://fight/1",
                event_date=date(2023, 1, 1),
                weight_class="Welterweight",
            )
        )
        tracker.update(
            _outcome(
                fight_url="http://fight/2",
                event_date=date(2023, 6, 1),
                weight_class="Welterweight",
            )
        )
        tracker.update(
            _outcome(
                fight_url="http://fight/3",
                event_date=date(2024, 1, 1),
                weight_class="Lightweight",
            )
        )

        snapshot = tracker.freeze()
        ctx = _make_context(snapshot, weight_class="Lightweight")
        emitter = WeightCutEmitter()
        result = emitter.emit(ctx)
        assert result["moving_down_in_weight"] == 1.0

    def test_tracker_no_migration_same_class(self) -> None:
        """Fighter stays in the same weight class throughout."""
        tracker = WeightClassTracker()
        tracker.update(
            _outcome(
                fight_url="http://fight/1",
                event_date=date(2023, 1, 1),
                weight_class="Middleweight",
            )
        )
        tracker.update(
            _outcome(
                fight_url="http://fight/2",
                event_date=date(2023, 6, 1),
                weight_class="Middleweight",
            )
        )

        snapshot = tracker.freeze()
        ctx = _make_context(snapshot, weight_class="Middleweight")
        emitter = WeightCutEmitter()
        result = emitter.emit(ctx)
        assert result["moving_down_in_weight"] == 0.0
