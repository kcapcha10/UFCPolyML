"""Tests for the weight dominance feature emitter.

Validates migration features (class change detection, direction sign, per-class
fight/win counts) and weight bully features (top-quartile size detection,
grappling utilization rate, product term score).
"""

from __future__ import annotations

from datetime import date

from ufc_edge.features.components.rolling_stats import (
    FightStats,
    RollingStatsSnapshot,
)
from ufc_edge.features.components.weight_class import (
    FighterWeightSnapshot,
    WeightClassFrozenState,
)
from ufc_edge.features.contracts import EmitContext, FighterProfile
from ufc_edge.features.emitters.weight import WeightDominanceEmitter

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_weight_state(
    fighters: dict[str, FighterWeightSnapshot] | None = None,
    thresholds: dict[str, dict[str, float]] | None = None,
) -> WeightClassFrozenState:
    """Build a WeightClassFrozenState from provided data."""
    return WeightClassFrozenState(
        fighters=fighters or {},
        thresholds=thresholds or {},
    )


def _make_rolling_stats(
    fighter_windows: dict[str, tuple[FightStats, ...]] | None = None,
) -> RollingStatsSnapshot:
    """Build a RollingStatsSnapshot from provided window data."""
    return RollingStatsSnapshot(fighter_windows=fighter_windows or {})


def _make_fight_stats(
    *,
    takedowns_attempted: int = 0,
    control_time_seconds: int = 0,
    fight_duration_minutes: float = 15.0,
) -> FightStats:
    """Build a FightStats with only grappling fields populated."""
    return FightStats(
        fight_duration_minutes=fight_duration_minutes,
        sig_strikes_landed=0,
        sig_strikes_attempted=0,
        sig_strikes_absorbed=0,
        sig_strikes_absorbed_attempted=0,
        total_strikes_landed=0,
        total_strikes_attempted=0,
        takedowns_landed=0,
        takedowns_attempted=takedowns_attempted,
        submissions_attempted=0,
        knockdowns=0,
        control_time_seconds=control_time_seconds,
        opponent_takedowns_landed=0,
        opponent_takedowns_attempted=0,
        opponent_control_time_seconds=0,
    )


def _make_context(
    *,
    fighter_url: str = "http://ufcstats.com/fighter/a",
    opponent_url: str = "http://ufcstats.com/fighter/b",
    weight_class: str = "Lightweight",
    height_cm: float | None = 180.0,
    reach_cm: float | None = 190.0,
    weight_state: WeightClassFrozenState | None = None,
    rolling_stats: RollingStatsSnapshot | None = None,
) -> EmitContext:
    """Build a minimal EmitContext for weight dominance emitter testing."""
    fighter_profile = FighterProfile(
        fighter_url=fighter_url,
        height_cm=height_cm,
        reach_cm=reach_cm,
        stance="Orthodox",
        dob=date(1990, 5, 20),
    )
    opponent_profile = FighterProfile(
        fighter_url=opponent_url,
        height_cm=175.0,
        reach_cm=180.0,
        stance="Southpaw",
        dob=date(1988, 1, 15),
    )
    components: dict[str, object] = {}
    if weight_state is not None:
        components["weight_class"] = weight_state
    if rolling_stats is not None:
        components["rolling_stats"] = rolling_stats

    return EmitContext(
        fighter_url=fighter_url,
        fighter_profile=fighter_profile,
        opponent_url=opponent_url,
        opponent_profile=opponent_profile,
        event_date=date(2024, 3, 16),
        event_url="http://ufcstats.com/event/e1",
        weight_class=weight_class,
        fight_url="http://ufcstats.com/fight/f1",
        bout_order=None,
        components=components,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestWeightDominanceEmitterProtocol:
    """WeightDominanceEmitter satisfies the FeatureEmitter protocol."""

    def test_has_name_attribute(self) -> None:
        emitter = WeightDominanceEmitter()
        assert isinstance(emitter.name, str)
        assert emitter.name == "weight_dominance"

    def test_emit_returns_dict(self) -> None:
        emitter = WeightDominanceEmitter()
        ctx = _make_context()
        result = emitter.emit(ctx)
        assert isinstance(result, dict)

    def test_all_keys_present_when_no_state(self) -> None:
        emitter = WeightDominanceEmitter()
        ctx = _make_context()
        result = emitter.emit(ctx)
        expected_keys = {
            "is_weight_class_change",
            "direction_of_change",
            "fights_at_current_class",
            "win_pct_at_current_class",
            "prior_class_win_pct",
            "is_large_for_class",
            "grappling_utilization_rate",
            "weight_bully_score",
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Migration features — class change detection
# ---------------------------------------------------------------------------


class TestClassChangeDetection:
    """Migration feature: is_weight_class_change detects class transitions."""

    def test_no_change_same_class(self) -> None:
        """Fighter staying at same class -> 0.0."""
        state = _make_weight_state(
            fighters={
                "http://ufcstats.com/fighter/a": FighterWeightSnapshot(
                    first_class="Lightweight",
                    current_class="Lightweight",
                    migration_count=0,
                    fights_at_class={"Lightweight": 5},
                    wins_at_class={"Lightweight": 3},
                ),
            }
        )
        ctx = _make_context(weight_class="Lightweight", weight_state=state)
        result = WeightDominanceEmitter().emit(ctx)
        assert result["is_weight_class_change"] == 0.0

    def test_change_detected_moving_up(self) -> None:
        """Fighter whose current_class differs from prior class -> 1.0."""
        state = _make_weight_state(
            fighters={
                "http://ufcstats.com/fighter/a": FighterWeightSnapshot(
                    first_class="Lightweight",
                    current_class="Welterweight",
                    migration_count=1,
                    fights_at_class={"Lightweight": 5, "Welterweight": 1},
                    wins_at_class={"Lightweight": 3, "Welterweight": 1},
                ),
            }
        )
        ctx = _make_context(weight_class="Welterweight", weight_state=state)
        result = WeightDominanceEmitter().emit(ctx)
        assert result["is_weight_class_change"] == 1.0

    def test_unknown_fighter_returns_none(self) -> None:
        """Fighter with no weight history -> None for all migration features."""
        state = _make_weight_state()
        ctx = _make_context(weight_state=state)
        result = WeightDominanceEmitter().emit(ctx)
        assert result["is_weight_class_change"] is None


# ---------------------------------------------------------------------------
# Migration features — direction of change
# ---------------------------------------------------------------------------


class TestDirectionOfChange:
    """Direction sign: positive = moving up, negative = moving down."""

    def test_moving_up_positive(self) -> None:
        """Lightweight -> Welterweight = +1.0."""
        state = _make_weight_state(
            fighters={
                "http://ufcstats.com/fighter/a": FighterWeightSnapshot(
                    first_class="Lightweight",
                    current_class="Welterweight",
                    migration_count=1,
                    fights_at_class={"Lightweight": 5, "Welterweight": 1},
                    wins_at_class={"Lightweight": 3},
                ),
            }
        )
        ctx = _make_context(weight_class="Welterweight", weight_state=state)
        result = WeightDominanceEmitter().emit(ctx)
        assert result["direction_of_change"] == 1.0

    def test_moving_down_negative(self) -> None:
        """Welterweight -> Lightweight = -1.0."""
        state = _make_weight_state(
            fighters={
                "http://ufcstats.com/fighter/a": FighterWeightSnapshot(
                    first_class="Welterweight",
                    current_class="Lightweight",
                    migration_count=1,
                    fights_at_class={"Welterweight": 5, "Lightweight": 1},
                    wins_at_class={"Welterweight": 3},
                ),
            }
        )
        ctx = _make_context(weight_class="Lightweight", weight_state=state)
        result = WeightDominanceEmitter().emit(ctx)
        assert result["direction_of_change"] == -1.0

    def test_no_change_zero(self) -> None:
        """Same class = 0.0."""
        state = _make_weight_state(
            fighters={
                "http://ufcstats.com/fighter/a": FighterWeightSnapshot(
                    first_class="Lightweight",
                    current_class="Lightweight",
                    migration_count=0,
                    fights_at_class={"Lightweight": 5},
                    wins_at_class={"Lightweight": 3},
                ),
            }
        )
        ctx = _make_context(weight_class="Lightweight", weight_state=state)
        result = WeightDominanceEmitter().emit(ctx)
        assert result["direction_of_change"] == 0.0

    def test_unknown_class_none(self) -> None:
        """Unrecognized weight class in ordering -> None."""
        state = _make_weight_state(
            fighters={
                "http://ufcstats.com/fighter/a": FighterWeightSnapshot(
                    first_class="Open Weight",
                    current_class="Lightweight",
                    migration_count=1,
                    fights_at_class={"Open Weight": 1, "Lightweight": 1},
                    wins_at_class={},
                ),
            }
        )
        ctx = _make_context(weight_class="Lightweight", weight_state=state)
        result = WeightDominanceEmitter().emit(ctx)
        assert result["direction_of_change"] is None


# ---------------------------------------------------------------------------
# Migration features — per-class fight counts and win percentages
# ---------------------------------------------------------------------------


class TestPerClassStats:
    """Fights and win percentage at current and prior weight classes."""

    def test_fights_at_current_class(self) -> None:
        state = _make_weight_state(
            fighters={
                "http://ufcstats.com/fighter/a": FighterWeightSnapshot(
                    first_class="Lightweight",
                    current_class="Welterweight",
                    migration_count=1,
                    fights_at_class={"Lightweight": 5, "Welterweight": 3},
                    wins_at_class={"Lightweight": 3, "Welterweight": 2},
                ),
            }
        )
        ctx = _make_context(weight_class="Welterweight", weight_state=state)
        result = WeightDominanceEmitter().emit(ctx)
        assert result["fights_at_current_class"] == 3.0

    def test_win_pct_at_current_class(self) -> None:
        state = _make_weight_state(
            fighters={
                "http://ufcstats.com/fighter/a": FighterWeightSnapshot(
                    first_class="Lightweight",
                    current_class="Welterweight",
                    migration_count=1,
                    fights_at_class={"Lightweight": 5, "Welterweight": 4},
                    wins_at_class={"Lightweight": 3, "Welterweight": 3},
                ),
            }
        )
        ctx = _make_context(weight_class="Welterweight", weight_state=state)
        result = WeightDominanceEmitter().emit(ctx)
        assert result["win_pct_at_current_class"] == 0.75

    def test_prior_class_win_pct(self) -> None:
        """Prior class = first_class when different from current."""
        state = _make_weight_state(
            fighters={
                "http://ufcstats.com/fighter/a": FighterWeightSnapshot(
                    first_class="Lightweight",
                    current_class="Welterweight",
                    migration_count=1,
                    fights_at_class={"Lightweight": 10, "Welterweight": 2},
                    wins_at_class={"Lightweight": 7, "Welterweight": 1},
                ),
            }
        )
        ctx = _make_context(weight_class="Welterweight", weight_state=state)
        result = WeightDominanceEmitter().emit(ctx)
        assert result["prior_class_win_pct"] == 0.7

    def test_no_prior_class_returns_none(self) -> None:
        """Fighter with only one class has no prior class win pct."""
        state = _make_weight_state(
            fighters={
                "http://ufcstats.com/fighter/a": FighterWeightSnapshot(
                    first_class="Lightweight",
                    current_class="Lightweight",
                    migration_count=0,
                    fights_at_class={"Lightweight": 5},
                    wins_at_class={"Lightweight": 3},
                ),
            }
        )
        ctx = _make_context(weight_class="Lightweight", weight_state=state)
        result = WeightDominanceEmitter().emit(ctx)
        assert result["prior_class_win_pct"] is None


# ---------------------------------------------------------------------------
# Weight bully — top-quartile detection
# ---------------------------------------------------------------------------


class TestLargeForClass:
    """is_large_for_class delegates to WeightClassFrozenState."""

    def test_large_fighter_detected(self) -> None:
        """Both reach and height exceed 75th percentile -> 1.0."""
        state = _make_weight_state(
            fighters={
                "http://ufcstats.com/fighter/a": FighterWeightSnapshot(
                    first_class="Lightweight",
                    current_class="Lightweight",
                    migration_count=0,
                    fights_at_class={"Lightweight": 5},
                    wins_at_class={"Lightweight": 3},
                ),
            },
            thresholds={"Lightweight": {"reach_cm": 185.0, "height_cm": 178.0}},
        )
        # fighter has reach=190, height=180 -> both exceed thresholds
        ctx = _make_context(
            weight_class="Lightweight",
            weight_state=state,
            height_cm=180.0,
            reach_cm=190.0,
        )
        result = WeightDominanceEmitter().emit(ctx)
        assert result["is_large_for_class"] == 1.0

    def test_not_large_when_below_threshold(self) -> None:
        """Reach below 75th percentile -> 0.0."""
        state = _make_weight_state(
            fighters={
                "http://ufcstats.com/fighter/a": FighterWeightSnapshot(
                    first_class="Lightweight",
                    current_class="Lightweight",
                    migration_count=0,
                    fights_at_class={"Lightweight": 5},
                    wins_at_class={"Lightweight": 3},
                ),
            },
            thresholds={"Lightweight": {"reach_cm": 195.0, "height_cm": 178.0}},
        )
        # fighter has reach=190 < 195 threshold
        ctx = _make_context(
            weight_class="Lightweight",
            weight_state=state,
            height_cm=180.0,
            reach_cm=190.0,
        )
        result = WeightDominanceEmitter().emit(ctx)
        assert result["is_large_for_class"] == 0.0

    def test_missing_measurements_returns_zero(self) -> None:
        """Missing reach/height -> 0.0 (cannot determine size)."""
        state = _make_weight_state(
            fighters={
                "http://ufcstats.com/fighter/a": FighterWeightSnapshot(
                    first_class="Lightweight",
                    current_class="Lightweight",
                    migration_count=0,
                    fights_at_class={"Lightweight": 5},
                    wins_at_class={"Lightweight": 3},
                ),
            },
            thresholds={"Lightweight": {"reach_cm": 185.0, "height_cm": 178.0}},
        )
        ctx = _make_context(
            weight_class="Lightweight",
            weight_state=state,
            height_cm=None,
            reach_cm=None,
        )
        result = WeightDominanceEmitter().emit(ctx)
        assert result["is_large_for_class"] == 0.0


# ---------------------------------------------------------------------------
# Weight bully — grappling utilization rate
# ---------------------------------------------------------------------------


class TestGrapplingUtilizationRate:
    """grappling_utilization_rate = (TD attempts + control time) / fight count."""

    def test_computed_from_rolling_window(self) -> None:
        """Average (td_attempted + control_time) per fight."""
        window = (
            _make_fight_stats(takedowns_attempted=4, control_time_seconds=120),
            _make_fight_stats(takedowns_attempted=6, control_time_seconds=180),
        )
        rolling = _make_rolling_stats(
            fighter_windows={"http://ufcstats.com/fighter/a": window}
        )
        weight_state = _make_weight_state(
            fighters={
                "http://ufcstats.com/fighter/a": FighterWeightSnapshot(
                    first_class="Lightweight",
                    current_class="Lightweight",
                    migration_count=0,
                    fights_at_class={"Lightweight": 2},
                    wins_at_class={"Lightweight": 1},
                ),
            }
        )
        ctx = _make_context(
            weight_class="Lightweight",
            weight_state=weight_state,
            rolling_stats=rolling,
        )
        result = WeightDominanceEmitter().emit(ctx)
        # (4 + 120 + 6 + 180) / 2 = 155.0
        assert result["grappling_utilization_rate"] == 155.0

    def test_none_when_no_rolling_stats(self) -> None:
        """No rolling_stats component -> None."""
        weight_state = _make_weight_state(
            fighters={
                "http://ufcstats.com/fighter/a": FighterWeightSnapshot(
                    first_class="Lightweight",
                    current_class="Lightweight",
                    migration_count=0,
                    fights_at_class={"Lightweight": 2},
                    wins_at_class={"Lightweight": 1},
                ),
            }
        )
        ctx = _make_context(weight_class="Lightweight", weight_state=weight_state)
        result = WeightDominanceEmitter().emit(ctx)
        assert result["grappling_utilization_rate"] is None

    def test_none_when_fighter_not_in_window(self) -> None:
        """Fighter has no window entries -> None."""
        rolling = _make_rolling_stats(fighter_windows={})
        weight_state = _make_weight_state(
            fighters={
                "http://ufcstats.com/fighter/a": FighterWeightSnapshot(
                    first_class="Lightweight",
                    current_class="Lightweight",
                    migration_count=0,
                    fights_at_class={"Lightweight": 2},
                    wins_at_class={"Lightweight": 1},
                ),
            }
        )
        ctx = _make_context(
            weight_class="Lightweight",
            weight_state=weight_state,
            rolling_stats=rolling,
        )
        result = WeightDominanceEmitter().emit(ctx)
        assert result["grappling_utilization_rate"] is None


# ---------------------------------------------------------------------------
# Weight bully — product term
# ---------------------------------------------------------------------------


class TestWeightBullyScore:
    """weight_bully_score = is_large_for_class * grappling_utilization_rate."""

    def test_product_term_positive(self) -> None:
        """Large fighter with grappling activity produces non-zero score."""
        window = (
            _make_fight_stats(takedowns_attempted=5, control_time_seconds=200),
            _make_fight_stats(takedowns_attempted=3, control_time_seconds=100),
        )
        rolling = _make_rolling_stats(
            fighter_windows={"http://ufcstats.com/fighter/a": window}
        )
        weight_state = _make_weight_state(
            fighters={
                "http://ufcstats.com/fighter/a": FighterWeightSnapshot(
                    first_class="Lightweight",
                    current_class="Lightweight",
                    migration_count=0,
                    fights_at_class={"Lightweight": 2},
                    wins_at_class={"Lightweight": 1},
                ),
            },
            thresholds={"Lightweight": {"reach_cm": 185.0, "height_cm": 178.0}},
        )
        ctx = _make_context(
            weight_class="Lightweight",
            weight_state=weight_state,
            rolling_stats=rolling,
            height_cm=180.0,
            reach_cm=190.0,
        )
        result = WeightDominanceEmitter().emit(ctx)
        # is_large = 1.0, grappling_util = (5+200+3+100)/2 = 154.0
        # score = 1.0 * 154.0 = 154.0
        assert result["weight_bully_score"] == 154.0

    def test_not_large_produces_zero_score(self) -> None:
        """Non-large fighter -> score is 0.0 regardless of grappling."""
        window = (
            _make_fight_stats(takedowns_attempted=5, control_time_seconds=200),
        )
        rolling = _make_rolling_stats(
            fighter_windows={"http://ufcstats.com/fighter/a": window}
        )
        weight_state = _make_weight_state(
            fighters={
                "http://ufcstats.com/fighter/a": FighterWeightSnapshot(
                    first_class="Lightweight",
                    current_class="Lightweight",
                    migration_count=0,
                    fights_at_class={"Lightweight": 1},
                    wins_at_class={},
                ),
            },
            # threshold higher than fighter's measurements
            thresholds={"Lightweight": {"reach_cm": 195.0, "height_cm": 185.0}},
        )
        ctx = _make_context(
            weight_class="Lightweight",
            weight_state=weight_state,
            rolling_stats=rolling,
            height_cm=180.0,
            reach_cm=190.0,
        )
        result = WeightDominanceEmitter().emit(ctx)
        assert result["weight_bully_score"] == 0.0

    def test_none_when_grappling_unavailable(self) -> None:
        """Missing grappling rate -> score is None."""
        weight_state = _make_weight_state(
            fighters={
                "http://ufcstats.com/fighter/a": FighterWeightSnapshot(
                    first_class="Lightweight",
                    current_class="Lightweight",
                    migration_count=0,
                    fights_at_class={"Lightweight": 5},
                    wins_at_class={"Lightweight": 3},
                ),
            },
            thresholds={"Lightweight": {"reach_cm": 185.0, "height_cm": 178.0}},
        )
        ctx = _make_context(
            weight_class="Lightweight",
            weight_state=weight_state,
            height_cm=180.0,
            reach_cm=190.0,
        )
        result = WeightDominanceEmitter().emit(ctx)
        assert result["weight_bully_score"] is None
