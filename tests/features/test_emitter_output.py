"""Tests for OutputEmitter — thin adapter over RollingStatsSnapshot.

Validates that the emitter correctly reads frozen rolling averages, prefixes
column names, handles missing fighter history (returns all None), and handles
zero-denominator edge cases via the underlying snapshot.
"""

from __future__ import annotations

from datetime import date

import pytest

from ufc_edge.features.components.rolling_stats import (
    FightStats,
    RollingStatsAccumulator,
    RollingStatsSnapshot,
)
from ufc_edge.features.contracts import EmitContext, FighterProfile, FrozenState
from ufc_edge.features.emitters.output import OutputEmitter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIGHTER_A = "http://fighter/a"
FIGHTER_B = "http://fighter/b"


def _profile(url: str = FIGHTER_A) -> FighterProfile:
    return FighterProfile(fighter_url=url, height_cm=180.0, reach_cm=185.0)


def _make_snapshot(
    fighter_url: str, fights: list[FightStats]
) -> RollingStatsSnapshot:
    """Build a snapshot with a single fighter's window pre-populated."""
    return RollingStatsSnapshot({fighter_url: tuple(fights)})


def _make_context(
    snapshot: RollingStatsSnapshot,
    fighter_url: str = FIGHTER_A,
    opponent_url: str = FIGHTER_B,
) -> EmitContext:
    """Build an EmitContext with rolling_stats component populated."""
    components: dict[str, FrozenState] = {"rolling_stats": snapshot}
    return EmitContext(
        fighter_url=fighter_url,
        fighter_profile=_profile(fighter_url),
        opponent_url=opponent_url,
        opponent_profile=_profile(opponent_url),
        event_date=date(2024, 6, 1),
        event_url="http://event/1",
        weight_class="Lightweight",
        fight_url="http://fight/1",
        bout_order=None,
        components=components,
    )


def _fight_stats(
    fight_duration_minutes: float = 15.0,
    sig_strikes_landed: int = 60,
    sig_strikes_attempted: int = 100,
    sig_strikes_absorbed: int = 40,
    sig_strikes_absorbed_attempted: int = 80,
    total_strikes_landed: int = 80,
    total_strikes_attempted: int = 150,
    takedowns_landed: int = 3,
    takedowns_attempted: int = 5,
    submissions_attempted: int = 1,
    knockdowns: int = 1,
    control_time_seconds: int = 120,
    opponent_takedowns_landed: int = 2,
    opponent_takedowns_attempted: int = 4,
    opponent_control_time_seconds: int = 60,
) -> FightStats:
    return FightStats(
        fight_duration_minutes=fight_duration_minutes,
        sig_strikes_landed=sig_strikes_landed,
        sig_strikes_attempted=sig_strikes_attempted,
        sig_strikes_absorbed=sig_strikes_absorbed,
        sig_strikes_absorbed_attempted=sig_strikes_absorbed_attempted,
        total_strikes_landed=total_strikes_landed,
        total_strikes_attempted=total_strikes_attempted,
        takedowns_landed=takedowns_landed,
        takedowns_attempted=takedowns_attempted,
        submissions_attempted=submissions_attempted,
        knockdowns=knockdowns,
        control_time_seconds=control_time_seconds,
        opponent_takedowns_landed=opponent_takedowns_landed,
        opponent_takedowns_attempted=opponent_takedowns_attempted,
        opponent_control_time_seconds=opponent_control_time_seconds,
    )


# ---------------------------------------------------------------------------
# Tests: Basic protocol conformance
# ---------------------------------------------------------------------------


class TestOutputEmitterProtocol:
    """OutputEmitter conforms to the FeatureEmitter protocol."""

    def test_has_name_attribute(self) -> None:
        emitter = OutputEmitter()
        assert emitter.name == "output"

    def test_emit_returns_dict(self) -> None:
        snapshot = _make_snapshot(FIGHTER_A, [_fight_stats()])
        ctx = _make_context(snapshot)
        emitter = OutputEmitter()
        result = emitter.emit(ctx)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Tests: All expected keys present
# ---------------------------------------------------------------------------


class TestOutputEmitterKeys:
    """Emitter returns all expected output feature columns."""

    def test_all_expected_keys_present(self) -> None:
        snapshot = _make_snapshot(FIGHTER_A, [_fight_stats()])
        ctx = _make_context(snapshot)
        emitter = OutputEmitter()
        result = emitter.emit(ctx)

        expected_keys = {
            "sig_strikes_per_min",
            "sig_strikes_absorbed_per_min",
            "striking_accuracy_pct",
            "striking_defense_pct",
            "td_per_15min",
            "td_accuracy_pct",
            "td_defense_pct",
            "sub_attempts_per_15min",
            "knockdown_rate",
            "damage_ratio",
            "grappling_dominance",
            "control_time_per_fight",
        }
        assert expected_keys == set(result.keys())


# ---------------------------------------------------------------------------
# Tests: Rolling window correctness
# ---------------------------------------------------------------------------


class TestOutputEmitterRollingWindow:
    """Emitter reads rolling averages from the snapshot correctly."""

    def test_single_fight_values_match_rolling_averages(self) -> None:
        stats = _fight_stats(
            fight_duration_minutes=15.0,
            sig_strikes_landed=60,
            sig_strikes_attempted=100,
            sig_strikes_absorbed=40,
            sig_strikes_absorbed_attempted=80,
            takedowns_landed=3,
            takedowns_attempted=5,
            submissions_attempted=1,
            knockdowns=1,
            control_time_seconds=120,
            opponent_takedowns_landed=2,
            opponent_takedowns_attempted=4,
            opponent_control_time_seconds=60,
        )
        snapshot = _make_snapshot(FIGHTER_A, [stats])
        ctx = _make_context(snapshot)
        emitter = OutputEmitter()
        result = emitter.emit(ctx)

        assert result["sig_strikes_per_min"] == pytest.approx(4.0)
        assert result["sig_strikes_absorbed_per_min"] == pytest.approx(40.0 / 15.0)
        assert result["striking_accuracy_pct"] == pytest.approx(0.6)
        # defense: (80 - 40) / 80 = 0.5
        assert result["striking_defense_pct"] == pytest.approx(0.5)
        # td_per_15min: 3 * 15 / 15 = 3.0
        assert result["td_per_15min"] == pytest.approx(3.0)
        # td_accuracy: 3/5 = 0.6
        assert result["td_accuracy_pct"] == pytest.approx(0.6)
        # td_defense: (4 - 2) / 4 = 0.5
        assert result["td_defense_pct"] == pytest.approx(0.5)
        # sub_per_15min: 1 * 15 / 15 = 1.0
        assert result["sub_attempts_per_15min"] == pytest.approx(1.0)
        # knockdown_rate: 1/100 = 0.01
        assert result["knockdown_rate"] == pytest.approx(0.01)
        # control_time_per_fight: 120.0
        assert result["control_time_per_fight"] == pytest.approx(120.0)

    def test_multiple_fight_window_averages(self) -> None:
        stats1 = _fight_stats(
            fight_duration_minutes=15.0,
            sig_strikes_landed=30,
            sig_strikes_attempted=60,
            sig_strikes_absorbed=20,
            sig_strikes_absorbed_attempted=50,
            takedowns_landed=2,
            takedowns_attempted=4,
            submissions_attempted=1,
            knockdowns=0,
            control_time_seconds=100,
            opponent_takedowns_landed=1,
            opponent_takedowns_attempted=3,
            opponent_control_time_seconds=40,
        )
        stats2 = _fight_stats(
            fight_duration_minutes=8.0,
            sig_strikes_landed=60,
            sig_strikes_attempted=90,
            sig_strikes_absorbed=10,
            sig_strikes_absorbed_attempted=30,
            takedowns_landed=4,
            takedowns_attempted=6,
            submissions_attempted=2,
            knockdowns=2,
            control_time_seconds=200,
            opponent_takedowns_landed=0,
            opponent_takedowns_attempted=2,
            opponent_control_time_seconds=20,
        )
        snapshot = _make_snapshot(FIGHTER_A, [stats1, stats2])
        ctx = _make_context(snapshot)
        emitter = OutputEmitter()
        result = emitter.emit(ctx)

        # sig_strikes_per_min: fight1=30/15=2.0, fight2=60/8=7.5, avg=4.75
        assert result["sig_strikes_per_min"] == pytest.approx(4.75)
        # control_time_per_fight: (100 + 200) / 2 = 150.0
        assert result["control_time_per_fight"] == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# Tests: damage_ratio formula
# ---------------------------------------------------------------------------


class TestOutputEmitterDamageRatio:
    """damage_ratio = sig_strikes_landed / sig_strikes_absorbed."""

    def test_basic_damage_ratio(self) -> None:
        stats = _fight_stats(sig_strikes_landed=80, sig_strikes_absorbed=40)
        snapshot = _make_snapshot(FIGHTER_A, [stats])
        ctx = _make_context(snapshot)
        emitter = OutputEmitter()
        result = emitter.emit(ctx)

        assert result["damage_ratio"] == pytest.approx(2.0)

    def test_damage_ratio_less_than_one(self) -> None:
        stats = _fight_stats(sig_strikes_landed=20, sig_strikes_absorbed=60)
        snapshot = _make_snapshot(FIGHTER_A, [stats])
        ctx = _make_context(snapshot)
        emitter = OutputEmitter()
        result = emitter.emit(ctx)

        assert result["damage_ratio"] == pytest.approx(20.0 / 60.0)

    def test_damage_ratio_averaged_over_window(self) -> None:
        stats1 = _fight_stats(sig_strikes_landed=60, sig_strikes_absorbed=30)
        stats2 = _fight_stats(sig_strikes_landed=40, sig_strikes_absorbed=40)
        snapshot = _make_snapshot(FIGHTER_A, [stats1, stats2])
        ctx = _make_context(snapshot)
        emitter = OutputEmitter()
        result = emitter.emit(ctx)

        # fight1: 60/30=2.0, fight2: 40/40=1.0, avg=1.5
        assert result["damage_ratio"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Tests: Zero denominator → None
# ---------------------------------------------------------------------------


class TestOutputEmitterZeroDenominator:
    """Zero denominators produce None rather than division errors."""

    def test_damage_ratio_zero_absorbed_yields_none(self) -> None:
        stats = _fight_stats(sig_strikes_landed=60, sig_strikes_absorbed=0)
        snapshot = _make_snapshot(FIGHTER_A, [stats])
        ctx = _make_context(snapshot)
        emitter = OutputEmitter()
        result = emitter.emit(ctx)

        assert result["damage_ratio"] is None

    def test_grappling_dominance_zero_opponent_yields_none(self) -> None:
        stats = _fight_stats(
            takedowns_landed=3,
            control_time_seconds=120,
            opponent_takedowns_landed=0,
            opponent_control_time_seconds=0,
        )
        snapshot = _make_snapshot(FIGHTER_A, [stats])
        ctx = _make_context(snapshot)
        emitter = OutputEmitter()
        result = emitter.emit(ctx)

        assert result["grappling_dominance"] is None

    def test_td_accuracy_zero_attempts_yields_none(self) -> None:
        stats = _fight_stats(takedowns_landed=0, takedowns_attempted=0)
        snapshot = _make_snapshot(FIGHTER_A, [stats])
        ctx = _make_context(snapshot)
        emitter = OutputEmitter()
        result = emitter.emit(ctx)

        assert result["td_accuracy_pct"] is None

    def test_knockdown_rate_zero_attempts_yields_none(self) -> None:
        stats = _fight_stats(knockdowns=0, sig_strikes_attempted=0)
        snapshot = _make_snapshot(FIGHTER_A, [stats])
        ctx = _make_context(snapshot)
        emitter = OutputEmitter()
        result = emitter.emit(ctx)

        assert result["knockdown_rate"] is None

    def test_striking_defense_zero_absorbed_attempts_yields_none(self) -> None:
        stats = _fight_stats(sig_strikes_absorbed=0, sig_strikes_absorbed_attempted=0)
        snapshot = _make_snapshot(FIGHTER_A, [stats])
        ctx = _make_context(snapshot)
        emitter = OutputEmitter()
        result = emitter.emit(ctx)

        assert result["striking_defense_pct"] is None

    def test_td_defense_zero_opponent_attempts_yields_none(self) -> None:
        stats = _fight_stats(opponent_takedowns_landed=0, opponent_takedowns_attempted=0)
        snapshot = _make_snapshot(FIGHTER_A, [stats])
        ctx = _make_context(snapshot)
        emitter = OutputEmitter()
        result = emitter.emit(ctx)

        assert result["td_defense_pct"] is None


# ---------------------------------------------------------------------------
# Tests: No fighter history → all None
# ---------------------------------------------------------------------------


class TestOutputEmitterNoHistory:
    """Fighter with no rolling history returns all None values."""

    def test_unknown_fighter_returns_all_none(self) -> None:
        snapshot = RollingStatsSnapshot({})
        ctx = _make_context(snapshot, fighter_url=FIGHTER_A)
        emitter = OutputEmitter()
        result = emitter.emit(ctx)

        for key, value in result.items():
            assert value is None, f"Expected None for {key}, got {value}"

    def test_missing_component_returns_all_none(self) -> None:
        """EmitContext with no rolling_stats component returns all None."""
        ctx = EmitContext(
            fighter_url=FIGHTER_A,
            fighter_profile=_profile(FIGHTER_A),
            opponent_url=FIGHTER_B,
            opponent_profile=_profile(FIGHTER_B),
            event_date=date(2024, 6, 1),
            event_url="http://event/1",
            weight_class="Lightweight",
            fight_url="http://fight/1",
            bout_order=None,
            components={},
        )
        emitter = OutputEmitter()
        result = emitter.emit(ctx)

        for key, value in result.items():
            assert value is None, f"Expected None for {key}, got {value}"


# ---------------------------------------------------------------------------
# Tests: Grappling dominance formula
# ---------------------------------------------------------------------------


class TestOutputEmitterGrapplingDominance:
    """grappling_dominance = (TD_landed + control) / (opp_TD_landed + opp_control)."""

    def test_grappling_dominance_value(self) -> None:
        stats = _fight_stats(
            takedowns_landed=4,
            control_time_seconds=180,
            opponent_takedowns_landed=2,
            opponent_control_time_seconds=60,
        )
        snapshot = _make_snapshot(FIGHTER_A, [stats])
        ctx = _make_context(snapshot)
        emitter = OutputEmitter()
        result = emitter.emit(ctx)

        # (4 + 180) / (2 + 60) = 184/62
        assert result["grappling_dominance"] == pytest.approx(184.0 / 62.0)


# ---------------------------------------------------------------------------
# Tests: Integration with RollingStatsAccumulator
# ---------------------------------------------------------------------------


class TestOutputEmitterIntegration:
    """OutputEmitter reads from a real RollingStatsAccumulator freeze."""

    def test_end_to_end_with_accumulator(self) -> None:
        from ufc_edge.features.contracts import FightOutcomeView, FightTotals

        acc = RollingStatsAccumulator(window_size=5)
        outcome = FightOutcomeView(
            fight_url="http://fight/1",
            event_url="http://event/1",
            event_date=date(2024, 6, 1),
            fighter_a_url=FIGHTER_A,
            fighter_b_url=FIGHTER_B,
            winner_url=FIGHTER_A,
            method="Decision - Unanimous",
            ending_round=3,
            ending_time="5:00",
            weight_class="Lightweight",
            bout_order=None,
        )
        totals_a = FightTotals(
            knockdowns=2,
            sig_strikes_landed=75,
            sig_strikes_attempted=120,
            total_strikes_landed=100,
            total_strikes_attempted=180,
            takedowns_landed=5,
            takedowns_attempted=8,
            submissions_attempted=1,
            reversals=0,
            control_time_seconds=200,
        )
        totals_b = FightTotals(
            knockdowns=0,
            sig_strikes_landed=50,
            sig_strikes_attempted=90,
            total_strikes_landed=70,
            total_strikes_attempted=130,
            takedowns_landed=1,
            takedowns_attempted=4,
            submissions_attempted=0,
            reversals=0,
            control_time_seconds=30,
        )
        acc.update_with_totals(outcome, totals_a, totals_b)

        snapshot = acc.freeze()
        ctx = _make_context(snapshot, fighter_url=FIGHTER_A)
        emitter = OutputEmitter()
        result = emitter.emit(ctx)

        # Verify key values from accumulator
        assert result["sig_strikes_per_min"] == pytest.approx(75.0 / 15.0)
        assert result["striking_accuracy_pct"] == pytest.approx(75.0 / 120.0)
        assert result["damage_ratio"] == pytest.approx(75.0 / 50.0)
        assert result["knockdown_rate"] == pytest.approx(2.0 / 120.0)
        assert result["control_time_per_fight"] == pytest.approx(200.0)
