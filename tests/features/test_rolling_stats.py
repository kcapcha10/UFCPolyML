"""Tests for RollingStatsAccumulator state component.

Validates rolling-window averages over per-fighter fight statistics,
partial windows, empty state, variance computation, and freeze immutability.
"""

from __future__ import annotations

from datetime import date

import pytest

from ufc_edge.features.components.rolling_stats import (
    FightStats,
    RollingStatsAccumulator,
)
from ufc_edge.features.contracts import FightOutcomeView, FightTotals


def _make_outcome(
    fight_url: str = "http://fight/1",
    event_date: date = date(2024, 6, 1),
    fighter_a: str = "http://fighter/a",
    fighter_b: str = "http://fighter/b",
    winner: str | None = "http://fighter/a",
    method: str = "Decision - Unanimous",
    ending_round: int = 3,
    ending_time: str = "5:00",
    weight_class: str = "Lightweight",
) -> FightOutcomeView:
    return FightOutcomeView(
        fight_url=fight_url,
        event_url="http://event/1",
        event_date=event_date,
        fighter_a_url=fighter_a,
        fighter_b_url=fighter_b,
        winner_url=winner,
        method=method,
        ending_round=ending_round,
        ending_time=ending_time,
        weight_class=weight_class,
        bout_order=None,
    )


def _make_totals(
    sig_strikes_landed: int = 50,
    sig_strikes_attempted: int = 100,
    total_strikes_landed: int = 80,
    total_strikes_attempted: int = 150,
    takedowns_landed: int = 3,
    takedowns_attempted: int = 5,
    submissions_attempted: int = 1,
    knockdowns: int = 1,
    reversals: int = 0,
    control_time_seconds: int = 120,
) -> FightTotals:
    return FightTotals(
        knockdowns=knockdowns,
        sig_strikes_landed=sig_strikes_landed,
        sig_strikes_attempted=sig_strikes_attempted,
        total_strikes_landed=total_strikes_landed,
        total_strikes_attempted=total_strikes_attempted,
        takedowns_landed=takedowns_landed,
        takedowns_attempted=takedowns_attempted,
        submissions_attempted=submissions_attempted,
        reversals=reversals,
        control_time_seconds=control_time_seconds,
    )


class TestFightStats:
    """FightStats captures derived per-fight metrics from raw totals and duration."""

    def test_from_totals_computes_fight_duration_minutes(self) -> None:
        stats = FightStats.from_totals(
            totals=_make_totals(),
            opponent_totals=_make_totals(),
            ending_round=3,
            ending_time="5:00",
        )
        # 3 rounds × 5 min = 15 min
        assert stats.fight_duration_minutes == pytest.approx(15.0)

    def test_from_totals_computes_partial_round_duration(self) -> None:
        stats = FightStats.from_totals(
            totals=_make_totals(),
            opponent_totals=_make_totals(),
            ending_round=2,
            ending_time="3:30",
        )
        # 1 full round (5 min) + 3:30 = 8.5 min
        assert stats.fight_duration_minutes == pytest.approx(8.5)

    def test_from_totals_stores_striking_fields(self) -> None:
        stats = FightStats.from_totals(
            totals=_make_totals(sig_strikes_landed=60, sig_strikes_attempted=120),
            opponent_totals=_make_totals(sig_strikes_landed=40),
            ending_round=3,
            ending_time="5:00",
        )
        assert stats.sig_strikes_landed == 60
        assert stats.sig_strikes_attempted == 120
        assert stats.sig_strikes_absorbed == 40

    def test_from_totals_stores_grappling_fields(self) -> None:
        stats = FightStats.from_totals(
            totals=_make_totals(takedowns_landed=4, takedowns_attempted=8),
            opponent_totals=_make_totals(takedowns_landed=2, control_time_seconds=60),
            ending_round=3,
            ending_time="5:00",
        )
        assert stats.takedowns_landed == 4
        assert stats.takedowns_attempted == 8
        assert stats.opponent_takedowns_landed == 2
        assert stats.opponent_control_time_seconds == 60

    def test_from_totals_with_none_totals_returns_none(self) -> None:
        result = FightStats.from_totals(
            totals=None,
            opponent_totals=None,
            ending_round=3,
            ending_time="5:00",
        )
        assert result is None


class TestRollingStatsAccumulatorEmpty:
    """Empty accumulator returns None for all rolling averages."""

    def test_freeze_empty_returns_none_for_fighter(self) -> None:
        acc = RollingStatsAccumulator(window_size=5)
        snapshot = acc.freeze()
        result = snapshot.get_rolling_averages("http://fighter/unknown")
        assert result is None

    def test_freeze_empty_returns_none_variance(self) -> None:
        acc = RollingStatsAccumulator(window_size=5)
        snapshot = acc.freeze()
        result = snapshot.get_rolling_variance("http://fighter/unknown")
        assert result is None


class TestRollingStatsAccumulatorPartialWindow:
    """Partial window (fewer fights than window_size) still returns valid averages."""

    def test_single_fight_returns_that_fights_stats(self) -> None:
        acc = RollingStatsAccumulator(window_size=5)
        outcome = _make_outcome(ending_round=3, ending_time="5:00")
        totals_a = _make_totals(sig_strikes_landed=60, sig_strikes_attempted=100)
        totals_b = _make_totals(sig_strikes_landed=40, sig_strikes_attempted=80)
        acc.update_with_totals(outcome, totals_a, totals_b)

        snapshot = acc.freeze()
        avgs = snapshot.get_rolling_averages("http://fighter/a")
        assert avgs is not None
        # 60 sig strikes / 15 min = 4.0 per min
        assert avgs["sig_strikes_per_min"] == pytest.approx(4.0)
        # 60/100 = 0.6
        assert avgs["striking_accuracy_pct"] == pytest.approx(0.6)

    def test_two_fights_averages_correctly(self) -> None:
        acc = RollingStatsAccumulator(window_size=5)

        # Fight 1: 3 round decision
        outcome1 = _make_outcome(
            fight_url="http://fight/1", ending_round=3, ending_time="5:00"
        )
        totals_a1 = _make_totals(sig_strikes_landed=30, sig_strikes_attempted=60)
        totals_b1 = _make_totals(sig_strikes_landed=20, sig_strikes_attempted=50)
        acc.update_with_totals(outcome1, totals_a1, totals_b1)

        # Fight 2: 2nd round TKO
        outcome2 = _make_outcome(
            fight_url="http://fight/2",
            ending_round=2,
            ending_time="3:00",
            method="KO/TKO",
        )
        totals_a2 = _make_totals(sig_strikes_landed=60, sig_strikes_attempted=90)
        totals_b2 = _make_totals(sig_strikes_landed=10, sig_strikes_attempted=30)
        acc.update_with_totals(outcome2, totals_a2, totals_b2)

        snapshot = acc.freeze()
        avgs = snapshot.get_rolling_averages("http://fighter/a")
        assert avgs is not None

        # Fight 1: 30/15min = 2.0; Fight 2: 60/8min = 7.5; avg = (2.0+7.5)/2 = 4.75
        assert avgs["sig_strikes_per_min"] == pytest.approx(4.75)


class TestRollingStatsAccumulatorFullWindow:
    """Full window: oldest fight drops when new one enters."""

    def test_window_evicts_oldest_entry(self) -> None:
        acc = RollingStatsAccumulator(window_size=2)

        # Fight 1
        outcome1 = _make_outcome(fight_url="http://fight/1", ending_round=3, ending_time="5:00")
        totals_a1 = _make_totals(sig_strikes_landed=30, sig_strikes_attempted=100)
        totals_b1 = _make_totals(sig_strikes_landed=20)
        acc.update_with_totals(outcome1, totals_a1, totals_b1)

        # Fight 2
        outcome2 = _make_outcome(fight_url="http://fight/2", ending_round=3, ending_time="5:00")
        totals_a2 = _make_totals(sig_strikes_landed=60, sig_strikes_attempted=100)
        totals_b2 = _make_totals(sig_strikes_landed=30)
        acc.update_with_totals(outcome2, totals_a2, totals_b2)

        # Fight 3 — pushes fight 1 out of window
        outcome3 = _make_outcome(fight_url="http://fight/3", ending_round=3, ending_time="5:00")
        totals_a3 = _make_totals(sig_strikes_landed=90, sig_strikes_attempted=100)
        totals_b3 = _make_totals(sig_strikes_landed=10)
        acc.update_with_totals(outcome3, totals_a3, totals_b3)

        snapshot = acc.freeze()
        avgs = snapshot.get_rolling_averages("http://fighter/a")
        assert avgs is not None

        # Only fights 2 and 3: (60/15 + 90/15) / 2 = (4.0 + 6.0) / 2 = 5.0
        assert avgs["sig_strikes_per_min"] == pytest.approx(5.0)


class TestRollingStatsVariance:
    """Variance computation requires at least 2 fights; single fight returns None."""

    def test_variance_none_with_single_fight(self) -> None:
        acc = RollingStatsAccumulator(window_size=5)
        outcome = _make_outcome(ending_round=3, ending_time="5:00")
        totals_a = _make_totals(sig_strikes_landed=60)
        totals_b = _make_totals(sig_strikes_landed=40)
        acc.update_with_totals(outcome, totals_a, totals_b)

        snapshot = acc.freeze()
        var = snapshot.get_rolling_variance("http://fighter/a")
        assert var is None

    def test_variance_computed_with_two_fights(self) -> None:
        acc = RollingStatsAccumulator(window_size=5)

        # Fight 1: sig_strikes_per_min = 30/15 = 2.0
        outcome1 = _make_outcome(fight_url="http://fight/1", ending_round=3, ending_time="5:00")
        totals_a1 = _make_totals(sig_strikes_landed=30, sig_strikes_attempted=60)
        totals_b1 = _make_totals(sig_strikes_landed=20)
        acc.update_with_totals(outcome1, totals_a1, totals_b1)

        # Fight 2: sig_strikes_per_min = 60/15 = 4.0
        outcome2 = _make_outcome(fight_url="http://fight/2", ending_round=3, ending_time="5:00")
        totals_a2 = _make_totals(sig_strikes_landed=60, sig_strikes_attempted=90)
        totals_b2 = _make_totals(sig_strikes_landed=30)
        acc.update_with_totals(outcome2, totals_a2, totals_b2)

        snapshot = acc.freeze()
        var = snapshot.get_rolling_variance("http://fighter/a")
        assert var is not None

        # mean = 3.0; var = ((2.0-3.0)^2 + (4.0-3.0)^2) / 2 = 1.0
        assert var["sig_strikes_per_min"] == pytest.approx(1.0)


class TestRollingStatsSnapshotFrozen:
    """Snapshot from freeze() must be immutable."""

    def test_cannot_set_attribute(self) -> None:
        acc = RollingStatsAccumulator(window_size=5)
        snapshot = acc.freeze()
        with pytest.raises(AttributeError):
            snapshot.some_attr = "anything"  # type: ignore[attr-defined]

    def test_internal_data_not_mutable_through_snapshot(self) -> None:
        acc = RollingStatsAccumulator(window_size=5)
        outcome = _make_outcome(ending_round=3, ending_time="5:00")
        totals_a = _make_totals(sig_strikes_landed=60)
        totals_b = _make_totals(sig_strikes_landed=40)
        acc.update_with_totals(outcome, totals_a, totals_b)

        snapshot = acc.freeze()
        # Mutating the accumulator after freeze should not affect snapshot
        outcome2 = _make_outcome(
            fight_url="http://fight/2", ending_round=1, ending_time="1:00"
        )
        totals_a2 = _make_totals(sig_strikes_landed=100)
        totals_b2 = _make_totals(sig_strikes_landed=5)
        acc.update_with_totals(outcome2, totals_a2, totals_b2)

        # Snapshot still reflects state at freeze time (1 fight)
        avgs = snapshot.get_rolling_averages("http://fighter/a")
        assert avgs is not None
        assert avgs["sig_strikes_per_min"] == pytest.approx(4.0)  # 60 / 15min


class TestRollingStatsOutputMetrics:
    """All expected output metrics are present in rolling averages."""

    def test_all_output_metrics_present(self) -> None:
        acc = RollingStatsAccumulator(window_size=5)
        outcome = _make_outcome(ending_round=3, ending_time="5:00")
        totals_a = _make_totals(
            sig_strikes_landed=60,
            sig_strikes_attempted=100,
            total_strikes_landed=80,
            total_strikes_attempted=150,
            takedowns_landed=4,
            takedowns_attempted=6,
            submissions_attempted=2,
            knockdowns=1,
            control_time_seconds=180,
        )
        totals_b = _make_totals(
            sig_strikes_landed=40,
            sig_strikes_attempted=80,
            takedowns_landed=2,
            takedowns_attempted=5,
            control_time_seconds=60,
        )
        acc.update_with_totals(outcome, totals_a, totals_b)

        snapshot = acc.freeze()
        avgs = snapshot.get_rolling_averages("http://fighter/a")
        assert avgs is not None

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
        assert expected_keys.issubset(avgs.keys())

    def test_damage_ratio_computation(self) -> None:
        acc = RollingStatsAccumulator(window_size=5)
        outcome = _make_outcome(ending_round=3, ending_time="5:00")
        totals_a = _make_totals(sig_strikes_landed=80)
        totals_b = _make_totals(sig_strikes_landed=40)
        acc.update_with_totals(outcome, totals_a, totals_b)

        snapshot = acc.freeze()
        avgs = snapshot.get_rolling_averages("http://fighter/a")
        assert avgs is not None
        # damage_ratio = sig_landed / sig_absorbed = 80/40 = 2.0
        assert avgs["damage_ratio"] == pytest.approx(2.0)

    def test_grappling_dominance_computation(self) -> None:
        acc = RollingStatsAccumulator(window_size=5)
        outcome = _make_outcome(ending_round=3, ending_time="5:00")
        # Fighter A: TD landed=4, control=180s; opponent: TD landed=2, control=60s
        totals_a = _make_totals(takedowns_landed=4, control_time_seconds=180)
        totals_b = _make_totals(takedowns_landed=2, control_time_seconds=60)
        acc.update_with_totals(outcome, totals_a, totals_b)

        snapshot = acc.freeze()
        avgs = snapshot.get_rolling_averages("http://fighter/a")
        assert avgs is not None
        # grappling_dominance = (TD_landed + control_time) / (opp_TD_landed + opp_control)
        # = (4 + 180) / (2 + 60) = 184/62 ≈ 2.9677
        assert avgs["grappling_dominance"] == pytest.approx(184.0 / 62.0)

    def test_td_per_15min_computation(self) -> None:
        acc = RollingStatsAccumulator(window_size=5)
        outcome = _make_outcome(ending_round=3, ending_time="5:00")
        totals_a = _make_totals(takedowns_landed=3, takedowns_attempted=5)
        totals_b = _make_totals()
        acc.update_with_totals(outcome, totals_a, totals_b)

        snapshot = acc.freeze()
        avgs = snapshot.get_rolling_averages("http://fighter/a")
        assert avgs is not None
        # td_per_15min = takedowns_landed / fight_duration * 15 = 3/15 * 15 = 3.0
        assert avgs["td_per_15min"] == pytest.approx(3.0)

    def test_zero_division_handled_for_damage_ratio(self) -> None:
        acc = RollingStatsAccumulator(window_size=5)
        outcome = _make_outcome(ending_round=3, ending_time="5:00")
        totals_a = _make_totals(sig_strikes_landed=60)
        # Opponent landed 0 sig strikes
        totals_b = _make_totals(sig_strikes_landed=0)
        acc.update_with_totals(outcome, totals_a, totals_b)

        snapshot = acc.freeze()
        avgs = snapshot.get_rolling_averages("http://fighter/a")
        assert avgs is not None
        # When opponent landed 0, damage_ratio should be None (undefined)
        assert avgs["damage_ratio"] is None


class TestRollingStatsUpdatesForBothFighters:
    """update_with_totals records stats for both fighters in the fight."""

    def test_both_fighters_tracked(self) -> None:
        acc = RollingStatsAccumulator(window_size=5)
        outcome = _make_outcome(ending_round=3, ending_time="5:00")
        totals_a = _make_totals(sig_strikes_landed=60)
        totals_b = _make_totals(sig_strikes_landed=40)
        acc.update_with_totals(outcome, totals_a, totals_b)

        snapshot = acc.freeze()
        avgs_a = snapshot.get_rolling_averages("http://fighter/a")
        avgs_b = snapshot.get_rolling_averages("http://fighter/b")
        assert avgs_a is not None
        assert avgs_b is not None
        # Fighter A absorbed what B landed, and vice versa
        assert avgs_a["sig_strikes_per_min"] == pytest.approx(4.0)  # 60/15
        assert avgs_b["sig_strikes_per_min"] == pytest.approx(40.0 / 15.0)


class TestRollingStatsConfigurableWindow:
    """Window size is configurable via constructor."""

    def test_custom_window_size(self) -> None:
        acc = RollingStatsAccumulator(window_size=3)
        base_totals = _make_totals(sig_strikes_landed=30)
        opp_totals = _make_totals(sig_strikes_landed=20)

        for i in range(5):
            outcome = _make_outcome(
                fight_url=f"http://fight/{i}", ending_round=3, ending_time="5:00"
            )
            acc.update_with_totals(outcome, base_totals, opp_totals)

        snapshot = acc.freeze()
        avgs = snapshot.get_rolling_averages("http://fighter/a")
        assert avgs is not None
        # Only last 3 fights matter, all identical: 30/15 = 2.0
        assert avgs["sig_strikes_per_min"] == pytest.approx(2.0)
