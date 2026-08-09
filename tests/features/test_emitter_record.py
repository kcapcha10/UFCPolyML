"""Tests for RecordEmitter feature emitter.

Validates win percentage calculations, streak sign convention, debut flag logic,
contender series detection, and zero-fights-returns-None behavior.
"""

from __future__ import annotations

from datetime import date

import pytest

from ufc_edge.features.components.career import CareerAccumulator, CareerSnapshot
from ufc_edge.features.contracts import (
    EmitContext,
    FighterProfile,
    FightOutcomeView,
)
from ufc_edge.features.emitters.record import RecordEmitter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIGHTER_A = "http://fighter/a"
FIGHTER_B = "http://fighter/b"
FIGHTER_C = "http://fighter/c"


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


def _profile(fighter_url: str = FIGHTER_A) -> FighterProfile:
    return FighterProfile(fighter_url=fighter_url)


def _build_context(
    career_snap: CareerSnapshot,
    *,
    fighter_url: str = FIGHTER_A,
    opponent_url: str = FIGHTER_B,
    event_url: str = "http://event/99",
) -> EmitContext:
    """Build an EmitContext with a career snapshot in components."""
    return EmitContext(
        fighter_url=fighter_url,
        fighter_profile=_profile(fighter_url),
        opponent_url=opponent_url,
        opponent_profile=_profile(opponent_url),
        event_date=date(2025, 1, 1),
        event_url=event_url,
        weight_class="Lightweight",
        fight_url="http://fight/test",
        bout_order=None,
        components={"career": career_snap},
    )


def _accumulate(outcomes: list[FightOutcomeView]) -> CareerSnapshot:
    """Feed a list of outcomes into the accumulator and freeze."""
    acc = CareerAccumulator()
    for o in outcomes:
        acc.update(o)
    return acc.freeze()


# ---------------------------------------------------------------------------
# Zero fights → None for all percentage features
# ---------------------------------------------------------------------------


class TestZeroFights:
    """A fighter with no prior history emits None for percentage-based features."""

    def test_unknown_fighter_returns_none_for_pcts(self) -> None:
        snap = _accumulate([])
        ctx = _build_context(snap)
        emitter = RecordEmitter()
        result = emitter.emit(ctx)

        assert result["win_pct_all"] is None
        assert result["win_pct_last3"] is None
        assert result["win_pct_last5"] is None
        assert result["ufc_win_pct"] is None
        assert result["current_streak"] == 0
        assert result["is_ufc_debut"] == 1.0

    def test_opponent_with_no_fights_debut_fields_are_none(self) -> None:
        snap = _accumulate([])
        ctx = _build_context(snap)
        emitter = RecordEmitter()
        result = emitter.emit(ctx)

        # Debut opponent fields are None when opponent has no UFC history
        assert result["debut_opponent_ufc_experience"] is None
        assert result["debut_opponent_ufc_win_pct"] is None


# ---------------------------------------------------------------------------
# Win percentage calculations
# ---------------------------------------------------------------------------


class TestWinPercentages:
    """Exact win percentage computations."""

    def test_win_pct_all_one_win(self) -> None:
        snap = _accumulate([_outcome(winner_url=FIGHTER_A)])
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        assert result["win_pct_all"] == pytest.approx(1.0)

    def test_win_pct_all_one_loss(self) -> None:
        snap = _accumulate([_outcome(winner_url=FIGHTER_B)])
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        assert result["win_pct_all"] == pytest.approx(0.0)

    def test_win_pct_all_mixed_record(self) -> None:
        outcomes = [
            _outcome(fight_url="http://fight/1", winner_url=FIGHTER_A),
            _outcome(fight_url="http://fight/2", winner_url=FIGHTER_B),
            _outcome(fight_url="http://fight/3", winner_url=FIGHTER_A),
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        # 2 wins out of 3 fights
        assert result["win_pct_all"] == pytest.approx(2.0 / 3.0)

    def test_win_pct_last3_with_enough_fights(self) -> None:
        # 5 fights: 3W 2L — overall 3/5
        outcomes = [
            _outcome(
                fight_url=f"http://fight/{i}",
                event_date=date(2024, 1 + i, 1),
                winner_url=FIGHTER_A if i < 3 else FIGHTER_B,
            )
            for i in range(5)
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        # With >= 3 fights, win_pct_last3 equals overall win pct (state limitation)
        assert result["win_pct_last3"] == pytest.approx(3.0 / 5.0)

    def test_win_pct_last5_with_fewer_than_5_fights(self) -> None:
        outcomes = [
            _outcome(fight_url="http://fight/1", winner_url=FIGHTER_A),
            _outcome(fight_url="http://fight/2", winner_url=FIGHTER_A),
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        # Only 2 fights, fewer than 5 → None
        assert result["win_pct_last5"] is None

    def test_win_pct_last3_with_fewer_than_3_fights(self) -> None:
        outcomes = [
            _outcome(fight_url="http://fight/1", winner_url=FIGHTER_A),
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        assert result["win_pct_last3"] is None

    def test_win_pct_last5_exact_calculation(self) -> None:
        # 7 fights: 6W 1L → overall 6/7
        outcomes = [
            _outcome(
                fight_url=f"http://fight/{i}",
                event_date=date(2024, 1 + i, 1),
                winner_url=FIGHTER_B if i == 4 else FIGHTER_A,
            )
            for i in range(7)
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        # With >= 5 fights, win_pct_last5 equals overall win pct (state limitation)
        assert result["win_pct_last5"] == pytest.approx(6.0 / 7.0)

    def test_draws_and_no_contests_counted_in_total_fights(self) -> None:
        outcomes = [
            _outcome(fight_url="http://fight/1", winner_url=FIGHTER_A),
            _outcome(
                fight_url="http://fight/2", winner_url=None, method="Draw"
            ),
            _outcome(fight_url="http://fight/3", winner_url=FIGHTER_B),
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        # 1 win, 3 total fights → 1/3
        assert result["win_pct_all"] == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# Finish and decision percentages
# ---------------------------------------------------------------------------


class TestFinishDecisionPcts:
    """Finish and decision percentage calculations."""

    def test_win_pct_by_finish_all_kos(self) -> None:
        outcomes = [
            _outcome(fight_url="http://fight/1", winner_url=FIGHTER_A, method="KO/TKO"),
            _outcome(fight_url="http://fight/2", winner_url=FIGHTER_A, method="KO/TKO"),
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        assert result["win_pct_by_finish"] == pytest.approx(1.0)
        assert result["win_pct_by_decision"] == pytest.approx(0.0)

    def test_win_pct_by_decision_all_decisions(self) -> None:
        outcomes = [
            _outcome(
                fight_url="http://fight/1",
                winner_url=FIGHTER_A,
                method="Decision - Unanimous",
            ),
            _outcome(
                fight_url="http://fight/2",
                winner_url=FIGHTER_A,
                method="Decision - Split",
            ),
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        assert result["win_pct_by_finish"] == pytest.approx(0.0)
        assert result["win_pct_by_decision"] == pytest.approx(1.0)

    def test_finish_pct_mixed_wins(self) -> None:
        outcomes = [
            _outcome(fight_url="http://fight/1", winner_url=FIGHTER_A, method="KO/TKO"),
            _outcome(
                fight_url="http://fight/2",
                winner_url=FIGHTER_A,
                method="Submission",
            ),
            _outcome(
                fight_url="http://fight/3",
                winner_url=FIGHTER_A,
                method="Decision - Unanimous",
            ),
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        # 2 finishes (KO + Sub) out of 3 wins
        assert result["win_pct_by_finish"] == pytest.approx(2.0 / 3.0)
        assert result["win_pct_by_decision"] == pytest.approx(1.0 / 3.0)

    def test_loss_pct_by_finish_and_decision(self) -> None:
        outcomes = [
            _outcome(fight_url="http://fight/1", winner_url=FIGHTER_B, method="KO/TKO"),
            _outcome(
                fight_url="http://fight/2",
                winner_url=FIGHTER_B,
                method="Decision - Unanimous",
            ),
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        # Fighter A lost both: 1 by finish, 1 by decision
        assert result["loss_pct_by_finish"] == pytest.approx(0.5)
        assert result["loss_pct_by_decision"] == pytest.approx(0.5)

    def test_zero_wins_finish_pct_is_none(self) -> None:
        outcomes = [
            _outcome(fight_url="http://fight/1", winner_url=FIGHTER_B),
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        assert result["win_pct_by_finish"] is None
        assert result["win_pct_by_decision"] is None

    def test_zero_losses_loss_pct_is_none(self) -> None:
        outcomes = [
            _outcome(fight_url="http://fight/1", winner_url=FIGHTER_A),
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        assert result["loss_pct_by_finish"] is None
        assert result["loss_pct_by_decision"] is None


# ---------------------------------------------------------------------------
# Streak sign convention
# ---------------------------------------------------------------------------


class TestStreak:
    """Streak tracking: positive = win streak, negative = loss streak."""

    def test_win_streak_positive(self) -> None:
        outcomes = [
            _outcome(fight_url="http://fight/1", winner_url=FIGHTER_A),
            _outcome(fight_url="http://fight/2", winner_url=FIGHTER_A),
            _outcome(fight_url="http://fight/3", winner_url=FIGHTER_A),
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        assert result["current_streak"] == 3.0

    def test_loss_streak_negative(self) -> None:
        outcomes = [
            _outcome(fight_url="http://fight/1", winner_url=FIGHTER_B),
            _outcome(fight_url="http://fight/2", winner_url=FIGHTER_B),
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        assert result["current_streak"] == -2.0

    def test_streak_resets_on_opposite_result(self) -> None:
        outcomes = [
            _outcome(fight_url="http://fight/1", winner_url=FIGHTER_A),
            _outcome(fight_url="http://fight/2", winner_url=FIGHTER_A),
            _outcome(fight_url="http://fight/3", winner_url=FIGHTER_B),
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        assert result["current_streak"] == -1.0


# ---------------------------------------------------------------------------
# UFC win percentage and fight count
# ---------------------------------------------------------------------------


class TestUfcWinPct:
    """UFC-specific win percentage uses total UFC fights as denominator."""

    def test_ufc_win_pct_all_wins(self) -> None:
        outcomes = [
            _outcome(fight_url="http://fight/1", winner_url=FIGHTER_A),
            _outcome(fight_url="http://fight/2", winner_url=FIGHTER_A),
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        assert result["ufc_win_pct"] == pytest.approx(1.0)
        assert result["ufc_record_fights_count"] == 2.0

    def test_ufc_win_pct_mixed(self) -> None:
        outcomes = [
            _outcome(fight_url="http://fight/1", winner_url=FIGHTER_A),
            _outcome(fight_url="http://fight/2", winner_url=FIGHTER_B),
            _outcome(fight_url="http://fight/3", winner_url=FIGHTER_A),
            _outcome(fight_url="http://fight/4", winner_url=FIGHTER_B),
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        assert result["ufc_win_pct"] == pytest.approx(0.5)
        assert result["ufc_record_fights_count"] == 4.0


# ---------------------------------------------------------------------------
# Debut flag logic
# ---------------------------------------------------------------------------


class TestDebutFlag:
    """is_ufc_debut and debut opponent fields."""

    def test_debut_fighter_flag_is_true(self) -> None:
        snap = _accumulate([])
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        assert result["is_ufc_debut"] == 1.0

    def test_non_debut_fighter_flag_is_false(self) -> None:
        outcomes = [
            _outcome(fight_url="http://fight/1", winner_url=FIGHTER_A),
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        assert result["is_ufc_debut"] == 0.0

    def test_debut_opponent_experience_from_opponent_state(self) -> None:
        # Fighter A debuting, opponent B has 5 fights
        outcomes = [
            _outcome(
                fight_url=f"http://fight/{i}",
                event_date=date(2024, 1 + i, 1),
                fighter_a_url=FIGHTER_B,
                fighter_b_url=FIGHTER_C,
                winner_url=FIGHTER_B if i < 3 else FIGHTER_C,
            )
            for i in range(5)
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(snap, fighter_url=FIGHTER_A, opponent_url=FIGHTER_B)
        result = RecordEmitter().emit(ctx)

        # A is debuting, so debut_opponent fields are populated from B's state
        assert result["is_ufc_debut"] == 1.0
        assert result["debut_opponent_ufc_experience"] == 5.0
        assert result["debut_opponent_ufc_win_pct"] == pytest.approx(3.0 / 5.0)

    def test_non_debut_fighter_has_none_debut_opponent_fields(self) -> None:
        # Fighter A has 1 fight — not debuting, debut fields should be None
        outcomes = [
            _outcome(fight_url="http://fight/1", winner_url=FIGHTER_A),
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        assert result["is_ufc_debut"] == 0.0
        assert result["debut_opponent_ufc_experience"] is None
        assert result["debut_opponent_ufc_win_pct"] is None


# ---------------------------------------------------------------------------
# Contender series detection
# ---------------------------------------------------------------------------


class TestContenderSeries:
    """DWCS detection from event_url pattern."""

    def test_contender_series_event_detected(self) -> None:
        outcomes = [
            _outcome(fight_url="http://fight/1", winner_url=FIGHTER_A),
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(
            snap,
            event_url="http://ufcstats.com/event-details/dana-white-contender-series-week-5",
        )
        result = RecordEmitter().emit(ctx)

        assert result["contender_series_win"] == 1.0

    def test_non_contender_series_event(self) -> None:
        outcomes = [
            _outcome(fight_url="http://fight/1", winner_url=FIGHTER_A),
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(snap, event_url="http://ufcstats.com/event-details/ufc-300")
        result = RecordEmitter().emit(ctx)

        assert result["contender_series_win"] == 0.0

    def test_contender_series_case_insensitive(self) -> None:
        outcomes = [
            _outcome(fight_url="http://fight/1", winner_url=FIGHTER_A),
        ]
        snap = _accumulate(outcomes)
        ctx = _build_context(
            snap,
            event_url="http://ufcstats.com/event-details/Dana-White-Contender-Series",
        )
        result = RecordEmitter().emit(ctx)

        assert result["contender_series_win"] == 1.0


# ---------------------------------------------------------------------------
# Emitter protocol compliance
# ---------------------------------------------------------------------------


class TestEmitterProtocol:
    """RecordEmitter meets FeatureEmitter protocol requirements."""

    def test_name_attribute_exists(self) -> None:
        emitter = RecordEmitter()
        assert emitter.name == "record"

    def test_emit_returns_dict(self) -> None:
        snap = _accumulate([])
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)
        assert isinstance(result, dict)

    def test_all_expected_keys_present(self) -> None:
        snap = _accumulate([])
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        expected_keys = {
            "win_pct_all",
            "win_pct_last3",
            "win_pct_last5",
            "current_streak",
            "win_pct_by_finish",
            "win_pct_by_decision",
            "loss_pct_by_finish",
            "loss_pct_by_decision",
            "ufc_win_pct",
            "ufc_record_fights_count",
            "is_ufc_debut",
            "debut_opponent_ufc_experience",
            "debut_opponent_ufc_win_pct",
            "contender_series_win",
        }
        assert set(result.keys()) == expected_keys

    def test_values_are_float_or_none(self) -> None:
        snap = _accumulate([
            _outcome(fight_url="http://fight/1", winner_url=FIGHTER_A),
        ])
        ctx = _build_context(snap)
        result = RecordEmitter().emit(ctx)

        for key, val in result.items():
            assert val is None or isinstance(val, float), (
                f"{key} has invalid type {type(val)}"
            )
