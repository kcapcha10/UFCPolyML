"""Tests for FinishingEmitter.

Validates finish rates, defensive finishing stats, fight duration averages and
variance, and the opponent-dependent interaction term.
"""

from __future__ import annotations

from datetime import date

import pytest

from ufc_edge.features.components.career import CareerFighterState, CareerSnapshot
from ufc_edge.features.components.rolling_stats import (
    FightStats,
    RollingStatsSnapshot,
)
from ufc_edge.features.contracts import EmitContext, FighterProfile
from ufc_edge.features.emitters.finishing import FinishingEmitter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIGHTER_A = "http://fighter/a"
FIGHTER_B = "http://fighter/b"


def _profile(fighter_url: str = FIGHTER_A) -> FighterProfile:
    return FighterProfile(fighter_url=fighter_url)


def _career_state(
    *,
    wins: int = 0,
    losses: int = 0,
    draws: int = 0,
    no_contests: int = 0,
    ko_wins: int = 0,
    submission_wins: int = 0,
    decision_wins: int = 0,
    times_finished_by_ko: int = 0,
    times_finished_by_sub: int = 0,
    round_one_finishes: int = 0,
    been_finished_round_one: bool = False,
    current_streak: int = 0,
    total_fights: int | None = None,
) -> CareerFighterState:
    fights = total_fights if total_fights is not None else wins + losses + draws + no_contests
    return CareerFighterState(
        wins=wins,
        losses=losses,
        draws=draws,
        no_contests=no_contests,
        ko_wins=ko_wins,
        submission_wins=submission_wins,
        decision_wins=decision_wins,
        times_finished_by_ko=times_finished_by_ko,
        times_finished_by_sub=times_finished_by_sub,
        round_one_finishes=round_one_finishes,
        been_finished_round_one=been_finished_round_one,
        current_streak=current_streak,
        fight_dates=(),
        last_fight_date=None,
        last_fight_method=None,
        debut_date=None,
        total_fights=fights,
        weight_classes=(),
    )


def _fight_stats(duration_minutes: float = 10.0) -> FightStats:
    """Create a minimal FightStats with only fight duration relevant."""
    return FightStats(
        fight_duration_minutes=duration_minutes,
        sig_strikes_landed=0,
        sig_strikes_attempted=0,
        sig_strikes_absorbed=0,
        sig_strikes_absorbed_attempted=0,
        total_strikes_landed=0,
        total_strikes_attempted=0,
        takedowns_landed=0,
        takedowns_attempted=0,
        submissions_attempted=0,
        knockdowns=0,
        control_time_seconds=0,
        opponent_takedowns_landed=0,
        opponent_takedowns_attempted=0,
        opponent_control_time_seconds=0,
    )


def _context(
    *,
    fighter_career: CareerFighterState | None = None,
    opponent_career: CareerFighterState | None = None,
    fighter_durations: tuple[FightStats, ...] | None = None,
    opponent_durations: tuple[FightStats, ...] | None = None,
) -> EmitContext:
    """Build a minimal EmitContext with career and optional rolling stats."""
    career_states: dict[str, CareerFighterState] = {}
    if fighter_career is not None:
        career_states[FIGHTER_A] = fighter_career
    if opponent_career is not None:
        career_states[FIGHTER_B] = opponent_career

    rolling_windows: dict[str, tuple[FightStats, ...]] = {}
    if fighter_durations is not None:
        rolling_windows[FIGHTER_A] = fighter_durations
    if opponent_durations is not None:
        rolling_windows[FIGHTER_B] = opponent_durations

    components: dict[str, object] = {
        "career": CareerSnapshot(career_states),
        "rolling_stats": RollingStatsSnapshot(rolling_windows),
    }

    return EmitContext(
        fighter_url=FIGHTER_A,
        fighter_profile=_profile(FIGHTER_A),
        opponent_url=FIGHTER_B,
        opponent_profile=_profile(FIGHTER_B),
        event_date=date(2024, 6, 1),
        event_url="http://event/1",
        weight_class="Lightweight",
        fight_url="http://fight/1",
        bout_order=None,
        components=components,
    )


# ---------------------------------------------------------------------------
# Finish rates with zero wins
# ---------------------------------------------------------------------------


class TestZeroWins:
    """When a fighter has zero wins, all rate features are 0.0."""

    def test_finish_rate_zero_with_no_wins(self) -> None:
        ctx = _context(fighter_career=_career_state(wins=0, losses=3))
        result = FinishingEmitter().emit(ctx)
        assert result["finish_rate"] == 0.0

    def test_ko_rate_zero_with_no_wins(self) -> None:
        ctx = _context(fighter_career=_career_state(wins=0, losses=2))
        result = FinishingEmitter().emit(ctx)
        assert result["ko_rate"] == 0.0

    def test_submission_rate_zero_with_no_wins(self) -> None:
        ctx = _context(fighter_career=_career_state(wins=0, losses=1))
        result = FinishingEmitter().emit(ctx)
        assert result["submission_rate"] == 0.0

    def test_early_finish_rate_zero_with_no_wins(self) -> None:
        ctx = _context(fighter_career=_career_state(wins=0, losses=4))
        result = FinishingEmitter().emit(ctx)
        assert result["early_finish_rate"] == 0.0


# ---------------------------------------------------------------------------
# Finish rates with wins
# ---------------------------------------------------------------------------


class TestFinishRates:
    """Correct computation of offensive finishing rate features."""

    def test_finish_rate_all_ko_wins(self) -> None:
        ctx = _context(
            fighter_career=_career_state(wins=4, ko_wins=4, losses=1)
        )
        result = FinishingEmitter().emit(ctx)
        assert result["finish_rate"] == 1.0

    def test_finish_rate_mixed_wins(self) -> None:
        ctx = _context(
            fighter_career=_career_state(
                wins=10, ko_wins=3, submission_wins=2, decision_wins=5, losses=2
            )
        )
        result = FinishingEmitter().emit(ctx)
        assert result["finish_rate"] == pytest.approx(0.5)

    def test_ko_rate(self) -> None:
        ctx = _context(
            fighter_career=_career_state(wins=8, ko_wins=4, losses=2)
        )
        result = FinishingEmitter().emit(ctx)
        assert result["ko_rate"] == pytest.approx(0.5)

    def test_submission_rate(self) -> None:
        ctx = _context(
            fighter_career=_career_state(wins=5, submission_wins=2, losses=1)
        )
        result = FinishingEmitter().emit(ctx)
        assert result["submission_rate"] == pytest.approx(0.4)

    def test_early_finish_rate(self) -> None:
        ctx = _context(
            fighter_career=_career_state(wins=6, round_one_finishes=3, losses=2)
        )
        result = FinishingEmitter().emit(ctx)
        assert result["early_finish_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Defensive finishing stats
# ---------------------------------------------------------------------------


class TestDefensiveFinishing:
    """Tests for has_ever_been_finished, times_finished, and related flags."""

    def test_has_ever_been_finished_true(self) -> None:
        ctx = _context(
            fighter_career=_career_state(
                wins=5, losses=3, times_finished_by_ko=1
            )
        )
        result = FinishingEmitter().emit(ctx)
        assert result["has_ever_been_finished"] == 1.0

    def test_has_ever_been_finished_false(self) -> None:
        ctx = _context(
            fighter_career=_career_state(wins=5, losses=2, decision_wins=5)
        )
        result = FinishingEmitter().emit(ctx)
        assert result["has_ever_been_finished"] == 0.0

    def test_times_finished_by_ko(self) -> None:
        ctx = _context(
            fighter_career=_career_state(
                wins=3, losses=4, times_finished_by_ko=2
            )
        )
        result = FinishingEmitter().emit(ctx)
        assert result["times_finished_by_ko"] == 2.0

    def test_times_finished_by_sub(self) -> None:
        ctx = _context(
            fighter_career=_career_state(
                wins=3, losses=4, times_finished_by_sub=3
            )
        )
        result = FinishingEmitter().emit(ctx)
        assert result["times_finished_by_sub"] == 3.0

    def test_has_been_finished_r1_true(self) -> None:
        ctx = _context(
            fighter_career=_career_state(
                wins=5, losses=2, been_finished_round_one=True
            )
        )
        result = FinishingEmitter().emit(ctx)
        assert result["has_been_finished_r1"] == 1.0

    def test_has_been_finished_r1_false(self) -> None:
        ctx = _context(
            fighter_career=_career_state(wins=5, losses=2)
        )
        result = FinishingEmitter().emit(ctx)
        assert result["has_been_finished_r1"] == 0.0


# ---------------------------------------------------------------------------
# Never-been-finished and interaction term
# ---------------------------------------------------------------------------


class TestNeverBeenFinished:
    """Tests for the never_been_finished flag and interaction with opponent."""

    def test_never_been_finished_true(self) -> None:
        ctx = _context(
            fighter_career=_career_state(wins=10, losses=2)
        )
        result = FinishingEmitter().emit(ctx)
        assert result["never_been_finished"] == 1.0

    def test_never_been_finished_false_with_ko(self) -> None:
        ctx = _context(
            fighter_career=_career_state(
                wins=5, losses=2, times_finished_by_ko=1
            )
        )
        result = FinishingEmitter().emit(ctx)
        assert result["never_been_finished"] == 0.0

    def test_never_been_finished_false_with_sub(self) -> None:
        ctx = _context(
            fighter_career=_career_state(
                wins=5, losses=3, times_finished_by_sub=2
            )
        )
        result = FinishingEmitter().emit(ctx)
        assert result["never_been_finished"] == 0.0

    def test_interaction_term_when_never_finished(self) -> None:
        """Interaction = 1.0 * opponent_finish_rate."""
        ctx = _context(
            fighter_career=_career_state(wins=8, losses=1),
            opponent_career=_career_state(wins=10, ko_wins=6, submission_wins=2),
        )
        result = FinishingEmitter().emit(ctx)
        # Opponent finish_rate = (6+2)/10 = 0.8
        assert result["never_been_finished_x_opp_finish_rate"] == pytest.approx(0.8)

    def test_interaction_term_when_has_been_finished(self) -> None:
        """Interaction = 0.0 when fighter has been finished."""
        ctx = _context(
            fighter_career=_career_state(
                wins=5, losses=3, times_finished_by_ko=1
            ),
            opponent_career=_career_state(wins=10, ko_wins=8),
        )
        result = FinishingEmitter().emit(ctx)
        assert result["never_been_finished_x_opp_finish_rate"] == 0.0

    def test_interaction_term_opponent_zero_wins(self) -> None:
        """Opponent with zero wins yields 0.0 interaction."""
        ctx = _context(
            fighter_career=_career_state(wins=8, losses=0),
            opponent_career=_career_state(wins=0, losses=3),
        )
        result = FinishingEmitter().emit(ctx)
        assert result["never_been_finished_x_opp_finish_rate"] == 0.0

    def test_interaction_term_opponent_not_tracked(self) -> None:
        """Opponent with no career state yields 0.0 interaction."""
        ctx = _context(
            fighter_career=_career_state(wins=8, losses=0),
            opponent_career=None,
        )
        result = FinishingEmitter().emit(ctx)
        assert result["never_been_finished_x_opp_finish_rate"] == 0.0


# ---------------------------------------------------------------------------
# Fight duration features
# ---------------------------------------------------------------------------


class TestFightDuration:
    """Tests for avg and variance of fight duration."""

    def test_avg_duration_with_fights(self) -> None:
        durations = (
            _fight_stats(10.0),
            _fight_stats(15.0),
            _fight_stats(5.0),
        )
        ctx = _context(
            fighter_career=_career_state(wins=3),
            fighter_durations=durations,
        )
        result = FinishingEmitter().emit(ctx)
        assert result["avg_fight_duration_sec"] == pytest.approx(10.0 * 60.0)

    def test_avg_duration_no_rolling_data(self) -> None:
        ctx = _context(
            fighter_career=_career_state(wins=3),
            fighter_durations=None,
        )
        result = FinishingEmitter().emit(ctx)
        assert result["avg_fight_duration_sec"] is None

    def test_variance_with_three_or_more_fights(self) -> None:
        durations = (
            _fight_stats(10.0),
            _fight_stats(20.0),
            _fight_stats(15.0),
        )
        ctx = _context(
            fighter_career=_career_state(wins=3),
            fighter_durations=durations,
        )
        result = FinishingEmitter().emit(ctx)
        # Mean = 15min, variance = ((10-15)^2 + (20-15)^2 + (15-15)^2) / 3 = 50/3
        # In seconds: variance_min * 3600 = (50/3) * 3600
        mean_min = 15.0
        var_min = ((10.0 - mean_min) ** 2 + (20.0 - mean_min) ** 2 + (15.0 - mean_min) ** 2) / 3
        expected_sec = var_min * 3600.0
        assert result["fight_duration_variance"] == pytest.approx(expected_sec)

    def test_variance_with_fewer_than_three_fights_is_none(self) -> None:
        durations = (_fight_stats(10.0), _fight_stats(20.0))
        ctx = _context(
            fighter_career=_career_state(wins=2),
            fighter_durations=durations,
        )
        result = FinishingEmitter().emit(ctx)
        assert result["fight_duration_variance"] is None

    def test_variance_single_fight_is_none(self) -> None:
        durations = (_fight_stats(15.0),)
        ctx = _context(
            fighter_career=_career_state(wins=1),
            fighter_durations=durations,
        )
        result = FinishingEmitter().emit(ctx)
        assert result["fight_duration_variance"] is None


# ---------------------------------------------------------------------------
# No career state (debut fighter)
# ---------------------------------------------------------------------------


class TestNoCareerState:
    """Fighter with no prior fights returns zero rates and None for duration."""

    def test_debut_fighter_all_rates_zero(self) -> None:
        ctx = _context(fighter_career=None)
        result = FinishingEmitter().emit(ctx)
        assert result["finish_rate"] == 0.0
        assert result["ko_rate"] == 0.0
        assert result["submission_rate"] == 0.0
        assert result["early_finish_rate"] == 0.0

    def test_debut_fighter_defensive_stats_zero(self) -> None:
        ctx = _context(fighter_career=None)
        result = FinishingEmitter().emit(ctx)
        assert result["has_ever_been_finished"] == 0.0
        assert result["times_finished_by_ko"] == 0.0
        assert result["times_finished_by_sub"] == 0.0
        assert result["has_been_finished_r1"] == 0.0
        assert result["never_been_finished"] == 1.0

    def test_debut_fighter_duration_none(self) -> None:
        ctx = _context(fighter_career=None)
        result = FinishingEmitter().emit(ctx)
        assert result["avg_fight_duration_sec"] is None
        assert result["fight_duration_variance"] is None
