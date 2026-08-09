"""Tests for MatchupEmitter.

Validates matchup delta computations (A−B convention), grappling sub-type
scores and mismatches, style interaction features, and orientation symmetry
(signs flip when fighter/opponent are reversed).
"""

from __future__ import annotations

from datetime import date

import pytest

from ufc_edge.features.components.career import CareerFighterState, CareerSnapshot
from ufc_edge.features.components.elo import EloRecord, _FrozenEloSnapshot
from ufc_edge.features.components.pagerank import PageRankFrozenState
from ufc_edge.features.components.rolling_stats import (
    FightStats,
    RollingStatsSnapshot,
)
from ufc_edge.features.contracts import EmitContext, FighterProfile
from ufc_edge.features.emitters.matchup import MatchupEmitter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIGHTER_A = "http://fighter/a"
FIGHTER_B = "http://fighter/b"


def _profile(
    fighter_url: str = FIGHTER_A,
    *,
    height_cm: float | None = 180.0,
    reach_cm: float | None = 185.0,
    stance: str | None = "Orthodox",
    dob: date | None = date(1990, 1, 1),
) -> FighterProfile:
    return FighterProfile(
        fighter_url=fighter_url,
        height_cm=height_cm,
        reach_cm=reach_cm,
        stance=stance,
        dob=dob,
    )


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


def _elo_record(rating: float = 1500.0) -> EloRecord:
    return EloRecord(
        rating=rating,
        peak=rating,
        history=(rating,),
        last_fight_date=date(2024, 1, 1),
        fight_count=5,
    )


def _fight_stats(
    *,
    duration_minutes: float = 10.0,
    sig_strikes_landed: int = 30,
    sig_strikes_attempted: int = 60,
    sig_strikes_absorbed: int = 20,
    sig_strikes_absorbed_attempted: int = 50,
    total_strikes_landed: int = 50,
    total_strikes_attempted: int = 80,
    takedowns_landed: int = 3,
    takedowns_attempted: int = 5,
    submissions_attempted: int = 1,
    knockdowns: int = 0,
    control_time_seconds: int = 120,
    opponent_takedowns_landed: int = 1,
    opponent_takedowns_attempted: int = 3,
    opponent_control_time_seconds: int = 30,
) -> FightStats:
    return FightStats(
        fight_duration_minutes=duration_minutes,
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


def _context(
    *,
    fighter_profile: FighterProfile | None = None,
    opponent_profile: FighterProfile | None = None,
    fighter_career: CareerFighterState | None = None,
    opponent_career: CareerFighterState | None = None,
    fighter_rolling: tuple[FightStats, ...] | None = None,
    opponent_rolling: tuple[FightStats, ...] | None = None,
    fighter_elo: float | None = None,
    opponent_elo: float | None = None,
    pagerank_scores: dict[str, float] | None = None,
    event_date: date = date(2024, 6, 1),
) -> EmitContext:
    """Build a minimal EmitContext with required components for matchup features."""
    f_profile = fighter_profile or _profile(FIGHTER_A)
    o_profile = opponent_profile or _profile(FIGHTER_B)

    career_states: dict[str, CareerFighterState] = {}
    if fighter_career is not None:
        career_states[FIGHTER_A] = fighter_career
    if opponent_career is not None:
        career_states[FIGHTER_B] = opponent_career

    rolling_windows: dict[str, tuple[FightStats, ...]] = {}
    if fighter_rolling is not None:
        rolling_windows[FIGHTER_A] = fighter_rolling
    if opponent_rolling is not None:
        rolling_windows[FIGHTER_B] = opponent_rolling

    elo_records: dict[str, EloRecord] = {}
    if fighter_elo is not None:
        elo_records[FIGHTER_A] = _elo_record(fighter_elo)
    if opponent_elo is not None:
        elo_records[FIGHTER_B] = _elo_record(opponent_elo)

    pr_scores = pagerank_scores if pagerank_scores is not None else {}

    components: dict[str, object] = {
        "career": CareerSnapshot(career_states),
        "rolling_stats": RollingStatsSnapshot(rolling_windows),
        "elo": _FrozenEloSnapshot(elo_records),
        "pagerank": PageRankFrozenState(pr_scores),
    }

    return EmitContext(
        fighter_url=FIGHTER_A,
        fighter_profile=f_profile,
        opponent_url=FIGHTER_B,
        opponent_profile=o_profile,
        event_date=event_date,
        event_url="http://event/1",
        weight_class="Lightweight",
        fight_url="http://fight/1",
        bout_order=None,
        components=components,
    )


def _reversed_context(ctx: EmitContext) -> EmitContext:
    """Build a context with fighter/opponent reversed to test symmetry."""
    return EmitContext(
        fighter_url=ctx.opponent_url,
        fighter_profile=ctx.opponent_profile,
        opponent_url=ctx.fighter_url,
        opponent_profile=ctx.fighter_profile,
        event_date=ctx.event_date,
        event_url=ctx.event_url,
        weight_class=ctx.weight_class,
        fight_url=ctx.fight_url,
        bout_order=ctx.bout_order,
        components=ctx.components,
    )


# ---------------------------------------------------------------------------
# Physical deltas
# ---------------------------------------------------------------------------


class TestPhysicalDeltas:
    """Reach, height, and age deltas follow A−B convention."""

    def test_reach_delta_positive_when_a_longer(self) -> None:
        ctx = _context(
            fighter_profile=_profile(FIGHTER_A, reach_cm=200.0),
            opponent_profile=_profile(FIGHTER_B, reach_cm=180.0),
        )
        result = MatchupEmitter().emit(ctx)
        assert result["reach_delta"] == pytest.approx(20.0)

    def test_height_delta_negative_when_a_shorter(self) -> None:
        ctx = _context(
            fighter_profile=_profile(FIGHTER_A, height_cm=170.0),
            opponent_profile=_profile(FIGHTER_B, height_cm=185.0),
        )
        result = MatchupEmitter().emit(ctx)
        assert result["height_delta"] == pytest.approx(-15.0)

    def test_age_delta_positive_when_a_older(self) -> None:
        ctx = _context(
            fighter_profile=_profile(FIGHTER_A, dob=date(1985, 1, 1)),
            opponent_profile=_profile(FIGHTER_B, dob=date(1990, 1, 1)),
            event_date=date(2024, 6, 1),
        )
        result = MatchupEmitter().emit(ctx)
        # A is 39, B is 34 => delta = 5.0
        assert result["age_delta"] == pytest.approx(5.0)

    def test_physical_deltas_none_when_measurements_missing(self) -> None:
        ctx = _context(
            fighter_profile=_profile(FIGHTER_A, reach_cm=None, height_cm=None, dob=None),
            opponent_profile=_profile(FIGHTER_B, reach_cm=180.0, height_cm=175.0),
        )
        result = MatchupEmitter().emit(ctx)
        assert result["reach_delta"] is None
        assert result["height_delta"] is None
        assert result["age_delta"] is None


# ---------------------------------------------------------------------------
# Stance matchup
# ---------------------------------------------------------------------------


class TestStanceMatchup:
    """Stance categorization follows ortho_v_ortho, ortho_v_south, etc."""

    def test_ortho_v_ortho(self) -> None:
        ctx = _context(
            fighter_profile=_profile(FIGHTER_A, stance="Orthodox"),
            opponent_profile=_profile(FIGHTER_B, stance="Orthodox"),
        )
        result = MatchupEmitter().emit(ctx)
        assert result["stance_matchup"] == "ortho_v_ortho"

    def test_ortho_v_south(self) -> None:
        ctx = _context(
            fighter_profile=_profile(FIGHTER_A, stance="Orthodox"),
            opponent_profile=_profile(FIGHTER_B, stance="Southpaw"),
        )
        result = MatchupEmitter().emit(ctx)
        assert result["stance_matchup"] == "ortho_v_south"

    def test_south_v_south(self) -> None:
        ctx = _context(
            fighter_profile=_profile(FIGHTER_A, stance="Southpaw"),
            opponent_profile=_profile(FIGHTER_B, stance="Southpaw"),
        )
        result = MatchupEmitter().emit(ctx)
        assert result["stance_matchup"] == "south_v_south"

    def test_switch_involved(self) -> None:
        ctx = _context(
            fighter_profile=_profile(FIGHTER_A, stance="Switch"),
            opponent_profile=_profile(FIGHTER_B, stance="Orthodox"),
        )
        result = MatchupEmitter().emit(ctx)
        assert result["stance_matchup"] == "switch_involved"

    def test_southpaw_matchup_flag(self) -> None:
        ctx = _context(
            fighter_profile=_profile(FIGHTER_A, stance="Orthodox"),
            opponent_profile=_profile(FIGHTER_B, stance="Southpaw"),
        )
        result = MatchupEmitter().emit(ctx)
        assert result["southpaw_matchup"] == 1.0

    def test_no_southpaw_matchup(self) -> None:
        ctx = _context(
            fighter_profile=_profile(FIGHTER_A, stance="Orthodox"),
            opponent_profile=_profile(FIGHTER_B, stance="Orthodox"),
        )
        result = MatchupEmitter().emit(ctx)
        assert result["southpaw_matchup"] == 0.0

    def test_stance_none_returns_none(self) -> None:
        ctx = _context(
            fighter_profile=_profile(FIGHTER_A, stance=None),
            opponent_profile=_profile(FIGHTER_B, stance="Orthodox"),
        )
        result = MatchupEmitter().emit(ctx)
        assert result["stance_matchup"] is None
        assert result["southpaw_matchup"] is None


# ---------------------------------------------------------------------------
# Rating deltas
# ---------------------------------------------------------------------------


class TestRatingDeltas:
    """Elo and PageRank deltas follow A−B convention."""

    def test_elo_delta(self) -> None:
        ctx = _context(fighter_elo=1600.0, opponent_elo=1450.0)
        result = MatchupEmitter().emit(ctx)
        assert result["elo_delta"] == pytest.approx(150.0)

    def test_elo_delta_negative_when_lower(self) -> None:
        ctx = _context(fighter_elo=1400.0, opponent_elo=1550.0)
        result = MatchupEmitter().emit(ctx)
        assert result["elo_delta"] == pytest.approx(-150.0)

    def test_elo_delta_none_when_missing(self) -> None:
        ctx = _context(fighter_elo=1500.0, opponent_elo=None)
        result = MatchupEmitter().emit(ctx)
        assert result["elo_delta"] is None

    def test_pagerank_delta(self) -> None:
        ctx = _context(
            pagerank_scores={FIGHTER_A: 0.05, FIGHTER_B: 0.02}
        )
        result = MatchupEmitter().emit(ctx)
        assert result["pagerank_delta"] == pytest.approx(0.03)

    def test_pagerank_delta_none_when_missing(self) -> None:
        ctx = _context(pagerank_scores={FIGHTER_A: 0.05})
        result = MatchupEmitter().emit(ctx)
        assert result["pagerank_delta"] is None


# ---------------------------------------------------------------------------
# Career/output-derived deltas
# ---------------------------------------------------------------------------


class TestDerivedDeltas:
    """Finish rate, efficiency, and experience deltas from career/rolling stats."""

    def test_finish_rate_delta(self) -> None:
        ctx = _context(
            fighter_career=_career_state(wins=10, ko_wins=5, submission_wins=3),
            opponent_career=_career_state(wins=8, ko_wins=2, submission_wins=1),
        )
        result = MatchupEmitter().emit(ctx)
        # A: (5+3)/10 = 0.8, B: (2+1)/8 = 0.375 => delta = 0.425
        assert result["finish_rate_delta"] == pytest.approx(0.425)

    def test_striking_efficiency_delta(self) -> None:
        a_stats = _fight_stats(sig_strikes_landed=50, sig_strikes_attempted=100)
        b_stats = _fight_stats(sig_strikes_landed=30, sig_strikes_attempted=100)
        ctx = _context(
            fighter_rolling=(a_stats,),
            opponent_rolling=(b_stats,),
        )
        result = MatchupEmitter().emit(ctx)
        # A accuracy: 50/100=0.5, B accuracy: 30/100=0.3 => delta = 0.2
        assert result["striking_efficiency_delta"] == pytest.approx(0.2)

    def test_td_accuracy_delta(self) -> None:
        a_stats = _fight_stats(takedowns_landed=4, takedowns_attempted=5)
        b_stats = _fight_stats(takedowns_landed=2, takedowns_attempted=8)
        ctx = _context(
            fighter_rolling=(a_stats,),
            opponent_rolling=(b_stats,),
        )
        result = MatchupEmitter().emit(ctx)
        # A: 4/5=0.8, B: 2/8=0.25 => delta = 0.55
        assert result["td_accuracy_delta"] == pytest.approx(0.55)

    def test_damage_ratio_delta(self) -> None:
        a_stats = _fight_stats(sig_strikes_landed=40, sig_strikes_absorbed=20)
        b_stats = _fight_stats(sig_strikes_landed=25, sig_strikes_absorbed=30)
        ctx = _context(
            fighter_rolling=(a_stats,),
            opponent_rolling=(b_stats,),
        )
        result = MatchupEmitter().emit(ctx)
        # A: 40/20=2.0, B: 25/30=0.833 => delta ≈ 1.167
        assert result["damage_ratio_delta"] == pytest.approx(2.0 - 25 / 30)

    def test_avg_fight_duration_delta(self) -> None:
        a_stats = _fight_stats(duration_minutes=15.0)
        b_stats = _fight_stats(duration_minutes=8.0)
        ctx = _context(
            fighter_rolling=(a_stats,),
            opponent_rolling=(b_stats,),
        )
        result = MatchupEmitter().emit(ctx)
        # Duration is emitted in seconds: A=900, B=480 => delta=420
        assert result["avg_fight_duration_delta"] == pytest.approx(420.0)

    def test_fight_duration_variance_delta_requires_3_fights(self) -> None:
        a_stats = (_fight_stats(duration_minutes=10.0), _fight_stats(duration_minutes=5.0))
        b_stats = (_fight_stats(duration_minutes=12.0), _fight_stats(duration_minutes=8.0))
        ctx = _context(fighter_rolling=a_stats, opponent_rolling=b_stats)
        result = MatchupEmitter().emit(ctx)
        # Less than 3 fights => None for variance
        assert result["fight_duration_variance_delta"] is None

    def test_fight_duration_variance_delta_with_sufficient_history(self) -> None:
        a_stats = (
            _fight_stats(duration_minutes=10.0),
            _fight_stats(duration_minutes=10.0),
            _fight_stats(duration_minutes=10.0),
        )
        b_stats = (
            _fight_stats(duration_minutes=5.0),
            _fight_stats(duration_minutes=15.0),
            _fight_stats(duration_minutes=10.0),
        )
        ctx = _context(fighter_rolling=a_stats, opponent_rolling=b_stats)
        result = MatchupEmitter().emit(ctx)
        # A variance: 0 (all same). B variance: var([5,15,10]) in seconds^2
        # B mean = 10 min, var = ((5-10)^2+(15-10)^2+(10-10)^2)/3 = 50/3 min^2
        # In seconds^2: 0 - 50/3*3600 = negative
        a_var_sec = 0.0
        b_mean = 10.0
        b_var_min = ((5.0 - b_mean) ** 2 + (15.0 - b_mean) ** 2 + (10.0 - b_mean) ** 2) / 3
        b_var_sec = b_var_min * 3600.0
        expected = a_var_sec - b_var_sec
        assert result["fight_duration_variance_delta"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Experience deltas
# ---------------------------------------------------------------------------


class TestExperienceDeltas:
    """UFC experience, five-round experience, and title fight experience deltas."""

    def test_ufc_experience_delta(self) -> None:
        ctx = _context(
            fighter_career=_career_state(total_fights=15),
            opponent_career=_career_state(total_fights=8),
        )
        result = MatchupEmitter().emit(ctx)
        assert result["ufc_experience_delta"] == pytest.approx(7.0)

    def test_five_round_experience_delta_none_when_no_career(self) -> None:
        ctx = _context()
        result = MatchupEmitter().emit(ctx)
        assert result["five_round_experience_delta"] is None

    def test_title_fight_exp_delta_none_when_no_career(self) -> None:
        ctx = _context()
        result = MatchupEmitter().emit(ctx)
        assert result["title_fight_exp_delta"] is None


# ---------------------------------------------------------------------------
# Grappling sub-type matchup
# ---------------------------------------------------------------------------


class TestGrapplingSubType:
    """Wrestler/submission scores, deltas, and type mismatch detection."""

    def test_wrestler_score_computation(self) -> None:
        # wrestler_score = td_accuracy * td_per_15 * td_defense
        stats = _fight_stats(
            duration_minutes=10.0,
            takedowns_landed=4,
            takedowns_attempted=5,
            opponent_takedowns_landed=1,
            opponent_takedowns_attempted=4,
        )
        ctx = _context(fighter_rolling=(stats,))
        result = MatchupEmitter().emit(ctx)
        # td_accuracy = 4/5 = 0.8
        # td_per_15 = 4*15/10 = 6.0
        # td_defense = (4-1)/4 = 0.75
        expected = 0.8 * 6.0 * 0.75
        assert result["wrestler_score_a"] == pytest.approx(expected)

    def test_submission_score_computation(self) -> None:
        # submission_score = sub_attempts_per_15 * submission_rate
        stats = _fight_stats(duration_minutes=10.0, submissions_attempted=3)
        career = _career_state(wins=10, submission_wins=6)
        ctx = _context(fighter_rolling=(stats,), fighter_career=career)
        result = MatchupEmitter().emit(ctx)
        # sub_per_15 = 3*15/10 = 4.5
        # sub_rate = 6/10 = 0.6
        expected = 4.5 * 0.6
        assert result["submission_score_a"] == pytest.approx(expected)

    def test_wrestling_delta(self) -> None:
        a_stats = _fight_stats(
            duration_minutes=10.0,
            takedowns_landed=5,
            takedowns_attempted=6,
            opponent_takedowns_landed=0,
            opponent_takedowns_attempted=3,
        )
        b_stats = _fight_stats(
            duration_minutes=10.0,
            takedowns_landed=1,
            takedowns_attempted=5,
            opponent_takedowns_landed=3,
            opponent_takedowns_attempted=5,
        )
        ctx = _context(fighter_rolling=(a_stats,), opponent_rolling=(b_stats,))
        result = MatchupEmitter().emit(ctx)
        # A: td_acc=5/6, td_per_15=5*15/10=7.5, td_def=3/3=1.0
        # => 5/6 * 7.5 * 1.0 = 6.25
        # B: td_acc=1/5=0.2, td_per_15=1*15/10=1.5, td_def=(5-3)/5=0.4
        # => 0.2 * 1.5 * 0.4 = 0.12
        a_score = (5 / 6) * 7.5 * 1.0
        b_score = 0.2 * 1.5 * 0.4
        assert result["wrestling_delta"] == pytest.approx(a_score - b_score)

    def test_grappling_type_mismatch_detected(self) -> None:
        # High wrestler score on A, high submission score on B
        a_stats = _fight_stats(
            duration_minutes=10.0,
            takedowns_landed=5,
            takedowns_attempted=6,
            opponent_takedowns_landed=0,
            opponent_takedowns_attempted=4,
            submissions_attempted=0,
        )
        b_stats = _fight_stats(
            duration_minutes=10.0,
            takedowns_landed=0,
            takedowns_attempted=1,
            opponent_takedowns_landed=2,
            opponent_takedowns_attempted=3,
            submissions_attempted=5,
        )
        a_career = _career_state(wins=8, ko_wins=6, submission_wins=0)
        b_career = _career_state(wins=8, submission_wins=6, ko_wins=0)
        ctx = _context(
            fighter_rolling=(a_stats,),
            opponent_rolling=(b_stats,),
            fighter_career=a_career,
            opponent_career=b_career,
        )
        result = MatchupEmitter().emit(ctx)
        # Should detect mismatch: one side wrestler-heavy, other sub-heavy
        assert result["grappling_type_mismatch"] == 1.0

    def test_grappling_type_mismatch_not_detected_when_similar(self) -> None:
        stats = _fight_stats(
            duration_minutes=10.0,
            takedowns_landed=3,
            takedowns_attempted=5,
            opponent_takedowns_landed=2,
            opponent_takedowns_attempted=4,
            submissions_attempted=2,
        )
        career = _career_state(wins=10, submission_wins=3, ko_wins=3, decision_wins=4)
        ctx = _context(
            fighter_rolling=(stats,),
            opponent_rolling=(stats,),
            fighter_career=career,
            opponent_career=career,
        )
        result = MatchupEmitter().emit(ctx)
        assert result["grappling_type_mismatch"] == 0.0


# ---------------------------------------------------------------------------
# Style interactions
# ---------------------------------------------------------------------------


class TestStyleInteractions:
    """Striker vs grappler, pressure vs counter, pace mismatch, and southpaw history."""

    def test_striker_vs_grappler_detected(self) -> None:
        # A is a striker (high striking, low TD), B is a grappler (high TD, low striking)
        a_stats = _fight_stats(
            duration_minutes=10.0,
            sig_strikes_landed=60,
            sig_strikes_attempted=100,
            takedowns_landed=0,
            takedowns_attempted=1,
            submissions_attempted=0,
        )
        b_stats = _fight_stats(
            duration_minutes=10.0,
            sig_strikes_landed=15,
            sig_strikes_attempted=40,
            takedowns_landed=5,
            takedowns_attempted=7,
            submissions_attempted=3,
        )
        ctx = _context(fighter_rolling=(a_stats,), opponent_rolling=(b_stats,))
        result = MatchupEmitter().emit(ctx)
        assert result["striker_vs_grappler"] == 1.0

    def test_striker_vs_grappler_not_detected_both_strikers(self) -> None:
        stats = _fight_stats(
            duration_minutes=10.0,
            sig_strikes_landed=50,
            sig_strikes_attempted=90,
            takedowns_landed=0,
            takedowns_attempted=1,
            submissions_attempted=0,
        )
        ctx = _context(fighter_rolling=(stats,), opponent_rolling=(stats,))
        result = MatchupEmitter().emit(ctx)
        assert result["striker_vs_grappler"] == 0.0

    def test_pace_mismatch_score(self) -> None:
        # pace_mismatch = sig_strikes_per_min_delta * fight_duration_variance_delta
        a_stats = (
            _fight_stats(duration_minutes=10.0, sig_strikes_landed=60),
            _fight_stats(duration_minutes=10.0, sig_strikes_landed=60),
            _fight_stats(duration_minutes=10.0, sig_strikes_landed=60),
        )
        b_stats = (
            _fight_stats(duration_minutes=10.0, sig_strikes_landed=30),
            _fight_stats(duration_minutes=10.0, sig_strikes_landed=30),
            _fight_stats(duration_minutes=10.0, sig_strikes_landed=30),
        )
        ctx = _context(fighter_rolling=a_stats, opponent_rolling=b_stats)
        result = MatchupEmitter().emit(ctx)
        # Both have zero variance, so pace_mismatch = strikes_delta * 0 = 0
        assert result["pace_mismatch_score"] == pytest.approx(0.0)

    def test_southpaw_orthodox_history_none_without_data(self) -> None:
        ctx = _context()
        result = MatchupEmitter().emit(ctx)
        # Without southpaw fight history, returns None
        assert result["southpaw_orthodox_history"] is None

    def test_pressure_vs_counter_detected(self) -> None:
        # Pressure fighter: high output (sig_strikes_per_min >= 5.0)
        # Counter fighter: high damage ratio (>= 1.5) with low volume (< 4.0)
        a_stats = _fight_stats(
            duration_minutes=10.0,
            sig_strikes_landed=60,
            sig_strikes_attempted=100,
            sig_strikes_absorbed=50,
            sig_strikes_absorbed_attempted=80,
        )
        b_stats = _fight_stats(
            duration_minutes=10.0,
            sig_strikes_landed=35,
            sig_strikes_attempted=50,
            sig_strikes_absorbed=15,
            sig_strikes_absorbed_attempted=60,
        )
        ctx = _context(fighter_rolling=(a_stats,), opponent_rolling=(b_stats,))
        result = MatchupEmitter().emit(ctx)
        # A: sig_per_min = 60/10 = 6.0 >= 5.0 => pressure
        # B: sig_per_min = 35/10 = 3.5 < 4.0, damage_ratio = 35/15 = 2.33 >= 1.5 => counter
        assert result["pressure_vs_counter"] == 1.0


# ---------------------------------------------------------------------------
# Orientation symmetry — signs flip when fighters are reversed
# ---------------------------------------------------------------------------


class TestOrientationSymmetry:
    """Delta signs flip exactly when fighter and opponent are reversed."""

    def test_physical_deltas_flip(self) -> None:
        ctx = _context(
            fighter_profile=_profile(
                FIGHTER_A, height_cm=190.0, reach_cm=200.0, dob=date(1988, 1, 1)
            ),
            opponent_profile=_profile(
                FIGHTER_B, height_cm=175.0, reach_cm=180.0, dob=date(1992, 1, 1)
            ),
            event_date=date(2024, 6, 1),
        )
        fwd = MatchupEmitter().emit(ctx)
        rev = MatchupEmitter().emit(_reversed_context(ctx))

        assert fwd["reach_delta"] == pytest.approx(-rev["reach_delta"])
        assert fwd["height_delta"] == pytest.approx(-rev["height_delta"])
        assert fwd["age_delta"] == pytest.approx(-rev["age_delta"])

    def test_elo_delta_flips(self) -> None:
        ctx = _context(fighter_elo=1650.0, opponent_elo=1450.0)
        fwd = MatchupEmitter().emit(ctx)
        rev = MatchupEmitter().emit(_reversed_context(ctx))
        assert fwd["elo_delta"] == pytest.approx(-rev["elo_delta"])

    def test_pagerank_delta_flips(self) -> None:
        ctx = _context(pagerank_scores={FIGHTER_A: 0.08, FIGHTER_B: 0.03})
        fwd = MatchupEmitter().emit(ctx)
        rev = MatchupEmitter().emit(_reversed_context(ctx))
        assert fwd["pagerank_delta"] == pytest.approx(-rev["pagerank_delta"])

    def test_finish_rate_delta_flips(self) -> None:
        ctx = _context(
            fighter_career=_career_state(wins=10, ko_wins=7, submission_wins=2),
            opponent_career=_career_state(wins=6, ko_wins=1, submission_wins=1),
        )
        fwd = MatchupEmitter().emit(ctx)
        rev = MatchupEmitter().emit(_reversed_context(ctx))
        assert fwd["finish_rate_delta"] == pytest.approx(-rev["finish_rate_delta"])

    def test_wrestling_delta_flips(self) -> None:
        a_stats = _fight_stats(
            takedowns_landed=5, takedowns_attempted=6,
            opponent_takedowns_landed=1, opponent_takedowns_attempted=3,
        )
        b_stats = _fight_stats(
            takedowns_landed=1, takedowns_attempted=4,
            opponent_takedowns_landed=2, opponent_takedowns_attempted=5,
        )
        ctx = _context(fighter_rolling=(a_stats,), opponent_rolling=(b_stats,))
        fwd = MatchupEmitter().emit(ctx)
        rev = MatchupEmitter().emit(_reversed_context(ctx))
        assert fwd["wrestling_delta"] == pytest.approx(-rev["wrestling_delta"])

    def test_ufc_experience_delta_flips(self) -> None:
        ctx = _context(
            fighter_career=_career_state(total_fights=20),
            opponent_career=_career_state(total_fights=7),
        )
        fwd = MatchupEmitter().emit(ctx)
        rev = MatchupEmitter().emit(_reversed_context(ctx))
        assert fwd["ufc_experience_delta"] == pytest.approx(-rev["ufc_experience_delta"])


# ---------------------------------------------------------------------------
# Edge cases and missing data
# ---------------------------------------------------------------------------


class TestMissingData:
    """Graceful degradation when components or fighter data are missing."""

    def test_all_none_when_no_components(self) -> None:
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
        result = MatchupEmitter().emit(ctx)
        # All numeric deltas should be None when data is missing
        assert result["elo_delta"] is None
        assert result["pagerank_delta"] is None
        assert result["finish_rate_delta"] is None

    def test_one_sided_career_still_returns_deltas(self) -> None:
        # Only fighter A has career data; deltas with missing opponent are None
        ctx = _context(
            fighter_career=_career_state(wins=10, ko_wins=5, submission_wins=3),
        )
        result = MatchupEmitter().emit(ctx)
        assert result["finish_rate_delta"] is None

    def test_emitter_name(self) -> None:
        assert MatchupEmitter().name == "matchup"
