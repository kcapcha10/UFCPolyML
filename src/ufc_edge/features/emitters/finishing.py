"""Finishing profile emitter — offensive and defensive finishing features.

Derives finish rates (KO, submission, early) from career win breakdown,
defensive stats (times finished, round-1 vulnerability), fight duration
statistics, and an interaction term capturing whether the fighter has
never been stopped against an opponent who finishes at a high rate.
"""

from __future__ import annotations

from ufc_edge.features.components.career import CareerFighterState, CareerSnapshot
from ufc_edge.features.components.rolling_stats import RollingStatsSnapshot
from ufc_edge.features.contracts import EmitContext

# Minimum fight count to compute meaningful variance
_MIN_FIGHTS_FOR_VARIANCE = 3


class FinishingEmitter:
    """Stateless emitter for finishing profile features.

    Reads CareerSnapshot for finish counts and RollingStatsSnapshot for
    fight duration statistics. Computes both offensive rates (finish_rate,
    ko_rate, etc.) and defensive vulnerability (has_ever_been_finished,
    never_been_finished interaction).
    """

    name: str = "finishing"

    def emit(self, context: EmitContext) -> dict[str, float | str | None]:
        """Emit finishing profile features for the focal fighter."""
        career_snap = context.components.get("career")
        fighter_state = (
            career_snap.get(context.fighter_url)
            if isinstance(career_snap, CareerSnapshot)
            else None
        )

        rates = _compute_finish_rates(fighter_state)
        defensive = _compute_defensive_stats(fighter_state)
        duration = _compute_duration_stats(context)
        interaction = _compute_interaction_term(context, fighter_state)

        return {**rates, **defensive, **duration, **interaction}


# ---------------------------------------------------------------------------
# Offensive finish rates
# ---------------------------------------------------------------------------


def _compute_finish_rates(
    state: CareerFighterState | None,
) -> dict[str, float | None]:
    """Compute offensive finishing rate features.

    All rates are finishes/wins. Zero wins yields 0.0 (not None) since the
    fighter simply has no offensive finishing history yet.
    """
    if state is None or state.wins == 0:
        return {
            "finish_rate": 0.0,
            "ko_rate": 0.0,
            "submission_rate": 0.0,
            "early_finish_rate": 0.0,
        }

    wins = state.wins
    total_finishes = state.ko_wins + state.submission_wins
    return {
        "finish_rate": total_finishes / wins,
        "ko_rate": state.ko_wins / wins,
        "submission_rate": state.submission_wins / wins,
        "early_finish_rate": state.round_one_finishes / wins,
    }


# ---------------------------------------------------------------------------
# Defensive finishing stats
# ---------------------------------------------------------------------------


def _compute_defensive_stats(
    state: CareerFighterState | None,
) -> dict[str, float | None]:
    """Compute defensive finishing vulnerability features.

    Boolean features encoded as 1.0/0.0 for XGBoost consumption.
    A debut fighter (no state) has never been finished.
    """
    if state is None:
        return {
            "has_ever_been_finished": 0.0,
            "times_finished_by_ko": 0.0,
            "times_finished_by_sub": 0.0,
            "has_been_finished_r1": 0.0,
            "never_been_finished": 1.0,
        }

    total_times_finished = state.times_finished_by_ko + state.times_finished_by_sub
    return {
        "has_ever_been_finished": 1.0 if total_times_finished > 0 else 0.0,
        "times_finished_by_ko": float(state.times_finished_by_ko),
        "times_finished_by_sub": float(state.times_finished_by_sub),
        "has_been_finished_r1": 1.0 if state.been_finished_round_one else 0.0,
        "never_been_finished": 1.0 if total_times_finished == 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Fight duration statistics
# ---------------------------------------------------------------------------


def _compute_duration_stats(context: EmitContext) -> dict[str, float | None]:
    """Compute avg and variance of fight duration from rolling stats window.

    Duration is stored in the RollingStatsSnapshot per-fight window. Returns
    None when no rolling data exists. Variance requires at least 3 fights.
    Duration is converted to seconds for the feature output.
    """
    rolling_snap = context.components.get("rolling_stats")
    if not isinstance(rolling_snap, RollingStatsSnapshot):
        return {
            "avg_fight_duration_sec": None,
            "fight_duration_variance": None,
        }

    window = rolling_snap._fighter_windows.get(context.fighter_url)
    if not window:
        return {
            "avg_fight_duration_sec": None,
            "fight_duration_variance": None,
        }

    durations_min = [stats.fight_duration_minutes for stats in window]
    n = len(durations_min)

    mean_min = sum(durations_min) / n
    avg_sec = mean_min * 60.0

    if n < _MIN_FIGHTS_FOR_VARIANCE:
        return {
            "avg_fight_duration_sec": avg_sec,
            "fight_duration_variance": None,
        }

    variance_min = sum((d - mean_min) ** 2 for d in durations_min) / n
    variance_sec = variance_min * 3600.0

    return {
        "avg_fight_duration_sec": avg_sec,
        "fight_duration_variance": variance_sec,
    }


# ---------------------------------------------------------------------------
# Interaction term: never_been_finished × opponent finish rate
# ---------------------------------------------------------------------------


def _compute_interaction_term(
    context: EmitContext,
    fighter_state: CareerFighterState | None,
) -> dict[str, float | None]:
    """Compute never_been_finished × opponent's finish rate.

    The interaction captures whether an unfinished fighter faces a high-finish
    opponent. Returns 0.0 when the fighter has been finished (term is zero),
    or when the opponent's career is unknown or they have zero wins.
    """
    never_finished = _is_never_been_finished(fighter_state)
    if not never_finished:
        return {"never_been_finished_x_opp_finish_rate": 0.0}

    opp_finish_rate = _get_opponent_finish_rate(context)
    return {"never_been_finished_x_opp_finish_rate": opp_finish_rate}


def _is_never_been_finished(state: CareerFighterState | None) -> bool:
    """True when the fighter has never been stopped by KO or submission."""
    if state is None:
        return True
    return (state.times_finished_by_ko + state.times_finished_by_sub) == 0


def _get_opponent_finish_rate(context: EmitContext) -> float:
    """Retrieve the opponent's offensive finish rate from career snapshot.

    Returns 0.0 when opponent is not tracked or has zero wins.
    """
    career_snap = context.components.get("career")
    if not isinstance(career_snap, CareerSnapshot):
        return 0.0

    opp_state = career_snap.get(context.opponent_url)
    if opp_state is None or opp_state.wins == 0:
        return 0.0

    total_finishes = opp_state.ko_wins + opp_state.submission_wins
    return total_finishes / opp_state.wins
