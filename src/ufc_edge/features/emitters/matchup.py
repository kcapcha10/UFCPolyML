"""MatchupEmitter — pairwise matchup deltas and style interaction features.

Computes all matchup-level features using the A−B convention (positive values
favor the focal fighter). Reads physical profiles, career snapshots, rolling
stats, Elo ratings, and PageRank scores from EmitContext to derive:
  - Physical deltas (reach, height, age)
  - Stance matchup classification
  - Rating deltas (Elo, PageRank)
  - Career-derived deltas (finish rate, experience)
  - Output-efficiency deltas (striking accuracy, TD accuracy, damage ratio)
  - Duration and variance deltas (pace mismatch proxy)
  - Grappling sub-type scores (wrestler, submission) and mismatch detection
  - Style interactions (striker vs grappler, pressure vs counter, pace mismatch)
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from ufc_edge.features.components.career import CareerFighterState, CareerSnapshot
from ufc_edge.features.components.elo import _FrozenEloSnapshot
from ufc_edge.features.components.pagerank import PageRankFrozenState
from ufc_edge.features.components.rolling_stats import RollingStatsSnapshot

if TYPE_CHECKING:
    from ufc_edge.features.contracts import EmitContext

# Minimum fights to compute meaningful variance
_MIN_FIGHTS_FOR_VARIANCE = 3

# Thresholds for style classification
_STRIKER_SIG_PER_MIN_THRESHOLD = 4.0
_GRAPPLER_TD_PER_15_THRESHOLD = 3.0
_PRESSURE_OUTPUT_THRESHOLD = 5.0
_COUNTER_DAMAGE_RATIO_THRESHOLD = 1.5
_COUNTER_LOW_OUTPUT_THRESHOLD = 4.0


class MatchupEmitter:
    """Stateless emitter for matchup-level pairwise features.

    All deltas use the A−B convention: focal fighter value minus opponent value.
    When either side's data is unavailable, the corresponding delta is None.
    """

    name: str = "matchup"

    def emit(self, context: EmitContext) -> dict[str, float | str | None]:
        """Emit matchup features comparing the focal fighter against the opponent."""
        physical = _compute_physical_deltas(context)
        stance = _compute_stance_features(context)
        ratings = _compute_rating_deltas(context)
        career_deltas = _compute_career_deltas(context)
        output_deltas = _compute_output_deltas(context)
        duration = _compute_duration_deltas(context)
        grappling = _compute_grappling_subtype(context)
        style = _compute_style_interactions(context)

        return {
            **physical,
            **stance,
            **ratings,
            **career_deltas,
            **output_deltas,
            **duration,
            **grappling,
            **style,
        }


# ---------------------------------------------------------------------------
# Physical deltas
# ---------------------------------------------------------------------------


def _compute_physical_deltas(context: EmitContext) -> dict[str, float | None]:
    """Compute reach, height, and age deltas (A−B)."""
    f_prof = context.fighter_profile
    o_prof = context.opponent_profile

    reach_delta = _safe_delta(f_prof.reach_cm, o_prof.reach_cm)
    height_delta = _safe_delta(f_prof.height_cm, o_prof.height_cm)
    age_delta = _compute_age_delta(f_prof.dob, o_prof.dob, context.event_date)

    return {
        "reach_delta": reach_delta,
        "height_delta": height_delta,
        "age_delta": age_delta,
    }


def _compute_age_delta(
    dob_a: date | None, dob_b: date | None, event_date: date
) -> float | None:
    """Compute age difference (A−B) in years as of event date."""
    if dob_a is None or dob_b is None:
        return None
    age_a = _compute_age(dob_a, event_date)
    age_b = _compute_age(dob_b, event_date)
    return age_a - age_b


def _compute_age(dob: date, event_date: date) -> float:
    """Compute integer age as of a given date."""
    age = event_date.year - dob.year
    if (event_date.month, event_date.day) < (dob.month, dob.day):
        age -= 1
    return float(age)


# ---------------------------------------------------------------------------
# Stance features
# ---------------------------------------------------------------------------


def _compute_stance_features(context: EmitContext) -> dict[str, float | str | None]:
    """Classify stance matchup and southpaw flag."""
    f_stance = context.fighter_profile.stance
    o_stance = context.opponent_profile.stance

    if f_stance is None or o_stance is None:
        return {
            "stance_matchup": None,
            "southpaw_matchup": None,
        }

    matchup = _classify_stance(f_stance, o_stance)
    has_southpaw = "Southpaw" in (f_stance, o_stance)

    return {
        "stance_matchup": matchup,
        "southpaw_matchup": 1.0 if has_southpaw else 0.0,
    }


def _classify_stance(stance_a: str, stance_b: str) -> str:
    """Produce a canonical stance matchup label.

    Categories: ortho_v_ortho, ortho_v_south, south_v_south, switch_involved.
    """
    if "Switch" in (stance_a, stance_b):
        return "switch_involved"

    a_south = stance_a == "Southpaw"
    b_south = stance_b == "Southpaw"

    if a_south and b_south:
        return "south_v_south"
    if a_south or b_south:
        return "ortho_v_south"
    return "ortho_v_ortho"


# ---------------------------------------------------------------------------
# Rating deltas (Elo, PageRank)
# ---------------------------------------------------------------------------


def _compute_rating_deltas(context: EmitContext) -> dict[str, float | None]:
    """Compute Elo and PageRank deltas (A−B)."""
    elo_delta = _compute_elo_delta(context)
    pagerank_delta = _compute_pagerank_delta(context)

    return {
        "elo_delta": elo_delta,
        "pagerank_delta": pagerank_delta,
    }


def _compute_elo_delta(context: EmitContext) -> float | None:
    """Retrieve Elo ratings for both fighters and compute delta."""
    elo_snap = context.components.get("elo")
    if not isinstance(elo_snap, _FrozenEloSnapshot):
        return None

    f_record = elo_snap.get(context.fighter_url)
    o_record = elo_snap.get(context.opponent_url)

    if f_record is None or o_record is None:
        return None
    return f_record.rating - o_record.rating


def _compute_pagerank_delta(context: EmitContext) -> float | None:
    """Retrieve PageRank scores for both fighters and compute delta."""
    pr_snap = context.components.get("pagerank")
    if not isinstance(pr_snap, PageRankFrozenState):
        return None

    scores = pr_snap.scores
    f_score = scores.get(context.fighter_url)
    o_score = scores.get(context.opponent_url)

    if f_score is None or o_score is None:
        return None
    return f_score - o_score


# ---------------------------------------------------------------------------
# Career-derived deltas
# ---------------------------------------------------------------------------


def _compute_career_deltas(context: EmitContext) -> dict[str, float | None]:
    """Compute deltas from career state: finish rate and experience features."""
    career_snap = context.components.get("career")
    if not isinstance(career_snap, CareerSnapshot):
        return {
            "finish_rate_delta": None,
            "five_round_experience_delta": None,
            "ufc_experience_delta": None,
            "title_fight_exp_delta": None,
        }

    f_state = career_snap.get(context.fighter_url)
    o_state = career_snap.get(context.opponent_url)

    finish_rate_delta = _delta_from_career(f_state, o_state, _get_finish_rate)
    ufc_exp_delta = _delta_from_career(f_state, o_state, _get_ufc_experience)

    # Five-round and title fight experience tracked via total_fights in career
    # These are placeholders referencing experience component (not yet built);
    # emit None until the ExperienceEmitter's state is available.
    five_round_delta = None
    title_fight_delta = None

    return {
        "finish_rate_delta": finish_rate_delta,
        "five_round_experience_delta": five_round_delta,
        "ufc_experience_delta": ufc_exp_delta,
        "title_fight_exp_delta": title_fight_delta,
    }


def _get_finish_rate(state: CareerFighterState) -> float | None:
    """Compute finish rate (finishes/wins) for a fighter."""
    if state.wins == 0:
        return None
    total_finishes = state.ko_wins + state.submission_wins
    return total_finishes / state.wins


def _get_ufc_experience(state: CareerFighterState) -> float | None:
    """Get UFC fight count as experience metric."""
    return float(state.total_fights)


# ---------------------------------------------------------------------------
# Output-efficiency deltas from rolling stats
# ---------------------------------------------------------------------------


def _compute_output_deltas(context: EmitContext) -> dict[str, float | None]:
    """Compute striking efficiency, TD accuracy, and damage ratio deltas."""
    rolling_snap = context.components.get("rolling_stats")
    if not isinstance(rolling_snap, RollingStatsSnapshot):
        return {
            "striking_efficiency_delta": None,
            "td_accuracy_delta": None,
            "damage_ratio_delta": None,
        }

    f_avgs = rolling_snap.get_rolling_averages(context.fighter_url)
    o_avgs = rolling_snap.get_rolling_averages(context.opponent_url)

    striking_eff_delta = _delta_from_averages(
        f_avgs, o_avgs, "striking_accuracy_pct"
    )
    td_acc_delta = _delta_from_averages(f_avgs, o_avgs, "td_accuracy_pct")
    damage_ratio_delta = _delta_from_averages(f_avgs, o_avgs, "damage_ratio")

    return {
        "striking_efficiency_delta": striking_eff_delta,
        "td_accuracy_delta": td_acc_delta,
        "damage_ratio_delta": damage_ratio_delta,
    }


# ---------------------------------------------------------------------------
# Duration and variance deltas
# ---------------------------------------------------------------------------


def _compute_duration_deltas(context: EmitContext) -> dict[str, float | None]:
    """Compute avg fight duration and duration variance deltas (in seconds)."""
    rolling_snap = context.components.get("rolling_stats")
    if not isinstance(rolling_snap, RollingStatsSnapshot):
        return {
            "avg_fight_duration_delta": None,
            "fight_duration_variance_delta": None,
        }

    f_dur = _get_duration_stats(rolling_snap, context.fighter_url)
    o_dur = _get_duration_stats(rolling_snap, context.opponent_url)

    avg_delta = _safe_delta(
        f_dur[0] if f_dur else None,
        o_dur[0] if o_dur else None,
    )
    var_delta = _safe_delta(
        f_dur[1] if f_dur else None,
        o_dur[1] if o_dur else None,
    )

    return {
        "avg_fight_duration_delta": avg_delta,
        "fight_duration_variance_delta": var_delta,
    }


def _get_duration_stats(
    snap: RollingStatsSnapshot, fighter_url: str
) -> tuple[float, float | None] | None:
    """Get (avg_duration_sec, variance_sec) for a fighter from rolling window.

    Returns None when no rolling data exists. Variance is None when fewer than
    3 fights are available.
    """
    window = snap._fighter_windows.get(fighter_url)
    if not window:
        return None

    durations_min = [stats.fight_duration_minutes for stats in window]
    n = len(durations_min)
    mean_min = sum(durations_min) / n
    avg_sec = mean_min * 60.0

    if n < _MIN_FIGHTS_FOR_VARIANCE:
        return (avg_sec, None)

    variance_min = sum((d - mean_min) ** 2 for d in durations_min) / n
    variance_sec = variance_min * 3600.0
    return (avg_sec, variance_sec)


# ---------------------------------------------------------------------------
# Grappling sub-type matchup
# ---------------------------------------------------------------------------


def _compute_grappling_subtype(context: EmitContext) -> dict[str, float | None]:
    """Compute wrestler/submission scores, deltas, and type mismatch flag.

    wrestler_score = td_accuracy * td_per_15min * td_defense
    submission_score = sub_attempts_per_15min * submission_rate
    """
    rolling_snap = context.components.get("rolling_stats")
    career_snap = context.components.get("career")

    f_wrestler = _compute_wrestler_score(rolling_snap, context.fighter_url)
    o_wrestler = _compute_wrestler_score(rolling_snap, context.opponent_url)
    f_sub = _compute_submission_score(rolling_snap, career_snap, context.fighter_url)
    o_sub = _compute_submission_score(rolling_snap, career_snap, context.opponent_url)

    wrestling_delta = _safe_delta(f_wrestler, o_wrestler)
    submission_delta = _safe_delta(f_sub, o_sub)

    mismatch = _detect_grappling_mismatch(f_wrestler, o_wrestler, f_sub, o_sub)

    return {
        "wrestler_score_a": f_wrestler,
        "wrestler_score_b": o_wrestler,
        "submission_score_a": f_sub,
        "submission_score_b": o_sub,
        "wrestling_delta": wrestling_delta,
        "submission_delta": submission_delta,
        "grappling_type_mismatch": mismatch,
    }


def _compute_wrestler_score(
    rolling_snap: object | None, fighter_url: str
) -> float | None:
    """Compute wrestler_score = td_accuracy * td_per_15 * td_defense."""
    if not isinstance(rolling_snap, RollingStatsSnapshot):
        return None

    avgs = rolling_snap.get_rolling_averages(fighter_url)
    if avgs is None:
        return None

    td_acc = avgs.get("td_accuracy_pct")
    td_per_15 = avgs.get("td_per_15min")
    td_def = avgs.get("td_defense_pct")

    if td_acc is None or td_per_15 is None or td_def is None:
        return None
    return td_acc * td_per_15 * td_def


def _compute_submission_score(
    rolling_snap: object | None,
    career_snap: object | None,
    fighter_url: str,
) -> float | None:
    """Compute submission_score = sub_attempts_per_15 * submission_rate."""
    if not isinstance(rolling_snap, RollingStatsSnapshot):
        return None

    avgs = rolling_snap.get_rolling_averages(fighter_url)
    if avgs is None:
        return None

    sub_per_15 = avgs.get("sub_attempts_per_15min")
    if sub_per_15 is None:
        return None

    sub_rate = _get_submission_rate(career_snap, fighter_url)
    if sub_rate is None:
        return None

    return sub_per_15 * sub_rate


def _get_submission_rate(career_snap: object | None, fighter_url: str) -> float | None:
    """Get submission win rate from career state."""
    if not isinstance(career_snap, CareerSnapshot):
        return None
    state = career_snap.get(fighter_url)
    if state is None or state.wins == 0:
        return None
    return state.submission_wins / state.wins


def _detect_grappling_mismatch(
    f_wrestler: float | None,
    o_wrestler: float | None,
    f_sub: float | None,
    o_sub: float | None,
) -> float | None:
    """Detect when one side is wrestler-heavy and the other is submission-heavy.

    Returns 1.0 when one fighter has a significantly higher wrestler score and
    the other has a significantly higher submission score. Returns 0.0 otherwise.
    Returns None when scores are unavailable.
    """
    if any(v is None for v in (f_wrestler, o_wrestler, f_sub, o_sub)):
        return None

    # Asymmetry: one side wrestler-dominant, other side submission-dominant
    a_wrestle_dominant = f_wrestler > o_wrestler and o_sub > f_sub
    b_wrestle_dominant = o_wrestler > f_wrestler and f_sub > o_sub

    if a_wrestle_dominant or b_wrestle_dominant:
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Style interactions
# ---------------------------------------------------------------------------


def _compute_style_interactions(context: EmitContext) -> dict[str, float | None]:
    """Compute striker_vs_grappler, pressure_vs_counter, pace mismatch, southpaw history."""
    rolling_snap = context.components.get("rolling_stats")
    if not isinstance(rolling_snap, RollingStatsSnapshot):
        return {
            "striker_vs_grappler": None,
            "pressure_vs_counter": None,
            "pace_mismatch_score": None,
            "southpaw_orthodox_history": None,
        }

    f_avgs = rolling_snap.get_rolling_averages(context.fighter_url)
    o_avgs = rolling_snap.get_rolling_averages(context.opponent_url)

    striker_grappler = _detect_striker_vs_grappler(f_avgs, o_avgs)
    pressure_counter = _detect_pressure_vs_counter(f_avgs, o_avgs)
    pace_mismatch = _compute_pace_mismatch(context)
    southpaw_history = _compute_southpaw_history(context)

    return {
        "striker_vs_grappler": striker_grappler,
        "pressure_vs_counter": pressure_counter,
        "pace_mismatch_score": pace_mismatch,
        "southpaw_orthodox_history": southpaw_history,
    }


def _detect_striker_vs_grappler(
    f_avgs: dict[str, float | None] | None,
    o_avgs: dict[str, float | None] | None,
) -> float | None:
    """Detect a striker-vs-grappler stylistic clash.

    A fighter is classified as a striker when their sig_strikes_per_min exceeds
    the threshold and their td_per_15min is low. A grappler has high td_per_15min.
    Returns 1.0 when one side is a striker and the other is a grappler.
    """
    if f_avgs is None or o_avgs is None:
        return None

    f_striker = _is_striker(f_avgs)
    f_grappler = _is_grappler(f_avgs)
    o_striker = _is_striker(o_avgs)
    o_grappler = _is_grappler(o_avgs)

    if (f_striker and o_grappler) or (f_grappler and o_striker):
        return 1.0
    return 0.0


def _is_striker(avgs: dict[str, float | None]) -> bool:
    """True when a fighter's output profile is striker-dominant."""
    sig_per_min = avgs.get("sig_strikes_per_min")
    td_per_15 = avgs.get("td_per_15min")
    if sig_per_min is None:
        return False
    return sig_per_min >= _STRIKER_SIG_PER_MIN_THRESHOLD and (
        td_per_15 is None or td_per_15 < _GRAPPLER_TD_PER_15_THRESHOLD
    )


def _is_grappler(avgs: dict[str, float | None]) -> bool:
    """True when a fighter's output profile is grappler-dominant."""
    td_per_15 = avgs.get("td_per_15min")
    if td_per_15 is None:
        return False
    return td_per_15 >= _GRAPPLER_TD_PER_15_THRESHOLD


def _detect_pressure_vs_counter(
    f_avgs: dict[str, float | None] | None,
    o_avgs: dict[str, float | None] | None,
) -> float | None:
    """Detect a pressure-fighter vs counter-fighter stylistic clash.

    Pressure: high sig_strikes_per_min output.
    Counter: high damage_ratio with relatively low output volume.
    Returns 1.0 when one side is pressure and the other is counter.
    """
    if f_avgs is None or o_avgs is None:
        return None

    f_pressure = _is_pressure(f_avgs)
    f_counter = _is_counter(f_avgs)
    o_pressure = _is_pressure(o_avgs)
    o_counter = _is_counter(o_avgs)

    if (f_pressure and o_counter) or (f_counter and o_pressure):
        return 1.0
    return 0.0


def _is_pressure(avgs: dict[str, float | None]) -> bool:
    """True when a fighter has pressure-style output (high volume)."""
    sig_per_min = avgs.get("sig_strikes_per_min")
    if sig_per_min is None:
        return False
    return sig_per_min >= _PRESSURE_OUTPUT_THRESHOLD


def _is_counter(avgs: dict[str, float | None]) -> bool:
    """True when a fighter has counter-style output (high efficiency, low volume)."""
    damage_ratio = avgs.get("damage_ratio")
    sig_per_min = avgs.get("sig_strikes_per_min")
    if damage_ratio is None or sig_per_min is None:
        return False
    return (
        damage_ratio >= _COUNTER_DAMAGE_RATIO_THRESHOLD
        and sig_per_min < _COUNTER_LOW_OUTPUT_THRESHOLD
    )


def _compute_pace_mismatch(context: EmitContext) -> float | None:
    """Compute pace mismatch = sig_strikes_per_min delta × duration variance delta.

    Captures the interaction between pace difference and variance difference:
    a fast-paced fighter with consistent durations vs a slow fighter with
    erratic durations signals a meaningful pace clash.
    """
    rolling_snap = context.components.get("rolling_stats")
    if not isinstance(rolling_snap, RollingStatsSnapshot):
        return None

    f_avgs = rolling_snap.get_rolling_averages(context.fighter_url)
    o_avgs = rolling_snap.get_rolling_averages(context.opponent_url)

    if f_avgs is None or o_avgs is None:
        return None

    f_sig = f_avgs.get("sig_strikes_per_min")
    o_sig = o_avgs.get("sig_strikes_per_min")
    if f_sig is None or o_sig is None:
        return None

    sig_delta = f_sig - o_sig

    # Get variance delta (in seconds^2)
    f_dur = _get_duration_stats(rolling_snap, context.fighter_url)
    o_dur = _get_duration_stats(rolling_snap, context.opponent_url)

    f_var = f_dur[1] if f_dur else None
    o_var = o_dur[1] if o_dur else None

    if f_var is None or o_var is None:
        return None

    var_delta = f_var - o_var
    return sig_delta * var_delta


def _compute_southpaw_history(context: EmitContext) -> float | None:
    """Compute fighter-specific historical win rate vs southpaws.

    This requires fight-by-fight opponent stance data which is not currently
    tracked in the career or rolling stats components. Returns None until
    the requisite historical data is available from the replay engine.
    """
    # Southpaw-specific fight history is not yet tracked in any component.
    # The feature requires per-fight opponent stance tracking in the career
    # accumulator or a dedicated component. Emit None for now.
    return None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _safe_delta(a: float | None, b: float | None) -> float | None:
    """Compute A−B, returning None if either value is unavailable."""
    if a is None or b is None:
        return None
    return a - b


def _delta_from_career(
    f_state: CareerFighterState | None,
    o_state: CareerFighterState | None,
    extractor: object,
) -> float | None:
    """Compute delta from a career field extractor function.

    Returns None when either fighter's state or extracted value is unavailable.
    """
    if f_state is None or o_state is None:
        return None
    f_val = extractor(f_state)
    o_val = extractor(o_state)
    return _safe_delta(f_val, o_val)


def _delta_from_averages(
    f_avgs: dict[str, float | None] | None,
    o_avgs: dict[str, float | None] | None,
    key: str,
) -> float | None:
    """Compute delta for a specific rolling average key."""
    if f_avgs is None or o_avgs is None:
        return None
    f_val = f_avgs.get(key)
    o_val = o_avgs.get(key)
    return _safe_delta(f_val, o_val)
