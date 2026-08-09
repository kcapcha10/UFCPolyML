"""Weight dominance feature emitter.

Thin adapter over WeightClassFrozenState and RollingStatsSnapshot. Emits
migration features (class change detection, directional sign, per-class
fight counts and win percentages) and weight bully features (physical size
relative to class via top-quartile detection, grappling utilization rate,
and their product term).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ufc_edge.features.components.rolling_stats import RollingStatsSnapshot
from ufc_edge.features.components.weight_class import WeightClassFrozenState

if TYPE_CHECKING:
    from ufc_edge.features.contracts import EmitContext

# ---------------------------------------------------------------------------
# Weight class ordering (lightest to heaviest) for direction computation
# ---------------------------------------------------------------------------

_WEIGHT_CLASS_ORDER: dict[str, int] = {
    "Women's Strawweight": 0,
    "Women's Flyweight": 1,
    "Women's Bantamweight": 2,
    "Women's Featherweight": 3,
    "Strawweight": 4,
    "Flyweight": 5,
    "Bantamweight": 6,
    "Featherweight": 7,
    "Lightweight": 8,
    "Welterweight": 9,
    "Middleweight": 10,
    "Light Heavyweight": 11,
    "Heavyweight": 12,
}


class WeightDominanceEmitter:
    """Stateless emitter for weight migration and weight bully features.

    Reads WeightClassFrozenState for migration and top-quartile signals.
    Reads RollingStatsSnapshot for grappling utilization. Combines size and
    grappling into the weight_bully_score product term.
    """

    name: str = "weight_dominance"

    def emit(self, context: EmitContext) -> dict[str, float | str | None]:
        """Emit weight dominance features for the focal fighter."""
        weight_state = context.components.get("weight_class")
        if not isinstance(weight_state, WeightClassFrozenState):
            return _all_none()

        fighter_snap = weight_state.get_fighter_state(context.fighter_url)
        if fighter_snap is None:
            return _all_none()

        # Migration features
        is_change = _is_weight_class_change(fighter_snap.first_class, fighter_snap.current_class)
        direction = _direction_of_change(fighter_snap.first_class, fighter_snap.current_class)
        fights_current = float(fighter_snap.fights_at_class.get(fighter_snap.current_class, 0))
        win_pct_current = _win_pct(fighter_snap, fighter_snap.current_class)
        prior_win_pct = _prior_class_win_pct(fighter_snap)

        # Weight bully features
        is_large = _is_large_for_class(weight_state, context)
        grappling_rate = _grappling_utilization_rate(context)
        bully_score = _weight_bully_score(is_large, grappling_rate)

        return {
            "is_weight_class_change": is_change,
            "direction_of_change": direction,
            "fights_at_current_class": fights_current,
            "win_pct_at_current_class": win_pct_current,
            "prior_class_win_pct": prior_win_pct,
            "is_large_for_class": is_large,
            "grappling_utilization_rate": grappling_rate,
            "weight_bully_score": bully_score,
        }


# ---------------------------------------------------------------------------
# Migration helpers
# ---------------------------------------------------------------------------


def _is_weight_class_change(first_class: str, current_class: str) -> float:
    """1.0 if fighter has migrated from their original class, else 0.0."""
    return 1.0 if first_class != current_class else 0.0


def _direction_of_change(first_class: str, current_class: str) -> float | None:
    """Signed direction: +1 up, -1 down, 0 same. None if class not in ordering."""
    first_rank = _WEIGHT_CLASS_ORDER.get(first_class)
    current_rank = _WEIGHT_CLASS_ORDER.get(current_class)
    if first_rank is None or current_rank is None:
        return None
    diff = current_rank - first_rank
    if diff > 0:
        return 1.0
    if diff < 0:
        return -1.0
    return 0.0


def _win_pct(fighter_snap: object, weight_class: str) -> float | None:
    """Compute win percentage at a specific weight class."""
    fights = fighter_snap.fights_at_class.get(weight_class, 0)  # type: ignore[attr-defined]
    if fights == 0:
        return None
    wins = fighter_snap.wins_at_class.get(weight_class, 0)  # type: ignore[attr-defined]
    return wins / fights


def _prior_class_win_pct(fighter_snap: object) -> float | None:
    """Win percentage at the prior (first) class. None if no migration occurred."""
    first = fighter_snap.first_class  # type: ignore[attr-defined]
    current = fighter_snap.current_class  # type: ignore[attr-defined]
    if first == current:
        return None
    return _win_pct(fighter_snap, first)


# ---------------------------------------------------------------------------
# Weight bully helpers
# ---------------------------------------------------------------------------


def _is_large_for_class(
    weight_state: WeightClassFrozenState, context: EmitContext
) -> float:
    """Delegate to WeightClassFrozenState.is_large_for_class; return 1.0/0.0."""
    large = weight_state.is_large_for_class(
        context.fighter_url,
        reach_cm=context.fighter_profile.reach_cm,
        height_cm=context.fighter_profile.height_cm,
    )
    return 1.0 if large else 0.0


def _grappling_utilization_rate(context: EmitContext) -> float | None:
    """Compute (TD attempts + control time) averaged per fight from rolling window.

    Accesses the raw per-fight stats window to sum takedown attempts and control
    time seconds, then divides by fight count. Returns None if no window data.
    """
    rolling = context.components.get("rolling_stats")
    if not isinstance(rolling, RollingStatsSnapshot):
        return None

    window = rolling._fighter_windows.get(context.fighter_url)  # noqa: SLF001
    if not window:
        return None

    total = sum(s.takedowns_attempted + s.control_time_seconds for s in window)
    return total / len(window)


def _weight_bully_score(is_large: float, grappling_rate: float | None) -> float | None:
    """Product term: is_large_for_class × grappling_utilization_rate.

    Returns None when grappling rate is unavailable. Returns 0.0 when
    fighter is not large for class (regardless of grappling).
    """
    if grappling_rate is None:
        return None
    return is_large * grappling_rate


# ---------------------------------------------------------------------------
# Default None result
# ---------------------------------------------------------------------------


def _all_none() -> dict[str, float | str | None]:
    """Return dict with all weight dominance columns set to None."""
    return {
        "is_weight_class_change": None,
        "direction_of_change": None,
        "fights_at_current_class": None,
        "win_pct_at_current_class": None,
        "prior_class_win_pct": None,
        "is_large_for_class": None,
        "grappling_utilization_rate": None,
        "weight_bully_score": None,
    }
