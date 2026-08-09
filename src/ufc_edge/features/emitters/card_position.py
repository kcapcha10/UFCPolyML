"""Card-position emitter: splits output metrics by main-card vs prelim placement.

Gated on bout_order availability. When the data layer does not populate
bout_order (currently the case — the scraper extension has not been built),
this emitter emits None for all output columns and logs a warning. When
bout_order history becomes available via a dedicated card-position history
component, the emitter applies the main-card heuristic (top-5 bouts by
bout_order on numbered events) to split striking and grappling averages.
"""

from __future__ import annotations

import logging
from typing import Any

from ufc_edge.features.contracts import EmitContext

logger = logging.getLogger(__name__)

# Component key expected in EmitContext.components for card-position history
_HISTORY_COMPONENT_KEY = "card_position_history"

# Main-card heuristic: top-N bouts by bout_order constitute the main card
_MAIN_CARD_TOP_N = 5

# Minimum number of historical fights required for variance features
_MIN_FIGHTS_FOR_VARIANCE = 3


class CardPositionEmitter:
    """Emits output metrics split by card position (main card vs prelim).

    Checks bout_order availability on the current fight context and whether a
    card-position history component is registered. When either is missing, all
    outputs are None. When available, splits historical fights into main-card
    (top-5 by bout_order) and prelim groups, computing per-group averages.
    """

    name: str = "card_position"

    def emit(self, context: EmitContext) -> dict[str, float | str | None]:
        """Emit card-position features for the focal fighter.

        Returns dict with keys: sig_strikes_main_card_avg, sig_strikes_prelim_avg,
        td_rate_main_card_avg, td_rate_prelim_avg, grappling_abandonment_delta,
        output_variance_by_position.
        """
        # Gate: check if bout_order data is available
        if context.bout_order is None:
            logger.warning(
                "bout_order unavailable for fight %s — card-position features "
                "gated until scraper extension persists bout_order data",
                context.fight_url,
            )
            return _all_none()

        # Gate: check if card-position history component is registered
        history_component = context.components.get(_HISTORY_COMPONENT_KEY)
        if history_component is None:
            logger.warning(
                "card_position_history component not registered — card-position "
                "features gated until component tracks per-fight bout_order history"
            )
            return _all_none()

        # Retrieve fighter's historical fight records with bout_order data
        fight_history = history_component.get_fight_history(context.fighter_url)  # type: ignore[attr-defined]
        if not fight_history:
            return _all_none()

        return _compute_card_position_features(fight_history)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _all_none() -> dict[str, float | str | None]:
    """Return the full output dict with all values set to None."""
    return {
        "sig_strikes_main_card_avg": None,
        "sig_strikes_prelim_avg": None,
        "td_rate_main_card_avg": None,
        "td_rate_prelim_avg": None,
        "grappling_abandonment_delta": None,
        "output_variance_by_position": None,
    }


def _compute_card_position_features(
    fight_history: list[dict[str, Any]],
) -> dict[str, float | str | None]:
    """Split fights by card position and compute per-group metrics.

    Main-card heuristic: top-5 fights by bout_order are main card, the rest
    are prelim. Computes average sig_strikes_per_min and td_per_15min for each
    group, plus grappling abandonment delta and output variance by position.
    """
    # Sort by bout_order descending; top-N are main card
    sorted_fights = sorted(
        fight_history, key=lambda f: f["bout_order"], reverse=True
    )
    main_card = sorted_fights[:_MAIN_CARD_TOP_N]
    prelim = sorted_fights[_MAIN_CARD_TOP_N:]

    # Compute per-group averages
    main_sig = _avg(main_card, "sig_strikes_per_min")
    prelim_sig = _avg(prelim, "sig_strikes_per_min")
    main_td = _avg(main_card, "td_per_15min")
    prelim_td = _avg(prelim, "td_per_15min")

    # Grappling abandonment delta: main_card_avg - prelim_avg
    main_grappling = _avg(main_card, "grappling_dominance")
    prelim_grappling = _avg(prelim, "grappling_dominance")
    grappling_delta: float | None = None
    if main_grappling is not None and prelim_grappling is not None:
        grappling_delta = main_grappling - prelim_grappling

    # Output variance by position: requires 3+ total fights
    variance: float | None = None
    total_fights = len(fight_history)
    if total_fights >= _MIN_FIGHTS_FOR_VARIANCE:
        variance = _position_variance(main_sig, prelim_sig)

    return {
        "sig_strikes_main_card_avg": main_sig,
        "sig_strikes_prelim_avg": prelim_sig,
        "td_rate_main_card_avg": main_td,
        "td_rate_prelim_avg": prelim_td,
        "grappling_abandonment_delta": grappling_delta,
        "output_variance_by_position": variance,
    }


def _avg(fights: list[dict[str, Any]], key: str) -> float | None:
    """Compute average of a metric across fights. Returns None if list is empty."""
    if not fights:
        return None
    values = [f[key] for f in fights if f.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _position_variance(
    main_avg: float | None, prelim_avg: float | None
) -> float | None:
    """Compute population variance of position-grouped sig_strike averages.

    Treats main_card_avg and prelim_avg as two data points. Returns None when
    either group has no data (cannot compute meaningful variance).
    """
    if main_avg is None or prelim_avg is None:
        return None
    mean = (main_avg + prelim_avg) / 2.0
    variance = ((main_avg - mean) ** 2 + (prelim_avg - mean) ** 2) / 2.0
    return variance
