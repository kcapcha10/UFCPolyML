"""Physical profile feature emitter.

Emits static fighter attributes from the FighterProfile stored on EmitContext:
height, reach, reach-to-height ratio, stance, age at fight time, and the
scheduled weight class. All derived values gracefully degrade to None when
source measurements are missing.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ufc_edge.features.contracts import EmitContext


class PhysicalEmitter:
    """Emits physical profile features for the focal fighter.

    Reads height, reach, stance, and date of birth from the fighter's profile
    on EmitContext. Computes reach-to-height ratio and age at fight time.
    Returns None for any derived field when its source data is missing.
    """

    name: str = "physical"

    def emit(self, context: EmitContext) -> dict[str, float | str | None]:
        """Emit physical profile features for the focal fighter."""
        profile = context.fighter_profile

        height_cm = profile.height_cm
        reach_cm = profile.reach_cm
        stance = profile.stance
        weight_class = context.weight_class

        reach_to_height_ratio = _compute_reach_to_height_ratio(height_cm, reach_cm)
        age_at_fight = _compute_age_at_fight(profile.dob, context.event_date)

        return {
            "height_cm": height_cm,
            "reach_cm": reach_cm,
            "reach_to_height_ratio": reach_to_height_ratio,
            "stance": stance,
            "age_at_fight": age_at_fight,
            "weight_class": weight_class,
        }


def _compute_reach_to_height_ratio(
    height_cm: float | None, reach_cm: float | None
) -> float | None:
    """Compute reach/height ratio, returning None if either measurement is missing."""
    if height_cm is None or reach_cm is None:
        return None
    return reach_cm / height_cm


def _compute_age_at_fight(dob: date | None, event_date: date) -> float | None:
    """Compute fighter's integer age on fight day, returning None if DOB is missing.

    Age is the number of full years lived as of event_date (standard birthday rule).
    """
    if dob is None:
        return None
    age = event_date.year - dob.year
    if (event_date.month, event_date.day) < (dob.month, dob.day):
        age -= 1
    return float(age)
