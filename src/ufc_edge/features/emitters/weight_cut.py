"""Weight-cut and short-notice feature emitter.

Emits weight-class migration direction and placeholder fields for data that
cannot yet be derived from structured ufcstats sources.

Derivable fields:
    - moving_down_in_weight: 1.0 if the fighter's current weight class is
      lighter than their first (natural) class, 0.0 otherwise. Uses an
      explicit weight class hierarchy ordered from lightest to heaviest.

Not yet derivable (always None):
    - missed_weight_last_3: Number of times the fighter missed weight in
      their last 3 fights. UFCStats does not track weigh-in failures.
    - missed_weight_career: Career total missed-weight count. Same data gap.
    - short_notice: Whether the fighter took the bout on short notice.
      No structured announcement/booking date data exists.
    - full_camp: Whether the fighter had a full training camp. Same data gap.

These None-valued fields are present so downstream consumers have a stable
schema. When a structured data source becomes available (e.g. Kaggle dataset,
manual annotations), the emitter can be extended without schema changes.
"""

from __future__ import annotations

from ufc_edge.features.components.weight_class import WeightClassFrozenState
from ufc_edge.features.contracts import EmitContext

# ---------------------------------------------------------------------------
# Weight class hierarchy (lightest to heaviest)
# ---------------------------------------------------------------------------

_WEIGHT_CLASS_RANK: dict[str, int] = {
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


# ---------------------------------------------------------------------------
# WeightCutEmitter
# ---------------------------------------------------------------------------


class WeightCutEmitter:
    """Stateless emitter for weight-cut and camp-readiness features.

    Reads the WeightClassFrozenState from EmitContext to determine if the
    focal fighter has moved down in weight. Missed-weight and short-notice
    fields return None because no structured source exists in ufcstats.
    """

    name: str = "weight_cut"

    def emit(self, context: EmitContext) -> dict[str, float | str | None]:
        """Emit weight-cut features for the focal fighter."""
        weight_state: WeightClassFrozenState = context.components["weight_class"]  # type: ignore[assignment]
        moving_down = self._compute_moving_down(weight_state, context.fighter_url)

        return {
            "missed_weight_last_3": None,
            "missed_weight_career": None,
            "moving_down_in_weight": moving_down,
            "short_notice": None,
            "full_camp": None,
        }

    def _compute_moving_down(
        self,
        weight_state: WeightClassFrozenState,
        fighter_url: str,
    ) -> float | None:
        """Determine if the fighter moved to a lighter division.

        Compares the rank of the fighter's first UFC weight class to their
        current class. Returns 1.0 if current is lighter, 0.0 if same or
        heavier, None if the fighter is unknown or either class is not in
        the recognized hierarchy.
        """
        fighter_snapshot = weight_state.get_fighter_state(fighter_url)
        if fighter_snapshot is None:
            return None

        first_rank = _WEIGHT_CLASS_RANK.get(fighter_snapshot.first_class)
        current_rank = _WEIGHT_CLASS_RANK.get(fighter_snapshot.current_class)

        if first_rank is None or current_rank is None:
            return None

        if current_rank < first_rank:
            return 1.0
        return 0.0
