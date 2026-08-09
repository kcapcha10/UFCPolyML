"""Activity and inactivity feature emitter.

Derives time-based activity features from the frozen CareerSnapshot: days since
last fight, windowed fight counts (12-month, 3-year, 5-year), total UFC fights,
last-fight injury stoppage flag, age × inactivity interaction term, and bucketed
inactivity tier.
"""

from __future__ import annotations

from datetime import date, timedelta

from ufc_edge.features.components.career import CareerFighterState, CareerSnapshot
from ufc_edge.features.contracts import EmitContext

# ---------------------------------------------------------------------------
# Inactivity tier boundaries (in days)
# ---------------------------------------------------------------------------

_TIER_1_THRESHOLD = 180  # 6 months
_TIER_2_THRESHOLD = 365  # 1 year
_TIER_3_THRESHOLD = 730  # 2 years

# Windowed count boundaries (in days)
_WINDOW_12MO = 365
_WINDOW_3YR = 1095
_WINDOW_5YR = 1825

# Methods indicating an injury stoppage — fight is rating-neutral.
_INJURY_METHODS = frozenset(
    {
        "Could Not Continue",
        "Could Not Continue - Injury",
        "TKO - Doctor's Stoppage",
        "Doctor's Stoppage",
        "Overturned - Injury",
    }
)


# ---------------------------------------------------------------------------
# ActivityEmitter
# ---------------------------------------------------------------------------


class ActivityEmitter:
    """Stateless emitter for activity and inactivity features.

    Reads CareerSnapshot from the EmitContext components mapping. Returns None
    for features that cannot be computed (debut fighters, missing DOB).
    """

    name: str = "activity"

    def emit(self, context: EmitContext) -> dict[str, float | str | None]:
        """Emit activity features for the focal fighter."""
        snapshot: CareerSnapshot = context.components["career"]  # type: ignore[assignment]
        fighter_state = snapshot.get(context.fighter_url)

        if fighter_state is None:
            return self._debut_result()

        days_since = self._days_since_last_fight(fighter_state, context.event_date)
        fights_12 = self._count_fights_in_window(fighter_state, context.event_date, _WINDOW_12MO)
        fights_3yr = self._count_fights_in_window(fighter_state, context.event_date, _WINDOW_3YR)
        fights_5yr = self._count_fights_in_window(fighter_state, context.event_date, _WINDOW_5YR)
        total = fighter_state.total_fights
        injury = self._last_fight_injury_stoppage(fighter_state)
        age_inact = self._age_x_inactivity(context, days_since)
        tier = self._inactivity_tier(days_since)

        return {
            "days_since_last_fight": days_since,
            "fights_last_12mo": float(fights_12),
            "fights_last_3yr": float(fights_3yr),
            "fights_last_5yr": float(fights_5yr),
            "total_ufc_fights": float(total),
            "last_fight_injury_stoppage": injury,
            "age_x_inactivity": age_inact,
            "inactivity_tier": tier,
        }

    # ------------------------------------------------------------------
    # Private computation methods
    # ------------------------------------------------------------------

    def _debut_result(self) -> dict[str, float | str | None]:
        """Feature values for a fighter making their UFC debut."""
        return {
            "days_since_last_fight": None,
            "fights_last_12mo": 0.0,
            "fights_last_3yr": 0.0,
            "fights_last_5yr": 0.0,
            "total_ufc_fights": 0.0,
            "last_fight_injury_stoppage": None,
            "age_x_inactivity": None,
            "inactivity_tier": None,
        }

    def _days_since_last_fight(
        self, state: CareerFighterState, event_date: date
    ) -> float | None:
        """Days between the fighter's most recent fight and the current event."""
        if state.last_fight_date is None:
            return None
        return float((event_date - state.last_fight_date).days)

    def _count_fights_in_window(
        self, state: CareerFighterState, event_date: date, window_days: int
    ) -> int:
        """Count fights within a rolling window ending at event_date."""
        cutoff = event_date - timedelta(days=window_days)
        return sum(1 for d in state.fight_dates if d > cutoff)

    def _last_fight_injury_stoppage(self, state: CareerFighterState) -> float | None:
        """1.0 if the most recent fight ended by injury stoppage, else 0.0."""
        if state.last_fight_method is None:
            return None
        return 1.0 if state.last_fight_method in _INJURY_METHODS else 0.0

    def _age_x_inactivity(
        self, context: EmitContext, days_since: float | None
    ) -> float | None:
        """Interaction term: fighter age at fight × days since last fight."""
        if days_since is None:
            return None
        dob = context.fighter_profile.dob
        if dob is None:
            return None
        age_years = (context.event_date - dob).days / 365.25
        return age_years * days_since

    def _inactivity_tier(self, days_since: float | None) -> float | None:
        """Bucketed inactivity: 0=<180d, 1=180–365d, 2=366–730d, 3=731d+."""
        if days_since is None:
            return None
        days = int(days_since)
        if days < _TIER_1_THRESHOLD:
            return 0.0
        if days <= _TIER_2_THRESHOLD:
            return 1.0
        if days <= _TIER_3_THRESHOLD:
            return 2.0
        return 3.0
