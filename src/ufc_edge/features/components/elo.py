"""Elo rating tracker for UFC fighters.

Accumulates Elo ratings with variable K-factor, method-based bonuses,
inactivity decay toward the population mean, and special handling for
injury stoppages and disqualifications. Exposes a frozen EloRecord per
fighter via freeze().
"""

from __future__ import annotations

import math
from collections import deque
from datetime import date
from typing import TYPE_CHECKING

from ufc_edge.features.contracts import FrozenState

if TYPE_CHECKING:
    from ufc_edge.features.contracts import FightOutcomeView

# ---------------------------------------------------------------------------
# Configuration defaults — values awaiting human specification use reasonable
# placeholders drawn from standard Elo literature (chess/sports).
# ---------------------------------------------------------------------------

_INITIAL_RATING = 1500
_HISTORY_CAP = 20

# Awaiting human-specified value; 32 is a standard starting point for
# competitive rating systems with moderate volatility.
_K_BASE = 32

# Method bonuses scale the effective K-factor for decisive finishes.
# Awaiting human-specified values; these are reasonable starting estimates
# calibrated so a KO earns ~40% more K than a decision.
_METHOD_BONUS_MAP: dict[str, float] = {
    "KO/TKO": 0.4,
    "Submission": 0.3,
    "Decision": 0.0,
}

# Awaiting human-specified value; half-life of 365 days means a fight's
# recency contribution halves after roughly a year of inactivity.
_RECENCY_WEIGHT_HALFLIFE_DAYS = 365

# Awaiting human-specified value; 0.1 per inactivity period gives moderate
# convergence toward the mean without aggressive penalty.
_INACTIVITY_DECAY_RATE = 0.1

_INACTIVITY_PERIOD_DAYS = 180
_DQ_K_MULTIPLIER = 0.1
_INJURY_STOPPAGE_K = 0

# Methods that indicate an injury stoppage — rating-neutral outcomes.
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
# Frozen record exposed via freeze()
# ---------------------------------------------------------------------------


class EloRecord(FrozenState):
    """Immutable snapshot of a single fighter's Elo state.

    Attributes:
        rating: Current Elo rating.
        peak: Highest rating achieved.
        history: Last N post-fight ratings for trajectory analysis.
        last_fight_date: Date of most recent fight processed.
        fight_count: Total fights processed for this fighter.
    """

    __slots__ = ("rating", "peak", "history", "last_fight_date", "fight_count")

    def __init__(
        self,
        rating: float,
        peak: float,
        history: tuple[float, ...],
        last_fight_date: date,
        fight_count: int,
    ) -> None:
        object.__setattr__(self, "rating", rating)
        object.__setattr__(self, "peak", peak)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "last_fight_date", last_fight_date)
        object.__setattr__(self, "fight_count", fight_count)


# ---------------------------------------------------------------------------
# Mutable internal state per fighter
# ---------------------------------------------------------------------------


class _FighterEloState:
    """Mutable working state for a single fighter within the tracker."""

    __slots__ = ("rating", "peak", "history", "last_fight_date", "fight_count")

    def __init__(self) -> None:
        self.rating: float = _INITIAL_RATING
        self.peak: float = _INITIAL_RATING
        self.history: deque[float] = deque(maxlen=_HISTORY_CAP)
        self.last_fight_date: date | None = None
        self.fight_count: int = 0


# ---------------------------------------------------------------------------
# Frozen snapshot mapping — returned by EloTracker.freeze()
# ---------------------------------------------------------------------------


class _FrozenEloSnapshot(FrozenState):
    """Frozen mapping of fighter URLs to their EloRecord snapshots.

    Implements a dict-like .get() interface for emitter access.
    """

    __slots__ = ("_records",)

    def __init__(self, records: dict[str, EloRecord]) -> None:
        object.__setattr__(self, "_records", records)

    def get(self, fighter_url: str) -> EloRecord | None:
        """Retrieve the frozen EloRecord for a fighter, or None if unseen."""
        return self._records.get(fighter_url)


# ---------------------------------------------------------------------------
# EloTracker — the StateComponent implementation
# ---------------------------------------------------------------------------


class EloTracker:
    """Elo rating state component with variable K-factor and inactivity decay.

    Implements the StateComponent protocol: update() accumulates fight outcomes,
    freeze() returns a deeply-frozen snapshot independent of further mutations.

    K-factor varies by method (finishes earn bonuses), fight recency, and
    special outcome types. Injury stoppages are rating-neutral (K=0), and DQ
    outcomes apply a 0.1 multiplier to K. Inactive fighters decay toward the
    population mean rating (1500) at a configurable rate per inactivity period.
    """

    def __init__(self) -> None:
        self._fighters: dict[str, _FighterEloState] = {}

    def update(self, fight: FightOutcomeView) -> None:
        """Apply one fight outcome to internal Elo state for both fighters."""
        fighter_a = self._ensure_fighter(fight.fighter_a_url)
        fighter_b = self._ensure_fighter(fight.fighter_b_url)

        # Apply inactivity decay before processing the fight
        self._apply_inactivity_decay(fighter_a, fight.event_date)
        self._apply_inactivity_decay(fighter_b, fight.event_date)

        # Determine effective K-factor
        effective_k = self._compute_effective_k(fight)

        if effective_k > 0:
            # Standard Elo update
            expected_a = self._expected_score(fighter_a.rating, fighter_b.rating)
            expected_b = 1.0 - expected_a

            actual_a, actual_b = self._actual_scores(fight)

            fighter_a.rating += effective_k * (actual_a - expected_a)
            fighter_b.rating += effective_k * (actual_b - expected_b)

        # Track peak ratings
        fighter_a.peak = max(fighter_a.peak, fighter_a.rating)
        fighter_b.peak = max(fighter_b.peak, fighter_b.rating)

        # Record history and metadata
        fighter_a.history.append(fighter_a.rating)
        fighter_b.history.append(fighter_b.rating)
        fighter_a.last_fight_date = fight.event_date
        fighter_b.last_fight_date = fight.event_date
        fighter_a.fight_count += 1
        fighter_b.fight_count += 1

    def freeze(self) -> _FrozenEloSnapshot:
        """Return a deeply-frozen snapshot of all fighter Elo records."""
        records: dict[str, EloRecord] = {}
        for url, state in self._fighters.items():
            records[url] = EloRecord(
                rating=state.rating,
                peak=state.peak,
                history=tuple(state.history),
                last_fight_date=state.last_fight_date,  # type: ignore[arg-type]
                fight_count=state.fight_count,
            )
        return _FrozenEloSnapshot(records)

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _ensure_fighter(self, fighter_url: str) -> _FighterEloState:
        """Get or create mutable state for a fighter."""
        if fighter_url not in self._fighters:
            self._fighters[fighter_url] = _FighterEloState()
        return self._fighters[fighter_url]

    def _is_injury_stoppage(self, method: str) -> bool:
        """Check if a method string indicates an injury/doctor stoppage."""
        method_lower = method.lower()
        for injury in _INJURY_METHODS:
            if injury.lower() in method_lower:
                return True
        # Also catch partial matches for doctor/injury stoppages
        if "injury" in method_lower and "stoppage" in method_lower:
            return True
        return "doctor" in method_lower and "stoppage" in method_lower

    def _is_dq(self, method: str) -> bool:
        """Check if the method indicates a disqualification."""
        return "dq" in method.lower() or "disqualification" in method.lower()

    def _method_category(self, method: str) -> str:
        """Map a UFC method string to a bonus category key."""
        method_upper = method.upper()
        if "KO" in method_upper or "TKO" in method_upper:
            return "KO/TKO"
        if "SUBMISSION" in method_upper or "SUB" in method_upper:
            return "Submission"
        return "Decision"

    def _compute_effective_k(self, fight: FightOutcomeView) -> float:
        """Compute the effective K-factor for this fight outcome.

        Returns 0 for injury stoppages, K*0.1 for DQ, and K*(1+method_bonus)
        for normal outcomes.
        """
        if self._is_injury_stoppage(fight.method):
            return _INJURY_STOPPAGE_K

        k = _K_BASE

        if self._is_dq(fight.method):
            return k * _DQ_K_MULTIPLIER

        # Apply method bonus for decisive finishes
        category = self._method_category(fight.method)
        bonus = _METHOD_BONUS_MAP.get(category, 0.0)
        k *= 1.0 + bonus

        return k

    def _apply_inactivity_decay(self, fighter: _FighterEloState, event_date: date) -> None:
        """Decay a fighter's rating toward 1500 based on time since last fight.

        Each full inactivity period (default 180 days) decays the distance from
        1500 by the configured decay rate. Multiple periods compound multiplicatively.
        """
        if fighter.last_fight_date is None:
            return

        days_inactive = (event_date - fighter.last_fight_date).days
        if days_inactive <= _INACTIVITY_PERIOD_DAYS:
            return

        periods = days_inactive / _INACTIVITY_PERIOD_DAYS
        # Multiplicative decay: each period retains (1 - decay_rate) of distance from mean
        retention = math.pow(1.0 - _INACTIVITY_DECAY_RATE, periods)
        distance_from_mean = fighter.rating - _INITIAL_RATING
        fighter.rating = _INITIAL_RATING + distance_from_mean * retention

    @staticmethod
    def _expected_score(rating_a: float, rating_b: float) -> float:
        """Standard Elo expected score for player A vs player B."""
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400))

    @staticmethod
    def _actual_scores(fight: FightOutcomeView) -> tuple[float, float]:
        """Determine actual scores for fighters A and B.

        Win=1.0, Loss=0.0, Draw/NC=0.5 for both.
        """
        if fight.winner_url is None:
            return 0.5, 0.5
        if fight.winner_url == fight.fighter_a_url:
            return 1.0, 0.0
        return 0.0, 1.0
