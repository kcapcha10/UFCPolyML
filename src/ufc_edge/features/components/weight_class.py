"""Weight class tracking state component.

Tracks per-fighter weight class history: first (natural) class derived from UFC
data, current class, migration count, fight counts and wins per class. Provides
top-quartile determination for the "large for class" signal used in the weight
bully product term.

Top-quartile interface design:
    Rather than coupling this component to a database query or fixed population,
    thresholds are injected via set_class_thresholds(). The replay engine computes
    75th-percentile reach and height cutoffs per weight class from the point-in-time
    population and passes them in before each tick. This keeps the component pure
    (no I/O, no temporal leakage from future populations) while allowing the
    thresholds to evolve as more fighters enter the dataset.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ufc_edge.features.contracts import FightOutcomeView, FrozenState

# ---------------------------------------------------------------------------
# Internal mutable per-fighter state
# ---------------------------------------------------------------------------


@dataclass
class _FighterWeightHistory:
    """Mutable per-fighter weight class accumulator."""

    first_class: str = ""
    current_class: str = ""
    migration_count: int = 0
    fights_at_class: dict[str, int] = field(default_factory=dict)
    wins_at_class: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Frozen per-fighter state exposed through the snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FighterWeightSnapshot:
    """Immutable view of a single fighter's weight class history."""

    first_class: str
    current_class: str
    migration_count: int
    fights_at_class: dict[str, int]
    wins_at_class: dict[str, int]


# ---------------------------------------------------------------------------
# Frozen state (returned by freeze())
# ---------------------------------------------------------------------------


class WeightClassFrozenState(FrozenState):
    """Read-only snapshot of all fighters' weight class histories.

    Exposes per-fighter queries and top-quartile determination against
    injected thresholds.
    """

    __slots__ = ("_fighters", "_thresholds")

    def __init__(
        self,
        fighters: dict[str, FighterWeightSnapshot],
        thresholds: dict[str, dict[str, float]],
    ) -> None:
        # Bypass FrozenState's __setattr__ guard via object.__setattr__
        object.__setattr__(self, "_fighters", fighters)
        object.__setattr__(self, "_thresholds", thresholds)

    def get_fighter_state(self, fighter_url: str) -> FighterWeightSnapshot | None:
        """Return the frozen weight class snapshot for a fighter, or None if unknown."""
        return self._fighters.get(fighter_url)

    def is_large_for_class(
        self,
        fighter_url: str,
        *,
        reach_cm: float | None,
        height_cm: float | None,
    ) -> bool:
        """Determine whether a fighter is physically large for their current class.

        Both reach AND height must exceed the 75th-percentile threshold for the
        fighter's current weight class. Returns False when measurements are missing,
        fighter is unknown, or no thresholds exist for the class.
        """
        if reach_cm is None or height_cm is None:
            return False

        fighter = self._fighters.get(fighter_url)
        if fighter is None:
            return False

        class_thresholds = self._thresholds.get(fighter.current_class)
        if class_thresholds is None:
            return False

        reach_threshold = class_thresholds.get("reach_cm")
        height_threshold = class_thresholds.get("height_cm")
        if reach_threshold is None or height_threshold is None:
            return False

        return reach_cm > reach_threshold and height_cm > height_threshold


# ---------------------------------------------------------------------------
# WeightClassTracker — the StateComponent implementation
# ---------------------------------------------------------------------------


class WeightClassTracker:
    """Accumulates per-fighter weight class history across events.

    Tracks natural (first) class, current class, migration count, and per-class
    fight/win counts. Accepts injected 75th-percentile thresholds for top-quartile
    determination.
    """

    def __init__(self) -> None:
        self._fighters: dict[str, _FighterWeightHistory] = {}
        self._thresholds: dict[str, dict[str, float]] = {}

    def set_class_thresholds(self, thresholds: dict[str, dict[str, float]]) -> None:
        """Inject 75th-percentile reach/height cutoffs per weight class.

        Called by the replay engine before each tick with point-in-time thresholds.
        Keys are weight class names, values are dicts with 'reach_cm' and 'height_cm'.
        """
        self._thresholds = thresholds

    def update(self, fight: FightOutcomeView) -> None:
        """Apply one fight outcome to internal state for both fighters."""
        self._update_fighter(fight.fighter_a_url, fight)
        self._update_fighter(fight.fighter_b_url, fight)

    def freeze(self) -> WeightClassFrozenState:
        """Return a deeply-frozen snapshot of current state."""
        frozen_fighters: dict[str, FighterWeightSnapshot] = {}
        for url, history in self._fighters.items():
            frozen_fighters[url] = FighterWeightSnapshot(
                first_class=history.first_class,
                current_class=history.current_class,
                migration_count=history.migration_count,
                fights_at_class=copy.copy(history.fights_at_class),
                wins_at_class=copy.copy(history.wins_at_class),
            )
        return WeightClassFrozenState(
            fighters=frozen_fighters,
            thresholds=copy.deepcopy(self._thresholds),
        )

    def _update_fighter(self, fighter_url: str, fight: FightOutcomeView) -> None:
        """Update a single fighter's weight class history from a fight."""
        weight_class = fight.weight_class

        if fighter_url not in self._fighters:
            self._fighters[fighter_url] = _FighterWeightHistory(
                first_class=weight_class,
                current_class=weight_class,
            )
        else:
            history = self._fighters[fighter_url]
            if weight_class != history.current_class:
                history.migration_count += 1
                history.current_class = weight_class

        history = self._fighters[fighter_url]
        history.fights_at_class[weight_class] = (
            history.fights_at_class.get(weight_class, 0) + 1
        )

        if fight.winner_url == fighter_url:
            history.wins_at_class[weight_class] = (
                history.wins_at_class.get(weight_class, 0) + 1
            )
