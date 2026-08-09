"""Experience and championship context emitter and accumulator.

Tracks per-fighter title fight participation, championship wins, main event
appearances, and five-round fight history. The ExperienceAccumulator is a
StateComponent whose frozen snapshot is consumed by the stateless
ExperienceEmitter to derive six features: title_fight_experience,
has_been_champion, days_as_champion, main_event_experience,
five_round_experience, and five_round_win_pct.

Title fights are detected from weight_class containing "Title". Five-round
fights are detected from the time_format field. Main events are identified
as fights scheduled for five rounds (title bouts and five-round non-title
headliners).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ufc_edge.features.contracts import EmitContext, FightOutcomeView, FrozenState

# ---------------------------------------------------------------------------
# Internal mutable per-fighter experience state
# ---------------------------------------------------------------------------


@dataclass
class _FighterExperience:
    """Mutable internal accumulator for a single fighter's experience history."""

    title_fights: int = 0
    has_been_champion: bool = False
    first_title_win_date: date | None = None
    main_event_fights: int = 0
    five_round_fights: int = 0
    five_round_wins: int = 0


# ---------------------------------------------------------------------------
# Frozen snapshot — immutable read-only view of one fighter's experience
# ---------------------------------------------------------------------------


class ExperienceFighterState(FrozenState):
    """Immutable snapshot of a single fighter's experience state.

    All fields are read-only via FrozenState's __setattr__ guard.
    """

    __slots__ = (
        "title_fights",
        "has_been_champion",
        "first_title_win_date",
        "main_event_fights",
        "five_round_fights",
        "five_round_wins",
    )

    def __init__(
        self,
        *,
        title_fights: int,
        has_been_champion: bool,
        first_title_win_date: date | None,
        main_event_fights: int,
        five_round_fights: int,
        five_round_wins: int,
    ) -> None:
        object.__setattr__(self, "title_fights", title_fights)
        object.__setattr__(self, "has_been_champion", has_been_champion)
        object.__setattr__(self, "first_title_win_date", first_title_win_date)
        object.__setattr__(self, "main_event_fights", main_event_fights)
        object.__setattr__(self, "five_round_fights", five_round_fights)
        object.__setattr__(self, "five_round_wins", five_round_wins)


# ---------------------------------------------------------------------------
# Frozen snapshot — mapping of all fighters' experience states
# ---------------------------------------------------------------------------


class ExperienceSnapshot(FrozenState):
    """Immutable snapshot of all fighters' experience states.

    Keyed by fighter URL. Returns None for unknown fighters.
    """

    __slots__ = ("_states",)

    def __init__(self, states: dict[str, ExperienceFighterState]) -> None:
        object.__setattr__(self, "_states", states)

    def get(self, fighter_url: str) -> ExperienceFighterState | None:
        """Retrieve a fighter's experience state, or None if not yet tracked."""
        return self._states.get(fighter_url)

    def __contains__(self, fighter_url: str) -> bool:
        return fighter_url in self._states


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _is_title_fight(weight_class: str) -> bool:
    """True when the weight_class indicates a title bout."""
    return "title" in weight_class.lower()


def _is_five_round(time_format: str) -> bool:
    """True when the time_format indicates a five-round fight."""
    return time_format.strip().startswith("5")


# ---------------------------------------------------------------------------
# ExperienceAccumulator — StateComponent implementation
# ---------------------------------------------------------------------------


class ExperienceAccumulator:
    """Accumulates per-fighter experience statistics from fight outcomes.

    Tracks title fight participation, championship status, main event
    appearances, and five-round fight history. Uses update_with_context()
    to receive time_format alongside the standard FightOutcomeView; the
    protocol-mandated update() is a no-op since time_format is not available
    in FightOutcomeView.
    """

    def __init__(self) -> None:
        self._fighters: dict[str, _FighterExperience] = {}

    def _get_or_create(self, fighter_url: str) -> _FighterExperience:
        """Get or initialize a fighter's experience record."""
        if fighter_url not in self._fighters:
            self._fighters[fighter_url] = _FighterExperience()
        return self._fighters[fighter_url]

    def update(self, fight: FightOutcomeView) -> None:
        """No-op: use update_with_context which accepts time_format.

        The base StateComponent protocol's update() only receives outcome data.
        ExperienceAccumulator requires time_format to detect five-round fights,
        so the replay engine calls update_with_context() directly.
        """

    def update_with_context(
        self, fight: FightOutcomeView, *, time_format: str
    ) -> None:
        """Record experience data for both fighters from a fight outcome.

        Params:
            fight: standard outcome view with weight_class and winner.
            time_format: round format string (e.g. "5 Rnd (5-5-5-5-5)").
        """
        fighter_a = self._get_or_create(fight.fighter_a_url)
        fighter_b = self._get_or_create(fight.fighter_b_url)

        is_title = _is_title_fight(fight.weight_class)
        is_five = _is_five_round(time_format)

        # Title fight experience for both participants
        if is_title:
            fighter_a.title_fights += 1
            fighter_b.title_fights += 1

        # Championship: only the winner of a title fight becomes champion
        if is_title and fight.winner_url is not None:
            winner = self._get_or_create(fight.winner_url)
            if not winner.has_been_champion:
                winner.has_been_champion = True
                winner.first_title_win_date = fight.event_date

        # Main event detection: five-round fights are main events or co-mains
        if is_five:
            fighter_a.main_event_fights += 1
            fighter_b.main_event_fights += 1

        # Five-round tracking
        if is_five:
            fighter_a.five_round_fights += 1
            fighter_b.five_round_fights += 1
            if fight.winner_url == fight.fighter_a_url:
                fighter_a.five_round_wins += 1
            elif fight.winner_url == fight.fighter_b_url:
                fighter_b.five_round_wins += 1

    def freeze(self) -> ExperienceSnapshot:
        """Return a deeply-frozen snapshot of all fighters' experience states."""
        frozen_states: dict[str, ExperienceFighterState] = {}
        for url, exp in self._fighters.items():
            frozen_states[url] = ExperienceFighterState(
                title_fights=exp.title_fights,
                has_been_champion=exp.has_been_champion,
                first_title_win_date=exp.first_title_win_date,
                main_event_fights=exp.main_event_fights,
                five_round_fights=exp.five_round_fights,
                five_round_wins=exp.five_round_wins,
            )
        return ExperienceSnapshot(frozen_states)


# ---------------------------------------------------------------------------
# ExperienceEmitter — stateless feature emitter
# ---------------------------------------------------------------------------


class ExperienceEmitter:
    """Stateless emitter for experience and championship features.

    Reads ExperienceSnapshot from EmitContext components mapping. Returns
    zero-valued features for unknown fighters and None for ratios with
    zero denominator.
    """

    name: str = "experience"

    def emit(self, context: EmitContext) -> dict[str, float | str | None]:
        """Emit experience features for the focal fighter."""
        snapshot: ExperienceSnapshot = context.components["experience"]  # type: ignore[assignment]
        state = snapshot.get(context.fighter_url)

        if state is None:
            return self._zero_result()

        title_exp = float(state.title_fights)
        champion = 1.0 if state.has_been_champion else 0.0
        days_champ = self._days_as_champion(state, context.event_date)
        main_event = float(state.main_event_fights)
        five_round = float(state.five_round_fights)
        five_round_pct = self._five_round_win_pct(state)

        return {
            "title_fight_experience": title_exp,
            "has_been_champion": champion,
            "days_as_champion": days_champ,
            "main_event_experience": main_event,
            "five_round_experience": five_round,
            "five_round_win_pct": five_round_pct,
        }

    # ------------------------------------------------------------------
    # Private computation methods
    # ------------------------------------------------------------------

    def _zero_result(self) -> dict[str, float | str | None]:
        """Feature values for a fighter with no tracked experience."""
        return {
            "title_fight_experience": 0.0,
            "has_been_champion": 0.0,
            "days_as_champion": 0.0,
            "main_event_experience": 0.0,
            "five_round_experience": 0.0,
            "five_round_win_pct": None,
        }

    def _days_as_champion(
        self, state: ExperienceFighterState, event_date: date
    ) -> float:
        """Days since first title win, or 0.0 if never been champion."""
        if not state.has_been_champion or state.first_title_win_date is None:
            return 0.0
        return float((event_date - state.first_title_win_date).days)

    def _five_round_win_pct(self, state: ExperienceFighterState) -> float | None:
        """Win percentage in five-round fights, or None if no five-round bouts."""
        if state.five_round_fights == 0:
            return None
        return state.five_round_wins / state.five_round_fights
