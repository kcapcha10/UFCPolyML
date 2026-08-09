"""Rematch detection emitter and supporting state component.

Derives rematch features from a dedicated RematchAccumulator that tracks
per-matchup fight history including method, result, and timing. The emitter
identifies whether the focal fighter has previously faced the current opponent,
and if so, reports details of their first meeting: result, method, whether it
was competitive (decision-only), and the ending round as a dominance proxy.

The RematchAccumulator is a lightweight StateComponent that stores only the
fields needed for rematch analysis (opponent pairing, method, winner, round).
It complements the existing CareerAccumulator which tracks aggregate career
stats but not per-opponent history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ufc_edge.features.contracts import EmitContext, FightOutcomeView, FrozenState

# ---------------------------------------------------------------------------
# Matchup record — one historical meeting between two fighters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MatchupRecord:
    """One historical fight between a specific pair of fighters.

    Stores the minimum fields needed for rematch analysis: who fought, who won,
    when, the method of victory, and the ending round.
    """

    fight_url: str
    event_date: date
    fighter_a_url: str
    fighter_b_url: str
    winner_url: str | None
    method: str
    ending_round: int


# ---------------------------------------------------------------------------
# Frozen snapshot
# ---------------------------------------------------------------------------


class RematchFrozenState(FrozenState):
    """Immutable snapshot of all tracked matchup histories.

    Stores per-matchup fight records keyed by a canonical pair key (alphabetically
    sorted fighter URLs) to ensure bidirectional lookup returns the same history.
    Also stores per-fighter fight counts for fights_since calculation.
    """

    __slots__ = ("_matchups", "_fight_counts")

    def __init__(
        self,
        matchups: dict[tuple[str, str], list[MatchupRecord]],
        fight_counts: dict[str, int],
    ) -> None:
        # Deep-copy to decouple from the mutable accumulator
        copied: dict[tuple[str, str], list[MatchupRecord]] = {}
        for key, records in matchups.items():
            copied[key] = list(records)
        object.__setattr__(self, "_matchups", copied)
        object.__setattr__(self, "_fight_counts", dict(fight_counts))

    def get_matchup_history(
        self, fighter_url: str, opponent_url: str
    ) -> list[MatchupRecord]:
        """Return chronologically ordered fights between this pair."""
        key = _canonical_key(fighter_url, opponent_url)
        return self._matchups.get(key, [])

    def get_fight_count(self, fighter_url: str) -> int:
        """Total fights tracked for a fighter (all opponents)."""
        return self._fight_counts.get(fighter_url, 0)

    def get_fight_count_at_date(
        self, fighter_url: str, after_date: date
    ) -> int:
        """Count of fighter's fights strictly after the given date.

        Scans all matchups involving this fighter and counts entries with
        event_date strictly after the specified date.
        """
        count = 0
        for key, records in self._matchups.items():
            if fighter_url not in key:
                continue
            for record in records:
                if record.event_date > after_date:
                    count += 1
        return count


# ---------------------------------------------------------------------------
# RematchAccumulator — StateComponent
# ---------------------------------------------------------------------------


class RematchAccumulator:
    """Tracks per-matchup fight history for rematch detection.

    Uses a canonical key (alphabetically sorted fighter URLs) so that the same
    history is accessible regardless of which fighter is queried first. Also
    maintains per-fighter fight counts for the fights_since_first_meeting metric.
    """

    def __init__(self) -> None:
        self._matchups: dict[tuple[str, str], list[MatchupRecord]] = {}
        self._fight_counts: dict[str, int] = {}

    def update(self, fight: FightOutcomeView) -> None:
        """Record a fight between two fighters."""
        key = _canonical_key(fight.fighter_a_url, fight.fighter_b_url)
        record = MatchupRecord(
            fight_url=fight.fight_url,
            event_date=fight.event_date,
            fighter_a_url=fight.fighter_a_url,
            fighter_b_url=fight.fighter_b_url,
            winner_url=fight.winner_url,
            method=fight.method,
            ending_round=fight.ending_round,
        )
        self._matchups.setdefault(key, []).append(record)

        # Increment fight counts for both participants
        self._fight_counts[fight.fighter_a_url] = (
            self._fight_counts.get(fight.fighter_a_url, 0) + 1
        )
        self._fight_counts[fight.fighter_b_url] = (
            self._fight_counts.get(fight.fighter_b_url, 0) + 1
        )

    def freeze(self) -> RematchFrozenState:
        """Return a deeply-frozen snapshot of matchup histories."""
        return RematchFrozenState(
            matchups=self._matchups,
            fight_counts=self._fight_counts,
        )


# ---------------------------------------------------------------------------
# RematchEmitter — stateless feature emitter
# ---------------------------------------------------------------------------


class RematchEmitter:
    """Emits rematch features for the focal fighter vs the current opponent.

    Reads RematchFrozenState from EmitContext.components["rematch"]. If the
    fighters have never met, returns is_rematch=0.0 with all detail fields None.
    If they have met, returns details of their first encounter.
    """

    name: str = "rematch"

    def emit(self, context: EmitContext) -> dict[str, float | str | None]:
        """Emit rematch features for the focal fighter."""
        snapshot: RematchFrozenState = context.components["rematch"]  # type: ignore[assignment]
        history = snapshot.get_matchup_history(
            context.fighter_url, context.opponent_url
        )

        if not history:
            return _no_rematch_result()

        first_meeting = history[0]
        fights_since = _compute_fights_since(
            snapshot, context.fighter_url, first_meeting.event_date
        )
        result = _determine_result(first_meeting, context.fighter_url)
        competitive = _is_competitive(first_meeting.method)
        score_delta = float(first_meeting.ending_round)

        return {
            "is_rematch": 1.0,
            "fights_since_first_meeting": float(fights_since),
            "result_of_first_meeting": result,
            "first_meeting_method": first_meeting.method,
            "first_meeting_competitive": 1.0 if competitive else 0.0,
            "first_meeting_score_delta": score_delta,
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _no_rematch_result() -> dict[str, float | str | None]:
    """Feature values when fighters have not previously met."""
    return {
        "is_rematch": 0.0,
        "fights_since_first_meeting": None,
        "result_of_first_meeting": None,
        "first_meeting_method": None,
        "first_meeting_competitive": None,
        "first_meeting_score_delta": None,
    }


def _canonical_key(fighter_a: str, fighter_b: str) -> tuple[str, str]:
    """Produce a canonical matchup key by sorting fighter URLs alphabetically."""
    if fighter_a <= fighter_b:
        return (fighter_a, fighter_b)
    return (fighter_b, fighter_a)


def _compute_fights_since(
    snapshot: RematchFrozenState, fighter_url: str, first_meeting_date: date
) -> int:
    """Count the focal fighter's fights strictly after the first meeting date."""
    return snapshot.get_fight_count_at_date(fighter_url, first_meeting_date)


def _determine_result(record: MatchupRecord, focal_fighter_url: str) -> float:
    """Determine the focal fighter's result: 1.0=win, 0.0=loss, 0.5=draw/NC."""
    if record.winner_url is None:
        return 0.5
    if record.winner_url == focal_fighter_url:
        return 1.0
    return 0.0


def _is_competitive(method: str) -> bool:
    """True when the fight went to a judges' decision (any type).

    A fight is considered competitive if it was decided by the judges rather
    than stopped by the referee, doctor, or corner. This includes unanimous,
    split, and majority decisions.
    """
    return "DECISION" in method.upper()
