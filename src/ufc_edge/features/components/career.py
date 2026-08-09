"""Career accumulator state component.

Tracks per-fighter career statistics across UFC fights: win/loss/draw/NC counts,
finish-type breakdown, streak tracking (positive = win streak, negative = loss
streak), windowed fight date history, last fight details, debut date, and
weight-class history.

Downstream emitters read the frozen snapshot to derive activity, record, and
finishing features without recomputing from raw data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ufc_edge.features.contracts import FightOutcomeView, FrozenState

# ---------------------------------------------------------------------------
# Method classification helpers
# ---------------------------------------------------------------------------


def _is_ko(method: str) -> bool:
    """True when the fight ended by KO or TKO."""
    upper = method.upper()
    return "KO" in upper and "COULD NOT" not in upper


def _is_submission(method: str) -> bool:
    """True when the fight ended by submission."""
    return "SUBMISSION" in method.upper()


def _is_decision(method: str) -> bool:
    """True when the fight went to a judges' decision."""
    return "DECISION" in method.upper()


def _is_draw(method: str) -> bool:
    """True for draws (no winner, explicit draw ruling)."""
    return "DRAW" in method.upper()


def _is_finish(method: str) -> bool:
    """True when a fight ended by stoppage (KO/TKO or submission)."""
    return _is_ko(method) or _is_submission(method)


# ---------------------------------------------------------------------------
# Internal mutable per-fighter state
# ---------------------------------------------------------------------------


@dataclass
class _FighterCareer:
    """Mutable internal accumulator for a single fighter's career."""

    wins: int = 0
    losses: int = 0
    draws: int = 0
    no_contests: int = 0

    # Finish-type wins
    ko_wins: int = 0
    submission_wins: int = 0
    decision_wins: int = 0

    # Finish-type losses
    times_finished_by_ko: int = 0
    times_finished_by_sub: int = 0

    # Round-1 finishes (wins) and being finished in round 1 (losses)
    round_one_finishes: int = 0
    been_finished_round_one: bool = False

    # Streak: positive = consecutive wins, negative = consecutive losses
    current_streak: int = 0

    # Fight date history for windowed counting by downstream emitters
    fight_dates: list[date] = field(default_factory=list)

    # Last fight metadata
    last_fight_date: date | None = None
    last_fight_method: str | None = None

    # Debut
    debut_date: date | None = None

    # Weight classes competed in (ordered by first appearance)
    weight_classes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Frozen snapshot — immutable read-only view of one fighter's career
# ---------------------------------------------------------------------------


class CareerFighterState(FrozenState):
    """Immutable snapshot of a single fighter's career state.

    All fields are read-only. Collection fields are tuples to prevent mutation.
    """

    __slots__ = (
        "wins",
        "losses",
        "draws",
        "no_contests",
        "ko_wins",
        "submission_wins",
        "decision_wins",
        "times_finished_by_ko",
        "times_finished_by_sub",
        "round_one_finishes",
        "been_finished_round_one",
        "current_streak",
        "fight_dates",
        "last_fight_date",
        "last_fight_method",
        "debut_date",
        "total_fights",
        "weight_classes",
    )

    def __init__(
        self,
        *,
        wins: int,
        losses: int,
        draws: int,
        no_contests: int,
        ko_wins: int,
        submission_wins: int,
        decision_wins: int,
        times_finished_by_ko: int,
        times_finished_by_sub: int,
        round_one_finishes: int,
        been_finished_round_one: bool,
        current_streak: int,
        fight_dates: tuple[date, ...],
        last_fight_date: date | None,
        last_fight_method: str | None,
        debut_date: date | None,
        total_fights: int,
        weight_classes: tuple[str, ...],
    ) -> None:
        object.__setattr__(self, "wins", wins)
        object.__setattr__(self, "losses", losses)
        object.__setattr__(self, "draws", draws)
        object.__setattr__(self, "no_contests", no_contests)
        object.__setattr__(self, "ko_wins", ko_wins)
        object.__setattr__(self, "submission_wins", submission_wins)
        object.__setattr__(self, "decision_wins", decision_wins)
        object.__setattr__(self, "times_finished_by_ko", times_finished_by_ko)
        object.__setattr__(self, "times_finished_by_sub", times_finished_by_sub)
        object.__setattr__(self, "round_one_finishes", round_one_finishes)
        object.__setattr__(self, "been_finished_round_one", been_finished_round_one)
        object.__setattr__(self, "current_streak", current_streak)
        object.__setattr__(self, "fight_dates", fight_dates)
        object.__setattr__(self, "last_fight_date", last_fight_date)
        object.__setattr__(self, "last_fight_method", last_fight_method)
        object.__setattr__(self, "debut_date", debut_date)
        object.__setattr__(self, "total_fights", total_fights)
        object.__setattr__(self, "weight_classes", weight_classes)


# ---------------------------------------------------------------------------
# Frozen snapshot — mapping of all fighters' career states
# ---------------------------------------------------------------------------


class CareerSnapshot(FrozenState):
    """Immutable snapshot of all fighters' career states.

    Keyed by fighter URL. Returns None for unknown fighters.
    """

    __slots__ = ("_states",)

    def __init__(self, states: dict[str, CareerFighterState]) -> None:
        object.__setattr__(self, "_states", states)

    def get(self, fighter_url: str) -> CareerFighterState | None:
        """Retrieve a fighter's career state, or None if not yet tracked."""
        return self._states.get(fighter_url)

    def __contains__(self, fighter_url: str) -> bool:
        return fighter_url in self._states


# ---------------------------------------------------------------------------
# CareerAccumulator — the StateComponent implementation
# ---------------------------------------------------------------------------


class CareerAccumulator:
    """Accumulates per-fighter career statistics from fight outcomes.

    Tracks wins, losses, draws, no-contests, finish types, streaks, fight dates,
    last fight details, debut date, and weight class history. Freeze returns an
    immutable snapshot independent of further updates.
    """

    def __init__(self) -> None:
        self._fighters: dict[str, _FighterCareer] = {}

    def _get_or_create(self, fighter_url: str) -> _FighterCareer:
        """Get or initialize a fighter's career record."""
        if fighter_url not in self._fighters:
            self._fighters[fighter_url] = _FighterCareer()
        return self._fighters[fighter_url]

    def update(self, fight: FightOutcomeView) -> None:
        """Apply one fight outcome to both participants' career state."""
        fighter_a = self._get_or_create(fight.fighter_a_url)
        fighter_b = self._get_or_create(fight.fighter_b_url)

        # Record fight date and metadata for both fighters
        self._record_fight_metadata(fighter_a, fight)
        self._record_fight_metadata(fighter_b, fight)

        # Record weight class for both fighters
        self._record_weight_class(fighter_a, fight.weight_class)
        self._record_weight_class(fighter_b, fight.weight_class)

        # Determine outcome type and apply
        if fight.winner_url is None:
            self._apply_no_winner(fighter_a, fighter_b, fight.method)
        elif fight.winner_url == fight.fighter_a_url:
            self._apply_win_loss(fighter_a, fighter_b, fight)
        else:
            self._apply_win_loss(fighter_b, fighter_a, fight)

    def freeze(self) -> CareerSnapshot:
        """Return a deeply-frozen snapshot of all fighters' career states."""
        frozen_states: dict[str, CareerFighterState] = {}
        for url, career in self._fighters.items():
            frozen_states[url] = CareerFighterState(
                wins=career.wins,
                losses=career.losses,
                draws=career.draws,
                no_contests=career.no_contests,
                ko_wins=career.ko_wins,
                submission_wins=career.submission_wins,
                decision_wins=career.decision_wins,
                times_finished_by_ko=career.times_finished_by_ko,
                times_finished_by_sub=career.times_finished_by_sub,
                round_one_finishes=career.round_one_finishes,
                been_finished_round_one=career.been_finished_round_one,
                current_streak=career.current_streak,
                fight_dates=tuple(career.fight_dates),
                last_fight_date=career.last_fight_date,
                last_fight_method=career.last_fight_method,
                debut_date=career.debut_date,
                total_fights=len(career.fight_dates),
                weight_classes=tuple(career.weight_classes),
            )
        return CareerSnapshot(frozen_states)

    def _record_fight_metadata(
        self, fighter: _FighterCareer, fight: FightOutcomeView
    ) -> None:
        """Update fight date, last fight info, and debut for a fighter."""
        fighter.fight_dates.append(fight.event_date)
        fighter.last_fight_date = fight.event_date
        fighter.last_fight_method = fight.method
        if fighter.debut_date is None:
            fighter.debut_date = fight.event_date

    def _record_weight_class(self, fighter: _FighterCareer, weight_class: str) -> None:
        """Add weight class to history if not already present."""
        if weight_class not in fighter.weight_classes:
            fighter.weight_classes.append(weight_class)

    def _apply_no_winner(
        self, fighter_a: _FighterCareer, fighter_b: _FighterCareer, method: str
    ) -> None:
        """Handle draw or no-contest: no streak change, just record counts."""
        if _is_draw(method):
            fighter_a.draws += 1
            fighter_b.draws += 1
        else:
            fighter_a.no_contests += 1
            fighter_b.no_contests += 1

    def _apply_win_loss(
        self,
        winner: _FighterCareer,
        loser: _FighterCareer,
        fight: FightOutcomeView,
    ) -> None:
        """Apply a decisive outcome: update counts, streaks, and finish types."""
        method = fight.method
        ending_round = fight.ending_round

        # Win/loss counts
        winner.wins += 1
        loser.losses += 1

        # Streak tracking
        if winner.current_streak > 0:
            winner.current_streak += 1
        else:
            winner.current_streak = 1

        if loser.current_streak < 0:
            loser.current_streak -= 1
        else:
            loser.current_streak = -1

        # Finish-type classification for wins
        if _is_ko(method):
            winner.ko_wins += 1
            loser.times_finished_by_ko += 1
            if ending_round == 1:
                winner.round_one_finishes += 1
                loser.been_finished_round_one = True
        elif _is_submission(method):
            winner.submission_wins += 1
            loser.times_finished_by_sub += 1
            if ending_round == 1:
                winner.round_one_finishes += 1
                loser.been_finished_round_one = True
        elif _is_decision(method):
            winner.decision_wins += 1
