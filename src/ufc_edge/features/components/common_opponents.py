"""Common-opponent index for quality-weighted schedule comparison.

Tracks per-fighter fight history with timestamps and opponent identity. On
freeze, produces an immutable snapshot supporting intersection queries between
any two fighters within a configurable lookback window (default 3 years).
Quality weighting uses Elo and PageRank scores at freeze time; recency decay
applies exponential downweighting to older fights.

Accepts frozen Elo and PageRank state at construction time so quality scores
reflect the graph state at the moment of query rather than at fight time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from ufc_edge.features.contracts import FightOutcomeView, FrozenState

# ─── Configuration ────────────────────────────────────────────────────────────
# Placeholder values for TODO(human) config entries in configs/graph.yaml.
# These will be replaced with owner-specified values during training tuning.

# Lookback window: only fights within this many years of the query date count.
LOOKBACK_YEARS: int = 3

# Exponential recency decay: weight = exp(-lambda * years_since_fight)
# Placeholder lambda — half-life of roughly 1.7 years (ln(2)/0.4 ≈ 1.7)
RECENCY_DECAY_LAMBDA: float = 0.4

# Quality weighting blend between Elo and PageRank (must sum to 1.0).
# Placeholder — equal blend favoring neither signal.
QUALITY_WEIGHT_ELO: float = 0.5
QUALITY_WEIGHT_PAGERANK: float = 0.5


# ─── Value objects ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CommonOpponentRecord:
    """One common opponent with quality and recency metadata.

    Represents a single shared opponent between two fighters, including the
    quality score (Elo/PageRank blend) and averaged recency weight across
    both fighters' fights against this opponent.
    """

    opponent_url: str
    quality_score: float
    recency_weight: float


# ─── Internal fight record ────────────────────────────────────────────────────


class _FightEntry:
    """Internal record of one fight for a fighter, storing opponent and date."""

    __slots__ = ("opponent_url", "event_date", "won")

    def __init__(self, *, opponent_url: str, event_date: date, won: bool | None) -> None:
        self.opponent_url = opponent_url
        self.event_date = event_date
        self.won = won


# ─── Frozen snapshot ──────────────────────────────────────────────────────────


class CommonOpponentFrozenState(FrozenState):
    """Immutable snapshot of the common-opponent index.

    Supports querying the intersection of two fighters' recent opponent sets
    with quality weighting and recency decay. Attribute assignment raises
    AttributeError via the FrozenState base class.
    """

    __slots__ = ("_histories", "_elo_state", "_pagerank_state")

    def __init__(
        self,
        histories: dict[str, list[_FightEntry]],
        elo_state: object,
        pagerank_state: object,
    ) -> None:
        # Deep-copy histories so mutations on the mutable index don't propagate.
        copied: dict[str, list[_FightEntry]] = {}
        for fighter_url, entries in histories.items():
            copied[fighter_url] = [
                _FightEntry(
                    opponent_url=e.opponent_url,
                    event_date=e.event_date,
                    won=e.won,
                )
                for e in entries
            ]
        object.__setattr__(self, "_histories", copied)
        object.__setattr__(self, "_elo_state", elo_state)
        object.__setattr__(self, "_pagerank_state", pagerank_state)

    def get_common_opponents(
        self,
        fighter_a_url: str,
        fighter_b_url: str,
        *,
        as_of: date,
    ) -> list[CommonOpponentRecord]:
        """Return common opponents within the lookback window, quality-weighted.

        Finds the intersection of opponents fought by both fighters within
        `LOOKBACK_YEARS` of `as_of`. Each common opponent receives a quality
        score (Elo/PageRank blend) and recency weight (average of both fighters'
        most recent fight against that opponent, exponentially decayed).
        """
        cutoff = _lookback_cutoff(as_of)

        a_opponents = self._opponents_in_window(fighter_a_url, cutoff)
        b_opponents = self._opponents_in_window(fighter_b_url, cutoff)

        shared_urls = set(a_opponents.keys()) & set(b_opponents.keys())
        # Exclude the fighters themselves from being counted as common opponents
        shared_urls.discard(fighter_a_url)
        shared_urls.discard(fighter_b_url)

        if not shared_urls:
            return []

        results: list[CommonOpponentRecord] = []
        for opp_url in sorted(shared_urls):
            quality = self._compute_quality_score(opp_url)
            recency = self._compute_recency_weight(
                a_opponents[opp_url], b_opponents[opp_url], as_of
            )
            results.append(CommonOpponentRecord(
                opponent_url=opp_url,
                quality_score=quality,
                recency_weight=recency,
            ))

        return results

    def _opponents_in_window(
        self, fighter_url: str, cutoff: date
    ) -> dict[str, list[_FightEntry]]:
        """Return fights grouped by opponent within the lookback window."""
        entries = self._histories.get(fighter_url, [])
        grouped: dict[str, list[_FightEntry]] = {}
        for entry in entries:
            if entry.event_date >= cutoff:
                grouped.setdefault(entry.opponent_url, []).append(entry)
        return grouped

    def _compute_quality_score(self, opponent_url: str) -> float:
        """Blend Elo and PageRank scores for the given opponent."""
        elo_score = self._get_elo_rating(opponent_url)
        pagerank_score = self._get_pagerank_score(opponent_url)

        # Normalize Elo to a 0–1 scale (rough: 1000–2000 range maps to 0–1)
        elo_normalized = max(0.0, (elo_score - 1000.0) / 1000.0)

        return (
            QUALITY_WEIGHT_ELO * elo_normalized
            + QUALITY_WEIGHT_PAGERANK * pagerank_score
        )

    def _compute_recency_weight(
        self,
        a_entries: list[_FightEntry],
        b_entries: list[_FightEntry],
        as_of: date,
    ) -> float:
        """Average recency decay across both fighters' most recent fight vs this opponent."""
        a_most_recent = max(e.event_date for e in a_entries)
        b_most_recent = max(e.event_date for e in b_entries)

        decay_a = _recency_decay(a_most_recent, as_of)
        decay_b = _recency_decay(b_most_recent, as_of)

        return (decay_a + decay_b) / 2.0

    def _get_elo_rating(self, fighter_url: str) -> float:
        """Retrieve Elo rating from frozen state, defaulting to 1500."""
        elo_state = self._elo_state
        if hasattr(elo_state, "get"):
            record = elo_state.get(fighter_url)
            if record is not None and hasattr(record, "rating"):
                return record.rating
        return 1500.0

    def _get_pagerank_score(self, fighter_url: str) -> float:
        """Retrieve PageRank score from frozen state, defaulting to 0."""
        pr_state = self._pagerank_state
        if hasattr(pr_state, "scores"):
            scores = pr_state.scores
            if isinstance(scores, dict):
                return scores.get(fighter_url, 0.0)
        return 0.0


# ─── CommonOpponentIndex StateComponent ───────────────────────────────────────


class CommonOpponentIndex:
    """Per-fighter fight history index for common-opponent analysis.

    Tracks which opponents each fighter has faced and when. On freeze(),
    produces an immutable snapshot that can answer intersection queries
    with quality weighting from the injected Elo and PageRank frozen states.

    Constructor accepts frozen Elo and PageRank state snapshots so that
    quality scores reflect the rating graph at the time of the query rather
    than requiring re-computation.
    """

    def __init__(self, *, elo_state: object, pagerank_state: object) -> None:
        self._histories: dict[str, list[_FightEntry]] = {}
        self._elo_state = elo_state
        self._pagerank_state = pagerank_state

    def update(self, fight: FightOutcomeView) -> None:
        """Record both fighters' opponent relationships from this fight."""
        won_a = _determine_win(fight.winner_url, fight.fighter_a_url)
        won_b = _determine_win(fight.winner_url, fight.fighter_b_url)

        self._record_fight(
            fighter_url=fight.fighter_a_url,
            opponent_url=fight.fighter_b_url,
            event_date=fight.event_date,
            won=won_a,
        )
        self._record_fight(
            fighter_url=fight.fighter_b_url,
            opponent_url=fight.fighter_a_url,
            event_date=fight.event_date,
            won=won_b,
        )

    def freeze(self) -> CommonOpponentFrozenState:
        """Return a deeply-frozen snapshot of fight histories and quality state."""
        return CommonOpponentFrozenState(
            histories=self._histories,
            elo_state=self._elo_state,
            pagerank_state=self._pagerank_state,
        )

    def _record_fight(
        self, *, fighter_url: str, opponent_url: str, event_date: date, won: bool | None
    ) -> None:
        """Append a fight entry to the fighter's history."""
        self._histories.setdefault(fighter_url, []).append(
            _FightEntry(opponent_url=opponent_url, event_date=event_date, won=won)
        )


# ─── Module-level helpers ─────────────────────────────────────────────────────


def _lookback_cutoff(as_of: date) -> date:
    """Compute the earliest date within the lookback window."""
    year = as_of.year - LOOKBACK_YEARS
    # Handle leap-year edge case (Feb 29 → Feb 28)
    try:
        return as_of.replace(year=year)
    except ValueError:
        return as_of.replace(year=year, day=28)


def _recency_decay(fight_date: date, as_of: date) -> float:
    """Compute exponential recency decay for a fight relative to the query date."""
    years_elapsed = (as_of - fight_date).days / 365.25
    return math.exp(-RECENCY_DECAY_LAMBDA * max(0.0, years_elapsed))


def _determine_win(winner_url: str | None, fighter_url: str) -> bool | None:
    """Determine if the fighter won, lost, or drew (None for draw/NC)."""
    if winner_url is None:
        return None
    return winner_url == fighter_url
