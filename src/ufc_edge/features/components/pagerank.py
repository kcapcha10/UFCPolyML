"""PageRank-based fighter prestige on a directed win graph.

Maintains a directed graph where each edge points from loser to winner, weighted
by finish-type bonus, exponential recency decay, and early-finish bonus. Uses
networkx's PageRank implementation with configurable damping (α=0.85). Isolated
nodes (fighters with no directed edges) receive the global minimum PageRank score.
"""

from __future__ import annotations

import math
from datetime import date

import networkx as nx

from ufc_edge.features.contracts import FightOutcomeView, FrozenState

# ─── Configuration ────────────────────────────────────────────────────────────
# Placeholder values for TODO(human) config entries.
# These will be replaced with owner-specified values from configs/graph.yaml.

DAMPING: float = 0.85
CONVERGENCE_TOLERANCE: float = 1e-6
MAX_ITERATIONS: int = 100

# Finish-type bonus: multiplier on base edge weight (1.0).
# KO/TKO and Submission get bonus weight; Decision is baseline.
FINISH_TYPE_BONUS_MAP: dict[str, float] = {
    "KO/TKO": 0.3,          # placeholder — meaningful KO bonus
    "Submission": 0.2,       # placeholder — slightly smaller than KO
    "Decision": 0.0,
}

# Exponential recency decay: weight = exp(-lambda * years_since_fight)
# Placeholder lambda — half-life of roughly 2.3 years (ln(2)/0.3 ≈ 2.3)
RECENCY_DECAY_LAMBDA: float = 0.3

# Bonus for early finishes (round 1 or 2): additive weight bonus.
# Placeholder — small bonus for finishing early.
EARLY_FINISH_BONUS: float = 0.1


# ─── Frozen snapshot ──────────────────────────────────────────────────────────


class PageRankFrozenState(FrozenState):
    """Immutable snapshot of PageRank scores for all fighters in the graph.

    Accessing `.scores` returns a copy of the internal mapping. Attribute
    assignment raises AttributeError via the FrozenState base class.
    """

    __slots__ = ("_scores",)

    def __init__(self, scores: dict[str, float]) -> None:
        # Bypass FrozenState's __setattr__ guard for initialization.
        object.__setattr__(self, "_scores", dict(scores))

    @property
    def scores(self) -> dict[str, float]:
        """Return a copy of the PageRank score mapping (fighter_url → score)."""
        return dict(self._scores)


# ─── Edge record ──────────────────────────────────────────────────────────────


class _EdgeRecord:
    """Internal record for a single directed edge in the win graph."""

    __slots__ = ("winner_url", "loser_url", "method", "ending_round", "event_date")

    def __init__(
        self,
        *,
        winner_url: str,
        loser_url: str,
        method: str,
        ending_round: int,
        event_date: date,
    ) -> None:
        self.winner_url = winner_url
        self.loser_url = loser_url
        self.method = method
        self.ending_round = ending_round
        self.event_date = event_date


# ─── PageRankGraph StateComponent ─────────────────────────────────────────────


class PageRankGraph:
    """Directed win graph with weighted PageRank for fighter prestige.

    Edges point from loser to winner. Edge weight incorporates three factors:
    1. Finish-type bonus (KO > Submission > Decision)
    2. Exponential recency decay based on years since the fight
    3. Early-finish bonus for fights ending in rounds 1–2

    On freeze(), runs networkx PageRank and returns an immutable score mapping.
    Isolated nodes (no incoming or outgoing edges) receive the global minimum.
    """

    def __init__(self) -> None:
        self._edges: list[_EdgeRecord] = []
        self._all_fighters: set[str] = set()

    def update(self, fight: FightOutcomeView) -> None:
        """Add fight outcome to the graph. Draws/NCs add nodes but no edge."""
        self._all_fighters.add(fight.fighter_a_url)
        self._all_fighters.add(fight.fighter_b_url)

        if fight.winner_url is None:
            return

        loser_url = (
            fight.fighter_b_url
            if fight.winner_url == fight.fighter_a_url
            else fight.fighter_a_url
        )

        self._edges.append(_EdgeRecord(
            winner_url=fight.winner_url,
            loser_url=loser_url,
            method=fight.method,
            ending_round=fight.ending_round,
            event_date=fight.event_date,
        ))

    def freeze(self) -> PageRankFrozenState:
        """Compute PageRank and return an immutable score snapshot."""
        scores = self._compute_pagerank()
        return PageRankFrozenState(scores)

    def _compute_pagerank(self) -> dict[str, float]:
        """Build networkx DiGraph, compute weighted PageRank, assign minimums."""
        if not self._all_fighters:
            return {}

        reference_date = self._latest_event_date()
        graph = nx.DiGraph()
        graph.add_nodes_from(self._all_fighters)

        for edge in self._edges:
            weight = self._edge_weight(edge, reference_date)
            if graph.has_edge(edge.loser_url, edge.winner_url):
                graph[edge.loser_url][edge.winner_url]["weight"] += weight
            else:
                graph.add_edge(edge.loser_url, edge.winner_url, weight=weight)

        scores = nx.pagerank(
            graph,
            alpha=DAMPING,
            tol=CONVERGENCE_TOLERANCE,
            max_iter=MAX_ITERATIONS,
            weight="weight",
        )

        # Assign global minimum to truly isolated nodes (no in/out edges)
        if scores:
            min_score = min(scores.values())
            for fighter in self._all_fighters:
                if graph.degree(fighter) == 0:
                    scores[fighter] = min_score

        return scores

    def _edge_weight(self, edge: _EdgeRecord, reference_date: date) -> float:
        """Compute composite edge weight from finish bonus, recency, and early finish."""
        base_weight = 1.0

        # Finish-type bonus
        bonus = FINISH_TYPE_BONUS_MAP.get(edge.method, 0.0)
        weight = base_weight + bonus

        # Early-finish bonus (rounds 1 or 2)
        if edge.ending_round <= 2:
            weight += EARLY_FINISH_BONUS

        # Exponential recency decay: weight * exp(-lambda * years_elapsed)
        years_elapsed = (reference_date - edge.event_date).days / 365.25
        decay = math.exp(-RECENCY_DECAY_LAMBDA * years_elapsed)
        weight *= decay

        return weight

    def _latest_event_date(self) -> date:
        """Return the most recent event date across all recorded edges."""
        if not self._edges:
            return date(2024, 1, 1)
        return max(e.event_date for e in self._edges)
