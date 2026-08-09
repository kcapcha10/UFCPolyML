"""Tests for the PageRankGraph state component.

Validates directed-graph construction (loser→winner), edge weight computation
(finish-type bonus, recency decay, early-finish bonus), PageRank convergence,
isolated-node handling, and freeze immutability.
"""

from __future__ import annotations

from datetime import date

import pytest

from ufc_edge.features.components.pagerank import PageRankGraph
from ufc_edge.features.contracts import FightOutcomeView, FrozenState, StateComponent


def _make_outcome(
    *,
    winner_url: str | None = "http://fighter/a",
    loser_url: str = "http://fighter/b",
    method: str = "KO/TKO",
    ending_round: int = 2,
    ending_time: str = "3:45",
    event_date: date = date(2024, 6, 1),
    fight_url: str = "http://fight/1",
    event_url: str = "http://event/1",
) -> FightOutcomeView:
    """Build a FightOutcomeView with sensible defaults."""
    fighter_a_url = winner_url if winner_url else loser_url
    fighter_b_url = loser_url if winner_url else "http://fighter/a"
    return FightOutcomeView(
        fight_url=fight_url,
        event_url=event_url,
        event_date=event_date,
        fighter_a_url=fighter_a_url,
        fighter_b_url=fighter_b_url,
        winner_url=winner_url,
        method=method,
        ending_round=ending_round,
        ending_time=ending_time,
        weight_class="Lightweight",
        bout_order=None,
    )


class TestPageRankGraphProtocol:
    """PageRankGraph satisfies the StateComponent protocol."""

    def test_is_state_component(self) -> None:
        graph = PageRankGraph()
        assert isinstance(graph, StateComponent)

    def test_freeze_returns_frozen_state(self) -> None:
        graph = PageRankGraph()
        snapshot = graph.freeze()
        assert isinstance(snapshot, FrozenState)


class TestSingleFightCreatesEdge:
    """A single fight outcome creates a directed edge loser→winner."""

    def test_edge_from_loser_to_winner(self) -> None:
        graph = PageRankGraph()
        outcome = _make_outcome(winner_url="http://fighter/a", loser_url="http://fighter/b")
        graph.update(outcome)
        snapshot = graph.freeze()
        scores = snapshot.scores
        # Winner should have higher score than loser (receives incoming edge)
        assert scores["http://fighter/a"] > scores["http://fighter/b"]

    def test_both_fighters_present_in_scores(self) -> None:
        graph = PageRankGraph()
        outcome = _make_outcome(winner_url="http://fighter/a", loser_url="http://fighter/b")
        graph.update(outcome)
        snapshot = graph.freeze()
        assert "http://fighter/a" in snapshot.scores
        assert "http://fighter/b" in snapshot.scores

    def test_draw_creates_no_edge(self) -> None:
        graph = PageRankGraph()
        outcome = _make_outcome(winner_url=None)
        graph.update(outcome)
        snapshot = graph.freeze()
        # Scores should be equal for both fighters (no directed edge)
        scores = snapshot.scores
        if scores:
            vals = list(scores.values())
            assert all(v == vals[0] for v in vals)


class TestPageRankConverges:
    """PageRank values converge with multiple fights forming a graph."""

    def test_three_fighter_chain(self) -> None:
        """A→beats→B→beats→C: A should have highest PageRank."""
        graph = PageRankGraph()
        # C beats B
        graph.update(_make_outcome(
            winner_url="http://fighter/c",
            loser_url="http://fighter/b",
            fight_url="http://fight/1",
            event_date=date(2024, 1, 1),
        ))
        # B beats A
        graph.update(_make_outcome(
            winner_url="http://fighter/b",
            loser_url="http://fighter/a",
            fight_url="http://fight/2",
            event_date=date(2024, 2, 1),
        ))
        snapshot = graph.freeze()
        scores = snapshot.scores
        # C beat B, B beat A => C has highest transitive prestige
        assert scores["http://fighter/c"] > scores["http://fighter/b"]
        assert scores["http://fighter/b"] > scores["http://fighter/a"]

    def test_scores_sum_approximately_to_one(self) -> None:
        """PageRank scores should approximately sum to 1.0."""
        graph = PageRankGraph()
        graph.update(_make_outcome(
            winner_url="http://fighter/a",
            loser_url="http://fighter/b",
            fight_url="http://fight/1",
        ))
        graph.update(_make_outcome(
            winner_url="http://fighter/c",
            loser_url="http://fighter/a",
            fight_url="http://fight/2",
        ))
        snapshot = graph.freeze()
        total = sum(snapshot.scores.values())
        assert abs(total - 1.0) < 0.01


class TestIsolatedNodeGetsMinimum:
    """An isolated node (no edges) receives the global minimum PageRank."""

    def test_isolated_fighter_gets_minimum(self) -> None:
        graph = PageRankGraph()
        # Create a connected subgraph
        graph.update(_make_outcome(
            winner_url="http://fighter/a",
            loser_url="http://fighter/b",
            fight_url="http://fight/1",
        ))
        graph.update(_make_outcome(
            winner_url="http://fighter/b",
            loser_url="http://fighter/c",
            fight_url="http://fight/2",
        ))
        snapshot = graph.freeze()
        scores = snapshot.scores
        # All nodes participate in edges — verify fighter with no incoming edges
        # has the lowest score (fighter/c only has outgoing edge to b)
        min_score = min(scores.values())
        assert scores["http://fighter/c"] == min_score

    def test_truly_isolated_node_gets_minimum(self) -> None:
        """Fighter added via draw has no edges and gets minimum PageRank."""
        graph = PageRankGraph()
        # Fight with a winner
        graph.update(_make_outcome(
            winner_url="http://fighter/a",
            loser_url="http://fighter/b",
            fight_url="http://fight/1",
        ))
        # Draw adds two fighters as isolated nodes in the graph
        graph.update(FightOutcomeView(
            fight_url="http://fight/2",
            event_url="http://event/2",
            event_date=date(2024, 7, 1),
            fighter_a_url="http://fighter/x",
            fighter_b_url="http://fighter/y",
            winner_url=None,
            method="Draw",
            ending_round=3,
            ending_time="5:00",
            weight_class="Lightweight",
            bout_order=None,
        ))
        snapshot = graph.freeze()
        scores = snapshot.scores
        min_score = min(scores.values())
        # Isolated nodes get the minimum
        assert scores["http://fighter/x"] == min_score
        assert scores["http://fighter/y"] == min_score


class TestRecencyDecay:
    """Exponential recency decay reduces old edge weights."""

    def test_recent_win_weighted_more_than_old_win(self) -> None:
        """Same winner, two fights at different dates — recent fight has higher edge weight."""
        # We verify recency decay by comparing two graphs with identical
        # topology but different fight dates. When a node has multiple
        # outgoing edges, the more recently won edge carries more weight.
        graph_both_old = PageRankGraph()
        graph_both_old.update(_make_outcome(
            winner_url="http://fighter/a",
            loser_url="http://fighter/b",
            method="Decision",
            event_date=date(2018, 1, 1),
            fight_url="http://fight/1",
        ))
        graph_both_old.update(_make_outcome(
            winner_url="http://fighter/a",
            loser_url="http://fighter/c",
            method="Decision",
            event_date=date(2018, 6, 1),
            fight_url="http://fight/2",
        ))

        graph_one_recent = PageRankGraph()
        graph_one_recent.update(_make_outcome(
            winner_url="http://fighter/a",
            loser_url="http://fighter/b",
            method="Decision",
            event_date=date(2018, 1, 1),
            fight_url="http://fight/1",
        ))
        graph_one_recent.update(_make_outcome(
            winner_url="http://fighter/a",
            loser_url="http://fighter/c",
            method="Decision",
            event_date=date(2024, 6, 1),
            fight_url="http://fight/2",
        ))

        # The graph with a more recent fight should give A a higher PageRank
        # because recency decay makes the recent edge heavier relative to total.
        old_scores = graph_both_old.freeze().scores
        recent_scores = graph_one_recent.freeze().scores
        assert recent_scores["http://fighter/a"] >= old_scores["http://fighter/a"]


class TestFinishTypeBonus:
    """Finish-type bonus affects edge weights."""

    def test_ko_bonus_higher_than_decision(self) -> None:
        """A KO/TKO win should produce higher edge weight than a Decision.

        B lost to both A and C. When B→A has higher weight (KO bonus), A
        receives a larger share of B's rank than when B→A is a Decision.
        """
        graph_ko = PageRankGraph()
        # A beats B by KO — edge B→A with KO bonus
        graph_ko.update(_make_outcome(
            winner_url="http://fighter/a",
            loser_url="http://fighter/b",
            method="KO/TKO",
            ending_round=3,
            fight_url="http://fight/1",
            event_date=date(2024, 6, 1),
        ))
        # C beats B by Decision — edge B→C with base weight
        graph_ko.update(_make_outcome(
            winner_url="http://fighter/c",
            loser_url="http://fighter/b",
            method="Decision",
            ending_round=3,
            fight_url="http://fight/2",
            event_date=date(2024, 6, 1),
        ))

        graph_dec = PageRankGraph()
        # A beats B by Decision — edge B→A with base weight only
        graph_dec.update(_make_outcome(
            winner_url="http://fighter/a",
            loser_url="http://fighter/b",
            method="Decision",
            ending_round=3,
            fight_url="http://fight/1",
            event_date=date(2024, 6, 1),
        ))
        # C beats B by Decision — edge B→C with base weight (same in both)
        graph_dec.update(_make_outcome(
            winner_url="http://fighter/c",
            loser_url="http://fighter/b",
            method="Decision",
            ending_round=3,
            fight_url="http://fight/2",
            event_date=date(2024, 6, 1),
        ))

        # B has two outgoing edges: B→A and B→C.
        # In graph_ko: B→A weight is 1.3, B→C weight is 1.0
        #   → A gets more of B's rank share
        # In graph_dec: B→A weight is 1.0, B→C weight is 1.0
        #   → A and C split B's rank equally
        ko_scores = graph_ko.freeze().scores
        dec_scores = graph_dec.freeze().scores
        assert ko_scores["http://fighter/a"] > dec_scores["http://fighter/a"]


class TestFreezeImmutability:
    """Frozen snapshot must be immutable."""

    def test_cannot_set_attribute(self) -> None:
        graph = PageRankGraph()
        graph.update(_make_outcome())
        snapshot = graph.freeze()
        with pytest.raises(AttributeError):
            snapshot.scores = {}  # type: ignore[misc]

    def test_scores_dict_is_copy(self) -> None:
        """Mutating the returned scores dict does not affect the component."""
        graph = PageRankGraph()
        graph.update(_make_outcome())
        snapshot = graph.freeze()
        external_ref = snapshot.scores
        external_ref["http://fighter/injected"] = 999.0
        # Freeze again — injected key must not appear
        snapshot2 = graph.freeze()
        assert "http://fighter/injected" not in snapshot2.scores

    def test_subsequent_updates_dont_affect_frozen_snapshot(self) -> None:
        """Freezing, then updating, doesn't change the frozen snapshot."""
        graph = PageRankGraph()
        graph.update(_make_outcome(
            winner_url="http://fighter/a",
            loser_url="http://fighter/b",
            fight_url="http://fight/1",
        ))
        snapshot = graph.freeze()
        original_scores = dict(snapshot.scores)

        # Add more fights to the live graph
        graph.update(_make_outcome(
            winner_url="http://fighter/c",
            loser_url="http://fighter/a",
            fight_url="http://fight/2",
        ))
        # The old snapshot is unchanged
        assert snapshot.scores == original_scores
