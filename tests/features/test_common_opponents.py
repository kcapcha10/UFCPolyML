"""Tests for the CommonOpponentIndex state component.

Validates per-fighter fight history tracking, 3-year lookback windowing,
intersection computation, Elo/PageRank quality weighting, recency decay,
freeze immutability, and empty-set behaviour.
"""

from __future__ import annotations

from datetime import date

import pytest

from ufc_edge.features.components.common_opponents import (
    CommonOpponentFrozenState,
    CommonOpponentIndex,
)
from ufc_edge.features.components.elo import EloTracker
from ufc_edge.features.components.pagerank import PageRankGraph
from ufc_edge.features.contracts import FightOutcomeView, FrozenState


def _make_outcome(
    *,
    fighter_a_url: str = "http://fighter/a",
    fighter_b_url: str = "http://fighter/b",
    winner_url: str | None = "http://fighter/a",
    method: str = "Decision",
    ending_round: int = 3,
    ending_time: str = "5:00",
    event_date: date = date(2024, 6, 1),
    fight_url: str = "http://fight/1",
    event_url: str = "http://event/1",
    weight_class: str = "Lightweight",
) -> FightOutcomeView:
    """Build a FightOutcomeView with sensible defaults."""
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
        weight_class=weight_class,
        bout_order=None,
    )


def _build_elo_and_pagerank_state(
    fights: list[FightOutcomeView],
) -> tuple[object, object]:
    """Feed fights into EloTracker and PageRankGraph, return their frozen states."""
    elo = EloTracker()
    pr = PageRankGraph()
    for f in fights:
        elo.update(f)
        pr.update(f)
    return elo.freeze(), pr.freeze()


class TestCommonOpponentIndexProtocol:
    """CommonOpponentIndex satisfies expected interface contracts."""

    def test_freeze_returns_frozen_state(self) -> None:
        elo = EloTracker()
        pr = PageRankGraph()
        index = CommonOpponentIndex(elo_state=elo.freeze(), pagerank_state=pr.freeze())
        snapshot = index.freeze()
        assert isinstance(snapshot, FrozenState)

    def test_freeze_returns_common_opponent_frozen_state(self) -> None:
        elo = EloTracker()
        pr = PageRankGraph()
        index = CommonOpponentIndex(elo_state=elo.freeze(), pagerank_state=pr.freeze())
        snapshot = index.freeze()
        assert isinstance(snapshot, CommonOpponentFrozenState)


class TestSharedOpponentsWithinWindow:
    """Detects shared opponents when both fighters fought the same person recently."""

    def test_identifies_single_common_opponent(self) -> None:
        # Fighter A beats Opponent X, Fighter B beats Opponent X — X is common
        fight1 = _make_outcome(
            fighter_a_url="http://fighter/a",
            fighter_b_url="http://fighter/x",
            winner_url="http://fighter/a",
            event_date=date(2023, 3, 1),
            fight_url="http://fight/1",
            event_url="http://event/1",
        )
        fight2 = _make_outcome(
            fighter_a_url="http://fighter/b",
            fighter_b_url="http://fighter/x",
            winner_url="http://fighter/b",
            event_date=date(2023, 6, 1),
            fight_url="http://fight/2",
            event_url="http://event/2",
        )

        elo_state, pr_state = _build_elo_and_pagerank_state([fight1, fight2])
        index = CommonOpponentIndex(elo_state=elo_state, pagerank_state=pr_state)
        index.update(fight1)
        index.update(fight2)

        snapshot = index.freeze()
        common = snapshot.get_common_opponents(
            "http://fighter/a", "http://fighter/b", as_of=date(2024, 1, 1)
        )
        assert len(common) == 1
        assert common[0].opponent_url == "http://fighter/x"

    def test_identifies_multiple_common_opponents(self) -> None:
        fights = [
            _make_outcome(
                fighter_a_url="http://fighter/a",
                fighter_b_url="http://fighter/x",
                winner_url="http://fighter/a",
                event_date=date(2023, 1, 1),
                fight_url="http://fight/1",
            ),
            _make_outcome(
                fighter_a_url="http://fighter/a",
                fighter_b_url="http://fighter/y",
                winner_url="http://fighter/a",
                event_date=date(2023, 2, 1),
                fight_url="http://fight/2",
            ),
            _make_outcome(
                fighter_a_url="http://fighter/b",
                fighter_b_url="http://fighter/x",
                winner_url="http://fighter/b",
                event_date=date(2023, 3, 1),
                fight_url="http://fight/3",
            ),
            _make_outcome(
                fighter_a_url="http://fighter/b",
                fighter_b_url="http://fighter/y",
                winner_url="http://fighter/y",
                event_date=date(2023, 4, 1),
                fight_url="http://fight/4",
            ),
        ]

        elo_state, pr_state = _build_elo_and_pagerank_state(fights)
        index = CommonOpponentIndex(elo_state=elo_state, pagerank_state=pr_state)
        for f in fights:
            index.update(f)

        snapshot = index.freeze()
        common = snapshot.get_common_opponents(
            "http://fighter/a", "http://fighter/b", as_of=date(2024, 1, 1)
        )
        opponent_urls = {c.opponent_url for c in common}
        assert opponent_urls == {"http://fighter/x", "http://fighter/y"}


class TestLookbackWindowExclusion:
    """Fights older than 3 years from the query date are excluded."""

    def test_excludes_fights_outside_window(self) -> None:
        # Fight happened more than 3 years ago from the query date
        old_fight = _make_outcome(
            fighter_a_url="http://fighter/a",
            fighter_b_url="http://fighter/x",
            winner_url="http://fighter/a",
            event_date=date(2020, 1, 1),
            fight_url="http://fight/1",
        )
        # Fighter B also fought X but also long ago
        old_fight2 = _make_outcome(
            fighter_a_url="http://fighter/b",
            fighter_b_url="http://fighter/x",
            winner_url="http://fighter/b",
            event_date=date(2020, 2, 1),
            fight_url="http://fight/2",
        )

        elo_state, pr_state = _build_elo_and_pagerank_state([old_fight, old_fight2])
        index = CommonOpponentIndex(elo_state=elo_state, pagerank_state=pr_state)
        index.update(old_fight)
        index.update(old_fight2)

        snapshot = index.freeze()
        # Query as of 2024 — both fights are >3 years old
        common = snapshot.get_common_opponents(
            "http://fighter/a", "http://fighter/b", as_of=date(2024, 1, 1)
        )
        assert len(common) == 0

    def test_includes_fights_within_window(self) -> None:
        # Fight within 3 years of query date
        recent_fight = _make_outcome(
            fighter_a_url="http://fighter/a",
            fighter_b_url="http://fighter/x",
            winner_url="http://fighter/a",
            event_date=date(2023, 6, 1),
            fight_url="http://fight/1",
        )
        recent_fight2 = _make_outcome(
            fighter_a_url="http://fighter/b",
            fighter_b_url="http://fighter/x",
            winner_url="http://fighter/x",
            event_date=date(2023, 9, 1),
            fight_url="http://fight/2",
        )

        elo_state, pr_state = _build_elo_and_pagerank_state(
            [recent_fight, recent_fight2]
        )
        index = CommonOpponentIndex(elo_state=elo_state, pagerank_state=pr_state)
        index.update(recent_fight)
        index.update(recent_fight2)

        snapshot = index.freeze()
        common = snapshot.get_common_opponents(
            "http://fighter/a", "http://fighter/b", as_of=date(2024, 1, 1)
        )
        assert len(common) == 1

    def test_mixed_window_only_recent_included(self) -> None:
        # A fought X recently (within window), A fought Y long ago (outside window)
        # B fought both X and Y recently
        fights = [
            _make_outcome(
                fighter_a_url="http://fighter/a",
                fighter_b_url="http://fighter/x",
                winner_url="http://fighter/a",
                event_date=date(2023, 1, 1),
                fight_url="http://fight/1",
            ),
            _make_outcome(
                fighter_a_url="http://fighter/a",
                fighter_b_url="http://fighter/y",
                winner_url="http://fighter/a",
                event_date=date(2019, 1, 1),  # outside window
                fight_url="http://fight/2",
            ),
            _make_outcome(
                fighter_a_url="http://fighter/b",
                fighter_b_url="http://fighter/x",
                winner_url="http://fighter/b",
                event_date=date(2023, 6, 1),
                fight_url="http://fight/3",
            ),
            _make_outcome(
                fighter_a_url="http://fighter/b",
                fighter_b_url="http://fighter/y",
                winner_url="http://fighter/b",
                event_date=date(2023, 6, 1),
                fight_url="http://fight/4",
            ),
        ]

        elo_state, pr_state = _build_elo_and_pagerank_state(fights)
        index = CommonOpponentIndex(elo_state=elo_state, pagerank_state=pr_state)
        for f in fights:
            index.update(f)

        snapshot = index.freeze()
        # As of 2024-01-01: A's fight vs Y was in 2019 (>3yr), so Y not counted
        common = snapshot.get_common_opponents(
            "http://fighter/a", "http://fighter/b", as_of=date(2024, 1, 1)
        )
        opponent_urls = {c.opponent_url for c in common}
        assert "http://fighter/x" in opponent_urls
        assert "http://fighter/y" not in opponent_urls


class TestQualityWeighting:
    """Common opponent records include Elo and PageRank quality weights."""

    def test_quality_score_incorporates_elo(self) -> None:
        # Build state with fights so that Elo ratings differ
        fights = [
            _make_outcome(
                fighter_a_url="http://fighter/a",
                fighter_b_url="http://fighter/x",
                winner_url="http://fighter/a",
                event_date=date(2023, 1, 1),
                fight_url="http://fight/1",
            ),
            _make_outcome(
                fighter_a_url="http://fighter/b",
                fighter_b_url="http://fighter/x",
                winner_url="http://fighter/b",
                event_date=date(2023, 6, 1),
                fight_url="http://fight/2",
            ),
        ]

        elo_state, pr_state = _build_elo_and_pagerank_state(fights)
        index = CommonOpponentIndex(elo_state=elo_state, pagerank_state=pr_state)
        for f in fights:
            index.update(f)

        snapshot = index.freeze()
        common = snapshot.get_common_opponents(
            "http://fighter/a", "http://fighter/b", as_of=date(2024, 1, 1)
        )
        assert len(common) == 1
        # Quality score should be positive (incorporates Elo and PageRank)
        assert common[0].quality_score > 0.0

    def test_higher_rated_opponent_has_higher_quality(self) -> None:
        # X wins many fights (high Elo), Y loses many (low Elo)
        # Then A and B both fight X and Y
        setup_fights = [
            _make_outcome(
                fighter_a_url="http://fighter/x",
                fighter_b_url="http://fighter/z1",
                winner_url="http://fighter/x",
                event_date=date(2022, 1, 1),
                fight_url="http://fight/setup1",
            ),
            _make_outcome(
                fighter_a_url="http://fighter/x",
                fighter_b_url="http://fighter/z2",
                winner_url="http://fighter/x",
                event_date=date(2022, 3, 1),
                fight_url="http://fight/setup2",
            ),
            _make_outcome(
                fighter_a_url="http://fighter/y",
                fighter_b_url="http://fighter/z3",
                winner_url="http://fighter/z3",
                event_date=date(2022, 1, 1),
                fight_url="http://fight/setup3",
            ),
            _make_outcome(
                fighter_a_url="http://fighter/y",
                fighter_b_url="http://fighter/z4",
                winner_url="http://fighter/z4",
                event_date=date(2022, 3, 1),
                fight_url="http://fight/setup4",
            ),
        ]

        target_fights = [
            _make_outcome(
                fighter_a_url="http://fighter/a",
                fighter_b_url="http://fighter/x",
                winner_url="http://fighter/a",
                event_date=date(2023, 1, 1),
                fight_url="http://fight/t1",
            ),
            _make_outcome(
                fighter_a_url="http://fighter/a",
                fighter_b_url="http://fighter/y",
                winner_url="http://fighter/a",
                event_date=date(2023, 2, 1),
                fight_url="http://fight/t2",
            ),
            _make_outcome(
                fighter_a_url="http://fighter/b",
                fighter_b_url="http://fighter/x",
                winner_url="http://fighter/b",
                event_date=date(2023, 3, 1),
                fight_url="http://fight/t3",
            ),
            _make_outcome(
                fighter_a_url="http://fighter/b",
                fighter_b_url="http://fighter/y",
                winner_url="http://fighter/b",
                event_date=date(2023, 4, 1),
                fight_url="http://fight/t4",
            ),
        ]

        all_fights = setup_fights + target_fights
        elo_state, pr_state = _build_elo_and_pagerank_state(all_fights)
        index = CommonOpponentIndex(elo_state=elo_state, pagerank_state=pr_state)
        for f in target_fights:
            index.update(f)

        snapshot = index.freeze()
        common = snapshot.get_common_opponents(
            "http://fighter/a", "http://fighter/b", as_of=date(2024, 1, 1)
        )
        # X should have higher quality than Y because X has a higher Elo
        by_url = {c.opponent_url: c for c in common}
        assert by_url["http://fighter/x"].quality_score > by_url["http://fighter/y"].quality_score


class TestRecencyDecay:
    """More recent fights against common opponents receive higher weight."""

    def test_recent_fight_has_higher_recency_weight(self) -> None:
        fights = [
            _make_outcome(
                fighter_a_url="http://fighter/a",
                fighter_b_url="http://fighter/x",
                winner_url="http://fighter/a",
                event_date=date(2021, 6, 1),  # older
                fight_url="http://fight/1",
            ),
            _make_outcome(
                fighter_a_url="http://fighter/a",
                fighter_b_url="http://fighter/y",
                winner_url="http://fighter/a",
                event_date=date(2023, 6, 1),  # more recent
                fight_url="http://fight/2",
            ),
            _make_outcome(
                fighter_a_url="http://fighter/b",
                fighter_b_url="http://fighter/x",
                winner_url="http://fighter/b",
                event_date=date(2021, 7, 1),  # older
                fight_url="http://fight/3",
            ),
            _make_outcome(
                fighter_a_url="http://fighter/b",
                fighter_b_url="http://fighter/y",
                winner_url="http://fighter/b",
                event_date=date(2023, 7, 1),  # more recent
                fight_url="http://fight/4",
            ),
        ]

        elo_state, pr_state = _build_elo_and_pagerank_state(fights)
        index = CommonOpponentIndex(elo_state=elo_state, pagerank_state=pr_state)
        for f in fights:
            index.update(f)

        snapshot = index.freeze()
        common = snapshot.get_common_opponents(
            "http://fighter/a", "http://fighter/b", as_of=date(2024, 1, 1)
        )
        by_url = {c.opponent_url: c for c in common}
        # Y (fought more recently) should have higher recency weight than X
        assert by_url["http://fighter/y"].recency_weight > by_url["http://fighter/x"].recency_weight


class TestNoCommonOpponents:
    """Returns empty result when fighters share no opponents."""

    def test_no_overlap_returns_empty(self) -> None:
        fights = [
            _make_outcome(
                fighter_a_url="http://fighter/a",
                fighter_b_url="http://fighter/x",
                winner_url="http://fighter/a",
                event_date=date(2023, 1, 1),
                fight_url="http://fight/1",
            ),
            _make_outcome(
                fighter_a_url="http://fighter/b",
                fighter_b_url="http://fighter/y",
                winner_url="http://fighter/b",
                event_date=date(2023, 6, 1),
                fight_url="http://fight/2",
            ),
        ]

        elo_state, pr_state = _build_elo_and_pagerank_state(fights)
        index = CommonOpponentIndex(elo_state=elo_state, pagerank_state=pr_state)
        for f in fights:
            index.update(f)

        snapshot = index.freeze()
        common = snapshot.get_common_opponents(
            "http://fighter/a", "http://fighter/b", as_of=date(2024, 1, 1)
        )
        assert len(common) == 0

    def test_unknown_fighter_returns_empty(self) -> None:
        elo = EloTracker()
        pr = PageRankGraph()
        index = CommonOpponentIndex(elo_state=elo.freeze(), pagerank_state=pr.freeze())
        snapshot = index.freeze()
        common = snapshot.get_common_opponents(
            "http://fighter/unknown1", "http://fighter/unknown2", as_of=date(2024, 1, 1)
        )
        assert len(common) == 0


class TestFreezeImmutability:
    """Frozen snapshot is deeply immutable — mutations after freeze don't propagate."""

    def test_freeze_is_independent_of_later_updates(self) -> None:
        fight1 = _make_outcome(
            fighter_a_url="http://fighter/a",
            fighter_b_url="http://fighter/x",
            winner_url="http://fighter/a",
            event_date=date(2023, 1, 1),
            fight_url="http://fight/1",
        )
        fight2 = _make_outcome(
            fighter_a_url="http://fighter/b",
            fighter_b_url="http://fighter/x",
            winner_url="http://fighter/b",
            event_date=date(2023, 3, 1),
            fight_url="http://fight/2",
        )

        elo_state, pr_state = _build_elo_and_pagerank_state([fight1, fight2])
        index = CommonOpponentIndex(elo_state=elo_state, pagerank_state=pr_state)
        index.update(fight1)
        index.update(fight2)

        snapshot_before = index.freeze()

        # Add more fights after freeze
        fight3 = _make_outcome(
            fighter_a_url="http://fighter/a",
            fighter_b_url="http://fighter/z",
            winner_url="http://fighter/a",
            event_date=date(2023, 6, 1),
            fight_url="http://fight/3",
        )
        fight4 = _make_outcome(
            fighter_a_url="http://fighter/b",
            fighter_b_url="http://fighter/z",
            winner_url="http://fighter/b",
            event_date=date(2023, 7, 1),
            fight_url="http://fight/4",
        )
        index.update(fight3)
        index.update(fight4)

        # Original snapshot should not see fight/z as common
        common_before = snapshot_before.get_common_opponents(
            "http://fighter/a", "http://fighter/b", as_of=date(2024, 1, 1)
        )
        opponent_urls = {c.opponent_url for c in common_before}
        assert "http://fighter/z" not in opponent_urls
        assert "http://fighter/x" in opponent_urls

    def test_frozen_state_rejects_attribute_assignment(self) -> None:
        elo = EloTracker()
        pr = PageRankGraph()
        index = CommonOpponentIndex(elo_state=elo.freeze(), pagerank_state=pr.freeze())
        snapshot = index.freeze()
        with pytest.raises(AttributeError):
            snapshot.some_new_attr = "should fail"  # type: ignore[attr-defined]


class TestDrawsAndNoContests:
    """Draws and no-contests still record the opponent relationship."""

    def test_draw_records_opponent_history(self) -> None:
        fight1 = _make_outcome(
            fighter_a_url="http://fighter/a",
            fighter_b_url="http://fighter/x",
            winner_url=None,  # draw
            event_date=date(2023, 3, 1),
            fight_url="http://fight/1",
        )
        fight2 = _make_outcome(
            fighter_a_url="http://fighter/b",
            fighter_b_url="http://fighter/x",
            winner_url="http://fighter/b",
            event_date=date(2023, 6, 1),
            fight_url="http://fight/2",
        )

        elo_state, pr_state = _build_elo_and_pagerank_state([fight1, fight2])
        index = CommonOpponentIndex(elo_state=elo_state, pagerank_state=pr_state)
        index.update(fight1)
        index.update(fight2)

        snapshot = index.freeze()
        common = snapshot.get_common_opponents(
            "http://fighter/a", "http://fighter/b", as_of=date(2024, 1, 1)
        )
        assert len(common) == 1
        assert common[0].opponent_url == "http://fighter/x"
