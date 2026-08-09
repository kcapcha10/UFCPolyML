"""Tests for the graph feature emitter.

Validates that GraphEmitter correctly reads frozen Elo, Glicko-2, PageRank,
and CommonOpponent state, emits the specified features, computes trajectory
slope via linear regression, and returns None when fewer than 5 history
entries exist for trajectory.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from ufc_edge.features.components.common_opponents import (
    CommonOpponentFrozenState,
    _FightEntry,
)
from ufc_edge.features.components.elo import EloRecord, _FrozenEloSnapshot
from ufc_edge.features.components.glicko2 import Glicko2FrozenState, Glicko2Record
from ufc_edge.features.components.pagerank import PageRankFrozenState
from ufc_edge.features.contracts import EmitContext, FighterProfile
from ufc_edge.features.emitters.graph import GraphEmitter

FIGHTER_A = "http://ufcstats.com/fighter/a"
FIGHTER_B = "http://ufcstats.com/fighter/b"
COMMON_OPP = "http://ufcstats.com/fighter/c"


def _make_elo_snapshot(
    records: dict[str, EloRecord],
) -> _FrozenEloSnapshot:
    """Build a frozen Elo snapshot from a dict of fighter_url → EloRecord."""
    return _FrozenEloSnapshot(records)


def _make_glicko2_snapshot(
    records: dict[str, Glicko2Record],
) -> Glicko2FrozenState:
    """Build a frozen Glicko-2 snapshot from a dict of fighter_url → record."""
    return Glicko2FrozenState(
        records=records,
        initial_mu=1500.0,
        initial_rd=350.0,
        initial_sigma=0.06,
    )


def _make_pagerank_snapshot(scores: dict[str, float]) -> PageRankFrozenState:
    """Build a frozen PageRank snapshot from a scores dict."""
    return PageRankFrozenState(scores)


def _make_common_opp_snapshot(
    histories: dict[str, list[_FightEntry]],
    elo_state: object,
    pagerank_state: object,
) -> CommonOpponentFrozenState:
    """Build a frozen CommonOpponent snapshot from fight histories."""
    return CommonOpponentFrozenState(
        histories=histories,
        elo_state=elo_state,
        pagerank_state=pagerank_state,
    )


def _make_context(
    *,
    elo_snapshot: _FrozenEloSnapshot | None = None,
    glicko2_snapshot: Glicko2FrozenState | None = None,
    pagerank_snapshot: PageRankFrozenState | None = None,
    common_opp_snapshot: CommonOpponentFrozenState | None = None,
    event_date: date = date(2024, 6, 15),
) -> EmitContext:
    """Build a minimal EmitContext with graph component snapshots."""
    fighter_profile = FighterProfile(fighter_url=FIGHTER_A)
    opponent_profile = FighterProfile(fighter_url=FIGHTER_B)

    components: dict[str, object] = {}
    if elo_snapshot is not None:
        components["elo"] = elo_snapshot
    if glicko2_snapshot is not None:
        components["glicko2"] = glicko2_snapshot
    if pagerank_snapshot is not None:
        components["pagerank"] = pagerank_snapshot
    if common_opp_snapshot is not None:
        components["common_opponents"] = common_opp_snapshot

    return EmitContext(
        fighter_url=FIGHTER_A,
        fighter_profile=fighter_profile,
        opponent_url=FIGHTER_B,
        opponent_profile=opponent_profile,
        event_date=event_date,
        event_url="http://ufcstats.com/event/e1",
        weight_class="Lightweight",
        fight_url="http://ufcstats.com/fight/f1",
        bout_order=None,
        components=components,
    )


class TestGraphEmitterProtocol:
    """GraphEmitter satisfies the FeatureEmitter protocol."""

    def test_has_name_attribute(self) -> None:
        emitter = GraphEmitter()
        assert isinstance(emitter.name, str)
        assert emitter.name == "graph"

    def test_emit_returns_dict(self) -> None:
        emitter = GraphEmitter()
        ctx = _make_context()
        result = emitter.emit(ctx)
        assert isinstance(result, dict)


class TestEloFeatures:
    """Elo features are read from frozen component state."""

    def test_emits_elo_rating_from_snapshot(self) -> None:
        elo = _make_elo_snapshot({
            FIGHTER_A: EloRecord(
                rating=1650.0,
                peak=1700.0,
                history=(1500.0, 1550.0, 1580.0, 1620.0, 1650.0),
                last_fight_date=date(2024, 3, 1),
                fight_count=5,
            ),
        })
        ctx = _make_context(elo_snapshot=elo)
        result = GraphEmitter().emit(ctx)
        assert result["elo_rating"] == 1650.0

    def test_emits_elo_peak(self) -> None:
        elo = _make_elo_snapshot({
            FIGHTER_A: EloRecord(
                rating=1600.0,
                peak=1750.0,
                history=(1600.0,),
                last_fight_date=date(2024, 3, 1),
                fight_count=1,
            ),
        })
        ctx = _make_context(elo_snapshot=elo)
        result = GraphEmitter().emit(ctx)
        assert result["elo_peak"] == 1750.0

    def test_emits_elo_current_vs_peak(self) -> None:
        elo = _make_elo_snapshot({
            FIGHTER_A: EloRecord(
                rating=1600.0,
                peak=1800.0,
                history=(1600.0,),
                last_fight_date=date(2024, 3, 1),
                fight_count=1,
            ),
        })
        ctx = _make_context(elo_snapshot=elo)
        result = GraphEmitter().emit(ctx)
        # current_vs_peak = rating / peak
        expected = 1600.0 / 1800.0
        assert abs(result["elo_current_vs_peak"] - expected) < 1e-9

    def test_returns_none_for_unknown_fighter(self) -> None:
        elo = _make_elo_snapshot({})
        ctx = _make_context(elo_snapshot=elo)
        result = GraphEmitter().emit(ctx)
        assert result["elo_rating"] is None
        assert result["elo_peak"] is None
        assert result["elo_current_vs_peak"] is None
        assert result["elo_trajectory_last5"] is None


class TestEloTrajectory:
    """Trajectory slope uses linear regression over the last 5 history entries."""

    def test_trajectory_with_exactly_5_entries(self) -> None:
        # Ratings: [1500, 1520, 1540, 1560, 1580] — slope should be 20.0
        history = (1500.0, 1520.0, 1540.0, 1560.0, 1580.0)
        elo = _make_elo_snapshot({
            FIGHTER_A: EloRecord(
                rating=1580.0,
                peak=1580.0,
                history=history,
                last_fight_date=date(2024, 3, 1),
                fight_count=5,
            ),
        })
        ctx = _make_context(elo_snapshot=elo)
        result = GraphEmitter().emit(ctx)
        assert abs(result["elo_trajectory_last5"] - 20.0) < 1e-9

    def test_trajectory_with_more_than_5_entries_uses_last_5(self) -> None:
        # Full history: [1400, 1450, 1500, 1520, 1540, 1560, 1580]
        # Last 5: [1500, 1520, 1540, 1560, 1580] — slope = 20.0
        history = (1400.0, 1450.0, 1500.0, 1520.0, 1540.0, 1560.0, 1580.0)
        elo = _make_elo_snapshot({
            FIGHTER_A: EloRecord(
                rating=1580.0,
                peak=1580.0,
                history=history,
                last_fight_date=date(2024, 3, 1),
                fight_count=7,
            ),
        })
        ctx = _make_context(elo_snapshot=elo)
        result = GraphEmitter().emit(ctx)
        assert abs(result["elo_trajectory_last5"] - 20.0) < 1e-9

    def test_trajectory_none_when_fewer_than_5(self) -> None:
        history = (1500.0, 1520.0, 1540.0, 1560.0)
        elo = _make_elo_snapshot({
            FIGHTER_A: EloRecord(
                rating=1560.0,
                peak=1560.0,
                history=history,
                last_fight_date=date(2024, 3, 1),
                fight_count=4,
            ),
        })
        ctx = _make_context(elo_snapshot=elo)
        result = GraphEmitter().emit(ctx)
        assert result["elo_trajectory_last5"] is None

    def test_trajectory_none_when_empty_history(self) -> None:
        elo = _make_elo_snapshot({
            FIGHTER_A: EloRecord(
                rating=1500.0,
                peak=1500.0,
                history=(),
                last_fight_date=date(2024, 3, 1),
                fight_count=0,
            ),
        })
        ctx = _make_context(elo_snapshot=elo)
        result = GraphEmitter().emit(ctx)
        assert result["elo_trajectory_last5"] is None

    def test_trajectory_with_non_linear_ratings(self) -> None:
        # Non-linear history: use numpy polyfit to verify
        history = (1500.0, 1530.0, 1510.0, 1550.0, 1600.0)
        elo = _make_elo_snapshot({
            FIGHTER_A: EloRecord(
                rating=1600.0,
                peak=1600.0,
                history=history,
                last_fight_date=date(2024, 3, 1),
                fight_count=5,
            ),
        })
        ctx = _make_context(elo_snapshot=elo)
        result = GraphEmitter().emit(ctx)
        # Verify against numpy polyfit
        x = np.arange(5, dtype=np.float64)
        coeffs = np.polyfit(x, np.array(history, dtype=np.float64), 1)
        expected_slope = coeffs[0]
        assert abs(result["elo_trajectory_last5"] - expected_slope) < 1e-9


class TestGlicko2Features:
    """Glicko-2 features are read from frozen component state."""

    def test_emits_glicko2_rating(self) -> None:
        glicko2 = _make_glicko2_snapshot({
            FIGHTER_A: Glicko2Record(
                mu=1620.0,
                rd=80.0,
                sigma=0.05,
                last_fight_date=date(2024, 3, 1),
                fight_count=10,
            ),
        })
        ctx = _make_context(glicko2_snapshot=glicko2)
        result = GraphEmitter().emit(ctx)
        assert result["glicko2_rating"] == 1620.0
        assert result["glicko2_rd"] == 80.0

    def test_returns_defaults_for_unknown_fighter(self) -> None:
        glicko2 = _make_glicko2_snapshot({})
        ctx = _make_context(glicko2_snapshot=glicko2)
        result = GraphEmitter().emit(ctx)
        # Unknown fighters get initial defaults from Glicko2FrozenState.get_record
        assert result["glicko2_rating"] == 1500.0
        assert result["glicko2_rd"] == 350.0


class TestPageRankFeatures:
    """PageRank features are read from frozen component state."""

    def test_emits_pagerank_score(self) -> None:
        pr = _make_pagerank_snapshot({FIGHTER_A: 0.045, FIGHTER_B: 0.030})
        ctx = _make_context(pagerank_snapshot=pr)
        result = GraphEmitter().emit(ctx)
        assert result["pagerank_score"] == 0.045

    def test_returns_none_for_unknown_fighter(self) -> None:
        pr = _make_pagerank_snapshot({})
        ctx = _make_context(pagerank_snapshot=pr)
        result = GraphEmitter().emit(ctx)
        assert result["pagerank_score"] is None


class TestCommonOpponentFeatures:
    """Common-opponent features are computed from the frozen opponent index."""

    def _build_common_opp_context(self) -> EmitContext:
        """Build context where fighters A and B share one common opponent C."""
        elo = _make_elo_snapshot({
            COMMON_OPP: EloRecord(
                rating=1550.0,
                peak=1600.0,
                history=(1550.0,),
                last_fight_date=date(2024, 1, 1),
                fight_count=5,
            ),
        })
        pr = _make_pagerank_snapshot({COMMON_OPP: 0.02})

        # A beat C, B lost to C (within lookback window)
        histories: dict[str, list[_FightEntry]] = {
            FIGHTER_A: [
                _FightEntry(
                    opponent_url=COMMON_OPP,
                    event_date=date(2023, 6, 1),
                    won=True,
                ),
            ],
            FIGHTER_B: [
                _FightEntry(
                    opponent_url=COMMON_OPP,
                    event_date=date(2023, 9, 1),
                    won=False,
                ),
            ],
            COMMON_OPP: [
                _FightEntry(
                    opponent_url=FIGHTER_A,
                    event_date=date(2023, 6, 1),
                    won=False,
                ),
                _FightEntry(
                    opponent_url=FIGHTER_B,
                    event_date=date(2023, 9, 1),
                    won=True,
                ),
            ],
        }
        common_opp = _make_common_opp_snapshot(histories, elo, pr)
        return _make_context(
            elo_snapshot=elo,
            pagerank_snapshot=pr,
            common_opp_snapshot=common_opp,
            event_date=date(2024, 6, 15),
        )

    def test_emits_n_common_opponents(self) -> None:
        ctx = self._build_common_opp_context()
        result = GraphEmitter().emit(ctx)
        assert result["n_common_opponents"] == 1

    def test_emits_common_opp_score_a_and_b(self) -> None:
        ctx = self._build_common_opp_context()
        result = GraphEmitter().emit(ctx)
        # Score is quality_score * recency_weight — exact values depend on
        # the common opponent index computation, just verify they're float
        assert isinstance(result["common_opp_score_a"], float)
        assert isinstance(result["common_opp_score_b"], float)

    def test_emits_common_opp_score_delta(self) -> None:
        ctx = self._build_common_opp_context()
        result = GraphEmitter().emit(ctx)
        score_a = result["common_opp_score_a"]
        score_b = result["common_opp_score_b"]
        expected_delta = score_a - score_b
        assert abs(result["common_opp_score_delta"] - expected_delta) < 1e-9

    def test_emits_common_opp_win_rates(self) -> None:
        ctx = self._build_common_opp_context()
        result = GraphEmitter().emit(ctx)
        # Fighter A won against common opp C: win_rate = 1.0
        assert result["common_opp_a_win_rate"] == 1.0
        # Fighter B lost to common opp C: win_rate = 0.0
        assert result["common_opp_b_win_rate"] == 0.0

    def test_returns_none_when_no_common_opponents(self) -> None:
        elo = _make_elo_snapshot({})
        pr = _make_pagerank_snapshot({})
        # Fighters have no shared opponents
        histories: dict[str, list[_FightEntry]] = {
            FIGHTER_A: [
                _FightEntry(
                    opponent_url="http://ufcstats.com/fighter/x",
                    event_date=date(2023, 6, 1),
                    won=True,
                ),
            ],
            FIGHTER_B: [
                _FightEntry(
                    opponent_url="http://ufcstats.com/fighter/y",
                    event_date=date(2023, 9, 1),
                    won=True,
                ),
            ],
        }
        common_opp = _make_common_opp_snapshot(histories, elo, pr)
        ctx = _make_context(
            elo_snapshot=elo,
            pagerank_snapshot=pr,
            common_opp_snapshot=common_opp,
            event_date=date(2024, 6, 15),
        )
        result = GraphEmitter().emit(ctx)
        assert result["n_common_opponents"] == 0
        assert result["common_opp_score_a"] is None
        assert result["common_opp_score_b"] is None
        assert result["common_opp_score_delta"] is None
        assert result["common_opp_a_win_rate"] is None
        assert result["common_opp_b_win_rate"] is None

    def test_returns_none_when_component_missing(self) -> None:
        ctx = _make_context()
        result = GraphEmitter().emit(ctx)
        assert result["n_common_opponents"] == 0
        assert result["common_opp_score_a"] is None
        assert result["common_opp_score_b"] is None
        assert result["common_opp_score_delta"] is None
        assert result["common_opp_a_win_rate"] is None
        assert result["common_opp_b_win_rate"] is None


class TestMissingComponents:
    """Graceful degradation when components are absent from context."""

    def test_all_none_when_no_components(self) -> None:
        ctx = _make_context()
        result = GraphEmitter().emit(ctx)
        assert result["elo_rating"] is None
        assert result["elo_trajectory_last5"] is None
        assert result["elo_peak"] is None
        assert result["elo_current_vs_peak"] is None
        assert result["glicko2_rating"] is None
        assert result["glicko2_rd"] is None
        assert result["pagerank_score"] is None
        assert result["n_common_opponents"] == 0
