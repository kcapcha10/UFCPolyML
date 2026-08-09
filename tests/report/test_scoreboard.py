"""Tests for verdict scoreboard update and query logic.

Verifies that outcome_correct is computed correctly for each verdict type:
CONFIRM is correct when the signal-favored fighter wins (endorsement was
right), VETO is correct when the signal-favored fighter loses (warning was
justified). Also verifies that the query function returns per-verdict-type
counts, win rates, and mean mismatch magnitude.
"""

from __future__ import annotations

import duckdb
import pytest

from ufc_edge.report.schemas import DueDiligenceVerdictType, ScoreboardEntry
from ufc_edge.report.scoreboard import query_scoreboard, update_scoreboard
from ufc_edge.report.storage import REPORT_DDL

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with the verdict_scoreboard table initialized."""
    db = duckdb.connect(":memory:")
    for ddl in REPORT_DDL:
        db.execute(ddl)
    return db


# ── CONFIRM verdict tests ─────────────────────────────────────────────────────


def test_confirm_correct_when_signal_favored_fighter_wins(conn: duckdb.DuckDBPyConnection) -> None:
    """A CONFIRM verdict is correct when the fighter the signal favored actually won."""
    entry = update_scoreboard(
        fight_url="http://ufcstats.com/fight/aaa",
        report_run_id="run-001",
        verdict=DueDiligenceVerdictType.CONFIRM,
        mismatch_at_signal=0.15,
        actual_winner_url="http://ufcstats.com/fighter/alpha",
        signal_favored_url="http://ufcstats.com/fighter/alpha",
        conn=conn,
    )

    assert entry.outcome_correct is True
    assert entry.fight_resolved is True
    assert entry.verdict == DueDiligenceVerdictType.CONFIRM


def test_confirm_incorrect_when_signal_favored_fighter_loses(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """A CONFIRM verdict is incorrect when the favored fighter did not win."""
    entry = update_scoreboard(
        fight_url="http://ufcstats.com/fight/bbb",
        report_run_id="run-002",
        verdict=DueDiligenceVerdictType.CONFIRM,
        mismatch_at_signal=0.12,
        actual_winner_url="http://ufcstats.com/fighter/beta",
        signal_favored_url="http://ufcstats.com/fighter/alpha",
        conn=conn,
    )

    assert entry.outcome_correct is False
    assert entry.fight_resolved is True


# ── VETO verdict tests ────────────────────────────────────────────────────────


def test_veto_incorrect_when_signal_favored_fighter_wins(conn: duckdb.DuckDBPyConnection) -> None:
    """A VETO is incorrect when the signal-favored fighter wins anyway.

    The model warned the signal was unreliable, but the favored fighter
    still won — meaning the veto would have wrongly suppressed a good signal.
    """
    entry = update_scoreboard(
        fight_url="http://ufcstats.com/fight/ccc",
        report_run_id="run-003",
        verdict=DueDiligenceVerdictType.VETO,
        mismatch_at_signal=0.20,
        actual_winner_url="http://ufcstats.com/fighter/alpha",
        signal_favored_url="http://ufcstats.com/fighter/alpha",
        conn=conn,
    )

    assert entry.outcome_correct is False
    assert entry.fight_resolved is True
    assert entry.verdict == DueDiligenceVerdictType.VETO


def test_veto_correct_when_signal_favored_fighter_loses(conn: duckdb.DuckDBPyConnection) -> None:
    """A VETO is correct when the signal-favored fighter loses.

    The model warned the signal was unreliable, and indeed the favored fighter
    lost — the warning was justified.
    """
    entry = update_scoreboard(
        fight_url="http://ufcstats.com/fight/ddd",
        report_run_id="run-004",
        verdict=DueDiligenceVerdictType.VETO,
        mismatch_at_signal=0.18,
        actual_winner_url="http://ufcstats.com/fighter/beta",
        signal_favored_url="http://ufcstats.com/fighter/alpha",
        conn=conn,
    )

    assert entry.outcome_correct is True
    assert entry.fight_resolved is True


# ── QUALIFY verdict tests ─────────────────────────────────────────────────────


def test_qualify_uses_confirm_logic(conn: duckdb.DuckDBPyConnection) -> None:
    """QUALIFY endorsed the signal with caveats — correct when favored fighter wins."""
    entry = update_scoreboard(
        fight_url="http://ufcstats.com/fight/eee",
        report_run_id="run-005",
        verdict=DueDiligenceVerdictType.QUALIFY,
        mismatch_at_signal=0.10,
        actual_winner_url="http://ufcstats.com/fighter/alpha",
        signal_favored_url="http://ufcstats.com/fighter/alpha",
        conn=conn,
    )

    assert entry.outcome_correct is True


# ── Query scoreboard tests ────────────────────────────────────────────────────


def test_query_returns_per_verdict_aggregates(conn: duckdb.DuckDBPyConnection) -> None:
    """Query returns per-verdict-type counts, win rates, and mean mismatch."""
    # Two CONFIRM entries: one correct (0.15 mismatch), one incorrect (0.25)
    update_scoreboard(
        fight_url="http://ufcstats.com/fight/f1",
        report_run_id="run-q1",
        verdict=DueDiligenceVerdictType.CONFIRM,
        mismatch_at_signal=0.15,
        actual_winner_url="http://ufcstats.com/fighter/alpha",
        signal_favored_url="http://ufcstats.com/fighter/alpha",
        conn=conn,
    )
    update_scoreboard(
        fight_url="http://ufcstats.com/fight/f2",
        report_run_id="run-q2",
        verdict=DueDiligenceVerdictType.CONFIRM,
        mismatch_at_signal=0.25,
        actual_winner_url="http://ufcstats.com/fighter/beta",
        signal_favored_url="http://ufcstats.com/fighter/alpha",
        conn=conn,
    )

    # One VETO entry: correct (favored fighter lost, so veto was right)
    update_scoreboard(
        fight_url="http://ufcstats.com/fight/f3",
        report_run_id="run-q3",
        verdict=DueDiligenceVerdictType.VETO,
        mismatch_at_signal=-0.30,
        actual_winner_url="http://ufcstats.com/fighter/beta",
        signal_favored_url="http://ufcstats.com/fighter/alpha",
        conn=conn,
    )

    summary = query_scoreboard(conn)

    assert summary.total_resolved == 3

    confirm_stats = summary.by_verdict["CONFIRM"]
    assert confirm_stats.count == 2
    assert confirm_stats.correct == 1
    assert confirm_stats.incorrect == 1
    assert confirm_stats.win_rate == pytest.approx(0.5)
    assert confirm_stats.mean_mismatch_magnitude == pytest.approx(0.20)

    veto_stats = summary.by_verdict["VETO"]
    assert veto_stats.count == 1
    assert veto_stats.correct == 1
    assert veto_stats.incorrect == 0
    assert veto_stats.win_rate == pytest.approx(1.0)
    assert veto_stats.mean_mismatch_magnitude == pytest.approx(0.30)


def test_query_empty_scoreboard(conn: duckdb.DuckDBPyConnection) -> None:
    """Query on an empty scoreboard returns zero totals and empty dict."""
    summary = query_scoreboard(conn)

    assert summary.total_resolved == 0
    assert summary.by_verdict == {}


def test_query_ignores_unresolved_entries(conn: duckdb.DuckDBPyConnection) -> None:
    """Only resolved entries (fight_resolved=True) are included in aggregates."""
    # Insert a resolved entry via update_scoreboard
    update_scoreboard(
        fight_url="http://ufcstats.com/fight/g1",
        report_run_id="run-g1",
        verdict=DueDiligenceVerdictType.CONFIRM,
        mismatch_at_signal=0.12,
        actual_winner_url="http://ufcstats.com/fighter/alpha",
        signal_favored_url="http://ufcstats.com/fighter/alpha",
        conn=conn,
    )

    # Manually insert an unresolved row (simulates a pending fight)
    conn.execute(
        """
        INSERT INTO verdict_scoreboard
            (fight_url, report_run_id, verdict, mismatch_at_signal,
             fight_resolved, outcome_correct, resolved_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "http://ufcstats.com/fight/g2",
            "run-g2",
            "CONFIRM",
            0.10,
            False,
            None,
            None,
        ],
    )

    summary = query_scoreboard(conn)

    # Only the resolved entry should be counted
    assert summary.total_resolved == 1
    assert summary.by_verdict["CONFIRM"].count == 1


def test_mismatch_at_signal_none_handled(conn: duckdb.DuckDBPyConnection) -> None:
    """Entries with None mismatch don't break the query; mean excludes nulls."""
    update_scoreboard(
        fight_url="http://ufcstats.com/fight/h1",
        report_run_id="run-h1",
        verdict=DueDiligenceVerdictType.CONFIRM,
        mismatch_at_signal=None,
        actual_winner_url="http://ufcstats.com/fighter/alpha",
        signal_favored_url="http://ufcstats.com/fighter/alpha",
        conn=conn,
    )

    summary = query_scoreboard(conn)
    assert summary.total_resolved == 1
    # mean_mismatch_magnitude is None when all mismatch values are NULL
    assert summary.by_verdict["CONFIRM"].mean_mismatch_magnitude is None


def test_update_returns_frozen_scoreboard_entry(conn: duckdb.DuckDBPyConnection) -> None:
    """update_scoreboard returns a frozen ScoreboardEntry model."""
    entry = update_scoreboard(
        fight_url="http://ufcstats.com/fight/i1",
        report_run_id="run-i1",
        verdict=DueDiligenceVerdictType.VETO,
        mismatch_at_signal=0.22,
        actual_winner_url="http://ufcstats.com/fighter/beta",
        signal_favored_url="http://ufcstats.com/fighter/alpha",
        conn=conn,
    )

    assert isinstance(entry, ScoreboardEntry)
    assert entry.fight_url == "http://ufcstats.com/fight/i1"
    assert entry.report_run_id == "run-i1"
    assert entry.mismatch_at_signal == 0.22
    assert entry.resolved_at is not None
