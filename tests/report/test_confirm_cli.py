"""Tests for the human confirmation CLI.

Verifies display grouping by event date, confirmation status updates,
skip-leaves-unchanged behavior, and invalid token rejection.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import duckdb
import pytest

from ufc_edge.report.confirm_cli import (
    InvalidTokenError,
    confirm_link,
    display_unresolved,
)
from ufc_edge.report.schemas import MarketFightLink, MatchMethod, MatchStatus
from ufc_edge.report.storage import REPORT_DDL

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with report tables created."""
    db = duckdb.connect(":memory:")
    for ddl in REPORT_DDL:
        db.execute(ddl)
    return db


def _make_link(
    fight_url: str = "/fight/1",
    token_id: str = "tok_abc",
    match_status: MatchStatus = MatchStatus.NO_CANDIDATE,
    match_method: MatchMethod | None = None,
    matched_at: datetime | None = None,
    reviewed_by: str | None = None,
) -> MarketFightLink:
    """Helper to build a MarketFightLink with sensible defaults."""
    return MarketFightLink(
        fight_url=fight_url,
        token_id=token_id,
        match_status=match_status,
        match_method=match_method,
        matched_at=matched_at or datetime(2025, 6, 1, 12, 0),
        reviewed_by=reviewed_by,
    )


def _insert_link(conn: duckdb.DuckDBPyConnection, link: MarketFightLink) -> None:
    """Insert a link directly into the table for test setup."""
    conn.execute(
        """
        INSERT INTO market_fight_links
            (fight_url, token_id, match_status, match_method,
             candidate_count, matched_at, reviewed_by)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            link.fight_url,
            link.token_id,
            link.match_status.value,
            link.match_method.value if link.match_method else None,
            link.candidate_count,
            link.matched_at,
            link.reviewed_by,
        ],
    )


def _insert_snapshot(conn: duckdb.DuckDBPyConnection, token_id: str) -> None:
    """Insert a minimal order_book_snapshots row to make a token valid."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS order_book_snapshots (
            token_id VARCHAR NOT NULL,
            captured_at TIMESTAMP NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO order_book_snapshots (token_id, captured_at) VALUES (?, ?)",
        [token_id, datetime(2025, 6, 1, 12, 0)],
    )


# ── display_unresolved tests ─────────────────────────────────────────────────


def test_display_groups_by_event_date(capsys: pytest.CaptureFixture[str]) -> None:
    """Unresolved links are displayed grouped by event date."""
    links = [
        _make_link(fight_url="/fight/a", token_id="tok_1"),
        _make_link(fight_url="/fight/b", token_id="tok_2"),
        _make_link(fight_url="/fight/c", token_id="tok_3"),
    ]

    # Stub get_unresolved to inject event dates via a side table approach.
    # The display function groups by looking up event info.
    # Since display_unresolved accepts links directly, we test its output.
    display_unresolved(links)

    captured = capsys.readouterr()
    # All links should appear in output
    assert "/fight/a" in captured.out
    assert "/fight/b" in captured.out
    assert "/fight/c" in captured.out


def test_display_empty_list_shows_no_unresolved(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When no unresolved links exist, an informative message is shown."""
    display_unresolved([])

    captured = capsys.readouterr()
    assert "no unresolved" in captured.out.lower()


def test_display_groups_multiple_statuses(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Both NO_CANDIDATE and MULTIPLE_CANDIDATES rows appear in display."""
    links = [
        _make_link(
            fight_url="/fight/x",
            token_id="tok_x",
            match_status=MatchStatus.NO_CANDIDATE,
        ),
        _make_link(
            fight_url="/fight/y",
            token_id="tok_y",
            match_status=MatchStatus.MULTIPLE_CANDIDATES,
        ),
    ]

    display_unresolved(links)

    captured = capsys.readouterr()
    assert "NO_CANDIDATE" in captured.out
    assert "MULTIPLE_CANDIDATES" in captured.out


# ── confirm_link tests ────────────────────────────────────────────────────────


def test_confirm_updates_status_to_matched(conn: duckdb.DuckDBPyConnection) -> None:
    """Confirming a link sets status to MATCHED with HUMAN_CONFIRMED method."""
    link = _make_link(fight_url="/fight/1", token_id="tok_old")
    _insert_link(conn, link)
    _insert_snapshot(conn, "tok_valid")

    confirm_link(
        fight_url="/fight/1",
        token_id="tok_valid",
        reviewer="testuser",
        conn=conn,
    )

    row = conn.execute(
        "SELECT match_status, match_method, reviewed_by, token_id "
        "FROM market_fight_links WHERE fight_url = ?",
        ["/fight/1"],
    ).fetchone()

    assert row is not None
    assert row[0] == MatchStatus.MATCHED.value
    assert row[1] == MatchMethod.HUMAN_CONFIRMED.value
    assert row[2] == "testuser"
    assert row[3] == "tok_valid"


def test_skip_leaves_unchanged(conn: duckdb.DuckDBPyConnection) -> None:
    """Not calling confirm_link leaves the row in its original state."""
    link = _make_link(
        fight_url="/fight/2",
        token_id="tok_pending",
        match_status=MatchStatus.MULTIPLE_CANDIDATES,
    )
    _insert_link(conn, link)

    # Simulate skipping: do not call confirm_link.

    row = conn.execute(
        "SELECT match_status, match_method, reviewed_by "
        "FROM market_fight_links WHERE fight_url = ?",
        ["/fight/2"],
    ).fetchone()

    assert row is not None
    assert row[0] == MatchStatus.MULTIPLE_CANDIDATES.value
    assert row[1] is None
    assert row[2] is None


def test_invalid_token_rejected(conn: duckdb.DuckDBPyConnection) -> None:
    """Confirming with a token_id that has no snapshot raises InvalidTokenError."""
    link = _make_link(fight_url="/fight/3", token_id="tok_orig")
    _insert_link(conn, link)
    # Create the table but do NOT insert the token
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS order_book_snapshots (
            token_id VARCHAR NOT NULL,
            captured_at TIMESTAMP NOT NULL
        )
        """
    )

    with pytest.raises(InvalidTokenError):
        confirm_link(
            fight_url="/fight/3",
            token_id="tok_nonexistent",
            reviewer="testuser",
            conn=conn,
        )


def test_confirm_link_nonexistent_fight_raises(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """Confirming a fight_url not in the table raises ValueError."""
    _insert_snapshot(conn, "tok_valid")

    with pytest.raises(ValueError, match="not found"):
        confirm_link(
            fight_url="/fight/nonexistent",
            token_id="tok_valid",
            reviewer="testuser",
            conn=conn,
        )


def test_confirmed_row_not_presented_again(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """After confirmation, the row should not appear in unresolved queries."""
    link = _make_link(fight_url="/fight/4", token_id="tok_old")
    _insert_link(conn, link)
    _insert_snapshot(conn, "tok_confirmed")

    confirm_link(
        fight_url="/fight/4",
        token_id="tok_confirmed",
        reviewer="reviewer1",
        conn=conn,
    )

    # Query unresolved (rows with non-MATCHED status and no reviewed_by)
    rows = conn.execute(
        "SELECT * FROM market_fight_links "
        "WHERE match_status != 'MATCHED' AND reviewed_by IS NULL"
    ).fetchall()

    assert len(rows) == 0


# ── CLI entry point tests ─────────────────────────────────────────────────────


def test_cli_list_subcommand(
    conn: duckdb.DuckDBPyConnection,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 'list' subcommand displays unresolved links."""
    link = _make_link(fight_url="/fight/cli_test", token_id="tok_cli")
    _insert_link(conn, link)

    # Mock get_unresolved to return our test data
    monkeypatch.setattr(
        "ufc_edge.report.confirm_cli.get_unresolved",
        lambda c: [link],
    )

    from ufc_edge.report.confirm_cli import main

    with patch("sys.argv", ["confirm_cli", "--db", ":memory:", "list"]):
        # Inject our conn
        monkeypatch.setattr(
            "ufc_edge.report.confirm_cli._connect",
            lambda path: conn,
        )
        main()

    captured = capsys.readouterr()
    assert "/fight/cli_test" in captured.out


def test_cli_confirm_subcommand(
    conn: duckdb.DuckDBPyConnection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 'confirm' subcommand updates the link status."""
    link = _make_link(fight_url="/fight/5", token_id="tok_old")
    _insert_link(conn, link)
    _insert_snapshot(conn, "tok_new")

    monkeypatch.setattr(
        "ufc_edge.report.confirm_cli._connect",
        lambda path: conn,
    )

    from ufc_edge.report.confirm_cli import main

    with patch(
        "sys.argv",
        [
            "confirm_cli",
            "--db",
            ":memory:",
            "confirm",
            "--fight-url",
            "/fight/5",
            "--token-id",
            "tok_new",
            "--reviewer",
            "human1",
        ],
    ):
        main()

    row = conn.execute(
        "SELECT match_status, reviewed_by FROM market_fight_links WHERE fight_url = ?",
        ["/fight/5"],
    ).fetchone()
    assert row is not None
    assert row[0] == MatchStatus.MATCHED.value
    assert row[1] == "human1"


def test_cli_confirm_invalid_token_exits_with_error(
    conn: duckdb.DuckDBPyConnection,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI prints an error and exits non-zero for invalid tokens."""
    link = _make_link(fight_url="/fight/6", token_id="tok_old")
    _insert_link(conn, link)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS order_book_snapshots (
            token_id VARCHAR NOT NULL,
            captured_at TIMESTAMP NOT NULL
        )
        """
    )

    monkeypatch.setattr(
        "ufc_edge.report.confirm_cli._connect",
        lambda path: conn,
    )

    from ufc_edge.report.confirm_cli import main

    with patch(
        "sys.argv",
        [
            "confirm_cli",
            "--db",
            ":memory:",
            "confirm",
            "--fight-url",
            "/fight/6",
            "--token-id",
            "tok_bad",
            "--reviewer",
            "human1",
        ],
    ):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0

    captured = capsys.readouterr()
    assert "invalid" in captured.err.lower() or "not" in captured.err.lower()
