"""Tests for name normalization and fight-market link resolution.

Covers normalization edge cases (diacritics, hyphens, apostrophes, suffixes,
whitespace, case) and link resolution outcomes (exact match, no candidates,
multiple candidates, invalid input, missing snapshot). Also verifies
idempotency and immutability guarantees.
"""

from __future__ import annotations

import json
from datetime import datetime

import duckdb
import pytest

from ufc_edge.report.matching import get_unresolved, normalize_name, resolve_links
from ufc_edge.report.schemas import MatchMethod, MatchStatus
from ufc_edge.report.storage import LinkOverwriteError, write_link

# ── Fixtures ──────────────────────────────────────────────────────────────────

_SNAPSHOT_COLS = (
    "market_id, token_id, question, outcome, "
    "bids, asks, mid_price, spread, captured_at, tick_id"
)

_SNAPSHOT_INSERT = f"""
    INSERT INTO order_book_snapshots ({_SNAPSHOT_COLS})
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with fighters, order_book_snapshots, and market_fight_links."""
    db = duckdb.connect(":memory:")

    db.execute("""
        CREATE TABLE fighters (
            fighter_url VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            scraped_at TIMESTAMP NOT NULL
        )
    """)

    db.execute("""
        CREATE TABLE order_book_snapshots (
            market_id VARCHAR NOT NULL,
            token_id VARCHAR NOT NULL,
            question VARCHAR,
            outcome VARCHAR,
            bids JSON NOT NULL,
            asks JSON NOT NULL,
            mid_price DOUBLE,
            spread DOUBLE,
            captured_at TIMESTAMP NOT NULL,
            tick_id VARCHAR,
            PRIMARY KEY (token_id, captured_at)
        )
    """)

    db.execute("""
        CREATE TABLE market_fight_links (
            fight_url TEXT NOT NULL,
            token_id TEXT NOT NULL,
            match_status TEXT NOT NULL,
            match_method TEXT,
            candidate_count INTEGER,
            matched_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            reviewed_by TEXT,
            PRIMARY KEY (fight_url, token_id)
        )
    """)

    return db


@pytest.fixture
def populated_conn(conn: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """Conn with fighters and matching market snapshots already inserted."""
    now = datetime(2026, 7, 1, 12, 0, 0)

    # Insert fighters
    conn.execute(
        "INSERT INTO fighters VALUES (?, ?, ?)",
        ["http://ufcstats.com/fighter/aaa", "Conor McGregor", now],
    )
    conn.execute(
        "INSERT INTO fighters VALUES (?, ?, ?)",
        ["http://ufcstats.com/fighter/bbb", "Dustin Poirier", now],
    )
    conn.execute(
        "INSERT INTO fighters VALUES (?, ?, ?)",
        ["http://ufcstats.com/fighter/ccc", "Jiří Procházka", now],
    )
    conn.execute(
        "INSERT INTO fighters VALUES (?, ?, ?)",
        ["http://ufcstats.com/fighter/ddd", "Alex Pereira", now],
    )

    # Insert order_book_snapshots with UFC-tagged questions
    bids = json.dumps([{"price": 0.6, "size": 100}])
    asks = json.dumps([{"price": 0.4, "size": 80}])

    # McGregor vs Poirier market
    conn.execute(
        _SNAPSHOT_INSERT,
        [
            "market-1",
            "token-mcgregor-poirier",
            "UFC 310: Will Conor McGregor beat Dustin Poirier?",
            "Yes",
            bids,
            asks,
            0.5,
            0.2,
            datetime(2026, 7, 1, 10, 0, 0),
            "tick-1",
        ],
    )

    # Prochazka vs Pereira market
    conn.execute(
        _SNAPSHOT_INSERT,
        [
            "market-2",
            "token-prochazka-pereira",
            "UFC 310: Will Jiri Prochazka beat Alex Pereira?",
            "Yes",
            bids,
            asks,
            0.45,
            0.15,
            datetime(2026, 7, 1, 10, 0, 0),
            "tick-2",
        ],
    )

    return conn


# ── normalize_name tests ──────────────────────────────────────────────────────


class TestNormalizeName:
    """Name normalization applies lowercase, diacritic strip, punctuation removal."""

    def test_lowercase(self):
        assert normalize_name("Conor McGregor") == "conor mcgregor"

    def test_diacritics_stripped(self):
        assert normalize_name("Jiří") == "jiri"

    def test_diacritics_full_name(self):
        assert normalize_name("Jiří Procházka") == "jiri prochazka"

    def test_hyphens_removed(self):
        assert normalize_name("Jean-Claude") == "jeanclaude"

    def test_apostrophes_removed(self):
        assert normalize_name("O'Malley") == "omalley"

    def test_suffix_jr_removed(self):
        assert normalize_name("Deiveson Figueiredo Jr.") == "deiveson figueiredo"

    def test_suffix_iii_removed(self):
        assert normalize_name("Henry Cejudo III") == "henry cejudo"

    def test_suffix_ii_removed(self):
        assert normalize_name("Fighter Name II") == "fighter name"

    def test_double_spaces_collapsed(self):
        assert normalize_name("Conor  McGregor") == "conor mcgregor"

    def test_multiple_spaces_collapsed(self):
        assert normalize_name("Conor    McGregor") == "conor mcgregor"

    def test_leading_trailing_whitespace(self):
        assert normalize_name("  Conor McGregor  ") == "conor mcgregor"

    def test_combined_edge_case(self):
        # Diacritics + suffix + multiple spaces
        assert normalize_name("  Jiří  Procházka Jr.  ") == "jiri prochazka"

    def test_empty_string(self):
        assert normalize_name("") == ""

    def test_pure_punctuation(self):
        assert normalize_name("...") == ""


# ── resolve_links tests ───────────────────────────────────────────────────────


class TestResolveLinks:
    """Link resolution maps fights to markets based on normalized name matching."""

    def test_exact_one_match_returns_matched(
        self, populated_conn: duckdb.DuckDBPyConnection
    ):
        fights = [
            {
                "fight_url": "http://ufcstats.com/fight/123",
                "fighter_a_url": "http://ufcstats.com/fighter/aaa",
                "fighter_b_url": "http://ufcstats.com/fighter/bbb",
            }
        ]
        as_of = datetime(2026, 7, 1, 12, 0, 0)

        links = resolve_links(fights, as_of, populated_conn)

        assert len(links) == 1
        assert links[0].match_status == MatchStatus.MATCHED
        assert links[0].token_id == "token-mcgregor-poirier"
        assert links[0].fight_url == "http://ufcstats.com/fight/123"

    def test_zero_match_returns_no_candidate(
        self, populated_conn: duckdb.DuckDBPyConnection
    ):
        # Use fighter URLs that exist but have no matching market question
        populated_conn.execute(
            "INSERT INTO fighters VALUES (?, ?, ?)",
            [
                "http://ufcstats.com/fighter/zzz",
                "Unknown Fighter",
                datetime(2026, 7, 1),
            ],
        )
        fights = [
            {
                "fight_url": "http://ufcstats.com/fight/999",
                "fighter_a_url": "http://ufcstats.com/fighter/zzz",
                "fighter_b_url": "http://ufcstats.com/fighter/bbb",
            }
        ]
        as_of = datetime(2026, 7, 1, 12, 0, 0)

        links = resolve_links(fights, as_of, populated_conn)

        assert len(links) == 1
        assert links[0].match_status == MatchStatus.NO_CANDIDATE

    def test_multiple_match_returns_multiple_candidates(
        self, populated_conn: duckdb.DuckDBPyConnection
    ):
        # Add a duplicate market with same fighter names
        bids = json.dumps([{"price": 0.55, "size": 90}])
        asks = json.dumps([{"price": 0.45, "size": 70}])
        populated_conn.execute(
            _SNAPSHOT_INSERT,
            [
                "market-dup",
                "token-dup",
                "UFC 311: Will Conor McGregor beat Dustin Poirier?",
                "Yes",
                bids,
                asks,
                0.5,
                0.1,
                datetime(2026, 7, 1, 11, 0, 0),
                "tick-dup",
            ],
        )

        fights = [
            {
                "fight_url": "http://ufcstats.com/fight/456",
                "fighter_a_url": "http://ufcstats.com/fighter/aaa",
                "fighter_b_url": "http://ufcstats.com/fighter/bbb",
            }
        ]
        as_of = datetime(2026, 7, 1, 12, 0, 0)

        links = resolve_links(fights, as_of, populated_conn)

        assert len(links) == 1
        assert links[0].match_status == MatchStatus.MULTIPLE_CANDIDATES
        assert links[0].candidate_count == 2

    def test_invalid_fighter_url_returns_invalid_input(
        self, populated_conn: duckdb.DuckDBPyConnection
    ):
        fights = [
            {
                "fight_url": "http://ufcstats.com/fight/bad",
                "fighter_a_url": "http://ufcstats.com/fighter/NONEXISTENT",
                "fighter_b_url": "http://ufcstats.com/fighter/bbb",
            }
        ]
        as_of = datetime(2026, 7, 1, 12, 0, 0)

        links = resolve_links(fights, as_of, populated_conn)

        assert len(links) == 1
        assert links[0].match_status == MatchStatus.INVALID_INPUT

    def test_no_snapshot_returns_missing_snapshot(
        self, conn: duckdb.DuckDBPyConnection
    ):
        # Fighters exist but no snapshots in the DB at all
        now = datetime(2026, 7, 1, 12, 0, 0)
        conn.execute(
            "INSERT INTO fighters VALUES (?, ?, ?)",
            ["http://ufcstats.com/fighter/aaa", "Conor McGregor", now],
        )
        conn.execute(
            "INSERT INTO fighters VALUES (?, ?, ?)",
            ["http://ufcstats.com/fighter/bbb", "Dustin Poirier", now],
        )

        fights = [
            {
                "fight_url": "http://ufcstats.com/fight/no-snap",
                "fighter_a_url": "http://ufcstats.com/fighter/aaa",
                "fighter_b_url": "http://ufcstats.com/fighter/bbb",
            }
        ]
        as_of = datetime(2026, 7, 1, 12, 0, 0)

        links = resolve_links(fights, as_of, conn)

        assert len(links) == 1
        assert links[0].match_status == MatchStatus.MISSING_SNAPSHOT


# ── Idempotency tests ─────────────────────────────────────────────────────────


class TestIdempotency:
    """Re-running resolve_links with same inputs produces no new rows."""

    def test_rerun_produces_no_new_rows(
        self, populated_conn: duckdb.DuckDBPyConnection
    ):
        fights = [
            {
                "fight_url": "http://ufcstats.com/fight/123",
                "fighter_a_url": "http://ufcstats.com/fighter/aaa",
                "fighter_b_url": "http://ufcstats.com/fighter/bbb",
            }
        ]
        as_of = datetime(2026, 7, 1, 12, 0, 0)

        # First run writes links
        links_first = resolve_links(fights, as_of, populated_conn)
        count_after_first = populated_conn.execute(
            "SELECT COUNT(*) FROM market_fight_links"
        ).fetchone()[0]

        # Second run with same inputs
        links_second = resolve_links(fights, as_of, populated_conn)
        count_after_second = populated_conn.execute(
            "SELECT COUNT(*) FROM market_fight_links"
        ).fetchone()[0]

        assert count_after_first == count_after_second
        assert len(links_first) == len(links_second)
        assert links_first[0].match_status == links_second[0].match_status


# ── Immutability tests ────────────────────────────────────────────────────────


class TestImmutability:
    """Attempting to overwrite a MATCHED link raises LinkOverwriteError."""

    def test_overwrite_matched_raises(
        self, populated_conn: duckdb.DuckDBPyConnection
    ):
        from ufc_edge.report.schemas import MarketFightLink

        now = datetime(2026, 7, 1, 12, 0, 0)

        # Write a MATCHED link
        link = MarketFightLink(
            fight_url="http://ufcstats.com/fight/immutable",
            token_id="token-immutable",
            match_status=MatchStatus.MATCHED,
            match_method=MatchMethod.AUTO_NAME,
            matched_at=now,
        )
        write_link(populated_conn, link)

        # Attempting to overwrite should raise
        replacement = MarketFightLink(
            fight_url="http://ufcstats.com/fight/immutable",
            token_id="token-immutable",
            match_status=MatchStatus.MATCHED,
            match_method=MatchMethod.HUMAN_CONFIRMED,
            matched_at=now,
            reviewed_by="admin",
        )
        with pytest.raises(LinkOverwriteError):
            write_link(populated_conn, replacement)


# ── get_unresolved tests ──────────────────────────────────────────────────────


class TestGetUnresolved:
    """get_unresolved returns non-MATCHED rows without a reviewer."""

    def test_returns_non_matched_rows(
        self, populated_conn: duckdb.DuckDBPyConnection
    ):
        now = datetime(2026, 7, 1, 12, 0, 0)

        # Write a NO_CANDIDATE link (unresolved)
        populated_conn.execute(
            """INSERT INTO market_fight_links
               (fight_url, token_id, match_status, matched_at)
               VALUES (?, ?, ?, ?)""",
            [
                "http://ufcstats.com/fight/unresolved",
                "token-x",
                "NO_CANDIDATE",
                now,
            ],
        )

        # Write a MATCHED link (should not appear)
        populated_conn.execute(
            """INSERT INTO market_fight_links
               (fight_url, token_id, match_status, match_method, matched_at)
               VALUES (?, ?, ?, ?, ?)""",
            [
                "http://ufcstats.com/fight/resolved",
                "token-y",
                "MATCHED",
                "AUTO_NAME",
                now,
            ],
        )

        unresolved = get_unresolved(populated_conn)

        assert len(unresolved) == 1
        assert unresolved[0].fight_url == "http://ufcstats.com/fight/unresolved"
        assert unresolved[0].match_status == MatchStatus.NO_CANDIDATE

    def test_excludes_reviewed_rows(
        self, populated_conn: duckdb.DuckDBPyConnection
    ):
        now = datetime(2026, 7, 1, 12, 0, 0)

        # Write a MULTIPLE_CANDIDATES link that has been reviewed
        populated_conn.execute(
            """INSERT INTO market_fight_links
               (fight_url, token_id, match_status, candidate_count,
                matched_at, reviewed_by)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                "http://ufcstats.com/fight/reviewed",
                "token-rev",
                "MULTIPLE_CANDIDATES",
                3,
                now,
                "reviewer-alias",
            ],
        )

        unresolved = get_unresolved(populated_conn)

        # The reviewed row should not appear
        fight_urls = [link.fight_url for link in unresolved]
        assert "http://ufcstats.com/fight/reviewed" not in fight_urls
