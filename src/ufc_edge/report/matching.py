"""Entity resolution between Polymarket tokens and UFC fights.

Normalizes fighter names, queries active market snapshots for matches,
and persists link records. Provides get_unresolved for human confirmation flow.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime

import duckdb

from ufc_edge.report.schemas import MarketFightLink, MatchMethod, MatchStatus
from ufc_edge.report.storage import write_link

# Regex to strip Unicode combining marks (accents, diacritics) after NFD decomposition
_COMBINING_MARK_RE = re.compile(r"[\u0300-\u036f]")

# Suffixes to strip from names (Jr., Sr., II, III, IV, etc.)
_SUFFIX_RE = re.compile(r"\b(?:jr\.?|sr\.?|ii|iii|iv|v)\s*$", re.IGNORECASE)

# Non-alphanumeric and non-space characters (punctuation, hyphens, apostrophes)
_PUNCTUATION_RE = re.compile(r"[^\w\s]|_")

# Consecutive whitespace
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_name(raw: str) -> str:
    """Normalize a fighter name to a canonical lowercase form.

    Applies NFD decomposition with combining-mark removal, lowercasing,
    suffix stripping, punctuation removal, and whitespace collapse.
    """
    # NFD decomposition separates base chars from combining marks
    decomposed = unicodedata.normalize("NFD", raw)

    # Strip combining marks (diacritics)
    without_diacritics = _COMBINING_MARK_RE.sub("", decomposed)

    # Lowercase
    lowered = without_diacritics.lower()

    # Strip suffixes before punctuation removal (so "Jr." is caught)
    without_suffix = _SUFFIX_RE.sub("", lowered)

    # Remove punctuation (hyphens, apostrophes, periods, etc.)
    without_punct = _PUNCTUATION_RE.sub("", without_suffix)

    # Collapse whitespace and strip edges
    collapsed = _WHITESPACE_RE.sub(" ", without_punct).strip()

    return collapsed


def resolve_links(
    fights: list[dict],
    as_of: datetime,
    conn: duckdb.DuckDBPyConnection,
) -> list[MarketFightLink]:
    """Resolve fight-to-market links via normalized name matching.

    For each fight, checks for an existing link first (idempotent reuse).
    If absent, looks up fighter names, searches market snapshots for
    matching questions, and writes the result.

    Args:
        fights: List of dicts with fight_url, fighter_a_url, fighter_b_url.
        as_of: Timestamp cutoff for snapshot eligibility.
        conn: DuckDB connection with fighters, order_book_snapshots, and
              market_fight_links tables.

    Returns:
        List of MarketFightLink objects (one per fight).
    """
    results: list[MarketFightLink] = []

    for fight in fights:
        fight_url = fight["fight_url"]
        fighter_a_url = fight["fighter_a_url"]
        fighter_b_url = fight["fighter_b_url"]

        # Check for existing link (idempotency: skip if already resolved)
        existing = conn.execute(
            "SELECT match_status, token_id, match_method, candidate_count, matched_at, reviewed_by "
            "FROM market_fight_links WHERE fight_url = ?",
            [fight_url],
        ).fetchone()

        if existing is not None:
            link = MarketFightLink(
                fight_url=fight_url,
                token_id=existing[1],
                match_status=MatchStatus(existing[0]),
                match_method=MatchMethod(existing[2]) if existing[2] else None,
                candidate_count=existing[3],
                matched_at=existing[4],
                reviewed_by=existing[5],
            )
            results.append(link)
            continue

        # Look up fighter names from the fighters table
        fighter_a_row = conn.execute(
            "SELECT name FROM fighters WHERE fighter_url = ?", [fighter_a_url]
        ).fetchone()
        fighter_b_row = conn.execute(
            "SELECT name FROM fighters WHERE fighter_url = ?", [fighter_b_url]
        ).fetchone()

        if fighter_a_row is None or fighter_b_row is None:
            link = _make_link(fight_url, "", MatchStatus.INVALID_INPUT)
            write_link(conn, link)
            results.append(link)
            continue

        name_a = normalize_name(fighter_a_row[0])
        name_b = normalize_name(fighter_b_row[0])

        # Check if any snapshots exist at or before as_of
        snapshot_exists = conn.execute(
            "SELECT 1 FROM order_book_snapshots WHERE captured_at <= ? LIMIT 1",
            [as_of],
        ).fetchone()

        if snapshot_exists is None:
            link = _make_link(fight_url, "", MatchStatus.MISSING_SNAPSHOT)
            write_link(conn, link)
            results.append(link)
            continue

        # Find candidate markets: distinct tokens whose question contains both names
        candidates = conn.execute(
            """
            SELECT DISTINCT token_id
            FROM order_book_snapshots
            WHERE captured_at <= ?
              AND question IS NOT NULL
              AND LOWER(question) LIKE ?
              AND LOWER(question) LIKE ?
            """,
            [as_of, f"%{name_a}%", f"%{name_b}%"],
        ).fetchall()

        candidate_count = len(candidates)

        if candidate_count == 0:
            link = _make_link(fight_url, "", MatchStatus.NO_CANDIDATE)
            write_link(conn, link)
            results.append(link)
        elif candidate_count == 1:
            token_id = candidates[0][0]
            link = MarketFightLink(
                fight_url=fight_url,
                token_id=token_id,
                match_status=MatchStatus.MATCHED,
                match_method=MatchMethod.AUTO_NAME,
                candidate_count=1,
                matched_at=datetime.now(),
            )
            write_link(conn, link)
            results.append(link)
        else:
            # Multiple candidates — record ambiguity for human review
            first_token = candidates[0][0]
            link = MarketFightLink(
                fight_url=fight_url,
                token_id=first_token,
                match_status=MatchStatus.MULTIPLE_CANDIDATES,
                candidate_count=candidate_count,
                matched_at=datetime.now(),
            )
            write_link(conn, link)
            results.append(link)

    return results


def get_unresolved(conn: duckdb.DuckDBPyConnection) -> list[MarketFightLink]:
    """Return all link rows that are not MATCHED and have no reviewer.

    These are candidates for human confirmation via the CLI.
    """
    rows = conn.execute(
        """
        SELECT fight_url, token_id, match_status, match_method,
               candidate_count, matched_at, reviewed_by
        FROM market_fight_links
        WHERE match_status != ?
          AND reviewed_by IS NULL
        """,
        [MatchStatus.MATCHED],
    ).fetchall()

    return [
        MarketFightLink(
            fight_url=row[0],
            token_id=row[1],
            match_status=MatchStatus(row[2]),
            match_method=MatchMethod(row[3]) if row[3] else None,
            candidate_count=row[4],
            matched_at=row[5],
            reviewed_by=row[6],
        )
        for row in rows
    ]


def _make_link(fight_url: str, token_id: str, status: MatchStatus) -> MarketFightLink:
    """Build a MarketFightLink for a non-MATCHED outcome."""
    return MarketFightLink(
        fight_url=fight_url,
        token_id=token_id,
        match_status=status,
        matched_at=datetime.now(),
    )
