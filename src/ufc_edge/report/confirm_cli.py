"""Human confirmation CLI for unresolved market-fight links.

Displays unresolved entity-resolution rows grouped by event date and
provides a command to confirm a link with a valid token. Once confirmed,
the row is updated to MATCHED with HUMAN_CONFIRMED method and the
reviewer alias recorded.

Usage:
    python -m ufc_edge.report.confirm_cli list --db data/ufc_edge.duckdb
    python -m ufc_edge.report.confirm_cli confirm --fight-url ... --token-id ... --reviewer ...
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from itertools import groupby
from typing import TYPE_CHECKING

import duckdb

from ufc_edge.report.schemas import (
    MarketFightLink,
    MatchMethod,
    MatchStatus,
)

if TYPE_CHECKING:
    pass


# ── Exceptions ────────────────────────────────────────────────────────────────


class InvalidTokenError(Exception):
    """Raised when the provided token_id has no matching snapshot in capture data."""


# ── Connection helper ─────────────────────────────────────────────────────────


def _connect(db_path: str) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection to the given path."""
    return duckdb.connect(db_path)


# ── Core functions ────────────────────────────────────────────────────────────


def get_unresolved(conn: duckdb.DuckDBPyConnection) -> list[MarketFightLink]:
    """Return all unresolved links (non-MATCHED, no reviewer).

    This is a local stub matching the interface of report.matching.get_unresolved.
    It queries market_fight_links directly for rows awaiting human review.
    """
    rows = conn.execute(
        """
        SELECT fight_url, token_id, match_status, match_method,
               candidate_count, matched_at, reviewed_by
        FROM market_fight_links
        WHERE match_status != 'MATCHED'
          AND reviewed_by IS NULL
        ORDER BY matched_at
        """,
    ).fetchall()

    return [
        MarketFightLink(
            fight_url=r[0],
            token_id=r[1],
            match_status=MatchStatus(r[2]),
            match_method=MatchMethod(r[3]) if r[3] else None,
            candidate_count=r[4],
            matched_at=r[5],
            reviewed_by=r[6],
        )
        for r in rows
    ]


def display_unresolved(links: list[MarketFightLink]) -> None:
    """Print unresolved links to stdout, grouped by matched_at date.

    Each group header shows the date; rows show fight_url, token_id, and status.
    """
    if not links:
        print("No unresolved links to review.")
        return

    sorted_links = sorted(links, key=lambda lnk: lnk.matched_at)
    grouped = groupby(sorted_links, key=lambda lnk: lnk.matched_at.date())

    for event_date, group in grouped:
        print(f"\n── {event_date} ──")
        for link in group:
            print(
                f"  fight_url: {link.fight_url}  "
                f"token_id: {link.token_id}  "
                f"status: {link.match_status.value}"
            )


def confirm_link(
    fight_url: str,
    token_id: str,
    reviewer: str,
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """Confirm an unresolved link with a validated token.

    Validates that the token exists in order_book_snapshots, then updates
    the link row to MATCHED with HUMAN_CONFIRMED method.

    Raises InvalidTokenError if token has no snapshot.
    Raises ValueError if fight_url is not found in the links table.
    """
    existing = conn.execute(
        "SELECT fight_url FROM market_fight_links WHERE fight_url = ?",
        [fight_url],
    ).fetchone()

    if existing is None:
        msg = f"Fight URL not found in market_fight_links: {fight_url!r}"
        raise ValueError(msg)

    token_exists = conn.execute(
        "SELECT 1 FROM order_book_snapshots WHERE token_id = ? LIMIT 1",
        [token_id],
    ).fetchone()

    if token_exists is None:
        msg = f"Token ID not found in active market data: {token_id!r}"
        raise InvalidTokenError(msg)

    now = datetime.now()
    conn.execute(
        """
        UPDATE market_fight_links
        SET token_id = ?,
            match_status = ?,
            match_method = ?,
            reviewed_by = ?,
            matched_at = ?
        WHERE fight_url = ?
        """,
        [
            token_id,
            MatchStatus.MATCHED.value,
            MatchMethod.HUMAN_CONFIRMED.value,
            reviewer,
            now,
            fight_url,
        ],
    )


# ── CLI entry point ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Build argparse parser for the confirmation CLI."""
    parser = argparse.ArgumentParser(
        prog="confirm_cli",
        description="Review and confirm unresolved market-fight links.",
    )
    parser.add_argument(
        "--db",
        default="data/ufc_edge.duckdb",
        help="Path to the DuckDB database file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="Display all unresolved links.")

    confirm_parser = subparsers.add_parser(
        "confirm", help="Confirm a link with a valid token."
    )
    confirm_parser.add_argument(
        "--fight-url", required=True, help="Fight URL to confirm."
    )
    confirm_parser.add_argument(
        "--token-id", required=True, help="Valid Polymarket token ID."
    )
    confirm_parser.add_argument(
        "--reviewer", required=True, help="Reviewer alias."
    )

    return parser


def main() -> None:
    """CLI entry point for human confirmation of market-fight links."""
    parser = _build_parser()
    args = parser.parse_args()

    conn = _connect(args.db)

    if args.command == "list":
        links = get_unresolved(conn)
        display_unresolved(links)

    elif args.command == "confirm":
        try:
            confirm_link(
                fight_url=args.fight_url,
                token_id=args.token_id,
                reviewer=args.reviewer,
                conn=conn,
            )
            print(f"Confirmed: {args.fight_url} → {args.token_id}")
        except InvalidTokenError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
