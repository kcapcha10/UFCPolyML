"""DuckDB connection management and idempotent write helpers.

All writes use INSERT ... ON CONFLICT DO UPDATE so re-running a crawl or
capture tick never produces duplicate rows. Every table carries a scraped_at
(or captured_at) timestamp column for provenance.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import duckdb

from ufc_edge.data.schemas import (
    Event,
    Fight,
    Fighter,
    FightTotals,
    MarketInfo,
    OrderBookSnapshot,
    RoundStats,
    SigStrikeBreakdown,
)

DEFAULT_DB_PATH = Path("data/ufc_edge.duckdb")

# ── DDL ───────────────────────────────────────────────────────────────────────

_CREATE_EVENTS = """
CREATE TABLE IF NOT EXISTS events (
    event_url   VARCHAR PRIMARY KEY,
    name        VARCHAR NOT NULL,
    date        DATE    NOT NULL,
    location    VARCHAR,
    scraped_at  TIMESTAMP NOT NULL
);
"""

_CREATE_FIGHTERS = """
CREATE TABLE IF NOT EXISTS fighters (
    fighter_url   VARCHAR PRIMARY KEY,
    name          VARCHAR NOT NULL,
    height_cm     DOUBLE,
    weight_kg     DOUBLE,
    reach_cm      DOUBLE,
    stance        VARCHAR,
    date_of_birth DATE,
    scraped_at    TIMESTAMP NOT NULL
);
"""

_CREATE_FIGHTS = """
CREATE TABLE IF NOT EXISTS fights (
    fight_url    VARCHAR PRIMARY KEY,
    event_url    VARCHAR NOT NULL,
    fighter_a_url VARCHAR NOT NULL,
    fighter_b_url VARCHAR NOT NULL,
    winner_url   VARCHAR,
    method       VARCHAR NOT NULL,
    ending_round INTEGER NOT NULL,
    ending_time  VARCHAR NOT NULL,
    time_format  VARCHAR NOT NULL,
    referee      VARCHAR,
    weight_class VARCHAR,
    scraped_at   TIMESTAMP NOT NULL
);
"""

_CREATE_FIGHT_TOTALS = """
CREATE TABLE IF NOT EXISTS fight_totals (
    fight_url                    VARCHAR NOT NULL,
    fighter_url                  VARCHAR NOT NULL,
    knockdowns                   INTEGER DEFAULT 0,
    significant_strikes_landed   INTEGER DEFAULT 0,
    significant_strikes_attempted INTEGER DEFAULT 0,
    total_strikes_landed         INTEGER DEFAULT 0,
    total_strikes_attempted      INTEGER DEFAULT 0,
    takedowns_landed             INTEGER DEFAULT 0,
    takedowns_attempted          INTEGER DEFAULT 0,
    submission_attempts          INTEGER DEFAULT 0,
    reversals                    INTEGER DEFAULT 0,
    control_time_seconds         INTEGER,
    scraped_at                   TIMESTAMP NOT NULL,
    PRIMARY KEY (fight_url, fighter_url)
);
"""

_CREATE_ROUND_STATS = """
CREATE TABLE IF NOT EXISTS round_stats (
    fight_url                    VARCHAR NOT NULL,
    fighter_url                  VARCHAR NOT NULL,
    round                        INTEGER NOT NULL,
    knockdowns                   INTEGER DEFAULT 0,
    significant_strikes_landed   INTEGER DEFAULT 0,
    significant_strikes_attempted INTEGER DEFAULT 0,
    total_strikes_landed         INTEGER DEFAULT 0,
    total_strikes_attempted      INTEGER DEFAULT 0,
    takedowns_landed             INTEGER DEFAULT 0,
    takedowns_attempted          INTEGER DEFAULT 0,
    submission_attempts          INTEGER DEFAULT 0,
    reversals                    INTEGER DEFAULT 0,
    control_time_seconds         INTEGER,
    scraped_at                   TIMESTAMP NOT NULL,
    PRIMARY KEY (fight_url, fighter_url, round)
);
"""

_CREATE_SIG_STRIKE_BREAKDOWNS = """
CREATE TABLE IF NOT EXISTS sig_strike_breakdowns (
    fight_url        VARCHAR NOT NULL,
    fighter_url      VARCHAR NOT NULL,
    round            INTEGER,
    head_landed      INTEGER DEFAULT 0,
    head_attempted   INTEGER DEFAULT 0,
    body_landed      INTEGER DEFAULT 0,
    body_attempted   INTEGER DEFAULT 0,
    leg_landed       INTEGER DEFAULT 0,
    leg_attempted    INTEGER DEFAULT 0,
    distance_landed  INTEGER DEFAULT 0,
    distance_attempted INTEGER DEFAULT 0,
    clinch_landed    INTEGER DEFAULT 0,
    clinch_attempted INTEGER DEFAULT 0,
    ground_landed    INTEGER DEFAULT 0,
    ground_attempted INTEGER DEFAULT 0,
    scraped_at       TIMESTAMP NOT NULL,
    PRIMARY KEY (fight_url, fighter_url, round)
);
"""

_CREATE_ORDER_BOOK_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS order_book_snapshots (
    market_id    VARCHAR NOT NULL,
    token_id     VARCHAR NOT NULL,
    question     VARCHAR,
    outcome      VARCHAR,
    bids         JSON    NOT NULL,
    asks         JSON    NOT NULL,
    mid_price    DOUBLE,
    spread       DOUBLE,
    captured_at  TIMESTAMP NOT NULL,
    PRIMARY KEY (token_id, captured_at)
);
"""

_ALL_DDL = [
    _CREATE_EVENTS,
    _CREATE_FIGHTERS,
    _CREATE_FIGHTS,
    _CREATE_FIGHT_TOTALS,
    _CREATE_ROUND_STATS,
    _CREATE_SIG_STRIKE_BREAKDOWNS,
    _CREATE_ORDER_BOOK_SNAPSHOTS,
]


# ── Connection ────────────────────────────────────────────────────────────────


def db_path() -> Path:
    """Resolve the DuckDB file path from env or default."""
    raw = os.environ.get("DUCKDB_PATH", "")
    return Path(raw) if raw else DEFAULT_DB_PATH


@contextmanager
def get_connection(path: Path | None = None) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Open a DuckDB connection and ensure all tables exist.

    Yields the connection; caller should not close it — the context manager
    handles that. Creates the database file and parent directories on first use.
    """
    resolved = path or db_path()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(resolved))
    try:
        _ensure_tables(conn)
        yield conn
    finally:
        conn.close()


def _ensure_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all tables if they do not already exist."""
    for ddl in _ALL_DDL:
        conn.execute(ddl)


# ── Upsert helpers ────────────────────────────────────────────────────────────


def upsert_event(conn: duckdb.DuckDBPyConnection, event: Event) -> None:
    conn.execute(
        """
        INSERT INTO events (event_url, name, date, location, scraped_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (event_url) DO UPDATE SET
            name       = EXCLUDED.name,
            date       = EXCLUDED.date,
            location   = EXCLUDED.location,
            scraped_at = EXCLUDED.scraped_at
        """,
        [event.event_url, event.name, event.date, event.location, event.scraped_at],
    )


def upsert_fighter(conn: duckdb.DuckDBPyConnection, fighter: Fighter) -> None:
    conn.execute(
        """
        INSERT INTO fighters
            (fighter_url, name, height_cm, weight_kg, reach_cm, stance, date_of_birth, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (fighter_url) DO UPDATE SET
            name          = EXCLUDED.name,
            height_cm     = EXCLUDED.height_cm,
            weight_kg     = EXCLUDED.weight_kg,
            reach_cm      = EXCLUDED.reach_cm,
            stance        = EXCLUDED.stance,
            date_of_birth = EXCLUDED.date_of_birth,
            scraped_at    = EXCLUDED.scraped_at
        """,
        [
            fighter.fighter_url,
            fighter.name,
            fighter.height_cm,
            fighter.weight_kg,
            fighter.reach_cm,
            fighter.stance,
            fighter.date_of_birth,
            fighter.scraped_at,
        ],
    )


def upsert_fight(conn: duckdb.DuckDBPyConnection, fight: Fight) -> None:
    conn.execute(
        """
        INSERT INTO fights
            (fight_url, event_url, fighter_a_url, fighter_b_url, winner_url,
             method, ending_round, ending_time, time_format, referee, weight_class, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (fight_url) DO UPDATE SET
            winner_url   = EXCLUDED.winner_url,
            method       = EXCLUDED.method,
            ending_round = EXCLUDED.ending_round,
            ending_time  = EXCLUDED.ending_time,
            time_format  = EXCLUDED.time_format,
            referee      = EXCLUDED.referee,
            weight_class = EXCLUDED.weight_class,
            scraped_at   = EXCLUDED.scraped_at
        """,
        [
            fight.fight_url,
            fight.event_url,
            fight.fighter_a_url,
            fight.fighter_b_url,
            fight.winner_url,
            fight.method,
            fight.ending_round,
            fight.ending_time,
            fight.time_format,
            fight.referee,
            fight.weight_class,
            fight.scraped_at,
        ],
    )


def upsert_fight_totals(conn: duckdb.DuckDBPyConnection, totals: FightTotals) -> None:
    conn.execute(
        """
        INSERT INTO fight_totals
            (fight_url, fighter_url, knockdowns,
             significant_strikes_landed, significant_strikes_attempted,
             total_strikes_landed, total_strikes_attempted,
             takedowns_landed, takedowns_attempted,
             submission_attempts, reversals, control_time_seconds, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (fight_url, fighter_url) DO UPDATE SET
            knockdowns                    = EXCLUDED.knockdowns,
            significant_strikes_landed    = EXCLUDED.significant_strikes_landed,
            significant_strikes_attempted = EXCLUDED.significant_strikes_attempted,
            total_strikes_landed          = EXCLUDED.total_strikes_landed,
            total_strikes_attempted       = EXCLUDED.total_strikes_attempted,
            takedowns_landed              = EXCLUDED.takedowns_landed,
            takedowns_attempted           = EXCLUDED.takedowns_attempted,
            submission_attempts           = EXCLUDED.submission_attempts,
            reversals                     = EXCLUDED.reversals,
            control_time_seconds          = EXCLUDED.control_time_seconds,
            scraped_at                    = EXCLUDED.scraped_at
        """,
        [
            totals.fight_url, totals.fighter_url, totals.knockdowns,
            totals.significant_strikes_landed, totals.significant_strikes_attempted,
            totals.total_strikes_landed, totals.total_strikes_attempted,
            totals.takedowns_landed, totals.takedowns_attempted,
            totals.submission_attempts, totals.reversals,
            totals.control_time_seconds, totals.scraped_at,
        ],
    )


def upsert_round_stats(conn: duckdb.DuckDBPyConnection, stats: RoundStats) -> None:
    conn.execute(
        """
        INSERT INTO round_stats
            (fight_url, fighter_url, round, knockdowns,
             significant_strikes_landed, significant_strikes_attempted,
             total_strikes_landed, total_strikes_attempted,
             takedowns_landed, takedowns_attempted,
             submission_attempts, reversals, control_time_seconds, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (fight_url, fighter_url, round) DO UPDATE SET
            knockdowns                    = EXCLUDED.knockdowns,
            significant_strikes_landed    = EXCLUDED.significant_strikes_landed,
            significant_strikes_attempted = EXCLUDED.significant_strikes_attempted,
            total_strikes_landed          = EXCLUDED.total_strikes_landed,
            total_strikes_attempted       = EXCLUDED.total_strikes_attempted,
            takedowns_landed              = EXCLUDED.takedowns_landed,
            takedowns_attempted           = EXCLUDED.takedowns_attempted,
            submission_attempts           = EXCLUDED.submission_attempts,
            reversals                     = EXCLUDED.reversals,
            control_time_seconds          = EXCLUDED.control_time_seconds,
            scraped_at                    = EXCLUDED.scraped_at
        """,
        [
            stats.fight_url, stats.fighter_url, stats.round, stats.knockdowns,
            stats.significant_strikes_landed, stats.significant_strikes_attempted,
            stats.total_strikes_landed, stats.total_strikes_attempted,
            stats.takedowns_landed, stats.takedowns_attempted,
            stats.submission_attempts, stats.reversals,
            stats.control_time_seconds, stats.scraped_at,
        ],
    )


def upsert_sig_strike_breakdown(
    conn: duckdb.DuckDBPyConnection, row: SigStrikeBreakdown
) -> None:
    conn.execute(
        """
        INSERT INTO sig_strike_breakdowns
            (fight_url, fighter_url, round,
             head_landed, head_attempted,
             body_landed, body_attempted,
             leg_landed, leg_attempted,
             distance_landed, distance_attempted,
             clinch_landed, clinch_attempted,
             ground_landed, ground_attempted, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (fight_url, fighter_url, round) DO UPDATE SET
            head_landed       = EXCLUDED.head_landed,
            head_attempted    = EXCLUDED.head_attempted,
            body_landed       = EXCLUDED.body_landed,
            body_attempted    = EXCLUDED.body_attempted,
            leg_landed        = EXCLUDED.leg_landed,
            leg_attempted     = EXCLUDED.leg_attempted,
            distance_landed   = EXCLUDED.distance_landed,
            distance_attempted = EXCLUDED.distance_attempted,
            clinch_landed     = EXCLUDED.clinch_landed,
            clinch_attempted  = EXCLUDED.clinch_attempted,
            ground_landed     = EXCLUDED.ground_landed,
            ground_attempted  = EXCLUDED.ground_attempted,
            scraped_at        = EXCLUDED.scraped_at
        """,
        [
            row.fight_url, row.fighter_url, row.round,
            row.head_landed, row.head_attempted,
            row.body_landed, row.body_attempted,
            row.leg_landed, row.leg_attempted,
            row.distance_landed, row.distance_attempted,
            row.clinch_landed, row.clinch_attempted,
            row.ground_landed, row.ground_attempted,
            row.scraped_at,
        ],
    )


def upsert_order_book_snapshot(
    conn: duckdb.DuckDBPyConnection, snapshot: OrderBookSnapshot
) -> None:
    import json

    bids_json = json.dumps([{"price": b.price, "size": b.size} for b in snapshot.bids])
    asks_json = json.dumps([{"price": a.price, "size": a.size} for a in snapshot.asks])

    conn.execute(
        """
        INSERT INTO order_book_snapshots
            (market_id, token_id, question, outcome, bids, asks,
             mid_price, spread, captured_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (token_id, captured_at) DO NOTHING
        """,
        [
            snapshot.market_id,
            snapshot.token_id,
            snapshot.question,
            snapshot.outcome,
            bids_json,
            asks_json,
            snapshot.mid_price,
            snapshot.spread,
            snapshot.captured_at,
        ],
    )
