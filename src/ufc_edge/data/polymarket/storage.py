"""DuckDB DDL and idempotent writes for Polymarket order-book snapshots.

Snapshot writes use ON CONFLICT DO NOTHING because a given (token_id, captured_at)
pair is immutable once written — the book at a point in time cannot change.

`tick_id` groups every snapshot written by one capture tick into a coherent
cross-section (a tick spans ~90s of per-market fetches). Rows captured before
2026-07-04 predate the column and carry NULL — group those by time-bucketing
`captured_at`, and treat any bucket-boundary ambiguity as a known limitation.
"""

from __future__ import annotations

import json

import duckdb

from ufc_edge.data.polymarket.schemas import OrderBookSnapshot

# ── DDL ───────────────────────────────────────────────────────────────────────

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
    tick_id      VARCHAR,
    PRIMARY KEY (token_id, captured_at)
);
"""

# Adds tick_id to databases created before the column existed; no-op on fresh
# tables (which already include it, keeping column order identical either way).
_MIGRATE_ADD_TICK_ID = """
ALTER TABLE order_book_snapshots ADD COLUMN IF NOT EXISTS tick_id VARCHAR;
"""

# Adds top-of-book price and size columns for future fillability analysis.
# Existing rows retain NULL — the book data was already captured in the JSON
# columns but not denormalized until this migration.
_MIGRATE_ADD_BEST_BID = """
ALTER TABLE order_book_snapshots ADD COLUMN IF NOT EXISTS best_bid DOUBLE;
"""
_MIGRATE_ADD_BEST_ASK = """
ALTER TABLE order_book_snapshots ADD COLUMN IF NOT EXISTS best_ask DOUBLE;
"""
_MIGRATE_ADD_BEST_BID_SIZE = """
ALTER TABLE order_book_snapshots ADD COLUMN IF NOT EXISTS best_bid_size DOUBLE;
"""
_MIGRATE_ADD_BEST_ASK_SIZE = """
ALTER TABLE order_book_snapshots ADD COLUMN IF NOT EXISTS best_ask_size DOUBLE;
"""

POLYMARKET_DDL: list[str] = [
    _CREATE_ORDER_BOOK_SNAPSHOTS,
    _MIGRATE_ADD_TICK_ID,
    _MIGRATE_ADD_BEST_BID,
    _MIGRATE_ADD_BEST_ASK,
    _MIGRATE_ADD_BEST_BID_SIZE,
    _MIGRATE_ADD_BEST_ASK_SIZE,
]

# ── Upserts ───────────────────────────────────────────────────────────────────


def upsert_order_book_snapshot(
    conn: duckdb.DuckDBPyConnection, snapshot: OrderBookSnapshot
) -> None:
    bids_json = json.dumps([{"price": b.price, "size": b.size} for b in snapshot.bids])
    asks_json = json.dumps([{"price": a.price, "size": a.size} for a in snapshot.asks])

    conn.execute(
        """
        INSERT INTO order_book_snapshots
            (market_id, token_id, question, outcome, bids, asks,
             mid_price, spread, best_bid, best_ask, best_bid_size,
             best_ask_size, captured_at, tick_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            snapshot.best_bid,
            snapshot.best_ask,
            snapshot.best_bid_size,
            snapshot.best_ask_size,
            snapshot.captured_at,
            snapshot.tick_id,
        ],
    )
