"""DuckDB DDL and idempotent writes for Polymarket order-book snapshots.

Snapshot writes use ON CONFLICT DO NOTHING because a given (token_id, captured_at)
pair is immutable once written — the book at a point in time cannot change.
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
    PRIMARY KEY (token_id, captured_at)
);
"""

POLYMARKET_DDL: list[str] = [
    _CREATE_ORDER_BOOK_SNAPSHOTS,
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
