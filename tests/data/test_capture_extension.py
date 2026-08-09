"""Tests for bid/ask/depth extraction from the order book.

Verifies that the capture pipeline correctly denormalizes the top-of-book
price and size into dedicated fields, and that empty book sides produce NULL
values rather than errors — ensuring backward compatibility with markets that
have one-sided liquidity.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import duckdb

from ufc_edge.data.polymarket.capture import _top_of_side, fetch_order_book
from ufc_edge.data.polymarket.schemas import MarketInfo, OrderBookSnapshot, OrderLevel
from ufc_edge.data.polymarket.storage import (
    POLYMARKET_DDL,
    upsert_order_book_snapshot,
)

CAPTURED_AT = datetime(2026, 7, 15, 12, 0, 0)
TICK_ID = "test-tick-001"


# ── _top_of_side unit tests ──────────────────────────────────────────────────


def test_top_of_side_returns_best_level():
    levels = [OrderLevel(price=0.65, size=200.0), OrderLevel(price=0.60, size=100.0)]
    price, size = _top_of_side(levels)
    assert price == 0.65
    assert size == 200.0


def test_top_of_side_empty_book_returns_none():
    price, size = _top_of_side([])
    assert price is None
    assert size is None


# ── Snapshot construction with populated book ─────────────────────────────────


def test_snapshot_with_populated_book_extracts_top_of_book():
    snapshot = OrderBookSnapshot(
        market_id="market-1",
        token_id="token-1",
        question="Fighter A vs Fighter B",
        outcome="Fighter A",
        bids=[OrderLevel(price=0.55, size=150.0), OrderLevel(price=0.50, size=80.0)],
        asks=[OrderLevel(price=0.60, size=120.0), OrderLevel(price=0.65, size=50.0)],
        mid_price=0.575,
        spread=0.05,
        best_bid=0.55,
        best_ask=0.60,
        best_bid_size=150.0,
        best_ask_size=120.0,
        captured_at=CAPTURED_AT,
        tick_id=TICK_ID,
    )
    assert snapshot.best_bid == 0.55
    assert snapshot.best_ask == 0.60
    assert snapshot.best_bid_size == 150.0
    assert snapshot.best_ask_size == 120.0


def test_snapshot_with_empty_book_has_null_top_of_book():
    snapshot = OrderBookSnapshot(
        market_id="market-1",
        token_id="token-1",
        bids=[],
        asks=[],
        mid_price=None,
        spread=None,
        best_bid=None,
        best_ask=None,
        best_bid_size=None,
        best_ask_size=None,
        captured_at=CAPTURED_AT,
        tick_id=TICK_ID,
    )
    assert snapshot.best_bid is None
    assert snapshot.best_ask is None
    assert snapshot.best_bid_size is None
    assert snapshot.best_ask_size is None


def test_snapshot_with_one_sided_book():
    """Only bids present, no asks — ask-side fields are NULL."""
    snapshot = OrderBookSnapshot(
        market_id="market-1",
        token_id="token-1",
        bids=[OrderLevel(price=0.45, size=300.0)],
        asks=[],
        mid_price=None,
        spread=None,
        best_bid=0.45,
        best_ask=None,
        best_bid_size=300.0,
        best_ask_size=None,
        captured_at=CAPTURED_AT,
        tick_id=TICK_ID,
    )
    assert snapshot.best_bid == 0.45
    assert snapshot.best_bid_size == 300.0
    assert snapshot.best_ask is None
    assert snapshot.best_ask_size is None


# ── DDL migration adds columns without breaking existing schema ───────────────


def test_ddl_migration_adds_columns_to_existing_table():
    """Columns are added via ALTER TABLE IF NOT EXISTS — idempotent on fresh DB."""
    conn = duckdb.connect(":memory:")
    for stmt in POLYMARKET_DDL:
        conn.execute(stmt)

    # Verify columns exist by inserting a row with values for the new columns.
    conn.execute(
        """
        INSERT INTO order_book_snapshots
            (market_id, token_id, bids, asks, mid_price, spread,
             best_bid, best_ask, best_bid_size, best_ask_size,
             captured_at, tick_id)
        VALUES ('m', 't', '[]', '[]', 0.5, 0.1, 0.45, 0.55, 100.0, 80.0,
                '2026-07-15 12:00:00', 'tick-1')
        """
    )
    row = conn.execute(
        "SELECT best_bid, best_ask, best_bid_size, best_ask_size "
        "FROM order_book_snapshots WHERE token_id = 't'"
    ).fetchone()
    assert row == (0.45, 0.55, 100.0, 80.0)
    conn.close()


def test_ddl_migration_existing_rows_have_null_for_new_columns():
    """Legacy rows predate the new columns and retain NULL defaults."""
    conn = duckdb.connect(":memory:")
    # Simulate a legacy DB: create table without new columns first.
    conn.execute(
        """
        CREATE TABLE order_book_snapshots (
            market_id    VARCHAR NOT NULL,
            token_id     VARCHAR NOT NULL,
            question     VARCHAR,
            outcome      VARCHAR,
            bids         JSON NOT NULL,
            asks         JSON NOT NULL,
            mid_price    DOUBLE,
            spread       DOUBLE,
            captured_at  TIMESTAMP NOT NULL,
            PRIMARY KEY (token_id, captured_at)
        );
        """
    )
    # Insert a legacy row before migration.
    conn.execute(
        """
        INSERT INTO order_book_snapshots
            (market_id, token_id, bids, asks, mid_price, spread, captured_at)
        VALUES ('m', 't', '[]', '[]', 0.5, 0.1, '2026-06-01 10:00:00')
        """
    )
    # Now run migrations (tick_id + new columns).
    conn.execute(
        "ALTER TABLE order_book_snapshots ADD COLUMN IF NOT EXISTS tick_id VARCHAR"
    )
    conn.execute(
        "ALTER TABLE order_book_snapshots ADD COLUMN IF NOT EXISTS best_bid DOUBLE"
    )
    conn.execute(
        "ALTER TABLE order_book_snapshots ADD COLUMN IF NOT EXISTS best_ask DOUBLE"
    )
    conn.execute(
        "ALTER TABLE order_book_snapshots ADD COLUMN IF NOT EXISTS best_bid_size DOUBLE"
    )
    conn.execute(
        "ALTER TABLE order_book_snapshots ADD COLUMN IF NOT EXISTS best_ask_size DOUBLE"
    )

    row = conn.execute(
        "SELECT best_bid, best_ask, best_bid_size, best_ask_size "
        "FROM order_book_snapshots WHERE token_id = 't'"
    ).fetchone()
    assert row == (None, None, None, None)
    conn.close()


# ── Storage round-trip with new columns ───────────────────────────────────────


def test_upsert_persists_best_bid_ask_fields():
    """Full round-trip: Pydantic model → upsert → DuckDB → verify new columns."""
    conn = duckdb.connect(":memory:")
    for stmt in POLYMARKET_DDL:
        conn.execute(stmt)

    snapshot = OrderBookSnapshot(
        market_id="market-1",
        token_id="token-1",
        question="A vs B",
        outcome="A",
        bids=[OrderLevel(price=0.55, size=150.0)],
        asks=[OrderLevel(price=0.60, size=120.0)],
        mid_price=0.575,
        spread=0.05,
        best_bid=0.55,
        best_ask=0.60,
        best_bid_size=150.0,
        best_ask_size=120.0,
        captured_at=CAPTURED_AT,
        tick_id=TICK_ID,
    )
    upsert_order_book_snapshot(conn, snapshot)

    row = conn.execute(
        "SELECT best_bid, best_ask, best_bid_size, best_ask_size "
        "FROM order_book_snapshots WHERE token_id = 'token-1'"
    ).fetchone()
    assert row == (0.55, 0.60, 150.0, 120.0)
    conn.close()


def test_upsert_persists_null_when_book_empty():
    """Empty book → NULL for all top-of-book columns in DuckDB."""
    conn = duckdb.connect(":memory:")
    for stmt in POLYMARKET_DDL:
        conn.execute(stmt)

    snapshot = OrderBookSnapshot(
        market_id="market-1",
        token_id="token-2",
        bids=[],
        asks=[],
        mid_price=None,
        spread=None,
        best_bid=None,
        best_ask=None,
        best_bid_size=None,
        best_ask_size=None,
        captured_at=CAPTURED_AT,
        tick_id=TICK_ID,
    )
    upsert_order_book_snapshot(conn, snapshot)

    row = conn.execute(
        "SELECT best_bid, best_ask, best_bid_size, best_ask_size "
        "FROM order_book_snapshots WHERE token_id = 'token-2'"
    ).fetchone()
    assert row == (None, None, None, None)
    conn.close()


# ── fetch_order_book integration (mocked HTTP) ───────────────────────────────


def test_fetch_order_book_populates_best_bid_ask():
    """fetch_order_book extracts bids[0] and asks[0] into dedicated fields."""
    from ufc_edge.data.polymarket.capture import CaptureConfig

    config = CaptureConfig(
        interval_seconds=300,
        gamma_base_url="http://fake",
        clob_base_url="http://fake",
        tag_slug="ufc",
        orderbook_depth=5,
    )
    market = MarketInfo(
        market_id="cond-1",
        question="A vs B",
        token_id="tok-1",
        outcome="A",
        active=True,
        closed=False,
    )
    api_response = {
        "bids": [
            {"price": 0.55, "size": 200.0},
            {"price": 0.50, "size": 100.0},
        ],
        "asks": [
            {"price": 0.60, "size": 150.0},
            {"price": 0.65, "size": 75.0},
        ],
    }

    with patch(
        "ufc_edge.data.polymarket.capture._request_json", return_value=api_response
    ):
        snapshot = fetch_order_book(None, market, config, "tick-abc")

    assert snapshot.best_bid == 0.55
    assert snapshot.best_ask == 0.60
    assert snapshot.best_bid_size == 200.0
    assert snapshot.best_ask_size == 150.0


def test_fetch_order_book_empty_book_yields_null_fields():
    """Empty API response → None for all top-of-book fields."""
    from ufc_edge.data.polymarket.capture import CaptureConfig

    config = CaptureConfig(
        interval_seconds=300,
        gamma_base_url="http://fake",
        clob_base_url="http://fake",
        tag_slug="ufc",
        orderbook_depth=5,
    )
    market = MarketInfo(
        market_id="cond-1",
        question="A vs B",
        token_id="tok-1",
        outcome="A",
        active=True,
        closed=False,
    )
    api_response = {"bids": [], "asks": []}

    with patch(
        "ufc_edge.data.polymarket.capture._request_json", return_value=api_response
    ):
        snapshot = fetch_order_book(None, market, config, "tick-empty")

    assert snapshot.best_bid is None
    assert snapshot.best_ask is None
    assert snapshot.best_bid_size is None
    assert snapshot.best_ask_size is None
