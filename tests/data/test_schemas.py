"""Schema validation tests: malformed rows must fail loudly at the boundary."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from ufc_edge.data.polymarket.schemas import OrderBookSnapshot, OrderLevel
from ufc_edge.data.ufcstats.schemas import Event

SCRAPED_AT = datetime(2026, 6, 13)


def test_event_requires_scraped_at():
    with pytest.raises(ValidationError):
        Event(event_url="u", name="n", date="2026-01-01")  # missing scraped_at


def test_event_rejects_unparseable_date():
    with pytest.raises(ValidationError):
        Event(event_url="u", name="n", date="not-a-date", scraped_at=SCRAPED_AT)


def test_frozen_model_is_immutable():
    event = Event(event_url="u", name="n", date="2026-01-01", scraped_at=SCRAPED_AT)
    with pytest.raises(ValidationError):
        event.name = "mutated"


def test_order_level_rejects_price_out_of_range():
    with pytest.raises(ValidationError):
        OrderLevel(price=1.5, size=10)  # Polymarket prices live in (0, 1)
    with pytest.raises(ValidationError):
        OrderLevel(price=0.0, size=10)


def test_order_level_rejects_negative_size():
    with pytest.raises(ValidationError):
        OrderLevel(price=0.5, size=-1)


def test_order_book_snapshot_validates_nested_levels():
    snap = OrderBookSnapshot(
        market_id="m",
        token_id="t",
        bids=[OrderLevel(price=0.4, size=100)],
        asks=[OrderLevel(price=0.6, size=80)],
        mid_price=0.5,
        spread=0.2,
        captured_at=SCRAPED_AT,
    )
    assert snap.bids[0].price == 0.4
    with pytest.raises(ValidationError):
        OrderBookSnapshot(
            market_id="m",
            token_id="t",
            bids=[{"price": 9.9, "size": 1}],  # invalid nested level
            asks=[],
            captured_at=SCRAPED_AT,
        )
