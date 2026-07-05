"""Pydantic models for Polymarket entities.

Every model corresponds to a DuckDB table or a validated API response shape.
Validate at ingest; fail loudly on schema drift.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator

from ufc_edge.data.schemas import _FrozenModel


class OrderLevel(_FrozenModel):
    """A single price level in an order book."""

    price: float
    size: float

    @field_validator("price")
    @classmethod
    def price_in_range(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"Polymarket prices must be in (0, 1), got {v}")
        return v

    @field_validator("size")
    @classmethod
    def size_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Order size must be non-negative, got {v}")
        return v


class OrderBookSnapshot(_FrozenModel):
    """A timestamped order-book capture for one Polymarket token."""

    market_id: str = Field(description="Polymarket condition_id / market ID")
    token_id: str = Field(description="Polymarket token/outcome ID")
    question: str | None = None
    outcome: str | None = None
    bids: list[OrderLevel]
    asks: list[OrderLevel]
    mid_price: float | None = None
    spread: float | None = None
    captured_at: datetime
    tick_id: str = Field(
        description="UUID shared by every snapshot written in the same capture tick"
    )


class MarketInfo(_FrozenModel):
    """Metadata for a single Polymarket market as returned by the Gamma API."""

    market_id: str
    question: str
    token_id: str
    outcome: str
    active: bool
    closed: bool
    end_date: datetime | None = None
