"""Polymarket UFC order-book capture cron.

Every CAPTURE_INTERVAL_SECONDS: enumerate active UFC markets via the Gamma
API, pull the live CLOB order book for each token, and write a timestamped
snapshot to DuckDB. One failed market never aborts a tick; every tick logs a
summary (markets seen, snapshots written, errors). Safe to restart at any
point — snapshot writes are idempotent on (token_id, captured_at).

Run modes:
    python -m ufc_edge.market.capture --once   # single tick (local verification)
    python -m ufc_edge.market.capture          # loop forever (deployed cron)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from omegaconf import OmegaConf
from tenacity import retry, stop_after_attempt, wait_exponential

from ufc_edge.data import storage
from ufc_edge.data.schemas import MarketInfo, OrderBookSnapshot, OrderLevel

logger = logging.getLogger("ufc_edge.capture")

CONFIG_PATH = Path("configs/capture/default.yaml")

# Fallbacks if the config file is absent (e.g. a stripped-down container).
DEFAULT_CAPTURE_INTERVAL_SECONDS = 300
DEFAULT_GAMMA_API_BASE_URL = "https://gamma-api.polymarket.com"
DEFAULT_CLOB_API_BASE_URL = "https://clob.polymarket.com"
DEFAULT_MARKET_TAG_SLUG = "ufc"
DEFAULT_ORDERBOOK_DEPTH = 10

GAMMA_PAGE_SIZE = 100
HTTP_TIMEOUT_SECONDS = 15.0
MAX_REQUEST_ATTEMPTS = 3
RETRY_BACKOFF_BASE_SECONDS = 2.0


@dataclass(frozen=True)
class CaptureConfig:
    """Runtime settings for the capture cron."""

    interval_seconds: int
    gamma_base_url: str
    clob_base_url: str
    tag_slug: str
    orderbook_depth: int


def load_config() -> CaptureConfig:
    """Load capture settings from configs/capture/default.yaml, with defaults."""
    overrides = OmegaConf.load(CONFIG_PATH) if CONFIG_PATH.exists() else OmegaConf.create()
    return CaptureConfig(
        interval_seconds=int(
            overrides.get("capture_interval_seconds", DEFAULT_CAPTURE_INTERVAL_SECONDS)
        ),
        gamma_base_url=str(overrides.get("gamma_api_base_url", DEFAULT_GAMMA_API_BASE_URL)),
        clob_base_url=str(overrides.get("clob_api_base_url", DEFAULT_CLOB_API_BASE_URL)),
        tag_slug=str(overrides.get("market_tag_slug", DEFAULT_MARKET_TAG_SLUG)),
        orderbook_depth=int(overrides.get("orderbook_depth", DEFAULT_ORDERBOOK_DEPTH)),
    )


# ── HTTP plumbing ─────────────────────────────────────────────────────────────


@retry(
    stop=stop_after_attempt(MAX_REQUEST_ATTEMPTS),
    wait=wait_exponential(multiplier=RETRY_BACKOFF_BASE_SECONDS),
    reraise=True,
)
def _request_json(client: httpx.Client, url: str, params: dict | None = None) -> object:
    """GET a JSON payload with exponential-backoff retries; raise on HTTP error."""
    response = client.get(url, params=params)
    response.raise_for_status()
    return response.json()


# ── Market enumeration (Gamma API) ────────────────────────────────────────────


def fetch_active_ufc_markets(client: httpx.Client, config: CaptureConfig) -> list[MarketInfo]:
    """Enumerate all (market, token) pairs for open UFC markets via Gamma /events.

    Returns one MarketInfo per outcome token, so a two-outcome fight market
    yields two entries. Paginates until the API returns a short page.
    """
    markets: list[MarketInfo] = []
    offset = 0
    while True:
        events = _request_json(
            client,
            f"{config.gamma_base_url}/events",
            params={
                "tag_slug": config.tag_slug,
                "closed": "false",
                "limit": GAMMA_PAGE_SIZE,
                "offset": offset,
            },
        )
        if not events:
            break
        for event in events:
            markets.extend(_parse_event_markets(event))
        if len(events) < GAMMA_PAGE_SIZE:
            break
        offset += GAMMA_PAGE_SIZE
    return markets


def _parse_event_markets(event: dict) -> list[MarketInfo]:
    """Extract per-token MarketInfo rows from one Gamma event payload.

    Gamma encodes clobTokenIds and outcomes as JSON strings inside the JSON
    payload (a known API quirk) — both must be json.loads'd before zipping.
    """
    parsed: list[MarketInfo] = []
    for market in event.get("markets", []):
        if not market.get("active", False) or market.get("closed", True):
            continue
        token_ids = _loads_if_string(market.get("clobTokenIds"))
        outcomes = _loads_if_string(market.get("outcomes"))
        if not token_ids or not outcomes:
            continue
        for token_id, outcome in zip(token_ids, outcomes, strict=False):
            parsed.append(
                MarketInfo(
                    market_id=market.get("conditionId") or str(market.get("id")),
                    question=market.get("question", ""),
                    token_id=token_id,
                    outcome=outcome,
                    active=market.get("active", False),
                    closed=market.get("closed", False),
                    end_date=market.get("endDate"),
                )
            )
    return parsed


def _loads_if_string(value: object) -> list | None:
    """Decode Gamma's JSON-string-encoded list fields; pass lists through."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return list(value)  # type: ignore[arg-type]


# ── Order-book capture (CLOB API) ─────────────────────────────────────────────


def fetch_order_book(
    client: httpx.Client, market: MarketInfo, config: CaptureConfig
) -> OrderBookSnapshot:
    """Pull the live CLOB book for one token and build a validated snapshot."""
    payload = _request_json(
        client,
        f"{config.clob_base_url}/book",
        params={"token_id": market.token_id},
    )
    bids = _top_levels(payload.get("bids", []), descending=True, depth=config.orderbook_depth)
    asks = _top_levels(payload.get("asks", []), descending=False, depth=config.orderbook_depth)
    mid_price, spread = _mid_and_spread(bids, asks)
    return OrderBookSnapshot(
        market_id=market.market_id,
        token_id=market.token_id,
        question=market.question,
        outcome=market.outcome,
        bids=bids,
        asks=asks,
        mid_price=mid_price,
        spread=spread,
        captured_at=datetime.now(UTC),
    )


def _top_levels(raw_levels: list[dict], descending: bool, depth: int) -> list[OrderLevel]:
    """Validate raw book levels and keep the best `depth` levels.

    Best bid = highest price (descending sort); best ask = lowest (ascending).
    """
    levels = [OrderLevel(price=level["price"], size=level["size"]) for level in raw_levels]
    levels.sort(key=lambda level: level.price, reverse=descending)
    return levels[:depth]


def _mid_and_spread(
    bids: list[OrderLevel], asks: list[OrderLevel]
) -> tuple[float | None, float | None]:
    """Compute mid price and spread from best bid/ask; None if a side is empty."""
    if not bids or not asks:
        return None, None
    best_bid, best_ask = bids[0].price, asks[0].price
    return (best_bid + best_ask) / 2, best_ask - best_bid


# ── Tick orchestration ────────────────────────────────────────────────────────


def run_tick(config: CaptureConfig) -> dict[str, int]:
    """Run one capture tick: enumerate markets, snapshot each book, write to DuckDB.

    A failure on one market is logged and skipped; the tick continues. Returns
    summary counts for logging and tests.
    """
    stats = {"markets_seen": 0, "snapshots_written": 0, "errors": 0}
    with (
        httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as client,
        storage.get_connection() as conn,
    ):
        try:
            markets = fetch_active_ufc_markets(client, config)
        except Exception:
            logger.exception("market enumeration failed; aborting tick")
            stats["errors"] += 1
            return stats

        stats["markets_seen"] = len(markets)
        for market in markets:
            try:
                snapshot = fetch_order_book(client, market, config)
                storage.upsert_order_book_snapshot(conn, snapshot)
                stats["snapshots_written"] += 1
            except Exception as error:
                stats["errors"] += 1
                logger.warning(
                    "snapshot failed token_id=%s question=%r error=%s",
                    market.token_id,
                    market.question,
                    error,
                )
    logger.info(
        "tick complete markets_seen=%d snapshots_written=%d errors=%d",
        stats["markets_seen"],
        stats["snapshots_written"],
        stats["errors"],
    )
    return stats


def run_forever(config: CaptureConfig) -> None:
    """Loop run_tick on the configured interval; a failed tick never kills the loop."""
    logger.info("capture cron started interval_seconds=%d", config.interval_seconds)
    while True:
        tick_started = time.monotonic()
        try:
            run_tick(config)
        except Exception:
            logger.exception("tick crashed; will retry next interval")
        elapsed = time.monotonic() - tick_started
        time.sleep(max(0.0, config.interval_seconds - elapsed))


def main() -> None:
    """CLI entry point: --once for a single verification tick, default loops."""
    parser = argparse.ArgumentParser(description="Polymarket UFC order-book capture")
    parser.add_argument("--once", action="store_true", help="run a single tick and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    config = load_config()
    if args.once:
        run_tick(config)
    else:
        run_forever(config)


if __name__ == "__main__":
    main()
