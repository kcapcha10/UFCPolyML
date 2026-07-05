"""Scrapy settings for the ufcstats crawl.

Politeness values are sourced from `configs/scrape/default.yaml` so the crawl's
rate limits live in one place (and are documented in docs/DECISIONS.md). Hard-coded
fallbacks apply if the config file is missing.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

_CONFIG_PATH = Path("configs/scrape/default.yaml")
_cfg = OmegaConf.load(_CONFIG_PATH) if _CONFIG_PATH.exists() else OmegaConf.create()

# ── Spider discovery ──────────────────────────────────────────────────────────
SPIDER_MODULES = ["ufc_edge.data.ufcstats"]
NEWSPIDER_MODULE = "ufc_edge.data.ufcstats"

# ── Identity & target ─────────────────────────────────────────────────────────
BOT_NAME = "ufc_edge"
USER_AGENT = _cfg.get("user_agent", "ufc-edge-research-bot/0.1 (academic; contact via repo)")
UFCSTATS_BASE_URL = _cfg.get("ufcstats_base_url", "http://www.ufcstats.com")
UFCSTATS_EVENTS_URL = _cfg.get(
    "events_list_url", "http://www.ufcstats.com/statistics/events/completed?page=all"
)

# ── Politeness (see docs/DECISIONS.md: scraper rate limits) ────────────────────────
DOWNLOAD_DELAY = float(_cfg.get("download_delay", 2.0))
CONCURRENT_REQUESTS = int(_cfg.get("concurrent_requests", 1))
CONCURRENT_REQUESTS_PER_DOMAIN = int(_cfg.get("concurrent_requests", 1))
AUTOTHROTTLE_ENABLED = bool(_cfg.get("autothrottle_enabled", True))
AUTOTHROTTLE_TARGET_CONCURRENCY = float(_cfg.get("autothrottle_target_concurrency", 1.0))
AUTOTHROTTLE_START_DELAY = 2.0
ROBOTSTXT_OBEY = True

# ── Pipeline & middleware wiring ──────────────────────────────────────────────
ITEM_PIPELINES = {
    "ufc_edge.data.ufcstats.pipelines.DuckDBPipeline": 300,
}
DOWNLOADER_MIDDLEWARES = {
    "ufc_edge.data.ufcstats.middlewares.ChallengeSolverMiddleware": 550,
}

# ── Robustness ────────────────────────────────────────────────────────────────
RETRY_ENABLED = True
RETRY_TIMES = 3
COOKIES_ENABLED = True
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
LOG_LEVEL = "INFO"
