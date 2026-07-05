"""Scrapy downloader middleware that clears the ufcstats PoW challenge.

When a response is the challenge interstitial, the middleware solves it (see
`challenge.py`), caches the clearance cookies on the spider, and retries the
original request with those cookies attached. The solve is a brief synchronous
call; acceptable here because the crawl runs at concurrency 1 and the challenge
is rare (cookie is reused until it expires). See docs/DECISIONS.md.
"""

from __future__ import annotations

import logging

from scrapy import Request
from scrapy.http import Response

from ufc_edge.data.ufcstats import challenge

logger = logging.getLogger(__name__)


class ChallengeSolverMiddleware:
    """Detect the PoW interstitial and retry with clearance cookies."""

    def __init__(self, base_url: str, user_agent: str) -> None:
        self._base_url = base_url
        self._user_agent = user_agent
        self._cookies: dict[str, str] = {}

    @classmethod
    def from_crawler(cls, crawler):  # noqa: ANN001, ANN206 - Scrapy hook signature
        settings = crawler.settings
        return cls(
            base_url=settings.get("UFCSTATS_BASE_URL", "http://www.ufcstats.com"),
            user_agent=settings.get("USER_AGENT", "ufc-edge-research-bot/0.1"),
        )

    def process_request(self, request: Request, spider):  # noqa: ANN001, ANN201
        """Attach any cached clearance cookies to outgoing requests."""
        if self._cookies:
            request.cookies.update(self._cookies)
        return None

    def process_response(self, request: Request, response: Response, spider):  # noqa: ANN001, ANN201
        """If the response is the challenge, solve it and retry the request."""
        body = response.text
        if not challenge.is_challenge(body):
            return response
        logger.info("ufcstats challenge detected; solving proof-of-work")
        self._cookies = challenge.obtain_clearance_cookies(self._base_url, self._user_agent)
        if not self._cookies:
            logger.warning("challenge solve produced no clearance cookie; passing response through")
            return response
        retried = request.replace(cookies=self._cookies, dont_filter=True)
        return retried
