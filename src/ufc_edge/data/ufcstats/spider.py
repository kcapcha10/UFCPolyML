"""ufcstats.com spider: events → fights → per-fight stats and fighter attributes.

The crawl reads like an outline: list events, then for each event parse its fights,
then for each fight parse stats and follow both fighters. All HTML parsing is
delegated to `parsers.py`; this module only handles request flow and de-duplication
of fighter pages within a run.
"""

from __future__ import annotations

from datetime import UTC, datetime

import scrapy

from ufc_edge.data.ufcstats import parsers
from ufc_edge.data.ufcstats.items import ItemKind


class UFCStatsSpider(scrapy.Spider):
    """Crawl completed UFC events and persist fights, stats, and fighters."""

    name = "ufcstats"

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self._seen_fighter_urls: set[str] = set()

    async def start(self):  # noqa: ANN201
        """Begin at the completed-events listing (Scrapy 2.13+ entry point)."""
        events_url = self.settings.get("UFCSTATS_EVENTS_URL")
        yield scrapy.Request(events_url, callback=self.parse_events_list)

    def parse_events_list(self, response):  # noqa: ANN001, ANN201
        """Follow each event-detail page found in the listing."""
        for event_url in parsers.parse_events_list(response.text):
            yield response.follow(event_url, callback=self.parse_event)

    def parse_event(self, response):  # noqa: ANN001, ANN201
        """Persist the event and follow each of its fights."""
        now = datetime.now(UTC)
        event, fight_urls = parsers.parse_event(response.text, response.url, now)
        yield {"kind": ItemKind.EVENT, "record": event}
        for fight_url in fight_urls:
            yield response.follow(
                fight_url,
                callback=self.parse_fight,
                cb_kwargs={"event_url": response.url},
            )

    def parse_fight(self, response, event_url: str):  # noqa: ANN001, ANN201
        """Persist the fight + stat rows, then follow both fighter pages."""
        now = datetime.now(UTC)
        fight, totals, rounds, sig = parsers.parse_fight(
            response.text, response.url, event_url, now
        )
        yield {"kind": ItemKind.FIGHT, "record": fight}
        yield {"kind": ItemKind.FIGHT_TOTALS, "record": totals}
        yield {"kind": ItemKind.ROUND_STATS, "record": rounds}
        yield {"kind": ItemKind.SIG_BREAKDOWN, "record": sig}

        for fighter_url in (fight.fighter_a_url, fight.fighter_b_url):
            if fighter_url and fighter_url not in self._seen_fighter_urls:
                self._seen_fighter_urls.add(fighter_url)
                yield response.follow(fighter_url, callback=self.parse_fighter)

    def parse_fighter(self, response):  # noqa: ANN001, ANN201
        """Persist a fighter's biographical attributes."""
        now = datetime.now(UTC)
        fighter = parsers.parse_fighter(response.text, response.url, now)
        yield {"kind": ItemKind.FIGHTER, "record": fighter}
