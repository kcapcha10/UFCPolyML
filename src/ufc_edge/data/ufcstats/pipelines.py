"""Scrapy pipeline: route parsed Pydantic models to idempotent DuckDB upserts.

The spider yields dicts of the form {"kind": ItemKind, "record": model_or_list}.
This pipeline owns a single DuckDB connection for the crawl and dispatches each
record to the matching upsert helper in `data/ufcstats/storage.py`.
"""

from __future__ import annotations

from ufc_edge.data import storage as db
from ufc_edge.data.ufcstats import storage
from ufc_edge.data.ufcstats.items import ItemKind


class DuckDBPipeline:
    """Persist validated records to DuckDB via idempotent upserts."""

    def __init__(self) -> None:
        self._cm = None
        self._conn = None

    def open_spider(self, spider) -> None:  # noqa: ANN001 - Scrapy hook signature
        """Open the DuckDB connection for the lifetime of the crawl."""
        self._cm = db.get_connection()
        self._conn = self._cm.__enter__()

    def close_spider(self, spider) -> None:  # noqa: ANN001 - Scrapy hook signature
        """Close the DuckDB connection at the end of the crawl."""
        if self._cm is not None:
            self._cm.__exit__(None, None, None)

    def process_item(self, item: dict, spider):  # noqa: ANN001, ANN201
        """Dispatch one parsed record (or list of records) to its upsert."""
        kind = item["kind"]
        record = item["record"]
        dispatch = {
            ItemKind.EVENT: storage.upsert_event,
            ItemKind.FIGHTER: storage.upsert_fighter,
            ItemKind.FIGHT: storage.upsert_fight,
            ItemKind.FIGHT_TOTALS: storage.upsert_fight_totals,
            ItemKind.ROUND_STATS: storage.upsert_round_stats,
            ItemKind.SIG_BREAKDOWN: storage.upsert_sig_strike_breakdown,
        }
        upsert = dispatch[kind]
        for model in record if isinstance(record, list) else [record]:
            upsert(self._conn, model)
        return item
