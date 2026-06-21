"""Scrapy item wrapper carrying validated Pydantic models to the pipeline.

The spider yields plain dicts tagged by `kind`; each holds an already-validated
Pydantic model (or list of models). Validation happens in the parsers, so the
pipeline only has to route models to the correct DuckDB upsert.
"""

from __future__ import annotations

from enum import Enum


class ItemKind(str, Enum):
    """Discriminator for the kind of parsed record flowing to the pipeline."""

    EVENT = "event"
    FIGHTER = "fighter"
    FIGHT = "fight"
    FIGHT_TOTALS = "fight_totals"
    ROUND_STATS = "round_stats"
    SIG_BREAKDOWN = "sig_breakdown"
