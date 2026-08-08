"""Synthetic feature-table generator for model/eval testing.

Produces a DuckDB with 50 fight rows across 10 events, representative columns
from each feature family in the registry (physical, activity, record, output,
graph), deliberate NULL/NaN patterns, draw/NC outcomes for exclusion testing,
and one market-derived column to verify rejection logic.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import duckdb

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

WEIGHT_CLASSES = [
    "Lightweight",
    "Welterweight",
    "Middleweight",
    "Light Heavyweight",
    "Heavyweight",
    "Featherweight",
    "Bantamweight",
]

OUTCOMES = ["win_a", "win_b"]
SPECIAL_OUTCOMES = ["draw", "nc"]

# Market-derived column that must be rejected by the model assembler.
MARKET_COLUMNS: frozenset[str] = frozenset({"opening_implied_prob"})

# Feature families with representative column subsets.
PHYSICAL_COLUMNS = [
    "height_cm_a",
    "height_cm_b",
    "reach_cm_a",
    "reach_cm_b",
    "reach_to_height_ratio_a",
    "reach_to_height_ratio_b",
    "age_at_fight_a",
    "age_at_fight_b",
]

ACTIVITY_COLUMNS = [
    "days_since_last_fight_a",
    "days_since_last_fight_b",
    "fights_last_12mo_a",
    "fights_last_12mo_b",
    "total_ufc_fights_a",
    "total_ufc_fights_b",
    "inactivity_tier_a",
    "inactivity_tier_b",
]

RECORD_COLUMNS = [
    "win_pct_all_a",
    "win_pct_all_b",
    "win_pct_last5_a",
    "win_pct_last5_b",
    "current_streak_a",
    "current_streak_b",
    "finish_rate_a",
    "finish_rate_b",
]

OUTPUT_COLUMNS = [
    "sig_strikes_per_min_a",
    "sig_strikes_per_min_b",
    "striking_accuracy_pct_a",
    "striking_accuracy_pct_b",
    "td_accuracy_pct_a",
    "td_accuracy_pct_b",
    "damage_ratio_a",
    "damage_ratio_b",
]

GRAPH_COLUMNS = [
    "elo_rating_a",
    "elo_rating_b",
    "glicko2_rating_a",
    "glicko2_rating_b",
    "pagerank_score_a",
    "pagerank_score_b",
    "elo_delta",
    "pagerank_delta",
]

MATCHUP_COLUMNS = [
    "reach_delta",
    "height_delta",
    "age_delta",
    "striking_efficiency_delta",
]

# The single market-derived column (must not enter model training).
MARKET_COLUMN_LIST = ["opening_implied_prob"]

ALL_FEATURE_COLUMNS = (
    PHYSICAL_COLUMNS
    + ACTIVITY_COLUMNS
    + RECORD_COLUMNS
    + OUTPUT_COLUMNS
    + GRAPH_COLUMNS
    + MATCHUP_COLUMNS
    + MARKET_COLUMN_LIST
)

# Metadata columns present alongside features.
META_COLUMNS = [
    "fight_url",
    "event_url",
    "event_date",
    "fighter_url_a",
    "fighter_url_b",
    "weight_class",
    "outcome",
]

NUM_FIGHTS = 50
NUM_EVENTS = 10
FIGHTS_PER_EVENT = NUM_FIGHTS // NUM_EVENTS  # 5


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _generate_events(rng: random.Random) -> list[dict]:
    """Generate 10 events spanning 2022-2026 with realistic spacing."""
    base_date = date(2022, 3, 5)
    events = []
    for i in range(NUM_EVENTS):
        event_date = base_date + timedelta(weeks=i * 8 + rng.randint(0, 14))
        events.append(
            {
                "event_url": f"http://ufcstats.com/event-details/evt{i:03d}",
                "event_date": event_date,
            }
        )
    return events


def _generate_fighters(rng: random.Random, count: int) -> list[str]:
    """Generate unique fighter URLs. Need at least 2*NUM_FIGHTS distinct fighters."""
    return [f"http://ufcstats.com/fighter-details/ftr{i:04d}" for i in range(count)]


def _generate_feature_value(col: str, rng: random.Random, force_null: bool) -> float | None:
    """Generate a plausible synthetic value for a given column, or NULL."""
    if force_null:
        return None

    if "height_cm" in col:
        return round(rng.uniform(160.0, 200.0), 1)
    if "reach_cm" in col:
        return round(rng.uniform(160.0, 210.0), 1)
    if "reach_to_height" in col:
        return round(rng.uniform(0.95, 1.15), 3)
    if "age_at_fight" in col:
        return round(rng.uniform(21.0, 42.0), 1)
    if "days_since_last_fight" in col:
        return float(rng.randint(30, 900))
    if "fights_last_12mo" in col:
        return float(rng.randint(0, 4))
    if "total_ufc_fights" in col:
        return float(rng.randint(1, 30))
    if "inactivity_tier" in col:
        return float(rng.randint(0, 3))
    if "win_pct" in col:
        return round(rng.uniform(0.2, 1.0), 3)
    if "current_streak" in col:
        return float(rng.randint(-5, 8))
    if "finish_rate" in col or "rate" in col:
        return round(rng.uniform(0.0, 1.0), 3)
    if "sig_strikes_per_min" in col:
        return round(rng.uniform(2.0, 8.0), 2)
    if "accuracy_pct" in col:
        return round(rng.uniform(0.30, 0.65), 3)
    if "damage_ratio" in col:
        return round(rng.uniform(0.5, 3.0), 2)
    if "elo_rating" in col or "glicko2_rating" in col:
        return round(rng.uniform(1300.0, 1800.0), 1)
    if "pagerank" in col:
        return round(rng.uniform(0.001, 0.05), 4)
    if "delta" in col:
        return round(rng.uniform(-200.0, 200.0), 1)
    if "opening_implied_prob" in col:
        return round(rng.uniform(0.15, 0.85), 3)

    return round(rng.uniform(0.0, 1.0), 3)


def generate_feature_table(
    conn: duckdb.DuckDBPyConnection,
    *,
    table_name: str = "features_v1",
    seed: int = 42,
) -> None:
    """Populate a DuckDB connection with a synthetic feature table.

    Creates 50 fight rows across 10 events with:
    - Known outcomes (win_a/win_b plus 2 draws and 1 NC)
    - Multiple weight classes
    - Deliberate NULLs in ~10% of feature cells
    - One market-derived column (opening_implied_prob) for rejection testing
    """
    rng = random.Random(seed)

    events = _generate_events(rng)
    fighters = _generate_fighters(rng, count=NUM_FIGHTS * 2)

    # Column definitions for CREATE TABLE.
    col_defs = ["fight_url VARCHAR NOT NULL"]
    col_defs.append("event_url VARCHAR NOT NULL")
    col_defs.append("event_date DATE NOT NULL")
    col_defs.append("fighter_url_a VARCHAR NOT NULL")
    col_defs.append("fighter_url_b VARCHAR NOT NULL")
    col_defs.append("weight_class VARCHAR NOT NULL")
    col_defs.append("outcome VARCHAR NOT NULL")
    for col in ALL_FEATURE_COLUMNS:
        col_defs.append(f"{col} DOUBLE")

    create_sql = f"CREATE TABLE {table_name} ({', '.join(col_defs)})"
    conn.execute(create_sql)

    # Generate rows.
    fighter_idx = 0
    fight_idx = 0

    # Indices for special outcomes: fight indices 3 and 7 are draws, index 12 is NC.
    draw_indices = {3, 7}
    nc_indices = {12}

    for _event_idx, event in enumerate(events):
        for _bout in range(FIGHTS_PER_EVENT):
            fight_url = f"http://ufcstats.com/fight-details/fght{fight_idx:04d}"
            fighter_a = fighters[fighter_idx]
            fighter_b = fighters[fighter_idx + 1]
            fighter_idx += 2

            weight_class = WEIGHT_CLASSES[fight_idx % len(WEIGHT_CLASSES)]

            if fight_idx in draw_indices:
                outcome = "draw"
            elif fight_idx in nc_indices:
                outcome = "nc"
            else:
                outcome = rng.choice(OUTCOMES)

            # Build row values.
            row = [
                fight_url,
                event["event_url"],
                event["event_date"],
                fighter_a,
                fighter_b,
                weight_class,
                outcome,
            ]

            for col in ALL_FEATURE_COLUMNS:
                # ~10% NULL rate, but never for the first 5 fights (ensures some
                # fully-populated rows for basic tests).
                force_null = fight_idx >= 5 and rng.random() < 0.10
                row.append(_generate_feature_value(col, rng, force_null))

            placeholders = ", ".join(["?"] * len(row))
            conn.execute(f"INSERT INTO {table_name} VALUES ({placeholders})", row)

            fight_idx += 1
