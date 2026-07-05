"""Invariant checks over the ufcstats tables.

Each public function runs one invariant family and returns Violations. All
invariants target *impossible* data only; expected historical absence (e.g. no
per-round stats before ~2001) is skipped by construction — conditional
invariants only evaluate rows where the check is well-defined.
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb

from ufc_edge.data.validation.schemas import Violation

# Reason codes (stable identifiers — quarantine rows and reports key on these).
LANDED_GT_ATTEMPTED = "LANDED_GT_ATTEMPTED"
NEGATIVE_STAT = "NEGATIVE_STAT"
ROUND_SUM_MISMATCH = "ROUND_SUM_MISMATCH"
BREAKDOWN_PARTITION_MISMATCH = "BREAKDOWN_PARTITION_MISMATCH"
ORPHAN_ROW = "ORPHAN_ROW"
SELF_FIGHT = "SELF_FIGHT"
WINNER_NOT_IN_FIGHT = "WINNER_NOT_IN_FIGHT"
ENDING_ROUND_INVALID = "ENDING_ROUND_INVALID"
MEASUREMENT_OUT_OF_RANGE = "MEASUREMENT_OUT_OF_RANGE"
EVENT_DATE_OUT_OF_RANGE = "EVENT_DATE_OUT_OF_RANGE"
CONTROL_TIME_EXCEEDS_FIGHT = "CONTROL_TIME_EXCEEDS_FIGHT"

# (landed, attempted) column pairs per stats table.
_STAT_PAIRS: dict[str, list[tuple[str, str]]] = {
    "fight_totals": [
        ("significant_strikes_landed", "significant_strikes_attempted"),
        ("total_strikes_landed", "total_strikes_attempted"),
        ("takedowns_landed", "takedowns_attempted"),
    ],
    "round_stats": [
        ("significant_strikes_landed", "significant_strikes_attempted"),
        ("total_strikes_landed", "total_strikes_attempted"),
        ("takedowns_landed", "takedowns_attempted"),
    ],
    "sig_strike_breakdowns": [
        ("head_landed", "head_attempted"),
        ("body_landed", "body_attempted"),
        ("leg_landed", "leg_attempted"),
        ("distance_landed", "distance_attempted"),
        ("clinch_landed", "clinch_attempted"),
        ("ground_landed", "ground_attempted"),
    ],
}

# Count columns that must be non-negative, per stats table.
_COUNT_COLUMNS: dict[str, list[str]] = {
    "fight_totals": [
        "knockdowns",
        "significant_strikes_landed",
        "significant_strikes_attempted",
        "total_strikes_landed",
        "total_strikes_attempted",
        "takedowns_landed",
        "takedowns_attempted",
        "submission_attempts",
        "reversals",
        "control_time_seconds",
    ],
    "round_stats": [
        "knockdowns",
        "significant_strikes_landed",
        "significant_strikes_attempted",
        "takedowns_landed",
        "takedowns_attempted",
        "submission_attempts",
        "reversals",
        "control_time_seconds",
    ],
    "sig_strike_breakdowns": [
        column for pair in _STAT_PAIRS["sig_strike_breakdowns"] for column in pair
    ],
}

# Totals columns cross-checked against per-round sums (control_time excluded:
# nullable and subject to source rounding, so a mismatch is not "impossible").
_ROUND_SUM_COLUMNS = [
    "knockdowns",
    "significant_strikes_landed",
    "significant_strikes_attempted",
    "total_strikes_landed",
    "total_strikes_attempted",
    "takedowns_landed",
    "takedowns_attempted",
]

# Plausible human ranges; outside these the value is a parse error, not an outlier.
# Weight upper bound calibrated on real data (T-D2): early open-weight UFC had
# legitimate 400 lb fighters — Emmanuel Yarbrough weighs in at 349.3 kg.
HEIGHT_CM_RANGE = (120.0, 230.0)
REACH_CM_RANGE = (120.0, 250.0)
WEIGHT_KG_RANGE = (40.0, 360.0)

UFC_FIRST_EVENT_DATE = date(1993, 11, 12)
MAX_FUTURE_EVENT_DAYS = 400  # announced events exist; a date past this is a parse error
ROUND_LENGTH_SECONDS = 300  # standard 5-minute rounds; only '(5-5-5' formats are checked

# Joins that attach an event date to each table's rows for era scoping.
_STATS_FROM = (
    "FROM {table} t "
    "JOIN fights f ON t.fight_url = f.fight_url "
    "JOIN events e ON f.event_url = e.event_url"
)
_ROUND_KEYED_TABLES = ("round_stats", "sig_strike_breakdowns")


def _row_key(table: str, row: tuple) -> str:
    """Build the human-readable key from the leading key columns of a result row."""
    key_width = 3 if table in _ROUND_KEYED_TABLES else 2
    return "|".join(str(part) for part in row[:key_width])


def _key_columns_sql(table: str) -> str:
    base = "t.fight_url, t.fighter_url"
    return f"{base}, t.round" if table in _ROUND_KEYED_TABLES else base


def landed_exceeds_attempted(conn: duckdb.DuckDBPyConnection) -> list[Violation]:
    """A strike/takedown cannot land more times than it was attempted."""
    violations: list[Violation] = []
    for table, pairs in _STAT_PAIRS.items():
        for landed, attempted in pairs:
            sql = (
                f"SELECT {_key_columns_sql(table)}, e.date, t.{landed}, t.{attempted} "
                f"{_STATS_FROM.format(table=table)} WHERE t.{landed} > t.{attempted}"
            )
            for row in conn.execute(sql).fetchall():
                violations.append(
                    Violation(
                        table_name=table,
                        row_key=_row_key(table, row),
                        reason_code=LANDED_GT_ATTEMPTED,
                        detail=f"{landed}={row[-2]} > {attempted}={row[-1]}",
                        event_date=row[-3],
                    )
                )
    return violations


def negative_counts(conn: duckdb.DuckDBPyConnection) -> list[Violation]:
    """Count-valued stats can never be negative."""
    violations: list[Violation] = []
    for table, columns in _COUNT_COLUMNS.items():
        for column in columns:
            sql = (
                f"SELECT {_key_columns_sql(table)}, e.date, t.{column} "
                f"{_STATS_FROM.format(table=table)} WHERE t.{column} < 0"
            )
            for row in conn.execute(sql).fetchall():
                violations.append(
                    Violation(
                        table_name=table,
                        row_key=_row_key(table, row),
                        reason_code=NEGATIVE_STAT,
                        detail=f"{column}={row[-1]} is negative",
                        event_date=row[-2],
                    )
                )
    return violations


def round_sums_mismatch_totals(conn: duckdb.DuckDBPyConnection) -> list[Violation]:
    """Where per-round rows exist, they must sum exactly to the fight totals.

    Conditional invariant: fights with no round rows (early-era data) are
    skipped — absence is missingness, not corruption.
    """
    sum_selects = ", ".join(
        f"SUM(r.{c}) AS r_{c}, ANY_VALUE(t.{c}) AS t_{c}" for c in _ROUND_SUM_COLUMNS
    )
    predicate = " OR ".join(f"r_{c} != t_{c}" for c in _ROUND_SUM_COLUMNS)
    sql = (
        f"SELECT r.fight_url, r.fighter_url, ANY_VALUE(e.date), {sum_selects} "
        "FROM round_stats r "
        "JOIN fight_totals t ON r.fight_url = t.fight_url AND r.fighter_url = t.fighter_url "
        "JOIN fights f ON r.fight_url = f.fight_url "
        "JOIN events e ON f.event_url = e.event_url "
        "GROUP BY r.fight_url, r.fighter_url "
        f"HAVING {predicate}"
    )
    violations: list[Violation] = []
    for row in conn.execute(sql).fetchall():
        mismatches = [
            f"{c}: rounds={row[3 + 2 * i]} totals={row[4 + 2 * i]}"
            for i, c in enumerate(_ROUND_SUM_COLUMNS)
            if row[3 + 2 * i] != row[4 + 2 * i]
        ]
        violations.append(
            Violation(
                table_name="round_stats",
                row_key=f"{row[0]}|{row[1]}",
                reason_code=ROUND_SUM_MISMATCH,
                detail="; ".join(mismatches),
                event_date=row[2],
            )
        )
    return violations


def breakdown_partitions_disagree(conn: duckdb.DuckDBPyConnection) -> list[Violation]:
    """Target split (head+body+leg) and position split (distance+clinch+ground)
    partition the same significant strikes, so their sums must agree."""
    sql = (
        "SELECT t.fight_url, t.fighter_url, t.round, e.date, "
        "t.head_landed + t.body_landed + t.leg_landed AS target_landed, "
        "t.distance_landed + t.clinch_landed + t.ground_landed AS position_landed "
        f"{_STATS_FROM.format(table='sig_strike_breakdowns')} "
        "WHERE t.head_landed + t.body_landed + t.leg_landed "
        "!= t.distance_landed + t.clinch_landed + t.ground_landed"
    )
    return [
        Violation(
            table_name="sig_strike_breakdowns",
            row_key=f"{row[0]}|{row[1]}|{row[2]}",
            reason_code=BREAKDOWN_PARTITION_MISMATCH,
            detail=f"target split={row[4]} != position split={row[5]}",
            event_date=row[3],
        )
        for row in conn.execute(sql).fetchall()
    ]


def orphaned_rows(conn: duckdb.DuckDBPyConnection) -> list[Violation]:
    """Referential integrity: child rows must join to their parents.

    Orphans carry event_date=None and therefore always trip the alarm — a
    broken join is a crawl/parser defect in any era. Precondition: run after a
    crawl completes (mid-crawl, parents may legitimately not exist yet).
    """
    child_of_fights_sql = (
        "SELECT t.fight_url, t.fighter_url FROM {table} t "
        "LEFT JOIN fights f ON t.fight_url = f.fight_url WHERE f.fight_url IS NULL"
    )
    checks = [
        (
            "fights",
            "SELECT t.fight_url FROM fights t "
            "LEFT JOIN events e ON t.event_url = e.event_url WHERE e.event_url IS NULL",
            "event_url not in events",
        ),
    ] + [
        (table, child_of_fights_sql.format(table=table), "fight_url not in fights")
        for table in ("fight_totals", "round_stats", "sig_strike_breakdowns")
    ]
    violations: list[Violation] = []
    for table, sql, why in checks:
        for row in conn.execute(sql).fetchall():
            violations.append(
                Violation(
                    table_name=table,
                    row_key="|".join(str(part) for part in row),
                    reason_code=ORPHAN_ROW,
                    detail=why,
                    event_date=None,
                )
            )
    return violations


def self_fight(conn: duckdb.DuckDBPyConnection) -> list[Violation]:
    """A fighter cannot fight themselves."""
    sql = (
        "SELECT t.fight_url, e.date FROM fights t "
        "JOIN events e ON t.event_url = e.event_url "
        "WHERE t.fighter_a_url = t.fighter_b_url"
    )
    return [
        Violation(
            table_name="fights",
            row_key=str(row[0]),
            reason_code=SELF_FIGHT,
            detail="fighter_a_url == fighter_b_url",
            event_date=row[1],
        )
        for row in conn.execute(sql).fetchall()
    ]


def winner_not_in_fight(conn: duckdb.DuckDBPyConnection) -> list[Violation]:
    """A non-null winner must be one of the two participants."""
    sql = (
        "SELECT t.fight_url, e.date, t.winner_url FROM fights t "
        "JOIN events e ON t.event_url = e.event_url "
        "WHERE t.winner_url IS NOT NULL "
        "AND t.winner_url NOT IN (t.fighter_a_url, t.fighter_b_url)"
    )
    return [
        Violation(
            table_name="fights",
            row_key=str(row[0]),
            reason_code=WINNER_NOT_IN_FIGHT,
            detail=f"winner_url={row[2]} is neither participant",
            event_date=row[1],
        )
        for row in conn.execute(sql).fetchall()
    ]


def ending_round_invalid(conn: duckdb.DuckDBPyConnection) -> list[Violation]:
    """The ending round must lie within the scheduled round count.

    Conditional: only evaluated where time_format's leading round count parses
    (e.g. '3 Rnd (5-5-5)'). Overtime formats ('1 Rnd + OT', '1 Rnd + 2OT') are
    skipped — OT legitimately pushes ending_round past the leading digit
    (calibrated on real data, T-D2: 29 false positives on 1990s fights).
    """
    sql = (
        "SELECT t.fight_url, e.date, t.ending_round, t.time_format FROM fights t "
        "JOIN events e ON t.event_url = e.event_url "
        "WHERE TRY_CAST(substr(t.time_format, 1, 1) AS INTEGER) IS NOT NULL "
        "AND t.time_format NOT LIKE '%OT%' "
        "AND (t.ending_round < 1 "
        "     OR t.ending_round > TRY_CAST(substr(t.time_format, 1, 1) AS INTEGER))"
    )
    return [
        Violation(
            table_name="fights",
            row_key=str(row[0]),
            reason_code=ENDING_ROUND_INVALID,
            detail=f"ending_round={row[2]} outside schedule '{row[3]}'",
            event_date=row[1],
        )
        for row in conn.execute(sql).fetchall()
    ]


def fighter_measurements_out_of_range(conn: duckdb.DuckDBPyConnection) -> list[Violation]:
    """Height/reach/weight outside plausible human ranges are parse errors.

    Fighter rows have no event date of their own; each violation is dated by
    the fighter's most recent fight so a modern parser regression still trips
    the label-universe alarm.
    """
    ranges = {
        "height_cm": HEIGHT_CM_RANGE,
        "reach_cm": REACH_CM_RANGE,
        "weight_kg": WEIGHT_KG_RANGE,
    }
    last_fight_join = (
        "LEFT JOIN ("
        "  SELECT fighter_url, MAX(e.date) AS last_event_date FROM ("
        "    SELECT fighter_a_url AS fighter_url, event_url FROM fights "
        "    UNION ALL SELECT fighter_b_url, event_url FROM fights"
        "  ) fx JOIN events e ON fx.event_url = e.event_url GROUP BY fighter_url"
        ") le ON t.fighter_url = le.fighter_url"
    )
    violations: list[Violation] = []
    for column, (low, high) in ranges.items():
        sql = (
            f"SELECT t.fighter_url, le.last_event_date, t.{column} "
            f"FROM fighters t {last_fight_join} "
            f"WHERE t.{column} IS NOT NULL AND (t.{column} < {low} OR t.{column} > {high})"
        )
        for row in conn.execute(sql).fetchall():
            violations.append(
                Violation(
                    table_name="fighters",
                    row_key=str(row[0]),
                    reason_code=MEASUREMENT_OUT_OF_RANGE,
                    detail=f"{column}={row[2]} outside [{low}, {high}]",
                    event_date=row[1],
                )
            )
    return violations


def event_date_out_of_range(conn: duckdb.DuckDBPyConnection) -> list[Violation]:
    """Event dates before UFC 1 or far in the future are parse errors."""
    max_date = date.today() + timedelta(days=MAX_FUTURE_EVENT_DAYS)
    sql = "SELECT event_url, date FROM events WHERE date < ? OR date > ?"
    return [
        Violation(
            table_name="events",
            row_key=str(row[0]),
            reason_code=EVENT_DATE_OUT_OF_RANGE,
            detail=f"event date {row[1]} outside [{UFC_FIRST_EVENT_DATE}, {max_date}]",
            event_date=row[1],
        )
        for row in conn.execute(sql, [UFC_FIRST_EVENT_DATE, max_date]).fetchall()
    ]


def control_time_exceeds_fight_duration(conn: duckdb.DuckDBPyConnection) -> list[Violation]:
    """One fighter's control time cannot exceed the fight's elapsed duration.

    Conditional: only standard 5-minute-round formats ('(5-5-5' pattern) are
    checked; exotic formats have unknown round lengths.
    """
    duration_sql = (
        f"(f.ending_round - 1) * {ROUND_LENGTH_SECONDS} "
        "+ TRY_CAST(split_part(f.ending_time, ':', 1) AS INTEGER) * 60 "
        "+ TRY_CAST(split_part(f.ending_time, ':', 2) AS INTEGER)"
    )
    sql = (
        "SELECT t.fight_url, t.fighter_url, e.date, t.control_time_seconds, "
        f"{duration_sql} AS fight_seconds "
        f"{_STATS_FROM.format(table='fight_totals')} "
        "WHERE f.time_format LIKE '%(5-5-5%' "
        "AND t.control_time_seconds IS NOT NULL "
        f"AND t.control_time_seconds > {duration_sql}"
    )
    return [
        Violation(
            table_name="fight_totals",
            row_key=f"{row[0]}|{row[1]}",
            reason_code=CONTROL_TIME_EXCEEDS_FIGHT,
            detail=f"control_time_seconds={row[3]} > fight duration {row[4]}s",
            event_date=row[2],
        )
        for row in conn.execute(sql).fetchall()
    ]


ALL_INVARIANTS = [
    landed_exceeds_attempted,
    negative_counts,
    round_sums_mismatch_totals,
    breakdown_partitions_disagree,
    orphaned_rows,
    self_fight,
    winner_not_in_fight,
    ending_round_invalid,
    fighter_measurements_out_of_range,
    event_date_out_of_range,
    control_time_exceeds_fight_duration,
]
