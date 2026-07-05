"""Quarantine table DDL and refresh.

Quarantine records violations; it never moves or mutates source rows (D12 —
auto-fixing is rejected outright). Feature computation excludes quarantined
rows by anti-joining on (table_name, row_key). Each run fully replaces the
table so it always reflects the current state of the data — a violation fixed
by a re-scrape disappears on the next run instead of lingering.
"""

from __future__ import annotations

from datetime import UTC, datetime

import duckdb

from ufc_edge.data.validation.schemas import Violation

_CREATE_QUARANTINE = """
CREATE TABLE IF NOT EXISTS validation_quarantine (
    table_name   VARCHAR NOT NULL,
    row_key      VARCHAR NOT NULL,
    reason_code  VARCHAR NOT NULL,
    detail       VARCHAR,
    event_date   DATE,
    detected_at  TIMESTAMP NOT NULL,
    PRIMARY KEY (table_name, row_key, reason_code)
);
"""

VALIDATION_DDL: list[str] = [
    _CREATE_QUARANTINE,
]


def refresh_quarantine(conn: duckdb.DuckDBPyConnection, violations: list[Violation]) -> None:
    """Replace the quarantine table with the current run's findings."""
    conn.execute("DELETE FROM validation_quarantine")
    if not violations:
        return
    detected_at = datetime.now(UTC)
    conn.executemany(
        """
        INSERT INTO validation_quarantine
            (table_name, row_key, reason_code, detail, event_date, detected_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (table_name, row_key, reason_code) DO NOTHING
        """,
        [
            [v.table_name, v.row_key, v.reason_code, v.detail, v.event_date, detected_at]
            for v in violations
        ],
    )
