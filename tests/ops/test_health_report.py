"""Fixture tests for report-pipeline health metrics and freshness warnings."""

from __future__ import annotations

from datetime import datetime, timedelta

import duckdb
import pytest

from ufc_edge.data.polymarket.storage import POLYMARKET_DDL
from ufc_edge.ops.health import (
    HealthConfig,
    HealthWarning,
    build_health_report,
)
from ufc_edge.report.storage import REPORT_DDL

_NOW = datetime(2026, 8, 31, 12, 0, 0)


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    """Return an in-memory database with report and capture tables."""
    database = duckdb.connect(":memory:")
    for ddl in REPORT_DDL + POLYMARKET_DDL:
        database.execute(ddl)
    return database


@pytest.fixture
def config() -> HealthConfig:
    """Use short deterministic freshness windows for fixture assertions."""
    return HealthConfig(capture_max_age_minutes=30, report_max_age_days=14)


def test_empty_tables_report_missing_freshness_as_stale(
    conn: duckdb.DuckDBPyConnection, config: HealthConfig
) -> None:
    """Empty report and capture tables return zero metrics and stale warnings."""
    report = build_health_report(conn, now=_NOW, config=config)

    assert report.latest_report_run_at is None
    assert report.total_paper_signals == 0
    assert report.flagged_signal_count == 0
    assert report.unresolved_link_count == 0
    assert report.latest_capture_at is None
    assert report.warnings == (HealthWarning.STALE_CAPTURE, HealthWarning.STALE_REPORT)


def test_report_health_aggregates_report_signal_link_and_capture_metrics(
    conn: duckdb.DuckDBPyConnection, config: HealthConfig
) -> None:
    """Health metrics are computed from the latest report, signals, links, and capture."""
    conn.executemany(
        """
        INSERT INTO report_runs
            (report_run_id, as_of_timestamp, mlflow_run_id, data_revision,
             feature_version, config_hash, bout_count, flagged_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "run-old",
                _NOW - timedelta(days=2),
                "mlflow-old",
                "data",
                "features",
                "config",
                1,
                0,
                _NOW - timedelta(days=2),
            ),
            (
                "run-new",
                _NOW - timedelta(minutes=5),
                "mlflow-new",
                "data",
                "features",
                "config",
                2,
                1,
                _NOW - timedelta(minutes=5),
            ),
        ],
    )
    conn.executemany(
        """
        INSERT INTO paper_signals
            (signal_id, report_run_id, fight_url, fighter_a_url, fighter_b_url,
             fighter_a_name, fighter_b_name, gate_verdict, match_status,
             mlflow_run_id, data_revision, feature_version, config_hash, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "signal-flagged",
                "run-new",
                "/fight/1",
                "/fighter/a",
                "/fighter/b",
                "A",
                "B",
                "FLAGGED",
                "MATCHED",
                "mlflow-new",
                "data",
                "features",
                "config",
                _NOW,
            ),
            (
                "signal-noise",
                "run-new",
                "/fight/2",
                "/fighter/c",
                "/fighter/d",
                "C",
                "D",
                "WITHIN_NOISE",
                "MATCHED",
                "mlflow-new",
                "data",
                "features",
                "config",
                _NOW,
            ),
            (
                "signal-old",
                "run-old",
                "/fight/3",
                "/fighter/e",
                "/fighter/f",
                "E",
                "F",
                "NO_BUCKET_DATA",
                "MATCHED",
                "mlflow-old",
                "data",
                "features",
                "config",
                _NOW,
            ),
        ],
    )
    conn.executemany(
        """
        INSERT INTO market_fight_links
            (fight_url, token_id, match_status, matched_at, reviewed_by)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            ("/fight/1", "token-1", "MATCHED", _NOW, None),
            ("/fight/2", "token-2", "NO_CANDIDATE", _NOW, None),
            ("/fight/3", "token-3", "MULTIPLE_CANDIDATES", _NOW, "reviewer"),
        ],
    )
    conn.execute(
        """
        INSERT INTO order_book_snapshots
            (market_id, token_id, bids, asks, captured_at)
        VALUES ('market-1', 'token-1', '[]', '[]', ?)
        """,
        [_NOW - timedelta(minutes=30)],
    )

    report = build_health_report(conn, now=_NOW, config=config)

    assert report.latest_report_run_at == _NOW - timedelta(minutes=5)
    assert report.total_paper_signals == 3
    assert report.flagged_signal_count == 1
    assert report.unresolved_link_count == 1
    assert report.latest_capture_at == _NOW - timedelta(minutes=30)
    assert report.warnings == (HealthWarning.UNRESOLVED_LINKS,)


def test_staleness_warnings_trigger_only_after_configured_boundaries(
    conn: duckdb.DuckDBPyConnection, config: HealthConfig
) -> None:
    """Capture and report timestamps exactly at their limits are still healthy."""
    conn.execute(
        """
        INSERT INTO report_runs
            (report_run_id, as_of_timestamp, mlflow_run_id, data_revision,
             feature_version, config_hash, bout_count, flagged_count, created_at)
        VALUES ('run-1', ?, 'mlflow', 'data', 'features', 'config', 0, 0, ?)
        """,
        [_NOW - timedelta(days=14), _NOW - timedelta(days=14)],
    )
    conn.execute(
        """
        INSERT INTO order_book_snapshots
            (market_id, token_id, bids, asks, captured_at)
        VALUES ('market-1', 'token-1', '[]', '[]', ?)
        """,
        [_NOW - timedelta(minutes=30)],
    )

    at_boundary = build_health_report(conn, now=_NOW, config=config)
    stale = build_health_report(
        conn,
        now=_NOW + timedelta(seconds=1),
        config=config,
    )

    assert at_boundary.warnings == ()
    assert stale.warnings == (HealthWarning.STALE_CAPTURE, HealthWarning.STALE_REPORT)


def test_unresolved_link_warning_includes_count_without_requiring_capture_data(
    conn: duckdb.DuckDBPyConnection, config: HealthConfig
) -> None:
    """An unresolved link is warned independently of capture freshness."""
    conn.execute(
        """
        INSERT INTO market_fight_links
            (fight_url, token_id, match_status, matched_at)
        VALUES ('/fight/unresolved', 'token-unresolved', 'NO_CANDIDATE', ?)
        """,
        [_NOW],
    )

    report = build_health_report(conn, now=_NOW, config=config)

    assert report.unresolved_link_count == 1
    assert HealthWarning.UNRESOLVED_LINKS in report.warnings
    assert "UNRESOLVED_LINKS: 1 unresolved link(s)" in report.warning_messages
