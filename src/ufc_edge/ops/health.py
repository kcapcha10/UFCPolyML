"""Health metrics for the report pipeline and its market-capture inputs.

The reporter is intentionally read-only: it summarizes the existing DuckDB
report and capture tables without changing pipeline state or claiming that
fixture results represent production monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

import duckdb
from omegaconf import OmegaConf

CONFIG_PATH = Path("configs/report/default.yaml")
DEFAULT_CAPTURE_MAX_AGE_MINUTES = 30
DEFAULT_REPORT_MAX_AGE_DAYS = 14


class HealthWarning(StrEnum):
    """Stable warning codes emitted by the health report."""

    STALE_CAPTURE = "STALE_CAPTURE"
    STALE_REPORT = "STALE_REPORT"
    UNRESOLVED_LINKS = "UNRESOLVED_LINKS"


@dataclass(frozen=True)
class HealthConfig:
    """Configured maximum ages used to assess report and capture freshness."""

    capture_max_age_minutes: int = DEFAULT_CAPTURE_MAX_AGE_MINUTES
    report_max_age_days: int = DEFAULT_REPORT_MAX_AGE_DAYS

    def __post_init__(self) -> None:
        """Reject negative freshness windows that cannot describe a valid age."""
        if self.capture_max_age_minutes < 0:
            raise ValueError("capture_max_age_minutes must be non-negative")
        if self.report_max_age_days < 0:
            raise ValueError("report_max_age_days must be non-negative")


@dataclass(frozen=True)
class HealthReport:
    """Read-only report-pipeline metrics and deterministic warning codes."""

    latest_report_run_at: datetime | None
    total_paper_signals: int
    flagged_signal_count: int
    unresolved_link_count: int
    latest_capture_at: datetime | None
    warnings: tuple[HealthWarning, ...]

    @property
    def warning_messages(self) -> tuple[str, ...]:
        """Render stable human-readable messages, including unresolved-link count."""
        messages: list[str] = []
        for warning in self.warnings:
            if warning == HealthWarning.UNRESOLVED_LINKS:
                messages.append(f"{warning}: {self.unresolved_link_count} unresolved link(s)")
            else:
                messages.append(warning.value)
        return tuple(messages)


def load_health_config(path: Path = CONFIG_PATH) -> HealthConfig:
    """Load staleness settings from the report config, using documented defaults."""
    overrides = OmegaConf.load(path) if path.exists() else OmegaConf.create()
    staleness = overrides.get("staleness", {})
    return HealthConfig(
        capture_max_age_minutes=int(
            staleness.get("capture_max_age_minutes", DEFAULT_CAPTURE_MAX_AGE_MINUTES)
        ),
        report_max_age_days=int(staleness.get("report_max_age_days", DEFAULT_REPORT_MAX_AGE_DAYS)),
    )


def build_health_report(
    conn: duckdb.DuckDBPyConnection,
    *,
    now: datetime | None = None,
    config: HealthConfig | None = None,
) -> HealthReport:
    """Query report/capture metrics and evaluate configured freshness warnings.

    Args:
        conn: Open DuckDB connection with report and capture tables initialized.
        now: Comparison timestamp; defaults to the current UTC time. Tests should
            provide a frozen value.
        config: Staleness settings; loads the report YAML when omitted.

    Returns:
        A typed snapshot. Empty tables yield zero counts and a stale warning for
        each missing freshness timestamp; no rows are inserted or updated.
    """
    comparison_time = now or datetime.now(UTC)
    settings = config or load_health_config()
    latest_report_run_at = _latest_report_run_at(conn)
    total_paper_signals, flagged_signal_count = _signal_counts(conn)
    unresolved_link_count = _unresolved_link_count(conn)
    latest_capture_at = _latest_capture_at(conn)

    warnings: list[HealthWarning] = []
    if _is_stale(
        latest_capture_at,
        comparison_time,
        timedelta(minutes=settings.capture_max_age_minutes),
    ):
        warnings.append(HealthWarning.STALE_CAPTURE)
    if _is_stale(
        latest_report_run_at,
        comparison_time,
        timedelta(days=settings.report_max_age_days),
    ):
        warnings.append(HealthWarning.STALE_REPORT)
    if unresolved_link_count:
        warnings.append(HealthWarning.UNRESOLVED_LINKS)

    return HealthReport(
        latest_report_run_at=latest_report_run_at,
        total_paper_signals=total_paper_signals,
        flagged_signal_count=flagged_signal_count,
        unresolved_link_count=unresolved_link_count,
        latest_capture_at=latest_capture_at,
        warnings=tuple(warnings),
    )


def _latest_report_run_at(conn: duckdb.DuckDBPyConnection) -> datetime | None:
    """Return the execution time of the newest persisted report run."""
    row = conn.execute("SELECT MAX(created_at) FROM report_runs").fetchone()
    return row[0] if row else None


def _signal_counts(conn: duckdb.DuckDBPyConnection) -> tuple[int, int]:
    """Return total signals and the subset that passed the magnitude gate."""
    row = conn.execute(
        """
        SELECT COUNT(*) AS total_signals,
               COUNT(*) FILTER (WHERE gate_verdict = 'FLAGGED') AS flagged_signals
        FROM paper_signals
        """
    ).fetchone()
    if row is None:
        return 0, 0
    return int(row[0]), int(row[1])


def _unresolved_link_count(conn: duckdb.DuckDBPyConnection) -> int:
    """Count unreviewed links whose persisted resolution is not MATCHED."""
    # A reviewed non-MATCHED link is an intentional human disposition and is no
    # longer an actionable unresolved item for the health summary.
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM market_fight_links
        WHERE match_status <> 'MATCHED' AND reviewed_by IS NULL
        """
    ).fetchone()
    return int(row[0]) if row else 0


def _latest_capture_at(conn: duckdb.DuckDBPyConnection) -> datetime | None:
    """Return the newest timestamp in the append-only capture table."""
    row = conn.execute("SELECT MAX(captured_at) FROM order_book_snapshots").fetchone()
    return row[0] if row else None


def _is_stale(
    timestamp: datetime | None,
    now: datetime,
    maximum_age: timedelta,
) -> bool:
    """Treat missing data as stale and compare aware/naive UTC timestamps safely."""
    if timestamp is None:
        return True
    return _as_utc(now) - _as_utc(timestamp) > maximum_age


def _as_utc(timestamp: datetime) -> datetime:
    """Normalize naive database timestamps and aware caller timestamps to UTC."""
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)
