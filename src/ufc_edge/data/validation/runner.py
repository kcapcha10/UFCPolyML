"""Validation run orchestration and CLI entry point.

    uv run python -m ufc_edge.data.validation.runner   # or: make validate

Runs every invariant, refreshes the quarantine table, writes a JSON report,
and exits nonzero iff the suite fails (label-universe or undated violations
present) — so `make validate` works as a CI gate. Precondition: run after a
crawl completes; mid-crawl, referential checks see legitimately absent parents.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
from omegaconf import OmegaConf

from ufc_edge.data import storage
from ufc_edge.data.validation.invariants import ALL_INVARIANTS
from ufc_edge.data.validation.quarantine import refresh_quarantine
from ufc_edge.data.validation.schemas import ValidationReport, Violation

logger = logging.getLogger("ufc_edge.validation")

CONFIG_PATH = Path("configs/data/default.yaml")

# Fallbacks if the config file is absent (mirrors the capture cron's pattern).
DEFAULT_LABEL_START_DATE = date(2010, 1, 1)
DEFAULT_REPORT_PATH = Path("data/interim/validation_report.json")


def load_validation_config() -> tuple[date, Path]:
    """Read (label_start_date, report_path) from the data config, with defaults."""
    overrides = OmegaConf.load(CONFIG_PATH) if CONFIG_PATH.exists() else OmegaConf.create()
    label_start = date.fromisoformat(
        str(overrides.get("label_start_date", DEFAULT_LABEL_START_DATE))
    )
    report_path = Path(str(overrides.get("validation_report_path", DEFAULT_REPORT_PATH)))
    return label_start, report_path


def collect_violations(conn: duckdb.DuckDBPyConnection) -> list[Violation]:
    """Run every invariant and pool the findings."""
    violations: list[Violation] = []
    for invariant in ALL_INVARIANTS:
        found = invariant(conn)
        if found:
            logger.warning("%s: %d violation(s)", invariant.__name__, len(found))
        violations.extend(found)
    return violations


def build_report(violations: list[Violation], label_start_date: date) -> ValidationReport:
    """Aggregate violations into the era-scoped pass/fail report (D12)."""
    label_universe = [
        v for v in violations if v.event_date is not None and v.event_date >= label_start_date
    ]
    pre_cutoff = [
        v for v in violations if v.event_date is not None and v.event_date < label_start_date
    ]
    undated = [v for v in violations if v.event_date is None]
    return ValidationReport(
        ran_at=datetime.now(UTC),
        label_start_date=label_start_date,
        total_violations=len(violations),
        label_universe_violations=len(label_universe),
        pre_cutoff_violations=len(pre_cutoff),
        undated_violations=len(undated),
        counts_by_reason=dict(Counter(v.reason_code for v in violations)),
        passed=not label_universe and not undated,
    )


def run_validation(conn: duckdb.DuckDBPyConnection, label_start_date: date) -> ValidationReport:
    """One full validation pass: check, quarantine, report."""
    violations = collect_violations(conn)
    refresh_quarantine(conn, violations)
    return build_report(violations, label_start_date)


def _write_report(report: ValidationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2))


def main() -> None:
    """CLI entry point; exit code 1 on suite failure so CI/make can gate on it."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    label_start_date, report_path = load_validation_config()
    with storage.get_connection() as conn:
        report = run_validation(conn, label_start_date)
    _write_report(report, report_path)
    logger.info(
        "validation %s total=%d label_universe=%d pre_cutoff=%d undated=%d report=%s",
        "PASSED" if report.passed else "FAILED",
        report.total_violations,
        report.label_universe_violations,
        report.pre_cutoff_violations,
        report.undated_violations,
        report_path,
    )
    print(json.dumps(report.counts_by_reason, indent=2))
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
