"""Round-trip and append-only tests for mismatch-report DuckDB storage."""

from __future__ import annotations

from datetime import datetime

import duckdb
import pytest

from ufc_edge.report.schemas import (
    ChecklistFindings,
    DueDiligenceVerdict,
    DueDiligenceVerdictType,
    Finding,
    GateVerdict,
    MarketFightLink,
    MatchMethod,
    MatchStatus,
    PaperSignal,
    PostSignalSnapshot,
    ReportRun,
    ScoreboardEntry,
    SnapshotOffset,
    SnapshotStatus,
)
from ufc_edge.report.storage import (
    REPORT_DDL,
    LinkOverwriteError,
    write_link,
    write_paper_signal,
    write_post_signal_snapshot,
    write_report_run,
    write_scoreboard_entry,
    write_verdict,
)

_NOW = datetime(2026, 8, 31, 12, 0, 0)


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    """Return an in-memory database initialized with all report tables."""
    database = duckdb.connect(":memory:")
    for ddl in REPORT_DDL:
        database.execute(ddl)
    return database


def _run(run_id: str = "run-abc", **overrides: object) -> ReportRun:
    """Build a report run with deterministic provenance values."""
    values: dict[str, object] = {
        "report_run_id": run_id,
        "as_of_timestamp": _NOW,
        "mlflow_run_id": "mlflow-abc",
        "data_revision": "revision-abc",
        "feature_version": "features-abc",
        "config_hash": "config-abc",
        "bout_count": 2,
        "flagged_count": 1,
        "created_at": _NOW,
    }
    values.update(overrides)
    return ReportRun(**values)


def _signal(
    signal_id: str = "signal-abc", run_id: str = "run-abc", **overrides: object
) -> PaperSignal:
    """Build a paper signal that references the supplied report run ID."""
    values: dict[str, object] = {
        "signal_id": signal_id,
        "report_run_id": run_id,
        "fight_url": "/fight/abc",
        "event_date": "2026-09-01",
        "fighter_a_url": "/fighter/a",
        "fighter_b_url": "/fighter/b",
        "fighter_a_name": "Fighter A",
        "fighter_b_name": "Fighter B",
        "p_model": 0.7,
        "p_market_mid": 0.5,
        "mismatch": 0.2,
        "gate_verdict": GateVerdict.FLAGGED,
        "bucket_id": "0.5-0.7",
        "bucket_n": 100,
        "bucket_calibration_error": 0.04,
        "bucket_ci_lower": 0.02,
        "bucket_ci_upper": 0.06,
        "match_status": MatchStatus.MATCHED,
        "mlflow_run_id": "mlflow-abc",
        "data_revision": "revision-abc",
        "feature_version": "features-abc",
        "config_hash": "config-abc",
        "created_at": _NOW,
    }
    values.update(overrides)
    return PaperSignal(**values)


def _verdict(run_id: str = "run-abc", **overrides: object) -> DueDiligenceVerdict:
    """Build a due-diligence verdict with one evidence URL."""
    values: dict[str, object] = {
        "fight_url": "/fight/abc",
        "report_run_id": run_id,
        "verdict": DueDiligenceVerdictType.CONFIRM,
        "confidence": 0.8,
        "evidence_urls": ["https://example.com/evidence"],
        "summary": "No material concerns found.",
        "checklist_findings": ChecklistFindings(
            injury_news=Finding(
                present=False,
                detail="No injury news",
                source_url="https://example.com/evidence",
            )
        ),
        "prompt_version": "prompt-v1",
        "model_name": "model",
        "model_version": "model-v1",
        "invoked_at": _NOW,
    }
    values.update(overrides)
    return DueDiligenceVerdict(**values)


def _snapshot(signal_id: str = "signal-abc", **overrides: object) -> PostSignalSnapshot:
    """Build a captured post-signal snapshot for the supplied signal ID."""
    values: dict[str, object] = {
        "snapshot_id": "snapshot-abc",
        "signal_id": signal_id,
        "token_id": "token-abc",
        "scheduled_offset": SnapshotOffset.ONE_HOUR,
        "scheduled_at": _NOW,
        "actual_captured_at": _NOW,
        "status": SnapshotStatus.CAPTURED,
        "best_bid": 0.49,
        "best_ask": 0.51,
        "best_bid_size": 100.0,
        "best_ask_size": 120.0,
        "mid_price": 0.5,
        "captured_at": _NOW,
    }
    values.update(overrides)
    return PostSignalSnapshot(**values)


def _scoreboard_entry(**overrides: object) -> ScoreboardEntry:
    """Build a resolved scoreboard entry."""
    values: dict[str, object] = {
        "fight_url": "/fight/abc",
        "report_run_id": "run-abc",
        "verdict": DueDiligenceVerdictType.CONFIRM,
        "mismatch_at_signal": 0.2,
        "fight_resolved": True,
        "outcome_correct": True,
        "resolved_at": _NOW,
    }
    values.update(overrides)
    return ScoreboardEntry(**values)


def _seed_signal_parent(conn: duckdb.DuckDBPyConnection) -> None:
    """Insert the parent rows needed by paper-signal and snapshot foreign keys."""
    write_report_run(conn, _run())
    write_paper_signal(conn, _signal())


def test_all_write_functions_round_trip(conn: duckdb.DuckDBPyConnection) -> None:
    """Each storage writer persists the model fields in its target table."""
    write_report_run(conn, _run())
    write_link(
        conn,
        MarketFightLink(
            fight_url="/fight/abc",
            token_id="token-abc",
            match_status=MatchStatus.MATCHED,
            match_method=MatchMethod.AUTO_NAME,
            matched_at=_NOW,
        ),
    )
    write_paper_signal(conn, _signal())
    write_post_signal_snapshot(conn, _snapshot())
    write_verdict(conn, _verdict())
    write_scoreboard_entry(conn, _scoreboard_entry())

    assert conn.execute("SELECT COUNT(*) FROM report_runs").fetchone()[0] == 1
    assert conn.execute("SELECT match_status, match_method FROM market_fight_links").fetchone() == (
        "MATCHED",
        "AUTO_NAME",
    )
    assert conn.execute("SELECT p_model, gate_verdict FROM paper_signals").fetchone() == (
        0.7,
        "FLAGGED",
    )
    assert conn.execute("SELECT status, mid_price FROM post_signal_snapshots").fetchone() == (
        "CAPTURED",
        0.5,
    )
    assert conn.execute("SELECT verdict, evidence_urls FROM due_diligence_verdicts").fetchone() == (
        "CONFIRM",
        '["https://example.com/evidence"]',
    )
    assert conn.execute(
        "SELECT fight_resolved, outcome_correct FROM verdict_scoreboard"
    ).fetchone() == (True, True)


def test_duplicate_report_run_is_ignored(conn: duckdb.DuckDBPyConnection) -> None:
    """A duplicate report-run ID does not overwrite the original provenance."""
    write_report_run(conn, _run())
    write_report_run(conn, _run(bout_count=99, flagged_count=99, config_hash="changed"))

    row = conn.execute("SELECT bout_count, flagged_count, config_hash FROM report_runs").fetchone()

    assert row == (2, 1, "config-abc")


def test_duplicate_paper_signal_is_ignored(conn: duckdb.DuckDBPyConnection) -> None:
    """A duplicate signal ID does not overwrite the original signal values."""
    write_report_run(conn, _run())
    write_paper_signal(conn, _signal())
    write_paper_signal(
        conn,
        _signal(p_model=0.1, mismatch=-0.4, gate_verdict=GateVerdict.WITHIN_NOISE),
    )

    row = conn.execute("SELECT p_model, mismatch, gate_verdict FROM paper_signals").fetchone()

    assert row == (0.7, 0.2, "FLAGGED")


def test_duplicate_snapshot_verdict_and_scoreboard_rows_are_ignored(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """Duplicate keys preserve the first snapshot, verdict, and scoreboard values."""
    _seed_signal_parent(conn)
    write_post_signal_snapshot(conn, _snapshot())
    write_post_signal_snapshot(conn, _snapshot(mid_price=0.9, status=SnapshotStatus.MISSED))
    write_verdict(conn, _verdict())
    write_verdict(conn, _verdict(verdict=DueDiligenceVerdictType.VETO, confidence=0.1))
    write_scoreboard_entry(conn, _scoreboard_entry())
    write_scoreboard_entry(conn, _scoreboard_entry(outcome_correct=False))

    assert conn.execute("SELECT mid_price, status FROM post_signal_snapshots").fetchone() == (
        0.5,
        "CAPTURED",
    )
    assert conn.execute("SELECT verdict, confidence FROM due_diligence_verdicts").fetchone() == (
        "CONFIRM",
        0.8,
    )
    assert conn.execute("SELECT outcome_correct FROM verdict_scoreboard").fetchone() == (True,)


def test_matched_link_cannot_be_overwritten(conn: duckdb.DuckDBPyConnection) -> None:
    """A MATCHED link remains unchanged when a replacement is attempted."""
    original = MarketFightLink(
        fight_url="/fight/abc",
        token_id="token-abc",
        match_status=MatchStatus.MATCHED,
        match_method=MatchMethod.AUTO_NAME,
        matched_at=_NOW,
    )
    write_link(conn, original)

    replacement = original.model_copy(
        update={"match_method": MatchMethod.HUMAN_CONFIRMED, "reviewed_by": "reviewer"}
    )
    with pytest.raises(LinkOverwriteError):
        write_link(conn, replacement)

    assert conn.execute("SELECT match_method, reviewed_by FROM market_fight_links").fetchone() == (
        "AUTO_NAME",
        None,
    )


def test_nonmatched_link_can_be_promoted_once(conn: duckdb.DuckDBPyConnection) -> None:
    """A non-MATCHED link can be replaced by the human-confirmed MATCHED row."""
    unresolved = MarketFightLink(
        fight_url="/fight/abc",
        token_id="token-abc",
        match_status=MatchStatus.NO_CANDIDATE,
        matched_at=_NOW,
    )
    write_link(conn, unresolved)
    confirmed = unresolved.model_copy(
        update={
            "match_status": MatchStatus.MATCHED,
            "match_method": MatchMethod.HUMAN_CONFIRMED,
            "reviewed_by": "reviewer",
        }
    )
    write_link(conn, confirmed)

    assert conn.execute(
        "SELECT match_status, match_method, reviewed_by FROM market_fight_links"
    ).fetchone() == ("MATCHED", "HUMAN_CONFIRMED", "reviewer")


def test_paper_signal_requires_existing_report_run(conn: duckdb.DuckDBPyConnection) -> None:
    """A paper signal referencing an unknown report run is rejected by the FK."""
    with pytest.raises(duckdb.ConstraintException):
        write_paper_signal(conn, _signal(run_id="missing-run"))

    assert conn.execute("SELECT COUNT(*) FROM paper_signals").fetchone()[0] == 0


def test_snapshot_requires_existing_paper_signal(conn: duckdb.DuckDBPyConnection) -> None:
    """A post-signal snapshot referencing an unknown signal is rejected by the FK."""
    with pytest.raises(duckdb.ConstraintException):
        write_post_signal_snapshot(conn, _snapshot(signal_id="missing-signal"))

    assert conn.execute("SELECT COUNT(*) FROM post_signal_snapshots").fetchone()[0] == 0
