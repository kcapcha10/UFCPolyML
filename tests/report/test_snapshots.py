"""Tests for the post-signal snapshot scheduler.

Verifies that the scheduler correctly enqueues market re-reads at fixed
intervals after a signal fires, handles missing event times gracefully,
records failures with reasons, and writes full top-of-book data on success.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import duckdb
import pytest

from ufc_edge.report.schemas import (
    GateVerdict,
    MatchStatus,
    PaperSignal,
    SnapshotOffset,
    SnapshotStatus,
)
from ufc_edge.report.snapshots import execute_pending_snapshots, schedule_snapshots
from ufc_edge.report.storage import REPORT_DDL

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with report schema and seed data.

    Seeds a report_runs row and a paper_signals row so that snapshot
    inserts satisfy the foreign key on signal_id.
    """
    db = duckdb.connect(":memory:")
    for ddl in REPORT_DDL:
        db.execute(ddl)
    # Seed parent rows required by FK constraints
    db.execute(
        """
        INSERT INTO report_runs
            (report_run_id, as_of_timestamp, mlflow_run_id, data_revision,
             feature_version, config_hash, bout_count, flagged_count, created_at)
        VALUES ('run-001', '2026-03-15', 'mlflow-001', 'rev-1', 'v1', 'hash-1', 10, 2, '2026-03-15')
        """
    )
    db.execute(
        """
        INSERT INTO paper_signals
            (signal_id, report_run_id, fight_url, fighter_a_url, fighter_b_url,
             fighter_a_name, fighter_b_name, token_id, match_status,
             mlflow_run_id, data_revision, feature_version, config_hash, created_at)
        VALUES ('sig-001', 'run-001', '/fight/abc', '/fighter/a', '/fighter/b',
                'Fighter A', 'Fighter B', 'token-xyz', 'MATCHED',
                'mlflow-001', 'rev-1', 'v1', 'hash-1', '2026-03-15 14:00:00')
        """
    )
    return db


@pytest.fixture
def signal_time() -> datetime:
    """Fixed signal creation timestamp for deterministic tests."""
    return datetime(2026, 3, 15, 14, 0, 0)


@pytest.fixture
def event_start_time() -> datetime:
    """Fixed event start time for fight-time snapshot tests."""
    return datetime(2026, 3, 16, 22, 0, 0)


@pytest.fixture
def flagged_signal(signal_time: datetime) -> PaperSignal:
    """A flagged paper signal with all required fields for scheduling."""
    return PaperSignal(
        signal_id="sig-001",
        report_run_id="run-001",
        fight_url="/fight/abc",
        fighter_a_url="/fighter/a",
        fighter_b_url="/fighter/b",
        fighter_a_name="Fighter A",
        fighter_b_name="Fighter B",
        token_id="token-xyz",
        gate_verdict=GateVerdict.FLAGGED,
        match_status=MatchStatus.MATCHED,
        mlflow_run_id="mlflow-001",
        data_revision="rev-1",
        feature_version="v1",
        config_hash="hash-1",
        created_at=signal_time,
    )


def _make_successful_capture_fn() -> callable:
    """Return a capture function that returns valid top-of-book data."""

    def capture(token_id: str) -> dict[str, float]:
        return {
            "best_bid": 0.62,
            "best_ask": 0.65,
            "best_bid_size": 1500.0,
            "best_ask_size": 1200.0,
            "mid_price": 0.635,
        }

    return capture


def _make_failing_capture_fn(reason: str) -> callable:
    """Return a capture function that raises with a specific reason."""

    def capture(token_id: str) -> dict[str, float]:
        raise RuntimeError(reason)

    return capture


# ── schedule_snapshots: enqueue behavior ──────────────────────────────────────


class TestScheduleSnapshots:
    """Scheduling creates the correct set of snapshots for a flagged signal."""

    def test_flagged_signal_enqueues_four_snapshots(
        self,
        flagged_signal: PaperSignal,
        event_start_time: datetime,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        """A flagged signal with event time produces 4 snapshots: 1h, 4h, 24h, fight-time."""
        result = schedule_snapshots(flagged_signal, event_start_time, conn)

        assert len(result) == 4
        offsets = {s.scheduled_offset for s in result}
        assert offsets == {
            SnapshotOffset.ONE_HOUR,
            SnapshotOffset.FOUR_HOURS,
            SnapshotOffset.TWENTY_FOUR_HOURS,
            SnapshotOffset.FIGHT_TIME,
        }

    def test_scheduled_times_are_correct_relative_to_signal(
        self,
        flagged_signal: PaperSignal,
        signal_time: datetime,
        event_start_time: datetime,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        """Each timed offset is computed from the signal creation timestamp."""
        result = schedule_snapshots(flagged_signal, event_start_time, conn)

        by_offset = {s.scheduled_offset: s for s in result}
        assert by_offset[SnapshotOffset.ONE_HOUR].scheduled_at == signal_time + timedelta(hours=1)
        assert by_offset[SnapshotOffset.FOUR_HOURS].scheduled_at == signal_time + timedelta(hours=4)
        assert by_offset[SnapshotOffset.TWENTY_FOUR_HOURS].scheduled_at == (
            signal_time + timedelta(hours=24)
        )
        assert by_offset[SnapshotOffset.FIGHT_TIME].scheduled_at == event_start_time

    def test_snapshots_written_to_database(
        self,
        flagged_signal: PaperSignal,
        event_start_time: datetime,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        """All scheduled snapshots are persisted in the post_signal_snapshots table."""
        schedule_snapshots(flagged_signal, event_start_time, conn)

        row_count = conn.execute(
            "SELECT COUNT(*) FROM post_signal_snapshots WHERE signal_id = ?",
            [flagged_signal.signal_id],
        ).fetchone()[0]
        assert row_count == 4

    def test_scheduled_rows_have_pending_status_in_db(
        self,
        flagged_signal: PaperSignal,
        event_start_time: datetime,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        """Non-skipped snapshots are stored with PENDING status in the database."""
        schedule_snapshots(flagged_signal, event_start_time, conn)

        pending_count = conn.execute(
            "SELECT COUNT(*) FROM post_signal_snapshots WHERE status = 'PENDING'",
        ).fetchone()[0]
        assert pending_count == 4

    def test_all_snapshots_reference_correct_signal_and_token(
        self,
        flagged_signal: PaperSignal,
        event_start_time: datetime,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        """Every snapshot row carries the originating signal_id and token_id."""
        result = schedule_snapshots(flagged_signal, event_start_time, conn)

        for snapshot in result:
            assert snapshot.signal_id == flagged_signal.signal_id
            assert snapshot.token_id == flagged_signal.token_id

    def test_no_snapshots_for_signal_without_token_id(
        self,
        signal_time: datetime,
        event_start_time: datetime,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        """A signal with no token_id produces zero snapshots (nothing to capture)."""
        signal_no_token = PaperSignal(
            signal_id="sig-no-token",
            report_run_id="run-001",
            fight_url="/fight/abc",
            fighter_a_url="/fighter/a",
            fighter_b_url="/fighter/b",
            fighter_a_name="Fighter A",
            fighter_b_name="Fighter B",
            token_id=None,
            gate_verdict=GateVerdict.FLAGGED,
            match_status=MatchStatus.MATCHED,
            mlflow_run_id="mlflow-001",
            data_revision="rev-1",
            feature_version="v1",
            config_hash="hash-1",
            created_at=signal_time,
        )
        result = schedule_snapshots(signal_no_token, event_start_time, conn)
        assert result == []


# ── schedule_snapshots: missing event_start_time ──────────────────────────────


class TestMissingEventStartTime:
    """When event start time is unknown, fight-time snapshot is SKIPPED gracefully."""

    def test_fight_time_snapshot_marked_skipped(
        self,
        flagged_signal: PaperSignal,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        """Missing event_start_time marks the fight-time snapshot SKIPPED, not an error."""
        result = schedule_snapshots(flagged_signal, None, conn)

        fight_time = [s for s in result if s.scheduled_offset == SnapshotOffset.FIGHT_TIME]
        assert len(fight_time) == 1
        assert fight_time[0].status == SnapshotStatus.SKIPPED

    def test_skipped_snapshot_records_reason(
        self,
        flagged_signal: PaperSignal,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        """The SKIPPED fight-time snapshot includes a descriptive failure reason."""
        result = schedule_snapshots(flagged_signal, None, conn)

        fight_time = next(s for s in result if s.scheduled_offset == SnapshotOffset.FIGHT_TIME)
        assert fight_time.failure_reason is not None
        assert "event_start_time" in fight_time.failure_reason

    def test_other_offsets_still_scheduled(
        self,
        flagged_signal: PaperSignal,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        """The 1h, 4h, 24h snapshots are still scheduled even without event time."""
        result = schedule_snapshots(flagged_signal, None, conn)

        assert len(result) == 4
        non_fight = [s for s in result if s.scheduled_offset != SnapshotOffset.FIGHT_TIME]
        assert len(non_fight) == 3

    def test_skipped_snapshot_persisted_in_db(
        self,
        flagged_signal: PaperSignal,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        """SKIPPED status is written to the database (not just returned in memory)."""
        schedule_snapshots(flagged_signal, None, conn)

        skipped = conn.execute(
            "SELECT status FROM post_signal_snapshots WHERE scheduled_offset = 'FIGHT_TIME'",
        ).fetchone()
        assert skipped is not None
        assert skipped[0] == "SKIPPED"


# ── execute_pending_snapshots: failed capture ─────────────────────────────────


class TestFailedCapture:
    """Capture failures are recorded as MISSED with a reason, not raised."""

    def test_failed_capture_status_missed(
        self,
        flagged_signal: PaperSignal,
        event_start_time: datetime,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        """A capture function that raises produces a MISSED snapshot."""
        schedule_snapshots(flagged_signal, event_start_time, conn)

        # Force all PENDING rows to be schedulable by backdating them
        conn.execute(
            "UPDATE post_signal_snapshots SET scheduled_at = '2020-01-01' WHERE status = 'PENDING'"
        )

        capture_fn = _make_failing_capture_fn("API timeout after 30s")
        results = execute_pending_snapshots(conn, capture_fn)

        assert len(results) > 0
        assert all(s.status == SnapshotStatus.MISSED for s in results)

    def test_failed_capture_records_failure_reason(
        self,
        flagged_signal: PaperSignal,
        event_start_time: datetime,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        """The failure reason from the exception is stored on the snapshot."""
        schedule_snapshots(flagged_signal, event_start_time, conn)
        conn.execute(
            "UPDATE post_signal_snapshots SET scheduled_at = '2020-01-01' WHERE status = 'PENDING'"
        )

        error_msg = "Connection refused: market API unavailable"
        capture_fn = _make_failing_capture_fn(error_msg)
        results = execute_pending_snapshots(conn, capture_fn)

        for result in results:
            assert result.failure_reason == error_msg

    def test_failed_capture_persisted_to_database(
        self,
        flagged_signal: PaperSignal,
        event_start_time: datetime,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        """MISSED status and failure reason are persisted in the database."""
        schedule_snapshots(flagged_signal, event_start_time, conn)
        conn.execute(
            "UPDATE post_signal_snapshots SET scheduled_at = '2020-01-01' WHERE status = 'PENDING'"
        )

        capture_fn = _make_failing_capture_fn("network error")
        execute_pending_snapshots(conn, capture_fn)

        rows = conn.execute(
            "SELECT status, failure_reason FROM post_signal_snapshots WHERE status = 'MISSED'"
        ).fetchall()
        assert len(rows) > 0
        for status, reason in rows:
            assert status == "MISSED"
            assert reason == "network error"


# ── execute_pending_snapshots: successful capture ─────────────────────────────


class TestSuccessfulCapture:
    """Successful captures write full top-of-book fields to the snapshot."""

    def test_captured_snapshot_has_full_top_of_book(
        self,
        flagged_signal: PaperSignal,
        event_start_time: datetime,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        """A successful capture populates best_bid, best_ask, sizes, and mid_price."""
        schedule_snapshots(flagged_signal, event_start_time, conn)
        conn.execute(
            "UPDATE post_signal_snapshots SET scheduled_at = '2020-01-01' WHERE status = 'PENDING'"
        )

        capture_fn = _make_successful_capture_fn()
        results = execute_pending_snapshots(conn, capture_fn)

        assert len(results) > 0
        for snapshot in results:
            assert snapshot.status == SnapshotStatus.CAPTURED
            assert snapshot.best_bid == pytest.approx(0.62)
            assert snapshot.best_ask == pytest.approx(0.65)
            assert snapshot.best_bid_size == pytest.approx(1500.0)
            assert snapshot.best_ask_size == pytest.approx(1200.0)
            assert snapshot.mid_price == pytest.approx(0.635)

    def test_captured_snapshot_has_actual_captured_at(
        self,
        flagged_signal: PaperSignal,
        event_start_time: datetime,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        """A successful capture records the actual capture timestamp."""
        schedule_snapshots(flagged_signal, event_start_time, conn)
        conn.execute(
            "UPDATE post_signal_snapshots SET scheduled_at = '2020-01-01' WHERE status = 'PENDING'"
        )

        capture_fn = _make_successful_capture_fn()
        results = execute_pending_snapshots(conn, capture_fn)

        for snapshot in results:
            assert snapshot.actual_captured_at is not None
            assert snapshot.captured_at is not None

    def test_captured_data_persisted_to_database(
        self,
        flagged_signal: PaperSignal,
        event_start_time: datetime,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        """Full top-of-book fields are written to the database, not just returned."""
        schedule_snapshots(flagged_signal, event_start_time, conn)
        conn.execute(
            "UPDATE post_signal_snapshots SET scheduled_at = '2020-01-01' WHERE status = 'PENDING'"
        )

        capture_fn = _make_successful_capture_fn()
        execute_pending_snapshots(conn, capture_fn)

        rows = conn.execute(
            """
            SELECT best_bid, best_ask, best_bid_size, best_ask_size, mid_price, status
            FROM post_signal_snapshots
            WHERE status = 'CAPTURED'
            """
        ).fetchall()
        assert len(rows) > 0
        for bid, ask, bid_size, ask_size, mid, status in rows:
            assert bid == pytest.approx(0.62)
            assert ask == pytest.approx(0.65)
            assert bid_size == pytest.approx(1500.0)
            assert ask_size == pytest.approx(1200.0)
            assert mid == pytest.approx(0.635)
            assert status == "CAPTURED"

    def test_only_due_snapshots_are_executed(
        self,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        """Only snapshots with scheduled_at in the past are executed."""
        # Create a signal with a far-future timestamp so snapshots are not yet due
        future_time = datetime(2099, 1, 1, 0, 0, 0)
        future_signal = PaperSignal(
            signal_id="sig-001",
            report_run_id="run-001",
            fight_url="/fight/abc",
            fighter_a_url="/fighter/a",
            fighter_b_url="/fighter/b",
            fighter_a_name="Fighter A",
            fighter_b_name="Fighter B",
            token_id="token-xyz",
            gate_verdict=GateVerdict.FLAGGED,
            match_status=MatchStatus.MATCHED,
            mlflow_run_id="mlflow-001",
            data_revision="rev-1",
            feature_version="v1",
            config_hash="hash-1",
            created_at=future_time,
        )
        future_event = datetime(2099, 1, 2, 22, 0, 0)
        schedule_snapshots(future_signal, future_event, conn)

        capture_fn = _make_successful_capture_fn()
        results = execute_pending_snapshots(conn, capture_fn)

        # All snapshots are in the future, so none should be executed
        assert results == []

    def test_capture_fn_receives_correct_token_id(
        self,
        flagged_signal: PaperSignal,
        event_start_time: datetime,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        """The capture function is called with the token_id from the snapshot."""
        schedule_snapshots(flagged_signal, event_start_time, conn)
        conn.execute(
            "UPDATE post_signal_snapshots SET scheduled_at = '2020-01-01' WHERE status = 'PENDING'"
        )

        received_tokens: list[str] = []

        def tracking_capture(token_id: str) -> dict[str, float]:
            received_tokens.append(token_id)
            return {
                "best_bid": 0.50,
                "best_ask": 0.52,
                "best_bid_size": 100.0,
                "best_ask_size": 100.0,
                "mid_price": 0.51,
            }

        execute_pending_snapshots(conn, tracking_capture)

        assert all(t == flagged_signal.token_id for t in received_tokens)
