"""Post-signal snapshot scheduler and executor.

Captures the market price at fixed intervals after a signal fires so a later
analysis can see how the market moved, without committing to any particular
trading strategy now. The scheduler creates one row per offset; the executor
reads pending rows and attempts capture via an injected callable.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from ufc_edge.report.schemas import (
    PaperSignal,
    PostSignalSnapshot,
    SnapshotOffset,
    SnapshotStatus,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import duckdb

# ── Configuration defaults ────────────────────────────────────────────────────

_OFFSET_TO_HOURS: dict[SnapshotOffset, float] = {
    SnapshotOffset.ONE_HOUR: 1.0,
    SnapshotOffset.FOUR_HOURS: 4.0,
    SnapshotOffset.TWENTY_FOUR_HOURS: 24.0,
}

_DEFAULT_OFFSETS: list[SnapshotOffset] = [
    SnapshotOffset.ONE_HOUR,
    SnapshotOffset.FOUR_HOURS,
    SnapshotOffset.TWENTY_FOUR_HOURS,
]


# ── Public API ────────────────────────────────────────────────────────────────


def schedule_snapshots(
    signal: PaperSignal,
    event_start_time: datetime | None,
    conn: duckdb.DuckDBPyConnection,
) -> list[PostSignalSnapshot]:
    """Schedule post-signal snapshots for a flagged signal.

    Creates one snapshot per configured offset (1h, 4h, 24h) plus an optional
    fight-time snapshot. If event_start_time is None, the fight-time snapshot
    is marked SKIPPED immediately (not an error). All other snapshots are
    written as PENDING in the database, awaiting later execution.

    Returns the full list of PostSignalSnapshot objects representing what was
    scheduled (SKIPPED snapshots are terminal; others await execution).
    """
    if signal.token_id is None:
        return []

    signal_time = signal.created_at
    snapshots: list[PostSignalSnapshot] = []

    for offset in _DEFAULT_OFFSETS:
        hours = _OFFSET_TO_HOURS[offset]
        scheduled_at = signal_time + timedelta(hours=hours)
        snapshot = PostSignalSnapshot(
            snapshot_id=str(uuid.uuid4()),
            signal_id=signal.signal_id,
            token_id=signal.token_id,
            scheduled_offset=offset,
            scheduled_at=scheduled_at,
            status=SnapshotStatus.CAPTURED,  # placeholder for DB insert
        )
        snapshots.append(snapshot)

    fight_time_snapshot = _build_fight_time_snapshot(
        signal=signal,
        event_start_time=event_start_time,
    )
    snapshots.append(fight_time_snapshot)

    _persist_scheduled(snapshots, conn)
    return snapshots


def execute_pending_snapshots(
    conn: duckdb.DuckDBPyConnection,
    capture_fn: Callable[[str], dict[str, float]],
) -> list[PostSignalSnapshot]:
    """Execute all pending snapshots whose scheduled time has arrived.

    Reads rows with status='PENDING' and scheduled_at <= now, attempts capture
    via the injected capture_fn, and writes the result. capture_fn receives a
    token_id and returns a dict with keys: best_bid, best_ask, best_bid_size,
    best_ask_size, mid_price. If capture_fn raises, the snapshot is marked
    MISSED with the failure reason recorded.

    Returns the list of snapshots that were processed (now CAPTURED or MISSED).
    """
    now = datetime.now(UTC)
    pending_rows = conn.execute(
        """
        SELECT snapshot_id, signal_id, token_id, scheduled_offset, scheduled_at
        FROM post_signal_snapshots
        WHERE status = 'PENDING' AND scheduled_at <= ?
        """,
        [now],
    ).fetchall()

    results: list[PostSignalSnapshot] = []
    for row in pending_rows:
        snapshot_id, signal_id, token_id, offset_str, scheduled_at = row
        offset = SnapshotOffset(offset_str)
        captured = _attempt_capture(
            snapshot_id=snapshot_id,
            signal_id=signal_id,
            token_id=token_id,
            offset=offset,
            scheduled_at=scheduled_at,
            capture_fn=capture_fn,
            conn=conn,
        )
        results.append(captured)

    return results


# ── Internal helpers ──────────────────────────────────────────────────────────


def _build_fight_time_snapshot(
    signal: PaperSignal,
    event_start_time: datetime | None,
) -> PostSignalSnapshot:
    """Build the fight-time snapshot, marking it SKIPPED if no event time is known."""
    if event_start_time is None:
        return PostSignalSnapshot(
            snapshot_id=str(uuid.uuid4()),
            signal_id=signal.signal_id,
            token_id=signal.token_id,
            scheduled_offset=SnapshotOffset.FIGHT_TIME,
            scheduled_at=signal.created_at,
            status=SnapshotStatus.SKIPPED,
            failure_reason="event_start_time not available",
        )

    return PostSignalSnapshot(
        snapshot_id=str(uuid.uuid4()),
        signal_id=signal.signal_id,
        token_id=signal.token_id,
        scheduled_offset=SnapshotOffset.FIGHT_TIME,
        scheduled_at=event_start_time,
        status=SnapshotStatus.CAPTURED,  # placeholder for DB insert as PENDING
    )


def _persist_scheduled(
    snapshots: list[PostSignalSnapshot],
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """Write snapshot rows to the database with appropriate initial status.

    Terminal snapshots (SKIPPED) are written with their final status.
    All others are written as PENDING, awaiting execution.
    """
    for snapshot in snapshots:
        db_status = "SKIPPED" if snapshot.status == SnapshotStatus.SKIPPED else "PENDING"

        conn.execute(
            """
            INSERT INTO post_signal_snapshots
                (snapshot_id, signal_id, token_id, scheduled_offset, scheduled_at,
                 status, failure_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (snapshot_id) DO NOTHING
            """,
            [
                snapshot.snapshot_id,
                snapshot.signal_id,
                snapshot.token_id,
                snapshot.scheduled_offset.value,
                snapshot.scheduled_at,
                db_status,
                snapshot.failure_reason,
            ],
        )


def _attempt_capture(
    snapshot_id: str,
    signal_id: str,
    token_id: str,
    offset: SnapshotOffset,
    scheduled_at: datetime,
    capture_fn: Callable[[str], dict[str, float]],
    conn: duckdb.DuckDBPyConnection,
) -> PostSignalSnapshot:
    """Attempt a single snapshot capture and persist the result."""
    now = datetime.now(UTC)

    try:
        book = capture_fn(token_id)
        snapshot = PostSignalSnapshot(
            snapshot_id=snapshot_id,
            signal_id=signal_id,
            token_id=token_id,
            scheduled_offset=offset,
            scheduled_at=scheduled_at,
            actual_captured_at=now,
            status=SnapshotStatus.CAPTURED,
            best_bid=book["best_bid"],
            best_ask=book["best_ask"],
            best_bid_size=book["best_bid_size"],
            best_ask_size=book["best_ask_size"],
            mid_price=book["mid_price"],
            captured_at=now,
        )
    except Exception as exc:  # noqa: BLE001
        snapshot = PostSignalSnapshot(
            snapshot_id=snapshot_id,
            signal_id=signal_id,
            token_id=token_id,
            scheduled_offset=offset,
            scheduled_at=scheduled_at,
            actual_captured_at=now,
            status=SnapshotStatus.MISSED,
            failure_reason=str(exc),
        )

    conn.execute(
        """
        UPDATE post_signal_snapshots
        SET actual_captured_at = ?,
            status = ?,
            failure_reason = ?,
            best_bid = ?,
            best_ask = ?,
            best_bid_size = ?,
            best_ask_size = ?,
            mid_price = ?,
            captured_at = ?
        WHERE snapshot_id = ?
        """,
        [
            snapshot.actual_captured_at,
            snapshot.status.value,
            snapshot.failure_reason,
            snapshot.best_bid,
            snapshot.best_ask,
            snapshot.best_bid_size,
            snapshot.best_ask_size,
            snapshot.mid_price,
            snapshot.captured_at,
            snapshot_id,
        ],
    )

    return snapshot
