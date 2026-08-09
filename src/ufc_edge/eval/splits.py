"""Event-grouped temporal fold generator for expanding-window cross-validation.

This module is the temporal-leakage firewall for the evaluation pipeline. It
guarantees that:
- All fights from the same event stay in one partition (prevents intra-card leakage)
- Training events precede calibration events which precede test events chronologically
- No event from the holdout window (default: 2026) enters any development fold
- Calibration slices meet minimum sizing requirements for reliable probability mapping
- Fold generation is fully deterministic given the same input

The expanding-window design means each successive fold trains on all data up to its
test boundary, giving later folds more training history while earlier folds test on
historically earlier fights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from ufc_edge.eval.schemas import Fold

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventEntry:
    """One event in the temporal index used for fold generation.

    Represents an event's position in time and its contribution to fight counts.
    The event_url serves as both a unique identifier and the tiebreaker for events
    sharing the same date.
    """

    event_url: str
    event_date: date
    n_fights: int


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class HoldoutLeakageError(Exception):
    """Raised when a holdout-window event would enter a development fold.

    The holdout period is reserved for final one-time evaluation after model
    selection is frozen. Allowing holdout data into development folds would
    compromise the integrity of that final evaluation.
    """


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_folds(
    event_index: list[EventEntry],
    *,
    n_folds: int = 4,
    min_train_fights: int = 500,
    min_test_fights: int = 150,
    calibration_ratio: float = 0.20,
    calibration_min: int = 250,
    holdout_start: str = "2026-01-01",
    holdout_end: str = "2026-08-31",
) -> list[Fold]:
    """Generate expanding-window event-grouped temporal folds.

    Sorts events chronologically (ties broken by event_url for determinism),
    excludes holdout-window events, then partitions the remaining events into
    expanding training + calibration + test splits.

    Params:
        event_index: All events with their dates and fight counts.
        n_folds: Target number of folds (falls back to fewer if not supportable).
        min_train_fights: Minimum fights required in the training partition.
        min_test_fights: Minimum fights required in the test partition.
        calibration_ratio: Fraction of training-eligible fights for calibration.
        calibration_min: Absolute floor on calibration partition fight count.
        holdout_start: Start of the holdout window (inclusive), ISO format.
        holdout_end: End of the holdout window (inclusive), ISO format.

    Returns:
        List of Fold objects with disjoint train/calibration/test event sets.

    Raises:
        HoldoutLeakageError: If any event falls within the holdout window.
        ValueError: If data is insufficient for even 2 folds.
    """
    holdout_start_date = date.fromisoformat(holdout_start)
    holdout_end_date = date.fromisoformat(holdout_end)

    _guard_holdout(event_index, holdout_start_date, holdout_end_date)

    # Sort deterministically: primary by date, secondary by event_url for ties
    sorted_events = sorted(event_index, key=lambda e: (e.event_date, e.event_url))

    # Try the requested fold count, then fall back to fewer
    for target_folds in range(n_folds, 1, -1):
        folds = _try_generate_folds(
            sorted_events, target_folds, min_train_fights, min_test_fights,
            calibration_ratio, calibration_min,
        )
        if folds is not None:
            return folds

    msg = (
        f"Cannot generate folds: insufficient data for even 2 folds with "
        f"min_train_fights={min_train_fights}, min_test_fights={min_test_fights}"
    )
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Internal logic
# ---------------------------------------------------------------------------


def _guard_holdout(
    events: list[EventEntry],
    holdout_start: date,
    holdout_end: date,
) -> None:
    """Reject any event that falls within the holdout window.

    The holdout period exists so the final evaluation is never contaminated by
    data the model has seen during development. Any breach is a hard failure.
    """
    for event in events:
        if holdout_start <= event.event_date <= holdout_end:
            msg = (
                f"Event '{event.event_url}' (date={event.event_date}) falls within "
                f"the holdout window [{holdout_start}, {holdout_end}]. "
                f"Holdout events must not enter development folds."
            )
            raise HoldoutLeakageError(msg)


def _try_generate_folds(
    sorted_events: list[EventEntry],
    n_folds: int,
    min_train_fights: int,
    min_test_fights: int,
    calibration_ratio: float,
    calibration_min: int,
) -> list[Fold] | None:
    """Attempt to partition events into n expanding-window folds.

    Returns None if constraints cannot be satisfied.

    The strategy divides the timeline into (n_folds + 1) roughly equal segments
    by fight count. For fold k (0-indexed), the test partition is segment k+1 and
    training draws from segments 0..k. The calibration slice is carved from the
    trailing end of the training window.
    """
    n_events = len(sorted_events)

    if n_events < n_folds + 1:
        return None

    # Divide events into (n_folds + 1) segments of approximately equal fight count
    segment_boundaries = _compute_segment_boundaries(sorted_events, n_folds + 1)
    if segment_boundaries is None:
        return None

    folds: list[Fold] = []

    for fold_idx in range(n_folds):
        # Test partition: segment (fold_idx + 1)
        test_start = segment_boundaries[fold_idx + 1]
        test_end = segment_boundaries[fold_idx + 2]
        test_events = sorted_events[test_start:test_end]

        # Pre-test events: segments 0 through fold_idx (inclusive)
        pre_test_end = segment_boundaries[fold_idx + 1]
        pre_test_events = sorted_events[:pre_test_end]

        test_fights = sum(e.n_fights for e in test_events)
        pre_test_fights = sum(e.n_fights for e in pre_test_events)

        if test_fights < min_test_fights:
            return None

        # Calibration sized as max(calibration_min, ceil(ratio × pre_test_fights))
        cal_target = max(calibration_min, math.ceil(calibration_ratio * pre_test_fights))
        train_events, cal_events = _split_calibration(pre_test_events, cal_target)

        train_fights = sum(e.n_fights for e in train_events)
        if train_fights < min_train_fights:
            return None

        folds.append(
            Fold(
                fold_id=fold_idx,
                train_event_ids=frozenset(e.event_url for e in train_events),
                calibration_event_ids=frozenset(e.event_url for e in cal_events),
                test_event_ids=frozenset(e.event_url for e in test_events),
            )
        )

    return folds


def _compute_segment_boundaries(
    sorted_events: list[EventEntry],
    n_segments: int,
) -> list[int] | None:
    """Divide sorted events into n_segments of roughly equal total fight count.

    Returns a list of (n_segments + 1) boundary indices into sorted_events.
    Segment k spans [boundaries[k], boundaries[k+1]).
    Returns None if there aren't enough events for all segments.
    """
    n_events = len(sorted_events)
    if n_events < n_segments:
        return None

    total_fights = sum(e.n_fights for e in sorted_events)
    target_per_segment = total_fights / n_segments

    boundaries = [0]
    accumulated = 0

    for i, event in enumerate(sorted_events):
        accumulated += event.n_fights
        segments_placed = len(boundaries) - 1
        segments_remaining = n_segments - segments_placed

        # Place a boundary when we've accumulated enough for one segment,
        # but ensure at least one event remains per future segment
        events_remaining = n_events - (i + 1)
        if (
            segments_remaining > 1
            and accumulated >= target_per_segment
            and events_remaining >= segments_remaining - 1
        ):
            boundaries.append(i + 1)
            accumulated = 0

    # Final boundary
    boundaries.append(n_events)

    if len(boundaries) != n_segments + 1:
        return None

    # Verify each segment has at least one event
    for k in range(n_segments):
        if boundaries[k] >= boundaries[k + 1]:
            return None

    return boundaries


def _split_calibration(
    pre_test_events: list[EventEntry],
    cal_target: int,
) -> tuple[list[EventEntry], list[EventEntry]]:
    """Split pre-test events into training and calibration partitions.

    The calibration partition is the trailing (most recent) slice of the pre-test
    events, containing at least cal_target fights. We work backwards from the end
    to accumulate enough fights for calibration; the remainder forms the training set.

    This ensures calibration data is temporally between training and test data,
    which prevents the calibrator from being fitted on data distributions that
    the model hasn't seen.
    """
    cal_fights = 0
    split_idx = len(pre_test_events)

    # Walk backwards accumulating fights for calibration
    for i in range(len(pre_test_events) - 1, -1, -1):
        cal_fights += pre_test_events[i].n_fights
        split_idx = i
        if cal_fights >= cal_target:
            break

    train_events = pre_test_events[:split_idx]
    cal_events = pre_test_events[split_idx:]
    return train_events, cal_events
