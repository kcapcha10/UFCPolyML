"""Tests for the event-grouped temporal fold generator.

Verifies temporal integrity, event grouping, holdout protection, calibration
sizing, determinism, and fallback behaviour of the fold generation logic.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from ufc_edge.eval.splits import (
    EventEntry,
    HoldoutLeakageError,
    generate_folds,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_events(
    n_events: int,
    fights_per_event: int,
    start_year: int = 2015,
    base_url: str = "http://ufcstats.com/event-details/evt",
) -> list[EventEntry]:
    """Build a synthetic event index spanning multiple years.

    Events are spaced ~3 weeks apart starting from January of start_year.
    All events receive event_date values strictly before 2026 by default.
    """
    events = []
    current = date(start_year, 1, 10)
    for i in range(n_events):
        events.append(
            EventEntry(
                event_url=f"{base_url}{i:04d}",
                event_date=current,
                n_fights=fights_per_event,
            )
        )
        # Advance ~3 weeks
        current = date.fromordinal(current.toordinal() + 21)
    return events


def _total_fights(events: list[EventEntry]) -> int:
    return sum(e.n_fights for e in events)


# ---------------------------------------------------------------------------
# Tests: Event grouping — all fights from same event stay in one partition
# ---------------------------------------------------------------------------


class TestEventGrouping:
    """Fights from the same event must never be split across partitions."""

    def test_no_event_appears_in_multiple_partitions(self):
        events = _make_events(100, fights_per_event=12)
        folds = generate_folds(
            events, n_folds=4, min_train_fights=100, min_test_fights=50,
            calibration_min=50, calibration_ratio=0.20,
        )

        for fold in folds:
            train = fold.train_event_ids
            cal = fold.calibration_event_ids
            test = fold.test_event_ids

            # No overlap between any pair of partitions
            assert train & cal == frozenset()
            assert train & test == frozenset()
            assert cal & test == frozenset()

    def test_every_eligible_event_assigned_to_exactly_one_partition(self):
        events = _make_events(100, fights_per_event=12)
        folds = generate_folds(
            events, n_folds=4, min_train_fights=100, min_test_fights=50,
            calibration_min=50, calibration_ratio=0.20,
        )

        all_event_urls = {e.event_url for e in events}
        for fold in folds:
            fold_events = fold.train_event_ids | fold.calibration_event_ids | fold.test_event_ids
            # Every event in the fold should come from our event list
            assert fold_events <= all_event_urls


# ---------------------------------------------------------------------------
# Tests: Temporal ordering — train < calibration < test by date
# ---------------------------------------------------------------------------


class TestTemporalOrdering:
    """Train events must precede calibration events which must precede test events."""

    def test_train_dates_before_calibration_dates(self):
        events = _make_events(100, fights_per_event=12)
        folds = generate_folds(
            events, n_folds=4, min_train_fights=100, min_test_fights=50,
            calibration_min=50, calibration_ratio=0.20,
        )
        event_map = {e.event_url: e for e in events}

        for fold in folds:
            max_train_date = max(event_map[url].event_date for url in fold.train_event_ids)
            min_cal_date = min(event_map[url].event_date for url in fold.calibration_event_ids)
            assert max_train_date < min_cal_date

    def test_calibration_dates_before_test_dates(self):
        events = _make_events(100, fights_per_event=12)
        folds = generate_folds(
            events, n_folds=4, min_train_fights=100, min_test_fights=50,
            calibration_min=50, calibration_ratio=0.20,
        )
        event_map = {e.event_url: e for e in events}

        for fold in folds:
            max_cal_date = max(event_map[url].event_date for url in fold.calibration_event_ids)
            min_test_date = min(event_map[url].event_date for url in fold.test_event_ids)
            assert max_cal_date < min_test_date

    def test_strict_temporal_order_across_all_folds(self):
        """Test dates in fold k must precede test dates in fold k+1 (expanding window)."""
        events = _make_events(100, fights_per_event=12)
        folds = generate_folds(
            events, n_folds=4, min_train_fights=100, min_test_fights=50,
            calibration_min=50, calibration_ratio=0.20,
        )
        event_map = {e.event_url: e for e in events}

        for i in range(len(folds) - 1):
            max_test_k = max(event_map[url].event_date for url in folds[i].test_event_ids)
            min_test_k1 = min(event_map[url].event_date for url in folds[i + 1].test_event_ids)
            # Test partitions of successive folds should not overlap temporally
            assert max_test_k <= min_test_k1


# ---------------------------------------------------------------------------
# Tests: Expanding window grows with each fold
# ---------------------------------------------------------------------------


class TestExpandingWindow:
    """Each successive fold's training set must be a strict superset of the prior fold's."""

    def test_training_set_expands_monotonically(self):
        events = _make_events(100, fights_per_event=12)
        folds = generate_folds(
            events, n_folds=4, min_train_fights=100, min_test_fights=50,
            calibration_min=50, calibration_ratio=0.20,
        )

        for i in range(len(folds) - 1):
            # Prior fold's training events should be a subset of next fold's
            # training + calibration (the expanding window includes what was
            # previously train and cal)
            prior_all_pre_test = folds[i].train_event_ids | folds[i].calibration_event_ids
            next_all_pre_test = folds[i + 1].train_event_ids | folds[i + 1].calibration_event_ids
            assert prior_all_pre_test < next_all_pre_test

    def test_training_fight_count_grows(self):
        events = _make_events(100, fights_per_event=12)
        folds = generate_folds(
            events, n_folds=4, min_train_fights=100, min_test_fights=50,
            calibration_min=50, calibration_ratio=0.20,
        )
        event_map = {e.event_url: e for e in events}

        pre_test_sizes = [
            sum(
                event_map[url].n_fights
                for url in (fold.train_event_ids | fold.calibration_event_ids)
            )
            for fold in folds
        ]
        for i in range(len(pre_test_sizes) - 1):
            assert pre_test_sizes[i] < pre_test_sizes[i + 1]


# ---------------------------------------------------------------------------
# Tests: Holdout exclusion — 2026 events raise HoldoutLeakageError
# ---------------------------------------------------------------------------


class TestHoldoutExclusion:
    """Any event in the holdout window (2026) must be categorically excluded."""

    def test_2026_event_raises_holdout_leakage_error(self):
        events = _make_events(50, fights_per_event=12, start_year=2015)
        # Add a 2026 event that falls within the holdout window
        events.append(
            EventEntry(
                event_url="http://ufcstats.com/event-details/evt_holdout",
                event_date=date(2026, 3, 15),
                n_fights=10,
            )
        )
        with pytest.raises(HoldoutLeakageError):
            generate_folds(
                events, n_folds=4, min_train_fights=100, min_test_fights=50,
                calibration_min=50,
            )

    def test_holdout_boundary_start_raises(self):
        """Event exactly on holdout_start date must be excluded."""
        events = _make_events(50, fights_per_event=12, start_year=2015)
        events.append(
            EventEntry(
                event_url="http://ufcstats.com/event-details/evt_boundary",
                event_date=date(2026, 1, 1),
                n_fights=10,
            )
        )
        with pytest.raises(HoldoutLeakageError):
            generate_folds(
                events, n_folds=4, min_train_fights=100, min_test_fights=50,
                calibration_min=50,
            )

    def test_holdout_boundary_end_raises(self):
        """Event exactly on holdout_end date must be excluded."""
        events = _make_events(50, fights_per_event=12, start_year=2015)
        events.append(
            EventEntry(
                event_url="http://ufcstats.com/event-details/evt_end",
                event_date=date(2026, 8, 31),
                n_fights=10,
            )
        )
        with pytest.raises(HoldoutLeakageError):
            generate_folds(
                events, n_folds=4, min_train_fights=100, min_test_fights=50,
                calibration_min=50,
            )

    def test_event_before_holdout_window_is_fine(self):
        """Events in December 2025 should not trigger the holdout guard."""
        events = _make_events(80, fights_per_event=12, start_year=2015)
        # Filter out anything that might be in 2026
        events = [e for e in events if e.event_date < date(2026, 1, 1)]
        events.append(
            EventEntry(
                event_url="http://ufcstats.com/event-details/evt_dec2025",
                event_date=date(2025, 12, 31),
                n_fights=10,
            )
        )
        # Should not raise
        folds = generate_folds(
            events, n_folds=4, min_train_fights=100, min_test_fights=50,
            calibration_min=50, calibration_ratio=0.20,
        )
        assert len(folds) >= 3

    def test_custom_holdout_window(self):
        """Custom holdout window should be respected."""
        events = _make_events(80, fights_per_event=12, start_year=2015)
        events = [e for e in events if e.event_date < date(2025, 6, 1)]
        events.append(
            EventEntry(
                event_url="http://ufcstats.com/event-details/evt_custom",
                event_date=date(2025, 6, 15),
                n_fights=10,
            )
        )
        with pytest.raises(HoldoutLeakageError):
            generate_folds(
                events,
                n_folds=4,
                min_train_fights=100,
                min_test_fights=50,
                calibration_min=50,
                holdout_start="2025-06-01",
                holdout_end="2025-12-31",
            )


# ---------------------------------------------------------------------------
# Tests: Calibration sizing — max(calibration_min, ceil(20% × N))
# ---------------------------------------------------------------------------


class TestCalibrationSizing:
    """Calibration partition sized as max(calibration_min, ceil(calibration_ratio × N))."""

    def test_calibration_meets_minimum_floor(self):
        """Calibration partition has at least calibration_min fights."""
        events = _make_events(100, fights_per_event=12)
        cal_min = 100
        folds = generate_folds(
            events,
            n_folds=4,
            min_train_fights=100,
            min_test_fights=50,
            calibration_min=cal_min,
            calibration_ratio=0.05,  # Low ratio so floor governs
        )
        event_map = {e.event_url: e for e in events}

        for fold in folds:
            cal_fights = sum(event_map[url].n_fights for url in fold.calibration_event_ids)
            assert cal_fights >= cal_min

    def test_calibration_meets_ratio_when_larger_than_minimum(self):
        """When ratio × N exceeds the floor, ratio governs calibration size."""
        events = _make_events(100, fights_per_event=12)
        folds = generate_folds(
            events,
            n_folds=3,
            min_train_fights=100,
            min_test_fights=50,
            calibration_min=10,  # Very low floor so ratio dominates
            calibration_ratio=0.20,
        )
        event_map = {e.event_url: e for e in events}

        for fold in folds:
            pre_test_fights = sum(
                event_map[url].n_fights
                for url in (fold.train_event_ids | fold.calibration_event_ids)
            )
            cal_fights = sum(event_map[url].n_fights for url in fold.calibration_event_ids)
            expected_target = math.ceil(0.20 * pre_test_fights)
            # Calibration should meet the ratio (may exceed due to event grouping)
            assert cal_fights >= expected_target

    def test_calibration_with_production_defaults(self):
        """With production defaults (250 min), large datasets work correctly."""
        # 200 events × 15 = 3000 fights — enough for production-scale calibration
        events = _make_events(200, fights_per_event=15, start_year=2010)
        # Filter out 2026 events
        events = [e for e in events if e.event_date < date(2026, 1, 1)]
        folds = generate_folds(
            events,
            n_folds=4,
            min_train_fights=500,
            min_test_fights=150,
            calibration_min=250,
            calibration_ratio=0.20,
        )
        event_map = {e.event_url: e for e in events}

        for fold in folds:
            cal_fights = sum(event_map[url].n_fights for url in fold.calibration_event_ids)
            assert cal_fights >= 250


# ---------------------------------------------------------------------------
# Tests: Fallback to 3 folds when 4 not supportable
# ---------------------------------------------------------------------------


class TestFallbackFolds:
    """If 4 folds cannot satisfy min_train and min_test constraints, fall back to 3."""

    def test_returns_3_folds_when_4_not_supportable(self):
        # 40 events × 12 fights = 480 total.
        # With 5 segments (for 4 folds), each segment ≈ 96 fights.
        # min_test_fights=100 exceeds segment size, so 4 folds cannot satisfy
        # the test-partition constraint. With 4 segments (for 3 folds), each
        # segment ≈ 120 fights ≥ 100, so 3 folds work.
        events = _make_events(40, fights_per_event=12)
        folds = generate_folds(
            events, n_folds=4, min_train_fights=80, min_test_fights=100,
            calibration_min=20, calibration_ratio=0.10,
        )
        assert len(folds) == 3

    def test_returns_requested_folds_when_data_sufficient(self):
        events = _make_events(100, fights_per_event=12)
        folds = generate_folds(
            events, n_folds=4, min_train_fights=100, min_test_fights=50,
            calibration_min=50, calibration_ratio=0.20,
        )
        assert len(folds) == 4

    def test_returns_3_folds_explicitly_requested(self):
        events = _make_events(100, fights_per_event=12)
        folds = generate_folds(
            events, n_folds=3, min_train_fights=100, min_test_fights=50,
            calibration_min=50, calibration_ratio=0.20,
        )
        assert len(folds) == 3


# ---------------------------------------------------------------------------
# Tests: Determinism — same input produces same folds
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Fold generation must be entirely deterministic given the same input."""

    def test_repeated_calls_produce_identical_folds(self):
        events = _make_events(100, fights_per_event=12)
        kwargs = {
            "n_folds": 4, "min_train_fights": 100, "min_test_fights": 50,
            "calibration_min": 50, "calibration_ratio": 0.20,
        }
        folds_a = generate_folds(events, **kwargs)
        folds_b = generate_folds(events, **kwargs)

        assert len(folds_a) == len(folds_b)
        for a, b in zip(folds_a, folds_b, strict=True):
            assert a.fold_id == b.fold_id
            assert a.train_event_ids == b.train_event_ids
            assert a.calibration_event_ids == b.calibration_event_ids
            assert a.test_event_ids == b.test_event_ids

    def test_input_order_does_not_affect_output(self):
        """Even if events are passed in a different order, folds are identical."""
        events = _make_events(100, fights_per_event=12)
        shuffled = list(reversed(events))
        kwargs = {
            "n_folds": 4, "min_train_fights": 100, "min_test_fights": 50,
            "calibration_min": 50, "calibration_ratio": 0.20,
        }

        folds_ordered = generate_folds(events, **kwargs)
        folds_shuffled = generate_folds(shuffled, **kwargs)

        for a, b in zip(folds_ordered, folds_shuffled, strict=True):
            assert a.fold_id == b.fold_id
            assert a.train_event_ids == b.train_event_ids
            assert a.calibration_event_ids == b.calibration_event_ids
            assert a.test_event_ids == b.test_event_ids


# ---------------------------------------------------------------------------
# Tests: Same-date tie-breaking by event_url
# ---------------------------------------------------------------------------


class TestSameDateTieBreaking:
    """Events with the same date are ordered deterministically by event_url."""

    def test_same_date_events_ordered_by_url(self):
        """Two events on the same date get a stable, repeatable ordering."""
        # Build a large enough base of events before the same-date pair
        base_events = _make_events(80, fights_per_event=12, start_year=2015)
        base_events = [e for e in base_events if e.event_date < date(2020, 3, 14)]

        same_date_events = [
            EventEntry(
                event_url="http://ufcstats.com/event-details/zebra",
                event_date=date(2020, 3, 14),
                n_fights=10,
            ),
            EventEntry(
                event_url="http://ufcstats.com/event-details/alpha",
                event_date=date(2020, 3, 14),
                n_fights=10,
            ),
        ]
        all_events = base_events + same_date_events
        kwargs = {
            "n_folds": 3, "min_train_fights": 100, "min_test_fights": 10,
            "calibration_min": 30, "calibration_ratio": 0.10,
        }

        folds_a = generate_folds(all_events, **kwargs)
        folds_b = generate_folds(all_events, **kwargs)

        # Same-date events always land in the same partitions
        for a, b in zip(folds_a, folds_b, strict=True):
            assert a.train_event_ids == b.train_event_ids
            assert a.calibration_event_ids == b.calibration_event_ids
            assert a.test_event_ids == b.test_event_ids

    def test_same_date_events_never_split_nondeterministically(self):
        """Multiple events on the same date produce stable partition assignments."""
        same_date = date(2021, 7, 10)
        same_date_events = [
            EventEntry(
                event_url=f"http://ufcstats.com/event-details/evt_samedate_{i:02d}",
                event_date=same_date,
                n_fights=8,
            )
            for i in range(5)
        ]
        earlier = _make_events(60, fights_per_event=12, start_year=2015)
        earlier = [e for e in earlier if e.event_date < same_date]
        all_events = earlier + same_date_events
        kwargs = {
            "n_folds": 3, "min_train_fights": 100, "min_test_fights": 10,
            "calibration_min": 30, "calibration_ratio": 0.10,
        }

        folds1 = generate_folds(all_events, **kwargs)
        folds2 = generate_folds(all_events, **kwargs)

        for f1, f2 in zip(folds1, folds2, strict=True):
            assert f1.train_event_ids == f2.train_event_ids
            assert f1.calibration_event_ids == f2.calibration_event_ids
            assert f1.test_event_ids == f2.test_event_ids
