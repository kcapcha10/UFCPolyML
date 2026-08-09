"""Tests for WeightClassTracker state component.

Validates weight class history tracking per fighter, migration detection,
fight count accumulation per class, top-quartile (large-for-class) calculation,
and freeze immutability.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

from ufc_edge.features.contracts import FightOutcomeView

# Load weight_class module directly — avoids components/__init__.py which may
# reference sibling modules still being written by concurrent agents.
_mod_path = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "ufc_edge"
    / "features"
    / "components"
    / "weight_class.py"
)
_spec = importlib.util.spec_from_file_location(
    "ufc_edge.features.components.weight_class", _mod_path
)
assert _spec is not None and _spec.loader is not None
_wc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _wc
_spec.loader.exec_module(_wc)

WeightClassTracker = _wc.WeightClassTracker
WeightClassFrozenState = _wc.WeightClassFrozenState


def _make_fight(
    fight_url: str = "/fight/1",
    event_url: str = "/event/1",
    event_date: date = date(2023, 1, 1),
    fighter_a_url: str = "/fighter/a",
    fighter_b_url: str = "/fighter/b",
    winner_url: str | None = "/fighter/a",
    method: str = "Decision - Unanimous",
    ending_round: int = 3,
    ending_time: str = "5:00",
    weight_class: str = "Lightweight",
    bout_order: int | None = None,
) -> FightOutcomeView:
    return FightOutcomeView(
        fight_url=fight_url,
        event_url=event_url,
        event_date=event_date,
        fighter_a_url=fighter_a_url,
        fighter_b_url=fighter_b_url,
        winner_url=winner_url,
        method=method,
        ending_round=ending_round,
        ending_time=ending_time,
        weight_class=weight_class,
        bout_order=bout_order,
    )


class TestWeightClassTrackerUpdate:
    """Tests for state accumulation through update()."""

    def test_first_fight_records_natural_class(self) -> None:
        tracker = WeightClassTracker()
        fight = _make_fight(weight_class="Lightweight", fighter_a_url="/fighter/a")
        tracker.update(fight)

        state = tracker.freeze()
        fighter_state = state.get_fighter_state("/fighter/a")
        assert fighter_state is not None
        assert fighter_state.first_class == "Lightweight"

    def test_tracks_both_fighters(self) -> None:
        tracker = WeightClassTracker()
        fight = _make_fight(
            fighter_a_url="/fighter/a",
            fighter_b_url="/fighter/b",
            weight_class="Welterweight",
        )
        tracker.update(fight)

        state = tracker.freeze()
        assert state.get_fighter_state("/fighter/a") is not None
        assert state.get_fighter_state("/fighter/b") is not None

    def test_fight_count_accumulates_per_class(self) -> None:
        tracker = WeightClassTracker()
        for i in range(3):
            tracker.update(
                _make_fight(
                    fight_url=f"/fight/{i}",
                    weight_class="Middleweight",
                    event_date=date(2023, 1 + i, 1),
                )
            )

        state = tracker.freeze()
        fighter_state = state.get_fighter_state("/fighter/a")
        assert fighter_state is not None
        assert fighter_state.fights_at_class["Middleweight"] == 3

    def test_detects_class_change(self) -> None:
        tracker = WeightClassTracker()
        tracker.update(
            _make_fight(
                fight_url="/fight/1",
                weight_class="Lightweight",
                event_date=date(2023, 1, 1),
            )
        )
        tracker.update(
            _make_fight(
                fight_url="/fight/2",
                weight_class="Welterweight",
                event_date=date(2023, 6, 1),
            )
        )

        state = tracker.freeze()
        fighter_state = state.get_fighter_state("/fighter/a")
        assert fighter_state is not None
        assert fighter_state.first_class == "Lightweight"
        assert fighter_state.current_class == "Welterweight"
        assert fighter_state.migration_count == 1

    def test_tracks_multiple_migrations(self) -> None:
        tracker = WeightClassTracker()
        classes = ["Lightweight", "Welterweight", "Middleweight", "Welterweight"]
        for i, wc in enumerate(classes):
            tracker.update(
                _make_fight(
                    fight_url=f"/fight/{i}",
                    weight_class=wc,
                    event_date=date(2023, 1 + i, 1),
                )
            )

        state = tracker.freeze()
        fighter_state = state.get_fighter_state("/fighter/a")
        assert fighter_state is not None
        assert fighter_state.first_class == "Lightweight"
        assert fighter_state.current_class == "Welterweight"
        assert fighter_state.migration_count == 3

    def test_fight_count_per_class_multiple_classes(self) -> None:
        tracker = WeightClassTracker()
        tracker.update(
            _make_fight(
                fight_url="/fight/1",
                weight_class="Lightweight",
                event_date=date(2023, 1, 1),
            )
        )
        tracker.update(
            _make_fight(
                fight_url="/fight/2",
                weight_class="Lightweight",
                event_date=date(2023, 3, 1),
            )
        )
        tracker.update(
            _make_fight(
                fight_url="/fight/3",
                weight_class="Welterweight",
                event_date=date(2023, 6, 1),
            )
        )

        state = tracker.freeze()
        fighter_state = state.get_fighter_state("/fighter/a")
        assert fighter_state is not None
        assert fighter_state.fights_at_class["Lightweight"] == 2
        assert fighter_state.fights_at_class["Welterweight"] == 1

    def test_win_tracking_per_class(self) -> None:
        tracker = WeightClassTracker()
        # Fighter A wins fight 1
        tracker.update(
            _make_fight(
                fight_url="/fight/1",
                weight_class="Lightweight",
                winner_url="/fighter/a",
                event_date=date(2023, 1, 1),
            )
        )
        # Fighter A loses fight 2
        tracker.update(
            _make_fight(
                fight_url="/fight/2",
                weight_class="Lightweight",
                winner_url="/fighter/b",
                event_date=date(2023, 3, 1),
            )
        )

        state = tracker.freeze()
        fighter_state = state.get_fighter_state("/fighter/a")
        assert fighter_state is not None
        assert fighter_state.wins_at_class["Lightweight"] == 1
        assert fighter_state.fights_at_class["Lightweight"] == 2


class TestWeightClassTrackerLargeForClass:
    """Tests for top-quartile (large-for-class) identification.

    The tracker accepts a comparison population as precomputed percentile
    thresholds per weight class. This avoids coupling the component to a
    specific population query — the replay engine can supply thresholds
    computed from any point-in-time population snapshot.
    """

    def test_identifies_large_for_class_both_above_threshold(self) -> None:
        tracker = WeightClassTracker()
        # Thresholds represent 75th-percentile cutoffs for the weight class
        thresholds = {
            "Lightweight": {"reach_cm": 185.0, "height_cm": 180.0},
        }
        tracker.set_class_thresholds(thresholds)
        tracker.update(
            _make_fight(
                fight_url="/fight/1",
                weight_class="Lightweight",
                event_date=date(2023, 1, 1),
            )
        )

        state = tracker.freeze()
        # Fighter with reach 190 and height 183 — both above threshold
        assert state.is_large_for_class("/fighter/a", reach_cm=190.0, height_cm=183.0) is True

    def test_not_large_if_only_reach_above(self) -> None:
        tracker = WeightClassTracker()
        thresholds = {
            "Lightweight": {"reach_cm": 185.0, "height_cm": 180.0},
        }
        tracker.set_class_thresholds(thresholds)
        tracker.update(
            _make_fight(
                fight_url="/fight/1",
                weight_class="Lightweight",
                event_date=date(2023, 1, 1),
            )
        )

        state = tracker.freeze()
        # Reach above threshold, height below
        assert state.is_large_for_class("/fighter/a", reach_cm=190.0, height_cm=175.0) is False

    def test_not_large_if_only_height_above(self) -> None:
        tracker = WeightClassTracker()
        thresholds = {
            "Lightweight": {"reach_cm": 185.0, "height_cm": 180.0},
        }
        tracker.set_class_thresholds(thresholds)
        tracker.update(
            _make_fight(
                fight_url="/fight/1",
                weight_class="Lightweight",
                event_date=date(2023, 1, 1),
            )
        )

        state = tracker.freeze()
        # Height above threshold, reach below
        assert state.is_large_for_class("/fighter/a", reach_cm=180.0, height_cm=183.0) is False

    def test_not_large_when_measurements_missing(self) -> None:
        tracker = WeightClassTracker()
        thresholds = {
            "Lightweight": {"reach_cm": 185.0, "height_cm": 180.0},
        }
        tracker.set_class_thresholds(thresholds)
        tracker.update(
            _make_fight(
                fight_url="/fight/1",
                weight_class="Lightweight",
                event_date=date(2023, 1, 1),
            )
        )

        state = tracker.freeze()
        # Missing reach — cannot determine
        assert state.is_large_for_class("/fighter/a", reach_cm=None, height_cm=183.0) is False

    def test_not_large_when_no_thresholds_for_class(self) -> None:
        tracker = WeightClassTracker()
        # No thresholds set for Lightweight
        thresholds = {
            "Welterweight": {"reach_cm": 190.0, "height_cm": 183.0},
        }
        tracker.set_class_thresholds(thresholds)
        tracker.update(
            _make_fight(
                fight_url="/fight/1",
                weight_class="Lightweight",
                event_date=date(2023, 1, 1),
            )
        )

        state = tracker.freeze()
        assert state.is_large_for_class("/fighter/a", reach_cm=200.0, height_cm=200.0) is False

    def test_large_for_class_uses_current_class(self) -> None:
        """After migration, top-quartile check uses the fighter's current class."""
        tracker = WeightClassTracker()
        thresholds = {
            "Lightweight": {"reach_cm": 185.0, "height_cm": 180.0},
            "Welterweight": {"reach_cm": 190.0, "height_cm": 183.0},
        }
        tracker.set_class_thresholds(thresholds)
        tracker.update(
            _make_fight(
                fight_url="/fight/1",
                weight_class="Lightweight",
                event_date=date(2023, 1, 1),
            )
        )
        tracker.update(
            _make_fight(
                fight_url="/fight/2",
                weight_class="Welterweight",
                event_date=date(2023, 6, 1),
            )
        )

        state = tracker.freeze()
        # 188 reach and 182 height: above LW thresholds, but below WW thresholds
        assert state.is_large_for_class("/fighter/a", reach_cm=188.0, height_cm=182.0) is False
        # Above WW thresholds
        assert state.is_large_for_class("/fighter/a", reach_cm=195.0, height_cm=185.0) is True


class TestWeightClassTrackerFreeze:
    """Tests for freeze immutability."""

    def test_freeze_returns_frozen_state(self) -> None:
        tracker = WeightClassTracker()
        tracker.update(
            _make_fight(weight_class="Lightweight", event_date=date(2023, 1, 1))
        )
        state = tracker.freeze()
        assert isinstance(state, WeightClassFrozenState)

    def test_frozen_state_is_immutable(self) -> None:
        tracker = WeightClassTracker()
        tracker.update(
            _make_fight(weight_class="Lightweight", event_date=date(2023, 1, 1))
        )
        state = tracker.freeze()

        with pytest.raises(AttributeError, match="frozen"):
            state.some_attr = "value"  # type: ignore[attr-defined]

    def test_freeze_snapshot_not_affected_by_later_updates(self) -> None:
        tracker = WeightClassTracker()
        tracker.update(
            _make_fight(
                fight_url="/fight/1",
                weight_class="Lightweight",
                event_date=date(2023, 1, 1),
            )
        )
        state_before = tracker.freeze()

        tracker.update(
            _make_fight(
                fight_url="/fight/2",
                weight_class="Welterweight",
                event_date=date(2023, 6, 1),
            )
        )
        state_after = tracker.freeze()

        # The earlier snapshot must be unaffected by the later update
        fighter_before = state_before.get_fighter_state("/fighter/a")
        fighter_after = state_after.get_fighter_state("/fighter/a")
        assert fighter_before is not None
        assert fighter_after is not None
        assert fighter_before.current_class == "Lightweight"
        assert fighter_after.current_class == "Welterweight"
        assert fighter_before.migration_count == 0
        assert fighter_after.migration_count == 1

    def test_unknown_fighter_returns_none(self) -> None:
        tracker = WeightClassTracker()
        state = tracker.freeze()
        assert state.get_fighter_state("/unknown") is None


class TestWeightClassTrackerProtocol:
    """Verify WeightClassTracker satisfies the StateComponent protocol."""

    def test_satisfies_state_component_protocol(self) -> None:
        from ufc_edge.features.contracts import StateComponent

        tracker = WeightClassTracker()
        assert isinstance(tracker, StateComponent)

    def test_freeze_returns_frozen_state_subclass(self) -> None:
        from ufc_edge.features.contracts import FrozenState

        tracker = WeightClassTracker()
        tracker.update(_make_fight())
        state = tracker.freeze()
        assert isinstance(state, FrozenState)
