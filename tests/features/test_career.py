"""Tests for CareerAccumulator state component.

Validates win/loss counting, streak tracking, finish-type breakdown,
windowed fight counts, debut detection, and freeze immutability.
"""

from __future__ import annotations

from datetime import date

import pytest

from ufc_edge.features.components.career import CareerAccumulator
from ufc_edge.features.contracts import FightOutcomeView

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _outcome(
    *,
    fight_url: str = "http://fight/1",
    event_url: str = "http://event/1",
    event_date: date = date(2024, 6, 1),
    fighter_a_url: str = "http://fighter/a",
    fighter_b_url: str = "http://fighter/b",
    winner_url: str | None = "http://fighter/a",
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


FIGHTER_A = "http://fighter/a"
FIGHTER_B = "http://fighter/b"
FIGHTER_C = "http://fighter/c"


# ---------------------------------------------------------------------------
# Win/loss counting
# ---------------------------------------------------------------------------


class TestWinLossCounts:
    """Basic win and loss accumulation."""

    def test_initial_state_has_zero_counts(self) -> None:
        acc = CareerAccumulator()
        snap = acc.freeze()
        state = snap.get(FIGHTER_A)
        assert state is None

    def test_single_win_increments_win_count(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(winner_url=FIGHTER_A))
        snap = acc.freeze()
        state = snap.get(FIGHTER_A)
        assert state is not None
        assert state.wins == 1
        assert state.losses == 0

    def test_single_loss_increments_loss_count(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(winner_url=FIGHTER_B))
        snap = acc.freeze()
        state = snap.get(FIGHTER_A)
        assert state is not None
        assert state.wins == 0
        assert state.losses == 1

    def test_draw_does_not_increment_win_or_loss(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(winner_url=None, method="Draw"))
        snap = acc.freeze()
        state = snap.get(FIGHTER_A)
        assert state is not None
        assert state.wins == 0
        assert state.losses == 0
        assert state.draws == 1

    def test_no_contest_does_not_increment_win_or_loss(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(winner_url=None, method="Could not continue"))
        snap = acc.freeze()
        state = snap.get(FIGHTER_A)
        assert state is not None
        assert state.wins == 0
        assert state.losses == 0
        assert state.no_contests == 1

    def test_multiple_fights_accumulate(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(fight_url="http://f/1", winner_url=FIGHTER_A))
        acc.update(
            _outcome(
                fight_url="http://f/2",
                event_date=date(2024, 7, 1),
                winner_url=FIGHTER_B,
            )
        )
        acc.update(
            _outcome(
                fight_url="http://f/3",
                event_date=date(2024, 8, 1),
                winner_url=FIGHTER_A,
            )
        )
        snap = acc.freeze()
        state = snap.get(FIGHTER_A)
        assert state is not None
        assert state.wins == 2
        assert state.losses == 1


# ---------------------------------------------------------------------------
# Streak tracking
# ---------------------------------------------------------------------------


class TestStreakTracking:
    """Streak sign convention: positive = win streak, negative = loss streak."""

    def test_single_win_gives_positive_streak(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(winner_url=FIGHTER_A))
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).current_streak == 1

    def test_single_loss_gives_negative_streak(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(winner_url=FIGHTER_B))
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).current_streak == -1

    def test_consecutive_wins_build_positive_streak(self) -> None:
        acc = CareerAccumulator()
        for i in range(4):
            acc.update(
                _outcome(
                    fight_url=f"http://f/{i}",
                    event_date=date(2024, 1 + i, 1),
                    winner_url=FIGHTER_A,
                )
            )
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).current_streak == 4

    def test_loss_resets_win_streak_to_negative(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(fight_url="http://f/1", winner_url=FIGHTER_A))
        acc.update(
            _outcome(fight_url="http://f/2", winner_url=FIGHTER_A, event_date=date(2024, 7, 1))
        )
        acc.update(
            _outcome(
                fight_url="http://f/3",
                event_date=date(2024, 8, 1),
                winner_url=FIGHTER_B,
            )
        )
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).current_streak == -1

    def test_win_resets_loss_streak_to_positive(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(fight_url="http://f/1", winner_url=FIGHTER_B))
        acc.update(
            _outcome(fight_url="http://f/2", winner_url=FIGHTER_B, event_date=date(2024, 7, 1))
        )
        acc.update(
            _outcome(
                fight_url="http://f/3",
                event_date=date(2024, 8, 1),
                winner_url=FIGHTER_A,
            )
        )
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).current_streak == 1

    def test_draw_does_not_affect_streak(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(fight_url="http://f/1", winner_url=FIGHTER_A))
        acc.update(
            _outcome(
                fight_url="http://f/2",
                event_date=date(2024, 7, 1),
                winner_url=None,
                method="Draw",
            )
        )
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).current_streak == 1


# ---------------------------------------------------------------------------
# Finish type counts
# ---------------------------------------------------------------------------


class TestFinishCounts:
    """Finish-type breakdown: KO/TKO, submission, decision wins/losses."""

    def test_ko_win_increments_ko_wins(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(winner_url=FIGHTER_A, method="KO/TKO"))
        snap = acc.freeze()
        state = snap.get(FIGHTER_A)
        assert state.ko_wins == 1
        assert state.submission_wins == 0
        assert state.decision_wins == 0

    def test_submission_win_increments_submission_wins(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(winner_url=FIGHTER_A, method="Submission"))
        snap = acc.freeze()
        state = snap.get(FIGHTER_A)
        assert state.submission_wins == 1
        assert state.ko_wins == 0

    def test_decision_win_increments_decision_wins(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(winner_url=FIGHTER_A, method="Decision - Unanimous"))
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).decision_wins == 1

    def test_split_decision_counts_as_decision(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(winner_url=FIGHTER_A, method="Decision - Split"))
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).decision_wins == 1

    def test_ko_loss_increments_times_finished_by_ko(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(winner_url=FIGHTER_B, method="KO/TKO"))
        snap = acc.freeze()
        state = snap.get(FIGHTER_A)
        assert state.times_finished_by_ko == 1

    def test_submission_loss_increments_times_finished_by_sub(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(winner_url=FIGHTER_B, method="Submission"))
        snap = acc.freeze()
        state = snap.get(FIGHTER_A)
        assert state.times_finished_by_sub == 1

    def test_decision_loss_does_not_increment_finish_counts(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(winner_url=FIGHTER_B, method="Decision - Unanimous"))
        snap = acc.freeze()
        state = snap.get(FIGHTER_A)
        assert state.times_finished_by_ko == 0
        assert state.times_finished_by_sub == 0

    def test_round_one_finish_tracked(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(winner_url=FIGHTER_A, method="KO/TKO", ending_round=1))
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).round_one_finishes == 1

    def test_round_two_finish_not_counted_as_round_one(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(winner_url=FIGHTER_A, method="KO/TKO", ending_round=2))
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).round_one_finishes == 0

    def test_been_finished_round_one_tracked(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(winner_url=FIGHTER_B, method="KO/TKO", ending_round=1))
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).been_finished_round_one is True

    def test_not_been_finished_round_one_if_later_round(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(winner_url=FIGHTER_B, method="KO/TKO", ending_round=3))
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).been_finished_round_one is False


# ---------------------------------------------------------------------------
# Windowed fight counts
# ---------------------------------------------------------------------------


class TestWindowedFightCounts:
    """Per-window fight counts (12mo, 3yr, 5yr) respect event dates."""

    def test_fight_within_12_months_counted(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(event_date=date(2024, 1, 15), fight_url="http://f/1"))
        snap = acc.freeze()
        state = snap.get(FIGHTER_A)
        # Relative to the state, windows are computed at query time by emitters,
        # but the component stores fight dates for the emitter to compute windows.
        assert date(2024, 1, 15) in state.fight_dates

    def test_all_fight_dates_recorded(self) -> None:
        acc = CareerAccumulator()
        dates = [date(2022, 1, 1), date(2023, 6, 15), date(2024, 3, 10)]
        for i, d in enumerate(dates):
            acc.update(_outcome(event_date=d, fight_url=f"http://f/{i}"))
        snap = acc.freeze()
        state = snap.get(FIGHTER_A)
        assert len(state.fight_dates) == 3
        for d in dates:
            assert d in state.fight_dates

    def test_fights_last_12mo_computed_from_dates(self) -> None:
        acc = CareerAccumulator()
        # 4 fights, two within 12 months of reference date 2024-06-01
        fights = [
            date(2022, 1, 1),
            date(2023, 3, 1),
            date(2023, 9, 1),
            date(2024, 3, 1),
        ]
        for i, d in enumerate(fights):
            acc.update(_outcome(event_date=d, fight_url=f"http://f/{i}"))
        snap = acc.freeze()
        state = snap.get(FIGHTER_A)
        reference = date(2024, 6, 1)
        count_12mo = sum(1 for d in state.fight_dates if (reference - d).days <= 365)
        assert count_12mo == 2  # 2023-09-01 and 2024-03-01

    def test_fights_last_3yr_computed_from_dates(self) -> None:
        acc = CareerAccumulator()
        fights = [
            date(2020, 1, 1),
            date(2022, 1, 1),
            date(2023, 6, 1),
            date(2024, 1, 1),
        ]
        for i, d in enumerate(fights):
            acc.update(_outcome(event_date=d, fight_url=f"http://f/{i}"))
        snap = acc.freeze()
        state = snap.get(FIGHTER_A)
        reference = date(2024, 6, 1)
        count_3yr = sum(1 for d in state.fight_dates if (reference - d).days <= 3 * 365)
        assert count_3yr == 3  # 2022-01-01, 2023-06-01, 2024-01-01

    def test_fights_last_5yr_computed_from_dates(self) -> None:
        acc = CareerAccumulator()
        fights = [
            date(2018, 1, 1),
            date(2020, 1, 1),
            date(2022, 6, 1),
            date(2024, 1, 1),
        ]
        for i, d in enumerate(fights):
            acc.update(_outcome(event_date=d, fight_url=f"http://f/{i}"))
        snap = acc.freeze()
        state = snap.get(FIGHTER_A)
        reference = date(2024, 6, 1)
        count_5yr = sum(1 for d in state.fight_dates if (reference - d).days <= 5 * 365)
        assert count_5yr == 3  # 2020-01-01, 2022-06-01, 2024-01-01


# ---------------------------------------------------------------------------
# Last fight date and method
# ---------------------------------------------------------------------------


class TestLastFightTracking:
    """Last fight date and method are tracked for each fighter."""

    def test_last_fight_date_after_single_fight(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(event_date=date(2024, 6, 1)))
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).last_fight_date == date(2024, 6, 1)

    def test_last_fight_date_updates_with_later_fight(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(fight_url="http://f/1", event_date=date(2024, 1, 1)))
        acc.update(_outcome(fight_url="http://f/2", event_date=date(2024, 6, 1)))
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).last_fight_date == date(2024, 6, 1)

    def test_last_fight_method_tracked(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(method="KO/TKO"))
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).last_fight_method == "KO/TKO"

    def test_last_fight_method_updates(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(fight_url="http://f/1", method="KO/TKO"))
        acc.update(
            _outcome(
                fight_url="http://f/2",
                event_date=date(2024, 7, 1),
                method="Submission",
            )
        )
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).last_fight_method == "Submission"


# ---------------------------------------------------------------------------
# Debut tracking
# ---------------------------------------------------------------------------


class TestDebutTracking:
    """UFC debut detection — first fight in the system is debut."""

    def test_first_fight_marks_debut_date(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(event_date=date(2024, 3, 15)))
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).debut_date == date(2024, 3, 15)

    def test_debut_date_does_not_change_on_subsequent_fights(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(fight_url="http://f/1", event_date=date(2024, 1, 1)))
        acc.update(_outcome(fight_url="http://f/2", event_date=date(2024, 6, 1)))
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).debut_date == date(2024, 1, 1)

    def test_total_fights_count(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(fight_url="http://f/1"))
        acc.update(_outcome(fight_url="http://f/2", event_date=date(2024, 7, 1)))
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).total_fights == 2


# ---------------------------------------------------------------------------
# Weight class history
# ---------------------------------------------------------------------------


class TestWeightClassHistory:
    """Weight class history tracks unique classes the fighter has competed in."""

    def test_single_weight_class_recorded(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(weight_class="Lightweight"))
        snap = acc.freeze()
        assert "Lightweight" in snap.get(FIGHTER_A).weight_classes

    def test_multiple_weight_classes_recorded(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(fight_url="http://f/1", weight_class="Lightweight"))
        acc.update(
            _outcome(
                fight_url="http://f/2",
                event_date=date(2024, 7, 1),
                weight_class="Welterweight",
            )
        )
        snap = acc.freeze()
        wcs = snap.get(FIGHTER_A).weight_classes
        assert "Lightweight" in wcs
        assert "Welterweight" in wcs

    def test_duplicate_weight_class_not_doubled(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(fight_url="http://f/1", weight_class="Lightweight"))
        acc.update(
            _outcome(
                fight_url="http://f/2",
                event_date=date(2024, 7, 1),
                weight_class="Lightweight",
            )
        )
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).weight_classes.count("Lightweight") == 1


# ---------------------------------------------------------------------------
# Freeze immutability
# ---------------------------------------------------------------------------


class TestFreezeImmutability:
    """Frozen snapshots must be immutable — no mutation after freeze()."""

    def test_frozen_snapshot_rejects_attribute_assignment(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(winner_url=FIGHTER_A))
        snap = acc.freeze()
        state = snap.get(FIGHTER_A)
        with pytest.raises(AttributeError):
            state.wins = 99  # type: ignore[misc]

    def test_frozen_snapshot_independent_of_further_updates(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(fight_url="http://f/1", winner_url=FIGHTER_A))
        snap = acc.freeze()
        # Update the accumulator after freezing
        acc.update(
            _outcome(
                fight_url="http://f/2",
                event_date=date(2024, 7, 1),
                winner_url=FIGHTER_A,
            )
        )
        # Original snapshot unchanged
        assert snap.get(FIGHTER_A).wins == 1

    def test_multiple_freezes_are_independent(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(fight_url="http://f/1", winner_url=FIGHTER_A))
        snap1 = acc.freeze()
        acc.update(
            _outcome(
                fight_url="http://f/2",
                event_date=date(2024, 7, 1),
                winner_url=FIGHTER_A,
            )
        )
        snap2 = acc.freeze()
        assert snap1.get(FIGHTER_A).wins == 1
        assert snap2.get(FIGHTER_A).wins == 2

    def test_fight_dates_in_snapshot_not_mutable(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(event_date=date(2024, 1, 1)))
        snap = acc.freeze()
        state = snap.get(FIGHTER_A)
        # The returned dates tuple should not be appendable
        assert isinstance(state.fight_dates, tuple)

    def test_weight_classes_in_snapshot_not_mutable(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(weight_class="Lightweight"))
        snap = acc.freeze()
        state = snap.get(FIGHTER_A)
        assert isinstance(state.weight_classes, tuple)


# ---------------------------------------------------------------------------
# Both fighters updated per fight
# ---------------------------------------------------------------------------


class TestBothFightersUpdated:
    """Both fighters in a bout get their records updated."""

    def test_winner_and_loser_both_tracked(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(winner_url=FIGHTER_A))
        snap = acc.freeze()
        assert snap.get(FIGHTER_A) is not None
        assert snap.get(FIGHTER_B) is not None

    def test_winner_gets_win_loser_gets_loss(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(winner_url=FIGHTER_A))
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).wins == 1
        assert snap.get(FIGHTER_A).losses == 0
        assert snap.get(FIGHTER_B).wins == 0
        assert snap.get(FIGHTER_B).losses == 1

    def test_draw_gives_both_a_draw(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(winner_url=None, method="Draw"))
        snap = acc.freeze()
        assert snap.get(FIGHTER_A).draws == 1
        assert snap.get(FIGHTER_B).draws == 1


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """CareerAccumulator satisfies the StateComponent protocol."""

    def test_implements_state_component(self) -> None:
        from ufc_edge.features.contracts import StateComponent

        acc = CareerAccumulator()
        assert isinstance(acc, StateComponent)

    def test_freeze_returns_frozen_state(self) -> None:
        from ufc_edge.features.contracts import FrozenState

        acc = CareerAccumulator()
        snap = acc.freeze()
        assert isinstance(snap, FrozenState)
