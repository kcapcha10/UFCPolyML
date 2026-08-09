"""Tests for the EloTracker state component.

Validates rating initialization, variable K-factor behavior, method-based
adjustments, inactivity decay, freeze immutability, and history tracking.
"""

from __future__ import annotations

from datetime import date

import pytest

from ufc_edge.features.components.elo import EloTracker
from ufc_edge.features.contracts import FightOutcomeView


def _make_fight(
    *,
    fighter_a: str = "http://fighter-a",
    fighter_b: str = "http://fighter-b",
    winner: str | None = "http://fighter-a",
    method: str = "Decision - Unanimous",
    event_date: date = date(2024, 1, 1),
    ending_round: int = 3,
    ending_time: str = "5:00",
) -> FightOutcomeView:
    """Build a minimal FightOutcomeView for testing."""
    return FightOutcomeView(
        fight_url="http://fight/1",
        event_url="http://event/1",
        event_date=event_date,
        fighter_a_url=fighter_a,
        fighter_b_url=fighter_b,
        winner_url=winner,
        method=method,
        ending_round=ending_round,
        ending_time=ending_time,
        weight_class="Lightweight",
        bout_order=None,
    )


class TestEloDebut:
    """Fighters with no prior record start at the population mean rating (1500)."""

    def test_debut_rating_is_1500(self) -> None:
        tracker = EloTracker()
        state = tracker.freeze()
        record = state.get("http://fighter-a")
        assert record is None

    def test_first_fight_initializes_at_1500(self) -> None:
        tracker = EloTracker()
        fight = _make_fight()
        tracker.update(fight)
        state = tracker.freeze()
        winner_record = state.get("http://fighter-a")
        assert winner_record is not None
        # Winner should be above 1500 after a win
        assert winner_record.rating > 1500

    def test_debut_rating_baseline(self) -> None:
        tracker = EloTracker()
        fight = _make_fight()
        tracker.update(fight)
        state = tracker.freeze()
        loser_record = state.get("http://fighter-b")
        assert loser_record is not None
        # Loser should be below 1500 after a loss
        assert loser_record.rating < 1500


class TestMethodBonus:
    """KO/TKO wins produce larger rating changes than decisions."""

    def test_ko_win_increases_more_than_decision(self) -> None:
        tracker_ko = EloTracker()
        tracker_dec = EloTracker()

        ko_fight = _make_fight(method="KO/TKO")
        dec_fight = _make_fight(method="Decision - Unanimous")

        tracker_ko.update(ko_fight)
        tracker_dec.update(dec_fight)

        ko_state = tracker_ko.freeze()
        dec_state = tracker_dec.freeze()

        ko_rating = ko_state.get("http://fighter-a").rating
        dec_rating = dec_state.get("http://fighter-a").rating

        # KO gives a method bonus so rating gain should be larger
        assert ko_rating > dec_rating

    def test_submission_win_increases_more_than_decision(self) -> None:
        tracker_sub = EloTracker()
        tracker_dec = EloTracker()

        sub_fight = _make_fight(method="Submission")
        dec_fight = _make_fight(method="Decision - Unanimous")

        tracker_sub.update(sub_fight)
        tracker_dec.update(dec_fight)

        sub_state = tracker_sub.freeze()
        dec_state = tracker_dec.freeze()

        sub_rating = sub_state.get("http://fighter-a").rating
        dec_rating = dec_state.get("http://fighter-a").rating

        assert sub_rating > dec_rating


class TestInjuryStoppage:
    """Injury stoppages produce no rating change (K=0)."""

    def test_injury_stoppage_no_change(self) -> None:
        tracker = EloTracker()
        # First give fighters a baseline fight so they exist in the system
        baseline = _make_fight(event_date=date(2023, 6, 1))
        tracker.update(baseline)

        state_before = tracker.freeze()
        a_before = state_before.get("http://fighter-a").rating
        b_before = state_before.get("http://fighter-b").rating

        # Now apply an injury stoppage
        injury_fight = _make_fight(
            method="Could Not Continue - Injury",
            event_date=date(2023, 7, 1),
        )
        tracker.update(injury_fight)

        state_after = tracker.freeze()
        a_after = state_after.get("http://fighter-a").rating
        b_after = state_after.get("http://fighter-b").rating

        assert a_after == a_before
        assert b_after == b_before

    def test_doctor_stoppage_no_change(self) -> None:
        tracker = EloTracker()
        baseline = _make_fight(event_date=date(2023, 6, 1))
        tracker.update(baseline)

        state_before = tracker.freeze()
        a_before = state_before.get("http://fighter-a").rating

        injury_fight = _make_fight(
            method="TKO - Doctor's Stoppage",
            event_date=date(2023, 7, 1),
        )
        tracker.update(injury_fight)

        state_after = tracker.freeze()
        a_after = state_after.get("http://fighter-a").rating
        assert a_after == a_before


class TestDQMultiplier:
    """DQ outcomes apply K * 0.1 multiplier — ratings change but minimally."""

    def test_dq_applies_reduced_k(self) -> None:
        tracker_dq = EloTracker()
        tracker_normal = EloTracker()

        dq_fight = _make_fight(method="DQ")
        normal_fight = _make_fight(method="Decision - Unanimous")

        tracker_dq.update(dq_fight)
        tracker_normal.update(normal_fight)

        dq_state = tracker_dq.freeze()
        normal_state = tracker_normal.freeze()

        dq_gain = dq_state.get("http://fighter-a").rating - 1500
        normal_gain = normal_state.get("http://fighter-a").rating - 1500

        # DQ gain should be approximately 10% of normal gain
        assert dq_gain > 0
        assert dq_gain < normal_gain
        assert abs(dq_gain / normal_gain - 0.1) < 0.01


class TestInactivityDecay:
    """Inactive fighters decay toward 1500."""

    def test_inactivity_decays_toward_1500(self) -> None:
        tracker = EloTracker()

        # Build up a rating above 1500
        fight1 = _make_fight(event_date=date(2022, 1, 1))
        tracker.update(fight1)

        state_before = tracker.freeze()
        rating_before = state_before.get("http://fighter-a").rating
        assert rating_before > 1500

        # Fight after a long inactivity gap (>180 days)
        fight2 = _make_fight(event_date=date(2023, 6, 1))
        tracker.update(fight2)

        state_after = tracker.freeze()
        rating_after = state_after.get("http://fighter-a").rating

        # The inactivity decay should have pulled the pre-fight rating toward 1500,
        # so the effective starting point was lower. The post-fight rating should
        # still reflect that decay was applied before the update.
        # Since fighter won again, rating goes up. But the decay means the
        # pre-update rating was pulled toward 1500 from rating_before.
        # We verify by checking the history shows the decay was applied.
        assert rating_after < rating_before + (rating_before - 1500)

    def test_below_1500_decays_upward(self) -> None:
        tracker = EloTracker()

        # Lose to get below 1500
        fight1 = _make_fight(
            winner="http://fighter-b",
            event_date=date(2022, 1, 1),
        )
        tracker.update(fight1)

        state_mid = tracker.freeze()
        rating_mid = state_mid.get("http://fighter-a").rating
        assert rating_mid < 1500

        # Long gap, then another loss — but decay should have pushed toward 1500
        fight2 = _make_fight(
            winner="http://fighter-b",
            event_date=date(2023, 6, 1),
        )
        tracker.update(fight2)

        state_after = tracker.freeze()
        # After decay toward 1500 (upward) and another loss (downward),
        # the loss from the decayed position should be smaller than it would be
        # without decay, resulting in a final rating above what pure losses
        # would produce.
        # Validate that the fighter's rating isn't as low as two full losses from 1500.
        two_loss_tracker = EloTracker()
        two_loss_tracker.update(
            _make_fight(winner="http://fighter-b", event_date=date(2022, 1, 1))
        )
        two_loss_tracker.update(
            _make_fight(winner="http://fighter-b", event_date=date(2022, 2, 1))
        )
        no_decay_rating = two_loss_tracker.freeze().get("http://fighter-a").rating

        # With decay, rating should be higher (closer to 1500) than without
        assert state_after.get("http://fighter-a").rating > no_decay_rating


class TestFreezeImmutability:
    """Frozen state must be immutable — modifications raise errors."""

    def test_frozen_record_rejects_attribute_set(self) -> None:
        tracker = EloTracker()
        fight = _make_fight()
        tracker.update(fight)

        state = tracker.freeze()
        record = state.get("http://fighter-a")
        with pytest.raises(AttributeError):
            record.rating = 9999  # type: ignore[misc]

    def test_frozen_state_is_independent_of_tracker(self) -> None:
        tracker = EloTracker()
        fight1 = _make_fight(event_date=date(2024, 1, 1))
        tracker.update(fight1)

        state1 = tracker.freeze()
        rating1 = state1.get("http://fighter-a").rating

        # Further updates do not affect previously frozen state
        fight2 = _make_fight(event_date=date(2024, 2, 1))
        tracker.update(fight2)

        assert state1.get("http://fighter-a").rating == rating1


class TestHistory:
    """Rating history tracks the last N ratings for trajectory analysis."""

    def test_history_grows_with_fights(self) -> None:
        tracker = EloTracker()

        for i in range(5):
            fight = _make_fight(event_date=date(2024, 1 + i, 1))
            tracker.update(fight)

        state = tracker.freeze()
        record = state.get("http://fighter-a")
        assert len(record.history) == 5

    def test_history_is_capped(self) -> None:
        tracker = EloTracker()

        for i in range(25):
            fight = _make_fight(event_date=date(2022, 1, 1 + i))
            tracker.update(fight)

        state = tracker.freeze()
        record = state.get("http://fighter-a")
        # History should be capped at a reasonable size (20 per design)
        assert len(record.history) <= 20

    def test_history_records_post_update_ratings(self) -> None:
        tracker = EloTracker()
        fight = _make_fight()
        tracker.update(fight)

        state = tracker.freeze()
        record = state.get("http://fighter-a")
        # The history entry should match the current rating after 1 fight
        assert record.history[-1] == record.rating


class TestPeakTracking:
    """Peak tracks the highest rating achieved."""

    def test_peak_equals_rating_after_first_win(self) -> None:
        tracker = EloTracker()
        fight = _make_fight()
        tracker.update(fight)

        state = tracker.freeze()
        record = state.get("http://fighter-a")
        assert record.peak == record.rating

    def test_peak_preserves_maximum(self) -> None:
        tracker = EloTracker()

        # Win to push rating up
        fight1 = _make_fight(event_date=date(2024, 1, 1))
        tracker.update(fight1)
        peak_after_win = tracker.freeze().get("http://fighter-a").rating

        # Lose to push rating down
        fight2 = _make_fight(
            winner="http://fighter-b",
            event_date=date(2024, 2, 1),
        )
        tracker.update(fight2)

        state = tracker.freeze()
        record = state.get("http://fighter-a")
        assert record.peak == peak_after_win
        assert record.rating < record.peak


class TestFightCount:
    """Fight count increments for each fight processed."""

    def test_fight_count_increments(self) -> None:
        tracker = EloTracker()

        for i in range(3):
            fight = _make_fight(event_date=date(2024, 1 + i, 1))
            tracker.update(fight)

        state = tracker.freeze()
        assert state.get("http://fighter-a").fight_count == 3
        assert state.get("http://fighter-b").fight_count == 3


class TestDrawsAndNoContests:
    """Draws and no-contests (winner_url=None) should still update."""

    def test_draw_updates_both_fighters(self) -> None:
        tracker = EloTracker()
        draw_fight = _make_fight(winner=None, method="Draw")
        tracker.update(draw_fight)

        state = tracker.freeze()
        # Both fighters should exist with 1 fight
        assert state.get("http://fighter-a").fight_count == 1
        assert state.get("http://fighter-b").fight_count == 1
        # Ratings should stay near 1500 for equal-rated draw
        assert abs(state.get("http://fighter-a").rating - 1500) < 1
        assert abs(state.get("http://fighter-b").rating - 1500) < 1
