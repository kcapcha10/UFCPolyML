"""Tests for Glicko2Tracker state component.

Verifies the Glicko-2 rating system implementation: correct initialization
at debut, rating deviation reduction after rated fights, RD growth during
inactivity, injury-stoppage neutrality, and freeze immutability.
"""

from __future__ import annotations

import math
import sys
import types
from datetime import date

import pytest

from ufc_edge.features.contracts import FightOutcomeView, FrozenState, StateComponent

# Sibling state-component modules are developed concurrently. If any import
# referenced by components/__init__.py is missing, stub it so this test runs standalone.
_career_mod = "ufc_edge.features.components.career"
if _career_mod not in sys.modules:
    _stub = types.ModuleType(_career_mod)
    _stub.CareerAccumulator = type("CareerAccumulator", (), {})  # type: ignore[attr-defined]
    sys.modules[_career_mod] = _stub

from ufc_edge.features.components.glicko2 import Glicko2Tracker  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_outcome(
    *,
    event_date: date = date(2024, 6, 1),
    fighter_a: str = "http://fighter/a",
    fighter_b: str = "http://fighter/b",
    winner: str | None = "http://fighter/a",
    method: str = "Decision",
) -> FightOutcomeView:
    """Build a FightOutcomeView with sensible defaults."""
    return FightOutcomeView(
        fight_url=f"http://fight/{event_date.isoformat()}",
        event_url=f"http://event/{event_date.isoformat()}",
        event_date=event_date,
        fighter_a_url=fighter_a,
        fighter_b_url=fighter_b,
        winner_url=winner,
        method=method,
        ending_round=3,
        ending_time="5:00",
        weight_class="Lightweight",
        bout_order=None,
    )


# ---------------------------------------------------------------------------
# Debut initialization
# ---------------------------------------------------------------------------


class TestGlicko2Debut:
    """New fighters initialize at standard Glicko-2 defaults."""

    def test_debut_rd_equals_350(self) -> None:
        """A fighter with no fights has RD=350 (the configured initial value)."""
        tracker = Glicko2Tracker(initial_mu=1500.0, initial_rd=350.0, tau=0.5)
        state = tracker.freeze()
        record = state.get_record("http://fighter/unknown")
        assert record.mu == 1500.0
        assert record.rd == 350.0

    def test_debut_fight_reduces_rd(self) -> None:
        """After a debut fight, RD should be less than the initial 350."""
        tracker = Glicko2Tracker(initial_mu=1500.0, initial_rd=350.0, tau=0.5)
        tracker.update(_make_outcome())
        state = tracker.freeze()
        record_a = state.get_record("http://fighter/a")
        assert record_a.rd < 350.0

    def test_debut_fight_moves_mu_from_initial(self) -> None:
        """After a debut fight, winner's mu should increase from initial value."""
        tracker = Glicko2Tracker(initial_mu=1500.0, initial_rd=350.0, tau=0.5)
        tracker.update(_make_outcome(winner="http://fighter/a"))
        state = tracker.freeze()
        record_a = state.get_record("http://fighter/a")
        assert record_a.mu > 1500.0


# ---------------------------------------------------------------------------
# Win reduces RD (confidence increases after observed outcomes)
# ---------------------------------------------------------------------------


class TestWinReducesRD:
    """Rating deviation should decrease after a rated fight outcome.

    Glicko-2's core insight: observing a fight result gives information about
    a fighter's true skill, so uncertainty (RD) decreases.
    """

    def test_winner_rd_decreases_after_fight(self) -> None:
        tracker = Glicko2Tracker(initial_mu=1500.0, initial_rd=350.0, tau=0.5)
        tracker.update(_make_outcome(winner="http://fighter/a"))
        state = tracker.freeze()
        record_a = state.get_record("http://fighter/a")
        assert record_a.rd < 350.0

    def test_loser_rd_decreases_after_fight(self) -> None:
        """Even the loser gains information — their RD decreases too."""
        tracker = Glicko2Tracker(initial_mu=1500.0, initial_rd=350.0, tau=0.5)
        tracker.update(_make_outcome(winner="http://fighter/a"))
        state = tracker.freeze()
        record_b = state.get_record("http://fighter/b")
        assert record_b.rd < 350.0

    def test_multiple_fights_further_reduce_rd(self) -> None:
        """More fights give more information; RD should continue declining."""
        tracker = Glicko2Tracker(initial_mu=1500.0, initial_rd=350.0, tau=0.5)
        tracker.update(
            _make_outcome(event_date=date(2024, 1, 1), winner="http://fighter/a")
        )
        state1 = tracker.freeze()
        rd_after_1 = state1.get_record("http://fighter/a").rd

        tracker.update(
            _make_outcome(
                event_date=date(2024, 3, 1),
                fighter_a="http://fighter/a",
                fighter_b="http://fighter/c",
                winner="http://fighter/a",
            )
        )
        state2 = tracker.freeze()
        rd_after_2 = state2.get_record("http://fighter/a").rd
        assert rd_after_2 < rd_after_1


# ---------------------------------------------------------------------------
# Inactivity increases RD (uncertainty grows without observed outcomes)
# ---------------------------------------------------------------------------


class TestInactivityIncreasesRD:
    """RD grows over time when a fighter is inactive.

    In Glicko-2, confidence in a rating decays toward maximum uncertainty
    when no new outcomes are observed. This reflects that a fighter's true
    skill may drift during extended absence.
    """

    def test_rd_grows_with_elapsed_periods(self) -> None:
        tracker = Glicko2Tracker(
            initial_mu=1500.0, initial_rd=350.0, tau=0.5, rating_period_days=30
        )
        tracker.update(
            _make_outcome(event_date=date(2024, 1, 1), winner="http://fighter/a")
        )
        state_post_fight = tracker.freeze()
        _ = state_post_fight.get_record("http://fighter/a").rd

        # Second fight 6 months later — RD grows before the update
        tracker.update(
            _make_outcome(
                event_date=date(2024, 7, 1),
                fighter_a="http://fighter/a",
                fighter_b="http://fighter/d",
                winner="http://fighter/a",
            )
        )
        state_active = tracker.freeze()
        rd_active = state_active.get_record("http://fighter/a").rd

        # Compare: same fighter with fights close together
        tracker2 = Glicko2Tracker(
            initial_mu=1500.0, initial_rd=350.0, tau=0.5, rating_period_days=30
        )
        tracker2.update(
            _make_outcome(event_date=date(2024, 1, 1), winner="http://fighter/a")
        )
        tracker2.update(
            _make_outcome(
                event_date=date(2024, 2, 1),
                fighter_a="http://fighter/a",
                fighter_b="http://fighter/d",
                winner="http://fighter/a",
            )
        )
        state_dense = tracker2.freeze()
        rd_dense = state_dense.get_record("http://fighter/a").rd

        # Fighter with the long gap should have higher resulting RD
        assert rd_active > rd_dense

    def test_rd_does_not_exceed_initial_rd(self) -> None:
        """RD growth is capped at the initial_rd value."""
        tracker = Glicko2Tracker(
            initial_mu=1500.0, initial_rd=350.0, tau=0.5, rating_period_days=30
        )
        tracker.update(
            _make_outcome(event_date=date(2020, 1, 1), winner="http://fighter/a")
        )
        # Enormous gap — 5 years
        tracker.update(
            _make_outcome(
                event_date=date(2025, 1, 1),
                fighter_a="http://fighter/a",
                fighter_b="http://fighter/e",
                winner="http://fighter/a",
            )
        )
        state = tracker.freeze()
        record = state.get_record("http://fighter/a")
        assert record.rd <= 350.0


# ---------------------------------------------------------------------------
# Injury stoppage — no rating update
# ---------------------------------------------------------------------------


class TestInjuryNoUpdate:
    """Injury stoppages and NC-injuries are uninformative about skill.

    When a fight ends by injury (doctor stoppage or accidental injury),
    neither fighter's rating should change — the outcome tells us nothing
    about relative ability.
    """

    def test_injury_stoppage_does_not_change_ratings(self) -> None:
        tracker = Glicko2Tracker(initial_mu=1500.0, initial_rd=350.0, tau=0.5)
        tracker.update(
            _make_outcome(event_date=date(2024, 1, 1), winner="http://fighter/a")
        )
        state_before = tracker.freeze()
        mu_a_before = state_before.get_record("http://fighter/a").mu
        mu_b_before = state_before.get_record("http://fighter/b").mu
        rd_a_before = state_before.get_record("http://fighter/a").rd
        rd_b_before = state_before.get_record("http://fighter/b").rd

        tracker.update(
            _make_outcome(
                event_date=date(2024, 2, 1),
                winner="http://fighter/a",
                method="Could Not Continue",
            )
        )
        state_after = tracker.freeze()
        assert state_after.get_record("http://fighter/a").mu == mu_a_before
        assert state_after.get_record("http://fighter/b").mu == mu_b_before
        assert state_after.get_record("http://fighter/a").rd == rd_a_before
        assert state_after.get_record("http://fighter/b").rd == rd_b_before

    def test_doctor_stoppage_injury_does_not_change_ratings(self) -> None:
        tracker = Glicko2Tracker(initial_mu=1500.0, initial_rd=350.0, tau=0.5)
        tracker.update(
            _make_outcome(event_date=date(2024, 1, 1), winner="http://fighter/a")
        )
        state_before = tracker.freeze()
        mu_a = state_before.get_record("http://fighter/a").mu

        tracker.update(
            _make_outcome(
                event_date=date(2024, 2, 1),
                winner="http://fighter/a",
                method="TKO - Doctor's Stoppage (Injury)",
            )
        )
        state_after = tracker.freeze()
        assert state_after.get_record("http://fighter/a").mu == mu_a

    def test_no_contest_does_not_change_ratings(self) -> None:
        """No-contests (including NC-injury) are also uninformative."""
        tracker = Glicko2Tracker(initial_mu=1500.0, initial_rd=350.0, tau=0.5)
        tracker.update(
            _make_outcome(event_date=date(2024, 1, 1), winner="http://fighter/a")
        )
        state_before = tracker.freeze()
        mu_a = state_before.get_record("http://fighter/a").mu

        tracker.update(
            _make_outcome(
                event_date=date(2024, 2, 1),
                winner=None,
                method="No Contest",
            )
        )
        state_after = tracker.freeze()
        assert state_after.get_record("http://fighter/a").mu == mu_a


# ---------------------------------------------------------------------------
# Freeze immutability
# ---------------------------------------------------------------------------


class TestFreezeImmutability:
    """Frozen state snapshots must be deeply immutable."""

    def test_frozen_state_is_frozenstate_subclass(self) -> None:
        tracker = Glicko2Tracker(initial_mu=1500.0, initial_rd=350.0, tau=0.5)
        state = tracker.freeze()
        assert isinstance(state, FrozenState)

    def test_setattr_raises_on_frozen_state(self) -> None:
        tracker = Glicko2Tracker(initial_mu=1500.0, initial_rd=350.0, tau=0.5)
        tracker.update(_make_outcome())
        state = tracker.freeze()
        with pytest.raises(AttributeError):
            state.some_attr = "anything"  # type: ignore[attr-defined]

    def test_freeze_does_not_share_internal_state(self) -> None:
        """Updates after freeze must not affect the previously frozen snapshot."""
        tracker = Glicko2Tracker(initial_mu=1500.0, initial_rd=350.0, tau=0.5)
        tracker.update(
            _make_outcome(event_date=date(2024, 1, 1), winner="http://fighter/a")
        )
        frozen1 = tracker.freeze()
        mu_frozen1 = frozen1.get_record("http://fighter/a").mu

        tracker.update(
            _make_outcome(
                event_date=date(2024, 3, 1),
                fighter_a="http://fighter/a",
                fighter_b="http://fighter/c",
                winner="http://fighter/c",
            )
        )
        assert frozen1.get_record("http://fighter/a").mu == mu_frozen1


# ---------------------------------------------------------------------------
# Glicko-2 algorithm correctness
# ---------------------------------------------------------------------------


class TestGlicko2AlgorithmCorrectness:
    """Validates that the implementation follows Glickman's paper formulas.

    The Glicko-2 algorithm converts ratings to an internal scale (mu/173.7178),
    applies the g(phi) weighting function, computes expected score E, and
    updates rating, deviation, and volatility simultaneously.
    """

    def test_winner_rating_increases(self) -> None:
        tracker = Glicko2Tracker(initial_mu=1500.0, initial_rd=350.0, tau=0.5)
        tracker.update(_make_outcome(winner="http://fighter/a"))
        state = tracker.freeze()
        assert state.get_record("http://fighter/a").mu > 1500.0

    def test_loser_rating_decreases(self) -> None:
        tracker = Glicko2Tracker(initial_mu=1500.0, initial_rd=350.0, tau=0.5)
        tracker.update(_make_outcome(winner="http://fighter/a"))
        state = tracker.freeze()
        assert state.get_record("http://fighter/b").mu < 1500.0

    def test_equal_opponents_approximately_symmetric(self) -> None:
        """Against equal opponents, winner gain and loser loss are comparable.

        Not exactly equal because Glicko-2 updates each fighter independently
        with their own volatility adjustment, creating slight asymmetry.
        """
        tracker = Glicko2Tracker(initial_mu=1500.0, initial_rd=350.0, tau=0.5)
        tracker.update(_make_outcome(winner="http://fighter/a"))
        state = tracker.freeze()
        gain = state.get_record("http://fighter/a").mu - 1500.0
        loss = 1500.0 - state.get_record("http://fighter/b").mu
        # Both should be positive and of similar magnitude
        assert gain > 0
        assert loss > 0
        assert abs(gain - loss) < 50.0

    def test_volatility_is_positive(self) -> None:
        tracker = Glicko2Tracker(initial_mu=1500.0, initial_rd=350.0, tau=0.5)
        tracker.update(_make_outcome(winner="http://fighter/a"))
        state = tracker.freeze()
        assert state.get_record("http://fighter/a").sigma > 0
        assert state.get_record("http://fighter/b").sigma > 0

    def test_glicko2_scale_factor(self) -> None:
        """The Glicko-2 scale factor is 400/ln(10) ≈ 173.7178."""
        scale = 400.0 / math.log(10)
        assert abs(scale - 173.7178) < 0.001

    def test_draw_excluded_from_update(self) -> None:
        """Draws (winner_url=None with non-injury method) skip the update."""
        tracker = Glicko2Tracker(initial_mu=1500.0, initial_rd=350.0, tau=0.5)
        tracker.update(
            _make_outcome(event_date=date(2024, 1, 1), winner="http://fighter/a")
        )
        state_before = tracker.freeze()
        mu_a = state_before.get_record("http://fighter/a").mu

        tracker.update(
            _make_outcome(event_date=date(2024, 2, 1), winner=None, method="Draw")
        )
        state_after = tracker.freeze()
        assert state_after.get_record("http://fighter/a").mu == mu_a


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Glicko2Tracker must satisfy the StateComponent protocol."""

    def test_has_update_method(self) -> None:
        tracker = Glicko2Tracker(initial_mu=1500.0, initial_rd=350.0, tau=0.5)
        assert hasattr(tracker, "update")
        assert callable(tracker.update)

    def test_has_freeze_method(self) -> None:
        tracker = Glicko2Tracker(initial_mu=1500.0, initial_rd=350.0, tau=0.5)
        assert hasattr(tracker, "freeze")
        assert callable(tracker.freeze)

    def test_satisfies_state_component_protocol(self) -> None:
        assert isinstance(
            Glicko2Tracker(initial_mu=1500.0, initial_rd=350.0, tau=0.5),
            StateComponent,
        )
