"""Tests for ActivityEmitter.

Validates days-since-last-fight calculation, windowed fight counts, total UFC
fights, injury-stoppage detection, age-by-inactivity interaction, and
inactivity tier bucketing with exact boundary behavior.
"""

from __future__ import annotations

from datetime import date

import pytest

from ufc_edge.features.components.career import CareerAccumulator, CareerSnapshot
from ufc_edge.features.contracts import EmitContext, FighterProfile, FightOutcomeView
from ufc_edge.features.emitters.activity import ActivityEmitter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIGHTER_A = "http://fighter/a"
FIGHTER_B = "http://fighter/b"


def _outcome(
    *,
    fight_url: str = "http://fight/1",
    event_url: str = "http://event/1",
    event_date: date = date(2024, 6, 1),
    fighter_a_url: str = FIGHTER_A,
    fighter_b_url: str = FIGHTER_B,
    winner_url: str | None = FIGHTER_A,
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


def _build_context(
    snapshot: CareerSnapshot,
    *,
    fighter_url: str = FIGHTER_A,
    opponent_url: str = FIGHTER_B,
    event_date: date = date(2024, 6, 1),
    dob: date | None = date(1990, 1, 1),
) -> EmitContext:
    """Build a minimal EmitContext with a career snapshot."""
    return EmitContext(
        fighter_url=fighter_url,
        fighter_profile=FighterProfile(fighter_url=fighter_url, dob=dob),
        opponent_url=opponent_url,
        opponent_profile=FighterProfile(fighter_url=opponent_url),
        event_date=event_date,
        event_url="http://event/test",
        weight_class="Lightweight",
        fight_url="http://fight/test",
        bout_order=None,
        components={"career": snapshot},
    )


# ---------------------------------------------------------------------------
# Debut fighter tests
# ---------------------------------------------------------------------------


class TestDebutFighter:
    """A fighter with no prior fights (debut) produces None for days_since_last."""

    def test_debut_fighter_days_since_last_is_none(self) -> None:
        acc = CareerAccumulator()
        snapshot = acc.freeze()
        ctx = _build_context(snapshot)
        emitter = ActivityEmitter()
        result = emitter.emit(ctx)
        assert result["days_since_last_fight"] is None

    def test_debut_fighter_total_ufc_fights_is_zero(self) -> None:
        acc = CareerAccumulator()
        snapshot = acc.freeze()
        ctx = _build_context(snapshot)
        emitter = ActivityEmitter()
        result = emitter.emit(ctx)
        assert result["total_ufc_fights"] == 0

    def test_debut_fighter_windowed_counts_are_zero(self) -> None:
        acc = CareerAccumulator()
        snapshot = acc.freeze()
        ctx = _build_context(snapshot)
        emitter = ActivityEmitter()
        result = emitter.emit(ctx)
        assert result["fights_last_12mo"] == 0
        assert result["fights_last_3yr"] == 0
        assert result["fights_last_5yr"] == 0

    def test_debut_fighter_inactivity_tier_is_none(self) -> None:
        acc = CareerAccumulator()
        snapshot = acc.freeze()
        ctx = _build_context(snapshot)
        emitter = ActivityEmitter()
        result = emitter.emit(ctx)
        assert result["inactivity_tier"] is None

    def test_debut_fighter_age_x_inactivity_is_none(self) -> None:
        acc = CareerAccumulator()
        snapshot = acc.freeze()
        ctx = _build_context(snapshot)
        emitter = ActivityEmitter()
        result = emitter.emit(ctx)
        assert result["age_x_inactivity"] is None


# ---------------------------------------------------------------------------
# Inactivity tier boundary tests
# ---------------------------------------------------------------------------


class TestInactivityTierBoundaries:
    """Tier bucketing: 0=<180d, 1=180–365d, 2=366–730d, 3=731d+."""

    def _emit_with_gap(self, days_gap: int) -> dict[str, float | str | None]:
        """Set up one prior fight, then emit with the given gap in days."""
        acc = CareerAccumulator()
        fight_date = date(2023, 1, 1)
        acc.update(
            _outcome(event_date=fight_date, fight_url="http://fight/prior")
        )
        snapshot = acc.freeze()
        # Compute the event date as fight_date + days_gap
        from datetime import timedelta

        emit_date = fight_date + timedelta(days=days_gap)
        ctx = _build_context(snapshot, event_date=emit_date)
        emitter = ActivityEmitter()
        return emitter.emit(ctx)

    def test_179_days_is_tier_0(self) -> None:
        result = self._emit_with_gap(179)
        assert result["inactivity_tier"] == 0

    def test_180_days_is_tier_1(self) -> None:
        result = self._emit_with_gap(180)
        assert result["inactivity_tier"] == 1

    def test_365_days_is_tier_1(self) -> None:
        result = self._emit_with_gap(365)
        assert result["inactivity_tier"] == 1

    def test_366_days_is_tier_2(self) -> None:
        result = self._emit_with_gap(366)
        assert result["inactivity_tier"] == 2

    def test_730_days_is_tier_2(self) -> None:
        result = self._emit_with_gap(730)
        assert result["inactivity_tier"] == 2

    def test_731_days_is_tier_3(self) -> None:
        result = self._emit_with_gap(731)
        assert result["inactivity_tier"] == 3


# ---------------------------------------------------------------------------
# Days since last fight
# ---------------------------------------------------------------------------


class TestDaysSinceLastFight:
    """Correct day-count between last fight and current event."""

    def test_exact_days_count(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(event_date=date(2024, 1, 1)))
        snapshot = acc.freeze()
        ctx = _build_context(snapshot, event_date=date(2024, 4, 10))
        emitter = ActivityEmitter()
        result = emitter.emit(ctx)
        # Jan 1 to Apr 10 = 100 days
        assert result["days_since_last_fight"] == 100.0

    def test_same_day_fight_returns_zero(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(event_date=date(2024, 6, 1)))
        snapshot = acc.freeze()
        ctx = _build_context(snapshot, event_date=date(2024, 6, 1))
        emitter = ActivityEmitter()
        result = emitter.emit(ctx)
        assert result["days_since_last_fight"] == 0.0


# ---------------------------------------------------------------------------
# Windowed fight counts
# ---------------------------------------------------------------------------


class TestWindowedFightCounts:
    """Windowed fight counts respect 365/1095/1825 day boundaries."""

    def test_fights_within_12_months_counted(self) -> None:
        acc = CareerAccumulator()
        # Fight within 12 months of emit date
        acc.update(_outcome(event_date=date(2024, 1, 15), fight_url="http://f/1"))
        # Fight outside 12 months
        acc.update(_outcome(event_date=date(2022, 6, 1), fight_url="http://f/2"))
        snapshot = acc.freeze()
        ctx = _build_context(snapshot, event_date=date(2024, 6, 1))
        emitter = ActivityEmitter()
        result = emitter.emit(ctx)
        assert result["fights_last_12mo"] == 1

    def test_fights_within_3_years_counted(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(event_date=date(2022, 1, 1), fight_url="http://f/1"))
        acc.update(_outcome(event_date=date(2023, 6, 1), fight_url="http://f/2"))
        acc.update(_outcome(event_date=date(2019, 1, 1), fight_url="http://f/3"))
        snapshot = acc.freeze()
        ctx = _build_context(snapshot, event_date=date(2024, 6, 1))
        emitter = ActivityEmitter()
        result = emitter.emit(ctx)
        assert result["fights_last_3yr"] == 2

    def test_fights_within_5_years_counted(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(event_date=date(2020, 1, 1), fight_url="http://f/1"))
        acc.update(_outcome(event_date=date(2023, 1, 1), fight_url="http://f/2"))
        acc.update(_outcome(event_date=date(2018, 1, 1), fight_url="http://f/3"))
        snapshot = acc.freeze()
        ctx = _build_context(snapshot, event_date=date(2024, 6, 1))
        emitter = ActivityEmitter()
        result = emitter.emit(ctx)
        assert result["fights_last_5yr"] == 2


# ---------------------------------------------------------------------------
# Total UFC fights
# ---------------------------------------------------------------------------


class TestTotalUfcFights:
    """Total UFC fights count all fights regardless of window."""

    def test_counts_all_fights(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(event_date=date(2020, 1, 1), fight_url="http://f/1"))
        acc.update(_outcome(event_date=date(2021, 1, 1), fight_url="http://f/2"))
        acc.update(_outcome(event_date=date(2022, 1, 1), fight_url="http://f/3"))
        snapshot = acc.freeze()
        ctx = _build_context(snapshot, event_date=date(2024, 6, 1))
        emitter = ActivityEmitter()
        result = emitter.emit(ctx)
        assert result["total_ufc_fights"] == 3


# ---------------------------------------------------------------------------
# Last fight injury stoppage
# ---------------------------------------------------------------------------


class TestLastFightInjuryStoppage:
    """Detects whether the most recent fight ended by injury stoppage."""

    def test_injury_stoppage_detected(self) -> None:
        acc = CareerAccumulator()
        acc.update(
            _outcome(
                event_date=date(2024, 1, 1),
                method="TKO - Doctor's Stoppage",
            )
        )
        snapshot = acc.freeze()
        ctx = _build_context(snapshot, event_date=date(2024, 6, 1))
        emitter = ActivityEmitter()
        result = emitter.emit(ctx)
        assert result["last_fight_injury_stoppage"] == 1.0

    def test_non_injury_method_returns_zero(self) -> None:
        acc = CareerAccumulator()
        acc.update(
            _outcome(event_date=date(2024, 1, 1), method="KO/TKO")
        )
        snapshot = acc.freeze()
        ctx = _build_context(snapshot, event_date=date(2024, 6, 1))
        emitter = ActivityEmitter()
        result = emitter.emit(ctx)
        assert result["last_fight_injury_stoppage"] == 0.0

    def test_debut_fighter_injury_stoppage_is_none(self) -> None:
        acc = CareerAccumulator()
        snapshot = acc.freeze()
        ctx = _build_context(snapshot)
        emitter = ActivityEmitter()
        result = emitter.emit(ctx)
        assert result["last_fight_injury_stoppage"] is None


# ---------------------------------------------------------------------------
# Age × inactivity interaction
# ---------------------------------------------------------------------------


class TestAgeInactivityInteraction:
    """Interaction term: age_at_fight × days_since_last_fight."""

    def test_computed_correctly(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(event_date=date(2023, 1, 1)))
        snapshot = acc.freeze()
        # DOB 1990-01-01, event 2024-07-01 → age ~34.5
        # Last fight 2023-01-01, event 2024-07-01 → 547 days
        ctx = _build_context(
            snapshot, event_date=date(2024, 7, 1), dob=date(1990, 1, 1)
        )
        emitter = ActivityEmitter()
        result = emitter.emit(ctx)
        days = (date(2024, 7, 1) - date(2023, 1, 1)).days
        age = (date(2024, 7, 1) - date(1990, 1, 1)).days / 365.25
        expected = age * days
        assert result["age_x_inactivity"] == pytest.approx(expected, rel=1e-9)

    def test_none_when_no_dob(self) -> None:
        acc = CareerAccumulator()
        acc.update(_outcome(event_date=date(2023, 1, 1)))
        snapshot = acc.freeze()
        ctx = _build_context(snapshot, event_date=date(2024, 6, 1), dob=None)
        emitter = ActivityEmitter()
        result = emitter.emit(ctx)
        assert result["age_x_inactivity"] is None

    def test_none_when_debut(self) -> None:
        acc = CareerAccumulator()
        snapshot = acc.freeze()
        ctx = _build_context(snapshot, event_date=date(2024, 6, 1))
        emitter = ActivityEmitter()
        result = emitter.emit(ctx)
        assert result["age_x_inactivity"] is None


# ---------------------------------------------------------------------------
# Emitter protocol conformance
# ---------------------------------------------------------------------------


class TestEmitterProtocol:
    """ActivityEmitter conforms to the FeatureEmitter protocol."""

    def test_has_name_attribute(self) -> None:
        emitter = ActivityEmitter()
        assert emitter.name == "activity"

    def test_emit_returns_dict(self) -> None:
        acc = CareerAccumulator()
        snapshot = acc.freeze()
        ctx = _build_context(snapshot)
        emitter = ActivityEmitter()
        result = emitter.emit(ctx)
        assert isinstance(result, dict)

    def test_all_expected_keys_present(self) -> None:
        acc = CareerAccumulator()
        snapshot = acc.freeze()
        ctx = _build_context(snapshot)
        emitter = ActivityEmitter()
        result = emitter.emit(ctx)
        expected_keys = {
            "days_since_last_fight",
            "fights_last_12mo",
            "fights_last_3yr",
            "fights_last_5yr",
            "total_ufc_fights",
            "last_fight_injury_stoppage",
            "age_x_inactivity",
            "inactivity_tier",
        }
        assert set(result.keys()) == expected_keys
