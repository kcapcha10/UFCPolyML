"""Tests for CardPositionEmitter.

Validates the gated behavior when bout_order data is unavailable (all outputs
None with logged warning), the main-card heuristic when bout_order is present,
and the minimum-fights threshold for variance features.
"""

from __future__ import annotations

import logging
from datetime import date

import pytest

from ufc_edge.features.contracts import EmitContext, FighterProfile, FrozenState
from ufc_edge.features.emitters.card_position import CardPositionEmitter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_OUTPUT_KEYS = (
    "sig_strikes_main_card_avg",
    "sig_strikes_prelim_avg",
    "td_rate_main_card_avg",
    "td_rate_prelim_avg",
    "grappling_abandonment_delta",
    "output_variance_by_position",
)


def _make_profile(fighter_url: str = "http://fighter/a") -> FighterProfile:
    return FighterProfile(
        fighter_url=fighter_url,
        height_cm=180.0,
        reach_cm=185.0,
        stance="Orthodox",
        dob=date(1990, 1, 1),
    )


def _make_context(
    fighter_url: str = "http://fighter/a",
    opponent_url: str = "http://fighter/b",
    bout_order: int | None = None,
    components: dict[str, FrozenState] | None = None,
) -> EmitContext:
    return EmitContext(
        fighter_url=fighter_url,
        fighter_profile=_make_profile(fighter_url),
        opponent_url=opponent_url,
        opponent_profile=_make_profile(opponent_url),
        event_date=date(2024, 6, 15),
        event_url="http://event/ufc300",
        weight_class="Lightweight",
        fight_url="http://fight/1",
        bout_order=bout_order,
        components=components or {},
    )


# ---------------------------------------------------------------------------
# Gated branch: bout_order unavailable → all None with warning
# ---------------------------------------------------------------------------


class TestCardPositionEmitterGated:
    """When bout_order is None, all features are None and a warning is logged."""

    def test_all_outputs_none_when_bout_order_unavailable(self) -> None:
        emitter = CardPositionEmitter()
        context = _make_context(bout_order=None)

        result = emitter.emit(context)

        for key in _ALL_OUTPUT_KEYS:
            assert key in result, f"Missing output key: {key}"
            assert result[key] is None, f"Expected None for {key}, got {result[key]}"

    def test_logs_warning_when_bout_order_unavailable(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        emitter = CardPositionEmitter()
        context = _make_context(bout_order=None)

        with caplog.at_level(logging.WARNING):
            emitter.emit(context)

        assert any(
            "bout_order" in record.message.lower() for record in caplog.records
        ), "Expected a warning mentioning bout_order"

    def test_emitter_name_is_card_position(self) -> None:
        emitter = CardPositionEmitter()
        assert emitter.name == "card_position"

    def test_output_dict_has_exactly_expected_keys(self) -> None:
        emitter = CardPositionEmitter()
        context = _make_context(bout_order=None)

        result = emitter.emit(context)

        assert set(result.keys()) == set(_ALL_OUTPUT_KEYS)


# ---------------------------------------------------------------------------
# Bout order present but no card-position history component → gated path
# ---------------------------------------------------------------------------


class TestCardPositionEmitterNoHistoryComponent:
    """When bout_order is present on the current fight but no card-position
    history component is registered, the emitter still emits None with warning
    because it cannot split historical fights by card position."""

    def test_all_outputs_none_without_history_component(self) -> None:
        emitter = CardPositionEmitter()
        context = _make_context(bout_order=5, components={})

        result = emitter.emit(context)

        for key in _ALL_OUTPUT_KEYS:
            assert result[key] is None

    def test_logs_warning_without_history_component(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        emitter = CardPositionEmitter()
        context = _make_context(bout_order=5, components={})

        with caplog.at_level(logging.WARNING):
            emitter.emit(context)

        assert any(
            "bout_order" in record.message.lower()
            or "card_position" in record.message.lower()
            for record in caplog.records
        ), "Expected a warning about missing card-position history"


# ---------------------------------------------------------------------------
# Bout order present with card-position history → main-card heuristic
# ---------------------------------------------------------------------------


class _FakeCardPositionSnapshot(FrozenState):
    """Fake snapshot providing historical bout_order data for a fighter."""

    __slots__ = ("_history",)

    def __init__(self, history: dict[str, list[dict[str, float | int | None]]]) -> None:
        object.__setattr__(self, "_history", history)

    def get_fight_history(self, fighter_url: str) -> list[dict[str, float | int | None]] | None:
        return self._history.get(fighter_url)


def _fight_record(
    bout_order: int,
    sig_strikes_per_min: float,
    td_per_15min: float,
    grappling_dominance: float | None = 1.5,
) -> dict[str, float | int | None]:
    """Create a single historical fight record with card-position data."""
    return {
        "bout_order": bout_order,
        "sig_strikes_per_min": sig_strikes_per_min,
        "td_per_15min": td_per_15min,
        "grappling_dominance": grappling_dominance,
    }


class TestCardPositionEmitterWithHistory:
    """When bout_order data is available and a card-position history component
    provides historical fight records, the emitter splits by main/prelim."""

    def _make_history_context(
        self,
        fighter_url: str,
        fights: list[dict[str, float | int | None]],
        bout_order: int = 5,
    ) -> EmitContext:
        snapshot = _FakeCardPositionSnapshot({fighter_url: fights})
        return _make_context(
            fighter_url=fighter_url,
            bout_order=bout_order,
            components={"card_position_history": snapshot},
        )

    def test_splits_main_and_prelim_by_top_5_heuristic(self) -> None:
        """Main card = top-5 bouts by bout_order (highest values)."""
        emitter = CardPositionEmitter()
        fights = [
            _fight_record(bout_order=10, sig_strikes_per_min=6.0, td_per_15min=3.0),
            _fight_record(bout_order=9, sig_strikes_per_min=5.0, td_per_15min=2.5),
            _fight_record(bout_order=8, sig_strikes_per_min=4.0, td_per_15min=2.0),
            _fight_record(bout_order=7, sig_strikes_per_min=3.0, td_per_15min=1.5),
            _fight_record(bout_order=6, sig_strikes_per_min=2.0, td_per_15min=1.0),
            _fight_record(bout_order=3, sig_strikes_per_min=1.0, td_per_15min=0.5),
            _fight_record(bout_order=2, sig_strikes_per_min=0.5, td_per_15min=0.2),
        ]
        context = self._make_history_context("http://fighter/a", fights)

        result = emitter.emit(context)

        # Main card (top-5 by bout_order): bouts 10,9,8,7,6 → avg sig = 4.0
        assert result["sig_strikes_main_card_avg"] == pytest.approx(4.0)
        # Prelim (remaining): bouts 3,2 → avg sig = 0.75
        assert result["sig_strikes_prelim_avg"] == pytest.approx(0.75)
        # Main card td rate: (3.0+2.5+2.0+1.5+1.0)/5 = 2.0
        assert result["td_rate_main_card_avg"] == pytest.approx(2.0)
        # Prelim td rate: (0.5+0.2)/2 = 0.35
        assert result["td_rate_prelim_avg"] == pytest.approx(0.35)

    def test_grappling_abandonment_delta(self) -> None:
        """Delta = main_card_grappling_avg - prelim_grappling_avg."""
        emitter = CardPositionEmitter()
        fights = [
            _fight_record(
                bout_order=10, sig_strikes_per_min=5.0,
                td_per_15min=2.0, grappling_dominance=2.0,
            ),
            _fight_record(
                bout_order=9, sig_strikes_per_min=5.0,
                td_per_15min=2.0, grappling_dominance=1.8,
            ),
            _fight_record(
                bout_order=8, sig_strikes_per_min=5.0,
                td_per_15min=2.0, grappling_dominance=1.6,
            ),
            _fight_record(
                bout_order=7, sig_strikes_per_min=5.0,
                td_per_15min=2.0, grappling_dominance=1.4,
            ),
            _fight_record(
                bout_order=6, sig_strikes_per_min=5.0,
                td_per_15min=2.0, grappling_dominance=1.2,
            ),
            _fight_record(
                bout_order=2, sig_strikes_per_min=1.0,
                td_per_15min=0.5, grappling_dominance=3.0,
            ),
        ]
        context = self._make_history_context("http://fighter/a", fights)

        result = emitter.emit(context)

        # Main avg grappling: (2.0+1.8+1.6+1.4+1.2)/5 = 1.6
        # Prelim avg grappling: 3.0/1 = 3.0
        # Delta: 1.6 - 3.0 = -1.4
        assert result["grappling_abandonment_delta"] == pytest.approx(-1.4)

    def test_variance_none_when_fewer_than_3_fights(self) -> None:
        """output_variance_by_position requires 3+ total fights."""
        emitter = CardPositionEmitter()
        fights = [
            _fight_record(bout_order=10, sig_strikes_per_min=5.0, td_per_15min=2.0),
            _fight_record(bout_order=2, sig_strikes_per_min=1.0, td_per_15min=0.5),
        ]
        context = self._make_history_context("http://fighter/a", fights)

        result = emitter.emit(context)

        assert result["output_variance_by_position"] is None

    def test_variance_computed_with_3_plus_fights(self) -> None:
        """output_variance_by_position computed as variance of sig_strikes across
        position-grouped averages when 3+ fights available."""
        emitter = CardPositionEmitter()
        # Need 6+ fights so top-5 are main and remainder are prelim
        fights = [
            _fight_record(bout_order=12, sig_strikes_per_min=6.0, td_per_15min=3.0),
            _fight_record(bout_order=11, sig_strikes_per_min=6.0, td_per_15min=3.0),
            _fight_record(bout_order=10, sig_strikes_per_min=6.0, td_per_15min=3.0),
            _fight_record(bout_order=9, sig_strikes_per_min=4.0, td_per_15min=2.0),
            _fight_record(bout_order=8, sig_strikes_per_min=3.0, td_per_15min=1.5),
            _fight_record(bout_order=2, sig_strikes_per_min=1.0, td_per_15min=0.5),
        ]
        context = self._make_history_context("http://fighter/a", fights)

        result = emitter.emit(context)

        # Main card (top-5 by bout_order: 12,11,10,9,8) avg sig: (6+6+6+4+3)/5 = 5.0
        # Prelim (bout_order 2) avg sig: 1.0
        # Variance of [5.0, 1.0] = ((5-3)^2 + (1-3)^2) / 2 = (4+4)/2 = 4.0
        assert result["output_variance_by_position"] == pytest.approx(4.0)

    def test_all_fights_on_main_card_prelim_avg_is_none(self) -> None:
        """When all fights are main card, prelim averages are None."""
        emitter = CardPositionEmitter()
        fights = [
            _fight_record(bout_order=10, sig_strikes_per_min=5.0, td_per_15min=2.0),
            _fight_record(bout_order=9, sig_strikes_per_min=4.0, td_per_15min=1.5),
            _fight_record(bout_order=8, sig_strikes_per_min=3.0, td_per_15min=1.0),
        ]
        context = self._make_history_context("http://fighter/a", fights)

        result = emitter.emit(context)

        assert result["sig_strikes_main_card_avg"] == pytest.approx(4.0)
        assert result["sig_strikes_prelim_avg"] is None
        assert result["td_rate_main_card_avg"] == pytest.approx(1.5)
        assert result["td_rate_prelim_avg"] is None
        assert result["grappling_abandonment_delta"] is None
        assert result["output_variance_by_position"] is None

    def test_no_fight_history_for_fighter_returns_all_none(self) -> None:
        """When the history component has no data for this fighter, emit None."""
        emitter = CardPositionEmitter()
        snapshot = _FakeCardPositionSnapshot({"http://other_fighter": []})
        context = _make_context(
            fighter_url="http://fighter/a",
            bout_order=5,
            components={"card_position_history": snapshot},
        )

        result = emitter.emit(context)

        for key in _ALL_OUTPUT_KEYS:
            assert result[key] is None
