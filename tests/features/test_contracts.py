"""Tests for feature engine protocol contracts and data models.

Validates immutability, protocol signatures, and field exclusion rules that
enforce temporal isolation and market-data separation.
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime

import pytest
from pydantic import ValidationError


class TestEmitContextFrozen:
    """EmitContext must be immutable — assignment to any field raises."""

    def test_assignment_raises_on_fighter_url(self) -> None:
        ctx = _make_emit_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.fighter_url = "http://other"  # type: ignore[misc]

    def test_assignment_raises_on_event_date(self) -> None:
        ctx = _make_emit_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.event_date = date(2025, 1, 1)  # type: ignore[misc]

    def test_assignment_raises_on_components(self) -> None:
        ctx = _make_emit_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.components = {}  # type: ignore[misc]

    def test_assignment_raises_on_weight_class(self) -> None:
        ctx = _make_emit_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.weight_class = "Heavyweight"  # type: ignore[misc]


class TestFightOutcomeViewExclusions:
    """FightOutcomeView must NOT contain market-derived or label-leaking fields."""

    def test_no_market_price_field(self) -> None:
        from ufc_edge.features.contracts import FightOutcomeView

        fields = set(FightOutcomeView.model_fields.keys())
        market_fields = {"market_price", "order_book", "polymarket_price", "odds", "implied_prob"}
        assert fields.isdisjoint(market_fields)

    def test_has_required_outcome_fields(self) -> None:
        from ufc_edge.features.contracts import FightOutcomeView

        fields = set(FightOutcomeView.model_fields.keys())
        required = {
            "fight_url",
            "event_url",
            "event_date",
            "fighter_a_url",
            "fighter_b_url",
            "winner_url",
            "method",
            "ending_round",
            "ending_time",
            "weight_class",
            "bout_order",
        }
        assert required.issubset(fields)

    def test_is_frozen(self) -> None:
        from ufc_edge.features.contracts import FightOutcomeView

        view = FightOutcomeView(
            fight_url="http://fight/1",
            event_url="http://event/1",
            event_date=date(2024, 6, 1),
            fighter_a_url="http://fighter/a",
            fighter_b_url="http://fighter/b",
            winner_url="http://fighter/a",
            method="KO/TKO",
            ending_round=2,
            ending_time="3:45",
            weight_class="Lightweight",
            bout_order=None,
        )
        with pytest.raises(ValidationError):
            view.winner_url = "http://other"  # type: ignore[misc]


class TestEmitContextExclusions:
    """EmitContext must NOT expose labels, market data, or mutable refs."""

    def test_no_winner_url_field(self) -> None:
        from ufc_edge.features.contracts import EmitContext

        assert not hasattr(EmitContext, "winner_url")

    def test_no_market_fields(self) -> None:
        from ufc_edge.features.contracts import EmitContext

        fields = {f.name for f in EmitContext.__dataclass_fields__.values()}
        forbidden = {"market_price", "order_book", "polymarket_price", "odds"}
        assert fields.isdisjoint(forbidden)

    def test_no_duckdb_connection_field(self) -> None:
        from ufc_edge.features.contracts import EmitContext

        fields = {f.name for f in EmitContext.__dataclass_fields__.values()}
        assert "conn" not in fields
        assert "connection" not in fields
        assert "db" not in fields


class TestStateComponentProtocol:
    """StateComponent protocol has update(FightOutcomeView) and freeze() -> FrozenState."""

    def test_has_update_method(self) -> None:
        from ufc_edge.features.contracts import StateComponent

        assert hasattr(StateComponent, "update")

    def test_has_freeze_method(self) -> None:
        from ufc_edge.features.contracts import StateComponent

        assert hasattr(StateComponent, "freeze")

    def test_is_runtime_checkable(self) -> None:
        from ufc_edge.features.contracts import FightOutcomeView, FrozenState, StateComponent

        class _FakeComponent:
            def update(self, fight: FightOutcomeView) -> None:
                pass

            def freeze(self) -> FrozenState:
                return FrozenState()

        assert isinstance(_FakeComponent(), StateComponent)

    def test_non_conforming_class_fails_check(self) -> None:
        from ufc_edge.features.contracts import StateComponent

        class _BadComponent:
            pass

        assert not isinstance(_BadComponent(), StateComponent)


class TestFeatureEmitterProtocol:
    """FeatureEmitter protocol has name: str and emit(EmitContext) -> dict."""

    def test_has_name_attribute(self) -> None:
        from ufc_edge.features.contracts import FeatureEmitter

        assert "name" in FeatureEmitter.__protocol_attrs__

    def test_has_emit_method(self) -> None:
        from ufc_edge.features.contracts import FeatureEmitter

        assert hasattr(FeatureEmitter, "emit")

    def test_is_runtime_checkable(self) -> None:
        from ufc_edge.features.contracts import EmitContext, FeatureEmitter

        class _FakeEmitter:
            name: str = "fake"

            def emit(self, context: EmitContext) -> dict[str, float | str | None]:
                return {}

        assert isinstance(_FakeEmitter(), FeatureEmitter)

    def test_non_conforming_class_fails_check(self) -> None:
        from ufc_edge.features.contracts import FeatureEmitter

        class _BadEmitter:
            pass

        assert not isinstance(_BadEmitter(), FeatureEmitter)


class TestFrozenState:
    """FrozenState is the base for component snapshots."""

    def test_frozen_state_is_frozen(self) -> None:
        from ufc_edge.features.contracts import FrozenState

        state = FrozenState()
        with pytest.raises(AttributeError):
            state.new_attr = "value"  # type: ignore[attr-defined]


class TestHistoricalFight:
    """HistoricalFight carries all fields needed for replay."""

    def test_has_required_fields(self) -> None:
        from ufc_edge.features.contracts import HistoricalFight

        fields = set(HistoricalFight.model_fields.keys())
        required = {
            "fight_url",
            "event_url",
            "event_date",
            "fighter_a_url",
            "fighter_b_url",
            "winner_url",
            "method",
            "ending_round",
            "ending_time",
            "time_format",
            "weight_class",
            "bout_order",
            "fighter_a_profile",
            "fighter_b_profile",
            "fighter_a_totals",
            "fighter_b_totals",
        }
        assert required.issubset(fields)

    def test_is_frozen(self) -> None:
        from ufc_edge.features.contracts import FighterProfile, HistoricalFight

        fight = HistoricalFight(
            fight_url="http://fight/1",
            event_url="http://event/1",
            event_date=date(2024, 6, 1),
            fighter_a_url="http://fighter/a",
            fighter_b_url="http://fighter/b",
            winner_url="http://fighter/a",
            method="KO/TKO",
            ending_round=2,
            ending_time="3:45",
            time_format="5:00",
            weight_class="Lightweight",
            bout_order=None,
            fighter_a_profile=FighterProfile(fighter_url="http://fighter/a"),
            fighter_b_profile=FighterProfile(fighter_url="http://fighter/b"),
            fighter_a_totals=None,
            fighter_b_totals=None,
        )
        with pytest.raises(ValidationError):
            fight.method = "Decision"  # type: ignore[misc]


class TestEventTick:
    """EventTick groups fights for one event in an atomic tick."""

    def test_has_required_fields(self) -> None:
        from ufc_edge.features.contracts import EventTick

        fields = set(EventTick.model_fields.keys())
        assert {"event_url", "event_date", "fights"}.issubset(fields)

    def test_is_frozen(self) -> None:
        from ufc_edge.features.contracts import EventTick

        tick = EventTick(
            event_url="http://event/1",
            event_date=date(2024, 6, 1),
            fights=[],
        )
        with pytest.raises(ValidationError):
            tick.event_url = "http://other"  # type: ignore[misc]


class TestFeatureRow:
    """FeatureRow represents one emitted feature record."""

    def test_has_metadata_fields(self) -> None:
        from ufc_edge.features.contracts import FeatureRow

        fields = set(FeatureRow.model_fields.keys())
        metadata = {
            "fight_url",
            "fighter_url",
            "event_url",
            "event_date",
            "opponent_url",
            "weight_class",
            "feature_version",
            "generated_at",
            "features",
        }
        assert metadata.issubset(fields)

    def test_is_frozen(self) -> None:
        from ufc_edge.features.contracts import FeatureRow

        row = FeatureRow(
            fight_url="http://fight/1",
            fighter_url="http://fighter/a",
            event_url="http://event/1",
            event_date=date(2024, 6, 1),
            opponent_url="http://fighter/b",
            weight_class="Lightweight",
            feature_version="v1",
            generated_at=datetime(2024, 6, 1, 12, 0, 0),
            features={},
        )
        with pytest.raises(ValidationError):
            row.fight_url = "http://other"  # type: ignore[misc]


class TestPublicExports:
    """The __init__.py must export all public contract types."""

    def test_all_contracts_importable_from_package(self) -> None:
        from ufc_edge.features import (
            EmitContext,
            EventTick,
            FeatureEmitter,
            FeatureRow,
            FighterProfile,
            FightOutcomeView,
            FightTotals,
            FrozenState,
            HistoricalFight,
            StateComponent,
        )

        assert EmitContext is not None
        assert EventTick is not None
        assert FeatureEmitter is not None
        assert FeatureRow is not None
        assert FighterProfile is not None
        assert FightOutcomeView is not None
        assert FightTotals is not None
        assert FrozenState is not None
        assert HistoricalFight is not None
        assert StateComponent is not None


# --- Helpers ---


def _make_emit_context():
    """Build a minimal EmitContext for immutability tests."""
    from ufc_edge.features.contracts import EmitContext, FighterProfile

    return EmitContext(
        fighter_url="http://fighter/a",
        fighter_profile=FighterProfile(fighter_url="http://fighter/a"),
        opponent_url="http://fighter/b",
        opponent_profile=FighterProfile(fighter_url="http://fighter/b"),
        event_date=date(2024, 6, 1),
        event_url="http://event/1",
        weight_class="Lightweight",
        fight_url="http://fight/1",
        bout_order=None,
        components={},
    )
