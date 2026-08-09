"""Tests for the physical profile feature emitter.

Validates that height, reach, ratio, stance, age, and weight class are emitted
correctly from fighter profile data and event context. Covers known-profile
happy path, missing DOB producing None for age, and missing reach producing
None for reach-to-height ratio.
"""

from __future__ import annotations

from datetime import date

from ufc_edge.features.contracts import EmitContext, FighterProfile
from ufc_edge.features.emitters.physical import PhysicalEmitter


def _make_context(
    *,
    fighter_profile: FighterProfile | None = None,
    event_date: date = date(2024, 3, 16),
    weight_class: str = "Lightweight",
) -> EmitContext:
    """Build a minimal EmitContext for physical emitter testing."""
    default_profile = FighterProfile(
        fighter_url="http://ufcstats.com/fighter/a",
        height_cm=180.0,
        reach_cm=190.0,
        stance="Orthodox",
        dob=date(1990, 5, 20),
    )
    profile = fighter_profile if fighter_profile is not None else default_profile
    opponent_profile = FighterProfile(
        fighter_url="http://ufcstats.com/fighter/b",
        height_cm=175.0,
        reach_cm=180.0,
        stance="Southpaw",
        dob=date(1988, 1, 15),
    )
    return EmitContext(
        fighter_url=profile.fighter_url,
        fighter_profile=profile,
        opponent_url=opponent_profile.fighter_url,
        opponent_profile=opponent_profile,
        event_date=event_date,
        event_url="http://ufcstats.com/event/e1",
        weight_class=weight_class,
        fight_url="http://ufcstats.com/fight/f1",
        bout_order=None,
        components={},
    )


class TestPhysicalEmitterProtocol:
    """PhysicalEmitter satisfies the FeatureEmitter protocol."""

    def test_has_name_attribute(self) -> None:
        emitter = PhysicalEmitter()
        assert isinstance(emitter.name, str)
        assert emitter.name == "physical"

    def test_emit_returns_dict(self) -> None:
        emitter = PhysicalEmitter()
        ctx = _make_context()
        result = emitter.emit(ctx)
        assert isinstance(result, dict)


class TestPhysicalEmitterKnownProfile:
    """Known fighter profile produces exact expected feature values."""

    def test_height_cm(self) -> None:
        emitter = PhysicalEmitter()
        ctx = _make_context()
        result = emitter.emit(ctx)
        assert result["height_cm"] == 180.0

    def test_reach_cm(self) -> None:
        emitter = PhysicalEmitter()
        ctx = _make_context()
        result = emitter.emit(ctx)
        assert result["reach_cm"] == 190.0

    def test_reach_to_height_ratio(self) -> None:
        emitter = PhysicalEmitter()
        ctx = _make_context()
        result = emitter.emit(ctx)
        expected_ratio = 190.0 / 180.0
        assert result["reach_to_height_ratio"] == expected_ratio

    def test_stance(self) -> None:
        emitter = PhysicalEmitter()
        ctx = _make_context()
        result = emitter.emit(ctx)
        assert result["stance"] == "Orthodox"

    def test_age_at_fight(self) -> None:
        emitter = PhysicalEmitter()
        # DOB: 1990-05-20, event: 2024-03-16 → age is 33 (birthday not yet passed)
        ctx = _make_context(event_date=date(2024, 3, 16))
        result = emitter.emit(ctx)
        assert result["age_at_fight"] == 33.0

    def test_age_at_fight_birthday_passed(self) -> None:
        emitter = PhysicalEmitter()
        # DOB: 1990-05-20, event: 2024-06-01 → age is 34 (birthday passed)
        ctx = _make_context(event_date=date(2024, 6, 1))
        result = emitter.emit(ctx)
        assert result["age_at_fight"] == 34.0

    def test_age_at_fight_on_birthday(self) -> None:
        emitter = PhysicalEmitter()
        # DOB: 1990-05-20, event: 2024-05-20 → age is 34 (birthday is today)
        ctx = _make_context(event_date=date(2024, 5, 20))
        result = emitter.emit(ctx)
        assert result["age_at_fight"] == 34.0

    def test_weight_class(self) -> None:
        emitter = PhysicalEmitter()
        ctx = _make_context(weight_class="Welterweight")
        result = emitter.emit(ctx)
        assert result["weight_class"] == "Welterweight"

    def test_all_keys_present(self) -> None:
        emitter = PhysicalEmitter()
        ctx = _make_context()
        result = emitter.emit(ctx)
        expected_keys = {
            "height_cm",
            "reach_cm",
            "reach_to_height_ratio",
            "stance",
            "age_at_fight",
            "weight_class",
        }
        assert set(result.keys()) == expected_keys


class TestPhysicalEmitterMissingDob:
    """Missing date of birth produces None for age_at_fight."""

    def test_missing_dob_returns_none_age(self) -> None:
        emitter = PhysicalEmitter()
        profile = FighterProfile(
            fighter_url="http://ufcstats.com/fighter/a",
            height_cm=180.0,
            reach_cm=190.0,
            stance="Orthodox",
            dob=None,
        )
        ctx = _make_context(fighter_profile=profile)
        result = emitter.emit(ctx)
        assert result["age_at_fight"] is None

    def test_missing_dob_other_fields_still_present(self) -> None:
        emitter = PhysicalEmitter()
        profile = FighterProfile(
            fighter_url="http://ufcstats.com/fighter/a",
            height_cm=180.0,
            reach_cm=190.0,
            stance="Orthodox",
            dob=None,
        )
        ctx = _make_context(fighter_profile=profile)
        result = emitter.emit(ctx)
        assert result["height_cm"] == 180.0
        assert result["reach_cm"] == 190.0
        assert result["reach_to_height_ratio"] == 190.0 / 180.0
        assert result["stance"] == "Orthodox"


class TestPhysicalEmitterMissingReach:
    """Missing reach produces None for reach_cm and reach_to_height_ratio."""

    def test_missing_reach_returns_none_ratio(self) -> None:
        emitter = PhysicalEmitter()
        profile = FighterProfile(
            fighter_url="http://ufcstats.com/fighter/a",
            height_cm=180.0,
            reach_cm=None,
            stance="Orthodox",
            dob=date(1990, 5, 20),
        )
        ctx = _make_context(fighter_profile=profile)
        result = emitter.emit(ctx)
        assert result["reach_to_height_ratio"] is None

    def test_missing_reach_returns_none_reach_cm(self) -> None:
        emitter = PhysicalEmitter()
        profile = FighterProfile(
            fighter_url="http://ufcstats.com/fighter/a",
            height_cm=180.0,
            reach_cm=None,
            stance="Orthodox",
            dob=date(1990, 5, 20),
        )
        ctx = _make_context(fighter_profile=profile)
        result = emitter.emit(ctx)
        assert result["reach_cm"] is None


class TestPhysicalEmitterMissingHeight:
    """Missing height produces None for height_cm and reach_to_height_ratio."""

    def test_missing_height_returns_none_ratio(self) -> None:
        emitter = PhysicalEmitter()
        profile = FighterProfile(
            fighter_url="http://ufcstats.com/fighter/a",
            height_cm=None,
            reach_cm=190.0,
            stance="Orthodox",
            dob=date(1990, 5, 20),
        )
        ctx = _make_context(fighter_profile=profile)
        result = emitter.emit(ctx)
        assert result["reach_to_height_ratio"] is None

    def test_missing_height_returns_none_height_cm(self) -> None:
        emitter = PhysicalEmitter()
        profile = FighterProfile(
            fighter_url="http://ufcstats.com/fighter/a",
            height_cm=None,
            reach_cm=190.0,
            stance="Orthodox",
            dob=date(1990, 5, 20),
        )
        ctx = _make_context(fighter_profile=profile)
        result = emitter.emit(ctx)
        assert result["height_cm"] is None


class TestPhysicalEmitterAllMissing:
    """Fully missing profile fields produce None for derived features."""

    def test_all_physical_fields_none(self) -> None:
        emitter = PhysicalEmitter()
        profile = FighterProfile(
            fighter_url="http://ufcstats.com/fighter/a",
            height_cm=None,
            reach_cm=None,
            stance=None,
            dob=None,
        )
        ctx = _make_context(fighter_profile=profile)
        result = emitter.emit(ctx)
        assert result["height_cm"] is None
        assert result["reach_cm"] is None
        assert result["reach_to_height_ratio"] is None
        assert result["stance"] is None
        assert result["age_at_fight"] is None
        # weight_class comes from context, not profile
        assert result["weight_class"] is not None
