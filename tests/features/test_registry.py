"""Tests for the Feature Registry.

Validates schema, uniqueness, type safety, and family ordering.
"""

from __future__ import annotations

import pytest

from ufc_edge.features.registry import (
    FeatureFamily,
    FeatureRegistry,
    RegistryError,
)

# --- Fixtures ---


def _minimal_family(
    name: str = "test_family",
    columns: dict[str, type] | None = None,
    order: int = 0,
) -> FeatureFamily:
    """Build a FeatureFamily for testing with sensible defaults."""
    if columns is None:
        columns = {"col_a": float, "col_b": float}
    return FeatureFamily(name=name, columns=columns, order=order)


# --- RegistryError on duplicate columns ---


class TestDuplicateColumns:
    def test_duplicate_within_same_family_raises(self) -> None:
        """A family cannot declare the same column twice (caught at family level)."""
        # dict keys are inherently unique, so duplicate within one family
        # is structurally impossible with dict — test cross-family instead.
        family_a = _minimal_family("family_a", {"shared_col": float}, order=0)
        family_b = _minimal_family("family_b", {"shared_col": float}, order=1)

        with pytest.raises(RegistryError, match="shared_col"):
            FeatureRegistry(families=[family_a, family_b])

    def test_duplicate_across_families_names_both_families(self) -> None:
        """Error message identifies which families own the duplicate column."""
        family_a = _minimal_family("elo", {"elo_rating": float}, order=0)
        family_b = _minimal_family("activity", {"elo_rating": float}, order=1)

        with pytest.raises(RegistryError, match="elo") as exc_info:
            FeatureRegistry(families=[family_a, family_b])
        assert "activity" in str(exc_info.value)


# --- RegistryError on invalid types ---


class TestInvalidTypes:
    def test_int_type_raises(self) -> None:
        """Only float, str, and None are allowed column types."""
        family = _minimal_family("bad_family", {"count": int}, order=0)

        with pytest.raises(RegistryError, match="int"):
            FeatureRegistry(families=[family])

    def test_bool_type_raises(self) -> None:
        family = _minimal_family("bad_family", {"flag": bool}, order=0)

        with pytest.raises(RegistryError, match="bool"):
            FeatureRegistry(families=[family])

    def test_list_type_raises(self) -> None:
        family = _minimal_family("bad_family", {"items": list}, order=0)

        with pytest.raises(RegistryError, match="list"):
            FeatureRegistry(families=[family])

    def test_valid_types_accepted(self) -> None:
        """float, str, and type(None) are all valid."""
        family = _minimal_family(
            "valid_family",
            {"rating": float, "label": str, "optional_field": type(None)},
            order=0,
        )
        registry = FeatureRegistry(families=[family])
        assert "rating" in registry.schema()


# --- RegistryError on missing family ---


class TestMissingFamily:
    def test_required_family_missing_raises(self) -> None:
        """If required_families lists a name with no matching registered family, fail."""
        family = _minimal_family("elo", {"elo_rating": float}, order=0)

        with pytest.raises(RegistryError, match="activity"):
            FeatureRegistry(
                families=[family],
                required_families=["elo", "activity"],
            )

    def test_all_required_families_present_passes(self) -> None:
        family_a = _minimal_family("elo", {"elo_rating": float}, order=0)
        family_b = _minimal_family("activity", {"days_since": float}, order=1)

        registry = FeatureRegistry(
            families=[family_a, family_b],
            required_families=["elo", "activity"],
        )
        assert registry is not None


# --- schema() returns expected dict ---


class TestSchema:
    def test_schema_returns_all_columns_with_types(self) -> None:
        family_a = _minimal_family(
            "physical", {"height_cm": float, "stance": str}, order=0
        )
        family_b = _minimal_family(
            "graph", {"elo_rating": float, "glicko2_rd": float}, order=1
        )

        registry = FeatureRegistry(families=[family_a, family_b])
        schema = registry.schema()

        assert schema == {
            "height_cm": float,
            "stance": str,
            "elo_rating": float,
            "glicko2_rd": float,
        }

    def test_schema_includes_none_typed_columns(self) -> None:
        family = _minimal_family(
            "optional", {"maybe_field": type(None)}, order=0
        )
        registry = FeatureRegistry(families=[family])

        assert registry.schema() == {"maybe_field": type(None)}

    def test_schema_is_independent_copy(self) -> None:
        """Mutating the returned schema does not affect the registry."""
        family = _minimal_family("test", {"col": float}, order=0)
        registry = FeatureRegistry(families=[family])

        schema = registry.schema()
        schema["injected"] = str  # type: ignore[assignment]

        assert "injected" not in registry.schema()


# --- Ordered family iteration ---


class TestFamilyOrdering:
    def test_families_returned_in_order(self) -> None:
        """families() returns families sorted by their declared order."""
        family_c = _minimal_family("third", {"c": float}, order=2)
        family_a = _minimal_family("first", {"a": float}, order=0)
        family_b = _minimal_family("second", {"b": float}, order=1)

        # Pass in scrambled order — registry must sort by `order` field.
        registry = FeatureRegistry(families=[family_c, family_a, family_b])
        names = [f.name for f in registry.families()]

        assert names == ["first", "second", "third"]

    def test_schema_column_order_follows_family_order(self) -> None:
        """Columns in schema() follow family order then declaration order."""
        family_a = _minimal_family("alpha", {"a1": float, "a2": float}, order=0)
        family_b = _minimal_family("beta", {"b1": float, "b2": float}, order=1)

        registry = FeatureRegistry(families=[family_a, family_b])
        keys = list(registry.schema().keys())

        assert keys == ["a1", "a2", "b1", "b2"]


# --- Edge cases ---


class TestEdgeCases:
    def test_empty_families_list_accepted(self) -> None:
        """A registry with no families is valid (degenerate but not erroneous)."""
        registry = FeatureRegistry(families=[])
        assert registry.schema() == {}

    def test_family_with_no_columns_accepted(self) -> None:
        """A family with zero columns is structurally valid (placeholder)."""
        family = _minimal_family("empty_family", {}, order=0)
        registry = FeatureRegistry(families=[family])
        assert registry.schema() == {}

    def test_get_family_by_name(self) -> None:
        """Registry exposes lookup by family name."""
        family = _minimal_family("elo", {"elo_rating": float}, order=0)
        registry = FeatureRegistry(families=[family])

        result = registry.get_family("elo")
        assert result is not None
        assert result.name == "elo"

    def test_get_family_missing_returns_none(self) -> None:
        family = _minimal_family("elo", {"elo_rating": float}, order=0)
        registry = FeatureRegistry(families=[family])

        assert registry.get_family("nonexistent") is None
