"""Feature Registry — canonical schema owner for the feature engine.

Declares feature families with their output columns and types. Validates
at startup that no duplicate columns exist across families, all types are
within {float, str, NoneType}, and all required families are registered.
Exposes schema() for downstream storage validation and families() for
deterministic ordered iteration during feature generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Allowed column types for feature values.
ALLOWED_TYPES: frozenset[type] = frozenset({float, str, type(None)})


class RegistryError(Exception):
    """Raised when the feature registry detects a configuration violation."""


@dataclass(frozen=True)
class FeatureFamily:
    """A named group of feature columns with declared output types.

    Attributes:
        name: Unique identifier for this family (e.g., "elo", "activity").
        columns: Mapping of column name to its expected Python type.
        order: Numeric position controlling generation ordering.
    """

    name: str
    columns: dict[str, type] = field(default_factory=dict)
    order: int = 0


class FeatureRegistry:
    """Owns the canonical feature schema and enforces structural invariants.

    Validates at construction time:
    - No duplicate column names across families.
    - All column types are within {float, str, NoneType}.
    - All required families (if specified) have registered entries.

    Provides ordered iteration for deterministic generation.
    """

    def __init__(
        self,
        families: Sequence[FeatureFamily],
        required_families: Sequence[str] | None = None,
    ) -> None:
        self._families = sorted(families, key=lambda f: f.order)
        self._validate_types()
        self._validate_uniqueness()
        self._validate_required(required_families or [])

    def _validate_types(self) -> None:
        """Reject columns whose type is not in {float, str, NoneType}."""
        for family in self._families:
            for col_name, col_type in family.columns.items():
                if col_type not in ALLOWED_TYPES:
                    raise RegistryError(
                        f"Column '{col_name}' in family '{family.name}' declares "
                        f"invalid type '{col_type.__name__}'; "
                        f"allowed types are float, str, NoneType"
                    )

    def _validate_uniqueness(self) -> None:
        """Reject duplicate column names across families."""
        seen: dict[str, str] = {}
        for family in self._families:
            for col_name in family.columns:
                if col_name in seen:
                    raise RegistryError(
                        f"Duplicate column '{col_name}' declared by "
                        f"families '{seen[col_name]}' and '{family.name}'"
                    )
                seen[col_name] = family.name

    def _validate_required(self, required_families: Sequence[str]) -> None:
        """Reject if any required family name has no registered entry."""
        registered = {f.name for f in self._families}
        for name in required_families:
            if name not in registered:
                raise RegistryError(
                    f"Required family '{name}' has no registered emitter"
                )

    def schema(self) -> dict[str, type]:
        """Return the complete column→type map ordered by family then declaration.

        Returns a copy — mutations do not affect the registry.
        """
        result: dict[str, type] = {}
        for family in self._families:
            for col_name, col_type in family.columns.items():
                result[col_name] = col_type
        return result

    def families(self) -> list[FeatureFamily]:
        """Return families in declared generation order."""
        return list(self._families)

    def get_family(self, name: str) -> FeatureFamily | None:
        """Look up a family by name. Returns None if not found."""
        for family in self._families:
            if family.name == name:
                return family
        return None
