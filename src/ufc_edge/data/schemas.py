"""Shared Pydantic base for all ufc-edge data models.

Each data sub-package (ufcstats/, polymarket/) defines its own schema module and
imports _FrozenModel from here. This module contains nothing domain-specific.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _FrozenModel(BaseModel):
    """Base class: immutable by default to prevent accidental mutation."""

    model_config = ConfigDict(frozen=True)
