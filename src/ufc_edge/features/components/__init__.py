"""Feature engine state components — accumulate fighter state across events."""

from ufc_edge.features.components.career import CareerAccumulator as CareerAccumulator
from ufc_edge.features.components.rolling_stats import (
    RollingStatsAccumulator as RollingStatsAccumulator,
)
