"""OutputEmitter — thin adapter over RollingStatsSnapshot rolling averages.

Reads the frozen rolling-stats component from EmitContext and exposes per-fighter
output and efficiency metrics: striking rates, takedown rates, damage ratio,
grappling dominance, control time, and knockdown rate. All math lives in
RollingStatsAccumulator; this emitter simply reads the pre-computed averages.
"""

from __future__ import annotations

from ufc_edge.features.components.rolling_stats import RollingStatsSnapshot
from ufc_edge.features.contracts import EmitContext

# All output columns this emitter produces
_OUTPUT_COLUMNS: tuple[str, ...] = (
    "sig_strikes_per_min",
    "sig_strikes_absorbed_per_min",
    "striking_accuracy_pct",
    "striking_defense_pct",
    "td_per_15min",
    "td_accuracy_pct",
    "td_defense_pct",
    "sub_attempts_per_15min",
    "knockdown_rate",
    "damage_ratio",
    "grappling_dominance",
    "control_time_per_fight",
)


class OutputEmitter:
    """Stateless emitter that reads rolling output averages for the focal fighter.

    Delegates all computation to RollingStatsSnapshot.get_rolling_averages().
    Returns None for every column when the fighter has no recorded history or
    the rolling_stats component is absent from context.
    """

    name: str = "output"

    def emit(self, context: EmitContext) -> dict[str, float | None]:
        """Emit rolling output-stat features for the focal fighter."""
        snapshot = context.components.get("rolling_stats")
        if not isinstance(snapshot, RollingStatsSnapshot):
            return _all_none()

        averages = snapshot.get_rolling_averages(context.fighter_url)
        if averages is None:
            return _all_none()

        return {col: averages.get(col) for col in _OUTPUT_COLUMNS}


def _all_none() -> dict[str, float | None]:
    """Return dict with all output columns set to None."""
    return dict.fromkeys(_OUTPUT_COLUMNS, None)
