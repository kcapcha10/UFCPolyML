"""Rolling statistics accumulator for per-fighter output and efficiency metrics.

Maintains a sliding window (deque) of per-fight statistics for each fighter.
Computes rolling averages and variance over the window for metrics like striking
rate, takedown accuracy, damage ratio, and grappling dominance.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from collections.abc import Mapping

from ufc_edge.features.contracts import FightOutcomeView, FightTotals, FrozenState

# ---------------------------------------------------------------------------
# FightStats — per-fight derived metrics stored in the rolling window
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class FightStats:
    """Derived per-fight statistics computed from raw totals and fight duration.

    Stores both the fighter's own stats and opponent stats needed for defensive
    metrics (strikes absorbed, opponent takedowns, opponent control time).
    """

    fight_duration_minutes: float
    sig_strikes_landed: int
    sig_strikes_attempted: int
    sig_strikes_absorbed: int
    sig_strikes_absorbed_attempted: int
    total_strikes_landed: int
    total_strikes_attempted: int
    takedowns_landed: int
    takedowns_attempted: int
    submissions_attempted: int
    knockdowns: int
    control_time_seconds: int
    opponent_takedowns_landed: int
    opponent_takedowns_attempted: int
    opponent_control_time_seconds: int

    @classmethod
    def from_totals(
        cls,
        totals: FightTotals | None,
        opponent_totals: FightTotals | None,
        ending_round: int,
        ending_time: str,
    ) -> FightStats | None:
        """Build FightStats from raw totals and fight timing.

        Returns None when fighter totals are unavailable (early UFC events).
        Computes fight duration from rounds completed plus partial final round.
        """
        if totals is None:
            return None

        fight_duration = _compute_fight_duration_minutes(ending_round, ending_time)

        opp_sig_landed = (
            opponent_totals.sig_strikes_landed if opponent_totals else 0
        ) or 0
        opp_sig_attempted = (
            opponent_totals.sig_strikes_attempted if opponent_totals else 0
        ) or 0
        opp_td_landed = (
            opponent_totals.takedowns_landed if opponent_totals else 0
        ) or 0
        opp_td_attempted = (
            opponent_totals.takedowns_attempted if opponent_totals else 0
        ) or 0
        opp_control = (
            opponent_totals.control_time_seconds if opponent_totals else 0
        ) or 0

        return cls(
            fight_duration_minutes=fight_duration,
            sig_strikes_landed=totals.sig_strikes_landed or 0,
            sig_strikes_attempted=totals.sig_strikes_attempted or 0,
            sig_strikes_absorbed=opp_sig_landed,
            sig_strikes_absorbed_attempted=opp_sig_attempted,
            total_strikes_landed=totals.total_strikes_landed or 0,
            total_strikes_attempted=totals.total_strikes_attempted or 0,
            takedowns_landed=totals.takedowns_landed or 0,
            takedowns_attempted=totals.takedowns_attempted or 0,
            submissions_attempted=totals.submissions_attempted or 0,
            knockdowns=totals.knockdowns or 0,
            control_time_seconds=totals.control_time_seconds or 0,
            opponent_takedowns_landed=opp_td_landed,
            opponent_takedowns_attempted=opp_td_attempted,
            opponent_control_time_seconds=opp_control,
        )


# ---------------------------------------------------------------------------
# RollingStatsSnapshot — immutable frozen state returned by freeze()
# ---------------------------------------------------------------------------


class RollingStatsSnapshot(FrozenState):
    """Immutable snapshot of rolling statistics for all tracked fighters.

    Provides methods to compute rolling averages and variance from the
    frozen per-fighter fight history windows.
    """

    __slots__ = ("_fighter_windows",)

    def __init__(self, fighter_windows: Mapping[str, tuple[FightStats, ...]]) -> None:
        object.__setattr__(self, "_fighter_windows", fighter_windows)

    def get_rolling_averages(
        self, fighter_url: str
    ) -> dict[str, float | None] | None:
        """Compute rolling averages for a fighter. Returns None if no history."""
        window = self._fighter_windows.get(fighter_url)
        if not window:
            return None
        return _compute_averages(window)

    def get_rolling_variance(
        self, fighter_url: str
    ) -> dict[str, float | None] | None:
        """Compute rolling variance for a fighter. Requires 2+ fights."""
        window = self._fighter_windows.get(fighter_url)
        if not window or len(window) < 2:
            return None
        return _compute_variance(window)


# ---------------------------------------------------------------------------
# RollingStatsAccumulator — the StateComponent implementation
# ---------------------------------------------------------------------------


class RollingStatsAccumulator:
    """Accumulates per-fighter rolling statistics over a configurable window.

    Uses collections.deque with maxlen to automatically evict the oldest entry
    when the window is full. Implements the StateComponent protocol.
    """

    def __init__(self, window_size: int = 10) -> None:
        self._window_size = window_size
        self._fighters: dict[str, deque[FightStats]] = {}

    def update(self, fight: FightOutcomeView) -> None:
        """No-op: use update_with_totals which accepts totals data.

        The base StateComponent protocol's update() only receives outcome data.
        RollingStatsAccumulator requires fight totals to compute output metrics,
        so the replay engine calls update_with_totals() directly.
        """

    def update_with_totals(
        self,
        fight: FightOutcomeView,
        fighter_a_totals: FightTotals | None,
        fighter_b_totals: FightTotals | None,
    ) -> None:
        """Record per-fight stats for both fighters from totals data.

        Derives FightStats from raw totals and fight duration, then appends
        to each fighter's rolling window.
        """
        stats_a = FightStats.from_totals(
            totals=fighter_a_totals,
            opponent_totals=fighter_b_totals,
            ending_round=fight.ending_round,
            ending_time=fight.ending_time,
        )
        stats_b = FightStats.from_totals(
            totals=fighter_b_totals,
            opponent_totals=fighter_a_totals,
            ending_round=fight.ending_round,
            ending_time=fight.ending_time,
        )

        if stats_a is not None:
            self._get_or_create_window(fight.fighter_a_url).append(stats_a)
        if stats_b is not None:
            self._get_or_create_window(fight.fighter_b_url).append(stats_b)

    def freeze(self) -> RollingStatsSnapshot:
        """Return an immutable snapshot of current rolling windows."""
        frozen_windows: dict[str, tuple[FightStats, ...]] = {
            url: tuple(window) for url, window in self._fighters.items()
        }
        return RollingStatsSnapshot(frozen_windows)

    def _get_or_create_window(self, fighter_url: str) -> deque[FightStats]:
        if fighter_url not in self._fighters:
            self._fighters[fighter_url] = deque(maxlen=self._window_size)
        return self._fighters[fighter_url]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_fight_duration_minutes(ending_round: int, ending_time: str) -> float:
    """Compute total fight duration in minutes from round and time.

    Convention: rounds are 5 minutes each. ending_round and ending_time describe
    when the fight ended (round N at MM:SS into that round).
    """
    parts = ending_time.split(":")
    minutes = int(parts[0])
    seconds = int(parts[1]) if len(parts) > 1 else 0
    final_round_minutes = minutes + seconds / 60.0
    completed_rounds = ending_round - 1
    return completed_rounds * 5.0 + final_round_minutes


def _safe_divide(numerator: float, denominator: float) -> float | None:
    """Divide with zero-safety. Returns None when denominator is zero."""
    if denominator == 0:
        return None
    return numerator / denominator


def _compute_averages(window: tuple[FightStats, ...]) -> dict[str, float | None]:
    """Compute per-metric rolling averages across the window.

    Each metric is computed per-fight then averaged. Rate metrics (per-minute,
    per-15-min) are normalized by fight duration within each fight.
    """
    per_fight_metrics: list[dict[str, float | None]] = []

    for stats in window:
        duration = stats.fight_duration_minutes
        metrics: dict[str, float | None] = {}

        # Striking rates (per minute)
        metrics["sig_strikes_per_min"] = _safe_divide(
            stats.sig_strikes_landed, duration
        )
        metrics["sig_strikes_absorbed_per_min"] = _safe_divide(
            stats.sig_strikes_absorbed, duration
        )

        # Accuracy percentages
        metrics["striking_accuracy_pct"] = _safe_divide(
            stats.sig_strikes_landed, stats.sig_strikes_attempted
        )
        metrics["striking_defense_pct"] = _safe_divide(
            stats.sig_strikes_absorbed_attempted - stats.sig_strikes_absorbed,
            stats.sig_strikes_absorbed_attempted,
        ) if stats.sig_strikes_absorbed_attempted > 0 else None

        # Takedown rates (per 15 min)
        metrics["td_per_15min"] = _safe_divide(
            stats.takedowns_landed * 15.0, duration
        )
        metrics["td_accuracy_pct"] = _safe_divide(
            stats.takedowns_landed, stats.takedowns_attempted
        )

        # TD defense: proportion of opponent TDs defended
        opp_td_att = stats.opponent_takedowns_attempted
        metrics["td_defense_pct"] = _safe_divide(
            opp_td_att - stats.opponent_takedowns_landed, opp_td_att
        ) if opp_td_att > 0 else None

        # Submission rate (per 15 min)
        metrics["sub_attempts_per_15min"] = _safe_divide(
            stats.submissions_attempted * 15.0, duration
        )

        # Knockdown rate (knockdowns per sig strike thrown)
        metrics["knockdown_rate"] = _safe_divide(
            stats.knockdowns, stats.sig_strikes_attempted
        )

        # Damage ratio: sig landed / sig absorbed
        metrics["damage_ratio"] = _safe_divide(
            stats.sig_strikes_landed, stats.sig_strikes_absorbed
        )

        # Grappling dominance: (TD_landed + control) / (opp_TD_landed + opp_control)
        own_grappling = stats.takedowns_landed + stats.control_time_seconds
        opp_grappling = (
            stats.opponent_takedowns_landed + stats.opponent_control_time_seconds
        )
        metrics["grappling_dominance"] = _safe_divide(own_grappling, opp_grappling)

        # Control time per fight (raw seconds)
        metrics["control_time_per_fight"] = float(stats.control_time_seconds)

        per_fight_metrics.append(metrics)

    # Average across fights
    result: dict[str, float | None] = {}
    all_keys = per_fight_metrics[0].keys()
    for key in all_keys:
        values = [m[key] for m in per_fight_metrics if m[key] is not None]
        if values:
            result[key] = sum(values) / len(values)
        else:
            result[key] = None

    return result


def _compute_variance(window: tuple[FightStats, ...]) -> dict[str, float | None]:
    """Compute per-metric population variance across the window.

    Uses population variance (divide by N, not N-1) since the window is
    the complete recent history, not a sample from a larger population.
    """
    per_fight_metrics: list[dict[str, float | None]] = []

    for stats in window:
        duration = stats.fight_duration_minutes
        metrics: dict[str, float | None] = {}
        metrics["sig_strikes_per_min"] = _safe_divide(
            stats.sig_strikes_landed, duration
        )
        metrics["sig_strikes_absorbed_per_min"] = _safe_divide(
            stats.sig_strikes_absorbed, duration
        )
        metrics["striking_accuracy_pct"] = _safe_divide(
            stats.sig_strikes_landed, stats.sig_strikes_attempted
        )
        metrics["td_per_15min"] = _safe_divide(
            stats.takedowns_landed * 15.0, duration
        )
        metrics["damage_ratio"] = _safe_divide(
            stats.sig_strikes_landed, stats.sig_strikes_absorbed
        )
        per_fight_metrics.append(metrics)

    result: dict[str, float | None] = {}
    all_keys = per_fight_metrics[0].keys()
    for key in all_keys:
        values = [m[key] for m in per_fight_metrics if m[key] is not None]
        if len(values) < 2:
            result[key] = None
        else:
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            result[key] = variance

    return result
