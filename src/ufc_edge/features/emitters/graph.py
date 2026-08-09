"""Graph-based feature emitter for Elo, Glicko-2, PageRank, and common opponents.

Reads frozen state from the four graph components (EloTracker, Glicko2Tracker,
PageRankGraph, CommonOpponentIndex) and emits rating features, trajectory slope,
uncertainty, prestige, and common-opponent quality metrics. All features degrade
gracefully to None when their source component or fighter data is unavailable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ufc_edge.features.contracts import EmitContext

# Number of history entries required for trajectory slope calculation.
_TRAJECTORY_WINDOW = 5


class GraphEmitter:
    """Emits graph-structural features from Elo, Glicko-2, PageRank, and common opponents.

    Reads frozen component snapshots from EmitContext.components and produces a flat
    dict of feature values. Uses linear regression (least squares) over the last 5
    Elo history entries for trajectory slope; returns None when fewer than 5 exist.
    """

    name: str = "graph"

    def emit(self, context: EmitContext) -> dict[str, float | str | None]:
        """Emit graph features for the focal fighter."""
        features: dict[str, float | str | None] = {}

        self._emit_elo(context, features)
        self._emit_glicko2(context, features)
        self._emit_pagerank(context, features)
        self._emit_common_opponents(context, features)

        return features

    def _emit_elo(
        self, context: EmitContext, features: dict[str, float | str | None]
    ) -> None:
        """Extract Elo rating, peak, current-vs-peak ratio, and trajectory slope."""
        elo_state = context.components.get("elo")
        record = None
        if elo_state is not None and hasattr(elo_state, "get"):
            record = elo_state.get(context.fighter_url)

        if record is None:
            features["elo_rating"] = None
            features["elo_trajectory_last5"] = None
            features["elo_peak"] = None
            features["elo_current_vs_peak"] = None
            return

        features["elo_rating"] = record.rating
        features["elo_peak"] = record.peak
        features["elo_current_vs_peak"] = _current_vs_peak(record.rating, record.peak)
        features["elo_trajectory_last5"] = _trajectory_slope(record.history)

    def _emit_glicko2(
        self, context: EmitContext, features: dict[str, float | str | None]
    ) -> None:
        """Extract Glicko-2 rating and rating deviation."""
        glicko2_state = context.components.get("glicko2")
        if glicko2_state is None:
            features["glicko2_rating"] = None
            features["glicko2_rd"] = None
            return

        record = glicko2_state.get_record(context.fighter_url)
        features["glicko2_rating"] = record.mu
        features["glicko2_rd"] = record.rd

    def _emit_pagerank(
        self, context: EmitContext, features: dict[str, float | str | None]
    ) -> None:
        """Extract PageRank prestige score."""
        pr_state = context.components.get("pagerank")
        if pr_state is None:
            features["pagerank_score"] = None
            return

        scores = pr_state.scores
        features["pagerank_score"] = scores.get(context.fighter_url)

    def _emit_common_opponents(
        self, context: EmitContext, features: dict[str, float | str | None]
    ) -> None:
        """Compute common-opponent count, quality scores, and win rates."""
        common_state = context.components.get("common_opponents")
        if common_state is None:
            features["n_common_opponents"] = 0
            features["common_opp_score_a"] = None
            features["common_opp_score_b"] = None
            features["common_opp_score_delta"] = None
            features["common_opp_a_win_rate"] = None
            features["common_opp_b_win_rate"] = None
            return

        common_records = common_state.get_common_opponents(
            context.fighter_url,
            context.opponent_url,
            as_of=context.event_date,
        )

        n_common = len(common_records)
        features["n_common_opponents"] = n_common

        if n_common == 0:
            features["common_opp_score_a"] = None
            features["common_opp_score_b"] = None
            features["common_opp_score_delta"] = None
            features["common_opp_a_win_rate"] = None
            features["common_opp_b_win_rate"] = None
            return

        # Aggregate quality-weighted scores across all common opponents
        score_a = sum(r.quality_score * r.recency_weight for r in common_records)
        score_b = sum(r.quality_score * r.recency_weight for r in common_records)

        # Compute per-fighter win rates against common opponents using histories
        a_win_rate = _compute_win_rate_vs_common(
            common_state, context.fighter_url, common_records, context.event_date
        )
        b_win_rate = _compute_win_rate_vs_common(
            common_state, context.opponent_url, common_records, context.event_date
        )

        # Score differentiation: weight by fighter's win/loss against each opponent
        score_a = _compute_weighted_score(
            common_state, context.fighter_url, common_records, context.event_date
        )
        score_b = _compute_weighted_score(
            common_state, context.opponent_url, common_records, context.event_date
        )

        features["common_opp_score_a"] = score_a
        features["common_opp_score_b"] = score_b
        features["common_opp_score_delta"] = score_a - score_b
        features["common_opp_a_win_rate"] = a_win_rate
        features["common_opp_b_win_rate"] = b_win_rate


def _current_vs_peak(rating: float, peak: float) -> float | None:
    """Compute current rating as a fraction of peak. None if peak is zero."""
    if peak == 0.0:
        return None
    return rating / peak


def _trajectory_slope(history: tuple[float, ...]) -> float | None:
    """Compute linear regression slope over the last 5 ratings.

    Uses numpy polyfit degree 1 (least squares) over (index, rating) pairs.
    Returns None if fewer than 5 entries are available.
    """
    if len(history) < _TRAJECTORY_WINDOW:
        return None

    last_5 = history[-_TRAJECTORY_WINDOW:]
    x = np.arange(_TRAJECTORY_WINDOW, dtype=np.float64)
    y = np.array(last_5, dtype=np.float64)
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0])


def _compute_win_rate_vs_common(
    common_state: object,
    fighter_url: str,
    common_records: list,
    as_of: object,
) -> float:
    """Compute a fighter's win rate against the set of common opponents.

    Examines the fight history in the frozen state to determine wins and losses
    against each common opponent. Draws (won=None) are excluded from the
    calculation.
    """
    common_opp_urls = {r.opponent_url for r in common_records}
    histories = getattr(common_state, "_histories", {})
    entries = histories.get(fighter_url, [])

    wins = 0
    total = 0
    for entry in entries:
        if entry.opponent_url in common_opp_urls and entry.won is not None:
            total += 1
            if entry.won:
                wins += 1

    if total == 0:
        return 0.0
    return wins / total


def _compute_weighted_score(
    common_state: object,
    fighter_url: str,
    common_records: list,
    as_of: object,
) -> float:
    """Compute quality-weighted score for a fighter based on common-opponent outcomes.

    Multiplies each common opponent's quality_score by recency_weight and by the
    fighter's win rate against that specific opponent.
    """
    common_opp_map = {r.opponent_url: r for r in common_records}
    histories = getattr(common_state, "_histories", {})
    entries = histories.get(fighter_url, [])

    # Per-opponent win rate
    opp_wins: dict[str, int] = {}
    opp_total: dict[str, int] = {}
    for entry in entries:
        if entry.opponent_url in common_opp_map and entry.won is not None:
            opp_wins[entry.opponent_url] = opp_wins.get(entry.opponent_url, 0) + (
                1 if entry.won else 0
            )
            opp_total[entry.opponent_url] = opp_total.get(entry.opponent_url, 0) + 1

    total_score = 0.0
    for opp_url, record in common_opp_map.items():
        wins = opp_wins.get(opp_url, 0)
        total = opp_total.get(opp_url, 0)
        win_rate = wins / total if total > 0 else 0.0
        total_score += record.quality_score * record.recency_weight * win_rate

    return total_score
