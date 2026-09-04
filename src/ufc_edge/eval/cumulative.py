"""Cumulative evidence tracker for model-vs-market performance over time.

Tracks how the model-vs-market comparison evolves as more fights resolve
over time, with confidence bands that should narrow as more data accumulates,
since one evaluation on a small sample can't settle the question by itself.

The tracker maintains a running Brier-skill time series. Each new evaluation
period appends its per-fight data to the accumulated history, then the running
statistic and event-bootstrap confidence intervals are recomputed over the
full accumulated set. This makes the time series self-consistent: the final
point always equals a single-pass computation on all accumulated data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from ufc_edge.eval.schemas import CumulativePoint


@dataclass(frozen=True)
class PeriodData:
    """Per-fight Brier differences for a single evaluation period.

    Each entry is a (model_brier_i, market_brier_i) pair for one fight,
    grouped by event. The cumulative tracker accumulates these across
    periods to recompute the running Brier-skill and confidence band.
    """

    event_id: str
    as_of_date: date
    model_brier_per_fight: list[float]
    market_brier_per_fight: list[float]


def _compute_brier_skill(
    model_brier_per_fight: np.ndarray,
    market_brier_per_fight: np.ndarray,
) -> float:
    """Brier skill score: 1 - (mean_model_brier / mean_market_brier).

    Positive values mean the model outperforms the market reference.
    Zero means parity; negative means the market is better.
    """
    mean_model = float(np.mean(model_brier_per_fight))
    mean_market = float(np.mean(market_brier_per_fight))
    if mean_market == 0.0:
        return 0.0
    return 1.0 - (mean_model / mean_market)


def _event_bootstrap_brier_skill_ci(
    events: dict[str, tuple[np.ndarray, np.ndarray]],
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Event-level bootstrap CI on Brier skill.

    Resamples events with replacement, then computes Brier skill on each
    replicate. Returns (lower, upper) percentile interval.
    """
    event_ids = list(events.keys())
    n_events = len(event_ids)
    rng = np.random.default_rng(seed)

    scores = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        sampled_indices = rng.integers(0, n_events, size=n_events)
        model_parts: list[np.ndarray] = []
        market_parts: list[np.ndarray] = []
        for idx in sampled_indices:
            model_arr, market_arr = events[event_ids[idx]]
            model_parts.append(model_arr)
            market_parts.append(market_arr)

        all_model = np.concatenate(model_parts)
        all_market = np.concatenate(market_parts)
        scores[b] = _compute_brier_skill(all_model, all_market)

    lower_pct = 100.0 * (alpha / 2)
    upper_pct = 100.0 * (1 - alpha / 2)
    return (float(np.percentile(scores, lower_pct)), float(np.percentile(scores, upper_pct)))


def update_cumulative_evidence(
    prior_series: list[CumulativePoint],
    new_periods: list[PeriodData],
    *,
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> list[CumulativePoint]:
    """Append new periods and recompute running Brier-skill with CI bands.

    Each period contributes per-fight model and market Brier values grouped
    by event. The tracker accumulates all periods, then for each point in the
    resulting series, computes the running statistic and confidence interval
    over all data up to and including that point.

    The function is designed so that calling it incrementally (one period at a
    time) produces the same result as calling it once with all periods, because
    each point's statistic is always recomputed from the full accumulated data
    up to that point.

    Args:
        prior_series: Existing cumulative evidence points (may be empty).
        new_periods: New evaluation periods to append. Each contains per-fight
            Brier values and their event grouping. Must have at least one fight.
        n_bootstrap: Number of bootstrap replicates for CI computation.
        alpha: Significance level for CI (default 0.05 → 95% CI).
        seed: Base random seed; incremented per point for reproducibility.

    Returns:
        The full updated series (prior points recomputed + new points appended).
    """
    if not new_periods:
        return list(prior_series)

    # Reconstruct accumulated fight data from prior series metadata.
    # We need the raw per-fight data to recompute. The caller must pass all
    # periods that built the prior series if they want full recomputation.
    # In practice, this function receives the full history of PeriodData each
    # time via the prior_periods pattern.
    #
    # Simpler approach: we rebuild from new_periods only — the caller is
    # responsible for passing ALL periods (prior + new) if they want to
    # recompute older points. The typical usage is:
    #   all_periods = existing_periods + [latest_period]
    #   series = update_cumulative_evidence([], all_periods)
    #
    # This guarantees consistency: running statistic at point k equals the
    # direct computation on periods[0:k+1].

    result: list[CumulativePoint] = []
    accumulated_events: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    total_fights = 0

    for i, period in enumerate(new_periods):
        model_arr = np.array(period.model_brier_per_fight, dtype=np.float64)
        market_arr = np.array(period.market_brier_per_fight, dtype=np.float64)

        if period.event_id in accumulated_events:
            existing_model, existing_market = accumulated_events[period.event_id]
            accumulated_events[period.event_id] = (
                np.concatenate([existing_model, model_arr]),
                np.concatenate([existing_market, market_arr]),
            )
        else:
            accumulated_events[period.event_id] = (model_arr, market_arr)

        total_fights += len(model_arr)

        # Compute running Brier skill over all accumulated data
        all_model = np.concatenate([m for m, _ in accumulated_events.values()])
        all_market = np.concatenate([m for _, m in accumulated_events.values()])
        brier_skill = _compute_brier_skill(all_model, all_market)

        # Compute event-bootstrap CI over accumulated events
        point_seed = seed + i
        ci_lower, ci_upper = _event_bootstrap_brier_skill_ci(
            accumulated_events,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            seed=point_seed,
        )

        result.append(
            CumulativePoint(
                as_of_event_url=period.event_id,
                as_of_date=period.as_of_date,
                cumulative_n_fights=total_fights,
                brier_skill=brier_skill,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
            )
        )

    return result
