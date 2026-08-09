"""Scoring metrics for evaluation: Brier, log loss, ECE, and event-bootstrap CI.

Pure functions with no side effects, no DuckDB, and no MLflow dependencies.
The bootstrap resamples at the event level — all fights from a sampled event
are included together to respect event grouping.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FightPrediction:
    """A single fight prediction paired with its event grouping and true label.

    Used by bootstrap and stratified reliability to maintain event structure.
    """

    event_id: str
    prob: float
    label: int


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary outcomes.

    Returns 0.0 for a perfect predictor, 0.25 for constant 0.5, 1.0 for worst.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean((probs - labels) ** 2))


def log_loss_score(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-15) -> float:
    """Negative log-likelihood averaged over predictions.

    Clips probabilities to [eps, 1-eps] to avoid log(0).
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    clipped = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Weighted-average absolute calibration error across equal-width bins.

    Bins are constructed from [0, 1] in n_bins equal-width intervals. Empty bins
    are excluded from the average. The weight of each bin is its sample fraction.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n = len(probs)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (probs >= lower) & (probs < upper)
        else:
            # Last bin includes the right edge
            mask = (probs >= lower) & (probs <= upper)

        bin_count = int(np.sum(mask))
        if bin_count == 0:
            continue

        bin_probs = probs[mask]
        bin_labels = labels[mask]
        avg_predicted = float(np.mean(bin_probs))
        avg_observed = float(np.mean(bin_labels))
        ece += (bin_count / n) * abs(avg_predicted - avg_observed)

    return ece


def event_bootstrap_ci(
    predictions: list[FightPrediction],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Event-level bootstrap confidence interval for a scoring metric.

    Resamples events (not individual fights) with replacement. All fights
    from each sampled event are included in the bootstrap replicate. Returns
    the (alpha/2, 1-alpha/2) percentile interval.
    """
    # Group fights by event
    events: dict[str, list[FightPrediction]] = defaultdict(list)
    for pred in predictions:
        events[pred.event_id].append(pred)

    event_ids = list(events.keys())
    n_events = len(event_ids)
    rng = np.random.default_rng(seed)

    scores = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        # Resample events with replacement
        sampled_indices = rng.integers(0, n_events, size=n_events)
        probs_list: list[float] = []
        labels_list: list[int] = []
        for idx in sampled_indices:
            event_preds = events[event_ids[idx]]
            for pred in event_preds:
                probs_list.append(pred.prob)
                labels_list.append(pred.label)

        probs_arr = np.array(probs_list, dtype=np.float64)
        labels_arr = np.array(labels_list, dtype=np.float64)
        scores[b] = metric_fn(probs_arr, labels_arr)

    lower_pct = 100.0 * (alpha / 2)
    upper_pct = 100.0 * (1 - alpha / 2)
    return (float(np.percentile(scores, lower_pct)), float(np.percentile(scores, upper_pct)))
