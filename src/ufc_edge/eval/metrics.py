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


# ---------------------------------------------------------------------------
# History-depth stratification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StratumMetrics:
    """Brier and ECE for a single stratum of fights.

    Fighters with very few prior UFC fights are the least-informed predictions
    the model makes, so their calibration quality is measured and reported
    separately instead of being averaged away into the overall number.
    """

    n_fights: int
    brier: float
    ece: float


@dataclass(frozen=True)
class StratifiedMetrics:
    """Per-stratum and overall metrics split by fighter history depth.

    The sparse group contains fights where the lesser-experienced fighter has
    at most `threshold` prior UFC bouts. The remainder form the non-sparse group.
    """

    sparse: StratumMetrics
    non_sparse: StratumMetrics
    threshold: int


def stratify_by_history_depth(
    predictions: list[FightPrediction],
    min_prior_ufc_fights: list[int],
    threshold: int = 3,
    n_bins: int = 10,
) -> StratifiedMetrics:
    """Split predictions by fighter history depth and report Brier/ECE per stratum.

    Each fight's history depth is the lesser of both fighters' prior-UFC-fight
    counts (passed in via min_prior_ufc_fights). A fight is tagged SPARSE_HISTORY
    when its min_prior_ufc_fights is at or below the threshold.

    This separates out fights where at least one competitor has very little UFC
    track record — the predictions the model is least confident about — so their
    accuracy is visible rather than hidden inside aggregate numbers.

    Params: predictions — fight outcomes with predicted probabilities.
            min_prior_ufc_fights — per-fight minimum of both fighters' prior counts.
            threshold — at-or-below this count tags a fight as sparse (default 3).
            n_bins — number of equal-width bins for ECE computation.
    Returns: StratifiedMetrics with separate Brier/ECE for each group.
    Assumes: len(predictions) == len(min_prior_ufc_fights); all counts >= 0.
    """
    if len(predictions) != len(min_prior_ufc_fights):
        msg = (
            f"predictions length ({len(predictions)}) must match "
            f"min_prior_ufc_fights length ({len(min_prior_ufc_fights)})"
        )
        raise ValueError(msg)

    if any(count < 0 for count in min_prior_ufc_fights):
        raise ValueError("min_prior_ufc_fights must contain only non-negative counts")

    sparse_probs: list[float] = []
    sparse_labels: list[int] = []
    non_sparse_probs: list[float] = []
    non_sparse_labels: list[int] = []

    for pred, count in zip(predictions, min_prior_ufc_fights, strict=True):
        if count <= threshold:
            sparse_probs.append(pred.prob)
            sparse_labels.append(pred.label)
        else:
            non_sparse_probs.append(pred.prob)
            non_sparse_labels.append(pred.label)

    sparse_probs_arr = np.array(sparse_probs, dtype=np.float64)
    sparse_labels_arr = np.array(sparse_labels, dtype=np.float64)
    non_sparse_probs_arr = np.array(non_sparse_probs, dtype=np.float64)
    non_sparse_labels_arr = np.array(non_sparse_labels, dtype=np.float64)

    sparse_metrics = StratumMetrics(
        n_fights=len(sparse_probs),
        brier=brier_score(sparse_probs_arr, sparse_labels_arr) if sparse_probs else 0.0,
        ece=expected_calibration_error(sparse_probs_arr, sparse_labels_arr, n_bins=n_bins)
        if sparse_probs
        else 0.0,
    )

    non_sparse_metrics = StratumMetrics(
        n_fights=len(non_sparse_probs),
        brier=brier_score(non_sparse_probs_arr, non_sparse_labels_arr) if non_sparse_probs else 0.0,
        ece=expected_calibration_error(non_sparse_probs_arr, non_sparse_labels_arr, n_bins=n_bins)
        if non_sparse_probs
        else 0.0,
    )

    return StratifiedMetrics(
        sparse=sparse_metrics,
        non_sparse=non_sparse_metrics,
        threshold=threshold,
    )
