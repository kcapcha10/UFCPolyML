"""Stratified reliability reporting for calibrated predictions.

Reports how far off the model's predictions are within each probability range,
so users can see where the model is actually trustworthy versus where it isn't.
The output is a typed list of bucket results that any downstream module can
consume programmatically without coupling to a particular report format.

Four fixed probability buckets partition the prediction space:
[0.1–0.3], [0.3–0.5], [0.5–0.7], [0.7–0.9]. Each bucket reports the observed
win rate, mean predicted probability, absolute calibration error, and a
Clopper-Pearson exact 95% binomial confidence interval on the observed rate.
Buckets with fewer than 10 fights are flagged as low-support.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist

from ufc_edge.eval.metrics import FightPrediction
from ufc_edge.eval.schemas import ReliabilityBucket

# Four fixed probability ranges spanning the non-extreme prediction space.
FIXED_BUCKETS: list[tuple[float, float]] = [
    (0.1, 0.3),
    (0.3, 0.5),
    (0.5, 0.7),
    (0.7, 0.9),
]

# Threshold below which a bucket's estimates are considered unreliable.
LOW_SUPPORT: int = 10


def _clopper_pearson_ci(
    successes: int,
    trials: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Clopper-Pearson exact binomial confidence interval.

    Uses the relationship between the binomial distribution and the beta
    distribution to produce exact coverage-guaranteed bounds.

    Returns (lower, upper) at the (1 - alpha) confidence level.
    Handles edge cases: 0 successes → lower=0, all successes → upper=1.
    """
    if trials == 0:
        return (0.0, 1.0)

    if successes == 0:
        lower = 0.0
    else:
        lower = float(beta_dist.ppf(alpha / 2, successes, trials - successes + 1))

    if successes == trials:
        upper = 1.0
    else:
        upper = float(beta_dist.ppf(1 - alpha / 2, successes + 1, trials - successes))

    return (lower, upper)


def stratified_reliability(
    predictions: list[FightPrediction],
) -> list[ReliabilityBucket]:
    """Compute reliability metrics for each fixed probability bucket.

    Partitions predictions into FIXED_BUCKETS by their predicted probability.
    Bucket boundaries are [lower, upper) except the final bucket which is
    [lower, upper]. Predictions outside all buckets (below 0.1 or above 0.9)
    are not assigned to any bucket.

    Each bucket receives a Clopper-Pearson 95% confidence interval on the
    observed win rate and a low-support flag when the fight count is below 10.
    """
    probs = np.array([p.prob for p in predictions], dtype=np.float64)
    labels = np.array([p.label for p in predictions], dtype=np.int32)

    buckets: list[ReliabilityBucket] = []

    for i, (lower, upper) in enumerate(FIXED_BUCKETS):
        is_last_bucket = i == len(FIXED_BUCKETS) - 1

        if is_last_bucket:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)

        bucket_probs = probs[mask]
        bucket_labels = labels[mask]
        n_fights = int(np.sum(mask))

        if n_fights == 0:
            buckets.append(
                ReliabilityBucket(
                    lower=lower,
                    upper=upper,
                    n_fights=0,
                    mean_predicted=0.0,
                    observed_win_rate=0.0,
                    calibration_error=0.0,
                    ci_lower=0.0,
                    ci_upper=1.0,
                    low_support=True,
                )
            )
            continue

        mean_predicted = float(np.mean(bucket_probs))
        observed_win_rate = float(np.mean(bucket_labels))
        calibration_error = abs(mean_predicted - observed_win_rate)

        successes = int(np.sum(bucket_labels))
        ci_lower, ci_upper = _clopper_pearson_ci(successes, n_fights)

        buckets.append(
            ReliabilityBucket(
                lower=lower,
                upper=upper,
                n_fights=n_fights,
                mean_predicted=mean_predicted,
                observed_win_rate=observed_win_rate,
                calibration_error=calibration_error,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                low_support=n_fights < LOW_SUPPORT,
            )
        )

    return buckets
