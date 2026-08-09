"""Mismatch computation and magnitude gating.

Computes the signed difference between the model's calibrated probability and
the market midpoint, then applies a magnitude gate to decide whether the
mismatch is large enough to be meaningful.

The gating philosophy: a mismatch only counts as a signal if it's bigger than
the model's typical error in that probability range, otherwise it's likely just
noise. The threshold is k × calibration_error for the bucket that p_model falls
into, where k is a configurable multiplier (default 2.0).
"""

from __future__ import annotations

from typing import Protocol

from ufc_edge.report.schemas import GateResult, GateVerdict

# ── Bucket protocol ───────────────────────────────────────────────────────────


class BucketEntryLike(Protocol):
    """Structural type for a single bucket entry from the calibration artifact."""

    bucket_id: str
    lower: float
    upper: float
    n: int
    calibration_error: float
    ci_lower: float
    ci_upper: float


class BucketArtifactLike(Protocol):
    """Structural type for the bucket artifact produced by model evaluation."""

    buckets: list[BucketEntryLike]


# ── Public API ────────────────────────────────────────────────────────────────


def compute_mismatch(p_model: float, p_market_mid: float) -> float:
    """Compute signed mismatch: model probability minus market midpoint.

    Positive means the model thinks the fighter is more likely to win than
    the market does; negative means the model is lower.
    """
    return p_model - p_market_mid


def assign_bucket(
    p_model: float, buckets: list[BucketEntryLike]
) -> BucketEntryLike | None:
    """Find the calibration bucket for a model probability.

    Bucket boundaries use [lower, upper) — half-open on the right — except
    the final bucket which is inclusive on both ends [lower, upper]. A
    probability sitting exactly on a boundary between two buckets is assigned
    to the lower bucket. Probabilities outside [0.1, 0.9] map to the nearest
    boundary bucket.

    Returns the matching bucket entry, or None if the bucket list is empty.
    """
    if not buckets:
        return None

    sorted_buckets = sorted(buckets, key=lambda b: b.lower)

    # Clamp to the covered range for out-of-bounds probabilities
    first_lower = sorted_buckets[0].lower
    last_upper = sorted_buckets[-1].upper
    if p_model < first_lower:
        return sorted_buckets[0]
    if p_model > last_upper:
        return sorted_buckets[-1]

    # Walk buckets; a value on a boundary goes to the lower (earlier) bucket
    for i, bucket in enumerate(sorted_buckets):
        is_last = i == len(sorted_buckets) - 1
        if is_last:
            # Final bucket is inclusive on both ends: [lower, upper]
            if bucket.lower <= p_model <= bucket.upper:
                return bucket
        else:
            # Half-open: [lower, upper)
            # But boundary values (p_model == upper) go to this (lower) bucket
            next_lower = sorted_buckets[i + 1].lower
            if bucket.lower <= p_model and p_model < next_lower:
                return bucket
            # Also catch p_model exactly on this bucket's upper when it equals
            # next bucket's lower (boundary → lower bucket)
            if p_model == bucket.upper:
                return bucket

    # Fallback: nearest bucket by distance to midpoints
    return min(sorted_buckets, key=lambda b: abs(p_model - (b.lower + b.upper) / 2))


def apply_gate(
    mismatch: float,
    p_model: float,
    bucket_artifact: BucketArtifactLike,
    k: float,
) -> GateResult:
    """Apply magnitude gate: flag mismatches exceeding k × calibration error.

    A mismatch only counts as a signal if its absolute value is strictly
    greater than k times the bucket's calibration error. If it's at or below
    that threshold, it's within the model's expected noise for that
    probability range.

    Returns a GateResult with full bucket transparency columns so the human
    reader can see exactly what drove the decision.
    """
    bucket = assign_bucket(p_model, bucket_artifact.buckets)

    if bucket is None:
        return GateResult(
            verdict=GateVerdict.NO_BUCKET_DATA,
            bucket_id="",
            bucket_n=0,
            bucket_calibration_error=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
        )

    threshold = k * bucket.calibration_error
    verdict = (
        GateVerdict.FLAGGED
        if abs(mismatch) > threshold
        else GateVerdict.WITHIN_NOISE
    )

    return GateResult(
        verdict=verdict,
        bucket_id=bucket.bucket_id,
        bucket_n=bucket.n,
        bucket_calibration_error=bucket.calibration_error,
        ci_lower=bucket.ci_lower,
        ci_upper=bucket.ci_upper,
    )
