"""Tests for the stratified reliability artifact module.

Validates bucket boundaries, Clopper-Pearson confidence intervals, low-support
flagging, calibration error computation, and typed result structure.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from ufc_edge.eval.metrics import FightPrediction
from ufc_edge.eval.reliability import FIXED_BUCKETS, stratified_reliability
from ufc_edge.eval.schemas import ReliabilityBucket  # noqa: TC002 (used in isinstance)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bucket_predictions(
    bucket_assignments: dict[tuple[float, float], list[tuple[float, int]]],
) -> list[FightPrediction]:
    """Create FightPrediction objects from per-bucket (prob, label) pairs.

    Each prediction gets a unique event_id to simplify fixture construction.
    """
    predictions: list[FightPrediction] = []
    event_counter = 0
    for _bucket, entries in bucket_assignments.items():
        for prob, label in entries:
            predictions.append(
                FightPrediction(
                    event_id=f"event_{event_counter}",
                    prob=prob,
                    label=label,
                )
            )
            event_counter += 1
    return predictions


def _clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Reference Clopper-Pearson exact interval using scipy.stats.beta.

    Returns (ci_lower, ci_upper) for k successes in n trials.
    """
    if n == 0:
        return (0.0, 1.0)
    lower = float(stats.beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    upper = float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return (lower, upper)


# ---------------------------------------------------------------------------
# Test: four buckets produced with correct boundaries
# ---------------------------------------------------------------------------


class TestBucketBoundaries:
    """Stratified reliability produces exactly four buckets with fixed boundaries."""

    def test_exactly_four_buckets_returned(self) -> None:
        """Result contains exactly four reliability buckets."""
        predictions = _make_bucket_predictions({
            (0.1, 0.3): [(0.2, 1)] * 20,
            (0.3, 0.5): [(0.4, 0)] * 20,
            (0.5, 0.7): [(0.6, 1)] * 20,
            (0.7, 0.9): [(0.8, 0)] * 20,
        })
        result = stratified_reliability(predictions)
        assert len(result) == 4

    def test_bucket_boundaries_match_fixed_spec(self) -> None:
        """Each bucket has the expected (lower, upper) boundary pair."""
        predictions = _make_bucket_predictions({
            (0.1, 0.3): [(0.15, 1)] * 15,
            (0.3, 0.5): [(0.35, 0)] * 15,
            (0.5, 0.7): [(0.55, 1)] * 15,
            (0.7, 0.9): [(0.75, 0)] * 15,
        })
        result = stratified_reliability(predictions)
        expected_bounds = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9)]
        for bucket, (exp_lower, exp_upper) in zip(result, expected_bounds, strict=True):
            assert bucket.lower == pytest.approx(exp_lower)
            assert bucket.upper == pytest.approx(exp_upper)

    def test_fixed_buckets_constant_matches(self) -> None:
        """The module-level FIXED_BUCKETS constant has the documented values."""
        expected = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9)]
        assert FIXED_BUCKETS == expected  # noqa: SIM300

    def test_predictions_outside_buckets_excluded(self) -> None:
        """Predictions with probabilities outside [0.1, 0.9] are not counted."""
        predictions = [
            FightPrediction(event_id="e1", prob=0.05, label=1),
            FightPrediction(event_id="e2", prob=0.95, label=0),
            *[FightPrediction(event_id=f"e{i+3}", prob=0.5, label=1) for i in range(20)],
        ]
        result = stratified_reliability(predictions)
        total_counted = sum(b.n_fights for b in result)
        # Only the 20 predictions at 0.5 should land in the (0.5, 0.7) bucket
        assert total_counted == 20


# ---------------------------------------------------------------------------
# Test: Clopper-Pearson CI matches scipy reference
# ---------------------------------------------------------------------------


class TestClopperPearsonCI:
    """Binomial CIs use the Clopper-Pearson exact interval at 95% confidence."""

    @pytest.mark.parametrize(
        ("n", "k"),
        [
            (50, 25),   # balanced case
            (100, 70),  # skewed high
            (30, 3),    # skewed low
            (20, 0),    # no successes
            (20, 20),   # all successes
        ],
    )
    def test_ci_matches_scipy_reference(self, n: int, k: int) -> None:
        """CI bounds match scipy beta quantile computation for known n, k."""
        # Build predictions: k wins, n-k losses, all in the same bucket
        prob = 0.6  # places everything in the [0.5, 0.7] bucket
        predictions = [
            FightPrediction(event_id=f"e_{i}", prob=prob, label=1) for i in range(k)
        ] + [
            FightPrediction(event_id=f"e_{i+k}", prob=prob, label=0)
            for i in range(n - k)
        ]
        result = stratified_reliability(predictions)

        # Find the bucket that contains our predictions
        target_bucket = next(b for b in result if b.n_fights == n)

        expected_lower, expected_upper = _clopper_pearson(k, n, alpha=0.05)
        assert target_bucket.ci_lower == pytest.approx(expected_lower, abs=1e-8)
        assert target_bucket.ci_upper == pytest.approx(expected_upper, abs=1e-8)

    def test_ci_bounds_ordered(self) -> None:
        """Lower CI bound is always less than or equal to upper bound."""
        predictions = _make_bucket_predictions({
            (0.1, 0.3): [(0.2, 1)] * 12 + [(0.2, 0)] * 8,
            (0.3, 0.5): [(0.4, 1)] * 5 + [(0.4, 0)] * 15,
            (0.5, 0.7): [(0.6, 1)] * 18 + [(0.6, 0)] * 2,
            (0.7, 0.9): [(0.8, 1)] * 10 + [(0.8, 0)] * 10,
        })
        result = stratified_reliability(predictions)
        for bucket in result:
            assert bucket.ci_lower <= bucket.ci_upper

    def test_ci_contains_observed_rate(self) -> None:
        """The observed win rate lies within the confidence interval."""
        predictions = _make_bucket_predictions({
            (0.1, 0.3): [(0.2, 1)] * 15 + [(0.2, 0)] * 35,
            (0.3, 0.5): [(0.4, 1)] * 20 + [(0.4, 0)] * 30,
            (0.5, 0.7): [(0.6, 1)] * 30 + [(0.6, 0)] * 20,
            (0.7, 0.9): [(0.8, 1)] * 35 + [(0.8, 0)] * 15,
        })
        result = stratified_reliability(predictions)
        for bucket in result:
            assert bucket.ci_lower <= bucket.observed_win_rate <= bucket.ci_upper


# ---------------------------------------------------------------------------
# Test: LOW_SUPPORT flag fires when n < 10
# ---------------------------------------------------------------------------


class TestLowSupportFlag:
    """The low_support flag is True when a bucket has fewer than 10 fights."""

    def test_flag_true_below_threshold(self) -> None:
        """Bucket with fewer than 10 fights gets low_support=True."""
        predictions = [
            FightPrediction(event_id=f"e_{i}", prob=0.6, label=1) for i in range(5)
        ]
        result = stratified_reliability(predictions)
        target = next(b for b in result if b.n_fights == 5)
        assert target.low_support is True

    def test_flag_false_at_threshold(self) -> None:
        """Bucket with exactly 10 fights gets low_support=False."""
        predictions = [
            FightPrediction(event_id=f"e_{i}", prob=0.6, label=1) for i in range(10)
        ]
        result = stratified_reliability(predictions)
        target = next(b for b in result if b.n_fights == 10)
        assert target.low_support is False

    def test_flag_false_above_threshold(self) -> None:
        """Bucket with more than 10 fights gets low_support=False."""
        predictions = [
            FightPrediction(event_id=f"e_{i}", prob=0.6, label=i % 2) for i in range(50)
        ]
        result = stratified_reliability(predictions)
        target = next(b for b in result if b.n_fights == 50)
        assert target.low_support is False

    def test_empty_bucket_flagged_low_support(self) -> None:
        """A bucket with zero fights gets low_support=True."""
        # Only place predictions in the [0.5, 0.7] bucket
        predictions = [
            FightPrediction(event_id=f"e_{i}", prob=0.55, label=1) for i in range(20)
        ]
        result = stratified_reliability(predictions)
        empty_buckets = [b for b in result if b.n_fights == 0]
        for bucket in empty_buckets:
            assert bucket.low_support is True


# ---------------------------------------------------------------------------
# Test: calibration error = |mean_predicted − observed_win_rate|
# ---------------------------------------------------------------------------


class TestCalibrationError:
    """Calibration error equals the absolute gap between mean predicted and observed."""

    def test_calibration_error_formula(self) -> None:
        """calibration_error = |mean_predicted - observed_win_rate| for known values."""
        # All predictions at 0.6, all labels are 1 → observed_win_rate = 1.0
        predictions = [
            FightPrediction(event_id=f"e_{i}", prob=0.6, label=1) for i in range(30)
        ]
        result = stratified_reliability(predictions)
        target = next(b for b in result if b.n_fights == 30)
        expected_error = abs(0.6 - 1.0)
        assert target.calibration_error == pytest.approx(expected_error)

    def test_perfectly_calibrated_bucket_has_zero_error(self) -> None:
        """When mean predicted matches observed rate, calibration error is zero."""
        # 60% of predictions are wins → observed = 0.6, predicted = 0.6
        predictions = [
            FightPrediction(event_id=f"e_{i}", prob=0.6, label=1) for i in range(18)
        ] + [
            FightPrediction(event_id=f"e_{i+18}", prob=0.6, label=0) for i in range(12)
        ]
        result = stratified_reliability(predictions)
        target = next(b for b in result if b.n_fights == 30)
        assert target.calibration_error == pytest.approx(0.0, abs=1e-10)

    def test_overconfident_bucket_positive_error(self) -> None:
        """Overconfident predictions produce calibration error equal to the gap."""
        # Predict 0.8 (in [0.7, 0.9] bucket) but only 50% win
        predictions = [
            FightPrediction(event_id=f"e_{i}", prob=0.8, label=1) for i in range(10)
        ] + [
            FightPrediction(event_id=f"e_{i+10}", prob=0.8, label=0) for i in range(10)
        ]
        result = stratified_reliability(predictions)
        target = next(b for b in result if b.n_fights == 20)
        expected_error = abs(0.8 - 0.5)
        assert target.calibration_error == pytest.approx(expected_error)

    def test_mean_predicted_is_sample_mean_of_bucket_probs(self) -> None:
        """mean_predicted is the arithmetic mean of probabilities in the bucket."""
        # Mix of probabilities strictly within the [0.5, 0.7) bucket
        probs = [0.51, 0.55, 0.58, 0.60, 0.62, 0.63, 0.65, 0.66, 0.68, 0.69]
        predictions = [
            FightPrediction(event_id=f"e_{i}", prob=p, label=1)
            for i, p in enumerate(probs)
        ]
        result = stratified_reliability(predictions)
        target = next(b for b in result if b.n_fights == len(probs))
        expected_mean = float(np.mean(probs))
        assert target.mean_predicted == pytest.approx(expected_mean)


# ---------------------------------------------------------------------------
# Test: typed read function returns correctly structured result
# ---------------------------------------------------------------------------


class TestTypedResultStructure:
    """The return value conforms to the ReliabilityBucket schema for downstream use."""

    def test_returns_list_of_reliability_bucket_schema(self) -> None:
        """Each element in the result is a valid ReliabilityBucket schema instance."""
        predictions = _make_bucket_predictions({
            (0.1, 0.3): [(0.2, 1)] * 20,
            (0.3, 0.5): [(0.4, 0)] * 20,
            (0.5, 0.7): [(0.6, 1)] * 20,
            (0.7, 0.9): [(0.8, 0)] * 20,
        })
        result = stratified_reliability(predictions)
        assert isinstance(result, list)
        for bucket in result:
            assert isinstance(bucket, ReliabilityBucket)

    def test_bucket_fields_have_correct_types(self) -> None:
        """All bucket fields have the documented Python types."""
        predictions = [
            FightPrediction(event_id=f"e_{i}", prob=0.6, label=i % 2) for i in range(30)
        ]
        result = stratified_reliability(predictions)
        target = next(b for b in result if b.n_fights > 0)

        assert isinstance(target.lower, float)
        assert isinstance(target.upper, float)
        assert isinstance(target.n_fights, int)
        assert isinstance(target.mean_predicted, float)
        assert isinstance(target.observed_win_rate, float)
        assert isinstance(target.calibration_error, float)
        assert isinstance(target.ci_lower, float)
        assert isinstance(target.ci_upper, float)
        assert isinstance(target.low_support, bool)

    def test_buckets_immutable(self) -> None:
        """ReliabilityBucket instances are frozen (immutable)."""
        predictions = [
            FightPrediction(event_id=f"e_{i}", prob=0.6, label=1) for i in range(20)
        ]
        result = stratified_reliability(predictions)
        target = next(b for b in result if b.n_fights > 0)
        with pytest.raises((TypeError, ValueError)):
            target.n_fights = 999  # type: ignore[misc]

    def test_observed_win_rate_bounded_zero_one(self) -> None:
        """Observed win rate is always in [0, 1] for populated buckets."""
        rng = np.random.default_rng(123)
        predictions = [
            FightPrediction(
                event_id=f"e_{i}",
                prob=float(rng.uniform(0.1, 0.9)),
                label=int(rng.integers(0, 2)),
            )
            for i in range(200)
        ]
        result = stratified_reliability(predictions)
        for bucket in result:
            if bucket.n_fights > 0:
                assert 0.0 <= bucket.observed_win_rate <= 1.0

    def test_calibration_error_non_negative(self) -> None:
        """Calibration error is always non-negative (absolute value)."""
        rng = np.random.default_rng(456)
        predictions = [
            FightPrediction(
                event_id=f"e_{i}",
                prob=float(rng.uniform(0.1, 0.9)),
                label=int(rng.integers(0, 2)),
            )
            for i in range(150)
        ]
        result = stratified_reliability(predictions)
        for bucket in result:
            assert bucket.calibration_error >= 0.0
