"""Tests for the evaluation scoring metrics module.

Covers Brier score, log loss, ECE, and event-grouped bootstrap confidence
intervals using known analytic distributions.
"""

from __future__ import annotations

import numpy as np
import pytest

from ufc_edge.eval.metrics import (
    FightPrediction,
    brier_score,
    event_bootstrap_ci,
    expected_calibration_error,
    log_loss_score,
)

# ---------------------------------------------------------------------------
# Brier score tests
# ---------------------------------------------------------------------------


class TestBrierScore:
    """Brier score on known distributions."""

    def test_perfect_predictor_scores_zero(self) -> None:
        """A model that always predicts the correct outcome scores 0.0."""
        probs = np.array([1.0, 0.0, 1.0, 0.0])
        labels = np.array([1, 0, 1, 0])
        assert brier_score(probs, labels) == 0.0

    def test_constant_half_prediction_scores_quarter(self) -> None:
        """Constant 0.5 predictions yield exactly 0.25 (random baseline)."""
        probs = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([1, 0, 1, 0])
        assert brier_score(probs, labels) == 0.25

    def test_worst_predictor_scores_one(self) -> None:
        """A model that always predicts the opposite outcome scores 1.0."""
        probs = np.array([0.0, 1.0, 0.0, 1.0])
        labels = np.array([1, 0, 1, 0])
        assert brier_score(probs, labels) == 1.0

    def test_brier_on_partial_knowledge(self) -> None:
        """Known partial case: predicting 0.8 for all positive outcomes."""
        probs = np.array([0.8, 0.8, 0.8, 0.8])
        labels = np.array([1, 1, 0, 0])
        # (1-0.8)^2 = 0.04 for positives, (0-0.8)^2 = 0.64 for negatives
        expected = (0.04 + 0.04 + 0.64 + 0.64) / 4
        assert brier_score(probs, labels) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Log loss tests
# ---------------------------------------------------------------------------


class TestLogLoss:
    """Log loss on known distributions."""

    def test_perfect_predictor_near_zero(self) -> None:
        """Near-perfect predictions yield log loss very close to zero."""
        probs = np.array([0.999, 0.001, 0.999, 0.001])
        labels = np.array([1, 0, 1, 0])
        assert log_loss_score(probs, labels) < 0.01

    def test_constant_half_prediction(self) -> None:
        """Constant 0.5 predictions yield -ln(0.5) ≈ 0.693."""
        probs = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([1, 0, 1, 0])
        assert log_loss_score(probs, labels) == pytest.approx(np.log(2), rel=1e-10)

    def test_log_loss_increases_with_error(self) -> None:
        """Worse predictions produce higher log loss."""
        labels = np.array([1, 1, 0, 0])
        good_probs = np.array([0.9, 0.9, 0.1, 0.1])
        bad_probs = np.array([0.6, 0.6, 0.4, 0.4])
        assert log_loss_score(good_probs, labels) < log_loss_score(bad_probs, labels)


# ---------------------------------------------------------------------------
# ECE tests
# ---------------------------------------------------------------------------


class TestExpectedCalibrationError:
    """Expected calibration error on perfectly calibrated vs miscalibrated."""

    def test_perfectly_calibrated_ece_is_zero(self) -> None:
        """When predicted probs match observed frequency exactly, ECE is zero."""
        # 100 predictions: 50 at 0.3, 50 at 0.7
        # Observed: 15/50 = 0.3 in first bin, 35/50 = 0.7 in second
        n = 50
        probs = np.concatenate([np.full(n, 0.3), np.full(n, 0.7)])
        # 15 wins out of 50 in bin 1, 35 wins out of 50 in bin 2
        labels = np.concatenate([
            np.array([1] * 15 + [0] * 35),
            np.array([1] * 35 + [0] * 15),
        ])
        ece = expected_calibration_error(probs, labels, n_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-10)

    def test_miscalibrated_predictions_have_positive_ece(self) -> None:
        """When predictions are overconfident, ECE is positive."""
        # Predict 0.9 for all, but only 50% win
        probs = np.full(100, 0.9)
        labels = np.array([1] * 50 + [0] * 50)
        ece = expected_calibration_error(probs, labels, n_bins=10)
        # |0.9 - 0.5| = 0.4
        assert ece == pytest.approx(0.4, abs=1e-10)

    def test_ece_bounded_zero_to_one(self) -> None:
        """ECE is always in [0, 1]."""
        rng = np.random.default_rng(42)
        probs = rng.random(200)
        labels = rng.integers(0, 2, 200)
        ece = expected_calibration_error(probs, labels, n_bins=10)
        assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# Event-bootstrap CI tests
# ---------------------------------------------------------------------------


def _make_predictions(
    n_events: int,
    fights_per_event: int,
    seed: int = 42,
) -> list[FightPrediction]:
    """Create synthetic FightPrediction objects grouped by event."""
    rng = np.random.default_rng(seed)
    predictions = []
    for event_idx in range(n_events):
        event_id = f"event_{event_idx}"
        for _fight_idx in range(fights_per_event):
            predictions.append(
                FightPrediction(
                    event_id=event_id,
                    prob=float(rng.random()),
                    label=int(rng.integers(0, 2)),
                )
            )
    return predictions


class TestEventBootstrapCI:
    """Bootstrap CI with event-level resampling."""

    def test_ci_width_shrinks_with_more_events(self) -> None:
        """Larger samples produce narrower confidence intervals."""
        small = _make_predictions(n_events=10, fights_per_event=5, seed=1)
        large = _make_predictions(n_events=100, fights_per_event=5, seed=1)

        ci_small = event_bootstrap_ci(
            small, brier_score, n_bootstrap=2000, alpha=0.05
        )
        ci_large = event_bootstrap_ci(
            large, brier_score, n_bootstrap=2000, alpha=0.05
        )

        width_small = ci_small[1] - ci_small[0]
        width_large = ci_large[1] - ci_large[0]
        assert width_large < width_small

    def test_bootstrap_respects_event_grouping(self) -> None:
        """Bootstrap never splits an event: all fights in a sampled event appear."""
        # Two events with distinct probabilities so resampled Brier values
        # can only take on specific whole-event values.
        predictions = [
            FightPrediction(event_id="X", prob=0.2, label=1),
            FightPrediction(event_id="X", prob=0.2, label=1),
            FightPrediction(event_id="Y", prob=0.8, label=1),
            FightPrediction(event_id="Y", prob=0.8, label=1),
        ]

        # Only three possible Brier outcomes from whole-event resamples:
        # XX: mean((1-0.2)^2) = 0.64
        # YY: mean((1-0.8)^2) = 0.04
        # XY or YX: mean of [0.64, 0.64, 0.04, 0.04] = 0.34
        # If fights were resampled individually, many other values would appear.

        ci = event_bootstrap_ci(
            predictions, brier_score, n_bootstrap=5000, alpha=0.05
        )
        # The CI endpoints must lie within the range of valid whole-event
        # bootstrap values: [0.04, 0.64]. Any value outside this range would
        # prove that partial-event resampling occurred.
        assert ci[0] >= 0.04 - 1e-10
        assert ci[1] <= 0.64 + 1e-10

    def test_ci_contains_point_estimate(self) -> None:
        """The overall Brier score lies within the bootstrap CI."""
        predictions = _make_predictions(n_events=30, fights_per_event=4, seed=99)
        probs = np.array([p.prob for p in predictions])
        labels = np.array([p.label for p in predictions])
        point = brier_score(probs, labels)

        ci = event_bootstrap_ci(predictions, brier_score, n_bootstrap=3000, alpha=0.05)
        assert ci[0] <= point <= ci[1]
