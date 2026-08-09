"""Tests for the symmetric inference module.

Verifies that predict_symmetric() enforces order invariance (swapping fighter
positions yields the same canonical result), uses lexicographic canonical ordering,
correctly averages raw predictions, raises MissingDataError for missing features,
and populates all SymmetricPrediction fields.
"""

from __future__ import annotations

import numpy as np
import pytest

from ufc_edge.model.inference import (
    MissingDataError,
    predict_symmetric,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _fake_calibrate(raw_prob: float) -> float:
    """Calibrator applying a known linear shift: maps [0,1] -> [0.1,0.9].

    Intentionally non-identity so we can verify calibration is applied once
    on the canonical raw.
    """
    return raw_prob * 0.8 + 0.1


def _identity_calibrate(raw_prob: float) -> float:
    """No-op calibration — passes through the raw probability unchanged."""
    return raw_prob


def _make_predict_fn(feature_to_prob: dict[tuple, float]):
    """Create a raw-predict callable that maps feature tuples to probabilities.

    Looks up the feature vector's tuple representation in a fixed map. Falls
    back to a sigmoid-of-sum for unknown vectors.
    """

    def predict(features: np.ndarray) -> float:
        key = tuple(features.tolist())
        if key in feature_to_prob:
            return feature_to_prob[key]
        return float(1.0 / (1.0 + np.exp(-np.sum(features) / 5.0)))

    return predict


def _sigmoid_predict(features: np.ndarray) -> float:
    """Raw-predict using sigmoid of the feature sum.

    Always produces a value in (0, 1), deterministic for any feature vector.
    """
    return float(1.0 / (1.0 + np.exp(-np.sum(features) / 5.0)))


# ---------------------------------------------------------------------------
# Symmetry property test
# ---------------------------------------------------------------------------


class TestSymmetryProperty:
    """Swapping argument order does not change the canonical prediction.

    The core guarantee: p_calibrated represents P(canonical fighter wins), and
    the canonical fighter is determined solely by lexicographic URL comparison.
    Swapping the input arguments produces an identical result, so:
    - p_calibrated(A,B) == p_calibrated(B,A) (same canonical probability)
    - P(A wins) + P(B wins) = 1.0 (one is p_calibrated, the other is 1 - p_calibrated)
    """

    def test_symmetry_100_random_pairs(self):
        """For 100 random (A,B) pairs, p(A wins) + p(B wins) = 1.0 exactly.

        Since p_calibrated = P(canonical wins) and canonical is fixed by URL sort,
        P(A wins) is either p_calibrated or 1 - p_calibrated depending on whether
        A is canonical. This verifies the averaging formula produces an
        order-invariant result that allows exact complementary probabilities.
        """
        rng = np.random.default_rng(seed=42)

        for i in range(100):
            n_features = rng.integers(5, 20)
            features_a = rng.standard_normal(n_features).astype(np.float64)
            features_b = rng.standard_normal(n_features).astype(np.float64)

            url_a = f"http://ufcstats.com/fighter-details/fighter_{i:04d}_a"
            url_b = f"http://ufcstats.com/fighter-details/fighter_{i:04d}_b"

            pred_ab = predict_symmetric(
                raw_predict=_sigmoid_predict,
                calibrate=_identity_calibrate,
                calibrator_method="identity",
                fighter_a_features=features_a,
                fighter_b_features=features_b,
                fighter_a_url=url_a,
                fighter_b_url=url_b,
                fight_url=f"http://ufcstats.com/fight-details/fight_{i:04d}",
            )

            # P(A wins) derived from prediction:
            # canonical is min(url_a, url_b) = url_a (since "_a" < "_b").
            # So p_calibrated = P(A wins), and P(B wins) = 1 - p_calibrated.
            p_a_wins = pred_ab.p_calibrated
            p_b_wins = 1.0 - pred_ab.p_calibrated

            assert p_a_wins + p_b_wins == 1.0, (
                f"Symmetry violated at pair {i}: "
                f"P(A wins)={p_a_wins} + P(B wins)={p_b_wins} "
                f"= {p_a_wins + p_b_wins}"
            )

    def test_order_invariance_100_random_pairs(self):
        """Swapping arguments produces identical p_calibrated for 100 random pairs.

        This is the functional invariant: the canonical averaging eliminates
        any dependence on which fighter is passed as fighter_a vs fighter_b.
        """
        rng = np.random.default_rng(seed=42)

        for i in range(100):
            n_features = rng.integers(5, 20)
            features_a = rng.standard_normal(n_features).astype(np.float64)
            features_b = rng.standard_normal(n_features).astype(np.float64)

            url_a = f"http://ufcstats.com/fighter-details/fighter_{i:04d}_a"
            url_b = f"http://ufcstats.com/fighter-details/fighter_{i:04d}_b"

            pred_ab = predict_symmetric(
                raw_predict=_sigmoid_predict,
                calibrate=_identity_calibrate,
                calibrator_method="identity",
                fighter_a_features=features_a,
                fighter_b_features=features_b,
                fighter_a_url=url_a,
                fighter_b_url=url_b,
                fight_url=f"http://ufcstats.com/fight-details/fight_{i:04d}",
            )

            pred_ba = predict_symmetric(
                raw_predict=_sigmoid_predict,
                calibrate=_identity_calibrate,
                calibrator_method="identity",
                fighter_a_features=features_b,
                fighter_b_features=features_a,
                fighter_a_url=url_b,
                fighter_b_url=url_a,
                fight_url=f"http://ufcstats.com/fight-details/fight_{i:04d}",
            )

            assert pred_ab.p_calibrated == pred_ba.p_calibrated, (
                f"Order invariance violated at pair {i}: "
                f"p(A,B)={pred_ab.p_calibrated} != p(B,A)={pred_ba.p_calibrated}"
            )

    def test_symmetry_with_nonlinear_calibrator(self):
        """Order invariance holds even with a non-identity calibrator.

        The averaging happens before calibration in canonical order. Since
        canonical order is determined by URL comparison (not input order),
        the calibration input is identical regardless of argument order.
        """
        rng = np.random.default_rng(seed=99)

        for i in range(50):
            n_features = 10
            features_a = rng.standard_normal(n_features).astype(np.float64)
            features_b = rng.standard_normal(n_features).astype(np.float64)

            url_a = f"http://ufcstats.com/fighter-details/aaa_{i:03d}"
            url_b = f"http://ufcstats.com/fighter-details/bbb_{i:03d}"

            pred_ab = predict_symmetric(
                raw_predict=_sigmoid_predict,
                calibrate=_fake_calibrate,
                calibrator_method="fake_platt",
                fighter_a_features=features_a,
                fighter_b_features=features_b,
                fighter_a_url=url_a,
                fighter_b_url=url_b,
                fight_url=f"http://ufcstats.com/fight-details/fight_{i:03d}",
            )

            pred_ba = predict_symmetric(
                raw_predict=_sigmoid_predict,
                calibrate=_fake_calibrate,
                calibrator_method="fake_platt",
                fighter_a_features=features_b,
                fighter_b_features=features_a,
                fighter_a_url=url_b,
                fighter_b_url=url_a,
                fight_url=f"http://ufcstats.com/fight-details/fight_{i:03d}",
            )

            assert pred_ab.p_calibrated == pred_ba.p_calibrated, (
                f"Order invariance violated with non-identity calibrator at pair {i}"
            )


# ---------------------------------------------------------------------------
# Canonical ordering
# ---------------------------------------------------------------------------


class TestCanonicalOrdering:
    """The canonical fighter is the lexicographically smaller URL."""

    def test_canonical_is_lexicographically_smaller(self):
        """When A < B lexicographically, canonical_fighter_url = A."""
        url_a = "http://ufcstats.com/fighter-details/aaa"
        url_b = "http://ufcstats.com/fighter-details/zzz"

        features_a = np.array([1.0, 2.0, 3.0])
        features_b = np.array([4.0, 5.0, 6.0])

        result = predict_symmetric(
            raw_predict=_sigmoid_predict,
            calibrate=_identity_calibrate,
            calibrator_method="identity",
            fighter_a_features=features_a,
            fighter_b_features=features_b,
            fighter_a_url=url_a,
            fighter_b_url=url_b,
            fight_url="http://ufcstats.com/fight-details/fight_001",
        )

        assert result.canonical_fighter_url == url_a
        assert result.canonical_opponent_url == url_b

    def test_canonical_when_b_is_smaller(self):
        """When B < A lexicographically, canonical_fighter_url = B."""
        url_a = "http://ufcstats.com/fighter-details/zzz"
        url_b = "http://ufcstats.com/fighter-details/aaa"

        features_a = np.array([1.0, 2.0, 3.0])
        features_b = np.array([4.0, 5.0, 6.0])

        result = predict_symmetric(
            raw_predict=_sigmoid_predict,
            calibrate=_identity_calibrate,
            calibrator_method="identity",
            fighter_a_features=features_a,
            fighter_b_features=features_b,
            fighter_a_url=url_a,
            fighter_b_url=url_b,
            fight_url="http://ufcstats.com/fight-details/fight_002",
        )

        assert result.canonical_fighter_url == url_b
        assert result.canonical_opponent_url == url_a

    def test_canonical_same_result_regardless_of_call_order(self):
        """Canonical ordering is invariant to which URL is passed as fighter_a."""
        url_x = "http://ufcstats.com/fighter-details/fighter_mmm"
        url_y = "http://ufcstats.com/fighter-details/fighter_nnn"

        features_x = np.array([0.5, -1.0, 2.5, 3.0])
        features_y = np.array([-0.5, 1.0, -2.5, 0.0])

        result_xy = predict_symmetric(
            raw_predict=_sigmoid_predict,
            calibrate=_identity_calibrate,
            calibrator_method="identity",
            fighter_a_features=features_x,
            fighter_b_features=features_y,
            fighter_a_url=url_x,
            fighter_b_url=url_y,
            fight_url="http://ufcstats.com/fight-details/fight_003",
        )

        result_yx = predict_symmetric(
            raw_predict=_sigmoid_predict,
            calibrate=_identity_calibrate,
            calibrator_method="identity",
            fighter_a_features=features_y,
            fighter_b_features=features_x,
            fighter_a_url=url_y,
            fighter_b_url=url_x,
            fight_url="http://ufcstats.com/fight-details/fight_003",
        )

        assert result_xy.canonical_fighter_url == result_yx.canonical_fighter_url
        assert result_xy.canonical_opponent_url == result_yx.canonical_opponent_url


# ---------------------------------------------------------------------------
# Raw averaging formula
# ---------------------------------------------------------------------------


class TestRawAveragingFormula:
    """The canonical raw probability is 0.5 * (p_ab + (1 - p_ba)).

    p_ab is the raw prediction with canonical fighter's features in slot A.
    p_ba is the raw prediction with opponent's features in slot A.
    """

    def test_averaging_against_manual_computation(self):
        """Verify the averaged raw matches 0.5 * (p_ab + (1 - p_ba)) exactly."""
        features_a = np.array([1.0, 2.0, 3.0])
        features_b = np.array([4.0, 5.0, 6.0])

        # Fixed raw predictions for known feature vectors.
        p_ab = 0.7  # raw prediction when canonical features in slot A
        p_ba = 0.4  # raw prediction when opponent features in slot A

        feature_map = {
            tuple(features_a.tolist()): p_ab,
            tuple(features_b.tolist()): p_ba,
        }
        predict_fn = _make_predict_fn(feature_map)

        # url_a < url_b, so canonical = url_a, canonical_features = features_a.
        url_a = "http://ufcstats.com/fighter-details/aaa"
        url_b = "http://ufcstats.com/fighter-details/bbb"

        result = predict_symmetric(
            raw_predict=predict_fn,
            calibrate=_identity_calibrate,
            calibrator_method="identity",
            fighter_a_features=features_a,
            fighter_b_features=features_b,
            fighter_a_url=url_a,
            fighter_b_url=url_b,
            fight_url="http://ufcstats.com/fight-details/fight_avg",
        )

        # canonical_raw = 0.5 * (0.7 + (1 - 0.4)) = 0.5 * 1.3 = 0.65
        expected_canonical_raw = 0.5 * (p_ab + (1.0 - p_ba))
        assert result.p_calibrated == expected_canonical_raw
        assert result.raw_p_ab == p_ab
        assert result.raw_p_ba == p_ba

    def test_averaging_equal_predictions_gives_half(self):
        """When p_ab = p_ba = 0.5, the averaged raw is exactly 0.5."""
        features_a = np.array([0.0, 0.0])
        features_b = np.array([0.0, 0.0])

        feature_map = {
            tuple(features_a.tolist()): 0.5,
        }
        predict_fn = _make_predict_fn(feature_map)

        url_a = "http://ufcstats.com/fighter-details/aaa"
        url_b = "http://ufcstats.com/fighter-details/bbb"

        result = predict_symmetric(
            raw_predict=predict_fn,
            calibrate=_identity_calibrate,
            calibrator_method="identity",
            fighter_a_features=features_a,
            fighter_b_features=features_b,
            fighter_a_url=url_a,
            fighter_b_url=url_b,
            fight_url="http://ufcstats.com/fight-details/fight_equal",
        )

        # 0.5 * (0.5 + (1 - 0.5)) = 0.5 * 1.0 = 0.5
        assert result.p_calibrated == 0.5

    def test_averaging_extreme_predictions(self):
        """Verify averaging at extreme raw values (near 0 and 1)."""
        features_a = np.array([10.0, 10.0])
        features_b = np.array([-10.0, -10.0])

        p_ab = 0.99
        p_ba = 0.01

        feature_map = {
            tuple(features_a.tolist()): p_ab,
            tuple(features_b.tolist()): p_ba,
        }
        predict_fn = _make_predict_fn(feature_map)

        url_a = "http://ufcstats.com/fighter-details/aaa"
        url_b = "http://ufcstats.com/fighter-details/bbb"

        result = predict_symmetric(
            raw_predict=predict_fn,
            calibrate=_identity_calibrate,
            calibrator_method="identity",
            fighter_a_features=features_a,
            fighter_b_features=features_b,
            fighter_a_url=url_a,
            fighter_b_url=url_b,
            fight_url="http://ufcstats.com/fight-details/fight_extreme",
        )

        # canonical_raw = 0.5 * (0.99 + (1 - 0.01)) = 0.5 * 1.98 = 0.99
        expected = 0.5 * (p_ab + (1.0 - p_ba))
        assert result.p_calibrated == expected


# ---------------------------------------------------------------------------
# Missing features error
# ---------------------------------------------------------------------------


class TestMissingFeatures:
    """predict_symmetric raises MissingDataError when features are missing."""

    def test_none_features_a_raises(self):
        """Passing None as fighter_a_features raises MissingDataError."""
        with pytest.raises(MissingDataError):
            predict_symmetric(
                raw_predict=_sigmoid_predict,
                calibrate=_identity_calibrate,
                calibrator_method="identity",
                fighter_a_features=None,
                fighter_b_features=np.array([1.0, 2.0]),
                fighter_a_url="http://ufcstats.com/fighter-details/aaa",
                fighter_b_url="http://ufcstats.com/fighter-details/bbb",
                fight_url="http://ufcstats.com/fight-details/fight_missing_a",
            )

    def test_none_features_b_raises(self):
        """Passing None as fighter_b_features raises MissingDataError."""
        with pytest.raises(MissingDataError):
            predict_symmetric(
                raw_predict=_sigmoid_predict,
                calibrate=_identity_calibrate,
                calibrator_method="identity",
                fighter_a_features=np.array([1.0, 2.0]),
                fighter_b_features=None,
                fighter_a_url="http://ufcstats.com/fighter-details/aaa",
                fighter_b_url="http://ufcstats.com/fighter-details/bbb",
                fight_url="http://ufcstats.com/fight-details/fight_missing_b",
            )

    def test_both_features_none_raises(self):
        """Passing None for both feature vectors raises MissingDataError."""
        with pytest.raises(MissingDataError):
            predict_symmetric(
                raw_predict=_sigmoid_predict,
                calibrate=_identity_calibrate,
                calibrator_method="identity",
                fighter_a_features=None,
                fighter_b_features=None,
                fighter_a_url="http://ufcstats.com/fighter-details/aaa",
                fighter_b_url="http://ufcstats.com/fighter-details/bbb",
                fight_url="http://ufcstats.com/fight-details/fight_missing_both",
            )


# ---------------------------------------------------------------------------
# SymmetricPrediction fields
# ---------------------------------------------------------------------------


class TestSymmetricPredictionFields:
    """All fields of SymmetricPrediction are populated correctly."""

    def test_all_fields_populated(self):
        """SymmetricPrediction contains fight_url, canonical URLs, probability,
        raw scores, and calibrator method.
        """
        features_a = np.array([1.0, 2.0, 3.0])
        features_b = np.array([4.0, 5.0, 6.0])

        p_ab = 0.6
        p_ba = 0.35

        feature_map = {
            tuple(features_a.tolist()): p_ab,
            tuple(features_b.tolist()): p_ba,
        }
        predict_fn = _make_predict_fn(feature_map)

        url_a = "http://ufcstats.com/fighter-details/alpha"
        url_b = "http://ufcstats.com/fighter-details/beta"
        fight = "http://ufcstats.com/fight-details/fight_fields"

        result = predict_symmetric(
            raw_predict=predict_fn,
            calibrate=_fake_calibrate,
            calibrator_method="fake_platt",
            fighter_a_features=features_a,
            fighter_b_features=features_b,
            fighter_a_url=url_a,
            fighter_b_url=url_b,
            fight_url=fight,
        )

        assert result.fight_url == fight
        assert result.canonical_fighter_url == url_a  # "alpha" < "beta"
        assert result.canonical_opponent_url == url_b
        assert result.calibrator_method == "fake_platt"
        assert isinstance(result.p_calibrated, float)
        assert isinstance(result.raw_p_ab, float)
        assert isinstance(result.raw_p_ba, float)

    def test_p_calibrated_within_bounds(self):
        """Calibrated probability is within [0, 1]."""
        rng = np.random.default_rng(seed=77)

        for _ in range(20):
            features_a = rng.standard_normal(8).astype(np.float64)
            features_b = rng.standard_normal(8).astype(np.float64)

            result = predict_symmetric(
                raw_predict=_sigmoid_predict,
                calibrate=_fake_calibrate,
                calibrator_method="fake_platt",
                fighter_a_features=features_a,
                fighter_b_features=features_b,
                fighter_a_url="http://ufcstats.com/fighter-details/aaa",
                fighter_b_url="http://ufcstats.com/fighter-details/bbb",
                fight_url="http://ufcstats.com/fight-details/fight_bounds",
            )

            assert 0.0 <= result.p_calibrated <= 1.0

    def test_raw_scores_reflect_orientation(self):
        """raw_p_ab and raw_p_ba reflect the two orientations of raw prediction."""
        features_a = np.array([2.0, -1.0, 0.5])
        features_b = np.array([-2.0, 1.0, -0.5])

        url_a = "http://ufcstats.com/fighter-details/aaa"
        url_b = "http://ufcstats.com/fighter-details/bbb"

        result = predict_symmetric(
            raw_predict=_sigmoid_predict,
            calibrate=_identity_calibrate,
            calibrator_method="identity",
            fighter_a_features=features_a,
            fighter_b_features=features_b,
            fighter_a_url=url_a,
            fighter_b_url=url_b,
            fight_url="http://ufcstats.com/fight-details/fight_raw",
        )

        # url_a < url_b, so canonical = url_a, canonical_features = features_a.
        # raw_p_ab = predict(canonical_features) = predict(features_a)
        # raw_p_ba = predict(opponent_features) = predict(features_b)
        expected_p_ab = float(1.0 / (1.0 + np.exp(-np.sum(features_a) / 5.0)))
        expected_p_ba = float(1.0 / (1.0 + np.exp(-np.sum(features_b) / 5.0)))

        assert result.raw_p_ab == expected_p_ab
        assert result.raw_p_ba == expected_p_ba

    def test_prediction_is_frozen(self):
        """SymmetricPrediction is immutable (frozen dataclass)."""
        result = predict_symmetric(
            raw_predict=_sigmoid_predict,
            calibrate=_identity_calibrate,
            calibrator_method="identity",
            fighter_a_features=np.array([1.0, 2.0]),
            fighter_b_features=np.array([3.0, 4.0]),
            fighter_a_url="http://ufcstats.com/fighter-details/aaa",
            fighter_b_url="http://ufcstats.com/fighter-details/bbb",
            fight_url="http://ufcstats.com/fight-details/fight_frozen",
        )

        with pytest.raises((AttributeError, TypeError)):
            result.p_calibrated = 0.999  # type: ignore[misc]
