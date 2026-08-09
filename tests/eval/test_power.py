"""Tests for power analysis and paired permutation testing.

Verifies:
- MDE computation matches a reference scipy.stats.norm.ppf calculation.
- Permutation test rejects when a large effect is planted (non-null).
- Permutation test does NOT reject when differences are pure noise (null).
- Same seed produces identical p-values (reproducibility).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ufc_edge.eval.power import minimum_detectable_effect, paired_permutation_test


class TestMinimumDetectableEffect:
    """MDE should match the standard paired-difference power formula."""

    def test_matches_scipy_reference_for_known_inputs(self) -> None:
        """Known n=200, sigma=0.04, alpha=0.05, power=0.8 must match reference."""
        n = 200
        sigma = 0.04
        alpha = 0.05
        power_val = 0.8

        # Reference computation using scipy directly
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power_val)
        expected_mde = (z_alpha + z_beta) * sigma / np.sqrt(n)

        result = minimum_detectable_effect(n_fights=n, sigma=sigma, alpha=alpha, power=power_val)

        assert result.n_fights == n
        assert result.estimated_sigma == sigma
        assert result.alpha == alpha
        assert result.power == power_val
        assert abs(result.mde - expected_mde) < 1e-12

    def test_larger_n_reduces_mde(self) -> None:
        """Doubling the sample size should reduce the MDE."""
        sigma = 0.05
        result_small = minimum_detectable_effect(n_fights=100, sigma=sigma)
        result_large = minimum_detectable_effect(n_fights=400, sigma=sigma)

        assert result_large.mde < result_small.mde

    def test_larger_sigma_increases_mde(self) -> None:
        """Higher variance in score differences means a larger effect is needed."""
        n = 300
        result_low = minimum_detectable_effect(n_fights=n, sigma=0.02)
        result_high = minimum_detectable_effect(n_fights=n, sigma=0.08)

        assert result_high.mde > result_low.mde

    def test_custom_alpha_and_power(self) -> None:
        """Non-default alpha=0.01, power=0.9 must also match reference."""
        n = 500
        sigma = 0.03
        alpha = 0.01
        power_val = 0.9

        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power_val)
        expected_mde = (z_alpha + z_beta) * sigma / np.sqrt(n)

        result = minimum_detectable_effect(n_fights=n, sigma=sigma, alpha=alpha, power=power_val)

        assert abs(result.mde - expected_mde) < 1e-12


class TestPairedPermutationTest:
    """Paired permutation test on per-fight Brier score differences."""

    def test_rejects_known_non_null(self) -> None:
        """A large planted effect (model much better than market) must be detected."""
        rng = np.random.default_rng(123)
        n = 200

        # Model Brier scores much lower than market (model is better)
        model_brier = rng.uniform(0.05, 0.15, size=n)
        market_brier = model_brier + 0.10  # constant large advantage

        result = paired_permutation_test(model_brier, market_brier, seed=99)

        assert result.p_value < 0.01
        assert result.observed_diff < 0.0  # model is better → negative difference
        assert result.n_permutations == 10_000

    def test_does_not_reject_null(self) -> None:
        """Two identical score arrays should yield a non-significant p-value."""
        rng = np.random.default_rng(42)
        n = 200

        scores = rng.uniform(0.1, 0.3, size=n)
        # Identical arrays → observed diff is 0, p-value should be high
        result = paired_permutation_test(scores, scores.copy(), seed=7)

        assert result.p_value >= 0.5  # Should be ~1.0 for identical arrays
        assert abs(result.observed_diff) < 1e-12

    def test_does_not_reject_shuffled_null(self) -> None:
        """Shuffled copy has zero mean difference — should not reject."""
        rng = np.random.default_rng(55)
        n = 300

        model_brier = rng.uniform(0.1, 0.3, size=n)
        # Shuffle is a permutation of the same values — mean diff expected ~0
        market_brier = model_brier.copy()
        rng.shuffle(market_brier)

        result = paired_permutation_test(model_brier, market_brier, seed=88)

        # With shuffled values the mean difference is small but not exactly 0;
        # the test should not reject at conventional significance
        assert result.p_value > 0.05

    def test_reproducibility_same_seed(self) -> None:
        """Same seed must produce the exact same p-value."""
        rng = np.random.default_rng(10)
        n = 150
        model_brier = rng.uniform(0.1, 0.25, size=n)
        market_brier = rng.uniform(0.12, 0.28, size=n)

        result_a = paired_permutation_test(model_brier, market_brier, seed=42)
        result_b = paired_permutation_test(model_brier, market_brier, seed=42)

        assert result_a.p_value == result_b.p_value
        assert result_a.observed_diff == result_b.observed_diff

    def test_different_seed_may_differ(self) -> None:
        """Different seeds can produce slightly different p-values."""
        rng = np.random.default_rng(77)
        n = 150
        model_brier = rng.uniform(0.1, 0.25, size=n)
        market_brier = rng.uniform(0.12, 0.28, size=n)

        result_a = paired_permutation_test(model_brier, market_brier, seed=1)
        result_b = paired_permutation_test(model_brier, market_brier, seed=2)

        # Observed diff is deterministic (data-only), but p-value depends on seed
        assert result_a.observed_diff == result_b.observed_diff

    def test_result_fields_populated(self) -> None:
        """All PermutationResult fields must be present and sensible."""
        rng = np.random.default_rng(33)
        n = 100
        model_brier = rng.uniform(0.1, 0.3, size=n)
        market_brier = rng.uniform(0.1, 0.3, size=n)

        result = paired_permutation_test(model_brier, market_brier, seed=42)

        assert 0.0 <= result.p_value <= 1.0
        assert result.n_permutations == 10_000
        assert result.ci_lower <= result.observed_diff <= result.ci_upper
