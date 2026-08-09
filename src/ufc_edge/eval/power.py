"""Power analysis and paired permutation testing for model vs. market comparison.

Computes the smallest real effect this evaluation could reliably detect given how
much data it has, and tests whether the model actually beats the market rather than
just getting lucky on a small sample. Operates on paired per-fight Brier score
differences (model_brier_i - market_brier_i for each fight i).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ufc_edge.eval.schemas import PermutationResult, PowerResult


def minimum_detectable_effect(
    n_fights: int,
    sigma: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> PowerResult:
    """Compute the minimum detectable effect for a paired-difference test.

    Uses (z_alpha + z_beta) × σ / √n where σ is the standard deviation of
    per-fight Brier differences estimated from development folds.

    Returns a PowerResult with the MDE in Brier-score units.
    """
    z_alpha = float(norm.ppf(1 - alpha / 2))
    z_beta = float(norm.ppf(power))
    mde = (z_alpha + z_beta) * sigma / np.sqrt(n_fights)

    return PowerResult(
        n_fights=n_fights,
        estimated_sigma=sigma,
        mde=float(mde),
        alpha=alpha,
        power=power,
    )


def paired_permutation_test(
    model_brier_per_fight: np.ndarray,
    market_brier_per_fight: np.ndarray,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> PermutationResult:
    """Two-sided paired permutation test on per-fight Brier score differences.

    Tests whether the observed mean difference between model and market Brier
    scores is distinguishable from what random label-swapping would produce.
    A significant result means the model's accuracy advantage (or disadvantage)
    is unlikely due to chance alone.

    The test randomly flips the sign of each paired difference (equivalent to
    swapping model/market labels for that fight) and computes the mean of the
    permuted differences. The p-value is the fraction of permutation means at
    least as extreme as the observed mean (two-sided).
    """
    model_brier_per_fight = np.asarray(model_brier_per_fight, dtype=np.float64)
    market_brier_per_fight = np.asarray(market_brier_per_fight, dtype=np.float64)

    differences = model_brier_per_fight - market_brier_per_fight
    observed_diff = float(np.mean(differences))
    n = len(differences)

    rng = np.random.default_rng(seed)

    # Generate all sign-flip vectors at once: +1 or -1 for each fight per permutation
    signs = rng.choice([-1, 1], size=(n_permutations, n))
    permuted_means = np.mean(signs * differences, axis=1)

    # Two-sided p-value: fraction of permutation means as or more extreme
    p_value = float(np.mean(np.abs(permuted_means) >= np.abs(observed_diff)))

    # Bootstrap CI on the observed difference using the permutation null
    ci_lower = float(np.percentile(permuted_means, 2.5))
    ci_upper = float(np.percentile(permuted_means, 97.5))

    return PermutationResult(
        observed_diff=observed_diff,
        p_value=p_value,
        n_permutations=n_permutations,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )
