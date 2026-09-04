"""Tests for market-relative Brier skill evaluation.

Verifies that the model's skill relative to market-implied probabilities is
computed correctly: Brier_skill = 1 − (Brier_model / Brier_market). The market
is scored as if it were a predictor (using its own implied probabilities against
actual outcomes), so the model's accuracy can be compared against a real benchmark.
"""

from __future__ import annotations

import pytest

from ufc_edge.eval.market_relative import (
    BrierSkillResult,
    MarketFightRow,
    compute_brier_skill,
)

# ---------------------------------------------------------------------------
# Formula correctness
# ---------------------------------------------------------------------------


class TestBrierSkillFormula:
    """Verifies the Brier skill score formula with known numeric values."""

    def test_model_better_than_market_positive_skill(self) -> None:
        """Model Brier 0.2, market Brier 0.25 → skill = 0.2 exactly."""
        # Construct fights that produce exact Brier values:
        # 4 fights, all outcome=1.
        # model_prob: (1-p)^2 mean = 0.2  →  (1-p)^2 = 0.2  →  p = 1-√0.2
        # market_prob: (1-p)^2 mean = 0.25  →  p = 0.5
        #
        # Simpler: directly construct per-fight squared errors that average to
        # the desired Brier scores.
        # Fight 1: model_prob=0.9, outcome=1 → (0.1)^2 = 0.01
        # Fight 2: model_prob=0.5, outcome=1 → (0.5)^2 = 0.25
        # Fight 3: model_prob=0.7, outcome=1 → (0.3)^2 = 0.09
        # Fight 4: model_prob=0.7, outcome=0 → (0.7)^2 = 0.49
        # Mean model Brier = (0.01 + 0.25 + 0.09 + 0.49) / 4 = 0.21 ← not 0.2
        #
        # Use 2 fights:
        # Fight 1: model=0.8, outcome=1, market=0.5 →
        #   model_sq_err = 0.04, market_sq_err = 0.25
        # Fight 2: model=0.6, outcome=1, market=0.5 →
        #   model_sq_err = 0.16, market_sq_err = 0.25
        # Brier_model = (0.04 + 0.16) / 2 = 0.10
        # Brier_market = (0.25 + 0.25) / 2 = 0.25
        # Skill = 1 - 0.10/0.25 = 0.6
        #
        # For exactly model=0.2, market=0.25:
        # 5 fights all outcome=1:
        # model_probs yielding mean sq_err = 0.2: each (1-p)^2 = 0.2 → not clean
        #
        # Simplest: use the Brier values directly via 1 fight per known error.
        # Fight: model=0.6, outcome=0, market=0.5
        #   model_sq_err = (0 - 0.6)^2 = 0.36
        #   market_sq_err = (0 - 0.5)^2 = 0.25
        # That gives model worse. Flip:
        #
        # Just pick a set that gives model=0.2, market=0.25 exactly.
        # 4 fights, all outcome=1:
        #   model needs sum of (1-p_i)^2 = 0.8 across 4 fights.
        #   market needs sum of (1-q_i)^2 = 1.0 across 4 fights.
        #
        # market: all q=0.5 → each (0.5)^2 = 0.25 → sum = 1.0 ✓, mean = 0.25 ✓
        # model: pick p values so (1-p)^2 sum = 0.8
        #   e.g. two at 0.9: 2*(0.1)^2=0.02, two at p where 2*(1-p)^2 = 0.78
        #        (1-p)^2 = 0.39, p = 1 - √0.39 ≈ 0.375 ← messy
        #
        # Clean approach: 1 fight.
        # outcome=1, model_prob=sqrt(1-0.2)= ? No — Brier = (1-p)^2 for 1 fight.
        # (1-p)^2 = 0.2 → p = 1 - √0.2 ≈ 0.5528... not clean either.
        #
        # Use the injection approach: 2 fights that are easy to verify manually.
        # Fight 1: model=0.8, market=0.5, outcome=1
        #   model_err = (1-0.8)^2 = 0.04; market_err = (1-0.5)^2 = 0.25
        # Fight 2: model=0.4, market=0.5, outcome=1
        #   model_err = (1-0.4)^2 = 0.36; market_err = (1-0.5)^2 = 0.25
        # Brier_model = (0.04 + 0.36)/2 = 0.20
        # Brier_market = (0.25 + 0.25)/2 = 0.25
        # Skill = 1 - 0.20/0.25 = 1 - 0.8 = 0.20 ✓
        rows = [
            MarketFightRow(model_prob=0.8, market_prob=0.5, outcome=1),
            MarketFightRow(model_prob=0.4, market_prob=0.5, outcome=1),
        ]
        result = compute_brier_skill(rows)

        assert result.brier_model == pytest.approx(0.20)
        assert result.brier_market == pytest.approx(0.25)
        assert result.brier_skill == pytest.approx(0.20)

    def test_model_same_as_market_zero_skill(self) -> None:
        """When model and market have equal Brier scores, skill is exactly 0."""
        rows = [
            MarketFightRow(model_prob=0.7, market_prob=0.7, outcome=1),
            MarketFightRow(model_prob=0.3, market_prob=0.3, outcome=0),
        ]
        result = compute_brier_skill(rows)

        assert result.brier_model == result.brier_market
        assert result.brier_skill == pytest.approx(0.0)

    def test_model_worse_than_market_negative_skill(self) -> None:
        """Model worse than market yields negative Brier skill."""
        # Fight: model=0.3, market=0.8, outcome=1
        #   model_err = (1-0.3)^2 = 0.49; market_err = (1-0.8)^2 = 0.04
        # Skill = 1 - 0.49/0.04 = 1 - 12.25 = -11.25
        rows = [
            MarketFightRow(model_prob=0.3, market_prob=0.8, outcome=1),
        ]
        result = compute_brier_skill(rows)

        assert result.brier_skill < 0.0
        assert result.brier_skill == pytest.approx(1.0 - 0.49 / 0.04)


# ---------------------------------------------------------------------------
# Exclusion count
# ---------------------------------------------------------------------------


class TestExclusionCount:
    """Fights without a matched market probability are excluded and counted."""

    def test_none_market_prob_excluded(self) -> None:
        """Rows with market_prob=None are excluded; exclusion count is reported."""
        rows = [
            MarketFightRow(model_prob=0.8, market_prob=0.5, outcome=1),
            MarketFightRow(model_prob=0.6, market_prob=None, outcome=1),
            MarketFightRow(model_prob=0.7, market_prob=None, outcome=0),
            MarketFightRow(model_prob=0.4, market_prob=0.5, outcome=1),
        ]
        result = compute_brier_skill(rows)

        assert result.n_excluded == 2
        assert result.n_scored == 2

    def test_all_excluded_returns_none_skill(self) -> None:
        """When all rows lack market data, skill and Brier values are None."""
        rows = [
            MarketFightRow(model_prob=0.8, market_prob=None, outcome=1),
            MarketFightRow(model_prob=0.6, market_prob=None, outcome=0),
        ]
        result = compute_brier_skill(rows)

        assert result.brier_skill is None
        assert result.brier_model is None
        assert result.brier_market is None
        assert result.n_excluded == 2
        assert result.n_scored == 0

    def test_zero_exclusions_when_all_matched(self) -> None:
        """All rows matched → exclusion count is 0."""
        rows = [
            MarketFightRow(model_prob=0.9, market_prob=0.7, outcome=1),
            MarketFightRow(model_prob=0.2, market_prob=0.4, outcome=0),
        ]
        result = compute_brier_skill(rows)

        assert result.n_excluded == 0
        assert result.n_scored == 2


# ---------------------------------------------------------------------------
# Same denominator (identical fight set for model and market)
# ---------------------------------------------------------------------------


class TestSameDenominator:
    """Model and market are scored on the exact same subset of fights."""

    def test_scored_fights_identical_for_model_and_market(self) -> None:
        """Both Brier scores use only the fights where market data exists."""
        # 3 rows: 2 with market, 1 without
        rows = [
            MarketFightRow(model_prob=0.9, market_prob=0.6, outcome=1),
            MarketFightRow(model_prob=0.3, market_prob=None, outcome=0),
            MarketFightRow(model_prob=0.5, market_prob=0.5, outcome=0),
        ]
        result = compute_brier_skill(rows)

        # The model Brier should use only the 2 matched fights, not all 3
        # Fight 1: model_err = (1-0.9)^2 = 0.01
        # Fight 3: model_err = (0-0.5)^2 = 0.25
        # Brier_model = (0.01 + 0.25) / 2 = 0.13
        assert result.brier_model == pytest.approx(0.13)
        assert result.n_scored == 2

    def test_denominator_count_matches_n_scored(self) -> None:
        """n_scored + n_excluded equals total input length."""
        rows = [
            MarketFightRow(model_prob=0.7, market_prob=0.5, outcome=1),
            MarketFightRow(model_prob=0.4, market_prob=None, outcome=0),
            MarketFightRow(model_prob=0.6, market_prob=0.4, outcome=1),
            MarketFightRow(model_prob=0.2, market_prob=None, outcome=0),
            MarketFightRow(model_prob=0.8, market_prob=0.9, outcome=1),
        ]
        result = compute_brier_skill(rows)

        assert result.n_scored + result.n_excluded == len(rows)


# ---------------------------------------------------------------------------
# Result structure completeness
# ---------------------------------------------------------------------------


class TestResultStructure:
    """The result includes absolute Brier values and the relative skill score."""

    def test_result_contains_all_fields(self) -> None:
        """BrierSkillResult has brier_model, brier_market, brier_skill, and counts."""
        rows = [
            MarketFightRow(model_prob=0.7, market_prob=0.6, outcome=1),
        ]
        result = compute_brier_skill(rows)

        assert isinstance(result, BrierSkillResult)
        assert result.brier_model is not None
        assert result.brier_market is not None
        assert result.brier_skill is not None
        assert isinstance(result.n_scored, int)
        assert isinstance(result.n_excluded, int)

    def test_empty_input_returns_zero_counts(self) -> None:
        """Empty input list yields n_scored=0, n_excluded=0, skill=None."""
        result = compute_brier_skill([])

        assert result.n_scored == 0
        assert result.n_excluded == 0
        assert result.brier_skill is None
