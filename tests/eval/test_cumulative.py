"""Tests for cumulative evidence tracking.

Verifies that the running Brier-skill time series grows correctly, confidence
bands narrow with accumulating data, and incremental updates produce the same
result as a single-pass computation over all accumulated data.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from ufc_edge.eval.cumulative import (
    PeriodData,
    _compute_brier_skill,
    _event_bootstrap_brier_skill_ci,
    update_cumulative_evidence,
)
from ufc_edge.eval.schemas import CumulativePoint

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_period(
    event_id: str,
    as_of_date: date,
    n_fights: int,
    model_mean_brier: float = 0.20,
    market_mean_brier: float = 0.25,
    noise_scale: float = 0.02,
    seed: int = 0,
) -> PeriodData:
    """Generate a synthetic period with known Brier values plus small noise."""
    rng = np.random.default_rng(seed)
    model_brier = (
        (np.full(n_fights, model_mean_brier) + rng.normal(0, noise_scale, n_fights))
        .clip(0, 1)
        .tolist()
    )
    market_brier = (
        (np.full(n_fights, market_mean_brier) + rng.normal(0, noise_scale, n_fights))
        .clip(0, 1)
        .tolist()
    )
    return PeriodData(
        event_id=event_id,
        as_of_date=as_of_date,
        model_brier_per_fight=model_brier,
        market_brier_per_fight=market_brier,
    )


# ---------------------------------------------------------------------------
# _compute_brier_skill unit tests
# ---------------------------------------------------------------------------


class TestComputeBrierSkill:
    """Direct tests for the Brier skill calculation."""

    def test_perfect_model_scores_one(self) -> None:
        """A model with zero Brier loss achieves skill = 1.0."""
        model = np.array([0.0, 0.0, 0.0])
        market = np.array([0.25, 0.25, 0.25])
        assert _compute_brier_skill(model, market) == 1.0

    def test_equal_performance_scores_zero(self) -> None:
        """When model equals market, skill = 0.0."""
        brier_vals = np.array([0.2, 0.3, 0.15])
        assert _compute_brier_skill(brier_vals, brier_vals) == 0.0

    def test_worse_model_scores_negative(self) -> None:
        """A model worse than market has negative skill."""
        model = np.array([0.4, 0.5, 0.3])
        market = np.array([0.2, 0.2, 0.2])
        skill = _compute_brier_skill(model, market)
        assert skill < 0.0

    def test_known_value(self) -> None:
        """Verify exact computation: skill = 1 - (0.20 / 0.25) = 0.20."""
        model = np.array([0.20, 0.20, 0.20])
        market = np.array([0.25, 0.25, 0.25])
        assert _compute_brier_skill(model, market) == pytest.approx(0.20)

    def test_zero_market_brier_returns_zero(self) -> None:
        """Edge case: perfect market yields skill = 0 (avoid division by zero)."""
        model = np.array([0.1, 0.2])
        market = np.array([0.0, 0.0])
        assert _compute_brier_skill(model, market) == 0.0


# ---------------------------------------------------------------------------
# Series growth tests
# ---------------------------------------------------------------------------


class TestSeriesGrowth:
    """The series grows in length when new periods are added."""

    def test_single_period_produces_one_point(self) -> None:
        """One new period yields a series of length 1."""
        period = _make_period("event_1", date(2026, 3, 1), n_fights=10, seed=1)
        result = update_cumulative_evidence([], [period])
        assert len(result) == 1

    def test_multiple_periods_grow_series(self) -> None:
        """Adding 5 periods produces a 5-point series."""
        periods = [
            _make_period(f"event_{i}", date(2026, 3, i + 1), n_fights=8, seed=i) for i in range(5)
        ]
        result = update_cumulative_evidence([], periods)
        assert len(result) == 5

    def test_incremental_append_grows_by_one(self) -> None:
        """Adding one more period to an existing set grows by one."""
        periods_a = [
            _make_period(f"event_{i}", date(2026, 3, i + 1), n_fights=10, seed=i) for i in range(3)
        ]
        periods_b = periods_a + [_make_period("event_3", date(2026, 3, 4), n_fights=10, seed=3)]
        result_a = update_cumulative_evidence([], periods_a)
        result_b = update_cumulative_evidence([], periods_b)
        assert len(result_b) == len(result_a) + 1

    def test_empty_new_periods_returns_prior(self) -> None:
        """No new periods returns the prior series unchanged."""
        prior = [
            CumulativePoint(
                as_of_event_url="event_0",
                as_of_date=date(2026, 3, 1),
                cumulative_n_fights=10,
                brier_skill=0.15,
                ci_lower=0.05,
                ci_upper=0.25,
            )
        ]
        result = update_cumulative_evidence(prior, [])
        assert result == prior

    def test_cumulative_fight_count_increases(self) -> None:
        """Each point's cumulative_n_fights is the sum of all fights so far."""
        periods = [
            _make_period(f"event_{i}", date(2026, 3, i + 1), n_fights=5 + i, seed=i)
            for i in range(4)
        ]
        result = update_cumulative_evidence([], periods)
        expected_counts = [5, 11, 18, 26]  # 5, 5+6, 5+6+7, 5+6+7+8
        actual_counts = [p.cumulative_n_fights for p in result]
        assert actual_counts == expected_counts


# ---------------------------------------------------------------------------
# CI band narrowing tests
# ---------------------------------------------------------------------------


class TestCIBandNarrowing:
    """Confidence intervals should narrow as more data accumulates.

    Uses a synthetic sequence with consistent signal (model consistently
    better than market) so that variance in the estimate shrinks with N.
    """

    def test_ci_width_narrows_with_accumulating_events(self) -> None:
        """CI width at point N should be less than at point 1 for large N."""
        # Generate many periods with consistent model advantage
        periods = [
            _make_period(
                f"event_{i}",
                date(2026, 1, 1),
                n_fights=12,
                model_mean_brier=0.18,
                market_mean_brier=0.25,
                noise_scale=0.03,
                seed=i + 100,
            )
            for i in range(20)
        ]
        result = update_cumulative_evidence([], periods, n_bootstrap=2000, seed=99)

        # CI width at point 2 (after 2 events) vs point 19 (after 20 events)
        early_width = result[1].ci_upper - result[1].ci_lower
        late_width = result[-1].ci_upper - result[-1].ci_lower
        assert late_width < early_width

    def test_ci_width_monotonically_decreasing_trend(self) -> None:
        """Over a large accumulation, the general trend of CI width is down.

        We don't require strict monotonicity (bootstrap variance), but the
        last quarter should be tighter on average than the first quarter.
        Uses enough events and noise to make the narrowing reliably visible.
        """
        periods = [
            _make_period(
                f"event_{i}",
                date(2026, 1, 1),
                n_fights=12,
                model_mean_brier=0.18,
                market_mean_brier=0.25,
                noise_scale=0.05,
                seed=i + 200,
            )
            for i in range(24)
        ]
        result = update_cumulative_evidence([], periods, n_bootstrap=2000, seed=77)

        widths = [p.ci_upper - p.ci_lower for p in result]
        # Compare first 6 points (few events) vs last 6 points (many events)
        first_quarter_avg = np.mean(widths[:6])
        last_quarter_avg = np.mean(widths[18:])
        assert last_quarter_avg < first_quarter_avg


# ---------------------------------------------------------------------------
# Consistency with direct computation tests
# ---------------------------------------------------------------------------


class TestIncrementalConsistency:
    """Running the update incrementally must match a one-shot computation.

    The running statistic at any point k should equal computing Brier skill
    directly on all accumulated data from periods[0:k+1].
    """

    def test_incremental_matches_one_shot(self) -> None:
        """Incremental updates produce the same Brier skill as single pass."""
        periods = [
            _make_period(f"event_{i}", date(2026, 4, i + 1), n_fights=8, seed=i + 50)
            for i in range(6)
        ]

        # Incremental: add one at a time, keep only the last point
        incremental_skills: list[float] = []
        for k in range(1, len(periods) + 1):
            result = update_cumulative_evidence([], periods[:k], seed=42)
            incremental_skills.append(result[-1].brier_skill)

        # One-shot: compute for all prefixes at once
        one_shot = update_cumulative_evidence([], periods, seed=42)
        one_shot_skills = [p.brier_skill for p in one_shot]

        for inc_val, one_val in zip(incremental_skills, one_shot_skills, strict=True):
            assert inc_val == pytest.approx(one_val, abs=1e-12)

    def test_incremental_ci_matches_one_shot(self) -> None:
        """Incremental CI bounds match single-pass CI bounds."""
        periods = [
            _make_period(f"event_{i}", date(2026, 5, i + 1), n_fights=10, seed=i + 70)
            for i in range(4)
        ]

        # Compute incrementally
        incremental_results: list[CumulativePoint] = []
        for k in range(1, len(periods) + 1):
            result = update_cumulative_evidence([], periods[:k], n_bootstrap=1000, seed=42)
            incremental_results.append(result[-1])

        # Compute all at once
        one_shot = update_cumulative_evidence([], periods, n_bootstrap=1000, seed=42)

        for inc_pt, one_pt in zip(incremental_results, one_shot, strict=True):
            assert inc_pt.brier_skill == pytest.approx(one_pt.brier_skill, abs=1e-12)
            assert inc_pt.ci_lower == pytest.approx(one_pt.ci_lower, abs=1e-12)
            assert inc_pt.ci_upper == pytest.approx(one_pt.ci_upper, abs=1e-12)

    def test_direct_brier_skill_matches_cumulative_point(self) -> None:
        """The Brier skill in the final point equals direct computation."""
        periods = [
            _make_period(f"event_{i}", date(2026, 6, i + 1), n_fights=15, seed=i + 90)
            for i in range(5)
        ]
        result = update_cumulative_evidence([], periods, seed=42)

        # Direct computation over all fight data
        all_model = np.concatenate([np.array(p.model_brier_per_fight) for p in periods])
        all_market = np.concatenate([np.array(p.market_brier_per_fight) for p in periods])
        direct_skill = _compute_brier_skill(all_model, all_market)

        assert result[-1].brier_skill == pytest.approx(direct_skill, abs=1e-12)


# ---------------------------------------------------------------------------
# Edge cases and data integrity
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for robustness."""

    def test_single_fight_per_period(self) -> None:
        """Handles periods with just one fight each."""
        periods = [
            PeriodData(
                event_id=f"event_{i}",
                as_of_date=date(2026, 7, i + 1),
                model_brier_per_fight=[0.2],
                market_brier_per_fight=[0.25],
            )
            for i in range(3)
        ]
        result = update_cumulative_evidence([], periods, n_bootstrap=500, seed=10)
        assert len(result) == 3
        # All periods have same brier values, so skill should be consistent
        for pt in result:
            assert pt.brier_skill == pytest.approx(0.2, abs=1e-10)

    def test_dates_preserved_in_output(self) -> None:
        """Output points preserve the as_of_date from input periods."""
        dates = [date(2026, 1, 10), date(2026, 2, 15), date(2026, 3, 20)]
        periods = [_make_period(f"event_{i}", dates[i], n_fights=5, seed=i) for i in range(3)]
        result = update_cumulative_evidence([], periods)
        assert [p.as_of_date for p in result] == dates

    def test_event_urls_preserved_in_output(self) -> None:
        """Output points preserve the event_id as as_of_event_url."""
        event_ids = ["event_alpha", "event_beta", "event_gamma"]
        periods = [
            PeriodData(
                event_id=eid,
                as_of_date=date(2026, 4, i + 1),
                model_brier_per_fight=[0.2, 0.2],
                market_brier_per_fight=[0.25, 0.25],
            )
            for i, eid in enumerate(event_ids)
        ]
        result = update_cumulative_evidence([], periods, n_bootstrap=100, seed=1)
        assert [p.as_of_event_url for p in result] == event_ids

    def test_ci_contains_point_estimate(self) -> None:
        """The point estimate should generally fall within the CI bounds."""
        periods = [
            _make_period(f"event_{i}", date(2026, 8, i + 1), n_fights=20, seed=i + 300)
            for i in range(10)
        ]
        result = update_cumulative_evidence([], periods, n_bootstrap=3000, seed=55)
        # Check last few points (more data → more stable)
        for pt in result[5:]:
            assert pt.ci_lower <= pt.brier_skill <= pt.ci_upper


# ---------------------------------------------------------------------------
# Bootstrap CI direct tests
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    """Direct tests for the event-level bootstrap CI function."""

    def test_ci_bounds_ordered(self) -> None:
        """Lower bound is always <= upper bound."""
        events = {
            f"event_{i}": (
                np.array([0.2, 0.15, 0.22]),
                np.array([0.25, 0.28, 0.24]),
            )
            for i in range(5)
        }
        lower, upper = _event_bootstrap_brier_skill_ci(events, n_bootstrap=1000)
        assert lower <= upper

    def test_ci_narrows_with_more_events(self) -> None:
        """More events → tighter CI (law of large numbers)."""
        rng = np.random.default_rng(42)
        small_events = {
            f"event_{i}": (
                rng.uniform(0.15, 0.22, size=5),
                rng.uniform(0.23, 0.28, size=5),
            )
            for i in range(3)
        }
        large_events = {
            f"event_{i}": (
                rng.uniform(0.15, 0.22, size=5),
                rng.uniform(0.23, 0.28, size=5),
            )
            for i in range(30)
        }

        lo_small, hi_small = _event_bootstrap_brier_skill_ci(
            small_events, n_bootstrap=2000, seed=10
        )
        lo_large, hi_large = _event_bootstrap_brier_skill_ci(
            large_events, n_bootstrap=2000, seed=10
        )

        width_small = hi_small - lo_small
        width_large = hi_large - lo_large
        assert width_large < width_small
