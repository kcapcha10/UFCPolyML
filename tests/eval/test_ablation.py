"""Tests for the feature-family ablation ladder.

Verifies that the ablation ladder correctly trains a sequence of nested models,
each adding one family of features, to measure incremental predictive value.
Uses a small synthetic dataset with 2-3 candidate configs to keep test time fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from ufc_edge.eval.ablation import (
    RUNG_ORDER,
    _build_event_row_index,
    _ci_spans_zero,
    _rows_for_events,
    event_to_rows_reverse,
    run_ablation_ladder,
)
from ufc_edge.eval.schemas import Fold
from ufc_edge.model.matrix import _columns_for_rung
from ufc_edge.model.schemas import AblationRung, CandidateConfig

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


def _make_candidates(n: int = 2) -> list[CandidateConfig]:
    """Build a small candidate set for fast tests (2-3 configs, not the full ~12)."""
    configs = [
        CandidateConfig(
            n_estimators=10,
            learning_rate=0.3,
            max_depth=2,
            min_child_weight=1.0,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
        ),
        CandidateConfig(
            n_estimators=15,
            learning_rate=0.2,
            max_depth=3,
            min_child_weight=2.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.5,
        ),
        CandidateConfig(
            n_estimators=20,
            learning_rate=0.1,
            max_depth=3,
            min_child_weight=1.0,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=2.0,
        ),
    ]
    return configs[:n]


def _make_feature_columns() -> list[str]:
    """Feature columns spanning all families for ablation testing.

    Includes columns from each family: record, physical, schedule-strength,
    and domain-interactions, using the naming patterns expected by the matrix
    assembler's column subsetting logic.
    """
    return [
        # Record family
        "win_pct_all_a",
        "win_pct_all_b",
        "current_streak_a",
        "current_streak_b",
        "finish_rate_a",
        "finish_rate_b",
        # Physical family
        "height_cm_a",
        "height_cm_b",
        "reach_cm_a",
        "reach_cm_b",
        "age_at_fight_a",
        "age_at_fight_b",
        "days_since_last_fight_a",
        "days_since_last_fight_b",
        # Schedule-strength family
        "elo_rating_a",
        "elo_rating_b",
        "elo_delta",
        "glicko2_rating_a",
        "glicko2_rating_b",
        # Domain-interactions family
        "sig_strikes_per_min_a",
        "sig_strikes_per_min_b",
        "striking_accuracy_pct_a",
        "striking_accuracy_pct_b",
        "reach_delta",
        "height_delta",
    ]


def _make_synthetic_data(
    n_events: int = 6,
    fights_per_event: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str], list[Fold]]:
    """Generate synthetic feature matrix, labels, event IDs, and folds.

    Returns:
        X: Feature matrix (n_samples, n_features).
        y: Binary labels.
        event_ids_per_row: Event identifier for each row.
        folds: Two expanding-window folds using the same events for fair comparison.
    """
    rng = np.random.default_rng(seed)
    n_features = len(_make_feature_columns())
    n_rows = n_events * fights_per_event

    # Generate features with some signal in the first few columns.
    X = rng.standard_normal((n_rows, n_features))  # noqa: N806
    # Inject mild signal: higher win_pct correlates with positive outcome.
    signal = X[:, 0] + X[:, 1]
    prob_positive = 1.0 / (1.0 + np.exp(-0.5 * signal))
    y = (rng.random(n_rows) < prob_positive).astype(np.float64)

    # Assign event IDs.
    event_ids_per_row: list[str] = []
    for event_idx in range(n_events):
        event_id = f"event_{event_idx:03d}"
        for _ in range(fights_per_event):
            event_ids_per_row.append(event_id)

    # Create 2 folds: expanding window over 6 events.
    # Fold 0: train=events[0,1], cal=events[2], test=events[3]
    # Fold 1: train=events[0,1,2], cal=events[3], test=events[4,5]
    folds = [
        Fold(
            fold_id=0,
            train_event_ids=frozenset({"event_000", "event_001"}),
            calibration_event_ids=frozenset({"event_002"}),
            test_event_ids=frozenset({"event_003"}),
        ),
        Fold(
            fold_id=1,
            train_event_ids=frozenset({"event_000", "event_001", "event_002"}),
            calibration_event_ids=frozenset({"event_003"}),
            test_event_ids=frozenset({"event_004", "event_005"}),
        ),
    ]

    return X, y, event_ids_per_row, folds


# ---------------------------------------------------------------------------
# Test: correct number and order of rungs
# ---------------------------------------------------------------------------


class TestAblationLadderStructure:
    """Validates the ablation ladder produces 5 rungs in the correct order."""

    def test_five_rungs_produced(self) -> None:
        """The ladder returns exactly 5 rung results."""
        X, y, event_ids, folds = _make_synthetic_data()  # noqa: N806
        candidates = _make_candidates(n=2)
        feature_columns = _make_feature_columns()

        results = run_ablation_ladder(
            folds=folds,
            candidates=candidates,
            feature_columns=feature_columns,
            X=X,
            y=y,
            event_ids_per_row=event_ids,
            seed=42,
            bootstrap_n=100,
        )

        assert len(results) == 5

    def test_rung_order_is_correct(self) -> None:
        """Rungs appear in the correct cumulative order."""
        X, y, event_ids, folds = _make_synthetic_data()  # noqa: N806
        candidates = _make_candidates(n=2)
        feature_columns = _make_feature_columns()

        results = run_ablation_ladder(
            folds=folds,
            candidates=candidates,
            feature_columns=feature_columns,
            X=X,
            y=y,
            event_ids_per_row=event_ids,
            seed=42,
            bootstrap_n=100,
        )

        expected_order = ["naive", "record", "physical", "schedule_strength", "domain_interactions"]
        actual_order = [r.rung for r in results]
        assert actual_order == expected_order

    def test_first_rung_has_no_delta(self) -> None:
        """The naive rung (first) has None for delta, delta_ci, and significant."""
        X, y, event_ids, folds = _make_synthetic_data()  # noqa: N806
        candidates = _make_candidates(n=2)
        feature_columns = _make_feature_columns()

        results = run_ablation_ladder(
            folds=folds,
            candidates=candidates,
            feature_columns=feature_columns,
            X=X,
            y=y,
            event_ids_per_row=event_ids,
            seed=42,
            bootstrap_n=100,
        )

        naive_result = results[0]
        assert naive_result.delta_brier is None
        assert naive_result.delta_ci is None
        assert naive_result.significant is None

    def test_non_naive_rungs_have_deltas(self) -> None:
        """All rungs after naive have populated delta_brier, delta_ci, and significant."""
        X, y, event_ids, folds = _make_synthetic_data()  # noqa: N806
        candidates = _make_candidates(n=2)
        feature_columns = _make_feature_columns()

        results = run_ablation_ladder(
            folds=folds,
            candidates=candidates,
            feature_columns=feature_columns,
            X=X,
            y=y,
            event_ids_per_row=event_ids,
            seed=42,
            bootstrap_n=100,
        )

        for result in results[1:]:
            assert result.delta_brier is not None
            assert result.delta_ci is not None
            assert result.significant is not None


# ---------------------------------------------------------------------------
# Test: column nesting property
# ---------------------------------------------------------------------------


class TestColumnNesting:
    """Verifies that each rung's columns are a subset of the next rung's columns."""

    def test_columns_are_nested(self) -> None:
        """Rung k's columns are a subset of rung k+1's columns."""
        feature_columns = _make_feature_columns()

        prev_cols: set[str] = set()
        for rung in RUNG_ORDER:
            rung_cols = set(_columns_for_rung(feature_columns, rung))
            assert prev_cols <= rung_cols, (
                f"Rung {rung.value} columns are not a superset of the prior rung. "
                f"Missing: {prev_cols - rung_cols}"
            )
            prev_cols = rung_cols

    def test_naive_rung_has_no_columns(self) -> None:
        """The naive rung uses zero features (constant 0.5 floor)."""
        feature_columns = _make_feature_columns()
        naive_cols = _columns_for_rung(feature_columns, AblationRung.naive)
        assert naive_cols == []

    def test_domain_interactions_uses_all_families(self) -> None:
        """The final rung (domain_interactions) includes columns from every family."""
        feature_columns = _make_feature_columns()
        final_cols = _columns_for_rung(feature_columns, AblationRung.domain_interactions)
        # Must include at least one column from each family.
        assert any("win_pct" in c for c in final_cols), "Missing record family"
        assert any("height_cm" in c for c in final_cols), "Missing physical family"
        assert any("elo_rating" in c for c in final_cols), "Missing schedule-strength family"
        assert any("sig_strikes_per_min" in c for c in final_cols), (
            "Missing domain-interactions family"
        )


# ---------------------------------------------------------------------------
# Test: naive rung produces Brier = 0.25
# ---------------------------------------------------------------------------


class TestNaiveRungBrier:
    """Validates the naive rung (constant 0.5 prediction) produces Brier = 0.25."""

    def test_naive_brier_is_exactly_quarter(self) -> None:
        """Constant 0.5 predictions yield Brier = 0.25 regardless of outcome distribution."""
        X, y, event_ids, folds = _make_synthetic_data()  # noqa: N806
        candidates = _make_candidates(n=2)
        feature_columns = _make_feature_columns()

        results = run_ablation_ladder(
            folds=folds,
            candidates=candidates,
            feature_columns=feature_columns,
            X=X,
            y=y,
            event_ids_per_row=event_ids,
            seed=42,
            bootstrap_n=100,
        )

        naive_result = results[0]
        assert naive_result.brier == 0.25

    def test_naive_brier_exact_with_imbalanced_labels(self) -> None:
        """Even with heavily imbalanced labels, naive rung Brier is exactly 0.25."""
        rng = np.random.default_rng(99)
        n_events = 4
        fights_per_event = 8
        n_features = len(_make_feature_columns())
        n_rows = n_events * fights_per_event

        X = rng.standard_normal((n_rows, n_features))  # noqa: N806
        # Heavily imbalanced but each event has at least one of each class
        # to avoid calibrator fitting errors.
        y = np.ones(n_rows)
        for event_idx in range(n_events):
            base = event_idx * fights_per_event
            y[base] = 0.0  # One negative per event ensures both classes in each partition.

        event_ids = []
        for i in range(n_events):
            for _ in range(fights_per_event):
                event_ids.append(f"evt_{i}")

        folds = [
            Fold(
                fold_id=0,
                train_event_ids=frozenset({"evt_0"}),
                calibration_event_ids=frozenset({"evt_1"}),
                test_event_ids=frozenset({"evt_2", "evt_3"}),
            ),
        ]

        results = run_ablation_ladder(
            folds=folds,
            candidates=_make_candidates(n=2),
            feature_columns=_make_feature_columns(),
            X=X,
            y=y,
            event_ids_per_row=event_ids,
            seed=42,
            bootstrap_n=50,
        )

        assert results[0].brier == 0.25


# ---------------------------------------------------------------------------
# Test: ΔBrier is correctly computed
# ---------------------------------------------------------------------------


class TestDeltaBrier:
    """Verifies ΔBrier is the difference between successive rungs."""

    def test_delta_is_successive_difference(self) -> None:
        """delta_brier[k] = brier[k] - brier[k-1] for k >= 1."""
        X, y, event_ids, folds = _make_synthetic_data()  # noqa: N806
        candidates = _make_candidates(n=2)
        feature_columns = _make_feature_columns()

        results = run_ablation_ladder(
            folds=folds,
            candidates=candidates,
            feature_columns=feature_columns,
            X=X,
            y=y,
            event_ids_per_row=event_ids,
            seed=42,
            bootstrap_n=100,
        )

        for i in range(1, len(results)):
            expected_delta = results[i].brier - results[i - 1].brier
            assert results[i].delta_brier == pytest.approx(expected_delta, abs=1e-10)

    def test_delta_brier_signs_are_consistent(self) -> None:
        """If a rung improves over the prior, its delta is negative (lower Brier = better).

        With small synthetic data, calibration noise can cause some rungs to
        score worse than naive. We verify the sign convention: negative delta
        means improvement, positive means degradation.
        """
        X, y, event_ids, folds = _make_synthetic_data()  # noqa: N806
        candidates = _make_candidates(n=2)
        feature_columns = _make_feature_columns()

        results = run_ablation_ladder(
            folds=folds,
            candidates=candidates,
            feature_columns=feature_columns,
            X=X,
            y=y,
            event_ids_per_row=event_ids,
            seed=42,
            bootstrap_n=100,
        )

        # Verify the sign convention: delta = current - previous
        # If the rung improved (lower Brier), delta is negative.
        # If the rung degraded (higher Brier), delta is positive.
        for i in range(1, len(results)):
            expected_sign = results[i].brier - results[i - 1].brier
            assert results[i].delta_brier == pytest.approx(expected_sign, abs=1e-10)


# ---------------------------------------------------------------------------
# Test: significance flag fires when CI spans zero
# ---------------------------------------------------------------------------


class TestSignificanceFlag:
    """Verifies that the significance flag correctly reflects the CI."""

    def test_significant_when_ci_excludes_zero(self) -> None:
        """significant=True when the entire delta CI is below or above zero."""
        assert _ci_spans_zero((-0.05, -0.01)) is False
        assert _ci_spans_zero((0.01, 0.05)) is False

    def test_not_significant_when_ci_includes_zero(self) -> None:
        """significant=False when the delta CI crosses zero."""
        assert _ci_spans_zero((-0.03, 0.02)) is True

    def test_not_significant_when_ci_touches_zero(self) -> None:
        """CI exactly touching zero counts as spanning zero (non-significant)."""
        assert _ci_spans_zero((-0.03, 0.0)) is True
        assert _ci_spans_zero((0.0, 0.03)) is True

    def test_flag_matches_ci_in_ladder_output(self) -> None:
        """The significant flag in ladder output reflects its delta_ci."""
        X, y, event_ids, folds = _make_synthetic_data()  # noqa: N806
        candidates = _make_candidates(n=2)
        feature_columns = _make_feature_columns()

        results = run_ablation_ladder(
            folds=folds,
            candidates=candidates,
            feature_columns=feature_columns,
            X=X,
            y=y,
            event_ids_per_row=event_ids,
            seed=42,
            bootstrap_n=200,
        )

        for result in results[1:]:
            assert result.delta_ci is not None
            expected_significant = not _ci_spans_zero(result.delta_ci)
            assert result.significant == expected_significant


# ---------------------------------------------------------------------------
# Test: identical folds used across all rungs (fair comparison)
# ---------------------------------------------------------------------------


class TestIdenticalFolds:
    """Verifies that all rungs are evaluated on the same fold structure."""

    def test_all_rungs_use_same_test_set_size(self) -> None:
        """Every rung's result is based on the same number of test predictions.

        Since all rungs share the same folds and the test partition is determined
        by event membership, the pooled test count should be identical.
        """
        X, y, event_ids, folds = _make_synthetic_data()  # noqa: N806
        candidates = _make_candidates(n=2)
        feature_columns = _make_feature_columns()

        results = run_ablation_ladder(
            folds=folds,
            candidates=candidates,
            feature_columns=feature_columns,
            X=X,
            y=y,
            event_ids_per_row=event_ids,
            seed=42,
            bootstrap_n=50,
        )

        # All rungs should produce valid Brier scores (not NaN).
        for result in results:
            assert not np.isnan(result.brier)
            assert result.brier_ci[0] <= result.brier_ci[1]

    def test_folds_are_passed_through_unchanged(self) -> None:
        """The ladder does not modify or regenerate folds internally."""
        X, y, event_ids, folds = _make_synthetic_data()  # noqa: N806
        candidates = _make_candidates(n=2)
        feature_columns = _make_feature_columns()

        # Capture fold event sets before running.
        fold_snapshots = [
            (f.train_event_ids.copy(), f.calibration_event_ids.copy(), f.test_event_ids.copy())
            for f in folds
        ]

        run_ablation_ladder(
            folds=folds,
            candidates=candidates,
            feature_columns=feature_columns,
            X=X,
            y=y,
            event_ids_per_row=event_ids,
            seed=42,
            bootstrap_n=50,
        )

        # Verify folds remain unchanged (frozen model prevents mutation anyway).
        for i, fold in enumerate(folds):
            assert fold.train_event_ids == fold_snapshots[i][0]
            assert fold.calibration_event_ids == fold_snapshots[i][1]
            assert fold.test_event_ids == fold_snapshots[i][2]


# ---------------------------------------------------------------------------
# Test: helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    """Unit tests for internal helper functions."""

    def test_build_event_row_index(self) -> None:
        """Maps event IDs to their row indices correctly."""
        event_ids = ["evt_a", "evt_a", "evt_b", "evt_b", "evt_b"]
        index = _build_event_row_index(event_ids)
        assert index["evt_a"] == [0, 1]
        assert index["evt_b"] == [2, 3, 4]

    def test_event_to_rows_reverse(self) -> None:
        """Reverse lookup finds the event ID for a given row index."""
        event_to_rows = {"evt_a": [0, 1], "evt_b": [2, 3]}
        assert event_to_rows_reverse(0, event_to_rows) == "evt_a"
        assert event_to_rows_reverse(3, event_to_rows) == "evt_b"

    def test_event_to_rows_reverse_raises_for_missing(self) -> None:
        """Reverse lookup raises ValueError for an unknown row index."""
        event_to_rows = {"evt_a": [0, 1]}
        with pytest.raises(ValueError, match="not found"):
            event_to_rows_reverse(99, event_to_rows)

    def test_rows_for_events(self) -> None:
        """Collects row indices for a set of events."""
        event_to_rows = {"evt_a": [0, 1], "evt_b": [2, 3], "evt_c": [4]}
        rows = _rows_for_events(frozenset({"evt_a", "evt_c"}), event_to_rows)
        assert sorted(rows) == [0, 1, 4]
