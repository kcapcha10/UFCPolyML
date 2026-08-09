"""Tests for the XGBoost trainer module.

Verifies the fixed-round training invariant, absence of early stopping,
candidate selection by calibrated Brier with tiebreak logic, candidate
count guard bounds, and provenance recording (seed, version, elapsed time).
"""

from __future__ import annotations

import numpy as np
import pytest
import xgboost as xgb

from ufc_edge.model.schemas import CandidateConfig
from ufc_edge.model.train import (
    CandidateCountError,
    FoldCandidateResult,
    select_best_candidate,
    train_candidate,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_binary_data(
    n_rows: int = 60, n_features: int = 8, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate small synthetic binary classification data for fast tests.

    Returns features and labels with plausible shape for XGBoost training.
    Labels are balanced binary; features are standard normal.
    """
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n_rows, n_features)).astype(np.float64)
    labels = (rng.random(n_rows) > 0.5).astype(np.float64)
    return features, labels


@pytest.fixture
def small_data() -> tuple[np.ndarray, np.ndarray]:
    """60-row synthetic dataset for unit-level trainer tests."""
    return _synthetic_binary_data(n_rows=60, n_features=8, seed=42)


@pytest.fixture
def default_config() -> CandidateConfig:
    """A minimal valid CandidateConfig for testing."""
    return CandidateConfig(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
    )


def _make_config(n_est: int) -> CandidateConfig:
    """Helper to create distinct configs by n_estimators."""
    return CandidateConfig(
        n_estimators=n_est,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
    )


def _make_candidate_set(count: int) -> list[CandidateConfig]:
    """Create a list of distinct CandidateConfigs of the given size."""
    return [_make_config(100 + i) for i in range(count)]


def _fold_result(
    config: CandidateConfig,
    fold_id: int,
    brier: float,
    logloss: float,
    ece: float,
) -> FoldCandidateResult:
    """Shorthand factory for FoldCandidateResult with readable call sites."""
    return FoldCandidateResult(
        candidate_config=config,
        fold_id=fold_id,
        calibrated_brier=brier,
        log_loss=logloss,
        ece=ece,
    )


# ---------------------------------------------------------------------------
# Fixed-round invariant
# ---------------------------------------------------------------------------


class TestFixedRoundInvariant:
    """The trained model has exactly config.n_estimators trees."""

    def test_model_has_exact_tree_count(
        self,
        small_data: tuple[np.ndarray, np.ndarray],
        default_config: CandidateConfig,
    ) -> None:
        """Booster tree count matches n_estimators exactly."""
        features, labels = small_data
        booster, _ = train_candidate(
            features, labels, config=default_config, seed=42
        )
        assert booster.num_boosted_rounds() == default_config.n_estimators

    def test_varied_n_estimators_produces_exact_count(
        self,
        small_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Different n_estimators values all produce exact tree counts."""
        features, labels = small_data
        for n_est in (10, 25, 75):
            config = CandidateConfig(
                n_estimators=n_est,
                learning_rate=0.1,
                max_depth=3,
                min_child_weight=1.0,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
            )
            booster, _ = train_candidate(
                features, labels, config=config, seed=99
            )
            assert booster.num_boosted_rounds() == n_est


# ---------------------------------------------------------------------------
# No early stopping
# ---------------------------------------------------------------------------


class TestNoEarlyStopping:
    """Early stopping must never be passed to XGBoost."""

    def test_no_early_stopping_rounds_in_xgb_train(
        self,
        small_data: tuple[np.ndarray, np.ndarray],
        default_config: CandidateConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The xgboost.train call must not receive early_stopping_rounds."""
        features, labels = small_data
        captured_kwargs: dict = {}

        original_train = xgb.train

        def spy_train(*args, **kwargs):  # noqa: ANN002, ANN003
            captured_kwargs.update(kwargs)
            return original_train(*args, **kwargs)

        monkeypatch.setattr(
            "ufc_edge.model.train.xgb.train", spy_train
        )

        train_candidate(
            features, labels, config=default_config, seed=42
        )

        assert "early_stopping_rounds" not in captured_kwargs

    def test_no_early_stopping_in_params_dict(
        self,
        small_data: tuple[np.ndarray, np.ndarray],
        default_config: CandidateConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The params dict passed to xgb.train has no early_stopping_rounds."""
        features, labels = small_data
        captured_params: dict = {}

        original_train = xgb.train

        def spy_train(params, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            captured_params.update(params)
            return original_train(params, *args, **kwargs)

        monkeypatch.setattr(
            "ufc_edge.model.train.xgb.train", spy_train
        )

        train_candidate(
            features, labels, config=default_config, seed=42
        )

        assert "early_stopping_rounds" not in captured_params


# ---------------------------------------------------------------------------
# Candidate selection: lowest mean calibrated Brier
# ---------------------------------------------------------------------------


class TestSelectBestCandidate:
    """Selection picks the candidate with lowest mean calibrated Brier."""

    def test_picks_lowest_mean_calibrated_brier(self) -> None:
        """The candidate with the best (lowest) mean Brier wins."""
        candidates = _make_candidate_set(8)
        config_a = candidates[0]
        config_b = candidates[1]

        # config_a: mean Brier = (0.20 + 0.22) / 2 = 0.21
        # config_b: mean Brier = (0.18 + 0.19) / 2 = 0.185 (winner)
        results = [
            _fold_result(config_a, 0, 0.20, 0.5, 0.02),
            _fold_result(config_a, 1, 0.22, 0.5, 0.02),
            _fold_result(config_b, 0, 0.18, 0.5, 0.02),
            _fold_result(config_b, 1, 0.19, 0.5, 0.02),
        ]

        best = select_best_candidate(results, candidates)
        assert best == config_b

    def test_clear_winner_among_three(self) -> None:
        """Among three candidates, lowest mean Brier is selected."""
        candidates = _make_candidate_set(8)
        config_a = candidates[0]
        config_b = candidates[1]
        config_c = candidates[2]

        results = [
            _fold_result(config_a, 0, 0.25, 0.6, 0.03),
            _fold_result(config_a, 1, 0.24, 0.6, 0.03),
            _fold_result(config_b, 0, 0.22, 0.55, 0.025),
            _fold_result(config_b, 1, 0.21, 0.55, 0.025),
            _fold_result(config_c, 0, 0.19, 0.50, 0.02),
            _fold_result(config_c, 1, 0.20, 0.50, 0.02),
        ]

        best = select_best_candidate(results, candidates)
        assert best == config_c


# ---------------------------------------------------------------------------
# Tiebreak: Brier tied -> lower log loss wins
# ---------------------------------------------------------------------------


class TestTiebreakLogic:
    """When calibrated Brier is tied, lower log loss breaks the tie."""

    def test_tied_brier_selects_lower_log_loss(self) -> None:
        """Two candidates with identical mean Brier; lower log loss wins."""
        candidates = _make_candidate_set(8)
        config_a = candidates[0]
        config_b = candidates[1]

        # Both mean Brier = 0.20.
        # config_a mean log_loss = 0.59, config_b = 0.51.
        results = [
            _fold_result(config_a, 0, 0.20, 0.60, 0.02),
            _fold_result(config_a, 1, 0.20, 0.58, 0.02),
            _fold_result(config_b, 0, 0.20, 0.50, 0.02),
            _fold_result(config_b, 1, 0.20, 0.52, 0.02),
        ]

        best = select_best_candidate(results, candidates)
        assert best == config_b

    def test_three_way_brier_tie_resolved_by_log_loss(self) -> None:
        """Three candidates with identical mean Brier; log loss resolves."""
        candidates = _make_candidate_set(8)
        config_a = candidates[0]
        config_b = candidates[1]
        config_c = candidates[2]

        # All mean Brier = 0.22.
        # Mean log_loss: a=0.64, b=0.56, c=0.59 -> b wins.
        results = [
            _fold_result(config_a, 0, 0.22, 0.65, 0.03),
            _fold_result(config_a, 1, 0.22, 0.63, 0.03),
            _fold_result(config_b, 0, 0.22, 0.55, 0.025),
            _fold_result(config_b, 1, 0.22, 0.57, 0.025),
            _fold_result(config_c, 0, 0.22, 0.58, 0.02),
            _fold_result(config_c, 1, 0.22, 0.60, 0.02),
        ]

        best = select_best_candidate(results, candidates)
        assert best == config_b


# ---------------------------------------------------------------------------
# Candidate count guard (rejects <8 or >20)
# ---------------------------------------------------------------------------


class TestCandidateCountGuard:
    """select_best_candidate rejects candidate lists outside [8, 20]."""

    def test_rejects_fewer_than_eight_candidates(self) -> None:
        """Raises CandidateCountError when fewer than 8 candidates."""
        too_few = _make_candidate_set(7)
        results = [
            _fold_result(c, 0, 0.2, 0.5, 0.02) for c in too_few
        ]

        with pytest.raises(CandidateCountError):
            select_best_candidate(results, too_few)

    def test_rejects_more_than_twenty_candidates(self) -> None:
        """Raises CandidateCountError when more than 20 candidates."""
        too_many = _make_candidate_set(21)
        results = [
            _fold_result(c, 0, 0.2, 0.5, 0.02) for c in too_many
        ]

        with pytest.raises(CandidateCountError):
            select_best_candidate(results, too_many)

    def test_accepts_exactly_eight_candidates(self) -> None:
        """Eight candidates is the minimum allowed count."""
        configs = _make_candidate_set(8)
        results = [
            _fold_result(c, 0, 0.2, 0.5, 0.02) for c in configs
        ]

        best = select_best_candidate(results, configs)
        assert best in configs

    def test_accepts_exactly_twenty_candidates(self) -> None:
        """Twenty candidates is the maximum allowed count."""
        configs = _make_candidate_set(20)
        results = [
            _fold_result(c, 0, 0.2, 0.5, 0.02) for c in configs
        ]

        best = select_best_candidate(results, configs)
        assert best in configs

    def test_accepts_twelve_candidates(self) -> None:
        """Twelve candidates (the default config count) is accepted."""
        configs = _make_candidate_set(12)
        results = [
            _fold_result(c, 0, 0.2, 0.5, 0.02) for c in configs
        ]

        best = select_best_candidate(results, configs)
        assert best in configs


# ---------------------------------------------------------------------------
# Provenance: seed, version, elapsed time recorded per run
# ---------------------------------------------------------------------------


class TestProvenanceRecording:
    """Training must record seed and elapsed time in TrainResult."""

    def test_train_result_contains_seed(
        self,
        small_data: tuple[np.ndarray, np.ndarray],
        default_config: CandidateConfig,
    ) -> None:
        """The TrainResult captures the random seed used for training."""
        features, labels = small_data
        _, result = train_candidate(
            features, labels, config=default_config, seed=123
        )
        assert result.random_seed == 123

    def test_train_result_contains_different_seed(
        self,
        small_data: tuple[np.ndarray, np.ndarray],
        default_config: CandidateConfig,
    ) -> None:
        """Distinct seeds produce distinct recorded values."""
        features, labels = small_data
        _, result_a = train_candidate(
            features, labels, config=default_config, seed=42
        )
        _, result_b = train_candidate(
            features, labels, config=default_config, seed=99
        )
        assert result_a.random_seed == 42
        assert result_b.random_seed == 99

    def test_train_result_records_elapsed_time(
        self,
        small_data: tuple[np.ndarray, np.ndarray],
        default_config: CandidateConfig,
    ) -> None:
        """Elapsed time is a non-negative float."""
        features, labels = small_data
        _, result = train_candidate(
            features, labels, config=default_config, seed=42
        )
        assert result.elapsed_seconds >= 0.0

    def test_train_result_records_candidate_config(
        self,
        small_data: tuple[np.ndarray, np.ndarray],
        default_config: CandidateConfig,
    ) -> None:
        """The candidate config is preserved in the result."""
        features, labels = small_data
        _, result = train_candidate(
            features, labels, config=default_config, seed=42
        )
        assert result.candidate_config == default_config

    def test_elapsed_time_is_positive_for_real_training(
        self,
        small_data: tuple[np.ndarray, np.ndarray],
        default_config: CandidateConfig,
    ) -> None:
        """Training a real model takes measurable wall-clock time."""
        features, labels = small_data
        _, result = train_candidate(
            features, labels, config=default_config, seed=42
        )
        # Even a tiny model on 60 rows should take >0 seconds.
        assert result.elapsed_seconds > 0.0
