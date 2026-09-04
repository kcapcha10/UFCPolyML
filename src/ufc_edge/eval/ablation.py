"""Feature-family ablation ladder evaluation.

Trains a sequence of models that each add one more family of features, to
measure whether adding domain-specific features actually improves predictions
beyond generic stats, rather than assuming they do. Each successive rung is a
strict superset of the prior rung's columns, so the difference in Brier score
between adjacent rungs isolates the contribution of that family.

Rungs (cumulative):
  naive              → no features (constant 0.5 prediction, Brier = 0.25)
  record             → win/loss record features only
  physical           → + physical profile and activity
  schedule_strength  → + graph-derived (Elo, Glicko-2, PageRank)
  domain_interactions → + finishing, output, matchup interactions

All rungs train on identical folds for fair comparison. The naive rung skips
training entirely and produces constant 0.5, establishing the random-chance
baseline. ΔBrier between successive rungs is reported with event-bootstrap
confidence intervals; a rung is flagged non-significant when its CI spans zero.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import xgboost as xgb

from ufc_edge.eval.calibration import fit_calibrators, select_calibrator
from ufc_edge.eval.metrics import FightPrediction, brier_score, event_bootstrap_ci
from ufc_edge.eval.schemas import AblationRungResult, Fold
from ufc_edge.model.matrix import _columns_for_rung
from ufc_edge.model.schemas import AblationRung, CandidateConfig
from ufc_edge.model.train import train_candidate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ordered progression — each rung builds on the prior.
RUNG_ORDER: list[AblationRung] = [
    AblationRung.naive,
    AblationRung.record,
    AblationRung.physical,
    AblationRung.schedule_strength,
    AblationRung.domain_interactions,
]

_DEFAULT_BOOTSTRAP_N = 2000
_DEFAULT_BOOTSTRAP_ALPHA = 0.05
_DEFAULT_SEED = 42
_MIN_CALIBRATION_SIZE = 30


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_ablation_ladder(
    folds: list[Fold],
    candidates: list[CandidateConfig],
    feature_columns: list[str],
    X: np.ndarray,  # noqa: N803
    y: np.ndarray,
    event_ids_per_row: list[str],
    *,
    seed: int = _DEFAULT_SEED,
    bootstrap_n: int = _DEFAULT_BOOTSTRAP_N,
    bootstrap_alpha: float = _DEFAULT_BOOTSTRAP_ALPHA,
) -> list[AblationRungResult]:
    """Run the ablation ladder across all rungs on identical folds.

    For each rung, trains the candidate set on each fold's training partition,
    calibrates on the calibration partition, and evaluates on the test partition.
    The best candidate per rung is selected by mean Brier across folds. The
    naive rung produces constant 0.5 without training.

    Params:
        folds: Temporal folds with disjoint train/calibration/test event sets.
        candidates: Hyperparameter configs to evaluate at each rung.
        feature_columns: All available feature column names (full domain-interactions set).
        X: Full feature matrix (n_samples, n_features) matching feature_columns order.
        y: Binary labels (n_samples,).
        event_ids_per_row: Event identifier for each row (for bootstrap grouping).
        seed: Random seed for training reproducibility.
        bootstrap_n: Number of bootstrap replicates for confidence intervals.
        bootstrap_alpha: Significance level for confidence intervals.

    Returns:
        List of AblationRungResult in rung order (naive first).
    """
    # Build a row-index lookup by event_id for partition slicing.
    event_to_rows = _build_event_row_index(event_ids_per_row)

    results: list[AblationRungResult] = []
    prev_brier: float | None = None

    for rung in RUNG_ORDER:
        rung_columns = _columns_for_rung(feature_columns, rung)

        # Collect test predictions across all folds for this rung.
        all_predictions = _evaluate_rung_across_folds(
            rung=rung,
            rung_columns=rung_columns,
            feature_columns=feature_columns,
            folds=folds,
            candidates=candidates,
            X=X,
            y=y,
            event_to_rows=event_to_rows,
            seed=seed,
        )

        # Compute Brier score and bootstrap CI.
        probs_arr = np.array([p.prob for p in all_predictions])
        labels_arr = np.array([p.label for p in all_predictions])
        rung_brier = brier_score(probs_arr, labels_arr)

        brier_ci = event_bootstrap_ci(
            all_predictions,
            brier_score,
            n_bootstrap=bootstrap_n,
            alpha=bootstrap_alpha,
            seed=seed,
        )

        # Compute delta vs. prior rung.
        delta_brier: float | None = None
        delta_ci: tuple[float, float] | None = None
        significant: bool | None = None

        if prev_brier is not None:
            # Negative delta means improvement (lower Brier is better).
            delta_brier = rung_brier - prev_brier
            delta_ci = _bootstrap_delta_ci(
                all_predictions,
                prev_brier,
                bootstrap_n=bootstrap_n,
                alpha=bootstrap_alpha,
                seed=seed,
            )
            # Significant if the entire CI is below zero (improvement) or
            # above zero (degradation) — i.e., it does not span zero.
            significant = not (_ci_spans_zero(delta_ci))

        results.append(
            AblationRungResult(
                rung=rung.value,
                brier=rung_brier,
                brier_ci=brier_ci,
                delta_brier=delta_brier,
                delta_ci=delta_ci,
                significant=significant,
            )
        )
        prev_brier = rung_brier

    return results


# ---------------------------------------------------------------------------
# Internal: fold-level evaluation
# ---------------------------------------------------------------------------


def _evaluate_rung_across_folds(
    rung: AblationRung,
    rung_columns: list[str],
    feature_columns: list[str],
    folds: list[Fold],
    candidates: list[CandidateConfig],
    X: np.ndarray,  # noqa: N803
    y: np.ndarray,
    event_to_rows: dict[str, list[int]],
    seed: int,
) -> list[FightPrediction]:
    """Train candidates at a rung on each fold, return pooled test predictions.

    For the naive rung, returns constant 0.5 predictions without training.
    For other rungs, trains each candidate on each fold and picks the best
    candidate by mean Brier, then collects calibrated predictions on test sets.
    """
    if rung == AblationRung.naive:
        return _naive_predictions(folds, y, event_to_rows)

    # Column indices for this rung within the full feature matrix.
    col_indices = [feature_columns.index(c) for c in rung_columns]

    # Train each candidate on each fold, pick best by mean test Brier.
    best_candidate = _select_best_for_rung(
        col_indices=col_indices,
        folds=folds,
        candidates=candidates,
        X=X,
        y=y,
        event_to_rows=event_to_rows,
        seed=seed,
    )

    # Collect calibrated test predictions for the winning candidate.
    all_predictions: list[FightPrediction] = []

    for fold in folds:
        train_rows = _rows_for_events(fold.train_event_ids, event_to_rows)
        cal_rows = _rows_for_events(fold.calibration_event_ids, event_to_rows)
        test_rows = _rows_for_events(fold.test_event_ids, event_to_rows)

        X_train = X[train_rows][:, col_indices]  # noqa: N806
        y_train = y[train_rows]

        booster, _ = train_candidate(X_train, y_train, best_candidate, seed)

        # Get raw predictions for calibration and test.
        X_cal = X[cal_rows][:, col_indices]  # noqa: N806
        y_cal = y[cal_rows]
        X_test = X[test_rows][:, col_indices]  # noqa: N806

        raw_cal = _predict_raw(booster, X_cal)
        raw_test = _predict_raw(booster, X_test)

        # Fit and select calibrator on calibration slice.
        calibrated_test = _calibrate_predictions(raw_cal, y_cal, raw_test)

        # Build FightPrediction objects for bootstrap grouping.
        for i, row_idx in enumerate(test_rows):
            all_predictions.append(
                FightPrediction(
                    event_id=event_to_rows_reverse(row_idx, event_to_rows),
                    prob=float(calibrated_test[i]),
                    label=int(y[row_idx]),
                )
            )

    return all_predictions


def _naive_predictions(
    folds: list[Fold],
    y: np.ndarray,
    event_to_rows: dict[str, list[int]],
) -> list[FightPrediction]:
    """Generate constant 0.5 predictions for all test rows across folds."""
    predictions: list[FightPrediction] = []
    for fold in folds:
        test_rows = _rows_for_events(fold.test_event_ids, event_to_rows)
        for row_idx in test_rows:
            predictions.append(
                FightPrediction(
                    event_id=event_to_rows_reverse(row_idx, event_to_rows),
                    prob=0.5,
                    label=int(y[row_idx]),
                )
            )
    return predictions


def _select_best_for_rung(
    col_indices: list[int],
    folds: list[Fold],
    candidates: list[CandidateConfig],
    X: np.ndarray,  # noqa: N803
    y: np.ndarray,
    event_to_rows: dict[str, list[int]],
    seed: int,
) -> CandidateConfig:
    """Select the best candidate by mean calibrated Brier across folds.

    Trains each candidate on each fold's training set, calibrates on the
    calibration set, and evaluates on the test set. Returns the candidate
    with the lowest mean Brier score across folds.
    """
    candidate_scores: dict[int, list[float]] = defaultdict(list)

    for fold in folds:
        train_rows = _rows_for_events(fold.train_event_ids, event_to_rows)
        cal_rows = _rows_for_events(fold.calibration_event_ids, event_to_rows)
        test_rows = _rows_for_events(fold.test_event_ids, event_to_rows)

        X_train = X[train_rows][:, col_indices]  # noqa: N806
        y_train = y[train_rows]
        X_cal = X[cal_rows][:, col_indices]  # noqa: N806
        y_cal = y[cal_rows]
        X_test = X[test_rows][:, col_indices]  # noqa: N806
        y_test = y[test_rows]

        for idx, config in enumerate(candidates):
            booster, _ = train_candidate(X_train, y_train, config, seed)
            raw_cal = _predict_raw(booster, X_cal)
            raw_test = _predict_raw(booster, X_test)

            calibrated_test = _calibrate_predictions(raw_cal, y_cal, raw_test)
            fold_brier = brier_score(calibrated_test, y_test)
            candidate_scores[idx].append(fold_brier)

    # Pick candidate with lowest mean Brier.
    best_idx = min(candidate_scores, key=lambda k: np.mean(candidate_scores[k]))
    return candidates[best_idx]


# ---------------------------------------------------------------------------
# Internal: prediction and calibration helpers
# ---------------------------------------------------------------------------


def _predict_raw(booster: xgb.Booster, X: np.ndarray) -> np.ndarray:  # noqa: N803
    """Get raw XGBoost probabilities for a feature matrix."""
    dmatrix = xgb.DMatrix(X, nthread=1)
    return booster.predict(dmatrix)


def _calibrate_predictions(
    raw_cal: np.ndarray,
    y_cal: np.ndarray,
    raw_test: np.ndarray,
) -> np.ndarray:
    """Fit calibrators on the calibration set and apply the best to test.

    Uses a relaxed minimum calibration size for ablation evaluation since
    synthetic test sets can be small.
    """
    n_cal = len(raw_cal)
    min_size = min(_MIN_CALIBRATION_SIZE, n_cal)

    calibrators = fit_calibrators(raw_cal, y_cal, min_calibration_size=min_size)
    _, best_calibrator = select_calibrator(calibrators, raw_cal, y_cal)
    return best_calibrator.transform(raw_test)


# ---------------------------------------------------------------------------
# Internal: bootstrap delta CI
# ---------------------------------------------------------------------------


def _bootstrap_delta_ci(
    predictions: list[FightPrediction],
    prev_brier: float,
    *,
    bootstrap_n: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    """Compute bootstrap CI on (current_rung_brier - prev_rung_brier).

    Resamples events with replacement and computes the Brier difference for
    each replicate. The CI is the percentile interval on these differences.
    """
    # Group predictions by event.
    events: dict[str, list[FightPrediction]] = defaultdict(list)
    for pred in predictions:
        events[pred.event_id].append(pred)

    event_ids = list(events.keys())
    n_events = len(event_ids)
    rng = np.random.default_rng(seed)

    deltas = np.empty(bootstrap_n, dtype=np.float64)

    for b in range(bootstrap_n):
        sampled_indices = rng.integers(0, n_events, size=n_events)
        probs_list: list[float] = []
        labels_list: list[int] = []
        for idx in sampled_indices:
            for pred in events[event_ids[idx]]:
                probs_list.append(pred.prob)
                labels_list.append(pred.label)

        probs_arr = np.array(probs_list, dtype=np.float64)
        labels_arr = np.array(labels_list, dtype=np.float64)
        boot_brier = brier_score(probs_arr, labels_arr)
        deltas[b] = boot_brier - prev_brier

    lower_pct = 100.0 * (alpha / 2)
    upper_pct = 100.0 * (1 - alpha / 2)
    return (float(np.percentile(deltas, lower_pct)), float(np.percentile(deltas, upper_pct)))


def _ci_spans_zero(ci: tuple[float, float]) -> bool:
    """True if the confidence interval includes zero (non-significant)."""
    return ci[0] <= 0.0 <= ci[1]


# ---------------------------------------------------------------------------
# Internal: index helpers
# ---------------------------------------------------------------------------


def _build_event_row_index(event_ids_per_row: list[str]) -> dict[str, list[int]]:
    """Map event_id → list of row indices in the feature matrix."""
    index: dict[str, list[int]] = defaultdict(list)
    for i, event_id in enumerate(event_ids_per_row):
        index[event_id].append(i)
    return index


def event_to_rows_reverse(row_idx: int, event_to_rows: dict[str, list[int]]) -> str:
    """Find the event_id for a given row index (reverse lookup)."""
    for event_id, rows in event_to_rows.items():
        if row_idx in rows:
            return event_id
    msg = f"Row index {row_idx} not found in event-to-rows mapping"
    raise ValueError(msg)


def _rows_for_events(
    event_ids: frozenset[str],
    event_to_rows: dict[str, list[int]],
) -> list[int]:
    """Collect all row indices belonging to the given event set."""
    rows: list[int] = []
    for event_id in sorted(event_ids):
        rows.extend(event_to_rows.get(event_id, []))
    return rows
