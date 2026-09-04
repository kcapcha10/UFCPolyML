"""End-to-end model + evaluation pipeline test on synthetic fixture features.

Runs the full development pipeline against a synthetic ``features_v1`` DuckDB:
matrix assembly → expanding-window fold generation → per-fold XGBoost training →
per-fold calibrator selection → pooled evaluation (Brier, log loss, ECE, reliability)
→ ablation ladder → MLflow provenance logging. It then asserts the cross-cutting
correctness properties: determinism under a fixed seed, no temporal leakage in any
fold, exact order symmetry of predictions, absence of market columns, and that a
complete provenance run with every expected artifact is logged.

Everything runs against a local temporary sqlite MLflow store — never a live server.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import duckdb
import mlflow
import numpy as np
import pytest
import xgboost as xgb
from mlflow.tracking import MlflowClient

from tests.fixtures.fixture_features import generate_feature_table
from ufc_edge.eval.ablation import run_ablation_ladder
from ufc_edge.eval.calibration import fit_calibrators, select_calibrator
from ufc_edge.eval.metrics import (
    FightPrediction,
    brier_score,
    event_bootstrap_ci,
    expected_calibration_error,
    log_loss_score,
)
from ufc_edge.eval.power import minimum_detectable_effect
from ufc_edge.eval.provenance import log_training_run
from ufc_edge.eval.reliability import stratified_reliability
from ufc_edge.eval.schemas import EvaluationReport, FoldMetrics
from ufc_edge.eval.splits import EventEntry, generate_folds
from ufc_edge.model.inference import SymmetricPrediction, predict_symmetric
from ufc_edge.model.matrix import MARKET_COLUMNS, assemble_matrix
from ufc_edge.model.schemas import CandidateConfig
from ufc_edge.model.train import FoldCandidateResult, train_candidate

_FEATURE_VERSION = "v1"
_FEATURES_TABLE = "features_v1"
_SEED = 7
_BOOTSTRAP_N = 200
_ABLATION_BOOTSTRAP_N = 100

# Tiny fold thresholds: the 50-fight fixture cannot meet production minimums.
_FOLD_KWARGS = {
    "n_folds": 2,
    "min_train_fights": 2,
    "min_test_fights": 2,
    "calibration_ratio": 0.2,
    "calibration_min": 2,
}

# Small locked candidates keep the end-to-end run fast while exercising the ladder.
_LOCKED_CONFIG = CandidateConfig(
    n_estimators=10,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=1.0,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.0,
    reg_lambda=1.0,
)
_SECOND_CONFIG = CandidateConfig(
    n_estimators=6,
    learning_rate=0.2,
    max_depth=2,
    min_child_weight=1.0,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.0,
    reg_lambda=1.0,
)


# ---------------------------------------------------------------------------
# Pipeline result bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PipelineResult:
    """Everything the pipeline produced, captured once for all assertions."""

    manifest_columns: list[str]
    manifest_n_rows: int
    manifest_exclusions: dict[str, int]
    event_index: list[EventEntry]
    folds: list
    report: EvaluationReport
    reliability: list
    ablation: list
    run_id: str
    client: MlflowClient
    booster: xgb.Booster
    calibrator: object
    calibrator_method: str
    features_matrix: np.ndarray
    pooled_probs_first: np.ndarray
    pooled_probs_second: np.ndarray
    pooled_brier_first: float
    pooled_brier_second: float


# ---------------------------------------------------------------------------
# Pipeline construction helpers
# ---------------------------------------------------------------------------


def _event_index(conn: duckdb.DuckDBPyConnection) -> list[EventEntry]:
    """Build the temporal event index from valid (decision) fights only."""
    rows = conn.execute(
        """
        SELECT event_url, event_date, COUNT(*) AS n_fights
        FROM features_v1
        WHERE outcome IN ('win_a', 'win_b')
        GROUP BY event_url, event_date
        ORDER BY event_date, event_url
        """
    ).fetchall()
    return [EventEntry(event_url=row[0], event_date=row[1], n_fights=int(row[2])) for row in rows]


def _assemble_by_event(
    conn: duckdb.DuckDBPyConnection,
    event_index: list[EventEntry],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Assemble one matrix per event and concatenate, tracking each row's event.

    Per-event assembly gives an exact row → event mapping (the assembler itself
    does not emit one), which the fold slicing and event bootstrap both require.
    """
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    event_ids_per_row: list[str] = []
    columns: list[str] = []

    for entry in event_index:
        result = assemble_matrix(
            _FEATURES_TABLE,
            _FEATURE_VERSION,
            conn,
            event_ids=frozenset({entry.event_url}),
        )
        if result.X.shape[0] == 0:
            continue
        x_parts.append(result.X)
        y_parts.append(result.y)
        event_ids_per_row.extend([entry.event_url] * result.X.shape[0])
        columns = result.manifest.columns

    return np.vstack(x_parts), np.concatenate(y_parts), event_ids_per_row, columns


def _rows_for_events(event_ids_per_row: list[str], events: frozenset[str]) -> list[int]:
    """Row indices whose event belongs to the given event set."""
    return [index for index, event_id in enumerate(event_ids_per_row) if event_id in events]


def _raw_predict(booster: xgb.Booster, features: np.ndarray) -> np.ndarray:
    """Raw XGBoost probabilities for a feature matrix (single-threaded)."""
    return booster.predict(xgb.DMatrix(features, nthread=1))


def _evaluate_folds(
    features_matrix: np.ndarray,
    labels: np.ndarray,
    event_ids_per_row: list[str],
    folds: list,
    config: CandidateConfig,
) -> tuple[
    list[FightPrediction],
    list[FoldMetrics],
    list[FoldCandidateResult],
    xgb.Booster,
    object,
    str,
]:
    """Train + calibrate + score every fold; return pooled predictions and metrics.

    Trains the locked config on each fold's training rows, selects a calibrator on
    the calibration rows, and scores the test rows. Predictions are tagged with
    their event so the event-level bootstrap can respect card grouping.
    """
    pooled: list[FightPrediction] = []
    fold_metrics: list[FoldMetrics] = []
    candidate_results: list[FoldCandidateResult] = []
    last_booster: xgb.Booster | None = None
    last_calibrator: object | None = None
    last_method = ""

    for fold in folds:
        train_rows = _rows_for_events(event_ids_per_row, fold.train_event_ids)
        cal_rows = _rows_for_events(event_ids_per_row, fold.calibration_event_ids)
        test_rows = _rows_for_events(event_ids_per_row, fold.test_event_ids)

        booster, _ = train_candidate(features_matrix[train_rows], labels[train_rows], config, _SEED)

        raw_cal = _raw_predict(booster, features_matrix[cal_rows])
        calibrators = fit_calibrators(raw_cal, labels[cal_rows], min_calibration_size=1)
        method, calibrator = select_calibrator(calibrators, raw_cal, labels[cal_rows])

        fold_predictions: list[FightPrediction] = []
        for row in test_rows:
            raw = _raw_predict(booster, features_matrix[row : row + 1])
            calibrated = float(calibrator.transform(raw)[0])
            fold_predictions.append(
                FightPrediction(
                    event_id=event_ids_per_row[row],
                    prob=calibrated,
                    label=int(labels[row]),
                )
            )
        pooled.extend(fold_predictions)

        probs = np.array([p.prob for p in fold_predictions], dtype=np.float64)
        outcomes = np.array([p.label for p in fold_predictions], dtype=np.float64)
        fold_brier = brier_score(probs, outcomes)
        fold_log_loss = log_loss_score(probs, outcomes)
        fold_ece = expected_calibration_error(probs, outcomes)

        fold_metrics.append(
            FoldMetrics(
                fold_id=fold.fold_id,
                n_train_fights=len(train_rows) // 2,
                n_cal_fights=len(cal_rows) // 2,
                n_test_fights=len(test_rows) // 2,
                brier=fold_brier,
                log_loss=fold_log_loss,
                ece=fold_ece,
                calibrator_method=method,
                all_calibrator_scores={},
            )
        )
        candidate_results.append(
            FoldCandidateResult(
                candidate_config=config,
                fold_id=fold.fold_id,
                calibrated_brier=fold_brier,
                log_loss=fold_log_loss,
                ece=fold_ece,
            )
        )
        last_booster, last_calibrator, last_method = booster, calibrator, method

    return pooled, fold_metrics, candidate_results, last_booster, last_calibrator, last_method


def _build_report(
    run_id: str,
    pooled: list[FightPrediction],
    fold_metrics: list[FoldMetrics],
) -> EvaluationReport:
    """Assemble the pooled EvaluationReport from per-fold test predictions."""
    probs = np.array([p.prob for p in pooled], dtype=np.float64)
    labels = np.array([p.label for p in pooled], dtype=np.float64)
    per_fight = (probs - labels) ** 2
    sigma = float(np.std(per_fight, ddof=1)) if len(per_fight) > 1 else 0.0

    return EvaluationReport(
        run_id=run_id,
        fold_metrics=fold_metrics,
        pooled_brier=brier_score(probs, labels),
        pooled_log_loss=log_loss_score(probs, labels),
        pooled_ece=expected_calibration_error(probs, labels),
        brier_ci=event_bootstrap_ci(pooled, brier_score, n_bootstrap=_BOOTSTRAP_N, seed=_SEED),
        log_loss_ci=event_bootstrap_ci(
            pooled, log_loss_score, n_bootstrap=_BOOTSTRAP_N, seed=_SEED
        ),
        mde=minimum_detectable_effect(len(pooled), sigma).mde,
        n_fights=len(pooled),
        n_excluded=0,
        holdout=False,
        evaluated_at="2026-08-31T00:00:00Z",
    )


@pytest.fixture(scope="module")
def pipeline() -> Iterator[_PipelineResult]:
    """Run the full pipeline once against a fresh fixture DB + local MLflow store."""
    with tempfile.TemporaryDirectory(prefix="ufc-edge-integration-") as directory:
        workdir = Path(directory)
        conn = duckdb.connect(str(workdir / "features.duckdb"))
        generate_feature_table(conn)

        mlflow.set_tracking_uri(f"sqlite:///{workdir / 'mlflow.db'}")
        mlflow.set_experiment("integration-pipeline")

        manifest = assemble_matrix(_FEATURES_TABLE, _FEATURE_VERSION, conn).manifest
        event_index = _event_index(conn)
        features_matrix, labels, event_ids_per_row, columns = _assemble_by_event(conn, event_index)
        folds = generate_folds(event_index, **_FOLD_KWARGS)

        pooled, fold_metrics, candidate_results, booster, calibrator, method = _evaluate_folds(
            features_matrix, labels, event_ids_per_row, folds, _LOCKED_CONFIG
        )
        # Second identical run verifies determinism under a fixed seed.
        pooled_second, *_ = _evaluate_folds(
            features_matrix, labels, event_ids_per_row, folds, _LOCKED_CONFIG
        )

        report = _build_report("pending", pooled, fold_metrics)
        reliability = stratified_reliability(pooled)
        ablation = run_ablation_ladder(
            folds,
            [_LOCKED_CONFIG, _SECOND_CONFIG],
            columns,
            features_matrix,
            labels,
            event_ids_per_row,
            seed=_SEED,
            bootstrap_n=_ABLATION_BOOTSTRAP_N,
        )

        run_id = log_training_run(
            config={
                "random_seed": _SEED,
                "data_revision": "fixture",
                "feature_version": _FEATURE_VERSION,
                "package_version": "0.1.0",
                "elapsed_seconds": 0.0,
            },
            booster=booster,
            calibrator=calibrator,
            folds=folds,
            eval_report=report,
            reliability=reliability,
            ablation=ablation,
            manifest=manifest,
            candidate_results=candidate_results,
            selected_candidate=_LOCKED_CONFIG,
        )

        client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())

        pooled_probs_first = np.array([p.prob for p in pooled], dtype=np.float64)
        pooled_probs_second = np.array([p.prob for p in pooled_second], dtype=np.float64)
        pooled_labels = np.array([p.label for p in pooled], dtype=np.float64)

        result = _PipelineResult(
            manifest_columns=manifest.columns,
            manifest_n_rows=manifest.n_rows,
            manifest_exclusions=manifest.exclusions,
            event_index=event_index,
            folds=folds,
            report=report,
            reliability=reliability,
            ablation=ablation,
            run_id=run_id,
            client=client,
            booster=booster,
            calibrator=calibrator,
            calibrator_method=method,
            features_matrix=features_matrix,
            pooled_probs_first=pooled_probs_first,
            pooled_probs_second=pooled_probs_second,
            pooled_brier_first=brier_score(pooled_probs_first, pooled_labels),
            pooled_brier_second=brier_score(pooled_probs_second, pooled_labels),
        )
        conn.close()
        yield result
        mlflow.end_run()


# ---------------------------------------------------------------------------
# Assembly and structure
# ---------------------------------------------------------------------------


class TestPipelineAssembly:
    """Matrix assembly and fold generation produce the expected shapes."""

    def test_excludes_draws_and_no_contests(self, pipeline: _PipelineResult) -> None:
        """The two draws and one no-contest in the fixture are excluded by reason."""
        assert pipeline.manifest_exclusions == {"draw": 2, "nc": 1}

    def test_row_count_is_two_per_decision_fight(self, pipeline: _PipelineResult) -> None:
        """47 decision fights yield 94 mirrored training rows."""
        assert pipeline.manifest_n_rows == 2 * 47

    def test_generates_expanding_window_folds(self, pipeline: _PipelineResult) -> None:
        """The fixture supports the requested two expanding-window folds."""
        assert len(pipeline.folds) == 2


# ---------------------------------------------------------------------------
# Property: no market columns
# ---------------------------------------------------------------------------


class TestNoMarketLeakage:
    """The assembled matrix never carries a market-derived column."""

    def test_no_market_columns_in_matrix(self, pipeline: _PipelineResult) -> None:
        """The column set is disjoint from the market blocklist."""
        assert set(pipeline.manifest_columns).isdisjoint(MARKET_COLUMNS)

    def test_opening_implied_prob_stripped(self, pipeline: _PipelineResult) -> None:
        """The fixture's market column is not present among features."""
        assert "opening_implied_prob" not in pipeline.manifest_columns


# ---------------------------------------------------------------------------
# Property: no temporal leakage
# ---------------------------------------------------------------------------


class TestNoTemporalLeakage:
    """Every fold respects strict train < calibration < test temporal ordering."""

    def test_partitions_are_time_ordered(self, pipeline: _PipelineResult) -> None:
        """Train events precede calibration events which precede test events."""
        dates = {entry.event_url: entry.event_date for entry in pipeline.event_index}
        for fold in pipeline.folds:
            latest_train = max(dates[e] for e in fold.train_event_ids)
            earliest_cal = min(dates[e] for e in fold.calibration_event_ids)
            earliest_test = min(dates[e] for e in fold.test_event_ids)
            assert latest_train < earliest_cal < earliest_test

    def test_partitions_are_disjoint(self, pipeline: _PipelineResult) -> None:
        """No event appears in more than one partition of a fold."""
        for fold in pipeline.folds:
            assert fold.train_event_ids.isdisjoint(fold.calibration_event_ids)
            assert fold.train_event_ids.isdisjoint(fold.test_event_ids)
            assert fold.calibration_event_ids.isdisjoint(fold.test_event_ids)


# ---------------------------------------------------------------------------
# Property: determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """A fixed seed yields identical predictions and metrics across runs."""

    def test_pooled_predictions_are_identical(self, pipeline: _PipelineResult) -> None:
        """Two identically seeded pipeline runs produce identical probabilities."""
        assert np.array_equal(pipeline.pooled_probs_first, pipeline.pooled_probs_second)

    def test_pooled_brier_is_identical(self, pipeline: _PipelineResult) -> None:
        """The pooled Brier score is reproducible to the bit."""
        assert pipeline.pooled_brier_first == pipeline.pooled_brier_second


# ---------------------------------------------------------------------------
# Property: exact order symmetry
# ---------------------------------------------------------------------------


class TestSymmetry:
    """predict_symmetric is invariant to fighter argument order."""

    def _predict(
        self,
        pipeline: _PipelineResult,
        features_a: np.ndarray,
        features_b: np.ndarray,
        url_a: str,
        url_b: str,
    ) -> SymmetricPrediction:
        return predict_symmetric(
            lambda vector: float(_raw_predict(pipeline.booster, vector.reshape(1, -1))[0]),
            lambda prob: float(pipeline.calibrator.transform(np.array([prob]))[0]),
            pipeline.calibrator_method,
            features_a,
            features_b,
            url_a,
            url_b,
            "http://fight/x",
        )

    def test_reversed_order_sums_to_one(self, pipeline: _PipelineResult) -> None:
        """For many pairs, P(A wins) + P(B wins) equals one within tolerance."""
        rng = np.random.default_rng(_SEED)
        matrix = pipeline.features_matrix
        url_a, url_b = "http://fighter/aaa", "http://fighter/bbb"
        for _ in range(50):
            features_a = matrix[int(rng.integers(0, matrix.shape[0]))]
            features_b = matrix[int(rng.integers(0, matrix.shape[0]))]
            # Swap features and URLs together so each fighter keeps its own vector.
            forward = self._predict(pipeline, features_a, features_b, url_a, url_b)
            reverse = self._predict(pipeline, features_b, features_a, url_b, url_a)
            p_a_wins = forward.p_calibrated
            p_b_wins = 1.0 - reverse.p_calibrated
            assert abs(p_a_wins + p_b_wins - 1.0) < 1e-12

    def test_canonical_probability_is_order_independent(self, pipeline: _PipelineResult) -> None:
        """The canonical-fighter probability is identical regardless of arg order."""
        matrix = pipeline.features_matrix
        features_a, features_b = matrix[0], matrix[1]
        url_a, url_b = "http://fighter/aaa", "http://fighter/bbb"
        forward = self._predict(pipeline, features_a, features_b, url_a, url_b)
        reverse = self._predict(pipeline, features_b, features_a, url_b, url_a)
        assert forward.p_calibrated == reverse.p_calibrated


# ---------------------------------------------------------------------------
# Ablation ladder
# ---------------------------------------------------------------------------


class TestAblationLadder:
    """The ladder runs across all five rungs on identical folds."""

    def test_five_rungs_in_order(self, pipeline: _PipelineResult) -> None:
        """The five rungs appear in cumulative order, naive first."""
        assert [rung.rung for rung in pipeline.ablation] == [
            "naive",
            "record",
            "physical",
            "schedule_strength",
            "domain_interactions",
        ]

    def test_naive_rung_is_chance_baseline(self, pipeline: _PipelineResult) -> None:
        """The naive rung predicts constant 0.5, giving Brier 0.25."""
        assert pipeline.ablation[0].brier == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Property: provenance artifacts
# ---------------------------------------------------------------------------


class TestProvenanceLogged:
    """The pipeline logs one complete, finished MLflow run."""

    def test_run_is_finished(self, pipeline: _PipelineResult) -> None:
        """The provenance run terminates successfully."""
        assert pipeline.client.get_run(pipeline.run_id).info.status == "FINISHED"

    def test_all_expected_artifacts_present(self, pipeline: _PipelineResult) -> None:
        """Every typed artifact family is logged to the run."""
        artifacts = {artifact.path for artifact in pipeline.client.list_artifacts(pipeline.run_id)}
        assert artifacts == {
            "assembly_manifest.json",
            "config.json",
            "fold_assignments.json",
            "evaluation_report.json",
            "reliability.json",
            "candidate_comparison.json",
            "ablation.json",
            "calibrator.pkl",
            "run_manifest.json",
            "model",
        }

    def test_run_manifest_lists_every_artifact(self, pipeline: _PipelineResult) -> None:
        """The run manifest enumerates each artifact path and type."""
        manifest_path = pipeline.client.download_artifacts(pipeline.run_id, "run_manifest.json")
        manifest = json.loads(Path(manifest_path).read_text())
        assert manifest["run_id"] == pipeline.run_id
        paths = {entry["path"] for entry in manifest["artifacts"]}
        assert "evaluation_report.json" in paths
        assert "ablation.json" in paths
        assert "model" in paths

    def test_evaluation_report_matches_logged_artifact(self, pipeline: _PipelineResult) -> None:
        """The logged evaluation report carries the pooled Brier we computed."""
        report_path = pipeline.client.download_artifacts(pipeline.run_id, "evaluation_report.json")
        logged = json.loads(Path(report_path).read_text())
        assert logged["pooled_brier"] == pipeline.report.pooled_brier
        assert logged["holdout"] is False
