"""Integration tests for the end-to-end report runner.

Exercises run_report over a fixture DuckDB and a local temporary MLflow run:
matched bouts get full model-vs-market columns, unmatched bouts get null
probability fields, flagged signals trigger annotate-only due diligence and
post-signal snapshot scheduling, reruns append new rows without mutating prior
ones, and a run whose MLflow artifacts are missing fails before any write.

All external boundaries (model inference, LLM, search) are injected fakes; the
only "real" dependency is a file-backed MLflow store created under tmp_path.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import duckdb
import mlflow
import pytest
from mlflow.tracking import MlflowClient

from ufc_edge.data.polymarket.storage import POLYMARKET_DDL
from ufc_edge.report.runner import (
    MissingArtifactError,
    RunnerConfig,
    UpcomingFight,
    run_report,
)
from ufc_edge.report.schemas import GateVerdict, MatchStatus
from ufc_edge.report.storage import REPORT_DDL

# ── Constants shared across the fixture graph ─────────────────────────────────

_AS_OF = datetime(2026, 7, 1, 12, 0, 0)
_CAPTURED_AT = datetime(2026, 7, 1, 10, 0, 0)

_FIGHT_FLAGGED = "http://ufcstats.com/fight/mp"
_FIGHT_WITHIN_NOISE = "http://ufcstats.com/fight/pp"
_FIGHT_UNMATCHED = "http://ufcstats.com/fight/xx"

_FIGHTER = {
    "mcgregor": "http://ufcstats.com/fighter/aaa",
    "poirier": "http://ufcstats.com/fighter/bbb",
    "prochazka": "http://ufcstats.com/fighter/ccc",
    "pereira": "http://ufcstats.com/fighter/ddd",
    "nobody_one": "http://ufcstats.com/fighter/eee",
    "nobody_two": "http://ufcstats.com/fighter/fff",
}

_TOKEN_FLAGGED = "token-mcgregor-poirier"
_TOKEN_WITHIN_NOISE = "token-prochazka-pereira"

# p_model chosen against a market mid of 0.50 and bucket calibration error 0.05
# (gate threshold = k * error = 2.0 * 0.05 = 0.10):
#   0.75 -> |mismatch| 0.25 > 0.10 -> FLAGGED
#   0.52 -> |mismatch| 0.02 < 0.10 -> WITHIN_NOISE
_P_MODEL = {_FIGHT_FLAGGED: 0.75, _FIGHT_WITHIN_NOISE: 0.52}
_MARKET_MID = 0.50


# ── Injected fakes (deterministic, no network) ────────────────────────────────


class _RecordingPredictor:
    """Fake symmetric predictor returning known calibrated probabilities.

    Records every matchup it is asked about so tests can assert that market data
    never drives inference and that unmatched bouts skip the model entirely.
    """

    def __init__(self, values: dict[str, float]) -> None:
        self._values = values
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, a_url: str, b_url: str, fight_url: str) -> float | None:
        self.calls.append((a_url, b_url, fight_url))
        return self._values.get(fight_url)


class _RecordingLLM:
    """Fake LLM client returning a fixed valid verdict JSON, counting calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        return json.dumps(
            {
                "verdict": "CONFIRM",
                "confidence": 0.8,
                "evidence_urls": ["http://example.com/news"],
                "summary": "No material concerns found.",
                "checklist_findings": {
                    "injury_news": None,
                    "weight_cut_concern": None,
                    "short_notice_replacement": None,
                    "camp_change": None,
                    "other_material_news": None,
                },
            }
        )


def _fake_search(query: str) -> list[str]:
    return ["http://example.com/news"]


# ── Fixture DuckDB ─────────────────────────────────────────────────────────────


def _seed_snapshot(
    conn: duckdb.DuckDBPyConnection,
    *,
    market_id: str,
    token_id: str,
    question: str,
) -> None:
    conn.execute(
        """
        INSERT INTO order_book_snapshots
            (market_id, token_id, question, outcome, bids, asks, mid_price,
             spread, best_bid, best_ask, best_bid_size, best_ask_size,
             captured_at, tick_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            market_id,
            token_id,
            question,
            "Yes",
            json.dumps([{"price": 0.49, "size": 100}]),
            json.dumps([{"price": 0.51, "size": 80}]),
            _MARKET_MID,
            0.02,
            0.49,
            0.51,
            100.0,
            80.0,
            _CAPTURED_AT,
            f"tick-{token_id}",
        ],
    )


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with report, polymarket, and ufcstats-lite schemas seeded."""
    db = duckdb.connect(":memory:")

    for ddl in POLYMARKET_DDL:
        db.execute(ddl)
    for ddl in REPORT_DDL:
        db.execute(ddl)

    db.execute(
        "CREATE TABLE fighters (fighter_url VARCHAR PRIMARY KEY, "
        "name VARCHAR NOT NULL, scraped_at TIMESTAMP NOT NULL)"
    )
    # sparse_history joins these two; empty tables are sufficient (count -> 0).
    db.execute("CREATE TABLE events (event_url VARCHAR PRIMARY KEY, date DATE)")
    db.execute(
        "CREATE TABLE fights (fight_url VARCHAR PRIMARY KEY, event_url VARCHAR, "
        "fighter_a_url VARCHAR, fighter_b_url VARCHAR)"
    )

    names = {
        _FIGHTER["mcgregor"]: "Conor McGregor",
        _FIGHTER["poirier"]: "Dustin Poirier",
        _FIGHTER["prochazka"]: "Jiří Procházka",
        _FIGHTER["pereira"]: "Alex Pereira",
        _FIGHTER["nobody_one"]: "Nobody One",
        _FIGHTER["nobody_two"]: "Nobody Two",
    }
    for url, name in names.items():
        db.execute("INSERT INTO fighters VALUES (?, ?, ?)", [url, name, _CAPTURED_AT])

    _seed_snapshot(
        db,
        market_id="market-1",
        token_id=_TOKEN_FLAGGED,
        question="UFC 310: Will Conor McGregor beat Dustin Poirier?",
    )
    _seed_snapshot(
        db,
        market_id="market-2",
        token_id=_TOKEN_WITHIN_NOISE,
        question="UFC 310: Will Jiri Prochazka beat Alex Pereira?",
    )
    return db


@pytest.fixture
def fights() -> list[UpcomingFight]:
    """Three upcoming bouts: one flagged, one within-noise, one unmatched."""
    event_day = date(2026, 7, 5)
    event_start = datetime(2026, 7, 5, 22, 0, 0)
    return [
        UpcomingFight(
            fight_url=_FIGHT_FLAGGED,
            fighter_a_url=_FIGHTER["mcgregor"],
            fighter_b_url=_FIGHTER["poirier"],
            fighter_a_name="Conor McGregor",
            fighter_b_name="Dustin Poirier",
            weight_class="Lightweight",
            event_date=event_day,
            event_start_time=event_start,
        ),
        UpcomingFight(
            fight_url=_FIGHT_WITHIN_NOISE,
            fighter_a_url=_FIGHTER["prochazka"],
            fighter_b_url=_FIGHTER["pereira"],
            fighter_a_name="Jiří Procházka",
            fighter_b_name="Alex Pereira",
            weight_class="Light Heavyweight",
            event_date=event_day,
            event_start_time=event_start,
        ),
        UpcomingFight(
            fight_url=_FIGHT_UNMATCHED,
            fighter_a_url=_FIGHTER["nobody_one"],
            fighter_b_url=_FIGHTER["nobody_two"],
            fighter_a_name="Nobody One",
            fighter_b_name="Nobody Two",
            weight_class="Featherweight",
            event_date=event_day,
            event_start_time=event_start,
        ),
    ]


@pytest.fixture
def config() -> RunnerConfig:
    return RunnerConfig(
        gate_k=2.0,
        sparse_history_threshold=3,
        due_diligence_enabled=True,
        due_diligence_prompt_version="v1",
        due_diligence_model_name="test-model",
        due_diligence_model_version="1.0",
    )


@pytest.fixture
def predictor() -> _RecordingPredictor:
    return _RecordingPredictor(_P_MODEL)


@pytest.fixture
def llm() -> _RecordingLLM:
    return _RecordingLLM()


def _reliability_buckets() -> list[dict]:
    """Four fixed reliability buckets with a uniform 0.05 calibration error."""
    ranges = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9)]
    return [
        {
            "lower": lower,
            "upper": upper,
            "n_fights": 50,
            "mean_predicted": (lower + upper) / 2,
            "observed_win_rate": (lower + upper) / 2,
            "calibration_error": 0.05,
            "ci_lower": lower,
            "ci_upper": upper,
            "low_support": False,
        }
        for lower, upper in ranges
    ]


def _make_mlflow_run(base_dir: Path, *, with_reliability: bool) -> str:
    """Create a local sqlite-backed MLflow run with model/calibrator (+reliability).

    Uses a sqlite tracking backend (the file store is disabled in this MLflow
    version) and an artifact location isolated under base_dir. Stub artifacts
    stand in for the booster and calibrator — the runner only checks their
    presence, never loads them.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{base_dir}/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_name = "report-runner-test"
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name, artifact_location=(base_dir / "artifacts").as_uri()
        )
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.log_params({"feature_version": "v1", "data_revision": "rev-abc"})
        mlflow.log_text("stub booster", "model/booster.json")
        mlflow.log_text("stub calibrator", "calibrator.pkl")
        if with_reliability:
            mlflow.log_dict({"buckets": _reliability_buckets()}, "reliability.json")
        return run.info.run_id


@pytest.fixture
def mlflow_run_id(tmp_path) -> str:
    return _make_mlflow_run(tmp_path / "mlruns", with_reliability=True)


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestFullPipeline:
    """A full run produces one report_run and one paper_signal per bout."""

    def test_writes_one_report_run(self, conn, fights, config, predictor, llm, mlflow_run_id):
        run_report(
            _AS_OF,
            mlflow_run_id,
            fights,
            config,
            conn,
            predict_matchup=predictor,
            llm_client=llm,
            search_client=_fake_search,
        )
        count = conn.execute("SELECT COUNT(*) FROM report_runs").fetchone()[0]
        assert count == 1

    def test_writes_one_signal_per_bout(self, conn, fights, config, predictor, llm, mlflow_run_id):
        run_report(
            _AS_OF,
            mlflow_run_id,
            fights,
            config,
            conn,
            predict_matchup=predictor,
            llm_client=llm,
            search_client=_fake_search,
        )
        count = conn.execute("SELECT COUNT(*) FROM paper_signals").fetchone()[0]
        assert count == len(fights)

    def test_report_run_counts(self, conn, fights, config, predictor, llm, mlflow_run_id):
        report_run = run_report(
            _AS_OF,
            mlflow_run_id,
            fights,
            config,
            conn,
            predict_matchup=predictor,
            llm_client=llm,
            search_client=_fake_search,
        )
        assert report_run.bout_count == 3
        assert report_run.flagged_count == 1


class TestMatchedBout:
    """A matched bout carries full model, market, and gate columns."""

    def test_matched_flagged_row_is_fully_populated(
        self, conn, fights, config, predictor, llm, mlflow_run_id
    ):
        run_report(
            _AS_OF,
            mlflow_run_id,
            fights,
            config,
            conn,
            predict_matchup=predictor,
            llm_client=llm,
            search_client=_fake_search,
        )
        row = conn.execute(
            """
            SELECT p_model, p_market_mid, mismatch, gate_verdict, bucket_id,
                   bucket_n, token_id, snapshot_timestamp, match_status
            FROM paper_signals WHERE fight_url = ?
            """,
            [_FIGHT_FLAGGED],
        ).fetchone()
        p_model, mid, mismatch, verdict, bucket_id, bucket_n, token, snap, status = row
        assert p_model == pytest.approx(0.75)
        assert mid == pytest.approx(0.50)
        assert mismatch == pytest.approx(0.25)
        assert verdict == GateVerdict.FLAGGED.value
        assert bucket_id == "0.7-0.9"
        assert bucket_n == 50
        assert token == _TOKEN_FLAGGED
        assert snap == _CAPTURED_AT
        assert status == MatchStatus.MATCHED.value

    def test_within_noise_bout_is_not_flagged(
        self, conn, fights, config, predictor, llm, mlflow_run_id
    ):
        run_report(
            _AS_OF,
            mlflow_run_id,
            fights,
            config,
            conn,
            predict_matchup=predictor,
            llm_client=llm,
            search_client=_fake_search,
        )
        verdict = conn.execute(
            "SELECT gate_verdict FROM paper_signals WHERE fight_url = ?",
            [_FIGHT_WITHIN_NOISE],
        ).fetchone()[0]
        assert verdict == GateVerdict.WITHIN_NOISE.value


class TestUnmatchedBout:
    """An unmatched bout gets null probability fields and skips the model."""

    def test_null_probability_fields(self, conn, fights, config, predictor, llm, mlflow_run_id):
        run_report(
            _AS_OF,
            mlflow_run_id,
            fights,
            config,
            conn,
            predict_matchup=predictor,
            llm_client=llm,
            search_client=_fake_search,
        )
        row = conn.execute(
            """
            SELECT p_model, p_market_mid, mismatch, gate_verdict, match_status
            FROM paper_signals WHERE fight_url = ?
            """,
            [_FIGHT_UNMATCHED],
        ).fetchone()
        p_model, mid, mismatch, verdict, status = row
        assert p_model is None
        assert mid is None
        assert mismatch is None
        assert verdict is None
        assert status == MatchStatus.NO_CANDIDATE.value

    def test_model_not_invoked_for_unmatched_bout(
        self, conn, fights, config, predictor, llm, mlflow_run_id
    ):
        run_report(
            _AS_OF,
            mlflow_run_id,
            fights,
            config,
            conn,
            predict_matchup=predictor,
            llm_client=llm,
            search_client=_fake_search,
        )
        queried_fights = {fight_url for _, _, fight_url in predictor.calls}
        assert _FIGHT_UNMATCHED not in queried_fights


class TestDueDiligence:
    """Flagged signals trigger annotate-only due diligence; others do not."""

    def test_flagged_signal_invokes_llm(self, conn, fights, config, predictor, llm, mlflow_run_id):
        run_report(
            _AS_OF,
            mlflow_run_id,
            fights,
            config,
            conn,
            predict_matchup=predictor,
            llm_client=llm,
            search_client=_fake_search,
        )
        assert llm.calls == 1

    def test_verdict_persisted_for_flagged_fight(
        self, conn, fights, config, predictor, llm, mlflow_run_id
    ):
        run_report(
            _AS_OF,
            mlflow_run_id,
            fights,
            config,
            conn,
            predict_matchup=predictor,
            llm_client=llm,
            search_client=_fake_search,
        )
        count = conn.execute(
            "SELECT COUNT(*) FROM due_diligence_verdicts WHERE fight_url = ?",
            [_FIGHT_FLAGGED],
        ).fetchone()[0]
        assert count == 1

    def test_no_verdict_for_within_noise_fight(
        self, conn, fights, config, predictor, llm, mlflow_run_id
    ):
        run_report(
            _AS_OF,
            mlflow_run_id,
            fights,
            config,
            conn,
            predict_matchup=predictor,
            llm_client=llm,
            search_client=_fake_search,
        )
        count = conn.execute(
            "SELECT COUNT(*) FROM due_diligence_verdicts WHERE fight_url = ?",
            [_FIGHT_WITHIN_NOISE],
        ).fetchone()[0]
        assert count == 0


class TestSnapshotScheduling:
    """Flagged signals schedule post-signal snapshots."""

    def test_snapshots_scheduled_for_flagged_signal(
        self, conn, fights, config, predictor, llm, mlflow_run_id
    ):
        run_report(
            _AS_OF,
            mlflow_run_id,
            fights,
            config,
            conn,
            predict_matchup=predictor,
            llm_client=llm,
            search_client=_fake_search,
        )
        signal_id = conn.execute(
            "SELECT signal_id FROM paper_signals WHERE fight_url = ?",
            [_FIGHT_FLAGGED],
        ).fetchone()[0]
        count = conn.execute(
            "SELECT COUNT(*) FROM post_signal_snapshots WHERE signal_id = ?",
            [signal_id],
        ).fetchone()[0]
        # 1h, 4h, 24h, plus fight-time (event_start_time provided).
        assert count == 4


class TestAppendOnlyRerun:
    """Re-running appends a new report_run without mutating prior rows."""

    def test_rerun_creates_distinct_report_run(
        self, conn, fights, config, predictor, llm, mlflow_run_id
    ):
        first = run_report(
            _AS_OF,
            mlflow_run_id,
            fights,
            config,
            conn,
            predict_matchup=predictor,
            llm_client=llm,
            search_client=_fake_search,
            now=lambda: datetime(2026, 7, 1, 12, 0, 0),
        )
        second = run_report(
            _AS_OF,
            mlflow_run_id,
            fights,
            config,
            conn,
            predict_matchup=predictor,
            llm_client=llm,
            search_client=_fake_search,
            now=lambda: datetime(2026, 7, 2, 12, 0, 0),
        )
        assert first.report_run_id != second.report_run_id

    def test_rerun_doubles_row_counts(self, conn, fights, config, predictor, llm, mlflow_run_id):
        run_report(
            _AS_OF,
            mlflow_run_id,
            fights,
            config,
            conn,
            predict_matchup=predictor,
            llm_client=llm,
            search_client=_fake_search,
        )
        run_report(
            _AS_OF,
            mlflow_run_id,
            fights,
            config,
            conn,
            predict_matchup=predictor,
            llm_client=llm,
            search_client=_fake_search,
        )
        runs = conn.execute("SELECT COUNT(*) FROM report_runs").fetchone()[0]
        signals = conn.execute("SELECT COUNT(*) FROM paper_signals").fetchone()[0]
        assert runs == 2
        assert signals == 2 * len(fights)

    def test_rerun_leaves_first_run_row_unchanged(
        self, conn, fights, config, predictor, llm, mlflow_run_id
    ):
        first_time = datetime(2026, 7, 1, 12, 0, 0)
        first = run_report(
            _AS_OF,
            mlflow_run_id,
            fights,
            config,
            conn,
            predict_matchup=predictor,
            llm_client=llm,
            search_client=_fake_search,
            now=lambda: first_time,
        )
        run_report(
            _AS_OF,
            mlflow_run_id,
            fights,
            config,
            conn,
            predict_matchup=predictor,
            llm_client=llm,
            search_client=_fake_search,
            now=lambda: datetime(2026, 7, 2, 12, 0, 0),
        )
        created_at = conn.execute(
            "SELECT created_at FROM report_runs WHERE report_run_id = ?",
            [first.report_run_id],
        ).fetchone()[0]
        assert created_at == first_time


class TestMissingArtifact:
    """A run whose MLflow bucket artifact is absent fails before any write."""

    def test_raises_missing_artifact_error(self, conn, fights, config, predictor, llm, tmp_path):
        run_id = _make_mlflow_run(tmp_path / "mlruns", with_reliability=False)
        with pytest.raises(MissingArtifactError):
            run_report(
                _AS_OF,
                run_id,
                fights,
                config,
                conn,
                predict_matchup=predictor,
                llm_client=llm,
                search_client=_fake_search,
            )

    def test_no_rows_written_on_missing_artifact(
        self, conn, fights, config, predictor, llm, tmp_path
    ):
        run_id = _make_mlflow_run(tmp_path / "mlruns", with_reliability=False)
        with pytest.raises(MissingArtifactError):
            run_report(
                _AS_OF,
                run_id,
                fights,
                config,
                conn,
                predict_matchup=predictor,
                llm_client=llm,
                search_client=_fake_search,
            )
        runs = conn.execute("SELECT COUNT(*) FROM report_runs").fetchone()[0]
        signals = conn.execute("SELECT COUNT(*) FROM paper_signals").fetchone()[0]
        links = conn.execute("SELECT COUNT(*) FROM market_fight_links").fetchone()[0]
        assert runs == 0
        assert signals == 0
        assert links == 0
