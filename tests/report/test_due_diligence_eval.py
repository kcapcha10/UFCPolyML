"""Tests for the due-diligence eval harness.

Validates that precision/recall are computed correctly from known confusion-
matrix outcomes, that the gate check enforces minimum thresholds, and that
the harness exercises the production due_diligence.run_due_diligence code
path (not a reimplementation) with a mocked LLM client.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from ufc_edge.report.due_diligence import DueDiligenceError
from ufc_edge.report.due_diligence_eval import (
    EvalResult,
    LabeledFight,
    gate_check,
    load_labels,
    run_eval,
)
from ufc_edge.report.storage import REPORT_DDL

# ── Fixtures ──────────────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with all report tables created."""
    db = duckdb.connect(":memory:")
    for ddl in REPORT_DDL:
        db.execute(ddl)
    return db


def _make_llm_response(verdict: str) -> str:
    """Build a valid LLM JSON response with the specified verdict."""
    return json.dumps(
        {
            "verdict": verdict,
            "confidence": 0.85,
            "evidence_urls": ["https://example.com/evidence"],
            "summary": f"Test verdict: {verdict}",
            "checklist_findings": {
                "injury_news": {
                    "present": verdict != "CONFIRM",
                    "detail": "Test finding",
                    "source_url": "https://example.com/evidence",
                },
                "weight_cut_concern": None,
                "short_notice_replacement": None,
                "camp_change": None,
                "other_material_news": None,
            },
        }
    )


def _make_labels(entries: list[tuple[str, bool]]) -> list[LabeledFight]:
    """Build labeled fights from (fight_url, ground_truth_concern) pairs."""
    return [
        LabeledFight(
            fight_url=url,
            event_date="2025-01-01",
            fighter_a="Fighter A",
            fighter_b="Fighter B",
            ground_truth_concern=concern,
            notes="",
        )
        for url, concern in entries
    ]


def _search_client(query: str) -> list[str]:
    """Mock search client returning a fixed URL list."""
    return ["https://example.com/evidence"]


# ── Fixture loading tests ─────────────────────────────────────────────────────


class TestLoadLabels:
    """Fixture JSON is loaded correctly into LabeledFight objects."""

    def test_loads_fixture_file(self):
        """The development fixture file loads without error."""
        labels = load_labels(FIXTURES_DIR / "due_diligence_labels.json")
        assert len(labels) == 5

    def test_fields_populated(self):
        """Each loaded label has all expected fields."""
        labels = load_labels(FIXTURES_DIR / "due_diligence_labels.json")
        for label in labels:
            assert label.fight_url
            assert label.event_date
            assert label.fighter_a
            assert label.fighter_b
            assert isinstance(label.ground_truth_concern, bool)

    def test_ground_truth_distribution(self):
        """Fixture has a mix of concern=True and concern=False labels."""
        labels = load_labels(FIXTURES_DIR / "due_diligence_labels.json")
        concerns = [lbl.ground_truth_concern for lbl in labels]
        assert True in concerns
        assert False in concerns


# ── Precision/recall computation tests ────────────────────────────────────────


class TestPrecisionRecall:
    """The harness computes precision/recall correctly from known outcomes."""

    def test_perfect_predictions(self, conn):
        """All correct predictions yield precision=1.0, recall=1.0."""
        labels = _make_labels(
            [
                ("/fight/tp-1", True),
                ("/fight/tp-2", True),
                ("/fight/tn-1", False),
            ]
        )

        # LLM returns QUALIFY for concern=True fights, CONFIRM for concern=False
        call_order = iter(["QUALIFY", "QUALIFY", "CONFIRM"])

        def sequenced_llm(prompt: str) -> str:
            return _make_llm_response(next(call_order))

        result = run_eval(labels, sequenced_llm, _search_client, conn)

        assert result.true_positives == 2
        assert result.true_negatives == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0
        assert result.precision == 1.0
        assert result.recall == 1.0

    def test_known_confusion_matrix(self, conn):
        """Hand-constructed case: 2 TP, 1 FP, 1 FN, 1 TN => P=2/3, R=2/3."""
        labels = _make_labels(
            [
                ("/fight/a", True),  # LLM says QUALIFY => TP
                ("/fight/b", True),  # LLM says QUALIFY => TP
                ("/fight/c", True),  # LLM says CONFIRM => FN
                ("/fight/d", False),  # LLM says QUALIFY => FP
                ("/fight/e", False),  # LLM says CONFIRM => TN
            ]
        )

        verdicts = iter(["QUALIFY", "QUALIFY", "CONFIRM", "QUALIFY", "CONFIRM"])

        def sequenced_llm(prompt: str) -> str:
            return _make_llm_response(next(verdicts))

        result = run_eval(labels, sequenced_llm, _search_client, conn)

        assert result.true_positives == 2
        assert result.false_positives == 1
        assert result.false_negatives == 1
        assert result.true_negatives == 1
        assert result.precision == pytest.approx(2.0 / 3.0)
        assert result.recall == pytest.approx(2.0 / 3.0)
        assert result.total == 5

    def test_no_positive_predictions_yields_zero_precision(self, conn):
        """When the LLM never flags a concern, precision is 0.0."""
        labels = _make_labels(
            [
                ("/fight/fn-1", True),
                ("/fight/tn-1", False),
            ]
        )

        def always_confirm(prompt: str) -> str:
            return _make_llm_response("CONFIRM")

        result = run_eval(labels, always_confirm, _search_client, conn)

        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.false_negatives == 1
        assert result.true_negatives == 1

    def test_veto_counts_as_concern_flagged(self, conn):
        """VETO verdicts are treated as 'concern flagged' (same as QUALIFY)."""
        labels = _make_labels([("/fight/veto-tp", True)])

        def veto_llm(prompt: str) -> str:
            return _make_llm_response("VETO")

        result = run_eval(labels, veto_llm, _search_client, conn)

        assert result.true_positives == 1
        assert result.precision == 1.0
        assert result.recall == 1.0


# ── Gate check tests ──────────────────────────────────────────────────────────


class TestGateCheck:
    """gate_check enforces precision >= 0.80 and recall >= 0.60."""

    def test_passes_when_both_thresholds_met(self):
        """Gate passes when precision and recall both exceed their thresholds."""
        result = EvalResult(
            true_positives=4,
            false_positives=1,
            true_negatives=3,
            false_negatives=2,
            precision=0.80,
            recall=0.67,
            total=10,
        )
        assert gate_check(result) is True

    def test_fails_when_precision_below_threshold(self):
        """Gate fails when precision is below 0.80."""
        result = EvalResult(
            true_positives=3,
            false_positives=2,
            true_negatives=3,
            false_negatives=2,
            precision=0.60,
            recall=0.60,
            total=10,
        )
        assert gate_check(result) is False

    def test_fails_when_recall_below_threshold(self):
        """Gate fails when recall is below 0.60."""
        result = EvalResult(
            true_positives=2,
            false_positives=0,
            true_negatives=5,
            false_negatives=3,
            precision=1.0,
            recall=0.40,
            total=10,
        )
        assert gate_check(result) is False

    def test_fails_when_both_below_threshold(self):
        """Gate fails when both precision and recall are below thresholds."""
        result = EvalResult(
            true_positives=1,
            false_positives=3,
            true_negatives=2,
            false_negatives=4,
            precision=0.25,
            recall=0.20,
            total=10,
        )
        assert gate_check(result) is False

    def test_passes_at_exact_threshold_boundary(self):
        """Gate passes when precision=0.80 and recall=0.60 exactly."""
        result = EvalResult(
            true_positives=3,
            false_positives=0,
            true_negatives=2,
            false_negatives=0,
            precision=0.80,
            recall=0.60,
            total=5,
        )
        assert gate_check(result) is True


# ── Production code path tests ────────────────────────────────────────────────


class TestProductionCodePath:
    """The harness uses the production run_due_diligence function, not a stub."""

    def test_verdicts_persisted_to_database(self, conn):
        """run_eval persists verdicts via the production storage path."""
        labels = _make_labels([("/fight/persist-check", True)])

        def llm(prompt: str) -> str:
            return _make_llm_response("QUALIFY")

        run_eval(labels, llm, _search_client, conn)

        row = conn.execute(
            "SELECT verdict FROM due_diligence_verdicts WHERE fight_url = ? AND report_run_id = ?",
            ["/fight/persist-check", "eval-harness"],
        ).fetchone()

        assert row is not None
        assert row[0] == "QUALIFY"

    def test_run_logged_in_due_diligence_runs(self, conn):
        """run_eval logs each invocation via the production logging path."""
        labels = _make_labels([("/fight/log-check", False)])

        def llm(prompt: str) -> str:
            return _make_llm_response("CONFIRM")

        run_eval(labels, llm, _search_client, conn)

        row = conn.execute(
            "SELECT success, prompt_version, model_name "
            "FROM due_diligence_runs "
            "WHERE fight_url = ? AND report_run_id = ?",
            ["/fight/log-check", "eval-harness"],
        ).fetchone()

        assert row is not None
        assert row[0] is True  # success

    def test_schema_validation_applies(self, conn):
        """Invalid LLM responses are caught by the production parsing logic."""
        labels = _make_labels([("/fight/bad-response", True)])

        def bad_llm(prompt: str) -> str:
            return "not valid json at all"

        with pytest.raises(DueDiligenceError):
            run_eval(labels, bad_llm, _search_client, conn)
