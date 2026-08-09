"""Tests for the due-diligence runner.

Validates idempotency (a second call for the same fight+run returns the
persisted verdict without re-invoking the LLM), schema validation (malformed
LLM responses are caught and recorded as failures, never silently dropped),
logging (prompt_version, model_name, model_version, invoked_at persisted on
every attempt), and evidence_urls enforcement (the schema requires at least
one URL per successful verdict — a response without URLs is caught and logged
as a failure rather than raised uncaught).

All LLM and search calls are mocked — no real network I/O.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import duckdb
import pytest

from ufc_edge.report.due_diligence import (
    DueDiligenceError,
    run_due_diligence,
)
from ufc_edge.report.schemas import DueDiligenceVerdictType
from ufc_edge.report.storage import REPORT_DDL

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with all report tables created."""
    db = duckdb.connect(":memory:")
    for ddl in REPORT_DDL:
        db.execute(ddl)
    return db


@pytest.fixture
def checklist_context() -> dict[str, str]:
    """Minimal checklist context for the runner."""
    return {
        "fighter_a": "Conor McGregor",
        "fighter_b": "Dustin Poirier",
        "event_date": "2025-07-10",
    }


def _valid_llm_response() -> str:
    """A well-formed LLM JSON response matching the verdict schema."""
    return json.dumps({
        "verdict": "CONFIRM",
        "confidence": 0.85,
        "evidence_urls": ["https://mmajunkie.com/article-1"],
        "summary": "No material concerns found. Both fighters healthy.",
        "checklist_findings": {
            "injury_news": {"present": False, "detail": "No injuries reported", "source_url": "https://mmajunkie.com/article-1"},
            "weight_cut_concern": None,
            "short_notice_replacement": None,
            "camp_change": None,
            "other_material_news": None,
        },
    })


def _mock_search_client(urls: list[str] | None = None):
    """Return a search client callable that yields fixed URLs."""
    returned_urls = urls if urls is not None else ["https://mmajunkie.com/article-1"]

    def _search(query: str) -> list[str]:
        return returned_urls

    return _search


def _mock_llm_client(response: str | None = None):
    """Return an LLM client callable that yields a fixed response."""
    resp = response if response is not None else _valid_llm_response()

    def _llm(prompt: str) -> str:
        return resp

    return _llm


# ── Idempotency tests ─────────────────────────────────────────────────────────


class TestIdempotency:
    """Second call for the same (fight_url, report_run_id) returns persisted verdict."""

    def test_second_call_returns_persisted_verdict(self, conn, checklist_context):
        """Calling run_due_diligence twice returns the same verdict both times."""
        call_count = {"llm": 0}

        def counting_llm(prompt: str) -> str:
            call_count["llm"] += 1
            return _valid_llm_response()

        fight_url = "/fight/abc123"
        run_id = "run-001"

        first = run_due_diligence(
            fight_url, run_id, checklist_context,
            counting_llm, _mock_search_client(), conn,
        )
        second = run_due_diligence(
            fight_url, run_id, checklist_context,
            counting_llm, _mock_search_client(), conn,
        )

        assert first.verdict == second.verdict
        assert first.fight_url == second.fight_url
        assert first.report_run_id == second.report_run_id
        assert first.confidence == second.confidence

    def test_llm_not_invoked_on_second_call(self, conn, checklist_context):
        """The LLM client is called exactly once across two invocations."""
        call_count = {"llm": 0}

        def counting_llm(prompt: str) -> str:
            call_count["llm"] += 1
            return _valid_llm_response()

        fight_url = "/fight/abc123"
        run_id = "run-001"

        run_due_diligence(
            fight_url, run_id, checklist_context,
            counting_llm, _mock_search_client(), conn,
        )
        run_due_diligence(
            fight_url, run_id, checklist_context,
            counting_llm, _mock_search_client(), conn,
        )

        assert call_count["llm"] == 1

    def test_different_run_id_invokes_llm_again(self, conn, checklist_context):
        """A different report_run_id triggers a fresh LLM invocation."""
        call_count = {"llm": 0}

        def counting_llm(prompt: str) -> str:
            call_count["llm"] += 1
            return _valid_llm_response()

        fight_url = "/fight/abc123"

        run_due_diligence(
            fight_url, "run-001", checklist_context,
            counting_llm, _mock_search_client(), conn,
        )
        run_due_diligence(
            fight_url, "run-002", checklist_context,
            counting_llm, _mock_search_client(), conn,
        )

        assert call_count["llm"] == 2


# ── Schema validation tests ───────────────────────────────────────────────────


class TestSchemaValidation:
    """Malformed LLM responses raise DueDiligenceError and log the failure."""

    def test_invalid_json_raises(self, conn, checklist_context):
        """Non-JSON response raises DueDiligenceError."""
        bad_llm = _mock_llm_client("this is not json at all")

        with pytest.raises(DueDiligenceError, match="Invalid LLM response"):
            run_due_diligence(
                "/fight/bad-json", "run-bad", checklist_context,
                bad_llm, _mock_search_client(), conn,
            )

    def test_missing_required_field_raises(self, conn, checklist_context):
        """JSON missing a required field raises DueDiligenceError."""
        incomplete = json.dumps({"verdict": "CONFIRM", "confidence": 0.9})
        bad_llm = _mock_llm_client(incomplete)

        with pytest.raises(DueDiligenceError, match="Invalid LLM response"):
            run_due_diligence(
                "/fight/incomplete", "run-incomplete", checklist_context,
                bad_llm, _mock_search_client(), conn,
            )

    def test_invalid_verdict_value_raises(self, conn, checklist_context):
        """Invalid verdict enum value raises DueDiligenceError."""
        bad_verdict = json.dumps({
            "verdict": "INVALID_VALUE",
            "confidence": 0.5,
            "evidence_urls": ["https://example.com"],
            "summary": "test",
            "checklist_findings": {},
        })
        bad_llm = _mock_llm_client(bad_verdict)

        with pytest.raises(DueDiligenceError, match="Invalid LLM response"):
            run_due_diligence(
                "/fight/bad-verdict", "run-bad-verdict", checklist_context,
                bad_llm, _mock_search_client(), conn,
            )

    def test_failed_verdict_logged_with_error_reason(self, conn, checklist_context):
        """A failed LLM response still logs the attempt with an error reason."""
        bad_llm = _mock_llm_client("not json")

        with pytest.raises(DueDiligenceError):
            run_due_diligence(
                "/fight/logged-fail", "run-logged", checklist_context,
                bad_llm, _mock_search_client(), conn,
            )

        # Verify the failed run was logged
        row = conn.execute(
            "SELECT success, error_reason FROM due_diligence_runs "
            "WHERE fight_url = ? AND report_run_id = ?",
            ["/fight/logged-fail", "run-logged"],
        ).fetchone()

        assert row is not None
        assert row[0] is False
        assert row[1] is not None
        assert "Response validation failed" in row[1]

    def test_llm_exception_logged_with_error_reason(self, conn, checklist_context):
        """An LLM client exception is logged and not silently dropped."""
        def failing_llm(prompt: str) -> str:
            msg = "Provider timeout after 30s"
            raise TimeoutError(msg)

        with pytest.raises(DueDiligenceError, match="LLM client failed"):
            run_due_diligence(
                "/fight/timeout", "run-timeout", checklist_context,
                failing_llm, _mock_search_client(), conn,
            )

        row = conn.execute(
            "SELECT success, error_reason FROM due_diligence_runs "
            "WHERE fight_url = ? AND report_run_id = ?",
            ["/fight/timeout", "run-timeout"],
        ).fetchone()

        assert row is not None
        assert row[0] is False
        assert "LLM invocation failed" in row[1]

    def test_search_failure_logged(self, conn, checklist_context):
        """A search client failure is logged and raises DueDiligenceError."""
        def failing_search(query: str) -> list[str]:
            msg = "Search API unavailable"
            raise ConnectionError(msg)

        with pytest.raises(DueDiligenceError, match="Search client failed"):
            run_due_diligence(
                "/fight/search-fail", "run-search-fail", checklist_context,
                _mock_llm_client(), failing_search, conn,
            )

        row = conn.execute(
            "SELECT success, error_reason FROM due_diligence_runs "
            "WHERE fight_url = ? AND report_run_id = ?",
            ["/fight/search-fail", "run-search-fail"],
        ).fetchone()

        assert row is not None
        assert row[0] is False
        assert "Search failed" in row[1]


# ── Logging tests ─────────────────────────────────────────────────────────────


class TestLogging:
    """Every invocation persists prompt_version, model_name, model_version, invoked_at."""

    def test_successful_run_logged(self, conn, checklist_context):
        """A successful invocation logs all required fields."""
        run_due_diligence(
            "/fight/log-success", "run-log-ok", checklist_context,
            _mock_llm_client(), _mock_search_client(), conn,
            prompt_version="v2",
            model_name="test-model",
            model_version="2.0",
        )

        row = conn.execute(
            "SELECT prompt_version, model_name, model_version, invoked_at, success "
            "FROM due_diligence_runs WHERE fight_url = ? AND report_run_id = ?",
            ["/fight/log-success", "run-log-ok"],
        ).fetchone()

        assert row is not None
        assert row[0] == "v2"
        assert row[1] == "test-model"
        assert row[2] == "2.0"
        assert row[3] is not None  # invoked_at present
        assert row[4] is True

    def test_failed_run_logged_with_all_fields(self, conn, checklist_context):
        """A failed invocation still logs prompt_version, model_name, model_version, invoked_at."""
        bad_llm = _mock_llm_client("garbage")

        with pytest.raises(DueDiligenceError):
            run_due_diligence(
                "/fight/log-fail", "run-log-fail", checklist_context,
                bad_llm, _mock_search_client(), conn,
                prompt_version="v3",
                model_name="fail-model",
                model_version="3.0",
            )

        row = conn.execute(
            "SELECT prompt_version, model_name, model_version, invoked_at, success "
            "FROM due_diligence_runs WHERE fight_url = ? AND report_run_id = ?",
            ["/fight/log-fail", "run-log-fail"],
        ).fetchone()

        assert row is not None
        assert row[0] == "v3"
        assert row[1] == "fail-model"
        assert row[2] == "3.0"
        assert row[3] is not None  # invoked_at present even on failure
        assert row[4] is False

    def test_invoked_at_is_utc_timestamp(self, conn, checklist_context):
        """invoked_at is a real timestamp, not null or default."""
        before = datetime.now(UTC).replace(tzinfo=None)

        run_due_diligence(
            "/fight/ts-check", "run-ts", checklist_context,
            _mock_llm_client(), _mock_search_client(), conn,
        )

        after = datetime.now(UTC).replace(tzinfo=None)

        row = conn.execute(
            "SELECT invoked_at FROM due_diligence_runs "
            "WHERE fight_url = ? AND report_run_id = ?",
            ["/fight/ts-check", "run-ts"],
        ).fetchone()

        # Stored as naive UTC — must fall between before and after
        invoked_at = row[0]
        assert invoked_at is not None
        assert before <= invoked_at <= after


# ── Evidence URL tests ────────────────────────────────────────────────────────


class TestEvidenceUrls:
    """Schema enforces at least one evidence_url per successful verdict."""

    def test_empty_evidence_urls_caught_as_failure(self, conn, checklist_context):
        """A verdict with zero evidence URLs fails validation and is logged as error."""
        no_evidence = json.dumps({
            "verdict": "CONFIRM",
            "confidence": 0.9,
            "evidence_urls": [],
            "summary": "Nothing found but confirming anyway",
            "checklist_findings": {
                "injury_news": {"present": False, "detail": "None", "source_url": ""},
            },
        })
        bad_llm = _mock_llm_client(no_evidence)

        with pytest.raises(DueDiligenceError, match="Invalid LLM response"):
            run_due_diligence(
                "/fight/no-evidence", "run-no-ev", checklist_context,
                bad_llm, _mock_search_client(), conn,
            )

        # Failure is logged, not silently dropped
        row = conn.execute(
            "SELECT success, error_reason FROM due_diligence_runs "
            "WHERE fight_url = ? AND report_run_id = ?",
            ["/fight/no-evidence", "run-no-ev"],
        ).fetchone()

        assert row is not None
        assert row[0] is False
        assert "evidence_urls" in row[1]

    def test_valid_evidence_urls_persisted(self, conn, checklist_context):
        """A verdict with valid evidence URLs is accepted and persisted."""
        multi_evidence = json.dumps({
            "verdict": "QUALIFY",
            "confidence": 0.7,
            "evidence_urls": [
                "https://mmajunkie.com/injury-report",
                "https://espn.com/ufc-news",
            ],
            "summary": "Minor weight concern noted.",
            "checklist_findings": {
                "weight_cut_concern": {
                    "present": True,
                    "detail": "Fighter missed weight at last event",
                    "source_url": "https://mmajunkie.com/injury-report",
                },
            },
        })
        good_llm = _mock_llm_client(multi_evidence)

        verdict = run_due_diligence(
            "/fight/good-ev", "run-good-ev", checklist_context,
            good_llm, _mock_search_client(), conn,
        )

        assert len(verdict.evidence_urls) == 2
        assert verdict.verdict == DueDiligenceVerdictType.QUALIFY

    def test_single_evidence_url_accepted(self, conn, checklist_context):
        """The minimum case: exactly one evidence URL passes."""
        verdict = run_due_diligence(
            "/fight/one-url", "run-one-url", checklist_context,
            _mock_llm_client(), _mock_search_client(), conn,
        )

        assert len(verdict.evidence_urls) == 1


# ── Verdict persistence tests ─────────────────────────────────────────────────


class TestVerdictPersistence:
    """Validated verdicts are persisted to due_diligence_verdicts table."""

    def test_verdict_written_to_storage(self, conn, checklist_context):
        """A successful verdict appears in the due_diligence_verdicts table."""
        run_due_diligence(
            "/fight/persist", "run-persist", checklist_context,
            _mock_llm_client(), _mock_search_client(), conn,
        )

        row = conn.execute(
            "SELECT verdict, confidence, evidence_urls, summary, prompt_version, "
            "model_name, model_version, invoked_at "
            "FROM due_diligence_verdicts WHERE fight_url = ? AND report_run_id = ?",
            ["/fight/persist", "run-persist"],
        ).fetchone()

        assert row is not None
        assert row[0] == "CONFIRM"
        assert row[1] == 0.85
        assert "mmajunkie" in row[2]  # JSON-encoded evidence_urls
        assert row[3] == "No material concerns found. Both fighters healthy."

    def test_checklist_findings_persisted_as_json(self, conn, checklist_context):
        """Checklist findings are stored as a JSON string that round-trips."""
        run_due_diligence(
            "/fight/checklist", "run-checklist", checklist_context,
            _mock_llm_client(), _mock_search_client(), conn,
        )

        row = conn.execute(
            "SELECT checklist_json FROM due_diligence_verdicts "
            "WHERE fight_url = ? AND report_run_id = ?",
            ["/fight/checklist", "run-checklist"],
        ).fetchone()

        assert row is not None
        parsed = json.loads(row[0])
        assert "injury_news" in parsed
        assert parsed["injury_news"]["present"] is False
