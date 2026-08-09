"""Search-augmented LLM due-diligence runner for flagged signals.

Invoked only on signals that pass the magnitude gate. Runs a fixed
pre-fight checklist (injury news, weight-cut concern, short-notice
replacement, camp change, other material news) and returns a structured
verdict: CONFIRM, QUALIFY, or VETO.

The model only annotates a flagged signal with what it found — it never
decides to hide or suppress the signal, because the system has no track
record yet to trust it with that decision. Verdicts are informational:
every signal survives to the paper-signal log regardless of what the LLM
concludes.

Idempotent per (fight_url, report_run_id): a second call returns the
persisted verdict without invoking the LLM again.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Protocol

import duckdb
from pydantic import ValidationError

from ufc_edge.report.schemas import (
    ChecklistFindings,
    DueDiligenceVerdict,
    DueDiligenceVerdictType,
)
from ufc_edge.report.storage import write_verdict

# ── Protocols for injected dependencies ───────────────────────────────────────


def _utcnow() -> datetime:
    """Return current UTC time as a naive datetime (DuckDB-compatible)."""
    return datetime.now(UTC).replace(tzinfo=None)


class LLMClient(Protocol):
    """Callable that sends a prompt and returns a structured JSON string.

    Accepts the assembled prompt (search context + checklist instructions)
    and returns the raw LLM response text (expected to be valid JSON matching
    the verdict schema). Raises on timeout or provider errors.
    """

    def __call__(self, prompt: str) -> str: ...


class SearchClient(Protocol):
    """Callable that runs a search query and returns evidence URLs.

    Accepts a query string (typically fighter names + event context) and
    returns a list of URLs from which the LLM can draw its conclusions.
    """

    def __call__(self, query: str) -> list[str]: ...


# ── Internal helpers ──────────────────────────────────────────────────────────


def _check_existing_verdict(
    conn: duckdb.DuckDBPyConnection,
    fight_url: str,
    report_run_id: str,
) -> DueDiligenceVerdict | None:
    """Return the persisted verdict for this (fight, run) if one exists."""
    row = conn.execute(
        """
        SELECT verdict, confidence, evidence_urls, summary, checklist_json,
               prompt_version, model_name, model_version, invoked_at
        FROM due_diligence_verdicts
        WHERE fight_url = ? AND report_run_id = ?
        """,
        [fight_url, report_run_id],
    ).fetchone()

    if row is None:
        return None

    return DueDiligenceVerdict(
        fight_url=fight_url,
        report_run_id=report_run_id,
        verdict=DueDiligenceVerdictType(row[0]),
        confidence=row[1],
        evidence_urls=json.loads(row[2]),
        summary=row[3],
        checklist_findings=ChecklistFindings.model_validate_json(row[4]),
        prompt_version=row[5],
        model_name=row[6],
        model_version=row[7],
        invoked_at=row[8],
    )


def _log_run(
    conn: duckdb.DuckDBPyConnection,
    *,
    fight_url: str,
    report_run_id: str,
    prompt_version: str,
    model_name: str,
    model_version: str,
    invoked_at: datetime,
    response_latency_ms: int | None,
    success: bool,
    error_reason: str | None,
) -> None:
    """Append a row to due_diligence_runs for observability."""
    run_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO due_diligence_runs
            (run_id, fight_url, report_run_id, prompt_version, model_name,
             model_version, invoked_at, response_latency_ms, success, error_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            run_id,
            fight_url,
            report_run_id,
            prompt_version,
            model_name,
            model_version,
            invoked_at,
            response_latency_ms,
            success,
            error_reason,
        ],
    )


def _build_prompt(checklist_context: dict[str, str], evidence_urls: list[str]) -> str:
    """Assemble the checklist prompt from context and search results."""
    fighter_a = checklist_context.get("fighter_a", "Unknown")
    fighter_b = checklist_context.get("fighter_b", "Unknown")
    event_date = checklist_context.get("event_date", "Unknown")

    evidence_block = "\n".join(f"- {url}" for url in evidence_urls) if evidence_urls else "None"

    return (
        f"Pre-fight due-diligence checklist for {fighter_a} vs {fighter_b} "
        f"({event_date}).\n\n"
        f"Evidence sources:\n{evidence_block}\n\n"
        "Evaluate each category and respond with valid JSON:\n"
        "- injury_news\n- weight_cut_concern\n- short_notice_replacement\n"
        "- camp_change\n- other_material_news\n\n"
        "Return JSON with keys: verdict (CONFIRM/QUALIFY/VETO), confidence (0-1), "
        "evidence_urls (list), summary (str), checklist_findings (object with each "
        "category having present/detail/source_url)."
    )


def _parse_llm_response(
    raw_response: str,
    fight_url: str,
    report_run_id: str,
    prompt_version: str,
    model_name: str,
    model_version: str,
    invoked_at: datetime,
) -> DueDiligenceVerdict:
    """Parse and validate the LLM JSON response into a frozen verdict model.

    Raises ValidationError if the response doesn't conform to the schema
    (including the at-least-one-evidence-url requirement).
    """
    data = json.loads(raw_response)

    return DueDiligenceVerdict(
        fight_url=fight_url,
        report_run_id=report_run_id,
        verdict=DueDiligenceVerdictType(data["verdict"]),
        confidence=data["confidence"],
        evidence_urls=data["evidence_urls"],
        summary=data["summary"],
        checklist_findings=ChecklistFindings.model_validate(data["checklist_findings"]),
        prompt_version=prompt_version,
        model_name=model_name,
        model_version=model_version,
        invoked_at=invoked_at,
    )


# ── Public API ────────────────────────────────────────────────────────────────


class DueDiligenceError(Exception):
    """Raised when the LLM invocation fails or returns an unparseable response."""


def run_due_diligence(
    fight_url: str,
    report_run_id: str,
    checklist_context: dict[str, str],
    llm_client: LLMClient,
    search_client: SearchClient,
    conn: duckdb.DuckDBPyConnection,
    *,
    prompt_version: str = "v1",
    model_name: str = "claude-sonnet-4-20250514",
    model_version: str = "1.0",
) -> DueDiligenceVerdict:
    """Run due-diligence on a flagged signal. Idempotent per (fight_url, report_run_id).

    Checks storage for an existing verdict before invoking the LLM. If one
    exists, returns it directly without any network calls. Otherwise runs
    the search + LLM pipeline, validates and persists the result.

    On LLM failure or malformed response, logs the attempt with error_reason
    and raises DueDiligenceError — the caller records a null verdict row so
    the failure is visible rather than silently dropped.
    """
    # Idempotency: return persisted verdict if already computed
    existing = _check_existing_verdict(conn, fight_url, report_run_id)
    if existing is not None:
        return existing

    invoked_at = _utcnow()

    # Search phase: gather evidence URLs for the LLM prompt
    search_query = (
        f"{checklist_context.get('fighter_a', '')} "
        f"{checklist_context.get('fighter_b', '')} "
        f"UFC {checklist_context.get('event_date', '')}"
    )

    try:
        evidence_urls = search_client(search_query)
    except Exception as exc:
        elapsed_ms = int((_utcnow() - invoked_at).total_seconds() * 1000)
        _log_run(
            conn,
            fight_url=fight_url,
            report_run_id=report_run_id,
            prompt_version=prompt_version,
            model_name=model_name,
            model_version=model_version,
            invoked_at=invoked_at,
            response_latency_ms=elapsed_ms,
            success=False,
            error_reason=f"Search failed: {exc}",
        )
        raise DueDiligenceError(f"Search client failed: {exc}") from exc

    # LLM phase: build prompt and invoke
    prompt = _build_prompt(checklist_context, evidence_urls)

    try:
        raw_response = llm_client(prompt)
    except Exception as exc:
        elapsed_ms = int((_utcnow() - invoked_at).total_seconds() * 1000)
        _log_run(
            conn,
            fight_url=fight_url,
            report_run_id=report_run_id,
            prompt_version=prompt_version,
            model_name=model_name,
            model_version=model_version,
            invoked_at=invoked_at,
            response_latency_ms=elapsed_ms,
            success=False,
            error_reason=f"LLM invocation failed: {exc}",
        )
        raise DueDiligenceError(f"LLM client failed: {exc}") from exc

    # Parse and validate the response against the frozen schema
    try:
        verdict = _parse_llm_response(
            raw_response,
            fight_url=fight_url,
            report_run_id=report_run_id,
            prompt_version=prompt_version,
            model_name=model_name,
            model_version=model_version,
            invoked_at=invoked_at,
        )
    except (json.JSONDecodeError, ValidationError, KeyError, ValueError) as exc:
        elapsed_ms = int((_utcnow() - invoked_at).total_seconds() * 1000)
        _log_run(
            conn,
            fight_url=fight_url,
            report_run_id=report_run_id,
            prompt_version=prompt_version,
            model_name=model_name,
            model_version=model_version,
            invoked_at=invoked_at,
            response_latency_ms=elapsed_ms,
            success=False,
            error_reason=f"Response validation failed: {exc}",
        )
        raise DueDiligenceError(f"Invalid LLM response: {exc}") from exc

    # Persist the validated verdict and log the successful run
    elapsed_ms = int((_utcnow() - invoked_at).total_seconds() * 1000)
    write_verdict(conn, verdict)
    _log_run(
        conn,
        fight_url=fight_url,
        report_run_id=report_run_id,
        prompt_version=prompt_version,
        model_name=model_name,
        model_version=model_version,
        invoked_at=invoked_at,
        response_latency_ms=elapsed_ms,
        success=True,
        error_reason=None,
    )

    return verdict
