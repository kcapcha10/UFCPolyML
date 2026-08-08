"""DuckDB DDL and append-only write functions for report tables.

Owns seven tables: market_fight_links, report_runs, paper_signals,
post_signal_snapshots, due_diligence_verdicts, due_diligence_runs,
and verdict_scoreboard. All writes are append-only (INSERT or ON CONFLICT
DO NOTHING); write_link raises on attempt to overwrite a MATCHED row.
"""

from __future__ import annotations

import json

import duckdb

from ufc_edge.report.schemas import (
    DueDiligenceVerdict,
    MarketFightLink,
    MatchStatus,
    PaperSignal,
    PostSignalSnapshot,
    ReportRun,
    ScoreboardEntry,
)

# ── DDL ───────────────────────────────────────────────────────────────────────

_CREATE_MARKET_FIGHT_LINKS = """
CREATE TABLE IF NOT EXISTS market_fight_links (
    fight_url       TEXT NOT NULL,
    token_id        TEXT NOT NULL,
    match_status    TEXT NOT NULL,
    match_method    TEXT,
    candidate_count INTEGER,
    matched_at      TIMESTAMP NOT NULL DEFAULT current_timestamp,
    reviewed_by     TEXT,
    PRIMARY KEY (fight_url, token_id)
);
"""

_CREATE_REPORT_RUNS = """
CREATE TABLE IF NOT EXISTS report_runs (
    report_run_id   TEXT PRIMARY KEY,
    as_of_timestamp TIMESTAMP NOT NULL,
    mlflow_run_id   TEXT NOT NULL,
    data_revision   TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    config_hash     TEXT NOT NULL,
    bout_count      INTEGER NOT NULL,
    flagged_count   INTEGER NOT NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT current_timestamp
);
"""

_CREATE_PAPER_SIGNALS = """
CREATE TABLE IF NOT EXISTS paper_signals (
    signal_id                TEXT PRIMARY KEY,
    report_run_id            TEXT NOT NULL REFERENCES report_runs(report_run_id),
    fight_url                TEXT NOT NULL,
    event_date               DATE,
    fighter_a_url            TEXT NOT NULL,
    fighter_b_url            TEXT NOT NULL,
    fighter_a_name           TEXT NOT NULL,
    fighter_b_name           TEXT NOT NULL,
    weight_class             TEXT,
    market_id                TEXT,
    token_id                 TEXT,
    snapshot_timestamp       TIMESTAMP,
    tick_id                  TEXT,
    p_model                  DOUBLE,
    p_market_mid             DOUBLE,
    best_bid                 DOUBLE,
    best_ask                 DOUBLE,
    best_bid_size            DOUBLE,
    best_ask_size            DOUBLE,
    mismatch                 DOUBLE,
    gate_verdict             TEXT,
    bucket_id                TEXT,
    bucket_n                 INTEGER,
    bucket_calibration_error DOUBLE,
    bucket_ci_lower          DOUBLE,
    bucket_ci_upper          DOUBLE,
    min_prior_ufc_fights     INTEGER,
    sparse_history           BOOLEAN NOT NULL DEFAULT false,
    match_status             TEXT NOT NULL,
    mlflow_run_id            TEXT NOT NULL,
    data_revision            TEXT NOT NULL,
    feature_version          TEXT NOT NULL,
    config_hash              TEXT NOT NULL,
    created_at               TIMESTAMP NOT NULL DEFAULT current_timestamp
);
"""

_CREATE_POST_SIGNAL_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS post_signal_snapshots (
    snapshot_id        TEXT PRIMARY KEY,
    signal_id          TEXT NOT NULL REFERENCES paper_signals(signal_id),
    token_id           TEXT NOT NULL,
    scheduled_offset   TEXT NOT NULL,
    scheduled_at       TIMESTAMP NOT NULL,
    actual_captured_at TIMESTAMP,
    status             TEXT NOT NULL DEFAULT 'PENDING',
    failure_reason     TEXT,
    best_bid           DOUBLE,
    best_ask           DOUBLE,
    best_bid_size      DOUBLE,
    best_ask_size      DOUBLE,
    mid_price          DOUBLE,
    captured_at        TIMESTAMP
);
"""

_CREATE_DUE_DILIGENCE_VERDICTS = """
CREATE TABLE IF NOT EXISTS due_diligence_verdicts (
    fight_url       TEXT NOT NULL,
    report_run_id   TEXT NOT NULL,
    verdict         TEXT NOT NULL,
    confidence      DOUBLE NOT NULL,
    evidence_urls   TEXT NOT NULL,
    summary         TEXT NOT NULL,
    checklist_json  TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    invoked_at      TIMESTAMP NOT NULL,
    PRIMARY KEY (fight_url, report_run_id)
);
"""

_CREATE_DUE_DILIGENCE_RUNS = """
CREATE TABLE IF NOT EXISTS due_diligence_runs (
    run_id              TEXT PRIMARY KEY,
    fight_url           TEXT NOT NULL,
    report_run_id       TEXT NOT NULL,
    prompt_version      TEXT NOT NULL,
    model_name          TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    invoked_at          TIMESTAMP NOT NULL,
    response_latency_ms INTEGER,
    success             BOOLEAN NOT NULL,
    error_reason        TEXT
);
"""

_CREATE_VERDICT_SCOREBOARD = """
CREATE TABLE IF NOT EXISTS verdict_scoreboard (
    fight_url          TEXT NOT NULL,
    report_run_id      TEXT NOT NULL,
    verdict            TEXT NOT NULL,
    mismatch_at_signal DOUBLE,
    fight_resolved     BOOLEAN NOT NULL DEFAULT false,
    outcome_correct    BOOLEAN,
    resolved_at        TIMESTAMP,
    PRIMARY KEY (fight_url, report_run_id)
);
"""

REPORT_DDL: list[str] = [
    _CREATE_MARKET_FIGHT_LINKS,
    _CREATE_REPORT_RUNS,
    _CREATE_PAPER_SIGNALS,
    _CREATE_POST_SIGNAL_SNAPSHOTS,
    _CREATE_DUE_DILIGENCE_VERDICTS,
    _CREATE_DUE_DILIGENCE_RUNS,
    _CREATE_VERDICT_SCOREBOARD,
]


# ── Write functions ───────────────────────────────────────────────────────────


class LinkOverwriteError(Exception):
    """Raised when attempting to overwrite a MATCHED market_fight_links row."""


def write_link(conn: duckdb.DuckDBPyConnection, link: MarketFightLink) -> None:
    """Write a market-fight link row. Raises on overwrite of a MATCHED row.

    A non-MATCHED row (NO_CANDIDATE, MULTIPLE_CANDIDATES, ...) may be
    replaced, since this is how a human confirmation moves it to MATCHED.
    Once a row is MATCHED, this function refuses to change it.
    """
    existing = conn.execute(
        "SELECT match_status FROM market_fight_links WHERE fight_url = ? AND token_id = ?",
        [link.fight_url, link.token_id],
    ).fetchone()

    if existing and existing[0] == MatchStatus.MATCHED:
        msg = (
            f"Cannot overwrite MATCHED link: "
            f"fight_url={link.fight_url!r}, token_id={link.token_id!r}"
        )
        raise LinkOverwriteError(msg)

    conn.execute(
        """
        INSERT INTO market_fight_links
            (fight_url, token_id, match_status, match_method,
             candidate_count, matched_at, reviewed_by)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (fight_url, token_id) DO UPDATE SET
            match_status = excluded.match_status,
            match_method = excluded.match_method,
            candidate_count = excluded.candidate_count,
            matched_at = excluded.matched_at,
            reviewed_by = excluded.reviewed_by
        """,
        [
            link.fight_url,
            link.token_id,
            link.match_status.value,
            link.match_method.value if link.match_method else None,
            link.candidate_count,
            link.matched_at,
            link.reviewed_by,
        ],
    )


def write_report_run(conn: duckdb.DuckDBPyConnection, run: ReportRun) -> None:
    """Append a report-run metadata row. Ignores duplicates by run ID."""
    conn.execute(
        """
        INSERT INTO report_runs
            (report_run_id, as_of_timestamp, mlflow_run_id, data_revision,
             feature_version, config_hash, bout_count, flagged_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (report_run_id) DO NOTHING
        """,
        [
            run.report_run_id,
            run.as_of_timestamp,
            run.mlflow_run_id,
            run.data_revision,
            run.feature_version,
            run.config_hash,
            run.bout_count,
            run.flagged_count,
            run.created_at,
        ],
    )


def write_paper_signal(conn: duckdb.DuckDBPyConnection, signal: PaperSignal) -> None:
    """Append a paper-signal row. Ignores duplicates by signal ID."""
    conn.execute(
        """
        INSERT INTO paper_signals
            (signal_id, report_run_id, fight_url, event_date,
             fighter_a_url, fighter_b_url, fighter_a_name, fighter_b_name,
             weight_class, market_id, token_id, snapshot_timestamp, tick_id,
             p_model, p_market_mid, best_bid, best_ask, best_bid_size, best_ask_size,
             mismatch, gate_verdict, bucket_id, bucket_n,
             bucket_calibration_error, bucket_ci_lower, bucket_ci_upper,
             min_prior_ufc_fights, sparse_history, match_status,
             mlflow_run_id, data_revision, feature_version, config_hash, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (signal_id) DO NOTHING
        """,
        [
            signal.signal_id,
            signal.report_run_id,
            signal.fight_url,
            signal.event_date,
            signal.fighter_a_url,
            signal.fighter_b_url,
            signal.fighter_a_name,
            signal.fighter_b_name,
            signal.weight_class,
            signal.market_id,
            signal.token_id,
            signal.snapshot_timestamp,
            signal.tick_id,
            signal.p_model,
            signal.p_market_mid,
            signal.best_bid,
            signal.best_ask,
            signal.best_bid_size,
            signal.best_ask_size,
            signal.mismatch,
            signal.gate_verdict.value if signal.gate_verdict else None,
            signal.bucket_id,
            signal.bucket_n,
            signal.bucket_calibration_error,
            signal.bucket_ci_lower,
            signal.bucket_ci_upper,
            signal.min_prior_ufc_fights,
            signal.sparse_history,
            signal.match_status.value,
            signal.mlflow_run_id,
            signal.data_revision,
            signal.feature_version,
            signal.config_hash,
            signal.created_at,
        ],
    )


def write_verdict(conn: duckdb.DuckDBPyConnection, verdict: DueDiligenceVerdict) -> None:
    """Append a due-diligence verdict row. Ignores duplicates by (fight_url, report_run_id)."""
    evidence_json = json.dumps(verdict.evidence_urls)
    checklist_json = verdict.checklist_findings.model_dump_json()

    conn.execute(
        """
        INSERT INTO due_diligence_verdicts
            (fight_url, report_run_id, verdict, confidence, evidence_urls,
             summary, checklist_json, prompt_version, model_name,
             model_version, invoked_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (fight_url, report_run_id) DO NOTHING
        """,
        [
            verdict.fight_url,
            verdict.report_run_id,
            verdict.verdict.value,
            verdict.confidence,
            evidence_json,
            verdict.summary,
            checklist_json,
            verdict.prompt_version,
            verdict.model_name,
            verdict.model_version,
            verdict.invoked_at,
        ],
    )


def write_post_signal_snapshot(
    conn: duckdb.DuckDBPyConnection, snapshot: PostSignalSnapshot
) -> None:
    """Append a post-signal snapshot row. Ignores duplicates by snapshot ID."""
    conn.execute(
        """
        INSERT INTO post_signal_snapshots
            (snapshot_id, signal_id, token_id, scheduled_offset, scheduled_at,
             actual_captured_at, status, failure_reason, best_bid, best_ask,
             best_bid_size, best_ask_size, mid_price, captured_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (snapshot_id) DO NOTHING
        """,
        [
            snapshot.snapshot_id,
            snapshot.signal_id,
            snapshot.token_id,
            snapshot.scheduled_offset.value,
            snapshot.scheduled_at,
            snapshot.actual_captured_at,
            snapshot.status.value,
            snapshot.failure_reason,
            snapshot.best_bid,
            snapshot.best_ask,
            snapshot.best_bid_size,
            snapshot.best_ask_size,
            snapshot.mid_price,
            snapshot.captured_at,
        ],
    )


def write_scoreboard_entry(
    conn: duckdb.DuckDBPyConnection, entry: ScoreboardEntry
) -> None:
    """Append a verdict-scoreboard entry. Ignores duplicates by (fight_url, report_run_id)."""
    conn.execute(
        """
        INSERT INTO verdict_scoreboard
            (fight_url, report_run_id, verdict, mismatch_at_signal,
             fight_resolved, outcome_correct, resolved_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (fight_url, report_run_id) DO NOTHING
        """,
        [
            entry.fight_url,
            entry.report_run_id,
            entry.verdict.value,
            entry.mismatch_at_signal,
            entry.fight_resolved,
            entry.outcome_correct,
            entry.resolved_at,
        ],
    )
