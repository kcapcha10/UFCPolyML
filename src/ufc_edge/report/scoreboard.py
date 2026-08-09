"""Verdict scoreboard: tracks whether due-diligence verdicts were correct.

Records fight outcomes against past CONFIRM / QUALIFY / VETO verdicts so
the due-diligence model's reliability can be measured with real outcomes
instead of assumed. A running tally lets the system answer "when the model
said CONFIRM, how often was it right?" and "when it said VETO, did the
signal-favored fighter actually lose?"

Correctness semantics:
  - CONFIRM outcome_correct = True when the signal-favored fighter won.
    The model confirmed the signal was worth trusting, and the outcome
    validated that trust.
  - VETO outcome_correct = True when the signal-favored fighter LOST.
    The model warned the signal was unreliable, and the outcome proved
    the warning justified.
  - QUALIFY uses the same logic as CONFIRM (it supported the signal with
    caveats, so a win for the favored fighter still counts as correct).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import duckdb

from ufc_edge.report.schemas import (
    DueDiligenceVerdictType,
    ScoreboardEntry,
)
from ufc_edge.report.storage import write_scoreboard_entry


@dataclass(frozen=True)
class VerdictStats:
    """Aggregate stats for a single verdict type."""

    count: int
    correct: int
    incorrect: int
    win_rate: float
    mean_mismatch_magnitude: float | None


@dataclass(frozen=True)
class ScoreboardSummary:
    """Per-verdict-type aggregates from the resolved scoreboard entries."""

    by_verdict: dict[str, VerdictStats]
    total_resolved: int


def _compute_outcome_correct(
    verdict: DueDiligenceVerdictType,
    actual_winner_url: str,
    signal_favored_url: str,
) -> bool:
    """Determine whether the verdict turned out to be correct.

    For CONFIRM and QUALIFY: the model endorsed the signal, so correctness
    means the signal-favored fighter actually won.

    For VETO: the model warned the signal was unreliable, so correctness
    means the signal-favored fighter did NOT win (the warning was justified).
    """
    signal_favored_won = actual_winner_url == signal_favored_url

    if verdict == DueDiligenceVerdictType.VETO:
        return not signal_favored_won
    return signal_favored_won


def update_scoreboard(
    fight_url: str,
    report_run_id: str,
    verdict: DueDiligenceVerdictType,
    mismatch_at_signal: float | None,
    actual_winner_url: str,
    signal_favored_url: str,
    conn: duckdb.DuckDBPyConnection,
) -> ScoreboardEntry:
    """Record a resolved fight outcome against its due-diligence verdict.

    Computes whether the verdict was correct based on who actually won, then
    persists an entry to the scoreboard table. This enables tracking the
    due-diligence model's real-world accuracy over time.

    Args:
        fight_url: The resolved fight's URL identifier.
        report_run_id: The report run that produced the original signal.
        verdict: The due-diligence verdict that was issued (CONFIRM/QUALIFY/VETO).
        mismatch_at_signal: Model-vs-market mismatch magnitude when the signal fired.
        actual_winner_url: Fighter URL of the actual fight winner.
        signal_favored_url: Fighter URL the signal said was undervalued.
        conn: DuckDB connection with report DDL initialized.

    Returns:
        The persisted ScoreboardEntry with fight_resolved=True and outcome set.
    """
    outcome_correct = _compute_outcome_correct(
        verdict, actual_winner_url, signal_favored_url
    )

    entry = ScoreboardEntry(
        fight_url=fight_url,
        report_run_id=report_run_id,
        verdict=verdict,
        mismatch_at_signal=mismatch_at_signal,
        fight_resolved=True,
        outcome_correct=outcome_correct,
        resolved_at=datetime.now(),
    )

    write_scoreboard_entry(conn, entry)
    return entry


def query_scoreboard(conn: duckdb.DuckDBPyConnection) -> ScoreboardSummary:
    """Compute per-verdict-type aggregates from all resolved scoreboard entries.

    Returns counts, win rates (fraction of correct outcomes), and mean absolute
    mismatch magnitude grouped by verdict type. Only resolved entries (where
    fight_resolved is true) are included.
    """
    rows = conn.execute(
        """
        SELECT
            verdict,
            COUNT(*) AS cnt,
            SUM(CASE WHEN outcome_correct THEN 1 ELSE 0 END) AS correct,
            SUM(CASE WHEN NOT outcome_correct THEN 1 ELSE 0 END) AS incorrect,
            AVG(ABS(mismatch_at_signal)) AS mean_mismatch
        FROM verdict_scoreboard
        WHERE fight_resolved = true
        GROUP BY verdict
        ORDER BY verdict
        """
    ).fetchall()

    by_verdict: dict[str, VerdictStats] = {}
    total = 0

    for row in rows:
        verdict_str, cnt, correct, incorrect, mean_mismatch = row
        total += cnt
        win_rate = correct / cnt if cnt > 0 else 0.0
        by_verdict[verdict_str] = VerdictStats(
            count=cnt,
            correct=correct,
            incorrect=incorrect,
            win_rate=win_rate,
            mean_mismatch_magnitude=mean_mismatch,
        )

    return ScoreboardSummary(by_verdict=by_verdict, total_resolved=total)
