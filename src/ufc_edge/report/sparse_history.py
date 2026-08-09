"""Sparse-history tagging for the mismatch report.

Fighters with few prior UFC bouts have thin data behind their prediction,
so we flag it rather than pretending the model is equally confident about
everyone. This module counts each fighter's completed UFC appearances
before a given fight and tags the pair as sparse when either count falls
below a configurable threshold.
"""

from __future__ import annotations

import duckdb

from ufc_edge.report.schemas import SparseHistoryResult

_PRIOR_FIGHTS_SQL = """
    SELECT COUNT(*) AS cnt
    FROM fights f
    JOIN events e ON f.event_url = e.event_url
    JOIN events cur_e ON cur_e.event_url = (
        SELECT event_url FROM fights WHERE fight_url = ?
    )
    WHERE (f.fighter_a_url = ? OR f.fighter_b_url = ?)
      AND e.date < cur_e.date
      AND f.fight_url != ?
"""


def _count_prior_fights(
    fighter_url: str,
    fight_url: str,
    conn: duckdb.DuckDBPyConnection,
) -> int:
    """Count a fighter's UFC bouts strictly before the given fight's event date.

    Excludes the fight itself and any same-card fights (the date < constraint
    removes same-event fights because they share the same date).
    """
    params = [fight_url, fighter_url, fighter_url, fight_url]
    result = conn.execute(_PRIOR_FIGHTS_SQL, params).fetchone()
    return result[0] if result else 0


def tag_sparse_history(
    fighter_a_url: str,
    fighter_b_url: str,
    fight_url: str,
    conn: duckdb.DuckDBPyConnection,
    threshold: int,
) -> SparseHistoryResult:
    """Determine whether a fight involves a fighter with sparse UFC history.

    A fight is tagged sparse when either fighter has fewer than `threshold`
    prior UFC bouts. The minimum count across both fighters drives the tag.
    """
    count_a = _count_prior_fights(fighter_a_url, fight_url, conn)
    count_b = _count_prior_fights(fighter_b_url, fight_url, conn)
    min_prior = min(count_a, count_b)
    return SparseHistoryResult(
        min_prior_ufc_fights=min_prior,
        sparse_history=min_prior < threshold,
    )
