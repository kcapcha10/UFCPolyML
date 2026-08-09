"""HistoricalFightLoader — SQL reader for replay input.

Joins events, fights, fighters, and fight_totals tables, applies an anti-join
against validation_quarantine to exclude quarantined fights, and returns a
chronologically sorted list of HistoricalFight frozen model instances. This is
the sole data-ingestion entry point for the feature replay engine.

Note: round_stats and sig_strike_breakdowns joins are intentionally omitted.
No v1 emitter or state component consumes per-round data or positional/target
strike breakdowns (head/body/leg, distance/clinch/ground). The HistoricalFight
contract carries only FightTotals aggregates, which fully satisfy all 12 emitters
and 7 state components. If a future emitter requires round-level or breakdown
data, extend the HistoricalFight model and add the joins here.
"""

from __future__ import annotations

import duckdb

from ufc_edge.features.contracts import FighterProfile, FightTotals, HistoricalFight

_LOAD_QUERY = """
SELECT
    f.fight_url,
    f.event_url,
    e.date        AS event_date,
    f.fighter_a_url,
    f.fighter_b_url,
    f.winner_url,
    f.method,
    f.ending_round,
    f.ending_time,
    f.time_format,
    f.weight_class,
    -- Fighter A profile
    fa.fighter_url  AS fa_url,
    fa.height_cm    AS fa_height_cm,
    fa.reach_cm     AS fa_reach_cm,
    fa.stance       AS fa_stance,
    fa.date_of_birth AS fa_dob,
    -- Fighter B profile
    fb.fighter_url  AS fb_url,
    fb.height_cm    AS fb_height_cm,
    fb.reach_cm     AS fb_reach_cm,
    fb.stance       AS fb_stance,
    fb.date_of_birth AS fb_dob,
    -- Fighter A totals
    ta.knockdowns                    AS ta_kd,
    ta.significant_strikes_landed    AS ta_sig_landed,
    ta.significant_strikes_attempted AS ta_sig_att,
    ta.total_strikes_landed          AS ta_total_landed,
    ta.total_strikes_attempted       AS ta_total_att,
    ta.takedowns_landed              AS ta_td_landed,
    ta.takedowns_attempted           AS ta_td_att,
    ta.submission_attempts           AS ta_sub_att,
    ta.reversals                     AS ta_reversals,
    ta.control_time_seconds          AS ta_control,
    -- Fighter B totals
    tb.knockdowns                    AS tb_kd,
    tb.significant_strikes_landed    AS tb_sig_landed,
    tb.significant_strikes_attempted AS tb_sig_att,
    tb.total_strikes_landed          AS tb_total_landed,
    tb.total_strikes_attempted       AS tb_total_att,
    tb.takedowns_landed              AS tb_td_landed,
    tb.takedowns_attempted           AS tb_td_att,
    tb.submission_attempts           AS tb_sub_att,
    tb.reversals                     AS tb_reversals,
    tb.control_time_seconds          AS tb_control
FROM fights f
JOIN events e ON f.event_url = e.event_url
JOIN fighters fa ON f.fighter_a_url = fa.fighter_url
JOIN fighters fb ON f.fighter_b_url = fb.fighter_url
LEFT JOIN fight_totals ta ON f.fight_url = ta.fight_url AND f.fighter_a_url = ta.fighter_url
LEFT JOIN fight_totals tb ON f.fight_url = tb.fight_url AND f.fighter_b_url = tb.fighter_url
WHERE NOT EXISTS (
    SELECT 1 FROM validation_quarantine vq
    WHERE vq.table_name = 'fights'
      AND vq.row_key = f.fight_url
)
ORDER BY e.date, f.event_url, f.fight_url
"""


def load_historical_fights(conn: duckdb.DuckDBPyConnection) -> list[HistoricalFight]:
    """Load all non-quarantined fights in chronological order.

    Joins source tables and maps rows to HistoricalFight frozen models.
    Fights missing totals get None for the totals fields (expected for early-era).
    bout_order is always None until the scraper extension persists it.
    """
    rows = conn.execute(_LOAD_QUERY).fetchall()
    return [_row_to_historical_fight(row) for row in rows]


def _build_totals(
    kd: int | None,
    sig_landed: int | None,
    sig_att: int | None,
    total_landed: int | None,
    total_att: int | None,
    td_landed: int | None,
    td_att: int | None,
    sub_att: int | None,
    reversals: int | None,
    control: int | None,
) -> FightTotals | None:
    """Construct FightTotals from SQL row columns, returning None if no data."""
    if kd is None and sig_landed is None:
        return None
    return FightTotals(
        knockdowns=kd,
        sig_strikes_landed=sig_landed,
        sig_strikes_attempted=sig_att,
        total_strikes_landed=total_landed,
        total_strikes_attempted=total_att,
        takedowns_landed=td_landed,
        takedowns_attempted=td_att,
        submissions_attempted=sub_att,
        reversals=reversals,
        control_time_seconds=control,
    )


def _row_to_historical_fight(row: tuple) -> HistoricalFight:
    """Map a single SQL result row to a HistoricalFight instance."""
    return HistoricalFight(
        fight_url=row[0],
        event_url=row[1],
        event_date=row[2],
        fighter_a_url=row[3],
        fighter_b_url=row[4],
        winner_url=row[5],
        method=row[6],
        ending_round=row[7],
        ending_time=row[8],
        time_format=row[9],
        weight_class=row[10],
        bout_order=None,
        fighter_a_profile=FighterProfile(
            fighter_url=row[11],
            height_cm=row[12],
            reach_cm=row[13],
            stance=row[14],
            dob=row[15],
        ),
        fighter_b_profile=FighterProfile(
            fighter_url=row[16],
            height_cm=row[17],
            reach_cm=row[18],
            stance=row[19],
            dob=row[20],
        ),
        fighter_a_totals=_build_totals(*row[21:31]),
        fighter_b_totals=_build_totals(*row[31:41]),
    )
