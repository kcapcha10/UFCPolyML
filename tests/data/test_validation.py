"""Validation-suite tests: every invariant catches its planted violation, era
scoping decides pass/fail (D12), and quarantine reflects current state only."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from ufc_edge.data import storage
from ufc_edge.data.validation import invariants
from ufc_edge.data.validation.runner import build_report, run_validation

LABEL_START = date(2010, 1, 1)
SCRAPED_AT = datetime(2026, 7, 5)

OLD_EVENT = ("http://e/old", "UFC 50", date(2004, 10, 22), "Atlantic City", SCRAPED_AT)
NEW_EVENT = ("http://e/new", "UFC 300", date(2024, 4, 13), "Las Vegas", SCRAPED_AT)


@pytest.fixture()
def conn(tmp_path):
    with storage.get_connection(path=tmp_path / "test.duckdb") as connection:
        yield connection


def _insert_event(conn, event):
    conn.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?)", list(event))


def _insert_fighter(conn, url, name="Fighter", height=180.0, reach=185.0, weight=77.0):
    conn.execute(
        "INSERT INTO fighters VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [url, name, height, weight, reach, "Orthodox", date(1990, 1, 1), SCRAPED_AT],
    )


def _insert_fight(
    conn,
    fight_url,
    event_url,
    fighter_a="http://f/a",
    fighter_b="http://f/b",
    winner="http://f/a",
    ending_round=3,
    time_format="3 Rnd (5-5-5)",
):
    conn.execute(
        "INSERT INTO fights VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            fight_url,
            event_url,
            fighter_a,
            fighter_b,
            winner,
            "Decision - Unanimous",
            ending_round,
            "5:00",
            time_format,
            "Herb Dean",
            "Welterweight",
            SCRAPED_AT,
        ],
    )


def _insert_totals(conn, fight_url, fighter_url, sig_landed=30, sig_att=60, control=120):
    conn.execute(
        "INSERT INTO fight_totals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            fight_url,
            fighter_url,
            0,
            sig_landed,
            sig_att,
            sig_landed,
            sig_att,
            2,
            4,
            0,
            0,
            control,
            SCRAPED_AT,
        ],
    )


def _insert_round(conn, fight_url, fighter_url, rnd, sig_landed=10, sig_att=20):
    conn.execute(
        "INSERT INTO round_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            fight_url,
            fighter_url,
            rnd,
            0,
            sig_landed,
            sig_att,
            sig_landed,
            sig_att,
            1 if rnd == 1 else 0 if rnd == 2 else 1,
            2 if rnd == 1 else 1,
            0,
            0,
            40,
            SCRAPED_AT,
        ],
    )


def _clean_fight(conn, event=NEW_EVENT, fight_url="http://ft/clean", with_rounds=True):
    """A fully consistent fight: totals equal the sum of three round rows.

    with_rounds=False builds a totals-only fight (early-era shape) — used when a
    test needs to corrupt totals without also cascading a ROUND_SUM_MISMATCH.
    """
    _insert_event(conn, event)
    _insert_fighter(conn, "http://f/a", "Alpha")
    _insert_fighter(conn, "http://f/b", "Bravo")
    _insert_fight(conn, fight_url, event[0])
    for fighter in ("http://f/a", "http://f/b"):
        _insert_totals(conn, fight_url, fighter)
        if with_rounds:
            for rnd in (1, 2, 3):
                _insert_round(conn, fight_url, fighter, rnd)


def test_clean_database_passes(conn):
    _clean_fight(conn)
    report = run_validation(conn, LABEL_START)
    assert report.passed
    assert report.total_violations == 0


def test_landed_gt_attempted_fails_in_label_universe(conn):
    _clean_fight(conn)
    conn.execute(
        "UPDATE fight_totals SET takedowns_landed = 9, takedowns_attempted = 4 "
        "WHERE fighter_url = 'http://f/a'"
    )
    report = run_validation(conn, LABEL_START)
    assert not report.passed
    assert report.counts_by_reason[invariants.LANDED_GT_ATTEMPTED] == 1
    quarantined = conn.execute("SELECT reason_code FROM validation_quarantine").fetchall()
    assert (invariants.LANDED_GT_ATTEMPTED,) in quarantined


def test_pre_cutoff_violation_is_report_only(conn):
    _clean_fight(conn, event=OLD_EVENT, fight_url="http://ft/old", with_rounds=False)
    conn.execute(
        "UPDATE fight_totals SET takedowns_landed = 9, takedowns_attempted = 4 "
        "WHERE fighter_url = 'http://f/a'"
    )
    report = run_validation(conn, LABEL_START)
    assert report.passed  # old-era violation: visible, quarantined, non-blocking
    assert report.pre_cutoff_violations == 1
    assert conn.execute("SELECT COUNT(*) FROM validation_quarantine").fetchone()[0] == 1


def test_round_sum_mismatch_detected(conn):
    _clean_fight(conn)
    conn.execute(
        "UPDATE round_stats SET significant_strikes_landed = 11 "
        "WHERE fighter_url = 'http://f/a' AND round = 2"
    )
    found = invariants.round_sums_mismatch_totals(conn)
    assert len(found) == 1
    assert "significant_strikes_landed" in found[0].detail


def test_fight_without_round_rows_is_not_a_violation(conn):
    """Early-era absence of round data is missingness, not corruption."""
    _insert_event(conn, OLD_EVENT)
    _insert_fighter(conn, "http://f/a")
    _insert_fighter(conn, "http://f/b")
    _insert_fight(conn, "http://ft/norounds", OLD_EVENT[0])
    _insert_totals(conn, "http://ft/norounds", "http://f/a")
    assert invariants.round_sums_mismatch_totals(conn) == []


def test_orphan_row_is_undated_and_fails(conn):
    _clean_fight(conn)
    _insert_totals(conn, "http://ft/ghost", "http://f/a")  # no such fight
    report = run_validation(conn, LABEL_START)
    assert not report.passed
    assert report.undated_violations == 1


def test_self_fight_and_foreign_winner_detected(conn):
    _insert_event(conn, NEW_EVENT)
    _insert_fighter(conn, "http://f/a")
    _insert_fighter(conn, "http://f/b")
    _insert_fight(conn, "http://ft/self", NEW_EVENT[0], fighter_b="http://f/a")
    _insert_fight(conn, "http://ft/badwin", NEW_EVENT[0], winner="http://f/zzz")
    assert len(invariants.self_fight(conn)) == 1
    assert len(invariants.winner_not_in_fight(conn)) == 1


def test_ending_round_beyond_schedule_detected(conn):
    _insert_event(conn, NEW_EVENT)
    _insert_fight(conn, "http://ft/r9", NEW_EVENT[0], ending_round=9)
    found = invariants.ending_round_invalid(conn)
    assert len(found) == 1


def test_fighter_measurement_dated_by_last_fight(conn):
    """A parse-error reach on an active fighter must trip the modern alarm."""
    _clean_fight(conn)
    conn.execute("UPDATE fighters SET reach_cm = 999 WHERE fighter_url = 'http://f/a'")
    report = run_validation(conn, LABEL_START)
    assert not report.passed
    assert report.counts_by_reason[invariants.MEASUREMENT_OUT_OF_RANGE] == 1


def test_control_time_exceeding_duration_detected(conn):
    _clean_fight(conn)
    conn.execute(
        "UPDATE fight_totals SET control_time_seconds = 2000 WHERE fighter_url = 'http://f/a'"
    )
    found = invariants.control_time_exceeds_fight_duration(conn)
    assert len(found) == 1


def test_quarantine_reflects_current_state_only(conn):
    _clean_fight(conn, with_rounds=False)
    conn.execute("UPDATE fight_totals SET takedowns_landed = 9 WHERE fighter_url = 'http://f/a'")
    run_validation(conn, LABEL_START)
    assert conn.execute("SELECT COUNT(*) FROM validation_quarantine").fetchone()[0] == 1
    conn.execute("UPDATE fight_totals SET takedowns_landed = 2 WHERE fighter_url = 'http://f/a'")
    report = run_validation(conn, LABEL_START)
    assert report.passed
    assert conn.execute("SELECT COUNT(*) FROM validation_quarantine").fetchone()[0] == 0


def test_report_era_partition_is_exhaustive():
    """Every violation lands in exactly one of the three era buckets."""
    from ufc_edge.data.validation.schemas import Violation

    violations = [
        Violation(
            table_name="t", row_key="a", reason_code="X", detail="", event_date=date(2024, 1, 1)
        ),
        Violation(
            table_name="t", row_key="b", reason_code="X", detail="", event_date=date(2005, 1, 1)
        ),
        Violation(table_name="t", row_key="c", reason_code="Y", detail="", event_date=None),
    ]
    report = build_report(violations, LABEL_START)
    assert (
        report.label_universe_violations + report.pre_cutoff_violations + report.undated_violations
        == report.total_violations
        == 3
    )
    assert not report.passed
