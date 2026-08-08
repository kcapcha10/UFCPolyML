"""Smoke tests for the synthetic feature-table fixture.

Validates structural invariants: row count, event count, outcome
distribution, NULL presence, and market column existence.
"""

from __future__ import annotations


def test_fixture_produces_50_fight_rows(feature_db):
    """The feature table must contain exactly 50 fight rows."""
    (count,) = feature_db.execute("SELECT COUNT(*) FROM features_v1").fetchone()
    assert count == 50


def test_fixture_has_10_distinct_events(feature_db):
    """Events must span exactly 10 distinct event URLs."""
    (count,) = feature_db.execute(
        "SELECT COUNT(DISTINCT event_url) FROM features_v1"
    ).fetchone()
    assert count == 10


def test_fixture_has_draws_and_nc(feature_db):
    """At least 2 draws and 1 NC exist for exclusion testing."""
    rows = feature_db.execute(
        "SELECT outcome, COUNT(*) as cnt FROM features_v1 "
        "WHERE outcome IN ('draw', 'nc') GROUP BY outcome ORDER BY outcome"
    ).fetchall()
    outcome_map = {row[0]: row[1] for row in rows}
    assert outcome_map.get("draw", 0) >= 2
    assert outcome_map.get("nc", 0) >= 1


def test_fixture_has_multiple_weight_classes(feature_db):
    """At least 3 distinct weight classes present."""
    (count,) = feature_db.execute(
        "SELECT COUNT(DISTINCT weight_class) FROM features_v1"
    ).fetchone()
    assert count >= 3


def test_fixture_has_null_values(feature_db):
    """Some feature columns contain NULLs (for missingness handling tests)."""
    (null_count,) = feature_db.execute(
        "SELECT COUNT(*) FROM features_v1 WHERE elo_rating_a IS NULL"
    ).fetchone()
    assert null_count > 0


def test_fixture_contains_market_column(feature_db):
    """The market-derived column exists (for rejection testing by assembler)."""
    cols = feature_db.execute("DESCRIBE features_v1").fetchall()
    col_names = {row[0] for row in cols}
    assert "opening_implied_prob" in col_names


def test_fixture_event_dates_are_chronological(feature_db):
    """Events are ordered chronologically (temporal splits depend on this)."""
    dates = feature_db.execute(
        "SELECT DISTINCT event_date FROM features_v1 ORDER BY event_date"
    ).fetchall()
    for i in range(1, len(dates)):
        assert dates[i][0] >= dates[i - 1][0]
