"""Tests for sparse-history tagging.

Validates that fighters with few prior UFC bouts are correctly identified,
that the count excludes the current fight and same-card fights, and that
the threshold comparison drives the sparse flag.
"""

from __future__ import annotations

import duckdb
import pytest

from ufc_edge.report.sparse_history import tag_sparse_history


def _fight_row(
    fight_url: str,
    event_url: str,
    fighter_a: str,
    fighter_b: str,
    scraped: str,
) -> str:
    """Build a VALUES tuple for a fight row with sensible defaults."""
    return (
        f"('{fight_url}', '{event_url}', '{fighter_a}', '{fighter_b}', "
        f"NULL, 'Decision', 3, '5:00', '3 Rnd (5-5-5)', NULL, "
        f"'Lightweight', '{scraped}')"
    )


@pytest.fixture()
def conn() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with events and fights seeded with known history."""
    db = duckdb.connect(":memory:")
    db.execute("""
        CREATE TABLE events (
            event_url VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            date DATE NOT NULL,
            location VARCHAR,
            scraped_at TIMESTAMP NOT NULL
        )
    """)
    db.execute("""
        CREATE TABLE fights (
            fight_url VARCHAR PRIMARY KEY,
            event_url VARCHAR NOT NULL,
            fighter_a_url VARCHAR NOT NULL,
            fighter_b_url VARCHAR NOT NULL,
            winner_url VARCHAR,
            method VARCHAR NOT NULL,
            ending_round INTEGER NOT NULL,
            ending_time VARCHAR NOT NULL,
            time_format VARCHAR NOT NULL,
            referee VARCHAR,
            weight_class VARCHAR,
            scraped_at TIMESTAMP NOT NULL
        )
    """)

    # Four events in chronological order
    db.execute("""
        INSERT INTO events VALUES
            ('evt1', 'UFC 300', '2024-01-01', 'Vegas',
             '2024-01-02 00:00:00'),
            ('evt2', 'UFC 301', '2024-02-01', 'Vegas',
             '2024-02-02 00:00:00'),
            ('evt3', 'UFC 302', '2024-03-01', 'Vegas',
             '2024-03-02 00:00:00'),
            ('evt4', 'UFC 303', '2024-04-01', 'Vegas',
             '2024-04-02 00:00:00')
    """)

    # Fighter A ("sparse"): fights on evt1 and evt2 → 2 fights before evt4
    # Fighter B ("experienced"): fights on evt1 (x2), evt2 (x2), evt3 → 5
    fights_sql = f"""
        INSERT INTO fights VALUES
            {_fight_row('fight1', 'evt1', 'fighterA', 'fighterC',
                        '2024-01-02 00:00:00')},
            {_fight_row('fight2', 'evt2', 'fighterA', 'fighterC',
                        '2024-02-02 00:00:00')},
            {_fight_row('fight3', 'evt1', 'fighterB', 'fighterC',
                        '2024-01-02 00:00:00')},
            {_fight_row('fight4', 'evt2', 'fighterB', 'fighterC',
                        '2024-02-02 00:00:00')},
            {_fight_row('fight5', 'evt3', 'fighterB', 'fighterC',
                        '2024-03-02 00:00:00')},
            {_fight_row('fight6', 'evt1', 'fighterC', 'fighterB',
                        '2024-01-02 00:00:00')},
            {_fight_row('fight7', 'evt2', 'fighterC', 'fighterB',
                        '2024-02-02 00:00:00')},
            {_fight_row('target_fight', 'evt4', 'fighterA', 'fighterB',
                        '2024-04-02 00:00:00')}
    """
    db.execute(fights_sql)

    return db


def test_sparse_when_one_fighter_below_threshold(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """Fighter A has 2 prior, Fighter B has 5 → min=2, sparse=True."""
    result = tag_sparse_history(
        "fighterA", "fighterB", "target_fight", conn, threshold=3
    )

    assert result.min_prior_ufc_fights == 2
    assert result.sparse_history is True


def test_not_sparse_when_both_above_threshold(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """Both fighters with 4+ prior fights → sparse=False."""
    conn.execute("""
        INSERT INTO events VALUES
            ('evt0a', 'UFC 297', '2023-10-01', 'Vegas',
             '2023-10-02 00:00:00'),
            ('evt0b', 'UFC 298', '2023-11-01', 'Vegas',
             '2023-11-02 00:00:00')
    """)
    extra = f"""
        INSERT INTO fights VALUES
            {_fight_row('extra1', 'evt0a', 'fighterA', 'fighterC',
                        '2023-10-02 00:00:00')},
            {_fight_row('extra2', 'evt0b', 'fighterA', 'fighterC',
                        '2023-11-02 00:00:00')}
    """
    conn.execute(extra)

    # Fighter A now has 4 prior fights (2 original + 2 new), B has 5
    result = tag_sparse_history(
        "fighterA", "fighterB", "target_fight", conn, threshold=3
    )

    assert result.min_prior_ufc_fights == 4
    assert result.sparse_history is False


def test_count_excludes_current_fight(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """The fight being evaluated is never counted in either fighter's tally."""
    result = tag_sparse_history(
        "fighterA", "fighterB", "target_fight", conn, threshold=10
    )

    # Fighter A: fight1 (evt1) + fight2 (evt2) = 2, not 3 with target
    assert result.min_prior_ufc_fights == 2


def test_count_excludes_same_card_fights(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """Same-card fights are not counted as prior history.

    Fights on the same event are chronologically simultaneous, not prior,
    so the strict date < comparison naturally excludes them.
    """
    samecard = _fight_row(
        "samecard", "evt4", "fighterA", "fighterC", "2024-04-02 00:00:00"
    )
    conn.execute(f"INSERT INTO fights VALUES {samecard}")

    result = tag_sparse_history(
        "fighterA", "fighterB", "target_fight", conn, threshold=3
    )

    # Fighter A still has only 2 prior fights (evt1, evt2)
    assert result.min_prior_ufc_fights == 2
    assert result.sparse_history is True


def test_zero_prior_fights(conn: duckdb.DuckDBPyConnection) -> None:
    """A debuting fighter with no prior UFC bouts returns 0."""
    debut = _fight_row(
        "debut_fight", "evt4", "newcomer", "fighterB", "2024-04-02 00:00:00"
    )
    conn.execute(f"INSERT INTO fights VALUES {debut}")

    result = tag_sparse_history(
        "newcomer", "fighterB", "debut_fight", conn, threshold=3
    )

    assert result.min_prior_ufc_fights == 0
    assert result.sparse_history is True


def test_threshold_boundary_exactly_at_threshold(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """A fighter with exactly threshold fights is NOT flagged as sparse."""
    # Fighter A has 2 prior fights; set threshold=2 → 2 < 2 is False
    result = tag_sparse_history(
        "fighterA", "fighterB", "target_fight", conn, threshold=2
    )

    assert result.min_prior_ufc_fights == 2
    assert result.sparse_history is False
