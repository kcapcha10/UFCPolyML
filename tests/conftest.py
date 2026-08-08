"""Shared pytest fixtures: saved ufcstats HTML loaded from tests/fixtures/."""

from __future__ import annotations

import pathlib

import pytest

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures"


def _load(name: str) -> str:
    return (FIXTURE_DIR / name).read_text()


@pytest.fixture
def events_list_html() -> str:
    return _load("events_list.html")


@pytest.fixture
def event_detail_html() -> str:
    return _load("event_detail.html")


@pytest.fixture
def fight_detail_html() -> str:
    return _load("fight_detail.html")


@pytest.fixture
def fighter_detail_html() -> str:
    return _load("fighter_detail.html")


# ---------------------------------------------------------------------------
# Synthetic feature-table fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def feature_db(tmp_path):
    """In-memory-style DuckDB with 50 synthetic fight rows for model/eval tests.

    Returns a DuckDB connection with a `features_v1` table containing:
    - 50 fights across 10 events
    - Multiple weight classes
    - 2 draws + 1 NC for exclusion testing
    - ~10% NULLs in feature columns
    - One market-derived column (opening_implied_prob) for rejection testing
    """
    import duckdb

    from tests.fixtures.fixture_features import generate_feature_table

    db_path = tmp_path / "features_test.duckdb"
    conn = duckdb.connect(str(db_path))
    generate_feature_table(conn)
    yield conn
    conn.close()
