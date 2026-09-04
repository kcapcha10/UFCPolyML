"""Tests for HistoricalFightLoader.

Validates that load_historical_fights correctly joins events, fights, fighters,
and fight_totals tables, excludes quarantined rows via anti-join, and returns
chronologically sorted HistoricalFight instances with all fields populated.
"""

from __future__ import annotations

from datetime import date, datetime

import duckdb
import pytest

from ufc_edge.data.ufcstats.storage import UFCSTATS_DDL
from ufc_edge.data.validation.quarantine import VALIDATION_DDL
from ufc_edge.features.contracts import HistoricalFight
from ufc_edge.features.loader import load_historical_fights

SCRAPED_AT = datetime(2026, 7, 1)


@pytest.fixture()
def conn():
    """In-memory DuckDB with real production schema."""
    connection = duckdb.connect(":memory:")
    for ddl in UFCSTATS_DDL + VALIDATION_DDL:
        connection.execute(ddl)
    yield connection
    connection.close()


def _insert_event(conn, event_url: str, name: str, event_date: date) -> None:
    conn.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
        [event_url, name, event_date, "Las Vegas", SCRAPED_AT],
    )


def _insert_fighter(
    conn,
    fighter_url: str,
    name: str = "Fighter",
    height_cm: float | None = 180.0,
    reach_cm: float | None = 185.0,
    stance: str | None = "Orthodox",
    dob: date | None = date(1990, 5, 15),
) -> None:
    conn.execute(
        "INSERT INTO fighters VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [fighter_url, name, height_cm, 77.0, reach_cm, stance, dob, SCRAPED_AT],
    )


def _insert_fight(
    conn,
    fight_url: str,
    event_url: str,
    fighter_a_url: str = "http://f/a",
    fighter_b_url: str = "http://f/b",
    winner_url: str | None = "http://f/a",
    method: str = "Decision - Unanimous",
    ending_round: int = 3,
    ending_time: str = "5:00",
    time_format: str = "3 Rnd (5-5-5)",
    weight_class: str = "Welterweight",
) -> None:
    conn.execute(
        "INSERT INTO fights VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            fight_url,
            event_url,
            fighter_a_url,
            fighter_b_url,
            winner_url,
            method,
            ending_round,
            ending_time,
            time_format,
            "Herb Dean",
            weight_class,
            SCRAPED_AT,
        ],
    )


def _insert_totals(
    conn,
    fight_url: str,
    fighter_url: str,
    knockdowns: int = 1,
    sig_landed: int = 45,
    sig_att: int = 90,
    total_landed: int = 60,
    total_att: int = 110,
    td_landed: int = 3,
    td_att: int = 5,
    sub_att: int = 1,
    reversals: int = 0,
    control_seconds: int = 180,
) -> None:
    conn.execute(
        "INSERT INTO fight_totals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            fight_url,
            fighter_url,
            knockdowns,
            sig_landed,
            sig_att,
            total_landed,
            total_att,
            td_landed,
            td_att,
            sub_att,
            reversals,
            control_seconds,
            SCRAPED_AT,
        ],
    )


def _quarantine_fight(conn, fight_url: str) -> None:
    conn.execute(
        "INSERT INTO validation_quarantine VALUES (?, ?, ?, ?, ?, ?)",
        ["fights", fight_url, "TEST_REASON", "test detail", date(2024, 1, 1), SCRAPED_AT],
    )


def _seed_complete_fight(
    conn,
    fight_url: str,
    event_url: str,
    event_name: str,
    event_date: date,
    fighter_a_url: str = "http://f/a",
    fighter_b_url: str = "http://f/b",
    winner_url: str | None = "http://f/a",
) -> None:
    """Insert a fully-joined fight: event, fighters, fight, and totals."""
    _insert_event(conn, event_url, event_name, event_date)
    _insert_fighter(conn, fighter_a_url, "Alpha")
    _insert_fighter(conn, fighter_b_url, "Bravo")
    _insert_fight(
        conn,
        fight_url,
        event_url,
        fighter_a_url=fighter_a_url,
        fighter_b_url=fighter_b_url,
        winner_url=winner_url,
    )
    _insert_totals(conn, fight_url, fighter_a_url)
    _insert_totals(conn, fight_url, fighter_b_url)


class TestBasicLoading:
    """Loader returns correct HistoricalFight instances from joined tables."""

    def test_empty_database_returns_empty_list(self, conn) -> None:
        result = load_historical_fights(conn)
        assert result == []

    def test_single_fight_all_fields_populated(self, conn) -> None:
        _insert_event(conn, "http://e/1", "UFC 300", date(2024, 4, 13))
        _insert_fighter(
            conn,
            "http://f/a",
            "Alpha",
            height_cm=183.0,
            reach_cm=188.0,
            stance="Orthodox",
            dob=date(1988, 3, 20),
        )
        _insert_fighter(
            conn,
            "http://f/b",
            "Bravo",
            height_cm=178.0,
            reach_cm=180.0,
            stance="Southpaw",
            dob=date(1991, 11, 5),
        )
        _insert_fight(
            conn,
            "http://ft/1",
            "http://e/1",
            method="KO/TKO",
            ending_round=2,
            ending_time="3:45",
            time_format="3 Rnd (5-5-5)",
            weight_class="Lightweight",
        )
        _insert_totals(conn, "http://ft/1", "http://f/a", knockdowns=2, sig_landed=50)
        _insert_totals(conn, "http://ft/1", "http://f/b", knockdowns=0, sig_landed=25)

        result = load_historical_fights(conn)

        assert len(result) == 1
        fight = result[0]
        assert isinstance(fight, HistoricalFight)
        assert fight.fight_url == "http://ft/1"
        assert fight.event_url == "http://e/1"
        assert fight.event_date == date(2024, 4, 13)
        assert fight.fighter_a_url == "http://f/a"
        assert fight.fighter_b_url == "http://f/b"
        assert fight.winner_url == "http://f/a"
        assert fight.method == "KO/TKO"
        assert fight.ending_round == 2
        assert fight.ending_time == "3:45"
        assert fight.time_format == "3 Rnd (5-5-5)"
        assert fight.weight_class == "Lightweight"

    def test_fighter_profiles_populated(self, conn) -> None:
        _insert_event(conn, "http://e/1", "UFC 300", date(2024, 4, 13))
        _insert_fighter(
            conn,
            "http://f/a",
            "Alpha",
            height_cm=183.0,
            reach_cm=188.0,
            stance="Orthodox",
            dob=date(1988, 3, 20),
        )
        _insert_fighter(
            conn,
            "http://f/b",
            "Bravo",
            height_cm=178.0,
            reach_cm=180.0,
            stance="Southpaw",
            dob=date(1991, 11, 5),
        )
        _insert_fight(conn, "http://ft/1", "http://e/1")
        _insert_totals(conn, "http://ft/1", "http://f/a")
        _insert_totals(conn, "http://ft/1", "http://f/b")

        result = load_historical_fights(conn)
        fight = result[0]

        assert fight.fighter_a_profile.fighter_url == "http://f/a"
        assert fight.fighter_a_profile.height_cm == 183.0
        assert fight.fighter_a_profile.reach_cm == 188.0
        assert fight.fighter_a_profile.stance == "Orthodox"
        assert fight.fighter_a_profile.dob == date(1988, 3, 20)

        assert fight.fighter_b_profile.fighter_url == "http://f/b"
        assert fight.fighter_b_profile.height_cm == 178.0
        assert fight.fighter_b_profile.reach_cm == 180.0
        assert fight.fighter_b_profile.stance == "Southpaw"
        assert fight.fighter_b_profile.dob == date(1991, 11, 5)

    def test_fight_totals_populated(self, conn) -> None:
        _insert_event(conn, "http://e/1", "UFC 300", date(2024, 4, 13))
        _insert_fighter(conn, "http://f/a")
        _insert_fighter(conn, "http://f/b")
        _insert_fight(conn, "http://ft/1", "http://e/1")
        _insert_totals(
            conn,
            "http://ft/1",
            "http://f/a",
            knockdowns=2,
            sig_landed=50,
            sig_att=80,
            total_landed=70,
            total_att=100,
            td_landed=4,
            td_att=6,
            sub_att=2,
            reversals=1,
            control_seconds=200,
        )
        _insert_totals(
            conn,
            "http://ft/1",
            "http://f/b",
            knockdowns=0,
            sig_landed=30,
            sig_att=60,
            total_landed=40,
            total_att=70,
            td_landed=1,
            td_att=3,
            sub_att=0,
            reversals=0,
            control_seconds=90,
        )

        result = load_historical_fights(conn)
        fight = result[0]

        assert fight.fighter_a_totals is not None
        assert fight.fighter_a_totals.knockdowns == 2
        assert fight.fighter_a_totals.sig_strikes_landed == 50
        assert fight.fighter_a_totals.sig_strikes_attempted == 80
        assert fight.fighter_a_totals.total_strikes_landed == 70
        assert fight.fighter_a_totals.total_strikes_attempted == 100
        assert fight.fighter_a_totals.takedowns_landed == 4
        assert fight.fighter_a_totals.takedowns_attempted == 6
        assert fight.fighter_a_totals.submissions_attempted == 2
        assert fight.fighter_a_totals.reversals == 1
        assert fight.fighter_a_totals.control_time_seconds == 200

        assert fight.fighter_b_totals is not None
        assert fight.fighter_b_totals.knockdowns == 0
        assert fight.fighter_b_totals.sig_strikes_landed == 30

    def test_fight_without_totals_returns_none(self, conn) -> None:
        _insert_event(conn, "http://e/1", "UFC 1", date(1993, 11, 12))
        _insert_fighter(conn, "http://f/a")
        _insert_fighter(conn, "http://f/b")
        _insert_fight(conn, "http://ft/1", "http://e/1")

        result = load_historical_fights(conn)

        assert len(result) == 1
        assert result[0].fighter_a_totals is None
        assert result[0].fighter_b_totals is None

    def test_draw_has_null_winner(self, conn) -> None:
        _insert_event(conn, "http://e/1", "UFC 300", date(2024, 4, 13))
        _insert_fighter(conn, "http://f/a")
        _insert_fighter(conn, "http://f/b")
        _insert_fight(
            conn,
            "http://ft/1",
            "http://e/1",
            winner_url=None,
            method="Draw",
        )

        result = load_historical_fights(conn)

        assert result[0].winner_url is None


class TestChronologicalOrdering:
    """Loader returns fights sorted by event_date, then event_url, then fight_url."""

    def test_fights_sorted_by_event_date(self, conn) -> None:
        _insert_event(conn, "http://e/2", "UFC 301", date(2024, 5, 4))
        _insert_event(conn, "http://e/1", "UFC 300", date(2024, 4, 13))
        _insert_fighter(conn, "http://f/a")
        _insert_fighter(conn, "http://f/b")
        _insert_fight(conn, "http://ft/2", "http://e/2")
        _insert_fight(conn, "http://ft/1", "http://e/1")
        _insert_totals(conn, "http://ft/1", "http://f/a")
        _insert_totals(conn, "http://ft/1", "http://f/b")
        _insert_totals(conn, "http://ft/2", "http://f/a")
        _insert_totals(conn, "http://ft/2", "http://f/b")

        result = load_historical_fights(conn)

        assert result[0].event_date < result[1].event_date
        assert result[0].fight_url == "http://ft/1"
        assert result[1].fight_url == "http://ft/2"

    def test_same_event_sorted_by_fight_url(self, conn) -> None:
        _insert_event(conn, "http://e/1", "UFC 300", date(2024, 4, 13))
        _insert_fighter(conn, "http://f/a")
        _insert_fighter(conn, "http://f/b")
        _insert_fighter(conn, "http://f/c", "Charlie")
        _insert_fight(conn, "http://ft/b", "http://e/1", "http://f/a", "http://f/b")
        _insert_fight(conn, "http://ft/a", "http://e/1", "http://f/a", "http://f/c")

        result = load_historical_fights(conn)

        assert result[0].fight_url == "http://ft/a"
        assert result[1].fight_url == "http://ft/b"


class TestQuarantineExclusion:
    """Quarantined fights are excluded from the result set via anti-join."""

    def test_quarantined_fight_excluded(self, conn) -> None:
        _seed_complete_fight(
            conn,
            "http://ft/good",
            "http://e/1",
            "UFC 300",
            date(2024, 4, 13),
            fighter_a_url="http://f/a",
            fighter_b_url="http://f/b",
        )
        _insert_fighter(conn, "http://f/c", "Charlie")
        _insert_fighter(conn, "http://f/d", "Delta")
        _insert_fight(
            conn,
            "http://ft/bad",
            "http://e/1",
            fighter_a_url="http://f/c",
            fighter_b_url="http://f/d",
            winner_url="http://f/c",
        )
        _insert_totals(conn, "http://ft/bad", "http://f/c")
        _insert_totals(conn, "http://ft/bad", "http://f/d")
        _quarantine_fight(conn, "http://ft/bad")

        result = load_historical_fights(conn)

        assert len(result) == 1
        assert result[0].fight_url == "http://ft/good"

    def test_non_fight_quarantine_does_not_exclude(self, conn) -> None:
        """Quarantine entries for other tables don't affect fight loading."""
        _seed_complete_fight(
            conn,
            "http://ft/1",
            "http://e/1",
            "UFC 300",
            date(2024, 4, 13),
        )
        conn.execute(
            "INSERT INTO validation_quarantine VALUES (?, ?, ?, ?, ?, ?)",
            ["fight_totals", "http://ft/1", "BAD_TOTALS", "detail", date(2024, 4, 13), SCRAPED_AT],
        )

        result = load_historical_fights(conn)

        assert len(result) == 1

    def test_all_quarantined_returns_empty(self, conn) -> None:
        _seed_complete_fight(
            conn,
            "http://ft/1",
            "http://e/1",
            "UFC 300",
            date(2024, 4, 13),
        )
        _quarantine_fight(conn, "http://ft/1")

        result = load_historical_fights(conn)

        assert result == []


class TestBoutOrder:
    """bout_order is always None until the scraper extension persists it."""

    def test_bout_order_is_none(self, conn) -> None:
        _seed_complete_fight(
            conn,
            "http://ft/1",
            "http://e/1",
            "UFC 300",
            date(2024, 4, 13),
        )

        result = load_historical_fights(conn)

        assert result[0].bout_order is None


class TestMultipleFightsIntegration:
    """Integration test with multiple events and fights."""

    def test_multi_event_multi_fight_load(self, conn) -> None:
        _insert_event(conn, "http://e/1", "UFC 300", date(2024, 4, 13))
        _insert_event(conn, "http://e/2", "UFC 301", date(2024, 5, 4))
        _insert_fighter(conn, "http://f/a", "Alpha")
        _insert_fighter(conn, "http://f/b", "Bravo")
        _insert_fighter(conn, "http://f/c", "Charlie")
        _insert_fighter(conn, "http://f/d", "Delta")

        _insert_fight(conn, "http://ft/1", "http://e/1", "http://f/a", "http://f/b")
        _insert_fight(conn, "http://ft/2", "http://e/1", "http://f/c", "http://f/d")
        _insert_fight(conn, "http://ft/3", "http://e/2", "http://f/a", "http://f/c")

        _insert_totals(conn, "http://ft/1", "http://f/a")
        _insert_totals(conn, "http://ft/1", "http://f/b")
        _insert_totals(conn, "http://ft/2", "http://f/c")
        _insert_totals(conn, "http://ft/2", "http://f/d")
        _insert_totals(conn, "http://ft/3", "http://f/a")
        _insert_totals(conn, "http://ft/3", "http://f/c")

        _quarantine_fight(conn, "http://ft/2")

        result = load_historical_fights(conn)

        assert len(result) == 2
        assert result[0].fight_url == "http://ft/1"
        assert result[1].fight_url == "http://ft/3"
        assert result[0].event_date <= result[1].event_date
