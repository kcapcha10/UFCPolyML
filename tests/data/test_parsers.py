"""Parser tests against saved ufcstats fixtures. No network access.

Fixtures captured from a completed event (UFC Fight Night: Adesanya vs. Pyfer)
so the stat tables are populated. See tests/fixtures/.
"""

from __future__ import annotations

from datetime import datetime

from ufc_edge.data.ufcstats import parsers

SCRAPED_AT = datetime(2026, 6, 13, 0, 0, 0)
EVENT_URL = "http://www.ufcstats.com/event-details/5c38639f860a5542"
FIGHT_URL = "http://www.ufcstats.com/fight-details/85e94a6c071fd9fa"
FIGHTER_URL = "http://www.ufcstats.com/fighter-details/1338e2c7480bdf9e"


# ── Events list ───────────────────────────────────────────────────────────────


def test_events_list_extracts_event_urls(events_list_html):
    urls = parsers.parse_events_list(events_list_html)
    assert len(urls) > 100  # the completed-events page lists the full history
    assert all("event-details" in u for u in urls)
    assert len(urls) == len(set(urls))  # deduped


# ── Event detail ──────────────────────────────────────────────────────────────


def test_parse_event_metadata_and_fights(event_detail_html):
    event, fight_urls = parsers.parse_event(event_detail_html, EVENT_URL, SCRAPED_AT)
    assert event.event_url == EVENT_URL
    assert event.name  # non-empty title
    assert event.date is not None
    assert event.location
    assert len(fight_urls) > 0
    assert all("fight-details" in u for u in fight_urls)


# ── Fight detail ──────────────────────────────────────────────────────────────


def test_parse_fight_core_fields(fight_detail_html):
    fight, totals, rounds, sig = parsers.parse_fight(
        fight_detail_html, FIGHT_URL, EVENT_URL, SCRAPED_AT
    )
    assert fight.fight_url == FIGHT_URL
    assert fight.event_url == EVENT_URL
    assert fight.fighter_a_url and fight.fighter_b_url
    assert fight.fighter_a_url != fight.fighter_b_url
    # The winner must be one of the two fighters (this fight had a decisive result).
    assert fight.winner_url in {fight.fighter_a_url, fight.fighter_b_url}
    assert fight.method  # e.g. "KO/TKO"
    assert fight.ending_round >= 1


def test_parse_fight_totals_two_fighters(fight_detail_html):
    _, totals, _, _ = parsers.parse_fight(fight_detail_html, FIGHT_URL, EVENT_URL, SCRAPED_AT)
    assert len(totals) == 2
    for row in totals:
        assert row.significant_strikes_attempted >= row.significant_strikes_landed
        assert row.total_strikes_attempted >= row.total_strikes_landed


def test_parse_round_stats_numbered_from_one(fight_detail_html):
    fight, _, rounds, _ = parsers.parse_fight(fight_detail_html, FIGHT_URL, EVENT_URL, SCRAPED_AT)
    assert len(rounds) > 0
    round_numbers = {r.round for r in rounds}
    assert min(round_numbers) == 1
    # Two fighters per round, rounds run 1..ending_round.
    assert max(round_numbers) == fight.ending_round
    assert len(rounds) == 2 * fight.ending_round


def test_parse_sig_breakdown_targets_sum_sanely(fight_detail_html):
    _, _, _, sig = parsers.parse_fight(fight_detail_html, FIGHT_URL, EVENT_URL, SCRAPED_AT)
    assert len(sig) > 0
    full_rows = [s for s in sig if s.round == 0]  # round 0 = full-fight aggregate
    assert len(full_rows) == 2
    for row in full_rows:
        assert row.head_attempted >= row.head_landed
        assert row.distance_attempted >= row.distance_landed


# ── Fighter detail ────────────────────────────────────────────────────────────


def test_parse_fighter_attributes(fighter_detail_html):
    fighter = parsers.parse_fighter(fighter_detail_html, FIGHTER_URL, SCRAPED_AT)
    assert fighter.fighter_url == FIGHTER_URL
    assert fighter.name
    assert fighter.height_cm and 150 < fighter.height_cm < 220
    assert fighter.reach_cm and 150 < fighter.reach_cm < 230
    assert fighter.stance  # e.g. "Switch"
    assert fighter.date_of_birth is not None


# ── Scalar parser helpers ─────────────────────────────────────────────────────


def test_landed_attempted_split():
    assert parsers._landed_attempted("24 of 42") == (24, 42)
    assert parsers._landed_attempted("---") == (0, 0)


def test_control_seconds():
    assert parsers._control_seconds("1:01") == 61
    assert parsers._control_seconds("0:00") == 0
    assert parsers._control_seconds("--") is None


def test_height_to_cm():
    assert parsers._height_to_cm("6' 4\"") == 193.0  # 76 inches
    assert parsers._height_to_cm("--") is None
