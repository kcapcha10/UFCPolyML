"""Pure BeautifulSoup parsers for ufcstats.com pages.

Each public function takes raw HTML (and the source URL where needed) and returns
validated Pydantic models from `ufc_edge.data.ufcstats.schemas`. No network, no I/O — so the
parsers are unit-tested offline against saved fixtures in `tests/fixtures/`.

Parsing convention for ufcstats stat tables: every data `<td>` holds two `<p>`
elements (fighter A then fighter B). Count cells read "<landed> of <attempted>".
Columns are addressed by position, not header text, because ufcstats reuses
ambiguous header labels (e.g. "Td %" appears twice).
"""

from __future__ import annotations

import re
from datetime import date, datetime

from bs4 import BeautifulSoup, Tag

from ufc_edge.data.ufcstats.schemas import (
    Event,
    Fight,
    Fighter,
    FightTotals,
    RoundStats,
    SigStrikeBreakdown,
)

CM_PER_INCH = 2.54
KG_PER_LB = 0.45359237
INCHES_PER_FOOT = 12

# Column positions in the "Totals" stat tables (both full-fight and per-round).
_COL_KD = 1
_COL_SIG_STR = 2
_COL_TOTAL_STR = 4
_COL_TD = 5
_COL_SUB_ATT = 7
_COL_REV = 8
_COL_CTRL = 9

# Column positions in the "Significant Strikes" breakdown tables.
_COL_HEAD = 3
_COL_BODY = 4
_COL_LEG = 5
_COL_DISTANCE = 6
_COL_CLINCH = 7
_COL_GROUND = 8


# ── Small scalar parsers ──────────────────────────────────────────────────────


def _clean(text: str) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", text).strip()


def _to_int(text: str) -> int:
    """Parse an integer from a stat cell; '--' and blanks become 0."""
    digits = re.sub(r"[^0-9]", "", text)
    return int(digits) if digits else 0


def _landed_attempted(text: str) -> tuple[int, int]:
    """Split an 'X of Y' stat cell into (landed, attempted)."""
    match = re.search(r"(\d+)\s+of\s+(\d+)", text)
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def _control_seconds(text: str) -> int | None:
    """Parse a 'M:SS' control-time cell into seconds; '--' becomes None."""
    cleaned = _clean(text)
    match = re.match(r"(\d+):(\d{2})$", cleaned)
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def _height_to_cm(text: str) -> float | None:
    """Parse a height like \"6' 4\\\"\" into centimetres; '--' becomes None."""
    match = re.search(r"(\d+)'\s*(\d+)", text)
    if not match:
        return None
    inches = int(match.group(1)) * INCHES_PER_FOOT + int(match.group(2))
    return round(inches * CM_PER_INCH, 1)


def _reach_to_cm(text: str) -> float | None:
    """Parse a reach like '80\"' into centimetres; '--' becomes None."""
    match = re.search(r"(\d+)", text)
    return round(int(match.group(1)) * CM_PER_INCH, 1) if match else None


def _weight_to_kg(text: str) -> float | None:
    """Parse a weight like '185 lbs.' into kilograms; '--' becomes None."""
    match = re.search(r"(\d+)", text)
    return round(int(match.group(1)) * KG_PER_LB, 1) if match else None


def _parse_date(text: str, fmt: str) -> date | None:
    """Parse a date with the given strptime format; return None on failure."""
    try:
        return datetime.strptime(_clean(text), fmt).date()
    except ValueError:
        return None


# ── Events list ───────────────────────────────────────────────────────────────


def parse_events_list(html: str) -> list[str]:
    """Return all event-detail URLs from the completed-events listing page."""
    soup = BeautifulSoup(html, "lxml")
    urls = [
        anchor["href"]
        for anchor in soup.select("a.b-link.b-link_style_black")
        if "event-details" in anchor.get("href", "")
    ]
    # Dedupe while preserving order.
    return list(dict.fromkeys(urls))


# ── Event detail ──────────────────────────────────────────────────────────────


def parse_event(html: str, event_url: str, scraped_at: datetime) -> tuple[Event, list[str]]:
    """Parse an event page into an Event plus the list of its fight-detail URLs."""
    soup = BeautifulSoup(html, "lxml")
    event = Event(
        event_url=event_url,
        name=_event_name(soup),
        date=_event_field(soup, "Date") and _parse_date(_event_field(soup, "Date"), "%B %d, %Y"),
        location=_event_field(soup, "Location") or None,
        scraped_at=scraped_at,
    )
    fight_urls = [
        row["data-link"]
        for row in soup.select("tr.b-fight-details__table-row[data-link]")
        if "fight-details" in row.get("data-link", "")
    ]
    return event, list(dict.fromkeys(fight_urls))


def _event_name(soup: BeautifulSoup) -> str:
    """Extract the event title."""
    title = soup.select_one(".b-content__title-highlight")
    return _clean(title.get_text()) if title else ""


def _event_field(soup: BeautifulSoup, label: str) -> str:
    """Extract a labelled value from the event info box (e.g. 'Date', 'Location')."""
    for item in soup.select(".b-list__box-list-item"):
        text = _clean(item.get_text(" "))
        if text.startswith(f"{label}:"):
            return text.split(":", 1)[1].strip()
    return ""


# ── Fight detail ──────────────────────────────────────────────────────────────


def parse_fight(
    html: str, fight_url: str, event_url: str, scraped_at: datetime
) -> tuple[Fight, list[FightTotals], list[RoundStats], list[SigStrikeBreakdown]]:
    """Parse a fight page into the Fight plus its per-fighter stat rows.

    Returns (fight, totals_rows, round_rows, sig_breakdown_rows). Stat lists are
    empty for fights with no recorded statistics (e.g. upcoming bouts).
    """
    soup = BeautifulSoup(html, "lxml")
    fighter_urls = _fight_fighter_urls(soup)
    fight = Fight(
        fight_url=fight_url,
        event_url=event_url,
        fighter_a_url=fighter_urls[0],
        fighter_b_url=fighter_urls[1],
        winner_url=_winner_url(soup, fighter_urls),
        method=_fight_meta(soup, "Method"),
        ending_round=_to_int(_fight_meta(soup, "Round")),
        ending_time=_fight_meta(soup, "Time") or "0:00",
        time_format=_fight_meta(soup, "Time format"),
        referee=_fight_meta(soup, "Referee") or None,
        weight_class=_weight_class(soup) or None,
        scraped_at=scraped_at,
    )
    totals_tables = _stat_tables(soup, marker="KD")
    sig_tables = _stat_tables(soup, marker="Head")

    totals = _parse_fight_totals(totals_tables, fight_url, fighter_urls, scraped_at)
    rounds = _parse_round_stats(totals_tables, fight_url, fighter_urls, scraped_at)
    sig = _parse_sig_breakdowns(sig_tables, fight_url, fighter_urls, scraped_at)
    return fight, totals, rounds, sig


def _fight_fighter_urls(soup: BeautifulSoup) -> list[str]:
    """Return the two fighter-detail URLs, in displayed (A, B) order."""
    anchors = soup.select(".b-fight-details__person a.b-link")
    urls = [a["href"] for a in anchors if "fighter-details" in a.get("href", "")]
    return urls[:2]


def _winner_url(soup: BeautifulSoup, fighter_urls: list[str]) -> str | None:
    """Return the winner's fighter URL, or None for a draw / no-contest."""
    persons = soup.select(".b-fight-details__person")
    for person, url in zip(persons, fighter_urls, strict=False):
        status = person.select_one(".b-fight-details__person-status")
        if status and _clean(status.get_text()) == "W":
            return url
    return None


def _fight_meta(soup: BeautifulSoup, label: str) -> str:
    """Extract a labelled fight-detail value (Method / Round / Time / Referee...)."""
    for item in soup.select(".b-fight-details__text-item, .b-fight-details__text-item_first"):
        text = _clean(item.get_text(" "))
        if text.lower().startswith(f"{label.lower()}:"):
            return text.split(":", 1)[1].strip()
    return ""


def _weight_class(soup: BeautifulSoup) -> str:
    """Extract the weight-class / bout description from the fight title bar."""
    title = soup.select_one(".b-fight-details__fight-title")
    return _clean(title.get_text()) if title else ""


def _stat_tables(soup: BeautifulSoup, marker: str) -> list[Tag]:
    """Return stat tables whose header row contains `marker` (e.g. 'KD', 'Head')."""
    tables = []
    for table in soup.select("table"):
        headers = [_clean(th.get_text()) for th in table.select("thead th")]
        if any(marker in h for h in headers):
            tables.append(table)
    return tables


def _full_and_per_round(tables: list[Tag]) -> tuple[Tag | None, Tag | None]:
    """Split matching stat tables into (full-fight, per-round).

    The full-fight table has a single `<thead>`; the per-round table carries one
    extra `<thead>` per round (the 'Round N' labels).
    """
    full = next((t for t in tables if len(t.select("thead")) == 1), None)
    per_round = next((t for t in tables if len(t.select("thead")) > 1), None)
    return full, per_round


def _cell_pair(row: Tag, col: int) -> tuple[str, str]:
    """Return the (fighter A, fighter B) text for one column of a stat row."""
    cells = row.select("td")
    if col >= len(cells):
        return "", ""
    paragraphs = cells[col].select("p")
    texts = [_clean(p.get_text(" ")) for p in paragraphs]
    while len(texts) < 2:
        texts.append("")
    return texts[0], texts[1]


def _totals_row(
    fight_url: str,
    fighter_url: str,
    row: Tag,
    side: int,
    scraped_at: datetime,
    round_no: int | None,
) -> FightTotals | RoundStats:
    """Build a FightTotals (round_no None) or RoundStats row from one table row.

    `side` selects fighter A (0) or B (1) within each two-fighter cell.
    """
    sig_l, sig_a = _landed_attempted(_cell_pair(row, _COL_SIG_STR)[side])
    tot_l, tot_a = _landed_attempted(_cell_pair(row, _COL_TOTAL_STR)[side])
    td_l, td_a = _landed_attempted(_cell_pair(row, _COL_TD)[side])
    common = {
        "fight_url": fight_url,
        "fighter_url": fighter_url,
        "knockdowns": _to_int(_cell_pair(row, _COL_KD)[side]),
        "significant_strikes_landed": sig_l,
        "significant_strikes_attempted": sig_a,
        "total_strikes_landed": tot_l,
        "total_strikes_attempted": tot_a,
        "takedowns_landed": td_l,
        "takedowns_attempted": td_a,
        "submission_attempts": _to_int(_cell_pair(row, _COL_SUB_ATT)[side]),
        "reversals": _to_int(_cell_pair(row, _COL_REV)[side]),
        "control_time_seconds": _control_seconds(_cell_pair(row, _COL_CTRL)[side]),
        "scraped_at": scraped_at,
    }
    if round_no is None:
        return FightTotals(**common)
    return RoundStats(round=round_no, **common)


def _parse_fight_totals(
    tables: list[Tag], fight_url: str, fighter_urls: list[str], scraped_at: datetime
) -> list[FightTotals]:
    """Parse the full-fight totals table into one FightTotals per fighter."""
    full, _ = _full_and_per_round(tables)
    if full is None:
        return []
    row = full.select_one("tbody tr")
    if row is None:
        return []
    return [
        _totals_row(fight_url, fighter_urls[side], row, side, scraped_at, round_no=None)
        for side in range(2)
        if side < len(fighter_urls)
    ]


def _parse_round_stats(
    tables: list[Tag], fight_url: str, fighter_urls: list[str], scraped_at: datetime
) -> list[RoundStats]:
    """Parse the per-round totals table into RoundStats (round = row index + 1)."""
    _, per_round = _full_and_per_round(tables)
    if per_round is None:
        return []
    rows = per_round.select("tbody tr")
    stats: list[RoundStats] = []
    for index, row in enumerate(rows):
        for side in range(min(2, len(fighter_urls))):
            stats.append(
                _totals_row(
                    fight_url, fighter_urls[side], row, side, scraped_at, round_no=index + 1
                )
            )
    return stats


def _sig_row(
    fight_url: str, fighter_url: str, row: Tag, side: int, scraped_at: datetime, round_no: int
) -> SigStrikeBreakdown:
    """Build a SigStrikeBreakdown row from one significant-strikes table row."""
    head_l, head_a = _landed_attempted(_cell_pair(row, _COL_HEAD)[side])
    body_l, body_a = _landed_attempted(_cell_pair(row, _COL_BODY)[side])
    leg_l, leg_a = _landed_attempted(_cell_pair(row, _COL_LEG)[side])
    dist_l, dist_a = _landed_attempted(_cell_pair(row, _COL_DISTANCE)[side])
    clinch_l, clinch_a = _landed_attempted(_cell_pair(row, _COL_CLINCH)[side])
    ground_l, ground_a = _landed_attempted(_cell_pair(row, _COL_GROUND)[side])
    return SigStrikeBreakdown(
        fight_url=fight_url,
        fighter_url=fighter_url,
        round=round_no,
        head_landed=head_l,
        head_attempted=head_a,
        body_landed=body_l,
        body_attempted=body_a,
        leg_landed=leg_l,
        leg_attempted=leg_a,
        distance_landed=dist_l,
        distance_attempted=dist_a,
        clinch_landed=clinch_l,
        clinch_attempted=clinch_a,
        ground_landed=ground_l,
        ground_attempted=ground_a,
        scraped_at=scraped_at,
    )


def _parse_sig_breakdowns(
    tables: list[Tag], fight_url: str, fighter_urls: list[str], scraped_at: datetime
) -> list[SigStrikeBreakdown]:
    """Parse full-fight and per-round significant-strike breakdown tables."""
    full, per_round = _full_and_per_round(tables)
    rows: list[SigStrikeBreakdown] = []
    if full is not None and (row := full.select_one("tbody tr")) is not None:
        for side in range(min(2, len(fighter_urls))):
            # round=0 marks the full-fight aggregate (see SigStrikeBreakdown).
            rows.append(_sig_row(fight_url, fighter_urls[side], row, side, scraped_at, 0))
    if per_round is not None:
        for index, row in enumerate(per_round.select("tbody tr")):
            for side in range(min(2, len(fighter_urls))):
                rows.append(
                    _sig_row(fight_url, fighter_urls[side], row, side, scraped_at, index + 1)
                )
    return rows


# ── Fighter detail ────────────────────────────────────────────────────────────


def parse_fighter(html: str, fighter_url: str, scraped_at: datetime) -> Fighter:
    """Parse a fighter page into a Fighter model."""
    soup = BeautifulSoup(html, "lxml")
    fields = _fighter_fields(soup)
    return Fighter(
        fighter_url=fighter_url,
        name=_event_name(soup),
        height_cm=_height_to_cm(fields.get("Height", "")),
        weight_kg=_weight_to_kg(fields.get("Weight", "")),
        reach_cm=_reach_to_cm(fields.get("Reach", "")),
        stance=fields.get("STANCE") or None,
        date_of_birth=_parse_date(fields.get("DOB", ""), "%b %d, %Y"),
        scraped_at=scraped_at,
    )


def _fighter_fields(soup: BeautifulSoup) -> dict[str, str]:
    """Extract the 'Label: value' attribute box into a dict."""
    fields: dict[str, str] = {}
    for item in soup.select(".b-list__box-list-item"):
        text = _clean(item.get_text(" "))
        if ":" not in text:
            continue
        label, value = text.split(":", 1)
        value = value.strip()
        if value and value != "--":
            fields[label.strip()] = value
    return fields
