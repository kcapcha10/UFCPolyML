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
