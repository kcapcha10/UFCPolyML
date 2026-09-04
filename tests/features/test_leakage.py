"""Temporal leakage regression tests for the feature replay engine."""

from __future__ import annotations

from datetime import UTC, date, datetime

from ufc_edge.features.__main__ import (
    _build_components,
    _build_emitters,
    _build_feature_registry,
)
from ufc_edge.features.contracts import FeatureRow, FighterProfile, FightTotals, HistoricalFight
from ufc_edge.features.replay import replay

_FIXED_GENERATED_AT = datetime(2026, 1, 1, tzinfo=UTC)


_FIGHTER_PROFILES = {
    "fighter://a": FighterProfile(
        fighter_url="fighter://a",
        height_cm=180.0,
        reach_cm=185.0,
        stance="Orthodox",
        dob=date(1990, 1, 1),
    ),
    "fighter://b": FighterProfile(
        fighter_url="fighter://b",
        height_cm=175.0,
        reach_cm=180.0,
        stance="Southpaw",
        dob=date(1991, 2, 2),
    ),
    "fighter://c": FighterProfile(
        fighter_url="fighter://c",
        height_cm=183.0,
        reach_cm=188.0,
        stance="Orthodox",
        dob=date(1988, 3, 3),
    ),
    "fighter://d": FighterProfile(
        fighter_url="fighter://d",
        height_cm=178.0,
        reach_cm=181.0,
        stance="Switch",
        dob=date(1992, 4, 4),
    ),
}


def _make_totals(seed: int) -> FightTotals:
    """Create deterministic, non-empty totals for one fixture fight."""
    return FightTotals(
        knockdowns=seed % 3,
        sig_strikes_landed=40 + seed,
        sig_strikes_attempted=80 + seed,
        total_strikes_landed=70 + seed,
        total_strikes_attempted=120 + seed,
        takedowns_landed=seed % 4,
        takedowns_attempted=seed % 4 + 2,
        submissions_attempted=seed % 2,
        reversals=seed % 2,
        control_time_seconds=60 + seed * 10,
    )


def _make_fight(
    fight_url: str,
    event_number: int,
    event_date: date,
    fighter_a_url: str,
    fighter_b_url: str,
    winner_url: str,
    method: str,
    seed: int,
) -> HistoricalFight:
    """Create one chronological fixture fight with complete replay inputs."""
    return HistoricalFight(
        fight_url=fight_url,
        event_url=f"event://{event_number}",
        event_date=event_date,
        fighter_a_url=fighter_a_url,
        fighter_b_url=fighter_b_url,
        winner_url=winner_url,
        method=method,
        ending_round=1 if method == "KO/TKO" else 3,
        ending_time="2:30" if method == "KO/TKO" else "5:00",
        time_format="3 Rnd (5-5-5)",
        weight_class="Lightweight",
        bout_order=None,
        fighter_a_profile=_FIGHTER_PROFILES[fighter_a_url],
        fighter_b_profile=_FIGHTER_PROFILES[fighter_b_url],
        fighter_a_totals=_make_totals(seed),
        fighter_b_totals=_make_totals(seed + 1),
    )


def _fixture_fights() -> list[HistoricalFight]:
    """Return five event-separated fights with recurring fighters and outcomes."""
    return [
        _make_fight(
            "fight://1",
            1,
            date(2023, 1, 1),
            "fighter://a",
            "fighter://b",
            "fighter://a",
            "Decision",
            1,
        ),
        _make_fight(
            "fight://2",
            2,
            date(2023, 2, 1),
            "fighter://c",
            "fighter://d",
            "fighter://c",
            "KO/TKO",
            2,
        ),
        _make_fight(
            "fight://3",
            3,
            date(2023, 3, 1),
            "fighter://a",
            "fighter://c",
            "fighter://c",
            "Submission",
            3,
        ),
        _make_fight(
            "fight://4",
            4,
            date(2023, 4, 1),
            "fighter://b",
            "fighter://d",
            "fighter://d",
            "Decision",
            4,
        ),
        _make_fight(
            "fight://5",
            5,
            date(2023, 5, 1),
            "fighter://a",
            "fighter://d",
            "fighter://a",
            "KO/TKO",
            5,
        ),
    ]


def _run_replay(fights: list[HistoricalFight]) -> list[FeatureRow]:
    """Run the production feature assembly against the supplied fixture fights."""
    component_registry, experience_accumulator = _build_components()
    return replay(
        fights=fights,
        component_registry=component_registry,
        emitters=_build_emitters(),
        feature_registry=_build_feature_registry(),
        experience_accumulator=experience_accumulator,
    )


def _canonicalize_generated_at(rows: list[FeatureRow]) -> list[FeatureRow]:
    """Normalize replay timestamps so complete row comparisons remain meaningful."""
    return [row.model_copy(update={"generated_at": _FIXED_GENERATED_AT}) for row in rows]


def _rows_for_fight(rows: list[FeatureRow], fight_url: str) -> list[FeatureRow]:
    """Select both orientations for one fight without changing replay order."""
    return [row for row in rows if row.fight_url == fight_url]


def _assert_rows_identical(
    expected_rows: list[FeatureRow], actual_rows: list[FeatureRow], fight_url: str
) -> None:
    """Assert complete row equality and report metadata or feature differences."""
    differences: list[str] = []
    if len(expected_rows) != len(actual_rows):
        differences.append(f"row count expected={len(expected_rows)} actual={len(actual_rows)}")

    for index, (expected, actual) in enumerate(zip(expected_rows, actual_rows, strict=False)):
        if expected.model_dump_json() == actual.model_dump_json():
            continue

        for field in (
            "fight_url",
            "fighter_url",
            "event_url",
            "event_date",
            "opponent_url",
            "weight_class",
            "feature_version",
            "generated_at",
        ):
            expected_value = getattr(expected, field)
            actual_value = getattr(actual, field)
            if expected_value != actual_value:
                differences.append(
                    f"orientation {index} metadata {field}: "
                    f"expected={expected_value!r} actual={actual_value!r}"
                )

        feature_names = sorted(set(expected.features) | set(actual.features))
        for feature_name in feature_names:
            expected_value = expected.features.get(feature_name)
            actual_value = actual.features.get(feature_name)
            if expected_value != actual_value:
                differences.append(
                    f"orientation {index} feature {feature_name}: "
                    f"expected={expected_value!r} actual={actual_value!r}"
                )

    assert not differences, f"Deletion oracle mismatch for {fight_url}: {differences}"


def test_deletion_oracle() -> None:
    """A target fight must not depend on source data after its event."""
    fights = _fixture_fights()
    full_rows = _canonicalize_generated_at(_run_replay(fights))

    for target in fights:
        target_event_key = (target.event_date, target.event_url)
        # Canonicalize generated_at and retain only strict pre-event inputs plus the target.
        prior_and_target = [
            fight for fight in fights if (fight.event_date, fight.event_url) < target_event_key
        ] + [target]
        reduced_rows = _canonicalize_generated_at(_run_replay(prior_and_target))

        expected_target_rows = _rows_for_fight(full_rows, target.fight_url)
        actual_target_rows = _rows_for_fight(reduced_rows, target.fight_url)
        _assert_rows_identical(expected_target_rows, actual_target_rows, target.fight_url)
