"""Fixture tests for per-field Kaggle admission validation."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from ufc_edge.features.kaggle_validator import (
    AMBIGUOUS_JOIN_KEY,
    CROSS_CHECK_LOW_MATCH_RATE,
    DUPLICATE_JOIN_KEY,
    LEAKAGE_COLUMNS_REQUIRED,
    LEAKAGE_DETECTED,
    MISSING_JOIN_KEY,
    NONFINITE_LEAKAGE_CORRELATION,
    NONFINITE_LEAKAGE_INPUT,
    PROVENANCE_INCOMPLETE,
    cross_check_field,
    validate_field,
    write_validation_report,
)

PROVENANCE = {
    "dataset_version": "2026.01",
    "update_cadence": "annual",
    "author": "Known Author",
    "license": "CC BY 4.0",
    "methodology": "Documented UFC record extraction.",
}


def _rows(values: list[int], post_fight_values: list[int]) -> list[dict[str, object]]:
    return [
        {
            "fighter_url": f"https://example.test/fighters/{index}",
            "fight_url": f"https://example.test/fights/{index}",
            "pre_ufc_wins": value,
            "post_fight_column": post_fight_values[index],
        }
        for index, value in enumerate(values)
    ]


def test_clean_field_passes_and_writes_json_report(tmp_path: Path) -> None:
    kaggle_rows = _rows([1, 2, 3, 4], [4, 1, 3, 2])
    ufc_rows = _rows([1, 2, 3, 4], [4, 1, 3, 2])

    report = validate_field(
        field_name="pre_ufc_wins",
        source_dataset="ufc-history",
        kaggle_rows=kaggle_rows,
        ufc_rows=ufc_rows,
        provenance=PROVENANCE,
        post_fight_columns=["post_fight_column"],
        validation_date=date(2026, 8, 31),
    )

    assert report.passed is True
    report_path = write_validation_report(report, tmp_path)
    assert report_path == tmp_path / "pre_ufc_wins.json"
    assert json.loads(report_path.read_text())["source_dataset"] == "ufc-history"


def test_known_leaky_field_fails_with_reason() -> None:
    rows = _rows([1, 2, 3, 4], [1, 2, 3, 4])

    report = validate_field(
        field_name="pre_ufc_wins",
        source_dataset="ufc-history",
        kaggle_rows=rows,
        ufc_rows=rows,
        provenance=PROVENANCE,
        post_fight_columns=["post_fight_column"],
    )

    assert report.passed is False
    assert LEAKAGE_DETECTED in report.reason_codes


def test_low_match_rate_field_fails_with_reason() -> None:
    kaggle_rows = _rows([1, 2, 3, 4], [4, 1, 3, 2])
    ufc_rows = _rows([1, 9, 8, 7], [4, 1, 3, 2])

    report = validate_field(
        field_name="pre_ufc_wins",
        source_dataset="ufc-history",
        kaggle_rows=kaggle_rows,
        ufc_rows=ufc_rows,
        provenance=PROVENANCE,
        match_rate_threshold=0.75,
    )

    assert report.passed is False
    assert CROSS_CHECK_LOW_MATCH_RATE in report.reason_codes


def test_incomplete_provenance_rejects_without_mutating_rows() -> None:
    kaggle_rows = _rows([1, 2, 3, 4], [4, 1, 3, 2])
    original_rows = [row.copy() for row in kaggle_rows]
    provenance = {**PROVENANCE, "license": ""}

    report = validate_field(
        field_name="pre_ufc_wins",
        source_dataset="ufc-history",
        kaggle_rows=kaggle_rows,
        ufc_rows=kaggle_rows,
        provenance=provenance,
    )

    assert report.passed is False
    assert PROVENANCE_INCOMPLETE in report.reason_codes
    assert kaggle_rows == original_rows


def test_missing_leakage_columns_reject_through_admission_path() -> None:
    rows = _rows([1, 2, 3, 4], [4, 1, 3, 2])

    report = validate_field(
        field_name="pre_ufc_wins",
        source_dataset="ufc-history",
        kaggle_rows=rows,
        ufc_rows=rows,
        provenance=PROVENANCE,
    )

    assert report.passed is False
    assert LEAKAGE_COLUMNS_REQUIRED in report.reason_codes


def test_blank_leakage_column_rejects_through_admission_path() -> None:
    rows = _rows([1, 2, 3, 4], [4, 1, 3, 2])

    report = validate_field(
        field_name="pre_ufc_wins",
        source_dataset="ufc-history",
        kaggle_rows=rows,
        ufc_rows=rows,
        provenance=PROVENANCE,
        post_fight_columns=["   "],
    )

    assert report.passed is False
    assert LEAKAGE_COLUMNS_REQUIRED in report.reason_codes


def test_nonfinite_leakage_input_rejects_through_admission_path() -> None:
    rows = _rows([1, 2, 3, 4], [1, 2, 3, 4])
    rows[2]["post_fight_column"] = float("nan")

    report = validate_field(
        field_name="pre_ufc_wins",
        source_dataset="ufc-history",
        kaggle_rows=rows,
        ufc_rows=rows,
        provenance=PROVENANCE,
        post_fight_columns=["post_fight_column"],
    )

    assert report.passed is False
    assert NONFINITE_LEAKAGE_INPUT in report.reason_codes


def test_nonfinite_leakage_correlation_rejects_through_admission_path() -> None:
    rows = [
        {
            "fighter_url": "fighter-a",
            "pre_ufc_wins": 1.0e308,
            "post_fight_column": 1.0e308,
        },
        {
            "fighter_url": "fighter-b",
            "pre_ufc_wins": 1.0e308,
            "post_fight_column": -1.0e308,
        },
    ]

    report = validate_field(
        field_name="pre_ufc_wins",
        source_dataset="ufc-history",
        kaggle_rows=rows,
        ufc_rows=rows,
        provenance=PROVENANCE,
        post_fight_columns=["post_fight_column"],
    )

    assert report.passed is False
    assert NONFINITE_LEAKAGE_CORRELATION in report.reason_codes


@pytest.mark.parametrize("join_key_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_numeric_join_key_rejects_through_admission_path(
    join_key_value: float,
) -> None:
    rows = [
        {
            "join_id": join_key_value,
            "candidate": 1,
            "post_fight_column": 4,
        },
        {
            "join_id": 1.0,
            "candidate": 2,
            "post_fight_column": 1,
        },
        {
            "join_id": 2.0,
            "candidate": 3,
            "post_fight_column": 3,
        },
        {
            "join_id": 3.0,
            "candidate": 4,
            "post_fight_column": 2,
        },
    ]

    report = validate_field(
        field_name="candidate",
        source_dataset="fixture",
        kaggle_rows=rows,
        ufc_rows=rows,
        provenance=PROVENANCE,
        post_fight_columns=["post_fight_column"],
        join_key="join_id",
    )

    assert report.cross_check.reason_codes == (AMBIGUOUS_JOIN_KEY,)


def test_requested_join_key_is_not_replaced_by_fallback() -> None:
    candidate_rows = [
        {
            "fight_url": "fight-1",
            "candidate": 1,
        },
        {
            "fight_url": "fight-2",
            "candidate": 2,
        },
    ]
    source_rows = [
        {
            "fight_url": "fight-1",
            "candidate": 1,
        },
        {
            "fight_url": "fight-2",
            "candidate": 2,
        },
    ]

    result = cross_check_field(
        candidate_rows,
        source_rows,
        "candidate",
        join_key="fighter_url",
    )

    assert result.reason_codes == (MISSING_JOIN_KEY,)


def test_missing_requested_join_key_rejects_without_fallback() -> None:
    rows = [
        {
            "fight_url": "fight-1",
            "candidate": 1,
        }
    ]

    result = cross_check_field(rows, rows, "candidate", join_key="fighter_url")

    assert result.passed is False
    assert result.reason_codes == (MISSING_JOIN_KEY,)


def test_duplicate_source_join_key_rejects_conflicting_values() -> None:
    candidate_rows = [
        {
            "fighter_url": "fighter-a",
            "candidate": 1,
        }
    ]
    source_rows = [
        {
            "fighter_url": "fighter-a",
            "candidate": 1,
        },
        {
            "fighter_url": "fighter-a",
            "candidate": 2,
        },
    ]

    result = cross_check_field(candidate_rows, source_rows, "candidate")

    assert result.passed is False
    assert result.reason_codes == (DUPLICATE_JOIN_KEY,)


def test_unhashable_join_key_rejects_as_ambiguous() -> None:
    rows = [
        {
            "fighter_url": ["fighter-a"],
            "candidate": 1,
        }
    ]

    result = cross_check_field(rows, rows, "candidate")

    assert result.passed is False
    assert result.reason_codes == (AMBIGUOUS_JOIN_KEY,)
