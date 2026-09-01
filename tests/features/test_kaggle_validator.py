"""Fixture tests for per-field Kaggle admission validation."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ufc_edge.features.kaggle_validator import (
    CROSS_CHECK_LOW_MATCH_RATE,
    LEAKAGE_DETECTED,
    PROVENANCE_INCOMPLETE,
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
