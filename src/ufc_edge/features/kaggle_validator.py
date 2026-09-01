"""Per-field admission checks for external Kaggle feature candidates.

This module only inspects caller-provided rows. It does not fetch Kaggle or UFCStats
content and never mutates source rows. A candidate is admitted only when its overlap
check, provenance audit, and requested leakage checks all pass.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

DEFAULT_REPORT_DIRECTORY = Path("data/interim/kaggle_validation")
DEFAULT_MATCH_RATE_THRESHOLD = 0.8
DEFAULT_LEAKAGE_CORRELATION_THRESHOLD = 0.8

CROSS_CHECK_LOW_MATCH_RATE = "CROSS_CHECK_LOW_MATCH_RATE"
NO_CROSS_CHECK_OVERLAP = "NO_CROSS_CHECK_OVERLAP"
MISSING_CROSS_CHECK_FIELD = "MISSING_CROSS_CHECK_FIELD"
PROVENANCE_INCOMPLETE = "PROVENANCE_INCOMPLETE"
LEAKAGE_DETECTED = "LEAKAGE_DETECTED"
LEAKAGE_TEST_INCONCLUSIVE = "LEAKAGE_TEST_INCONCLUSIVE"

_REQUIRED_PROVENANCE_FIELDS = (
    "dataset_version",
    "update_cadence",
    "known_author",
    "license",
    "methodology",
)
Row = Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ProvenanceAudit:
    """The provenance fields required before a candidate can be admitted."""

    dataset_version: str
    update_cadence: str
    known_author: str
    license: str
    methodology: str

    @property
    def author(self) -> str:
        """Provide the short alias used by callers for the known author."""
        return self.known_author


@dataclass(frozen=True, slots=True)
class ProvenanceResult:
    """Outcome of validating a candidate dataset's provenance metadata."""

    passed: bool
    missing_fields: tuple[str, ...]
    reason_codes: tuple[str, ...]
    audit: ProvenanceAudit | None


@dataclass(frozen=True, slots=True)
class CrossCheckResult:
    """Overlap and value-match statistics for one candidate field."""

    field_name: str
    compared_count: int
    matching_count: int
    match_rate: float | None
    threshold: float
    passed: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LeakageResult:
    """Correlation results against the caller-selected post-fight columns."""

    field_name: str
    correlations: dict[str, float]
    threshold: float
    passed: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KaggleValidationReport:
    """Persistable decision record for one candidate field."""

    field_name: str
    source_dataset: str
    validation_date: date
    match_rate: float | None
    provenance_note: str
    passed: bool
    reason_codes: tuple[str, ...]
    cross_check: CrossCheckResult
    provenance: ProvenanceResult
    leakage: LeakageResult

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation of this report."""
        return asdict(self)


def audit_provenance(provenance: Mapping[str, object] | ProvenanceAudit) -> ProvenanceResult:
    """Validate the required provenance schema without contacting the source.

    Behavior/params/return: normalize a mapping or typed audit into a pass/fail result.
    Assumes/errors: blank values are treated as missing and malformed mappings fail closed.
    """
    if isinstance(provenance, ProvenanceAudit):
        values = {field: getattr(provenance, field) for field in _REQUIRED_PROVENANCE_FIELDS}
    else:
        values = dict(provenance)
        if "known_author" not in values and "author" in values:
            values["known_author"] = values["author"]

    missing_fields = tuple(
        field
        for field in _REQUIRED_PROVENANCE_FIELDS
        if not isinstance(values.get(field), str) or not values[field].strip()
    )
    if missing_fields:
        return ProvenanceResult(
            passed=False,
            missing_fields=missing_fields,
            reason_codes=(PROVENANCE_INCOMPLETE,),
            audit=None,
        )

    audit = ProvenanceAudit(**{field: str(values[field]) for field in _REQUIRED_PROVENANCE_FIELDS})
    return ProvenanceResult(passed=True, missing_fields=(), reason_codes=(), audit=audit)


def cross_check_field(
    kaggle_rows: Sequence[Row],
    ufc_rows: Sequence[Row],
    field_name: str,
    *,
    join_key: str = "fighter_url",
    ufc_field: str | None = None,
    sample_size: int | None = None,
    match_rate_threshold: float = DEFAULT_MATCH_RATE_THRESHOLD,
) -> CrossCheckResult:
    """Compare candidate values against overlapping UFC rows.

    Behavior/params/return: compare at most ``sample_size`` keyed, non-null pairs.
    Assumes/errors: matching is normalized for strings and tolerant for numeric values;
    invalid thresholds raise ``ValueError``.
    """
    _validate_threshold(match_rate_threshold, "match_rate_threshold")
    comparison_field = ufc_field or field_name
    candidate_rows = list(kaggle_rows[:sample_size] if sample_size is not None else kaggle_rows)
    source_rows = list(ufc_rows)
    resolved_join_key = _resolve_join_key(candidate_rows, source_rows, join_key)

    if resolved_join_key is None:
        return CrossCheckResult(
            field_name=field_name,
            compared_count=0,
            matching_count=0,
            match_rate=None,
            threshold=match_rate_threshold,
            passed=False,
            reason_codes=(NO_CROSS_CHECK_OVERLAP,),
        )

    source_by_key = {
        row[resolved_join_key]: row
        for row in source_rows
        if resolved_join_key in row and row[resolved_join_key] is not None
    }
    compared_count = 0
    matching_count = 0
    missing_field = False
    for candidate in candidate_rows:
        key = candidate.get(resolved_join_key)
        source = source_by_key.get(key)
        if source is None:
            continue
        candidate_value = candidate.get(field_name)
        source_value = source.get(comparison_field)
        if candidate_value is None or source_value is None:
            missing_field = True
            continue
        compared_count += 1
        matching_count += int(_values_match(candidate_value, source_value))

    if compared_count == 0:
        reasons = (MISSING_CROSS_CHECK_FIELD,) if missing_field else (NO_CROSS_CHECK_OVERLAP,)
        return CrossCheckResult(
            field_name=field_name,
            compared_count=0,
            matching_count=0,
            match_rate=None,
            threshold=match_rate_threshold,
            passed=False,
            reason_codes=reasons,
        )

    match_rate = matching_count / compared_count
    reasons = () if match_rate >= match_rate_threshold else (CROSS_CHECK_LOW_MATCH_RATE,)
    return CrossCheckResult(
        field_name=field_name,
        compared_count=compared_count,
        matching_count=matching_count,
        match_rate=match_rate,
        threshold=match_rate_threshold,
        passed=not reasons,
        reason_codes=reasons,
    )


def test_field_leakage(
    rows: Sequence[Row],
    field_name: str,
    post_fight_columns: Sequence[str],
    *,
    correlation_threshold: float = DEFAULT_LEAKAGE_CORRELATION_THRESHOLD,
) -> LeakageResult:
    """Check absolute Pearson correlation with supplied post-fight columns.

    Behavior/params/return: calculate correlations using complete numeric pairs.
    Assumes/errors: callers explicitly identify post-fight or odds-derived columns;
    a requested column with no usable pairs fails closed as inconclusive.
    """
    _validate_threshold(correlation_threshold, "correlation_threshold")
    columns = tuple(dict.fromkeys(post_fight_columns))
    if not columns:
        return LeakageResult(field_name, {}, correlation_threshold, True, ())

    correlations: dict[str, float] = {}
    reasons: list[str] = []
    for column in columns:
        pairs = [
            (candidate_value, post_value)
            for row in rows
            for candidate_value, post_value in [_numeric_pair(row.get(field_name), row.get(column))]
            if candidate_value is not None and post_value is not None
        ]
        if len(pairs) < 2:
            reasons.append(LEAKAGE_TEST_INCONCLUSIVE)
            continue
        correlations[column] = abs(_pearson_correlation(pairs))
        if correlations[column] >= correlation_threshold:
            reasons.append(LEAKAGE_DETECTED)

    return LeakageResult(
        field_name=field_name,
        correlations=correlations,
        threshold=correlation_threshold,
        passed=not reasons,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def validate_field(
    *,
    field_name: str,
    source_dataset: str,
    kaggle_rows: Sequence[Row],
    ufc_rows: Sequence[Row],
    provenance: Mapping[str, object] | ProvenanceAudit,
    post_fight_columns: Sequence[str] = (),
    join_key: str = "fighter_url",
    ufc_field: str | None = None,
    sample_size: int | None = None,
    match_rate_threshold: float = DEFAULT_MATCH_RATE_THRESHOLD,
    leakage_correlation_threshold: float = DEFAULT_LEAKAGE_CORRELATION_THRESHOLD,
    validation_date: date | None = None,
) -> KaggleValidationReport:
    """Run all admission checks for one field and return its decision record.

    Behavior/params/return: compose overlap, provenance, and leakage results.
    Assumes/errors: all row inputs are already fixture or offline extracts; no source
    row is modified, and any failed check rejects the field.
    """
    cross_check = cross_check_field(
        kaggle_rows,
        ufc_rows,
        field_name,
        join_key=join_key,
        ufc_field=ufc_field,
        sample_size=sample_size,
        match_rate_threshold=match_rate_threshold,
    )
    provenance_result = audit_provenance(provenance)
    leakage = test_field_leakage(
        kaggle_rows,
        field_name,
        post_fight_columns,
        correlation_threshold=leakage_correlation_threshold,
    )
    reason_codes = tuple(
        dict.fromkeys(
            (*cross_check.reason_codes, *provenance_result.reason_codes, *leakage.reason_codes)
        )
    )
    provenance_note = (
        provenance_result.audit.methodology if provenance_result.audit is not None else ""
    )
    return KaggleValidationReport(
        field_name=field_name,
        source_dataset=source_dataset,
        validation_date=validation_date or datetime.now(UTC).date(),
        match_rate=cross_check.match_rate,
        provenance_note=provenance_note,
        passed=not reason_codes,
        reason_codes=reason_codes,
        cross_check=cross_check,
        provenance=provenance_result,
        leakage=leakage,
    )


def write_validation_report(
    report: KaggleValidationReport,
    output_directory: Path = DEFAULT_REPORT_DIRECTORY,
) -> Path:
    """Write one stable JSON report per evaluated field and return its path."""
    safe_field_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", report.field_name).strip("._")
    if not safe_field_name:
        raise ValueError("field name must contain at least one filename-safe character")
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / f"{safe_field_name}.json"
    report_path.write_text(
        json.dumps(report.to_dict(), default=_json_default, indent=2, sort_keys=True) + "\n"
    )
    return report_path


def _resolve_join_key(
    candidate_rows: Sequence[Row], source_rows: Sequence[Row], requested_key: str
) -> str | None:
    """Use the requested key, with common offline fixture key fallbacks."""
    all_rows = (*candidate_rows, *source_rows)
    if candidate_rows and source_rows and all(requested_key in row for row in all_rows):
        return requested_key
    for key in ("fighter_url", "fight_url", "fighter_id", "id", "name"):
        if candidate_rows and source_rows and all(key in row for row in all_rows):
            return key
    return None


def _values_match(left: object, right: object) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-9)
    return str(left).strip().casefold() == str(right).strip().casefold()


def _numeric_pair(left: object, right: object) -> tuple[float | None, float | None]:
    if isinstance(left, bool) or isinstance(right, bool):
        return None, None
    try:
        return float(left), float(right)
    except (TypeError, ValueError):
        return None, None


def _pearson_correlation(pairs: Sequence[tuple[float, float]]) -> float:
    left_values = [pair[0] for pair in pairs]
    right_values = [pair[1] for pair in pairs]
    left_mean = sum(left_values) / len(left_values)
    right_mean = sum(right_values) / len(right_values)
    numerator = sum((left - left_mean) * (right - right_mean) for left, right in pairs)
    left_variance = sum((left - left_mean) ** 2 for left in left_values)
    right_variance = sum((right - right_mean) ** 2 for right in right_values)
    denominator = math.sqrt(left_variance * right_variance)
    return numerator / denominator if denominator else 0.0


def _validate_threshold(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _json_default(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")
