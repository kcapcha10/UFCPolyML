"""Boundary tests for frozen mismatch-report Pydantic schemas."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from ufc_edge.report.schemas import (
    ChecklistFindings,
    DueDiligenceVerdict,
    DueDiligenceVerdictType,
    Finding,
    GateResult,
    GateVerdict,
    MarketFightLink,
    MatchMethod,
    MatchStatus,
    PaperSignal,
    PostSignalSnapshot,
    ReportRun,
    SnapshotOffset,
    SnapshotStatus,
)

_NOW = datetime(2026, 8, 31, 12, 0, 0)


def _link(**overrides: object) -> MarketFightLink:
    """Build a valid link and allow one field to be varied by a test."""
    values: dict[str, object] = {
        "fight_url": "/fight/abc",
        "token_id": "token-abc",
        "match_status": MatchStatus.MATCHED,
        "match_method": MatchMethod.AUTO_NAME,
        "matched_at": _NOW,
    }
    values.update(overrides)
    return MarketFightLink(**values)


def _signal(**overrides: object) -> PaperSignal:
    """Build a valid paper signal and allow one field to be varied by a test."""
    values: dict[str, object] = {
        "signal_id": "signal-abc",
        "report_run_id": "run-abc",
        "fight_url": "/fight/abc",
        "fighter_a_url": "/fighter/a",
        "fighter_b_url": "/fighter/b",
        "fighter_a_name": "Fighter A",
        "fighter_b_name": "Fighter B",
        "match_status": MatchStatus.MATCHED,
        "mlflow_run_id": "mlflow-abc",
        "data_revision": "revision-abc",
        "feature_version": "features-abc",
        "config_hash": "config-abc",
        "created_at": _NOW,
    }
    values.update(overrides)
    return PaperSignal(**values)


def _verdict(**overrides: object) -> DueDiligenceVerdict:
    """Build a valid due-diligence verdict and allow one field to vary."""
    values: dict[str, object] = {
        "fight_url": "/fight/abc",
        "report_run_id": "run-abc",
        "verdict": DueDiligenceVerdictType.CONFIRM,
        "confidence": 0.8,
        "evidence_urls": ["https://example.com/evidence"],
        "summary": "No material concerns found.",
        "checklist_findings": ChecklistFindings(
            injury_news=Finding(
                present=False,
                detail="No injury news",
                source_url="https://example.com/evidence",
            )
        ),
        "prompt_version": "prompt-v1",
        "model_name": "model",
        "model_version": "model-v1",
        "invoked_at": _NOW,
    }
    values.update(overrides)
    return DueDiligenceVerdict(**values)


def test_report_models_are_frozen() -> None:
    """Assigning a new value to a report model field is rejected."""
    link = _link()

    with pytest.raises(ValidationError):
        link.fight_url = "/fight/changed"


def test_url_fields_reject_null_values() -> None:
    """Required fight, fighter, and evidence source URL fields reject nulls."""
    with pytest.raises(ValidationError):
        _link(fight_url=None)

    with pytest.raises(ValidationError):
        _signal(fighter_a_url=None)

    with pytest.raises(ValidationError):
        Finding(present=True, detail="An injury", source_url=None)


def test_enum_fields_reject_unknown_values() -> None:
    """Status, gate, snapshot, and verdict fields accept only declared labels."""
    with pytest.raises(ValidationError):
        _link(match_status="UNKNOWN")

    with pytest.raises(ValidationError):
        GateResult(
            verdict="UNKNOWN",
            bucket_id="0.5-0.7",
            bucket_n=10,
            bucket_calibration_error=0.04,
            ci_lower=0.01,
            ci_upper=0.07,
        )

    with pytest.raises(ValidationError):
        PostSignalSnapshot(
            snapshot_id="snapshot-abc",
            signal_id="signal-abc",
            token_id="token-abc",
            scheduled_offset="UNKNOWN",
            scheduled_at=_NOW,
            status=SnapshotStatus.CAPTURED,
        )

    with pytest.raises(ValidationError):
        _verdict(verdict="UNKNOWN")


def test_due_diligence_requires_at_least_one_evidence_url() -> None:
    """A verdict without evidence cannot cross the schema boundary."""
    with pytest.raises(ValidationError, match="evidence_urls"):
        _verdict(evidence_urls=[])


def test_report_models_round_trip_through_json() -> None:
    """JSON serialization preserves URLs, enum values, and timestamps."""
    signal = _signal(
        event_date="2026-09-01",
        gate_verdict=GateVerdict.FLAGGED,
        p_model=0.7,
        p_market_mid=0.5,
    )
    restored = PaperSignal.model_validate_json(signal.model_dump_json())

    assert restored == signal


def test_nested_checklist_and_enum_values_round_trip() -> None:
    """Nested findings and due-diligence enum values survive model serialization."""
    verdict = _verdict(verdict=DueDiligenceVerdictType.VETO)
    restored = DueDiligenceVerdict.model_validate_json(verdict.model_dump_json())

    assert restored.checklist_findings == verdict.checklist_findings
    assert restored.verdict is DueDiligenceVerdictType.VETO


def test_all_declared_enum_labels_are_accepted() -> None:
    """Each persisted enum accepts its serialized label."""
    assert _link(match_status="NO_CANDIDATE").match_status is MatchStatus.NO_CANDIDATE
    assert _link(match_method="MANUAL_OVERRIDE").match_method is MatchMethod.MANUAL_OVERRIDE
    assert _signal(gate_verdict="WITHIN_NOISE").gate_verdict is GateVerdict.WITHIN_NOISE
    assert (
        PostSignalSnapshot(
            snapshot_id="snapshot-abc",
            signal_id="signal-abc",
            token_id="token-abc",
            scheduled_offset="24H",
            scheduled_at=_NOW,
            status="SKIPPED",
        ).scheduled_offset
        is SnapshotOffset.TWENTY_FOUR_HOURS
    )
    assert _verdict(verdict="QUALIFY").verdict is DueDiligenceVerdictType.QUALIFY


def test_required_report_fields_are_enforced() -> None:
    """A report run cannot be created without its provenance fields."""
    with pytest.raises(ValidationError):
        ReportRun(
            report_run_id="run-abc",
            as_of_timestamp=_NOW,
            mlflow_run_id="mlflow-abc",
            data_revision="revision-abc",
            feature_version="features-abc",
            config_hash="config-abc",
            bout_count=1,
            flagged_count=0,
        )
