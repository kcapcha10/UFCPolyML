"""Schema and configuration tests for the model training layer.

Validates Pydantic frozen immutability, required field enforcement,
CandidateConfig constraint boundaries, AblationRung ordering, and
Hydra config loading via OmegaConf.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from ufc_edge.model.schemas import (
    AblationRung,
    AssemblyManifest,
    CandidateConfig,
    TrainResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_CANDIDATE = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 3.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

VALID_MANIFEST = {
    "n_rows": 1000,
    "n_features": 30,
    "feature_version": "v1",
    "feature_source_hash": "abc123",
    "columns": ["col_a", "col_b"],
    "assembled_at": datetime(2026, 1, 1, tzinfo=UTC),
}


# ---------------------------------------------------------------------------
# CandidateConfig immutability
# ---------------------------------------------------------------------------


def test_candidate_config_is_frozen():
    cfg = CandidateConfig(**VALID_CANDIDATE)
    with pytest.raises(ValidationError):
        cfg.learning_rate = 0.99


# ---------------------------------------------------------------------------
# CandidateConfig required fields
# ---------------------------------------------------------------------------


def test_candidate_config_requires_all_fields():
    for field in VALID_CANDIDATE:
        incomplete = {k: v for k, v in VALID_CANDIDATE.items() if k != field}
        with pytest.raises(ValidationError):
            CandidateConfig(**incomplete)


# ---------------------------------------------------------------------------
# CandidateConfig boundary validation
# ---------------------------------------------------------------------------


def test_candidate_rejects_zero_n_estimators():
    with pytest.raises(ValidationError):
        CandidateConfig(**{**VALID_CANDIDATE, "n_estimators": 0})


def test_candidate_rejects_negative_n_estimators():
    with pytest.raises(ValidationError):
        CandidateConfig(**{**VALID_CANDIDATE, "n_estimators": -1})


def test_candidate_rejects_zero_learning_rate():
    with pytest.raises(ValidationError):
        CandidateConfig(**{**VALID_CANDIDATE, "learning_rate": 0.0})


def test_candidate_rejects_negative_learning_rate():
    with pytest.raises(ValidationError):
        CandidateConfig(**{**VALID_CANDIDATE, "learning_rate": -0.01})


def test_candidate_rejects_learning_rate_above_one():
    with pytest.raises(ValidationError):
        CandidateConfig(**{**VALID_CANDIDATE, "learning_rate": 1.01})


def test_candidate_accepts_learning_rate_at_one():
    cfg = CandidateConfig(**{**VALID_CANDIDATE, "learning_rate": 1.0})
    assert cfg.learning_rate == 1.0


def test_candidate_rejects_zero_max_depth():
    with pytest.raises(ValidationError):
        CandidateConfig(**{**VALID_CANDIDATE, "max_depth": 0})


def test_candidate_rejects_max_depth_above_twenty():
    with pytest.raises(ValidationError):
        CandidateConfig(**{**VALID_CANDIDATE, "max_depth": 21})


def test_candidate_accepts_max_depth_at_twenty():
    cfg = CandidateConfig(**{**VALID_CANDIDATE, "max_depth": 20})
    assert cfg.max_depth == 20


def test_candidate_rejects_negative_min_child_weight():
    with pytest.raises(ValidationError):
        CandidateConfig(**{**VALID_CANDIDATE, "min_child_weight": -0.1})


def test_candidate_accepts_zero_min_child_weight():
    cfg = CandidateConfig(**{**VALID_CANDIDATE, "min_child_weight": 0.0})
    assert cfg.min_child_weight == 0.0


def test_candidate_rejects_zero_subsample():
    with pytest.raises(ValidationError):
        CandidateConfig(**{**VALID_CANDIDATE, "subsample": 0.0})


def test_candidate_rejects_subsample_above_one():
    with pytest.raises(ValidationError):
        CandidateConfig(**{**VALID_CANDIDATE, "subsample": 1.01})


def test_candidate_accepts_subsample_at_one():
    cfg = CandidateConfig(**{**VALID_CANDIDATE, "subsample": 1.0})
    assert cfg.subsample == 1.0


def test_candidate_rejects_zero_colsample_bytree():
    with pytest.raises(ValidationError):
        CandidateConfig(**{**VALID_CANDIDATE, "colsample_bytree": 0.0})


def test_candidate_rejects_colsample_bytree_above_one():
    with pytest.raises(ValidationError):
        CandidateConfig(**{**VALID_CANDIDATE, "colsample_bytree": 1.01})


def test_candidate_rejects_negative_reg_alpha():
    with pytest.raises(ValidationError):
        CandidateConfig(**{**VALID_CANDIDATE, "reg_alpha": -0.01})


def test_candidate_rejects_negative_reg_lambda():
    with pytest.raises(ValidationError):
        CandidateConfig(**{**VALID_CANDIDATE, "reg_lambda": -0.01})


def test_candidate_accepts_zero_regularization():
    cfg = CandidateConfig(**{**VALID_CANDIDATE, "reg_alpha": 0.0, "reg_lambda": 0.0})
    assert cfg.reg_alpha == 0.0
    assert cfg.reg_lambda == 0.0


# ---------------------------------------------------------------------------
# AssemblyManifest immutability and required fields
# ---------------------------------------------------------------------------


def test_assembly_manifest_is_frozen():
    m = AssemblyManifest(**VALID_MANIFEST)
    with pytest.raises(ValidationError):
        m.n_rows = 9999


def test_assembly_manifest_requires_fields():
    for field in ("n_rows", "n_features", "feature_version", "feature_source_hash",
                  "columns", "assembled_at"):
        incomplete = {k: v for k, v in VALID_MANIFEST.items() if k != field}
        with pytest.raises(ValidationError):
            AssemblyManifest(**incomplete)


def test_assembly_manifest_rejects_negative_n_rows():
    with pytest.raises(ValidationError):
        AssemblyManifest(**{**VALID_MANIFEST, "n_rows": -1})


# ---------------------------------------------------------------------------
# TrainResult immutability
# ---------------------------------------------------------------------------


def test_train_result_is_frozen():
    tr = TrainResult(
        candidate_config=CandidateConfig(**VALID_CANDIDATE),
        fold_id=0,
        random_seed=42,
        feature_version="v1",
        data_revision="rev1",
        elapsed_seconds=12.5,
    )
    with pytest.raises(ValidationError):
        tr.fold_id = 1


def test_train_result_rejects_negative_fold_id():
    with pytest.raises(ValidationError):
        TrainResult(
            candidate_config=CandidateConfig(**VALID_CANDIDATE),
            fold_id=-1,
            random_seed=42,
            feature_version="v1",
            data_revision="rev1",
            elapsed_seconds=12.5,
        )


def test_train_result_rejects_negative_elapsed():
    with pytest.raises(ValidationError):
        TrainResult(
            candidate_config=CandidateConfig(**VALID_CANDIDATE),
            fold_id=0,
            random_seed=42,
            feature_version="v1",
            data_revision="rev1",
            elapsed_seconds=-1.0,
        )


# ---------------------------------------------------------------------------
# AblationRung enum ordering
# ---------------------------------------------------------------------------


def test_ablation_rung_ordering_matches_ladder():
    expected = ["naive", "record", "physical", "schedule_strength", "domain_interactions"]
    actual = [r.value for r in AblationRung]
    assert actual == expected


def test_ablation_rung_is_str_enum():
    assert AblationRung.naive == "naive"
    assert isinstance(AblationRung.record, str)


# ---------------------------------------------------------------------------
# Hydra config loads via OmegaConf
# ---------------------------------------------------------------------------


def test_model_config_loads_via_omegaconf():
    cfg = OmegaConf.load("configs/model/default.yaml")
    assert "candidates" in cfg
    assert len(cfg.candidates) == 12
    assert cfg.random_seed == 42
    assert cfg.objective == "binary:logistic"
    assert cfg.eval_metric == "logloss"


def test_model_config_candidates_are_valid():
    cfg = OmegaConf.load("configs/model/default.yaml")
    for c in cfg.candidates:
        candidate = CandidateConfig(**c)
        assert candidate.n_estimators > 0
        assert 0.0 < candidate.learning_rate <= 1.0
