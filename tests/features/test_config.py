"""Tests for configs/graph.yaml: loadable with all expected keys present.

Production uses OmegaConf (declared in pyproject.toml); this test uses PyYAML
for environment portability since both parse the same YAML structure.
"""

from __future__ import annotations

import pathlib

import yaml

CONFIGS_DIR = pathlib.Path(__file__).resolve().parents[2] / "configs"


def _load_graph_config() -> dict:
    """Load graph.yaml and return the parsed dict."""
    path = CONFIGS_DIR / "graph.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


class TestGraphConfigLoads:
    """YAML loads without error."""

    def test_loads_without_error(self):
        cfg = _load_graph_config()
        assert cfg is not None
        assert isinstance(cfg, dict)


class TestEloKeys:
    """All Elo parameter keys are present."""

    def test_initial_rating(self):
        cfg = _load_graph_config()
        assert "initial_rating" in cfg["elo"]

    def test_k_base(self):
        cfg = _load_graph_config()
        assert "k_base" in cfg["elo"]

    def test_method_bonus_map(self):
        cfg = _load_graph_config()
        bonus_map = cfg["elo"]["method_bonus_map"]
        assert "KO/TKO" in bonus_map
        assert "Submission" in bonus_map
        assert "Decision" in bonus_map

    def test_recency_weight_halflife(self):
        cfg = _load_graph_config()
        assert "recency_weight_halflife" in cfg["elo"]

    def test_inactivity_decay_rate(self):
        cfg = _load_graph_config()
        assert "inactivity_decay_rate" in cfg["elo"]

    def test_inactivity_period_days(self):
        cfg = _load_graph_config()
        assert "inactivity_period_days" in cfg["elo"]

    def test_dq_k_multiplier(self):
        cfg = _load_graph_config()
        assert "dq_k_multiplier" in cfg["elo"]

    def test_injury_stoppage_k(self):
        cfg = _load_graph_config()
        assert "injury_stoppage_k" in cfg["elo"]


class TestEloOwnerValues:
    """Owner-specified Elo values are set correctly."""

    def test_initial_rating_value(self):
        cfg = _load_graph_config()
        assert cfg["elo"]["initial_rating"] == 1500

    def test_decision_bonus_zero(self):
        cfg = _load_graph_config()
        assert cfg["elo"]["method_bonus_map"]["Decision"] == 0.0

    def test_inactivity_period_days_value(self):
        cfg = _load_graph_config()
        assert cfg["elo"]["inactivity_period_days"] == 180

    def test_dq_k_multiplier_value(self):
        cfg = _load_graph_config()
        assert cfg["elo"]["dq_k_multiplier"] == 0.1

    def test_injury_stoppage_k_value(self):
        cfg = _load_graph_config()
        assert cfg["elo"]["injury_stoppage_k"] == 0


class TestGlicko2Keys:
    """All Glicko-2 parameter keys are present."""

    def test_initial_mu(self):
        cfg = _load_graph_config()
        assert "initial_mu" in cfg["glicko2"]

    def test_initial_rd(self):
        cfg = _load_graph_config()
        assert "initial_rd" in cfg["glicko2"]

    def test_tau(self):
        cfg = _load_graph_config()
        assert "tau" in cfg["glicko2"]

    def test_rating_period_days(self):
        cfg = _load_graph_config()
        assert "rating_period_days" in cfg["glicko2"]

    def test_high_uncertainty_threshold(self):
        cfg = _load_graph_config()
        assert "high_uncertainty_threshold" in cfg["glicko2"]


class TestGlicko2OwnerValues:
    """Owner-specified Glicko-2 values are set correctly."""

    def test_initial_mu_value(self):
        cfg = _load_graph_config()
        assert cfg["glicko2"]["initial_mu"] == 1500

    def test_initial_rd_value(self):
        cfg = _load_graph_config()
        assert cfg["glicko2"]["initial_rd"] == 350


class TestPageRankKeys:
    """All PageRank parameter keys are present."""

    def test_damping(self):
        cfg = _load_graph_config()
        assert "damping" in cfg["pagerank"]

    def test_finish_type_bonus_map(self):
        cfg = _load_graph_config()
        bonus_map = cfg["pagerank"]["finish_type_bonus_map"]
        assert "KO/TKO" in bonus_map
        assert "Submission" in bonus_map
        assert "Decision" in bonus_map

    def test_recency_decay_lambda(self):
        cfg = _load_graph_config()
        assert "recency_decay_lambda" in cfg["pagerank"]

    def test_early_finish_bonus(self):
        cfg = _load_graph_config()
        assert "early_finish_bonus" in cfg["pagerank"]

    def test_convergence_tolerance(self):
        cfg = _load_graph_config()
        assert "convergence_tolerance" in cfg["pagerank"]

    def test_max_iterations(self):
        cfg = _load_graph_config()
        assert "max_iterations" in cfg["pagerank"]


class TestPageRankOwnerValues:
    """Owner-specified PageRank values are set correctly."""

    def test_damping_value(self):
        cfg = _load_graph_config()
        assert cfg["pagerank"]["damping"] == 0.85

    def test_decision_bonus_zero(self):
        cfg = _load_graph_config()
        assert cfg["pagerank"]["finish_type_bonus_map"]["Decision"] == 0.0

    def test_convergence_tolerance_value(self):
        cfg = _load_graph_config()
        assert cfg["pagerank"]["convergence_tolerance"] == 1.0e-6

    def test_max_iterations_value(self):
        cfg = _load_graph_config()
        assert cfg["pagerank"]["max_iterations"] == 100


class TestCommonOpponentsKeys:
    """All common-opponent parameter keys are present."""

    def test_lookback_years(self):
        cfg = _load_graph_config()
        assert "lookback_years" in cfg["common_opponents"]

    def test_recency_decay_lambda(self):
        cfg = _load_graph_config()
        assert "recency_decay_lambda" in cfg["common_opponents"]

    def test_quality_weight_elo(self):
        cfg = _load_graph_config()
        assert "quality_weight_elo" in cfg["common_opponents"]

    def test_quality_weight_pagerank(self):
        cfg = _load_graph_config()
        assert "quality_weight_pagerank" in cfg["common_opponents"]


class TestCommonOpponentsOwnerValues:
    """Owner-specified common-opponent values are set correctly."""

    def test_lookback_years_value(self):
        cfg = _load_graph_config()
        assert cfg["common_opponents"]["lookback_years"] == 3


class TestTopLevelSections:
    """All four top-level graph sections are present."""

    def test_elo_section_exists(self):
        cfg = _load_graph_config()
        assert "elo" in cfg

    def test_glicko2_section_exists(self):
        cfg = _load_graph_config()
        assert "glicko2" in cfg

    def test_pagerank_section_exists(self):
        cfg = _load_graph_config()
        assert "pagerank" in cfg

    def test_common_opponents_section_exists(self):
        cfg = _load_graph_config()
        assert "common_opponents" in cfg
