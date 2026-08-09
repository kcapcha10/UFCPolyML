"""Feature engine CLI — batch replay and materialization entry point.

Runs the full feature replay pipeline: loads config, verifies source-hash
integrity, executes the event-atomic replay, writes the features table through
the staging/validation/swap storage layer, and exits with a meaningful code.

Usage:
    uv run python -m ufc_edge.features [--skip-integrity-check] [--dry-run]

    Or via Makefile:
        make features

Exit codes:
    0  — Success: replay completed and features table materialized.
    1  — Version integrity failure: source hash does not match manifest.
    2  — Replay or storage error: a component, emitter, or validation failed.
    3  — Configuration error: config file missing or invalid.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb
from omegaconf import OmegaConf

from ufc_edge.features.components import (
    CareerAccumulator,
    CommonOpponentIndex,
    EloTracker,
    Glicko2Tracker,
    PageRankGraph,
    RollingStatsAccumulator,
)
from ufc_edge.features.components.weight_class import WeightClassTracker
from ufc_edge.features.emitters import (
    ActivityEmitter,
    CardPositionEmitter,
    ExperienceEmitter,
    FinishingEmitter,
    GraphEmitter,
    MatchupEmitter,
    OutputEmitter,
    PhysicalEmitter,
    RecordEmitter,
    RematchEmitter,
    WeightCutEmitter,
    WeightDominanceEmitter,
)
from ufc_edge.features.emitters.experience import ExperienceAccumulator
from ufc_edge.features.emitters.rematch import RematchAccumulator
from ufc_edge.features.loader import load_historical_fights
from ufc_edge.features.registry import FeatureFamily, FeatureRegistry
from ufc_edge.features.replay import ComponentRegistry, ReplayConfig, replay
from ufc_edge.features.storage import StorageError, create_table, write_and_swap
from ufc_edge.features.versioning import (
    FEATURE_VERSION,
    check_version_integrity,
)

# Exit codes with semantic meaning for operators and CI.
EXIT_SUCCESS = 0
EXIT_INTEGRITY_FAILURE = 1
EXIT_REPLAY_ERROR = 2
EXIT_CONFIG_ERROR = 3

logger = logging.getLogger("ufc_edge.features")

# ---------------------------------------------------------------------------
# Production feature registry definition
# ---------------------------------------------------------------------------

_PRODUCTION_FAMILIES: list[FeatureFamily] = [
    FeatureFamily(
        name="physical",
        columns={
            "height_cm": float,
            "reach_cm": float,
            "reach_to_height_ratio": float,
            "stance": str,
            "age_at_fight": float,
            "weight_class": str,
        },
        order=0,
    ),
    FeatureFamily(
        name="activity",
        columns={
            "days_since_last_fight": float,
            "fights_last_12mo": float,
            "fights_last_3yr": float,
            "fights_last_5yr": float,
            "total_ufc_fights": float,
            "last_fight_injury_stoppage": float,
            "age_x_inactivity": float,
            "inactivity_tier": float,
        },
        order=1,
    ),
    FeatureFamily(
        name="record",
        columns={
            "win_pct_all": float,
            "win_pct_last3": float,
            "win_pct_last5": float,
            "current_streak": float,
            "win_pct_by_finish": float,
            "win_pct_by_decision": float,
            "loss_pct_by_finish": float,
            "loss_pct_by_decision": float,
            "ufc_win_pct": float,
            "ufc_record_fights_count": float,
            "is_ufc_debut": float,
            "debut_opponent_ufc_experience": float,
            "debut_opponent_ufc_win_pct": float,
            "contender_series_win": float,
        },
        order=2,
    ),
    FeatureFamily(
        name="finishing",
        columns={
            "finish_rate": float,
            "ko_rate": float,
            "submission_rate": float,
            "early_finish_rate": float,
            "avg_fight_duration_sec": float,
            "fight_duration_variance": float,
            "has_ever_been_finished": float,
            "times_finished_by_ko": float,
            "times_finished_by_sub": float,
            "has_been_finished_r1": float,
            "never_been_finished": float,
            "never_been_finished_x_opp_finish_rate": float,
        },
        order=3,
    ),
    FeatureFamily(
        name="output",
        columns={
            "sig_strikes_per_min": float,
            "sig_strikes_absorbed_per_min": float,
            "striking_accuracy_pct": float,
            "striking_defense_pct": float,
            "td_per_15min": float,
            "td_accuracy_pct": float,
            "td_defense_pct": float,
            "sub_attempts_per_15min": float,
            "knockdown_rate": float,
            "damage_ratio": float,
            "grappling_dominance": float,
            "control_time_per_fight": float,
        },
        order=4,
    ),
    FeatureFamily(
        name="card_position",
        columns={
            "sig_strikes_main_card_avg": float,
            "sig_strikes_prelim_avg": float,
            "td_rate_main_card_avg": float,
            "td_rate_prelim_avg": float,
            "output_variance_by_position": float,
            "grappling_abandonment_delta": float,
        },
        order=5,
    ),
    FeatureFamily(
        name="experience",
        columns={
            "title_fight_experience": float,
            "has_been_champion": float,
            "days_as_champion": float,
            "main_event_experience": float,
            "five_round_experience": float,
            "five_round_win_pct": float,
        },
        order=6,
    ),
    FeatureFamily(
        name="weight_dominance",
        columns={
            "is_weight_class_change": float,
            "direction_of_change": float,
            "fights_at_current_class": float,
            "win_pct_at_current_class": float,
            "prior_class_win_pct": float,
            "is_large_for_class": float,
            "grappling_utilization_rate": float,
            "weight_bully_score": float,
        },
        order=7,
    ),
    FeatureFamily(
        name="graph",
        columns={
            "elo_rating": float,
            "elo_trajectory_last5": float,
            "elo_peak": float,
            "elo_current_vs_peak": float,
            "glicko2_rating": float,
            "glicko2_rd": float,
            "pagerank_score": float,
            "n_common_opponents": float,
            "common_opp_score_a": float,
            "common_opp_score_b": float,
            "common_opp_score_delta": float,
            "common_opp_a_win_rate": float,
            "common_opp_b_win_rate": float,
        },
        order=8,
    ),
    FeatureFamily(
        name="matchup",
        columns={
            "reach_delta": float,
            "height_delta": float,
            "age_delta": float,
            "elo_delta": float,
            "pagerank_delta": float,
            "ufc_experience_delta": float,
            "title_fight_exp_delta": float,
            "five_round_experience_delta": float,
            "finish_rate_delta": float,
            "striking_efficiency_delta": float,
            "td_accuracy_delta": float,
            "avg_fight_duration_delta": float,
            "fight_duration_variance_delta": float,
            "damage_ratio_delta": float,
            "wrestler_score_a": float,
            "wrestler_score_b": float,
            "wrestling_delta": float,
            "submission_score_a": float,
            "submission_score_b": float,
            "submission_delta": float,
            "grappling_type_mismatch": float,
            "striker_vs_grappler": float,
            "pressure_vs_counter": float,
            "pace_mismatch_score": float,
            "southpaw_matchup": float,
            "southpaw_orthodox_history": float,
            "stance_matchup": str,
        },
        order=9,
    ),
    FeatureFamily(
        name="rematch",
        columns={
            "is_rematch": float,
            "fights_since_first_meeting": float,
            "result_of_first_meeting": float,
            "first_meeting_method": str,
            "first_meeting_competitive": float,
            "first_meeting_score_delta": float,
        },
        order=10,
    ),
    FeatureFamily(
        name="weight_cut",
        columns={
            "missed_weight_last_3": float,
            "missed_weight_career": float,
            "moving_down_in_weight": float,
            "short_notice": float,
            "full_camp": float,
        },
        order=11,
    ),
]


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m ufc_edge.features",
        description="Run the UFC feature replay and materialize the features table.",
    )
    parser.add_argument(
        "--skip-integrity-check",
        action="store_true",
        help="Skip the source-hash integrity verification (development only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the replay but do not write to the database.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------


def _resolve_config(project_root: Path) -> dict:
    """Load and merge feature-relevant configuration from Hydra YAML files.

    Reads configs/data/default.yaml for duckdb_path and configs/graph.yaml for
    graph hyperparameters. Resolves OmegaConf interpolations (including the
    DUCKDB_PATH environment variable override).
    """
    data_config_path = project_root / "configs" / "data" / "default.yaml"
    graph_config_path = project_root / "configs" / "graph.yaml"

    if not data_config_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_config_path}")

    data_cfg = OmegaConf.load(data_config_path)
    resolved_data = OmegaConf.to_container(data_cfg, resolve=True)

    graph_cfg = None
    if graph_config_path.exists():
        graph_cfg = OmegaConf.to_container(
            OmegaConf.load(graph_config_path), resolve=True
        )

    return {
        "duckdb_path": resolved_data["duckdb_path"],
        "graph": graph_cfg,
    }


# ---------------------------------------------------------------------------
# Pipeline component assembly
# ---------------------------------------------------------------------------


def _build_feature_registry() -> FeatureRegistry:
    """Construct the canonical production feature registry."""
    return FeatureRegistry(families=_PRODUCTION_FAMILIES)


# Columns that overlap with storage metadata — present in emitter output for
# validation but must not appear in the DDL as feature columns.
_METADATA_OVERLAP_COLUMNS: frozenset[str] = frozenset({"weight_class"})


def _build_storage_registry() -> FeatureRegistry:
    """Build a registry suitable for storage DDL (excludes metadata-overlap columns).

    The PhysicalEmitter emits weight_class for validation completeness, but
    storage.py already declares it as a metadata column. The storage registry
    filters it out to prevent duplicate DDL columns.
    """
    storage_families = []
    for family in _PRODUCTION_FAMILIES:
        filtered_columns = {
            k: v for k, v in family.columns.items()
            if k not in _METADATA_OVERLAP_COLUMNS
        }
        storage_families.append(
            FeatureFamily(
                name=family.name,
                columns=filtered_columns,
                order=family.order,
            )
        )
    return FeatureRegistry(families=storage_families)


def _build_components() -> tuple[ComponentRegistry, ExperienceAccumulator]:
    """Instantiate all state components and wire cross-references.

    Returns the ComponentRegistry for the replay engine plus the standalone
    ExperienceAccumulator (which needs update_with_context instead of update).
    """
    elo = EloTracker()
    glicko2 = Glicko2Tracker()
    pagerank = PageRankGraph()
    career = CareerAccumulator()
    rolling = RollingStatsAccumulator()
    weight_class = WeightClassTracker()
    common_opponents = CommonOpponentIndex(elo_state=elo, pagerank_state=pagerank)
    experience_acc = ExperienceAccumulator()
    rematch_acc = RematchAccumulator()

    components: dict[str, object] = {
        "elo": elo,
        "glicko2": glicko2,
        "pagerank": pagerank,
        "career": career,
        "rolling_stats": rolling,
        "weight_class": weight_class,
        "common_opponents": common_opponents,
        "experience": experience_acc,
        "rematch": rematch_acc,
    }

    return ComponentRegistry(components), experience_acc  # type: ignore[arg-type]


def _build_emitters() -> list[object]:
    """Instantiate all feature emitters in registry order."""
    return [
        PhysicalEmitter(),
        ActivityEmitter(),
        RecordEmitter(),
        FinishingEmitter(),
        OutputEmitter(),
        CardPositionEmitter(),
        ExperienceEmitter(),
        WeightDominanceEmitter(),
        GraphEmitter(),
        MatchupEmitter(),
        RematchEmitter(),
        WeightCutEmitter(),
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(argv: list[str] | None = None) -> int:
    """Execute the feature replay pipeline. Returns an exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args(argv)

    # Locate project root (parent of src/)
    features_dir = Path(__file__).resolve().parent
    src_dir = features_dir.parent.parent
    project_root = src_dir.parent

    # --- Configuration ---
    logger.info("Loading configuration from %s", project_root / "configs")
    try:
        config = _resolve_config(project_root)
    except (FileNotFoundError, Exception) as exc:
        logger.error("Configuration error: %s", exc)
        return EXIT_CONFIG_ERROR

    # --- Version integrity check ---
    manifest_path = project_root / "features_version_manifest.json"
    if not args.skip_integrity_check:
        logger.info("Checking version integrity (version=%s)", FEATURE_VERSION)
        result = check_version_integrity(features_dir, manifest_path)
        if not result.ok:
            logger.error("Version integrity check FAILED: %s", result.message)
            return EXIT_INTEGRITY_FAILURE
        logger.info("Version integrity check passed (hash=%s)", result.actual_hash)
    else:
        logger.warning("Skipping version integrity check (--skip-integrity-check)")

    # --- Build pipeline components ---
    logger.info("Building feature registry, components, and emitters")
    try:
        feature_registry = _build_feature_registry()
        storage_registry = _build_storage_registry()
        component_registry, experience_acc = _build_components()
        emitters = _build_emitters()
    except Exception as exc:
        logger.error("Component initialization failed: %s", exc)
        return EXIT_REPLAY_ERROR

    # --- Connect to DuckDB and load fights ---
    duckdb_path = config["duckdb_path"]
    logger.info("Connecting to DuckDB at %s", duckdb_path)
    try:
        conn = duckdb.connect(str(duckdb_path))
        fights = load_historical_fights(conn)
        logger.info("Loaded %d historical fights", len(fights))
    except Exception as exc:
        logger.error("Data loading failed: %s", exc)
        return EXIT_REPLAY_ERROR

    # --- Run replay ---
    logger.info("Starting feature replay (version=%s)", FEATURE_VERSION)
    try:
        replay_config = ReplayConfig(log_every_n_events=50)
        rows = replay(
            fights=fights,
            component_registry=component_registry,
            emitters=emitters,
            feature_registry=feature_registry,
            config=replay_config,
            experience_accumulator=experience_acc,
        )
        logger.info("Replay produced %d feature rows", len(rows))
    except Exception as exc:
        logger.error("Replay failed: %s", exc)
        return EXIT_REPLAY_ERROR

    # --- Write to storage ---
    if args.dry_run:
        logger.info(
            "Dry run: skipping storage write (%d rows would be written)", len(rows)
        )
        return EXIT_SUCCESS

    logger.info("Writing features to storage (staging → validate → swap)")
    try:
        create_table(conn, storage_registry)
        row_count = write_and_swap(conn, rows, storage_registry)
        logger.info(
            "Feature materialization complete: %d rows in features_%s",
            row_count,
            FEATURE_VERSION,
        )
    except StorageError as exc:
        logger.error("Storage validation failed: %s", exc)
        return EXIT_REPLAY_ERROR
    except Exception as exc:
        logger.error("Storage write failed: %s", exc)
        return EXIT_REPLAY_ERROR
    finally:
        conn.close()

    logger.info("Feature engine finished successfully")
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(run())
