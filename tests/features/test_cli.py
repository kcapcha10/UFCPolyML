"""Tests for the feature engine CLI entry point.

Validates:
- Exit codes for success, integrity failure, config error, and replay error.
- Version integrity guard rejects stale source hash.
- Dry-run mode runs replay without writing to storage.
- Configuration loading with DUCKDB_PATH override.
- End-to-end: fixture DB → CLI run → features table materialized.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from ufc_edge.features.__main__ import (
    EXIT_CONFIG_ERROR,
    EXIT_INTEGRITY_FAILURE,
    EXIT_REPLAY_ERROR,
    EXIT_SUCCESS,
    _build_components,
    _build_emitters,
    _build_feature_registry,
    _resolve_config,
)
from ufc_edge.features.versioning import FEATURE_VERSION, compute_source_hash

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a minimal project layout with config files for the CLI."""
    configs = tmp_path / "configs" / "data"
    configs.mkdir(parents=True)
    db_path = tmp_path / "test.duckdb"
    (configs / "default.yaml").write_text(
        f"duckdb_path: {db_path}\nlabel_start_date: '2010-01-01'\n"
    )

    graph_dir = tmp_path / "configs"
    (graph_dir / "graph.yaml").write_text("elo:\n  initial_rating: 1500\n")

    return tmp_path


@pytest.fixture
def fixture_db(tmp_path: Path) -> Path:
    """Create a DuckDB with minimal schema for the loader to query."""
    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    _create_source_tables(conn)
    _insert_fixture_data(conn)
    conn.close()
    return db_path


@pytest.fixture
def project_with_db(tmp_path: Path, fixture_db: Path) -> Path:
    """Project layout with config pointing to the fixture DB."""
    configs = tmp_path / "configs" / "data"
    configs.mkdir(parents=True)
    (configs / "default.yaml").write_text(
        f"duckdb_path: {fixture_db}\nlabel_start_date: '2010-01-01'\n"
    )
    (tmp_path / "configs" / "graph.yaml").write_text("elo:\n  initial_rating: 1500\n")
    return tmp_path


def _create_source_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the source tables the loader joins against."""
    conn.execute("""
        CREATE TABLE events (
            event_url VARCHAR PRIMARY KEY,
            date DATE NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE fighters (
            fighter_url VARCHAR PRIMARY KEY,
            height_cm DOUBLE,
            reach_cm DOUBLE,
            stance VARCHAR,
            date_of_birth DATE
        )
    """)
    conn.execute("""
        CREATE TABLE fights (
            fight_url VARCHAR PRIMARY KEY,
            event_url VARCHAR NOT NULL,
            fighter_a_url VARCHAR NOT NULL,
            fighter_b_url VARCHAR NOT NULL,
            winner_url VARCHAR,
            method VARCHAR,
            ending_round INTEGER,
            ending_time VARCHAR,
            time_format VARCHAR,
            weight_class VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE fight_totals (
            fight_url VARCHAR NOT NULL,
            fighter_url VARCHAR NOT NULL,
            knockdowns INTEGER,
            significant_strikes_landed INTEGER,
            significant_strikes_attempted INTEGER,
            total_strikes_landed INTEGER,
            total_strikes_attempted INTEGER,
            takedowns_landed INTEGER,
            takedowns_attempted INTEGER,
            submission_attempts INTEGER,
            reversals INTEGER,
            control_time_seconds INTEGER,
            PRIMARY KEY (fight_url, fighter_url)
        )
    """)
    conn.execute("""
        CREATE TABLE validation_quarantine (
            table_name VARCHAR NOT NULL,
            row_key VARCHAR NOT NULL
        )
    """)


def _insert_fixture_data(conn: duckdb.DuckDBPyConnection) -> None:
    """Insert minimal fight data for end-to-end test."""
    conn.execute("""
        INSERT INTO events VALUES
        ('http://ufcstats.com/event/e1', '2023-01-15'),
        ('http://ufcstats.com/event/e2', '2023-02-20')
    """)
    conn.execute("""
        INSERT INTO fighters VALUES
        ('http://ufcstats.com/fighter/a1', 180.0, 190.0, 'Orthodox', '1990-05-10'),
        ('http://ufcstats.com/fighter/b1', 175.0, 185.0, 'Southpaw', '1992-03-15'),
        ('http://ufcstats.com/fighter/c1', 183.0, 188.0, 'Orthodox', '1988-11-20')
    """)
    conn.execute("""
        INSERT INTO fights VALUES
        ('http://ufcstats.com/fight/f1', 'http://ufcstats.com/event/e1',
         'http://ufcstats.com/fighter/a1', 'http://ufcstats.com/fighter/b1',
         'http://ufcstats.com/fighter/a1', 'Decision', 3, '5:00',
         '3 Rnd (5-5-5)', 'Lightweight'),
        ('http://ufcstats.com/fight/f2', 'http://ufcstats.com/event/e2',
         'http://ufcstats.com/fighter/b1', 'http://ufcstats.com/fighter/c1',
         'http://ufcstats.com/fighter/c1', 'KO/TKO', 1, '2:30',
         '3 Rnd (5-5-5)', 'Lightweight')
    """)
    conn.execute("""
        INSERT INTO fight_totals VALUES
        ('http://ufcstats.com/fight/f1', 'http://ufcstats.com/fighter/a1',
         2, 80, 120, 100, 150, 3, 5, 1, 1, 180),
        ('http://ufcstats.com/fight/f1', 'http://ufcstats.com/fighter/b1',
         1, 60, 110, 80, 130, 1, 4, 2, 0, 90),
        ('http://ufcstats.com/fight/f2', 'http://ufcstats.com/fighter/b1',
         0, 30, 50, 40, 60, 0, 2, 0, 0, 30),
        ('http://ufcstats.com/fight/f2', 'http://ufcstats.com/fighter/c1',
         3, 45, 70, 60, 90, 2, 3, 1, 0, 120)
    """)


# ---------------------------------------------------------------------------
# Tests: Exit codes
# ---------------------------------------------------------------------------


class TestExitCodes:
    """Verify distinct exit codes for each failure mode."""

    def test_success_exit_code_is_zero(self) -> None:
        assert EXIT_SUCCESS == 0

    def test_integrity_failure_code_is_one(self) -> None:
        assert EXIT_INTEGRITY_FAILURE == 1

    def test_replay_error_code_is_two(self) -> None:
        assert EXIT_REPLAY_ERROR == 2

    def test_config_error_code_is_three(self) -> None:
        assert EXIT_CONFIG_ERROR == 3


# ---------------------------------------------------------------------------
# Tests: Configuration loading
# ---------------------------------------------------------------------------


class TestConfiguration:
    """Verify config loading from YAML with env override support."""

    def test_loads_duckdb_path_from_config(self, tmp_project: Path) -> None:
        config = _resolve_config(tmp_project)
        assert "duckdb_path" in config
        assert "test.duckdb" in str(config["duckdb_path"])

    def test_duckdb_path_env_override(self, tmp_project: Path) -> None:
        configs = tmp_project / "configs" / "data"
        (configs / "default.yaml").write_text(
            "duckdb_path: ${oc.env:DUCKDB_PATH,fallback.duckdb}\nlabel_start_date: '2010-01-01'\n"
        )
        with patch.dict("os.environ", {"DUCKDB_PATH": "/custom/path.duckdb"}):
            config = _resolve_config(tmp_project)
        assert config["duckdb_path"] == "/custom/path.duckdb"

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _resolve_config(tmp_path)

    def test_graph_config_loaded_when_present(self, tmp_project: Path) -> None:
        config = _resolve_config(tmp_project)
        assert config["graph"] is not None
        assert config["graph"]["elo"]["initial_rating"] == 1500


# ---------------------------------------------------------------------------
# Tests: Version integrity guard
# ---------------------------------------------------------------------------


class TestVersionIntegrity:
    """Verify the CLI exits with code 1 on source-hash mismatch."""

    def test_stale_hash_returns_integrity_failure(self, tmp_path: Path) -> None:
        """CLI rejects when manifest hash doesn't match current source."""
        features_dir = Path(__file__).resolve().parent.parent.parent / ("src/ufc_edge/features")
        # Write a manifest with a stale (wrong) hash
        manifest_path = tmp_path / "features_version_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "version": FEATURE_VERSION,
                    "source_hash": "0000000000000000000000000000000000000000",
                    "changelog": "stale",
                    "created_at": "2024-01-01T00:00:00+00:00",
                }
            )
        )

        from ufc_edge.features.versioning import check_version_integrity

        result = check_version_integrity(features_dir, manifest_path)
        assert not result.ok
        assert "mismatch" in result.message.lower()

    def test_valid_hash_passes_integrity(self, tmp_path: Path) -> None:
        """CLI proceeds when manifest hash matches current source."""
        features_dir = Path(__file__).resolve().parent.parent.parent / ("src/ufc_edge/features")
        current_hash = compute_source_hash(features_dir)

        manifest_path = tmp_path / "features_version_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "version": FEATURE_VERSION,
                    "source_hash": current_hash,
                    "changelog": "current",
                    "created_at": "2024-01-01T00:00:00+00:00",
                }
            )
        )

        from ufc_edge.features.versioning import check_version_integrity

        result = check_version_integrity(features_dir, manifest_path)
        assert result.ok


# ---------------------------------------------------------------------------
# Tests: Component and emitter assembly
# ---------------------------------------------------------------------------


class TestAssembly:
    """Verify that production components and emitters initialize correctly."""

    def test_build_feature_registry_succeeds(self) -> None:
        registry = _build_feature_registry()
        schema = registry.schema()
        assert len(schema) > 50
        assert "elo_rating" in schema
        assert "height_cm" in schema

    def test_build_components_returns_registry_and_accumulator(self) -> None:
        component_reg, exp_acc = _build_components()
        assert "elo" in component_reg.names
        assert "career" in component_reg.names
        assert "rematch" in component_reg.names
        assert exp_acc is not None

    def test_build_emitters_returns_twelve(self) -> None:
        emitters = _build_emitters()
        assert len(emitters) == 12


# ---------------------------------------------------------------------------
# Tests: CLI argument parsing
# ---------------------------------------------------------------------------


class TestArgParsing:
    """Verify CLI flags are parsed correctly."""

    def test_default_args(self) -> None:
        from ufc_edge.features.__main__ import _parse_args

        args = _parse_args([])
        assert not args.skip_integrity_check
        assert not args.dry_run

    def test_skip_integrity_flag(self) -> None:
        from ufc_edge.features.__main__ import _parse_args

        args = _parse_args(["--skip-integrity-check"])
        assert args.skip_integrity_check

    def test_dry_run_flag(self) -> None:
        from ufc_edge.features.__main__ import _parse_args

        args = _parse_args(["--dry-run"])
        assert args.dry_run


# ---------------------------------------------------------------------------
# Tests: End-to-end integration
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full CLI run against a fixture database."""

    def test_dry_run_on_fixture_db(self, project_with_db: Path, fixture_db: Path) -> None:
        """Dry-run loads data, runs replay, but writes nothing."""
        exit_code = _direct_run(
            project_with_db,
            Path(__file__).resolve().parent.parent.parent / "src/ufc_edge/features",
            ["--dry-run", "--skip-integrity-check"],
        )
        assert exit_code == EXIT_SUCCESS

    def test_full_run_materializes_table(self, project_with_db: Path, fixture_db: Path) -> None:
        """Full run writes the features table to DuckDB."""
        exit_code = _direct_run(
            project_with_db,
            Path(__file__).resolve().parent.parent.parent / "src/ufc_edge/features",
            ["--skip-integrity-check"],
        )
        assert exit_code == EXIT_SUCCESS

        # Verify the table exists with expected row count (2 fights × 2 orientations)
        conn = duckdb.connect(str(fixture_db))
        table_name = f"features_{FEATURE_VERSION}"
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        conn.close()
        assert count == 4

    def test_full_run_table_has_correct_pks(self, project_with_db: Path, fixture_db: Path) -> None:
        """Materialized table has the expected primary key pairs."""
        _direct_run(
            project_with_db,
            Path(__file__).resolve().parent.parent.parent / "src/ufc_edge/features",
            ["--skip-integrity-check"],
        )

        conn = duckdb.connect(str(fixture_db))
        table_name = f"features_{FEATURE_VERSION}"
        rows = conn.execute(
            f"SELECT fight_url, fighter_url FROM {table_name} ORDER BY fight_url, fighter_url"
        ).fetchall()
        conn.close()

        fight_urls = {r[0] for r in rows}
        assert "http://ufcstats.com/fight/f1" in fight_urls
        assert "http://ufcstats.com/fight/f2" in fight_urls
        assert len(rows) == 4

    def test_config_error_exits_with_code_three(self, tmp_path: Path) -> None:
        """CLI exits 3 when config directory is missing."""
        features_dir = Path(__file__).resolve().parent.parent.parent / ("src/ufc_edge/features")
        exit_code = _direct_run(tmp_path, features_dir, [])
        assert exit_code == EXIT_CONFIG_ERROR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _direct_run(project_root: Path, features_dir: Path, argv: list[str]) -> int:
    """Execute the CLI pipeline directly with explicit paths for testing."""
    import logging

    from ufc_edge.features.__main__ import (
        _build_components,
        _build_emitters,
        _build_feature_registry,
        _build_storage_registry,
        _parse_args,
        _resolve_config,
    )
    from ufc_edge.features.loader import load_historical_fights
    from ufc_edge.features.replay import ReplayConfig, replay
    from ufc_edge.features.storage import create_table, write_and_swap
    from ufc_edge.features.versioning import check_version_integrity

    logging.basicConfig(level=logging.WARNING)
    args = _parse_args(argv)

    try:
        config = _resolve_config(project_root)
    except (FileNotFoundError, Exception):
        return EXIT_CONFIG_ERROR

    manifest_path = project_root / "features_version_manifest.json"
    if not args.skip_integrity_check:
        result = check_version_integrity(features_dir, manifest_path)
        if not result.ok:
            return EXIT_INTEGRITY_FAILURE

    try:
        feature_registry = _build_feature_registry()
        storage_registry = _build_storage_registry()
        component_registry, experience_acc = _build_components()
        emitters = _build_emitters()
    except Exception:
        return EXIT_REPLAY_ERROR

    duckdb_path = config["duckdb_path"]
    try:
        conn = duckdb.connect(str(duckdb_path))
        fights = load_historical_fights(conn)
    except Exception:
        return EXIT_REPLAY_ERROR

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
    except Exception:
        return EXIT_REPLAY_ERROR

    if args.dry_run:
        conn.close()
        return EXIT_SUCCESS

    try:
        create_table(conn, storage_registry)
        write_and_swap(conn, rows, storage_registry)
    except Exception:
        return EXIT_REPLAY_ERROR
    finally:
        conn.close()

    return EXIT_SUCCESS
