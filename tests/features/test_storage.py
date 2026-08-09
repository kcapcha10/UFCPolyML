"""Tests for feature table storage: staging, validation, and atomic swap.

Validates DDL generation, staging writes, PK uniqueness enforcement, row-count
checks, atomic swap semantics, and schema conformance with the registry.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import duckdb
import pytest

from ufc_edge.features.contracts import FeatureRow
from ufc_edge.features.registry import FeatureFamily, FeatureRegistry
from ufc_edge.features.storage import (
    StorageError,
    atomic_swap,
    build_ddl,
    create_table,
    live_table_name,
    staging_table_name,
    validate_staging,
    write_and_swap,
    write_staging,
)
from ufc_edge.features.versioning import FEATURE_VERSION

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection for isolation between tests."""
    return duckdb.connect(":memory:")


@pytest.fixture
def registry() -> FeatureRegistry:
    """Minimal registry with two feature families for testing."""
    families = [
        FeatureFamily(
            name="physical",
            columns={"height_cm": float, "stance": str},
            order=0,
        ),
        FeatureFamily(
            name="graph",
            columns={"elo_rating": float},
            order=1,
        ),
    ]
    return FeatureRegistry(families=families)


@pytest.fixture
def sample_rows() -> list[FeatureRow]:
    """Two valid feature rows with distinct primary keys."""
    now = datetime.now(UTC)
    return [
        FeatureRow(
            fight_url="http://ufcstats.com/fight/f001",
            fighter_url="http://ufcstats.com/fighter/a001",
            event_url="http://ufcstats.com/event/e001",
            event_date=date(2024, 1, 15),
            opponent_url="http://ufcstats.com/fighter/b001",
            weight_class="Lightweight",
            feature_version=FEATURE_VERSION,
            generated_at=now,
            features={"height_cm": 180.0, "stance": "Orthodox", "elo_rating": 1520.0},
        ),
        FeatureRow(
            fight_url="http://ufcstats.com/fight/f001",
            fighter_url="http://ufcstats.com/fighter/b001",
            event_url="http://ufcstats.com/event/e001",
            event_date=date(2024, 1, 15),
            opponent_url="http://ufcstats.com/fighter/a001",
            weight_class="Lightweight",
            feature_version=FEATURE_VERSION,
            generated_at=now,
            features={"height_cm": 175.5, "stance": "Southpaw", "elo_rating": 1480.0},
        ),
    ]


def _make_duplicate_rows() -> list[FeatureRow]:
    """Two rows with identical primary keys to trigger validation failure."""
    now = datetime.now(UTC)
    row = FeatureRow(
        fight_url="http://ufcstats.com/fight/f001",
        fighter_url="http://ufcstats.com/fighter/a001",
        event_url="http://ufcstats.com/event/e001",
        event_date=date(2024, 1, 15),
        opponent_url="http://ufcstats.com/fighter/b001",
        weight_class="Lightweight",
        feature_version=FEATURE_VERSION,
        generated_at=now,
        features={"height_cm": 180.0, "stance": "Orthodox", "elo_rating": 1520.0},
    )
    return [row, row]


# ---------------------------------------------------------------------------
# Table naming
# ---------------------------------------------------------------------------


class TestTableNaming:
    def test_live_table_name_includes_version(self) -> None:
        assert live_table_name() == f"features_{FEATURE_VERSION}"

    def test_staging_table_name_prefixed(self) -> None:
        assert staging_table_name() == f"staging_features_{FEATURE_VERSION}"


# ---------------------------------------------------------------------------
# DDL generation
# ---------------------------------------------------------------------------


class TestBuildDDL:
    def test_ddl_contains_metadata_columns(self, registry: FeatureRegistry) -> None:
        ddl = build_ddl(registry)
        assert "fight_url VARCHAR NOT NULL" in ddl
        assert "fighter_url VARCHAR NOT NULL" in ddl
        assert "event_url VARCHAR NOT NULL" in ddl
        assert "event_date DATE NOT NULL" in ddl
        assert "feature_version VARCHAR NOT NULL" in ddl
        assert "generated_at TIMESTAMP NOT NULL" in ddl

    def test_ddl_contains_feature_columns(self, registry: FeatureRegistry) -> None:
        ddl = build_ddl(registry)
        assert "height_cm DOUBLE" in ddl
        assert "stance VARCHAR" in ddl
        assert "elo_rating DOUBLE" in ddl

    def test_ddl_contains_primary_key(self, registry: FeatureRegistry) -> None:
        ddl = build_ddl(registry)
        assert "PRIMARY KEY (fight_url, fighter_url)" in ddl

    def test_ddl_uses_live_table_name(self, registry: FeatureRegistry) -> None:
        ddl = build_ddl(registry)
        assert f"CREATE TABLE IF NOT EXISTS {live_table_name()}" in ddl

    def test_ddl_schema_matches_registry(self, registry: FeatureRegistry) -> None:
        """All registry columns appear in the DDL with correct DuckDB types."""
        ddl = build_ddl(registry)
        schema = registry.schema()
        type_map = {float: "DOUBLE", str: "VARCHAR", type(None): "DOUBLE"}
        for col_name, col_type in schema.items():
            expected_type = type_map[col_type]
            assert f"{col_name} {expected_type}" in ddl


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------


class TestCreateTable:
    def test_creates_table_in_database(
        self, conn: duckdb.DuckDBPyConnection, registry: FeatureRegistry
    ) -> None:
        create_table(conn, registry)
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [row[0] for row in tables]
        assert live_table_name() in table_names

    def test_idempotent_creation(
        self, conn: duckdb.DuckDBPyConnection, registry: FeatureRegistry
    ) -> None:
        create_table(conn, registry)
        create_table(conn, registry)
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [row[0] for row in tables]
        assert table_names.count(live_table_name()) == 1


# ---------------------------------------------------------------------------
# Staging writes
# ---------------------------------------------------------------------------


class TestWriteStaging:
    def test_writes_correct_row_count(
        self,
        conn: duckdb.DuckDBPyConnection,
        registry: FeatureRegistry,
        sample_rows: list[FeatureRow],
    ) -> None:
        count = write_staging(conn, sample_rows, registry)
        assert count == 2

    def test_staging_table_created(
        self,
        conn: duckdb.DuckDBPyConnection,
        registry: FeatureRegistry,
        sample_rows: list[FeatureRow],
    ) -> None:
        write_staging(conn, sample_rows, registry)
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [row[0] for row in tables]
        assert staging_table_name() in table_names

    def test_data_readable_from_staging(
        self,
        conn: duckdb.DuckDBPyConnection,
        registry: FeatureRegistry,
        sample_rows: list[FeatureRow],
    ) -> None:
        write_staging(conn, sample_rows, registry)
        staging = staging_table_name()
        result = conn.execute(f"SELECT fight_url, fighter_url FROM {staging}").fetchall()
        assert len(result) == 2

    def test_provenance_columns_populated(
        self,
        conn: duckdb.DuckDBPyConnection,
        registry: FeatureRegistry,
        sample_rows: list[FeatureRow],
    ) -> None:
        write_staging(conn, sample_rows, registry)
        staging = staging_table_name()
        rows = conn.execute(
            f"SELECT feature_version, generated_at FROM {staging}"
        ).fetchall()
        for row in rows:
            assert row[0] == FEATURE_VERSION
            assert row[1] is not None

    def test_drops_previous_staging_table(
        self,
        conn: duckdb.DuckDBPyConnection,
        registry: FeatureRegistry,
        sample_rows: list[FeatureRow],
    ) -> None:
        """Writing staging again replaces the prior staging table."""
        write_staging(conn, sample_rows, registry)
        write_staging(conn, sample_rows[:1], registry)
        staging = staging_table_name()
        count = conn.execute(f"SELECT COUNT(*) FROM {staging}").fetchone()[0]
        assert count == 1

    def test_feature_values_stored_correctly(
        self,
        conn: duckdb.DuckDBPyConnection,
        registry: FeatureRegistry,
        sample_rows: list[FeatureRow],
    ) -> None:
        write_staging(conn, sample_rows, registry)
        staging = staging_table_name()
        result = conn.execute(
            f"SELECT height_cm, stance, elo_rating FROM {staging} "
            f"WHERE fighter_url = 'http://ufcstats.com/fighter/a001'"
        ).fetchone()
        assert result[0] == pytest.approx(180.0)
        assert result[1] == "Orthodox"
        assert result[2] == pytest.approx(1520.0)

    def test_missing_feature_columns_stored_as_null(
        self,
        conn: duckdb.DuckDBPyConnection,
        registry: FeatureRegistry,
    ) -> None:
        """Feature columns not in the row's features dict become NULL."""
        now = datetime.now(UTC)
        row = FeatureRow(
            fight_url="http://ufcstats.com/fight/f002",
            fighter_url="http://ufcstats.com/fighter/c001",
            event_url="http://ufcstats.com/event/e002",
            event_date=date(2024, 2, 10),
            opponent_url="http://ufcstats.com/fighter/d001",
            weight_class="Welterweight",
            feature_version=FEATURE_VERSION,
            generated_at=now,
            features={"height_cm": 185.0},
        )
        write_staging(conn, [row], registry)
        staging = staging_table_name()
        result = conn.execute(
            f"SELECT stance, elo_rating FROM {staging}"
        ).fetchone()
        assert result[0] is None
        assert result[1] is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidateStaging:
    def test_valid_staging_passes(
        self,
        conn: duckdb.DuckDBPyConnection,
        registry: FeatureRegistry,
        sample_rows: list[FeatureRow],
    ) -> None:
        write_staging(conn, sample_rows, registry)
        validate_staging(conn, expected_row_count=2)

    def test_duplicate_primary_keys_raises(
        self,
        conn: duckdb.DuckDBPyConnection,
        registry: FeatureRegistry,
    ) -> None:
        """Staging table with duplicate PKs fails validation."""
        duplicate_rows = _make_duplicate_rows()
        # Bypass PK constraint by creating staging without PK enforcement
        staging = staging_table_name()
        conn.execute(f"DROP TABLE IF EXISTS {staging}")
        conn.execute(
            f"CREATE TABLE {staging} ("
            f"  fight_url VARCHAR NOT NULL,"
            f"  fighter_url VARCHAR NOT NULL,"
            f"  event_url VARCHAR NOT NULL,"
            f"  event_date DATE NOT NULL,"
            f"  opponent_url VARCHAR NOT NULL,"
            f"  weight_class VARCHAR,"
            f"  feature_version VARCHAR NOT NULL,"
            f"  generated_at TIMESTAMP NOT NULL,"
            f"  height_cm DOUBLE,"
            f"  stance VARCHAR,"
            f"  elo_rating DOUBLE"
            f")"
        )
        for row in duplicate_rows:
            conn.execute(
                f"INSERT INTO {staging} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    row.fight_url,
                    row.fighter_url,
                    row.event_url,
                    row.event_date,
                    row.opponent_url,
                    row.weight_class,
                    row.feature_version,
                    row.generated_at,
                    row.features.get("height_cm"),
                    row.features.get("stance"),
                    row.features.get("elo_rating"),
                ],
            )

        with pytest.raises(StorageError, match="uniqueness violation"):
            validate_staging(conn, expected_row_count=2)

    def test_row_count_mismatch_raises(
        self,
        conn: duckdb.DuckDBPyConnection,
        registry: FeatureRegistry,
        sample_rows: list[FeatureRow],
    ) -> None:
        """Validation fails when expected count does not match actual rows."""
        write_staging(conn, sample_rows, registry)

        with pytest.raises(StorageError, match="Row count mismatch"):
            validate_staging(conn, expected_row_count=5)


# ---------------------------------------------------------------------------
# Atomic swap
# ---------------------------------------------------------------------------


class TestAtomicSwap:
    def test_swap_replaces_live_table(
        self,
        conn: duckdb.DuckDBPyConnection,
        registry: FeatureRegistry,
        sample_rows: list[FeatureRow],
    ) -> None:
        """After swap, live table contains the staging data."""
        create_table(conn, registry)
        write_staging(conn, sample_rows, registry)
        atomic_swap(conn)

        live = live_table_name()
        count = conn.execute(f"SELECT COUNT(*) FROM {live}").fetchone()[0]
        assert count == 2

    def test_staging_table_gone_after_swap(
        self,
        conn: duckdb.DuckDBPyConnection,
        registry: FeatureRegistry,
        sample_rows: list[FeatureRow],
    ) -> None:
        """Staging table no longer exists after successful swap."""
        write_staging(conn, sample_rows, registry)
        atomic_swap(conn)

        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [row[0] for row in tables]
        assert staging_table_name() not in table_names

    def test_swap_works_without_existing_live_table(
        self,
        conn: duckdb.DuckDBPyConnection,
        registry: FeatureRegistry,
        sample_rows: list[FeatureRow],
    ) -> None:
        """Swap succeeds when no prior live table exists."""
        write_staging(conn, sample_rows, registry)
        atomic_swap(conn)

        live = live_table_name()
        count = conn.execute(f"SELECT COUNT(*) FROM {live}").fetchone()[0]
        assert count == 2

    def test_prior_live_table_unchanged_on_validation_failure(
        self,
        conn: duckdb.DuckDBPyConnection,
        registry: FeatureRegistry,
        sample_rows: list[FeatureRow],
    ) -> None:
        """If validation fails, the existing live table data is preserved."""
        # Write initial data to live table
        write_staging(conn, sample_rows, registry)
        atomic_swap(conn)

        # Verify initial state
        live = live_table_name()
        initial_count = conn.execute(f"SELECT COUNT(*) FROM {live}").fetchone()[0]
        assert initial_count == 2

        # Attempt write_and_swap with bad row count expectation
        # Create staging with data but validate with wrong count
        write_staging(conn, sample_rows[:1], registry)
        with pytest.raises(StorageError, match="Row count mismatch"):
            validate_staging(conn, expected_row_count=99)

        # Live table still has original 2 rows
        count = conn.execute(f"SELECT COUNT(*) FROM {live}").fetchone()[0]
        assert count == 2


# ---------------------------------------------------------------------------
# End-to-end write_and_swap
# ---------------------------------------------------------------------------


class TestWriteAndSwap:
    def test_successful_pipeline(
        self,
        conn: duckdb.DuckDBPyConnection,
        registry: FeatureRegistry,
        sample_rows: list[FeatureRow],
    ) -> None:
        count = write_and_swap(conn, sample_rows, registry)
        assert count == 2

        live = live_table_name()
        actual = conn.execute(f"SELECT COUNT(*) FROM {live}").fetchone()[0]
        assert actual == 2

    def test_replaces_existing_live_table(
        self,
        conn: duckdb.DuckDBPyConnection,
        registry: FeatureRegistry,
        sample_rows: list[FeatureRow],
    ) -> None:
        """Second write_and_swap replaces the first generation's data."""
        write_and_swap(conn, sample_rows, registry)
        write_and_swap(conn, sample_rows[:1], registry)

        live = live_table_name()
        count = conn.execute(f"SELECT COUNT(*) FROM {live}").fetchone()[0]
        assert count == 1

    def test_validation_failure_preserves_live_table(
        self,
        conn: duckdb.DuckDBPyConnection,
        registry: FeatureRegistry,
        sample_rows: list[FeatureRow],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On validation failure, write_and_swap leaves prior live table intact."""
        write_and_swap(conn, sample_rows, registry)

        live = live_table_name()
        count_before = conn.execute(f"SELECT COUNT(*) FROM {live}").fetchone()[0]
        assert count_before == 2

        # Force validation to fail after staging write succeeds.
        import ufc_edge.features.storage as storage_mod

        def _failing_validate(
            _conn: duckdb.DuckDBPyConnection, expected_row_count: int
        ) -> None:
            raise StorageError("Simulated validation failure")

        monkeypatch.setattr(storage_mod, "validate_staging", _failing_validate)

        with pytest.raises(StorageError, match="Simulated validation failure"):
            write_and_swap(conn, sample_rows[:1], registry)

        # Live table still has original 2 rows — swap never executed.
        count_after = conn.execute(f"SELECT COUNT(*) FROM {live}").fetchone()[0]
        assert count_after == 2

    def test_staging_cleaned_on_validation_failure(
        self,
        conn: duckdb.DuckDBPyConnection,
        registry: FeatureRegistry,
        sample_rows: list[FeatureRow],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Staging table is dropped when validation fails in write_and_swap."""
        import ufc_edge.features.storage as storage_mod

        def _failing_validate(
            _conn: duckdb.DuckDBPyConnection, expected_row_count: int
        ) -> None:
            raise StorageError("Forced failure for cleanup test")

        monkeypatch.setattr(storage_mod, "validate_staging", _failing_validate)

        with pytest.raises(StorageError, match="Forced failure for cleanup test"):
            write_and_swap(conn, sample_rows, registry)

        # Staging table must be cleaned up after the failure.
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [row[0] for row in tables]
        assert staging_table_name() not in table_names


# ---------------------------------------------------------------------------
# Schema conformance
# ---------------------------------------------------------------------------


class TestSchemaConformance:
    def test_live_table_columns_match_registry(
        self,
        conn: duckdb.DuckDBPyConnection,
        registry: FeatureRegistry,
        sample_rows: list[FeatureRow],
    ) -> None:
        """After write_and_swap, live table has all expected columns."""
        write_and_swap(conn, sample_rows, registry)

        live = live_table_name()
        columns = conn.execute(
            f"SELECT column_name FROM information_schema.columns "
            f"WHERE table_name = '{live}' ORDER BY ordinal_position"
        ).fetchall()
        column_names = [row[0] for row in columns]

        expected_metadata = list(
            {"fight_url", "fighter_url", "event_url", "event_date",
             "opponent_url", "weight_class", "feature_version", "generated_at"}
        )
        expected_features = list(registry.schema().keys())

        for col in expected_metadata:
            assert col in column_names
        for col in expected_features:
            assert col in column_names

    def test_column_types_match_registry_declarations(
        self,
        conn: duckdb.DuckDBPyConnection,
        registry: FeatureRegistry,
        sample_rows: list[FeatureRow],
    ) -> None:
        """DuckDB column types correspond to the registry type declarations."""
        write_and_swap(conn, sample_rows, registry)

        live = live_table_name()
        type_info = conn.execute(
            f"SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_name = '{live}'"
        ).fetchall()
        type_map = {row[0]: row[1] for row in type_info}

        python_to_duckdb = {float: "DOUBLE", str: "VARCHAR", type(None): "DOUBLE"}
        for col_name, col_type in registry.schema().items():
            expected_duckdb_type = python_to_duckdb[col_type]
            assert type_map[col_name] == expected_duckdb_type, (
                f"Column '{col_name}': expected {expected_duckdb_type}, "
                f"got {type_map[col_name]}"
            )
