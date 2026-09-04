"""Feature table storage: staging, validation, and atomic swap.

Manages the lifecycle of the versioned features table in DuckDB. Rows are first
written to a staging table, validated for schema conformance and primary-key
uniqueness, and then atomically promoted to the live `features_v{N}` table via
ALTER TABLE RENAME. If validation fails, the staging table is dropped and the
prior live table remains untouched.
"""

from __future__ import annotations

import duckdb

from ufc_edge.features.contracts import FeatureRow
from ufc_edge.features.registry import FeatureRegistry
from ufc_edge.features.versioning import FEATURE_VERSION

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_METADATA_COLUMNS: dict[str, str] = {
    "fight_url": "VARCHAR NOT NULL",
    "fighter_url": "VARCHAR NOT NULL",
    "event_url": "VARCHAR NOT NULL",
    "event_date": "DATE NOT NULL",
    "opponent_url": "VARCHAR NOT NULL",
    "weight_class": "VARCHAR",
    "feature_version": "VARCHAR NOT NULL",
    "generated_at": "TIMESTAMP NOT NULL",
}

_PRIMARY_KEY = ("fight_url", "fighter_url")

_TYPE_MAP: dict[type, str] = {
    float: "DOUBLE",
    str: "VARCHAR",
    type(None): "DOUBLE",
}


class StorageError(Exception):
    """Raised when storage validation or swap fails."""


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def live_table_name() -> str:
    """Return the canonical live table name for the current feature version."""
    return f"features_{FEATURE_VERSION}"


def staging_table_name() -> str:
    """Return the staging table name used during writes."""
    return f"staging_{live_table_name()}"


def build_ddl(registry: FeatureRegistry) -> str:
    """Generate CREATE TABLE DDL for the live features table.

    Columns come from the fixed metadata set plus all feature columns declared
    in the registry, mapped to DuckDB types. Returns the SQL string without
    executing it.
    """
    lines: list[str] = []
    for col_name, col_ddl in _METADATA_COLUMNS.items():
        lines.append(f"    {col_name} {col_ddl}")

    for col_name, col_type in registry.schema().items():
        duckdb_type = _TYPE_MAP[col_type]
        lines.append(f"    {col_name} {duckdb_type}")

    pk_cols = ", ".join(_PRIMARY_KEY)
    lines.append(f"    PRIMARY KEY ({pk_cols})")

    columns_sql = ",\n".join(lines)
    table = live_table_name()
    return f"CREATE TABLE IF NOT EXISTS {table} (\n{columns_sql}\n);"


def create_table(conn: duckdb.DuckDBPyConnection, registry: FeatureRegistry) -> None:
    """Create the live features table if it does not exist."""
    conn.execute(build_ddl(registry))


def write_staging(
    conn: duckdb.DuckDBPyConnection,
    rows: list[FeatureRow],
    registry: FeatureRegistry,
) -> int:
    """Write feature rows to a fresh staging table.

    Drops any pre-existing staging table, creates a new one with the same schema
    as the live table, and inserts all rows. Returns the number of rows written.
    """
    staging = staging_table_name()
    conn.execute(f"DROP TABLE IF EXISTS {staging}")

    ddl = build_ddl(registry).replace(live_table_name(), staging)
    ddl = ddl.replace("CREATE TABLE IF NOT EXISTS", "CREATE TABLE")
    conn.execute(ddl)

    schema = registry.schema()
    feature_columns = list(schema.keys())
    all_columns = list(_METADATA_COLUMNS.keys()) + feature_columns

    for row in rows:
        values = _row_to_values(row, feature_columns)
        placeholders = ", ".join(["?"] * len(all_columns))
        col_names = ", ".join(all_columns)
        conn.execute(
            f"INSERT INTO {staging} ({col_names}) VALUES ({placeholders})",  # noqa: S608
            values,
        )

    return len(rows)


def validate_staging(
    conn: duckdb.DuckDBPyConnection,
    expected_row_count: int,
) -> None:
    """Validate the staging table before swap.

    Checks:
    1. Row count matches the expected value.
    2. No duplicate primary keys (fight_url, fighter_url).

    Raises StorageError on any validation failure. The staging table is NOT
    dropped here — the caller decides cleanup strategy.
    """
    staging = staging_table_name()

    actual_count = conn.execute(f"SELECT COUNT(*) FROM {staging}").fetchone()[0]  # noqa: S608
    if actual_count != expected_row_count:
        raise StorageError(
            f"Row count mismatch: expected {expected_row_count}, "
            f"got {actual_count} in staging table '{staging}'"
        )

    pk_cols = ", ".join(_PRIMARY_KEY)
    duplicate_count = conn.execute(
        f"SELECT COUNT(*) FROM ("  # noqa: S608
        f"  SELECT {pk_cols} FROM {staging}"
        f"  GROUP BY {pk_cols} HAVING COUNT(*) > 1"
        f")"
    ).fetchone()[0]

    if duplicate_count > 0:
        raise StorageError(
            f"Primary key uniqueness violation: {duplicate_count} duplicate "
            f"({', '.join(_PRIMARY_KEY)}) groups found in staging table '{staging}'"
        )


def atomic_swap(conn: duckdb.DuckDBPyConnection) -> None:
    """Atomically promote the staging table to the live table.

    Strategy:
    1. Drop the existing live table (if any).
    2. Rename the staging table to the live table name.

    Both operations run inside a single transaction. If the rename fails, the
    transaction rolls back and the original live table remains intact.
    """
    live = live_table_name()
    staging = staging_table_name()

    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute(f"DROP TABLE IF EXISTS {live}")
        conn.execute(f"ALTER TABLE {staging} RENAME TO {live}")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def write_and_swap(
    conn: duckdb.DuckDBPyConnection,
    rows: list[FeatureRow],
    registry: FeatureRegistry,
) -> int:
    """Full pipeline: write staging, validate, swap atomically.

    This is the primary entry point for callers. Validates BEFORE swapping to
    ensure the live table is never replaced with invalid data. On validation
    failure, the staging table is dropped and the live table remains unchanged.

    Returns the number of rows in the new live table.
    """
    row_count = write_staging(conn, rows, registry)

    try:
        validate_staging(conn, expected_row_count=row_count)
    except StorageError:
        conn.execute(f"DROP TABLE IF EXISTS {staging_table_name()}")
        raise

    atomic_swap(conn)
    return row_count


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _row_to_values(row: FeatureRow, feature_columns: list[str]) -> list:
    """Convert a FeatureRow to a flat list of values for INSERT.

    Order: metadata columns first (matching _METADATA_COLUMNS), then feature
    columns in registry order.
    """
    values: list = [
        row.fight_url,
        row.fighter_url,
        row.event_url,
        row.event_date,
        row.opponent_url,
        row.weight_class,
        row.feature_version,
        row.generated_at,
    ]
    for col_name in feature_columns:
        values.append(row.features.get(col_name))
    return values
