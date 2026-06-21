"""DuckDB connection management.

Single entry point for opening a database connection. All tables are created
idempotently on first open. DDL is owned by each sub-package and aggregated here.

    with storage.get_connection() as conn:
        ufcstats_storage.upsert_event(conn, event)
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from ufc_edge.data.polymarket.storage import POLYMARKET_DDL
from ufc_edge.data.ufcstats.storage import UFCSTATS_DDL

DEFAULT_DB_PATH = Path("data/ufc_edge.duckdb")

_ALL_DDL = UFCSTATS_DDL + POLYMARKET_DDL


def db_path() -> Path:
    """Resolve the DuckDB file path from env or default."""
    raw = os.environ.get("DUCKDB_PATH", "")
    return Path(raw) if raw else DEFAULT_DB_PATH


@contextmanager
def get_connection(path: Path | None = None) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Open a DuckDB connection and ensure all tables exist.

    Yields the connection; caller should not close it — the context manager
    handles that. Creates the database file and parent directories on first use.
    """
    resolved = path or db_path()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(resolved))
    try:
        _ensure_tables(conn)
        yield conn
    finally:
        conn.close()


def _ensure_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all tables if they do not already exist."""
    for ddl in _ALL_DDL:
        conn.execute(ddl)
