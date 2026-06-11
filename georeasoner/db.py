"""DuckDB connection factory and schema bootstrap."""

from pathlib import Path

import duckdb

from georeasoner.config import settings


def get_connection(db_path: str | None = None) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection, attempting to load the spatial extension."""
    path = db_path or settings.duckdb_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(path)
    _load_spatial(conn)
    return conn


def _load_spatial(conn: duckdb.DuckDBPyConnection) -> None:
    """Install + load the DuckDB spatial extension; silently skip if unavailable."""
    try:
        conn.execute("LOAD spatial")
    except duckdb.IOException:
        try:
            conn.execute("INSTALL spatial")
            conn.execute("LOAD spatial")
        except Exception:
            # Offline CI or air-gapped environment — basic ops still work
            pass


def init_db(db_path: str | None = None) -> None:
    """Create application tables if they don't already exist."""
    conn = get_connection(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analysis_runs (
            id          VARCHAR   PRIMARY KEY,
            query       TEXT      NOT NULL,
            status      VARCHAR   NOT NULL DEFAULT 'pending',
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            result_path TEXT,
            error       TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_traces (
            run_id      VARCHAR   NOT NULL,
            agent_name  VARCHAR   NOT NULL,
            step_index  INTEGER   NOT NULL,
            tool_call   TEXT,
            tool_result TEXT,
            ts          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.close()
