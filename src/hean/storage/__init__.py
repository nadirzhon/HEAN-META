"""Storage module -- DuckDB persistence layer for ticks, physics, and analyses."""

from .duckdb_store import DuckDBStore

__all__ = [
    "DuckDBStore",
]
