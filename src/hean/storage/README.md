# storage -- Persistent Storage

DuckDB-based persistence layer for ticks, physics snapshots, brain analyses, and other time-series data.

## Architecture

The `DuckDBStore` subscribes to EventBus events and batches writes into DuckDB tables for efficient persistence. It maintains in-memory write buffers (ticks: 10,000, physics: 5,000, brain: 1,000) that are flushed periodically (default every 5 seconds) or when they reach the batch size threshold (default 500). This batching approach minimizes disk I/O while ensuring data is eventually persisted. DuckDB is an optional dependency -- if not installed, the store logs a warning and disables itself gracefully. Data is stored at `data/hean.duckdb` by default.

## Key Classes

- `DuckDBStore` (`duckdb_store.py`) -- Main storage class. Subscribes to TICK and CONTEXT_UPDATE events. Maintains three write buffers (ticks, physics, brain) with configurable max sizes. Background flush loop writes batched data to DuckDB tables. Provides methods for querying stored data. Tables are created on first start.

## Events

| Event | Direction | Description |
|-------|-----------|-------------|
| TICK | Subscribes | Market tick data persisted to ticks table |
| CONTEXT_UPDATE | Subscribes | Physics and brain data persisted to respective tables |

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| (none) | data/hean.duckdb | Database file path (hardcoded default, configurable via constructor) |

DuckDB is an optional dependency. Install with: `pip install duckdb`.
