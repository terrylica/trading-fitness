"""Source tick data fingerprinting for incremental feature validation.

Column mapping (ClickHouse → Python dict key):
    source_tick_xxhash64_checksum  <- fingerprint["xxhash64_checksum"]
    source_tick_row_count          <- fingerprint["row_count"]
    source_tick_first_timestamp_ms <- fingerprint["first_timestamp_ms"]
    source_tick_last_timestamp_ms  <- fingerprint["last_timestamp_ms"]
"""

from typing import Any

import polars as pl
import xxhash


def compute_source_tick_fingerprint(df: pl.DataFrame) -> dict[str, Any]:
    """Generate fingerprint for source tick DataFrame.

    Uses xxHash64 for speed (10x faster than SHA256 on large data).
    Returns dict with keys matching ClickHouse column suffixes.

    Args:
        df: Polars DataFrame of tick data. Must have 'timestamp' column.

    Returns:
        Dictionary with:
        - xxhash64_checksum: UInt64 hash of all column bytes
        - row_count: Number of rows in DataFrame
        - first_timestamp_ms: First tick timestamp (milliseconds)
        - last_timestamp_ms: Last tick timestamp (milliseconds)

    Example:
        >>> tick_df = pl.DataFrame({"timestamp": [1, 2, 3], "price": [100.0, 101.0, 102.0]})
        >>> fp = compute_source_tick_fingerprint(tick_df)
        >>> fp["row_count"]
        3
    """
    if df.is_empty():
        return {
            "xxhash64_checksum": 0,
            "row_count": 0,
            "first_timestamp_ms": 0,
            "last_timestamp_ms": 0,
        }

    # Compute xxHash64 over all column bytes (sorted for determinism)
    h = xxhash.xxh64()
    for col in sorted(df.columns):
        col_bytes = df[col].to_numpy().tobytes()
        h.update(col_bytes)

    # Extract timestamp bounds
    ts_col = df["timestamp"]
    first_ts = int(ts_col.min())  # type: ignore[arg-type]
    last_ts = int(ts_col.max())  # type: ignore[arg-type]

    return {
        "xxhash64_checksum": h.intdigest(),
        "row_count": len(df),
        "first_timestamp_ms": first_ts,
        "last_timestamp_ms": last_ts,
    }
