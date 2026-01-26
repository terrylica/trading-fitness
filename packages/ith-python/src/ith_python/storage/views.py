"""View generators for Long Format SSoT.

This module provides functions to generate various view formats from the
Long Format SSoT (Single Source of Truth).

Architecture: Multi-View Feature Architecture with Separation of Concerns
- Layer 2: Feature Storage
- See: docs/plans/2026-01-25-multi-view-feature-architecture-plan.md
"""

from __future__ import annotations

import polars as pl


def to_wide(
    df: pl.DataFrame,
    threshold: int | None = None,
    symbol: str | None = None,
    drop_warmup: bool = True,
) -> pl.DataFrame:
    """Generate wide format suitable for ML training.

    Pivots the Long Format to create columns like: ith_rb{t}_lb{lb}_{feature}

    Args:
        df: Long Format DataFrame
        threshold: Filter by threshold_dbps (optional)
        symbol: Filter by symbol (optional)
        drop_warmup: If True, drop rows with any null values (warmup period)

    Returns:
        Wide format DataFrame with columns: bar_index, symbol, ith_rb{t}_lb{lb}_{f}
    """
    result = df.clone()

    # Apply filters
    if threshold is not None:
        result = result.filter(pl.col("threshold_dbps") == threshold)
    if symbol is not None:
        result = result.filter(pl.col("symbol") == symbol)

    if len(result) == 0:
        return pl.DataFrame()

    # Create column name from threshold, lookback, feature
    result = result.with_columns([
        pl.format(
            "ith_rb{}_lb{}_{}",
            pl.col("threshold_dbps"),
            pl.col("lookback"),
            pl.col("feature"),
        ).alias("col_name")
    ])

    # Pivot to wide format
    wide = result.pivot(
        on="col_name",
        index=["bar_index", "symbol"],
        values="value",
    )

    # Sort columns for consistency
    meta_cols = ["bar_index", "symbol"]
    feature_cols = sorted([c for c in wide.columns if c not in meta_cols])
    wide = wide.select(meta_cols + feature_cols)

    if drop_warmup:
        # Drop rows where any feature column is null OR NaN
        # NaN values come from Rust during warmup period
        for col in feature_cols:
            wide = wide.filter(~pl.col(col).is_nan() & pl.col(col).is_not_null())

    return wide


def get_warmup_bars(lookbacks: list[int]) -> int:
    """Calculate the number of warmup bars needed for given lookbacks.

    Args:
        lookbacks: List of lookback window sizes

    Returns:
        Number of bars in warmup period (max_lookback - 1)
    """
    return max(lookbacks) - 1


def validate_warmup(
    n_bars: int,
    lookbacks: list[int],
    min_valid_bars: int = 100,
) -> tuple[bool, dict]:
    """Validate that we have enough data after warmup.

    This is a preflight check to ensure meaningful analysis is possible.

    Args:
        n_bars: Total number of input bars
        lookbacks: List of lookback window sizes
        min_valid_bars: Minimum required valid bars after warmup

    Returns:
        Tuple of (is_valid, info_dict)
    """
    warmup_bars = get_warmup_bars(lookbacks)
    valid_bars = n_bars - warmup_bars

    info = {
        "n_bars": n_bars,
        "warmup_bars": warmup_bars,
        "valid_bars": valid_bars,
        "min_required": min_valid_bars,
        "max_lookback": max(lookbacks),
    }

    is_valid = valid_bars >= min_valid_bars

    if not is_valid:
        info["error"] = (
            f"Insufficient data: {valid_bars} valid bars after warmup "
            f"(need at least {min_valid_bars}). "
            f"Fetch at least {warmup_bars + min_valid_bars} bars."
        )

    return is_valid, info


def to_nested(df: pl.DataFrame) -> list[dict]:
    """Generate nested JSON for semantic queries and API responses.

    Args:
        df: Long Format DataFrame

    Returns:
        List of nested dictionaries with structure:
        {bar_index, symbol, features: {rb{t}: {lb{lb}: {feature: value}}}}
    """
    results = []

    # Group by bar_index and symbol
    for (bar_idx, sym), group in df.group_by(["bar_index", "symbol"]):
        record = {
            "bar_index": bar_idx,
            "symbol": sym,
            "features": {},
        }

        # Group by threshold
        for threshold in group["threshold_dbps"].unique().to_list():
            t_key = f"rb{threshold}"
            record["features"][t_key] = {}

            t_group = group.filter(pl.col("threshold_dbps") == threshold)

            # Group by lookback
            for lookback in t_group["lookback"].unique().to_list():
                lb_key = f"lb{lookback}"
                lb_group = t_group.filter(pl.col("lookback") == lookback)

                record["features"][t_key][lb_key] = {
                    row["feature"]: row["value"]
                    for row in lb_group.iter_rows(named=True)
                    if row["valid"]
                }

        results.append(record)

    return results


def to_dense(df: pl.DataFrame, threshold: int) -> pl.DataFrame:
    """Generate dense format for a single threshold (no sparsity).

    Unlike to_wide with multiple thresholds (which creates sparse columns),
    this returns only columns for the specified threshold.

    Args:
        df: Long Format DataFrame
        threshold: The threshold_dbps to filter by

    Returns:
        Dense DataFrame with only the specified threshold's features
    """
    # Filter to single threshold
    filtered = df.filter(pl.col("threshold_dbps") == threshold)

    if len(filtered) == 0:
        return pl.DataFrame()

    # Create column name (omit threshold since it's single)
    filtered = filtered.with_columns([
        pl.format(
            "ith_lb{}_{}", pl.col("lookback"), pl.col("feature")
        ).alias("col_name")
    ])

    # Pivot to wide format
    wide = filtered.pivot(
        on="col_name",
        index=["bar_index", "symbol"],
        values="value",
    )

    # Sort columns
    meta_cols = ["bar_index", "symbol"]
    feature_cols = sorted([c for c in wide.columns if c not in meta_cols])
    wide = wide.select(meta_cols + feature_cols)

    return wide


def to_clickhouse(df: pl.DataFrame):
    """Generate Arrow RecordBatch for ClickHouse insertion.

    Args:
        df: Long Format DataFrame

    Returns:
        PyArrow RecordBatch suitable for ClickHouse bulk insert
    """
    import pyarrow as pa

    # Convert Polars to Arrow
    arrow_table = df.to_arrow()

    # ClickHouse prefers RecordBatches
    # Combine all chunks into a single batch
    batches = arrow_table.to_batches()

    if not batches:
        # Return empty batch with schema
        return pa.RecordBatch.from_pydict(
            {col: [] for col in df.columns},
            schema=arrow_table.schema,
        )

    # Combine batches if multiple
    if len(batches) == 1:
        return batches[0]

    combined_table = pa.Table.from_batches(batches)
    return combined_table.combine_chunks().to_batches()[0]
