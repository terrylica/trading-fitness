"""Storage schemas for Long Format SSoT.

This module defines the canonical schema for the Long Format feature storage.

Architecture: Multi-View Feature Architecture with Separation of Concerns
- Layer 2: Feature Storage
- See: docs/plans/2026-01-25-multi-view-feature-architecture-plan.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    pass

# Long format schema definition
LONG_SCHEMA = {
    # Primary Keys
    "bar_index": pl.UInt32,
    "symbol": pl.Categorical,
    "threshold_dbps": pl.UInt16,
    "lookback": pl.UInt16,
    "feature": pl.Categorical,
    # Value
    "value": pl.Float64,
    # Metadata
    "valid": pl.Boolean,
    "computed_at": pl.Datetime("us", "UTC"),
    "nav_hash": pl.Utf8,
}

# Feature short names (matches Rust FEATURE_SHORT_NAMES)
FEATURE_SHORT_NAMES = [
    "bull_ed",
    "bear_ed",
    "bull_eg",
    "bear_eg",
    "bull_cv",
    "bear_cv",
    "max_dd",
    "max_ru",
]

# Feature full names for documentation
FEATURE_FULL_NAMES = {
    "bull_ed": "bull_epoch_density",
    "bear_ed": "bear_epoch_density",
    "bull_eg": "bull_excess_gain",
    "bear_eg": "bear_excess_gain",
    "bull_cv": "bull_cv",
    "bear_cv": "bear_cv",
    "max_dd": "max_drawdown",
    "max_ru": "max_runup",
}


def validate_long_format(df: pl.DataFrame) -> tuple[bool, list[str]]:
    """Validate that a DataFrame conforms to the Long Format schema.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors: list[str] = []

    # Check required columns exist
    required_cols = {"bar_index", "symbol", "threshold_dbps", "lookback", "feature", "value"}
    missing = required_cols - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Check feature values are valid
    if "feature" in df.columns:
        unique_features = df["feature"].unique().to_list()
        invalid_features = [f for f in unique_features if f not in FEATURE_SHORT_NAMES]
        if invalid_features:
            errors.append(f"Invalid feature names: {invalid_features}")

    # Check value bounds (features should be [0, 1], null, or NaN)
    if "value" in df.columns:
        # Drop both null and NaN values before checking bounds
        values = df["value"].drop_nulls().drop_nans()
        out_of_bounds = values.filter((values < 0) | (values > 1))
        if len(out_of_bounds) > 0:
            errors.append(f"Found {len(out_of_bounds)} values outside [0, 1] range")

    # Check threshold_dbps are positive
    if "threshold_dbps" in df.columns and df["threshold_dbps"].min() <= 0:
        errors.append("threshold_dbps must be positive")

    # Check lookbacks are positive
    if "lookback" in df.columns and df["lookback"].min() <= 0:
        errors.append("lookback must be positive")

    return len(errors) == 0, errors
