"""Pandera validation schemas for ITH features.

Uses pandera[polars] for Polars-native validation (no pandas).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandera.polars as pa
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence


# Column name pattern for ITH features: ith_rb{threshold}_lb{lookback}_{feature_type}
ITH_COLUMN_PATTERN = re.compile(r"^ith_rb(\d+)_lb(\d+)_(bull|bear)_(ed|eg|cv)$")
DD_RU_PATTERN = re.compile(r"^ith_rb(\d+)_lb(\d+)_(max_dd|max_ru)$")


class IthFeatureSchema(pa.DataFrameModel):
    """Validate ITH feature DataFrame structure and bounds.

    ITH features are bounded [0, 1] for ed/eg/cv and max_dd/max_ru.
    NaN values are allowed for warmup periods.
    """

    class Config:
        coerce = True
        strict = "filter"  # Allow extra columns (like bar_index, threshold_dbps)


def validate_ith_features(
    df: pl.DataFrame,
    check_bounds: bool = True,
    allowed_nulls: bool = True,
) -> tuple[bool, list[str]]:
    """Validate ITH feature DataFrame.

    Args:
        df: Polars DataFrame with ITH features
        check_bounds: Whether to verify features are in [0, 1]
        allowed_nulls: Whether to allow null/NaN values (for warmup)

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors: list[str] = []

    # Get ITH feature columns
    ith_cols = [c for c in df.columns if ITH_COLUMN_PATTERN.match(c) or DD_RU_PATTERN.match(c)]

    if not ith_cols:
        errors.append("No ITH feature columns found (expected pattern: ith_rb{threshold}_lb{lookback}_{type})")
        return False, errors

    for col in ith_cols:
        series = df.get_column(col)

        # Check for nulls if not allowed
        if not allowed_nulls and series.null_count() > 0:
            errors.append(f"Column {col} contains {series.null_count()} null values")

        # Check bounds [0, 1]
        if check_bounds:
            valid = series.drop_nulls()
            if len(valid) > 0:
                min_val = valid.min()
                max_val = valid.max()
                if min_val is not None and min_val < 0:
                    errors.append(f"Column {col} has values below 0 (min: {min_val})")
                if max_val is not None and max_val > 1:
                    errors.append(f"Column {col} has values above 1 (max: {max_val})")

    return len(errors) == 0, errors


def get_expected_columns(
    thresholds: Sequence[int],
    lookbacks: Sequence[int],
) -> list[str]:
    """Generate expected column names for given thresholds and lookbacks.

    Args:
        thresholds: Range bar thresholds in dbps (e.g., [25, 50, 100])
        lookbacks: Lookback windows in bars (e.g., [20, 50, 100])

    Returns:
        List of expected column names
    """
    feature_types = ["bull_ed", "bear_ed", "bull_eg", "bear_eg", "bull_cv", "bear_cv", "max_dd", "max_ru"]
    columns = []

    for threshold in thresholds:
        for lookback in lookbacks:
            for feature_type in feature_types:
                columns.append(f"ith_rb{threshold}_lb{lookback}_{feature_type}")

    return columns


def check_column_completeness(
    df: pl.DataFrame,
    thresholds: Sequence[int],
    lookbacks: Sequence[int],
) -> tuple[bool, list[str], list[str]]:
    """Check if all expected columns are present.

    Args:
        df: DataFrame to check
        thresholds: Expected thresholds
        lookbacks: Expected lookbacks

    Returns:
        Tuple of (is_complete, missing_columns, extra_columns)
    """
    expected = set(get_expected_columns(thresholds, lookbacks))
    actual_ith = {c for c in df.columns if ITH_COLUMN_PATTERN.match(c) or DD_RU_PATTERN.match(c)}

    missing = expected - actual_ith
    extra = actual_ith - expected

    return len(missing) == 0, sorted(missing), sorted(extra)
