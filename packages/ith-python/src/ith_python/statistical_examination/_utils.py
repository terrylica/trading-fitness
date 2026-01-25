"""Utility functions for ITH statistical examination.

Provides column name parsing, warmup handling, and feature filtering.
All functions are Polars-native (no pandas).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

# Column name patterns
ITH_COLUMN_PATTERN = re.compile(r"^ith_rb(\d+)_lb(\d+)_(.+)$")
FEATURE_TYPES = frozenset(["bull_ed", "bear_ed", "bull_eg", "bear_eg", "bull_cv", "bear_cv", "max_dd", "max_ru"])


def get_warmup_bars(lookbacks: Sequence[int]) -> int:
    """Return max warmup period (max lookback - 1).

    Args:
        lookbacks: List of lookback windows

    Returns:
        Number of warmup bars where features are NaN
    """
    return max(lookbacks) - 1


def drop_warmup(df: pl.DataFrame, lookbacks: Sequence[int]) -> pl.DataFrame:
    """Drop warmup rows where features are NaN.

    Args:
        df: DataFrame with ITH features
        lookbacks: Lookback windows used

    Returns:
        DataFrame with warmup rows removed
    """
    warmup = get_warmup_bars(lookbacks)
    return df.slice(warmup)


def extract_threshold(col_name: str) -> int | None:
    """Extract threshold from column name.

    Args:
        col_name: Column name like 'ith_rb250_lb100_bull_ed'

    Returns:
        Threshold value (250) or None if not found
    """
    match = ITH_COLUMN_PATTERN.match(col_name)
    return int(match.group(1)) if match else None


def extract_lookback(col_name: str) -> int | None:
    """Extract lookback from column name.

    Args:
        col_name: Column name like 'ith_rb250_lb100_bull_ed'

    Returns:
        Lookback value (100) or None if not found
    """
    match = ITH_COLUMN_PATTERN.match(col_name)
    return int(match.group(2)) if match else None


def extract_feature_type(col_name: str) -> str | None:
    """Extract feature type from column name.

    Args:
        col_name: Column name like 'ith_rb250_lb100_bull_ed'

    Returns:
        Feature type ('bull_ed') or None if not found
    """
    match = ITH_COLUMN_PATTERN.match(col_name)
    return match.group(3) if match else None


def get_feature_columns(
    df: pl.DataFrame,
    threshold: int | None = None,
    lookback: int | None = None,
    feature_type: str | None = None,
) -> list[str]:
    """Get ITH feature columns matching criteria.

    Args:
        df: DataFrame with ITH features
        threshold: Filter by threshold (optional)
        lookback: Filter by lookback (optional)
        feature_type: Filter by feature type (optional)

    Returns:
        List of matching column names
    """
    cols = []
    for col in df.columns:
        match = ITH_COLUMN_PATTERN.match(col)
        if not match:
            continue

        col_threshold = int(match.group(1))
        col_lookback = int(match.group(2))
        col_feature_type = match.group(3)

        if threshold is not None and col_threshold != threshold:
            continue
        if lookback is not None and col_lookback != lookback:
            continue
        if feature_type is not None and col_feature_type != feature_type:
            continue

        cols.append(col)

    return sorted(cols)


def get_all_thresholds(df: pl.DataFrame) -> list[int]:
    """Get all unique thresholds from column names.

    Args:
        df: DataFrame with ITH features

    Returns:
        Sorted list of unique thresholds
    """
    thresholds = set()
    for col in df.columns:
        t = extract_threshold(col)
        if t is not None:
            thresholds.add(t)
    return sorted(thresholds)


def get_all_lookbacks(df: pl.DataFrame) -> list[int]:
    """Get all unique lookbacks from column names.

    Args:
        df: DataFrame with ITH features

    Returns:
        Sorted list of unique lookbacks
    """
    lookbacks = set()
    for col in df.columns:
        lb = extract_lookback(col)
        if lb is not None:
            lookbacks.add(lb)
    return sorted(lookbacks)


def get_all_feature_types(df: pl.DataFrame) -> list[str]:
    """Get all unique feature types from column names.

    Args:
        df: DataFrame with ITH features

    Returns:
        Sorted list of unique feature types
    """
    types = set()
    for col in df.columns:
        ft = extract_feature_type(col)
        if ft is not None:
            types.add(ft)
    return sorted(types)


def columns_by_feature_type(df: pl.DataFrame) -> dict[str, list[str]]:
    """Group columns by feature type.

    Args:
        df: DataFrame with ITH features

    Returns:
        Dict mapping feature type to list of columns
    """
    result: dict[str, list[str]] = {}
    for col in df.columns:
        ft = extract_feature_type(col)
        if ft is not None:
            if ft not in result:
                result[ft] = []
            result[ft].append(col)

    # Sort columns within each group
    for ft in result:
        result[ft] = sorted(result[ft])

    return result


def columns_by_lookback(df: pl.DataFrame) -> dict[int, list[str]]:
    """Group columns by lookback.

    Args:
        df: DataFrame with ITH features

    Returns:
        Dict mapping lookback to list of columns
    """
    result: dict[int, list[str]] = {}
    for col in df.columns:
        lb = extract_lookback(col)
        if lb is not None:
            if lb not in result:
                result[lb] = []
            result[lb].append(col)

    # Sort columns within each group
    for lb in result:
        result[lb] = sorted(result[lb])

    return result


def columns_by_threshold(df: pl.DataFrame) -> dict[int, list[str]]:
    """Group columns by threshold.

    Args:
        df: DataFrame with ITH features

    Returns:
        Dict mapping threshold to list of columns
    """
    result: dict[int, list[str]] = {}
    for col in df.columns:
        t = extract_threshold(col)
        if t is not None:
            if t not in result:
                result[t] = []
            result[t].append(col)

    # Sort columns within each group
    for t in result:
        result[t] = sorted(result[t])

    return result
