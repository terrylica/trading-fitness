"""Feature selection for ITH features.

Implements filter-based feature selection using:
- Variance threshold
- Correlation-based redundancy removal
- Mutual information with target
- Per-lookback selection for balanced representation

100% Polars-native for filtering, sklearn MI via numpy bridge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from ith_python.statistical_examination._utils import (
    extract_lookback,
    get_feature_columns,
)
from ith_python.statistical_examination.feature_importance import compute_mutual_information

if TYPE_CHECKING:
    from collections.abc import Sequence


def filter_by_variance(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    min_variance: float = 0.01,
) -> list[str]:
    """Filter features by minimum variance.

    100% Polars-native implementation.

    Args:
        df: DataFrame with ITH features
        feature_cols: Features to consider (auto-detect if None)
        min_variance: Minimum variance threshold

    Returns:
        List of features passing variance filter
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    selected = []
    for col in feature_cols:
        var = df.select(pl.col(col).var()).item()
        if var is not None and var >= min_variance:
            selected.append(col)

    return selected


def filter_by_correlation(
    df: pl.DataFrame,
    feature_cols: Sequence[str],
    max_correlation: float = 0.95,
) -> list[str]:
    """Remove highly correlated features (keep first encountered).

    100% Polars-native correlation computation.

    Args:
        df: DataFrame with ITH features
        feature_cols: Features to consider
        max_correlation: Maximum allowed pairwise correlation

    Returns:
        List of features after removing highly correlated ones
    """
    if len(feature_cols) < 2:
        return list(feature_cols)

    selected = []
    df_clean = df.select(feature_cols).drop_nulls()

    for col in feature_cols:
        too_correlated = False
        for existing in selected:
            corr = df_clean.select(pl.corr(col, existing)).item()
            if corr is not None and abs(corr) > max_correlation:
                too_correlated = True
                break

        if not too_correlated:
            selected.append(col)

    return selected


def filter_features(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    target_col: str | None = None,
    min_variance: float = 0.01,
    max_correlation: float = 0.95,
    min_mi_score: float = 0.01,
) -> list[str]:
    """Filter-based feature selection combining multiple criteria.

    Pipeline:
    1. Variance filter (Polars-native)
    2. Correlation filter (Polars-native)
    3. MI filter (sklearn via numpy bridge, if target provided)

    Args:
        df: DataFrame with ITH features
        feature_cols: Features to start with (auto-detect if None)
        target_col: Optional target for MI filtering
        min_variance: Minimum variance threshold
        max_correlation: Maximum pairwise correlation
        min_mi_score: Minimum MI score with target

    Returns:
        List of selected feature names
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    # Step 1: Variance filter
    selected = filter_by_variance(df, feature_cols, min_variance)

    if not selected:
        return []

    # Step 2: Correlation filter
    selected = filter_by_correlation(df, selected, max_correlation)

    if not selected:
        return []

    # Step 3: MI filter (if target provided)
    if target_col and target_col in df.columns and min_mi_score > 0:
        mi_df = compute_mutual_information(df, selected, target_col)
        selected = (
            mi_df.filter(pl.col("mutual_information") >= min_mi_score)["feature"].to_list()
        )

    return selected


def select_per_lookback(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    target_col: str | None = None,
    n_per_lookback: int = 2,
) -> list[str]:
    """Select top N features per lookback scale.

    Ensures balanced representation across time scales.

    Args:
        df: DataFrame with ITH features
        feature_cols: Features to consider (auto-detect if None)
        target_col: Target column for MI ranking (required)
        n_per_lookback: Number of features to select per lookback

    Returns:
        List of selected features (n_per_lookback * n_lookbacks)
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    if not target_col or target_col not in df.columns:
        msg = "target_col is required for per-lookback selection"
        raise ValueError(msg)

    # Compute MI scores
    mi_df = compute_mutual_information(df, feature_cols, target_col)

    # Add lookback column
    mi_df = mi_df.with_columns(
        pl.col("feature").map_elements(
            lambda x: extract_lookback(x) or 0,
            return_dtype=pl.Int64,
        ).alias("lookback")
    )

    # Group by lookback, take top N (Polars-native)
    selected = (
        mi_df.sort("mutual_information", descending=True)
        .group_by("lookback")
        .head(n_per_lookback)["feature"]
        .to_list()
    )

    return selected


def select_per_feature_type(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    target_col: str | None = None,
    n_per_type: int = 3,
) -> list[str]:
    """Select top N features per feature type.

    Ensures representation from all feature categories (bull_ed, bear_ed, etc.).

    Args:
        df: DataFrame with ITH features
        feature_cols: Features to consider (auto-detect if None)
        target_col: Target column for MI ranking (required)
        n_per_type: Number of features to select per type

    Returns:
        List of selected features
    """
    from ith_python.statistical_examination._utils import extract_feature_type

    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    if not target_col or target_col not in df.columns:
        msg = "target_col is required for per-type selection"
        raise ValueError(msg)

    # Compute MI scores
    mi_df = compute_mutual_information(df, feature_cols, target_col)

    # Add feature type column
    mi_df = mi_df.with_columns(
        pl.col("feature").map_elements(
            lambda x: extract_feature_type(x) or "unknown",
            return_dtype=pl.Utf8,
        ).alias("feature_type")
    )

    # Group by feature type, take top N
    selected = (
        mi_df.sort("mutual_information", descending=True)
        .group_by("feature_type")
        .head(n_per_type)["feature"]
        .to_list()
    )

    return selected


def select_optimal_subset(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    target_col: str | None = None,
    max_features: int = 24,
    min_variance: float = 0.01,
    max_correlation: float = 0.90,
) -> dict:
    """Select optimal feature subset combining multiple strategies.

    Strategy:
    1. Apply variance and correlation filters
    2. Rank by MI with target
    3. Take top features up to max_features

    Args:
        df: DataFrame with ITH features
        feature_cols: Features to start with (auto-detect if None)
        target_col: Target column (required for optimal selection)
        max_features: Maximum features in final subset
        min_variance: Minimum variance threshold
        max_correlation: Maximum pairwise correlation

    Returns:
        Dict with selected features and selection metadata
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    initial_count = len(feature_cols)

    # Apply filters
    filtered = filter_features(
        df,
        feature_cols,
        target_col=None,  # Don't MI filter here, we'll rank by MI
        min_variance=min_variance,
        max_correlation=max_correlation,
        min_mi_score=0.0,
    )

    after_filter_count = len(filtered)

    # If target provided, rank by MI and take top max_features
    if target_col and target_col in df.columns and filtered:
        mi_df = compute_mutual_information(df, filtered, target_col)
        selected = mi_df.head(max_features)["feature"].to_list()
        mi_scores = dict(
            zip(mi_df["feature"].to_list(), mi_df["mutual_information"].to_list())
        )
    else:
        selected = filtered[:max_features]
        mi_scores = {}

    return {
        "initial_features": initial_count,
        "after_variance_correlation_filter": after_filter_count,
        "final_selected": len(selected),
        "selected_features": selected,
        "mi_scores": {f: mi_scores.get(f) for f in selected} if mi_scores else None,
        "reduction_ratio": len(selected) / initial_count if initial_count > 0 else 0.0,
    }


def get_feature_selection_summary(selection_result: dict) -> str:
    """Generate human-readable summary of feature selection.

    Args:
        selection_result: Result from select_optimal_subset()

    Returns:
        Summary string
    """
    return (
        f"Feature selection: {selection_result['initial_features']} -> "
        f"{selection_result['after_variance_correlation_filter']} (filter) -> "
        f"{selection_result['final_selected']} (final). "
        f"Reduction: {(1 - selection_result['reduction_ratio']) * 100:.1f}%"
    )


if __name__ == "__main__":
    import sys

    print("Feature selection module")
    print("Run via: uv run python -m ith_python.statistical_examination.runner")
    sys.exit(0)
