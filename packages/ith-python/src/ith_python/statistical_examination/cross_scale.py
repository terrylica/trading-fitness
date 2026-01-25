"""Cross-scale correlation analysis for ITH features.

Analyzes how features correlate across different lookback windows to identify
redundancy and determine optimal feature scales.

100% Polars-native implementation (no pandas).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from ith_python.statistical_examination._utils import (
    extract_lookback,
    get_all_feature_types,
    get_feature_columns,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def compute_cross_scale_correlation(
    df: pl.DataFrame,
    feature_type: str,
    threshold: int | None = None,
    method: Literal["pearson", "spearman"] = "spearman",
) -> dict:
    """Compute correlation matrix for one feature type across lookbacks.

    Uses Polars-native correlation (pl.corr for Pearson, rank-transform for Spearman).

    Args:
        df: DataFrame with ITH features
        feature_type: Feature type to analyze (e.g., 'bull_ed')
        threshold: Optional threshold filter (analyzes all if None)
        method: Correlation method ('pearson' or 'spearman')

    Returns:
        Dict with correlation matrix and analysis results
    """
    # Get columns for this feature type
    cols = get_feature_columns(df, threshold=threshold, feature_type=feature_type)

    if len(cols) < 2:
        return {
            "feature_type": feature_type,
            "threshold": threshold,
            "method": method,
            "error": f"Insufficient columns ({len(cols)}) for correlation analysis",
        }

    # Extract lookbacks and sort columns by lookback
    col_lookbacks = [(c, extract_lookback(c) or 0) for c in cols]
    col_lookbacks.sort(key=lambda x: x[1])
    cols = [c for c, _ in col_lookbacks]
    lookbacks = [lb for _, lb in col_lookbacks]

    # Drop rows with any nulls in these columns
    df_clean = df.select(cols).drop_nulls()

    if len(df_clean) < 30:
        return {
            "feature_type": feature_type,
            "threshold": threshold,
            "method": method,
            "error": f"Insufficient non-null rows ({len(df_clean)}) for correlation",
        }

    # Build correlation matrix
    n = len(cols)
    corr_matrix = np.eye(n)

    # For Spearman, rank-transform first
    if method == "spearman":
        df_work = df_clean.select([pl.col(c).rank().alias(c) for c in cols])
    else:
        df_work = df_clean

    # Compute pairwise correlations using Polars
    for i in range(n):
        for j in range(i + 1, n):
            corr_val = df_work.select(pl.corr(cols[i], cols[j])).item()
            corr_matrix[i, j] = corr_val if corr_val is not None else np.nan
            corr_matrix[j, i] = corr_matrix[i, j]

    # Find highly correlated pairs (|r| > 0.9)
    highly_correlated = []
    for i in range(n):
        for j in range(i + 1, n):
            r = corr_matrix[i, j]
            if not np.isnan(r) and abs(r) > 0.9:
                highly_correlated.append({
                    "col1": cols[i],
                    "col2": cols[j],
                    "lookback1": lookbacks[i],
                    "lookback2": lookbacks[j],
                    "correlation": float(r),
                })

    # Compute mean correlation (upper triangle, excluding diagonal)
    upper_tri = corr_matrix[np.triu_indices(n, k=1)]
    valid_corrs = upper_tri[~np.isnan(upper_tri)]
    mean_corr = float(np.mean(valid_corrs)) if len(valid_corrs) > 0 else np.nan

    return {
        "feature_type": feature_type,
        "threshold": threshold,
        "method": method,
        "n_features": n,
        "n_samples": len(df_clean),
        "lookbacks": lookbacks,
        "columns": cols,
        "correlation_matrix": corr_matrix.tolist(),
        "mean_correlation": mean_corr,
        "highly_correlated_pairs": highly_correlated,
        "n_highly_correlated": len(highly_correlated),
    }


def compute_all_cross_scale_correlations(
    df: pl.DataFrame,
    thresholds: Sequence[int] | None = None,
    feature_types: Sequence[str] | None = None,
    method: Literal["pearson", "spearman"] = "spearman",
) -> dict:
    """Compute cross-scale correlations for all feature types.

    Args:
        df: DataFrame with ITH features
        thresholds: Thresholds to analyze (all if None)
        feature_types: Feature types to analyze (all if None)
        method: Correlation method

    Returns:
        Dict with results per feature type and summary statistics
    """
    if feature_types is None:
        feature_types = get_all_feature_types(df)

    # If analyzing specific thresholds, compute separately for each
    if thresholds is not None:
        results_by_threshold = {}
        for threshold in thresholds:
            results_by_threshold[threshold] = {}
            for ft in feature_types:
                result = compute_cross_scale_correlation(df, ft, threshold=threshold, method=method)
                results_by_threshold[threshold][ft] = result

        # Summary across all
        all_means = []
        total_highly_correlated = 0
        for threshold_results in results_by_threshold.values():
            for result in threshold_results.values():
                if "mean_correlation" in result and not np.isnan(result.get("mean_correlation", np.nan)):
                    all_means.append(result["mean_correlation"])
                total_highly_correlated += result.get("n_highly_correlated", 0)

        return {
            "by_threshold": results_by_threshold,
            "summary": {
                "method": method,
                "overall_mean_correlation": float(np.mean(all_means)) if all_means else np.nan,
                "total_highly_correlated_pairs": total_highly_correlated,
            },
        }

    # Otherwise, compute across all thresholds combined
    results_by_feature = {}
    for ft in feature_types:
        result = compute_cross_scale_correlation(df, ft, threshold=None, method=method)
        results_by_feature[ft] = result

    # Summary
    all_means = []
    total_highly_correlated = 0
    for result in results_by_feature.values():
        if "mean_correlation" in result and not np.isnan(result.get("mean_correlation", np.nan)):
            all_means.append(result["mean_correlation"])
        total_highly_correlated += result.get("n_highly_correlated", 0)

    return {
        "by_feature_type": results_by_feature,
        "summary": {
            "method": method,
            "mean_correlation_by_feature": {
                ft: r.get("mean_correlation") for ft, r in results_by_feature.items()
            },
            "overall_mean_correlation": float(np.mean(all_means)) if all_means else np.nan,
            "total_highly_correlated_pairs": total_highly_correlated,
        },
    }


def identify_redundant_scales(
    df: pl.DataFrame,
    feature_type: str,
    threshold: int | None = None,
    correlation_threshold: float = 0.95,
    method: Literal["pearson", "spearman"] = "spearman",
) -> dict:
    """Identify lookback scales that are redundant (highly correlated).

    Useful for feature selection - if two scales are highly correlated,
    only one needs to be retained.

    Args:
        df: DataFrame with ITH features
        feature_type: Feature type to analyze
        threshold: Optional threshold filter
        correlation_threshold: Threshold above which scales are considered redundant
        method: Correlation method

    Returns:
        Dict with redundant scale pairs and recommended scales to keep
    """
    corr_result = compute_cross_scale_correlation(df, feature_type, threshold, method)

    if "error" in corr_result:
        return corr_result

    lookbacks = corr_result["lookbacks"]
    corr_matrix = np.array(corr_result["correlation_matrix"])
    n = len(lookbacks)

    # Find redundant pairs
    redundant_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_matrix[i, j]) >= correlation_threshold:
                redundant_pairs.append({
                    "lookback1": lookbacks[i],
                    "lookback2": lookbacks[j],
                    "correlation": float(corr_matrix[i, j]),
                })

    # Greedy selection: keep scales that aren't redundant with already-kept scales
    kept_scales = []
    dropped_scales = []

    for i, lb in enumerate(lookbacks):
        is_redundant = False
        for kept_idx in [lookbacks.index(k) for k in kept_scales]:
            if abs(corr_matrix[i, kept_idx]) >= correlation_threshold:
                is_redundant = True
                dropped_scales.append(lb)
                break
        if not is_redundant:
            kept_scales.append(lb)

    return {
        "feature_type": feature_type,
        "threshold": threshold,
        "correlation_threshold": correlation_threshold,
        "method": method,
        "all_lookbacks": lookbacks,
        "redundant_pairs": redundant_pairs,
        "recommended_scales": kept_scales,
        "redundant_scales": dropped_scales,
        "compression_ratio": len(kept_scales) / len(lookbacks) if lookbacks else 1.0,
    }


if __name__ == "__main__":
    import sys

    print("Cross-scale correlation analysis module")
    print("Run via: uv run python -m ith_python.statistical_examination.runner")
    sys.exit(0)
