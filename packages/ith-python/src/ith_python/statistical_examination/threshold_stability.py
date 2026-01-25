"""Threshold stability analysis for ITH features.

Assesses whether features are stable across different range bar thresholds
or if they exhibit significant threshold-dependent behavior.

Uses Polars for data preparation, CV-based stability metrics.

NOTE: Friedman test was REMOVED (2026-01-23) due to independence violation.
The Friedman test assumes mutually independent blocks, but time series rows
are serially correlated. This inflates Type I error rates and produces
invalid p-values. CV-based stability is used instead.

Reference: docs/research/2026-01-23-statistical-methods-verification-gemini.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ith_python.statistical_examination._utils import (
    extract_threshold,
    get_all_feature_types,
    get_all_lookbacks,
    get_feature_columns,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


# Stability CV threshold - features with CV below this are considered stable
CV_STABILITY_THRESHOLD = 0.20


def compute_threshold_stability(
    df: pl.DataFrame,
    lookback: int,
    feature_type: str,
    min_samples: int = 50,
    cv_threshold: float = CV_STABILITY_THRESHOLD,
) -> dict:
    """Assess feature stability across different range bar thresholds.

    Uses coefficient of variation (CV) across threshold means to measure
    stability. CV < 0.20 indicates the feature is threshold-invariant.

    NOTE: Friedman test removed due to time series independence violation.
    Time series blocks are serially correlated, invalidating the test's
    assumption of mutually independent observations.

    Args:
        df: DataFrame with ITH features (must have threshold_dbps column or
            features from multiple thresholds in column names)
        lookback: Lookback window to analyze
        feature_type: Feature type (e.g., 'bull_ed')
        min_samples: Minimum samples required per threshold
        cv_threshold: CV threshold below which feature is considered stable

    Returns:
        Dict with stability analysis results
    """
    # Get columns for this lookback and feature type across thresholds
    cols = get_feature_columns(df, lookback=lookback, feature_type=feature_type)

    if len(cols) < 2:
        return {
            "lookback": lookback,
            "feature_type": feature_type,
            "error": f"Need at least 2 thresholds for stability analysis (found {len(cols)})",
        }

    # Extract threshold for each column
    col_thresholds = [(c, extract_threshold(c) or 0) for c in cols]
    col_thresholds.sort(key=lambda x: x[1])
    cols = [c for c, _ in col_thresholds]
    thresholds = [t for _, t in col_thresholds]

    # Extract numpy arrays for each threshold (drop nulls per column)
    arrays = []
    for col in cols:
        arr = df.get_column(col).drop_nulls().to_numpy()
        if len(arr) < min_samples:
            return {
                "lookback": lookback,
                "feature_type": feature_type,
                "error": f"Insufficient samples for threshold in column {col} ({len(arr)} < {min_samples})",
            }
        arrays.append(arr)

    # Compute mean and std for each threshold
    means = [float(np.mean(a)) for a in arrays]
    stds = [float(np.std(a)) for a in arrays]

    # Coefficient of variation across threshold means
    mean_of_means = np.mean(means)
    cv_across_thresholds = float(np.std(means) / mean_of_means) if mean_of_means > 0 else 0.0

    # Range of means (max - min) as additional stability metric
    mean_range = float(max(means) - min(means)) if means else 0.0

    # Stability determination: CV below threshold
    is_stable = cv_across_thresholds < cv_threshold

    # Compute min length for sample size reporting
    min_len = min(len(a) for a in arrays)

    return {
        "lookback": lookback,
        "feature_type": feature_type,
        "thresholds": thresholds,
        "columns": cols,
        "n_samples": min_len,
        "threshold_means": dict(zip(thresholds, means, strict=True)),
        "threshold_stds": dict(zip(thresholds, stds, strict=True)),
        "cv_across_thresholds": cv_across_thresholds,
        "mean_range": mean_range,
        "stable": is_stable,
        "cv_threshold_used": cv_threshold,
        "note": "Friedman test removed due to time series independence violation",
    }


def compute_all_threshold_stability(
    df: pl.DataFrame,
    lookbacks: Sequence[int] | None = None,
    feature_types: Sequence[str] | None = None,
    min_samples: int = 50,
    cv_threshold: float = CV_STABILITY_THRESHOLD,
) -> dict:
    """Compute threshold stability for all lookback/feature type combinations.

    Args:
        df: DataFrame with ITH features
        lookbacks: Lookbacks to analyze (all if None)
        feature_types: Feature types to analyze (all if None)
        min_samples: Minimum samples required
        cv_threshold: CV threshold for stability determination

    Returns:
        Dict with results and summary statistics
    """
    if lookbacks is None:
        lookbacks = get_all_lookbacks(df)
    if feature_types is None:
        feature_types = get_all_feature_types(df)

    results = []
    for lb in lookbacks:
        for ft in feature_types:
            result = compute_threshold_stability(df, lb, ft, min_samples, cv_threshold)
            results.append(result)

    # Summary statistics
    valid_results = [r for r in results if "error" not in r]
    stable_count = sum(1 for r in valid_results if r.get("stable", False))
    unstable_count = len(valid_results) - stable_count

    # Features with highest CV (most threshold-sensitive)
    high_cv_features = sorted(
        [
            {"lookback": r["lookback"], "feature_type": r["feature_type"], "cv": r["cv_across_thresholds"]}
            for r in valid_results
            if not r.get("stable", True)
        ],
        key=lambda x: x["cv"],
        reverse=True,
    )[:10]

    return {
        "results": results,
        "summary": {
            "total_analyzed": len(valid_results),
            "stable_count": stable_count,
            "unstable_count": unstable_count,
            "stability_rate": stable_count / len(valid_results) if valid_results else 0.0,
            "cv_threshold_used": cv_threshold,
            "high_cv_features": high_cv_features,
            "note": "CV-based stability (Friedman test removed due to independence violation)",
        },
    }


def identify_threshold_invariant_features(
    df: pl.DataFrame,
    stability_cv_threshold: float = 0.15,
    min_samples: int = 50,
) -> dict:
    """Identify features that are threshold-invariant.

    Threshold-invariant features can be computed at any threshold and
    will give similar values - useful for threshold-agnostic analysis.

    Args:
        df: DataFrame with ITH features
        stability_cv_threshold: CV threshold below which feature is considered invariant
        min_samples: Minimum samples required

    Returns:
        Dict with invariant and variant features
    """
    lookbacks = get_all_lookbacks(df)
    feature_types = get_all_feature_types(df)

    invariant = []
    variant = []

    for lb in lookbacks:
        for ft in feature_types:
            result = compute_threshold_stability(df, lb, ft, min_samples, stability_cv_threshold)

            if "error" in result:
                continue

            entry = {
                "lookback": lb,
                "feature_type": ft,
                "cv": result["cv_across_thresholds"],
                "mean_range": result["mean_range"],
            }

            # Invariant if CV below threshold
            if result["stable"]:
                invariant.append(entry)
            else:
                variant.append(entry)

    return {
        "invariant_features": sorted(invariant, key=lambda x: x["cv"]),
        "variant_features": sorted(variant, key=lambda x: -x["cv"]),
        "summary": {
            "n_invariant": len(invariant),
            "n_variant": len(variant),
            "invariance_rate": len(invariant) / (len(invariant) + len(variant)) if (invariant or variant) else 0.0,
            "cv_threshold_used": stability_cv_threshold,
        },
    }


if __name__ == "__main__":
    import sys

    print("Threshold stability analysis module")
    print("Run via: uv run python -m ith_python.statistical_examination.runner")
    sys.exit(0)
