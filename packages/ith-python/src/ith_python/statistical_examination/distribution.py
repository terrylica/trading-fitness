"""Distribution analysis for ITH features.

Analyzes empirical distributions of ITH features, including:
- Descriptive statistics (Polars-native)
- Normality tests (scipy bridge) - W as continuous metric, not binary filter
- Beta distribution fitting (natural for [0,1] bounded features)
- Anderson-Darling goodness-of-fit with parametric bootstrap

Uses Polars for descriptive stats, scipy for statistical tests.

NOTE: KS test replaced with Anderson-Darling (2026-01-23).
Standard KS p-values are invalid when params estimated from data.
AD emphasizes tails (critical for financial data) and uses parametric
bootstrap via scipy.stats.goodness_of_fit for valid p-values.

Reference: docs/research/2026-01-23-statistical-methods-verification-gemini.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from ith_python.statistical_examination._utils import get_feature_columns

if TYPE_CHECKING:
    from collections.abc import Sequence


# Shapiro-Wilk W interpretation thresholds
# W > 0.99: Practically normal
# 0.95 < W < 0.99: Minor departures, acceptable for parametric methods
# 0.90 < W < 0.95: Moderate non-normality
# W < 0.90: Substantial non-normality
SHAPIRO_THRESHOLDS = {
    "practically_normal": 0.99,
    "minor_departures": 0.95,
    "moderate_non_normality": 0.90,
}


def _classify_gaussianity(w_stat: float) -> str:
    """Classify Gaussianity based on Shapiro-Wilk W statistic.

    Args:
        w_stat: Shapiro-Wilk W statistic

    Returns:
        Classification string
    """
    if w_stat > SHAPIRO_THRESHOLDS["practically_normal"]:
        return "practically_normal"
    elif w_stat > SHAPIRO_THRESHOLDS["minor_departures"]:
        return "minor_departures"
    elif w_stat > SHAPIRO_THRESHOLDS["moderate_non_normality"]:
        return "moderate_non_normality"
    else:
        return "substantial_non_normality"


def analyze_beta_fit(
    values: np.ndarray,
    fast_screen: bool = True,
    random_state: int = 42,
) -> dict:
    """Two-stage Anderson-Darling test for Beta goodness-of-fit.

    Stage 1: B=99 iterations at alpha=0.10 (fast screening)
    Stage 2: B=999 only for borderline cases (0.001 < p < 0.10)

    AD test emphasizes tails (critical for financial data) and uses
    parametric bootstrap for valid p-values when params are estimated.

    Args:
        values: Data array (should be in [0,1])
        fast_screen: If True, use two-stage approach for efficiency
        random_state: Random seed for reproducibility

    Returns:
        Dict with AD statistic, p-value, and fit assessment
    """
    # Clip values slightly away from exact 0 and 1 for numerical stability
    clipped = np.clip(values, 1e-10, 1 - 1e-10)

    # Stage 1: Fast screen with B=99
    n_mc = 99 if fast_screen else 999
    try:
        result = sp_stats.goodness_of_fit(
            sp_stats.beta,
            clipped,
            statistic="ad",  # Anderson-Darling (better for tails)
            known_params={"loc": 0, "scale": 1},
            n_mc_samples=n_mc,
            random_state=random_state,
        )
        p_value = float(result.pvalue)
        ad_stat = float(result.statistic)
        stage = 1

        # Stage 2: Re-run with more iterations if borderline
        if fast_screen and 0.001 < p_value < 0.10:
            result = sp_stats.goodness_of_fit(
                sp_stats.beta,
                clipped,
                statistic="ad",
                known_params={"loc": 0, "scale": 1},
                n_mc_samples=999,
                random_state=random_state,
            )
            p_value = float(result.pvalue)
            ad_stat = float(result.statistic)
            stage = 2

        # Also fit Beta params for reporting
        alpha, beta_param, _, _ = sp_stats.beta.fit(clipped, floc=0, fscale=1)

        return {
            "alpha": float(alpha),
            "beta": float(beta_param),
            "ad_statistic": ad_stat,
            "ad_p_value": p_value,
            "fits_well": p_value > 0.05,
            "stage_used": stage,
            "method": "Anderson-Darling with parametric bootstrap",
        }
    except (ValueError, RuntimeError) as e:
        return {"error": str(e)}


def analyze_distribution(
    df: pl.DataFrame,
    feature_col: str,
    max_shapiro_samples: int = 5000,
    use_ad_test: bool = True,
) -> dict:
    """Comprehensive distribution analysis for a single feature.

    Polars-native: mean, std, skewness, kurtosis, quantiles
    Scipy bridge: Shapiro-Wilk (W as metric), Beta fit with AD test

    Args:
        df: DataFrame with ITH features
        feature_col: Column name to analyze
        max_shapiro_samples: Max samples for Shapiro test (default 5000)
        use_ad_test: If True, use Anderson-Darling for Beta fit (recommended)

    Returns:
        Dict with distribution statistics and test results
    """
    if feature_col not in df.columns:
        return {"feature": feature_col, "error": f"Column {feature_col} not found"}

    # Polars-native descriptive statistics
    stats_result = df.select([
        pl.col(feature_col).count().alias("n"),
        pl.col(feature_col).null_count().alias("n_null"),
        pl.col(feature_col).mean().alias("mean"),
        pl.col(feature_col).std().alias("std"),
        pl.col(feature_col).min().alias("min"),
        pl.col(feature_col).max().alias("max"),
        pl.col(feature_col).skew().alias("skewness"),
        pl.col(feature_col).kurtosis().alias("kurtosis"),
        pl.col(feature_col).quantile(0.01).alias("q01"),
        pl.col(feature_col).quantile(0.05).alias("q05"),
        pl.col(feature_col).quantile(0.25).alias("q25"),
        pl.col(feature_col).quantile(0.50).alias("q50"),
        pl.col(feature_col).quantile(0.75).alias("q75"),
        pl.col(feature_col).quantile(0.95).alias("q95"),
        pl.col(feature_col).quantile(0.99).alias("q99"),
    ]).row(0, named=True)

    # Convert Polars types to Python native
    stats = {k: float(v) if v is not None else None for k, v in stats_result.items()}
    stats["n"] = int(stats_result["n"]) if stats_result["n"] is not None else 0
    stats["n_null"] = int(stats_result["n_null"]) if stats_result["n_null"] is not None else 0

    # Scipy bridge for statistical tests
    values = df.get_column(feature_col).drop_nulls().to_numpy()

    if len(values) < 10:
        return {
            "feature": feature_col,
            **stats,
            "error": f"Insufficient non-null values ({len(values)}) for distribution tests",
        }

    # Shapiro-Wilk normality test (scipy limit is 5000)
    # NOTE: Use W as continuous metric, not p-value binary filter
    # At N>5000, test is overpowered and bounded data cannot be normal
    shapiro_sample = values[:max_shapiro_samples] if len(values) > max_shapiro_samples else values
    try:
        shapiro_stat, shapiro_p = sp_stats.shapiro(shapiro_sample)
        gaussianity = _classify_gaussianity(shapiro_stat)
    except ValueError:
        # Shapiro fails with constant or insufficient data
        shapiro_stat, shapiro_p = np.nan, np.nan
        gaussianity = "unknown"

    # D'Agostino-Pearson normality test (no sample limit)
    try:
        dagostino_stat, dagostino_p = sp_stats.normaltest(values)
    except ValueError:
        # normaltest fails with < 20 samples or constant data
        dagostino_stat, dagostino_p = np.nan, np.nan

    # Beta distribution fit
    if use_ad_test:
        # Use two-stage Anderson-Darling with parametric bootstrap
        beta_fit = analyze_beta_fit(values)
    else:
        # Legacy KS test (DEPRECATED - p-values invalid when params estimated)
        clipped = np.clip(values, 1e-10, 1 - 1e-10)
        try:
            alpha, beta_param, loc, scale = sp_stats.beta.fit(clipped, floc=0, fscale=1)
            ks_stat, ks_p = sp_stats.kstest(clipped, "beta", args=(alpha, beta_param, loc, scale))
            beta_fit = {
                "alpha": float(alpha),
                "beta": float(beta_param),
                "ks_stat": float(ks_stat),
                "ks_p": float(ks_p),
                "fits_well": ks_p > 0.05,
                "method": "KS test (DEPRECATED - use AD test)",
            }
        except (ValueError, RuntimeError) as e:
            beta_fit = {"error": str(e)}

    # Detect distribution shape
    shape = _classify_distribution_shape(stats, beta_fit)

    return {
        "feature": feature_col,
        **stats,
        "normality_test": {
            "shapiro_w": float(shapiro_stat) if not np.isnan(shapiro_stat) else None,
            "shapiro_p": float(shapiro_p) if not np.isnan(shapiro_p) else None,
            "gaussianity": gaussianity,
            "dagostino_stat": float(dagostino_stat) if not np.isnan(dagostino_stat) else None,
            "dagostino_p": float(dagostino_p) if not np.isnan(dagostino_p) else None,
            "note": "W is continuous Gaussianity metric; p-values unreliable at N>5000",
        },
        "beta_fit": beta_fit,
        "distribution_shape": shape,
    }


def _classify_distribution_shape(stats: dict, beta_fit: dict) -> dict:
    """Classify distribution shape based on statistics.

    Args:
        stats: Descriptive statistics dict
        beta_fit: Beta fit results dict

    Returns:
        Dict with shape classification
    """
    skewness = stats.get("skewness")
    kurtosis = stats.get("kurtosis")

    shape = "unknown"
    details = []

    if skewness is not None:
        if abs(skewness) < 0.5:
            shape = "symmetric"
            details.append("approximately symmetric")
        elif skewness > 0.5:
            shape = "right_skewed"
            details.append("right-skewed (positive skew)")
        else:
            shape = "left_skewed"
            details.append("left-skewed (negative skew)")

    if kurtosis is not None:
        if kurtosis > 0.5:
            details.append("leptokurtic (heavy tails)")
        elif kurtosis < -0.5:
            details.append("platykurtic (light tails)")
        else:
            details.append("mesokurtic (normal-like tails)")

    # Check for uniform distribution
    if "alpha" in beta_fit and "beta" in beta_fit:
        alpha = beta_fit["alpha"]
        beta_param = beta_fit["beta"]
        if 0.8 < alpha < 1.2 and 0.8 < beta_param < 1.2:
            shape = "uniform"
            details.append("approximately uniform (Beta(1,1))")
        elif alpha < 1 and beta_param < 1:
            shape = "u_shaped"
            details.append("U-shaped (Beta with alpha,beta < 1)")
        elif alpha > 1 and beta_param > 1:
            if abs(alpha - beta_param) < 0.5:
                shape = "bell_shaped"
                details.append("bell-shaped (symmetric Beta)")

    return {"shape": shape, "details": details}


def analyze_all_distributions(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    threshold: int | None = None,
    lookback: int | None = None,
    use_ad_test: bool = True,
) -> dict:
    """Analyze distributions for all specified features.

    Args:
        df: DataFrame with ITH features
        feature_cols: Columns to analyze (auto-detect if None)
        threshold: Optional threshold filter
        lookback: Optional lookback filter
        use_ad_test: If True, use Anderson-Darling for Beta fit

    Returns:
        Dict with all distribution results and summary
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df, threshold=threshold, lookback=lookback)

    results = []
    for col in feature_cols:
        result = analyze_distribution(df, col, use_ad_test=use_ad_test)
        results.append(result)

    # Summary statistics
    valid_results = [r for r in results if "error" not in r]

    # Count by Gaussianity classification
    gaussianity_counts = {}
    for r in valid_results:
        g = r.get("normality_test", {}).get("gaussianity", "unknown")
        gaussianity_counts[g] = gaussianity_counts.get(g, 0) + 1

    n_beta_fits = sum(
        1
        for r in valid_results
        if r.get("beta_fit", {}).get("fits_well", False)
    )

    # Group by shape
    shapes = {}
    for r in valid_results:
        shape = r.get("distribution_shape", {}).get("shape", "unknown")
        shapes[shape] = shapes.get(shape, 0) + 1

    return {
        "results": results,
        "summary": {
            "total_analyzed": len(valid_results),
            "gaussianity_distribution": gaussianity_counts,
            "beta_fits_well": n_beta_fits,
            "shapes": shapes,
            "beta_fit_rate": n_beta_fits / len(valid_results) if valid_results else 0.0,
            "method": "Anderson-Darling with parametric bootstrap" if use_ad_test else "KS test (deprecated)",
        },
    }


def compute_outlier_stats(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    iqr_multiplier: float = 1.5,
) -> pl.DataFrame:
    """Compute outlier statistics for features using IQR method.

    Args:
        df: DataFrame with ITH features
        feature_cols: Columns to analyze (auto-detect if None)
        iqr_multiplier: IQR multiplier for outlier bounds (default 1.5)

    Returns:
        Polars DataFrame with outlier statistics per feature
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    results = []
    for col in feature_cols:
        # Polars-native quantiles
        q1, q3 = df.select([
            pl.col(col).quantile(0.25).alias("q1"),
            pl.col(col).quantile(0.75).alias("q3"),
        ]).row(0)

        if q1 is None or q3 is None:
            continue

        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        # Count outliers (Polars-native)
        outlier_stats = df.select([
            pl.col(col).count().alias("n_total"),
            ((pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)).sum().alias("n_outliers"),
        ]).row(0, named=True)

        n_total = outlier_stats["n_total"] or 0
        n_outliers = outlier_stats["n_outliers"] or 0

        results.append({
            "feature": col,
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "n_total": int(n_total),
            "n_outliers": int(n_outliers),
            "outlier_rate": float(n_outliers / n_total) if n_total > 0 else 0.0,
        })

    return pl.DataFrame(results)


if __name__ == "__main__":
    import sys

    print("Distribution analysis module")
    print("Run via: uv run python -m ith_python.statistical_examination.runner")
    sys.exit(0)
