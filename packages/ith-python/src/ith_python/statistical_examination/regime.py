"""Regime dependence analysis for ITH features.

Analyzes how features behave in different market regimes:
- Trending markets (H > 0.55)
- Mean-reverting markets (H < 0.45)
- Random walk (0.45 <= H <= 0.55)

Uses Polars for data filtering, scipy.stats for statistical tests.

NOTE: Effect size metrics updated (2026-01-23):
- Cohen's d: Fixed to use weighted pooled SD formula for unequal samples
- Cliff's Delta: Added as primary non-parametric effect size
- Finance-specific thresholds: |d|<0.05 negligible, >0.15 strong

Reference: docs/research/2026-01-23-statistical-methods-verification-gemini.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from ith_python.statistical_examination._utils import get_feature_columns
from ith_python.telemetry.events import log_hypothesis_result

if TYPE_CHECKING:
    from collections.abc import Sequence


RegimeType = Literal["trending", "mean_reverting", "random", "warmup"]


# Finance-specific Cliff's Delta thresholds (NOT Cohen's behavioral science 0.2/0.5/0.8)
# In efficient markets, large effect sizes are suspicious (check for look-ahead bias)
CLIFFS_DELTA_THRESHOLDS = {
    "negligible": 0.05,  # |delta| < 0.05 - likely noise
    "small_tradable": 0.15,  # 0.05 <= |delta| < 0.15 - small but potentially tradable
    "medium_strong": 0.30,  # 0.15 <= |delta| < 0.30 - strong signal
    # |delta| >= 0.30 - large/suspicious (check for look-ahead bias)
}


def cohens_d_corrected(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d with correct weighted pooled SD for unequal samples.

    Uses the formula: s_pooled = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))
    NOT the simple average sqrt((s1^2 + s2^2)/2) which is wrong for unequal n.

    Args:
        x: First sample array
        y: Second sample array

    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return 0.0

    var1 = np.var(x, ddof=1)
    var2 = np.var(y, ddof=1)

    # Weighted pooled standard deviation (Welch's formula)
    pooled_std = np.sqrt(
        ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    )

    if pooled_std == 0:
        return 0.0

    return float((np.mean(x) - np.mean(y)) / pooled_std)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's Delta - non-parametric effect size.

    Mathematically identical to rank-biserial correlation.
    Measures the probability that a randomly selected value from x
    is greater than a randomly selected value from y, minus the
    probability of the reverse.

    Formula: delta = (2*U / (n1*n2)) - 1
    where U is the Mann-Whitney U statistic.

    Args:
        x: First sample array
        y: Second sample array

    Returns:
        Cliff's Delta in range [-1, 1]
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0.0

    # Use Mann-Whitney U to compute Cliff's Delta efficiently
    U, _ = sp_stats.mannwhitneyu(x, y, alternative="two-sided")

    # delta = 1 - 2*U/(n1*n2)
    # Note: mannwhitneyu returns U for x, so we use this formula
    delta = 1 - 2 * U / (n1 * n2)

    return float(delta)


def interpret_cliffs_delta(delta: float) -> str:
    """Finance-specific interpretation of Cliff's Delta.

    Uses thresholds appropriate for financial data, NOT Cohen's
    behavioral science thresholds (0.2/0.5/0.8).

    Args:
        delta: Cliff's Delta value

    Returns:
        Interpretation string
    """
    abs_d = abs(delta)
    if abs_d < CLIFFS_DELTA_THRESHOLDS["negligible"]:
        return "negligible"
    elif abs_d < CLIFFS_DELTA_THRESHOLDS["small_tradable"]:
        return "small_tradable"
    elif abs_d < CLIFFS_DELTA_THRESHOLDS["medium_strong"]:
        return "medium_strong"
    else:
        return "large_suspicious"  # Check for look-ahead bias!


def compute_hurst_exponent(series: np.ndarray) -> float:
    """Compute Hurst exponent using R/S analysis.

    H > 0.5: Trending (persistent)
    H < 0.5: Mean-reverting (anti-persistent)
    H = 0.5: Random walk

    Args:
        series: Price or NAV series

    Returns:
        Hurst exponent estimate
    """
    if len(series) < 20:
        return 0.5  # Default to random walk for insufficient data

    # R/S analysis
    n = len(series)
    max_k = min(n // 2, 100)  # Limit to reasonable number of lags

    if max_k < 10:
        return 0.5

    rs_values = []
    ns = []

    for k in range(10, max_k + 1, max(1, max_k // 20)):
        # Divide into k segments
        n_segments = n // k
        if n_segments < 1:
            continue

        rs_list = []
        for i in range(n_segments):
            segment = series[i * k : (i + 1) * k]
            if len(segment) < 2:
                continue

            mean = np.mean(segment)
            deviations = segment - mean
            cumulative = np.cumsum(deviations)
            r = np.max(cumulative) - np.min(cumulative)
            s = np.std(segment, ddof=1)

            if s > 0:
                rs_list.append(r / s)

        if rs_list:
            rs_values.append(np.mean(rs_list))
            ns.append(k)

    if len(ns) < 3:
        return 0.5

    # Linear regression in log-log space
    log_n = np.log(ns)
    log_rs = np.log(rs_values)

    # Simple OLS for slope (Hurst exponent)
    slope, _, _, _, _ = sp_stats.linregress(log_n, log_rs)

    # Clamp to valid range [0, 1]
    return float(np.clip(slope, 0.0, 1.0))


def detect_regime(
    nav: np.ndarray,
    lookback: int = 100,
    trending_threshold: float = 0.55,
    mean_rev_threshold: float = 0.45,
) -> np.ndarray:
    """Classify each bar into market regime using rolling Hurst exponent.

    Args:
        nav: NAV series (normalized prices)
        lookback: Rolling window for Hurst calculation
        trending_threshold: H above this is trending
        mean_rev_threshold: H below this is mean-reverting

    Returns:
        Array of regime labels ('trending', 'mean_reverting', 'random', 'warmup')
    """
    n = len(nav)
    regimes = np.empty(n, dtype="<U15")
    regimes[:lookback] = "warmup"

    for i in range(lookback, n):
        window = nav[i - lookback : i]
        h = compute_hurst_exponent(window)

        if h > trending_threshold:
            regimes[i] = "trending"
        elif h < mean_rev_threshold:
            regimes[i] = "mean_reverting"
        else:
            regimes[i] = "random"

    return regimes


def analyze_regime_dependence(
    df: pl.DataFrame,
    regimes: np.ndarray,
    feature_cols: Sequence[str] | None = None,
    min_samples_per_regime: int = 30,
) -> list[dict]:
    """Compare feature distributions across regimes.

    Uses Mann-Whitney U test to detect significant differences between
    trending and mean-reverting regimes. Reports both Cohen's d (corrected)
    and Cliff's Delta (non-parametric, primary effect size).

    Args:
        df: DataFrame with ITH features
        regimes: Array of regime labels (same length as df)
        feature_cols: Features to analyze (auto-detect if None)
        min_samples_per_regime: Minimum samples required per regime

    Returns:
        List of dicts with regime dependence analysis for each feature
    """
    if len(regimes) != len(df):
        msg = f"Regime array length ({len(regimes)}) != DataFrame length ({len(df)})"
        raise ValueError(msg)

    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    # Add regime column to Polars DataFrame
    df_with_regime = df.with_columns(pl.Series("_regime", regimes))

    results = []
    for col in feature_cols:
        # Polars-native filtering
        trending = (
            df_with_regime.filter(pl.col("_regime") == "trending")
            .get_column(col)
            .drop_nulls()
            .to_numpy()
        )
        mean_rev = (
            df_with_regime.filter(pl.col("_regime") == "mean_reverting")
            .get_column(col)
            .drop_nulls()
            .to_numpy()
        )
        random_walk = (
            df_with_regime.filter(pl.col("_regime") == "random")
            .get_column(col)
            .drop_nulls()
            .to_numpy()
        )

        # Check minimum samples
        if len(trending) < min_samples_per_regime or len(mean_rev) < min_samples_per_regime:
            results.append({
                "feature": col,
                "error": f"Insufficient samples (trending={len(trending)}, mean_rev={len(mean_rev)})",
            })
            continue

        # Mann-Whitney U test: trending vs mean-reverting
        try:
            stat, p_value = sp_stats.mannwhitneyu(trending, mean_rev, alternative="two-sided")
        except ValueError as e:
            results.append({"feature": col, "error": f"Mann-Whitney test failed: {e}"})
            continue

        # Effect sizes
        # 1. Cohen's d (corrected formula for unequal samples)
        d = cohens_d_corrected(trending, mean_rev)

        # 2. Cliff's Delta (non-parametric, primary effect size)
        delta = cliffs_delta(trending, mean_rev)
        delta_interpretation = interpret_cliffs_delta(delta)

        # Emit hypothesis telemetry for Mann-Whitney regime test
        log_hypothesis_result(
            hypothesis_id=f"regime_mann_whitney_{col}",
            test_name="mann_whitney_u",
            statistic=float(stat),
            p_value=float(p_value),
            decision="regime_dependent" if p_value < 0.05 else "regime_invariant",
            effect_size=delta,  # Cliff's Delta as primary effect size
            context={
                "feature": col,
                "n_trending": len(trending),
                "n_mean_reverting": len(mean_rev),
                "cohens_d": d,
                "cliffs_delta": delta,
                "effect_magnitude": delta_interpretation,
            },
        )

        results.append({
            "feature": col,
            "n_trending": len(trending),
            "n_mean_reverting": len(mean_rev),
            "n_random": len(random_walk),
            "trending_mean": float(np.mean(trending)),
            "trending_std": float(np.std(trending)),
            "mean_rev_mean": float(np.mean(mean_rev)),
            "mean_rev_std": float(np.std(mean_rev)),
            "random_mean": float(np.mean(random_walk)) if len(random_walk) > 0 else None,
            "mann_whitney_stat": float(stat),
            "mann_whitney_p": float(p_value),
            "cohens_d": d,
            "cliffs_delta": delta,
            "effect_magnitude": delta_interpretation,  # Based on Cliff's Delta (finance thresholds)
            "regime_dependent": p_value < 0.05,
        })

    return results


def summarize_regime_dependence(results: list[dict]) -> dict:
    """Summarize regime dependence analysis results.

    Args:
        results: List of per-feature regime analysis results

    Returns:
        Summary statistics dict
    """
    valid_results = [r for r in results if "error" not in r]

    n_regime_dependent = sum(1 for r in valid_results if r.get("regime_dependent", False))
    n_regime_invariant = len(valid_results) - n_regime_dependent

    # Group by effect magnitude (finance-specific thresholds)
    effect_counts = {"negligible": 0, "small_tradable": 0, "medium_strong": 0, "large_suspicious": 0}
    for r in valid_results:
        mag = r.get("effect_magnitude", "negligible")
        effect_counts[mag] = effect_counts.get(mag, 0) + 1

    # Features with strongest regime dependence (sorted by |Cliff's Delta|)
    strongest = sorted(
        [r for r in valid_results if r.get("regime_dependent", False)],
        key=lambda x: abs(x.get("cliffs_delta", 0)),
        reverse=True,
    )[:10]

    return {
        "total_analyzed": len(valid_results),
        "regime_dependent": n_regime_dependent,
        "regime_invariant": n_regime_invariant,
        "dependence_rate": n_regime_dependent / len(valid_results) if valid_results else 0.0,
        "effect_magnitude_distribution": effect_counts,
        "strongest_regime_effects": [
            {
                "feature": r["feature"],
                "cliffs_delta": r["cliffs_delta"],
                "cohens_d": r["cohens_d"],
                "p_value": r["mann_whitney_p"],
            }
            for r in strongest
        ],
        "note": "Effect magnitude uses finance-specific Cliff's Delta thresholds (not Cohen's 0.2/0.5/0.8)",
    }


def compute_regime_statistics(
    df: pl.DataFrame,
    regimes: np.ndarray,
) -> dict:
    """Compute overall regime statistics.

    Args:
        df: DataFrame (for length reference)
        regimes: Regime labels array

    Returns:
        Dict with regime distribution statistics
    """
    unique, counts = np.unique(regimes, return_counts=True)
    regime_counts = dict(zip(unique.tolist(), counts.tolist(), strict=True))

    total = len(regimes)
    return {
        "total_bars": total,
        "regime_counts": regime_counts,
        "regime_proportions": {k: v / total for k, v in regime_counts.items()},
    }


if __name__ == "__main__":
    import sys

    print("Regime dependence analysis module")
    print("Run via: uv run python -m ith_python.statistical_examination.runner")
    sys.exit(0)
