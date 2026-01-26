"""Temporal analysis for ITH features.

Analyzes temporal properties of features including:
- Autocorrelation (ACF) at multiple lags
- Stationarity testing (ADF)
- Feature persistence and half-life

Uses Polars for ACF (shift + correlation), scipy for ADF test.

NOTE: ADF interpretation for bounded [0,1] data (2026-01-23):
A true unit root process has variance growing linearly with time (Var(yt) = t*sigma^2).
As t -> infinity, variance approaches infinity. But a variable bounded in [0,1] has
maximum variance of 0.25. Therefore, a bounded variable CANNOT be a unit root process.

The ADF test on bounded data will almost certainly reject the null (boundaries force
mean reversion). Interpret as test of LOCAL PERSISTENCE rather than global stationarity.

Reference: docs/research/2026-01-23-statistical-methods-verification-gemini.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ith_python.statistical_examination._utils import get_feature_columns
from ith_python.telemetry.events import log_hypothesis_result

if TYPE_CHECKING:
    from collections.abc import Sequence


def compute_autocorrelation(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    lags: Sequence[int] = (1, 5, 10, 20, 50, 100),
) -> pl.DataFrame:
    """Compute ACF at multiple lags using Polars shift + correlation.

    100% Polars-native implementation.

    Args:
        df: DataFrame with ITH features
        feature_cols: Features to analyze (auto-detect if None)
        lags: Lag values to compute ACF for

    Returns:
        Polars DataFrame with ACF values and half-life estimates
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    results = []
    for col in feature_cols:
        acf_values = {}

        for lag in lags:
            # Polars-native: shift and correlate
            acf = df.select(pl.corr(col, pl.col(col).shift(lag))).item()
            acf_values[f"acf_lag{lag}"] = float(acf) if acf is not None else None

        # Estimate half-life: find first lag where ACF drops below 0.5
        half_life = None
        for lag in sorted(lags):
            acf_val = acf_values.get(f"acf_lag{lag}")
            if acf_val is not None and acf_val < 0.5:
                half_life = lag
                break

        # Compute first significant lag (ACF < 0.1 or not significant)
        first_insignificant = None
        for lag in sorted(lags):
            acf_val = acf_values.get(f"acf_lag{lag}")
            if acf_val is not None and abs(acf_val) < 0.1:
                first_insignificant = lag
                break

        results.append({
            "feature": col,
            **acf_values,
            "half_life": half_life,
            "first_insignificant_lag": first_insignificant,
        })

    return pl.DataFrame(results)


def compute_stationarity(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    significance_level: float = 0.05,
) -> pl.DataFrame:
    """Test stationarity using simplified ADF test via scipy.

    Implements ADF using OLS regression on differenced series.

    Args:
        df: DataFrame with ITH features
        feature_cols: Features to analyze (auto-detect if None)
        significance_level: P-value threshold for stationarity

    Returns:
        Polars DataFrame with stationarity test results
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    # ADF critical values at 5% (approximate, from Dickey-Fuller table)
    # These depend on sample size; using asymptotic values
    critical_value_5pct = -2.86

    results = []
    for col in feature_cols:
        x = df.get_column(col).drop_nulls().to_numpy()

        if len(x) < 50:
            results.append({
                "feature": col,
                "adf_stat": None,
                "critical_value_5pct": critical_value_5pct,
                "stationary": None,
                "error": f"Insufficient samples ({len(x)})",
            })
            continue

        # ADF test: regress diff(x) on lag(x) and lags of diff(x)
        try:
            adf_stat = _compute_adf_statistic(x)
            is_stationary = adf_stat < critical_value_5pct

            # Emit hypothesis telemetry for ADF stationarity test
            log_hypothesis_result(
                hypothesis_id=f"adf_stationarity_{col}",
                test_name="augmented_dickey_fuller",
                statistic=float(adf_stat),
                p_value=None,  # Using critical value comparison instead
                decision="stationary" if is_stationary else "non_stationary",
                effect_size=float(adf_stat - critical_value_5pct),  # Distance from critical value
                context={
                    "feature": col,
                    "n_samples": len(x),
                    "critical_value_5pct": critical_value_5pct,
                    "note": "For bounded [0,1] data, interpret as LOCAL PERSISTENCE test",
                },
            )

            results.append({
                "feature": col,
                "adf_stat": float(adf_stat),
                "critical_value_5pct": critical_value_5pct,
                "stationary": is_stationary,
                "error": None,
            })
        except (ValueError, np.linalg.LinAlgError) as e:
            results.append({
                "feature": col,
                "adf_stat": None,
                "critical_value_5pct": critical_value_5pct,
                "stationary": None,
                "error": str(e),
            })

    return pl.DataFrame(results)


def _compute_adf_statistic(x: np.ndarray, max_lags: int = 12) -> float:
    """Compute ADF test statistic using OLS.

    Args:
        x: Time series (numpy array)
        max_lags: Maximum number of lagged differences to include

    Returns:
        ADF t-statistic
    """
    n = len(x)

    # First difference
    dx = np.diff(x)

    # Determine optimal lag length using AIC
    n_lags = min(max_lags, int(np.floor((n - 1) ** (1 / 3))))

    # Build regression matrix
    # y = dx[lags:]
    # X = [1, x[lags:-1], dx_lag1, dx_lag2, ...]
    y = dx[n_lags:]
    T = len(y)

    X_cols = [np.ones(T)]  # Constant
    X_cols.append(x[n_lags:-1])  # Lagged level

    # Add lagged differences
    for lag in range(1, n_lags + 1):
        X_cols.append(dx[n_lags - lag : -lag] if lag < len(dx) else np.zeros(T))

    X = np.column_stack(X_cols)

    # OLS
    coeffs, residuals, _rank, _s = np.linalg.lstsq(X, y, rcond=None)

    if len(residuals) == 0:
        # Compute residuals manually
        y_pred = X @ coeffs
        residuals = y - y_pred
        sse = np.sum(residuals**2)
    else:
        sse = residuals[0]

    # Standard error of coefficient on lagged level
    mse = sse / (T - len(coeffs))
    var_coef = mse * np.linalg.inv(X.T @ X)
    se_gamma = np.sqrt(var_coef[1, 1])

    # t-statistic for gamma (coefficient on lagged level)
    gamma = coeffs[1]
    t_stat = gamma / se_gamma

    return t_stat


def analyze_temporal_structure(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    acf_lags: Sequence[int] = (1, 5, 10, 20, 50, 100),
) -> dict:
    """Comprehensive temporal analysis combining ACF and stationarity.

    Args:
        df: DataFrame with ITH features
        feature_cols: Features to analyze (auto-detect if None)
        acf_lags: Lags for ACF computation

    Returns:
        Dict with temporal analysis results and summary
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    # ACF analysis
    acf_df = compute_autocorrelation(df, feature_cols, acf_lags)

    # Stationarity tests
    stationarity_df = compute_stationarity(df, feature_cols)

    # Merge results
    combined = acf_df.join(
        stationarity_df.select(["feature", "adf_stat", "stationary"]),
        on="feature",
        how="left",
    )

    # Summary statistics
    valid_stationarity = stationarity_df.filter(pl.col("stationary").is_not_null())
    n_stationary = valid_stationarity.filter(pl.col("stationary").eq(True)).height
    n_nonstationary = valid_stationarity.filter(pl.col("stationary").eq(False)).height

    # Persistence classification based on half-life
    persistence_classes = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    for row in acf_df.iter_rows(named=True):
        hl = row.get("half_life")
        if hl is None:
            persistence_classes["high"] += 1  # No decay within measured lags
        elif hl <= 5:
            persistence_classes["low"] += 1
        elif hl <= 20:
            persistence_classes["medium"] += 1
        else:
            persistence_classes["high"] += 1

    # Recommended LSTM sequence length based on half-lives
    half_lives = [r["half_life"] for r in acf_df.iter_rows(named=True) if r["half_life"] is not None]
    if half_lives:
        median_half_life = float(np.median(half_lives))
        recommended_seq_len = max(10, min(200, int(median_half_life * 2)))
    else:
        median_half_life = None
        recommended_seq_len = 50  # Default

    # Emit hypothesis telemetry for temporal structure summary
    stationarity_rate = n_stationary / len(valid_stationarity) if len(valid_stationarity) > 0 else None
    log_hypothesis_result(
        hypothesis_id=f"temporal_structure_{len(feature_cols)}feat",
        test_name="temporal_structure_analysis",
        statistic=median_half_life if median_half_life is not None else 0.0,
        p_value=None,
        decision="mostly_stationary" if stationarity_rate and stationarity_rate > 0.8 else "mixed_stationarity",
        effect_size=stationarity_rate,
        context={
            "n_features": len(feature_cols),
            "n_stationary": n_stationary,
            "n_nonstationary": n_nonstationary,
            "stationarity_rate": stationarity_rate,
            "persistence_distribution": persistence_classes,
            "median_half_life": median_half_life,
            "recommended_lstm_seq_len": recommended_seq_len,
        },
    )

    return {
        "acf_results": acf_df.to_dicts(),
        "stationarity_results": stationarity_df.to_dicts(),
        "combined": combined.to_dicts(),
        "summary": {
            "n_features": len(feature_cols),
            "n_stationary": n_stationary,
            "n_nonstationary": n_nonstationary,
            "stationarity_rate": stationarity_rate,
            "persistence_distribution": persistence_classes,
            "median_half_life": median_half_life,
            "recommended_lstm_sequence_length": recommended_seq_len,
        },
    }


def identify_persistent_features(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    min_half_life: int = 20,
) -> list[str]:
    """Identify highly persistent features (slow decay).

    Useful for features that carry long-term information.

    Args:
        df: DataFrame with ITH features
        feature_cols: Features to analyze (auto-detect if None)
        min_half_life: Minimum half-life to be considered persistent

    Returns:
        List of persistent feature names
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    acf_df = compute_autocorrelation(df, feature_cols)

    # Features with half-life >= threshold or no half-life found (very persistent)
    persistent = []
    for row in acf_df.iter_rows(named=True):
        hl = row.get("half_life")
        if hl is None or hl >= min_half_life:
            persistent.append(row["feature"])

    return persistent


def identify_fast_decaying_features(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    max_half_life: int = 5,
) -> list[str]:
    """Identify fast-decaying features (quick decay).

    Useful for features that capture short-term dynamics.

    Args:
        df: DataFrame with ITH features
        feature_cols: Features to analyze (auto-detect if None)
        max_half_life: Maximum half-life to be considered fast-decaying

    Returns:
        List of fast-decaying feature names
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    acf_df = compute_autocorrelation(df, feature_cols)

    fast_decaying = []
    for row in acf_df.iter_rows(named=True):
        hl = row.get("half_life")
        if hl is not None and hl <= max_half_life:
            fast_decaying.append(row["feature"])

    return fast_decaying


if __name__ == "__main__":
    import sys

    print("Temporal analysis module")
    print("Run via: uv run python -m ith_python.statistical_examination.runner")
    sys.exit(0)
