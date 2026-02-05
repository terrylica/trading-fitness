"""PCMCI causal discovery for feature selection.

Implements Phase 3 of the principled feature selection pipeline:
- Causal discovery using Peter-Clark Momentary Conditional Independence
- Explicitly handles autocorrelation via MCI test
- Selects features with direct causal links to target

GitHub Issue: https://github.com/terrylica/cc-skills/issues/21

100% Polars-native for data handling, tigramite via numpy bridge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from ith_python.statistical_examination._utils import get_feature_columns
from ith_python.statistical_examination.suppression import filter_suppressed

if TYPE_CHECKING:
    from collections.abc import Sequence


def filter_pcmci(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    target_col: str = "target",
    alpha: float = 0.05,
    tau_max: int = 5,
    *,
    apply_suppression: bool = True,
    emit_telemetry: bool = False,
) -> list[str]:
    """PCMCI-based causal feature selection (Phase 3: 30→15).

    Uses PCMCI algorithm to identify features with direct causal links
    to the target, accounting for autocorrelation in time series.

    Args:
        df: DataFrame with features and target column
        feature_cols: Features to consider (auto-detect if None)
        target_col: Name of target column for causal analysis
        alpha: Significance level for MCI test (default 0.05)
        tau_max: Maximum lag for causal analysis (default 5)
        apply_suppression: If True, filter suppressed features first
        emit_telemetry: If True, log events to NDJSON

    Returns:
        List of features with significant causal links to target

    Raises:
        ValueError: If target_col not in DataFrame or insufficient data
    """
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    if target_col not in df.columns:
        msg = f"Target column '{target_col}' not found in DataFrame"
        raise ValueError(msg)

    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    if not feature_cols:
        return []

    if apply_suppression:
        feature_cols = filter_suppressed(feature_cols, emit_telemetry=emit_telemetry)

    if not feature_cols:
        return []

    # Prepare data for tigramite
    df_clean = df.select([*feature_cols, target_col]).drop_nulls()

    if df_clean.height < 50:
        msg = f"Insufficient data for PCMCI: {df_clean.height} rows (need >= 50)"
        raise ValueError(msg)

    # Build variable names (features + target)
    var_names = list(feature_cols) + [target_col]
    target_idx = len(feature_cols)  # Target is last variable

    # Convert to numpy array (T x N)
    data_array = df_clean.select(var_names).to_numpy()

    # Create tigramite dataframe
    dataframe = pp.DataFrame(data_array, var_names=var_names)

    # Run PCMCI with ParCorr test (handles autocorrelation)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)

    # Run causal discovery
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha)

    # Extract features with significant causal links to target
    p_matrix = results["p_matrix"]
    selected = []

    for i, feature in enumerate(feature_cols):
        # Check if feature has significant link to target at any lag
        for tau in range(tau_max + 1):
            # p_matrix[i, target_idx, tau] = p-value for feature→target at lag tau
            if p_matrix[i, target_idx, tau] < alpha:
                selected.append(feature)
                break  # Found significant link, no need to check other lags

    if emit_telemetry:
        _log_pcmci_selection(
            initial_count=len(feature_cols),
            selected_count=len(selected),
            alpha=alpha,
            tau_max=tau_max,
            selected_features=selected,
        )

    return selected


def compute_causal_strengths(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    target_col: str = "target",
    tau_max: int = 5,
    *,
    apply_suppression: bool = True,
) -> pl.DataFrame:
    """Compute causal strength from each feature to target.

    Returns detailed causal analysis including lag-specific p-values
    and effect sizes.

    Args:
        df: DataFrame with features and target column
        feature_cols: Features to analyze (auto-detect if None)
        target_col: Name of target column
        tau_max: Maximum lag for causal analysis

    Returns:
        DataFrame with columns: feature, min_pvalue, best_lag, causal_strength
    """
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    if target_col not in df.columns:
        msg = f"Target column '{target_col}' not found in DataFrame"
        raise ValueError(msg)

    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    if apply_suppression:
        feature_cols = filter_suppressed(feature_cols)

    if not feature_cols:
        return pl.DataFrame(schema={
            "feature": pl.Utf8,
            "min_pvalue": pl.Float64,
            "best_lag": pl.Int64,
            "causal_strength": pl.Float64,
        })

    # Prepare data
    df_clean = df.select([*feature_cols, target_col]).drop_nulls()

    if df_clean.height < 50:
        return pl.DataFrame(schema={
            "feature": pl.Utf8,
            "min_pvalue": pl.Float64,
            "best_lag": pl.Int64,
            "causal_strength": pl.Float64,
        })

    var_names = list(feature_cols) + [target_col]
    target_idx = len(feature_cols)

    data_array = df_clean.select(var_names).to_numpy()
    dataframe = pp.DataFrame(data_array, var_names=var_names)

    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=0.2)

    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]

    # Build results
    records = []
    for i, feature in enumerate(feature_cols):
        # Find minimum p-value and best lag
        min_pvalue = 1.0
        best_lag = 0
        best_strength = 0.0

        for tau in range(tau_max + 1):
            pval = p_matrix[i, target_idx, tau]
            strength = abs(val_matrix[i, target_idx, tau])

            if pval < min_pvalue:
                min_pvalue = pval
                best_lag = tau
                best_strength = strength

        records.append({
            "feature": feature,
            "min_pvalue": float(min_pvalue),
            "best_lag": int(best_lag),
            "causal_strength": float(best_strength),
        })

    return pl.DataFrame(records).sort("min_pvalue")


def get_pcmci_summary(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    target_col: str = "target",
    alpha: float = 0.05,
    tau_max: int = 5,
) -> dict:
    """Get summary of PCMCI causal discovery.

    Args:
        df: DataFrame with features
        feature_cols: Features to analyze
        target_col: Target column name
        alpha: Significance level
        tau_max: Maximum lag

    Returns:
        Dict with causal discovery metadata
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    initial_count = len(feature_cols)
    available = filter_suppressed(feature_cols)
    after_suppression = len(available)

    try:
        selected = filter_pcmci(
            df,
            feature_cols=available,
            target_col=target_col,
            alpha=alpha,
            tau_max=tau_max,
            apply_suppression=False,
        )
    except ValueError as e:
        return {
            "phase": "PCMCI",
            "error": str(e),
            "initial_features": initial_count,
            "after_suppression": after_suppression,
        }

    return {
        "phase": "PCMCI",
        "initial_features": initial_count,
        "after_suppression": after_suppression,
        "alpha": alpha,
        "tau_max": tau_max,
        "features_with_causal_link": len(selected),
        "selected_features": selected,
        "reduction_ratio": len(selected) / initial_count if initial_count > 0 else 0.0,
    }


def _log_pcmci_selection(
    initial_count: int,
    selected_count: int,
    alpha: float,
    tau_max: int,
    selected_features: list[str],
) -> None:
    """Log PCMCI selection to NDJSON telemetry."""
    try:
        from ith_python.ndjson_logger import log_ndjson_event

        log_ndjson_event(
            event_type="feature_selection",
            method="pcmci",
            initial_count=initial_count,
            selected_count=selected_count,
            alpha=alpha,
            tau_max=tau_max,
            reduction_pct=round((1 - selected_count / initial_count) * 100, 1) if initial_count > 0 else 0,
            top_5_features=selected_features[:5],
        )
    except ImportError:
        pass


if __name__ == "__main__":
    import sys

    print("PCMCI Causal Discovery Module")
    print("=" * 50)
    print()
    print("Phase 3 of the principled feature selection pipeline.")
    print("Identifies features with direct causal links to target.")
    print()
    print("Key properties:")
    print("  - Peter-Clark Momentary Conditional Independence (MCI) test")
    print("  - Explicitly handles autocorrelation in time series")
    print("  - Distinguishes direct causation from spurious correlation")
    print()
    print("Usage:")
    print("  from ith_python.statistical_examination.pcmci_filter import filter_pcmci")
    print("  selected = filter_pcmci(df, target_col='returns', alpha=0.05)")
    print()
    print("Pipeline integration:")
    print("  mRMR (160→50) → dCor (50→30) → PCMCI (30→15) → Stability (15→10)")
    print()
    print("GitHub Issue: https://github.com/terrylica/cc-skills/issues/21")

    sys.exit(0)
