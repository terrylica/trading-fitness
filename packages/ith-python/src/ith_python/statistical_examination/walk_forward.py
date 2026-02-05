"""Walk-Forward Importance Stability validation.

Implements Phase 4b of the principled feature selection pipeline:
- Time-series cross-validation respecting temporal order
- Gap between train/test to prevent lookahead
- Selects features with stable importance across folds

GitHub Issue: https://github.com/terrylica/cc-skills/issues/21

100% Polars-native for data handling, sklearn/tscv via numpy bridge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ith_python.statistical_examination._utils import get_feature_columns
from ith_python.statistical_examination.suppression import filter_suppressed

if TYPE_CHECKING:
    from collections.abc import Sequence


def compute_walk_forward_stability(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    target_col: str = "target",
    n_splits: int = 5,
    gap: int = 0,
    *,
    apply_suppression: bool = True,
    emit_telemetry: bool = False,
) -> pl.DataFrame:
    """Compute feature importance stability across walk-forward folds.

    Uses time-series CV to compute importance in each fold,
    then measures stability via coefficient of variation (CV).

    Args:
        df: DataFrame with features and target column
        feature_cols: Features to analyze (auto-detect if None)
        target_col: Name of target column
        n_splits: Number of CV splits (default 5)
        gap: Gap between train and test sets (default 0)
        apply_suppression: If True, filter suppressed features first
        emit_telemetry: If True, log events to NDJSON

    Returns:
        DataFrame with columns: feature, mean_importance, std_importance, cv, n_folds
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import TimeSeriesSplit

    if target_col not in df.columns:
        msg = f"Target column '{target_col}' not found in DataFrame"
        raise ValueError(msg)

    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    if not feature_cols:
        return pl.DataFrame(schema={
            "feature": pl.Utf8,
            "mean_importance": pl.Float64,
            "std_importance": pl.Float64,
            "cv": pl.Float64,
            "n_folds": pl.Int64,
        })

    if apply_suppression:
        feature_cols = filter_suppressed(feature_cols, emit_telemetry=emit_telemetry)

    if not feature_cols:
        return pl.DataFrame(schema={
            "feature": pl.Utf8,
            "mean_importance": pl.Float64,
            "std_importance": pl.Float64,
            "cv": pl.Float64,
            "n_folds": pl.Int64,
        })

    # Prepare data
    df_clean = df.select([*feature_cols, target_col]).drop_nulls()

    min_samples = (n_splits + 1) * 20  # Need enough for n_splits + 1 test sets
    if df_clean.height < min_samples:
        msg = f"Insufficient data for {n_splits} splits: {df_clean.height} rows (need >= {min_samples})"
        raise ValueError(msg)

    X = df_clean.select(feature_cols).to_numpy()
    y = df_clean.select(target_col).to_numpy().ravel()

    # Initialize time-series CV
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    # Collect importance scores across folds
    importance_samples = {f: [] for f in feature_cols}

    for train_idx, _test_idx in tscv.split(X):
        X_train = X[train_idx]
        y_train = y[train_idx]

        # Fit RF and get feature importances
        rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=1,
        )
        rf.fit(X_train, y_train)

        for i, f in enumerate(feature_cols):
            importance_samples[f].append(rf.feature_importances_[i])

    # Compute statistics
    records = []
    for f in feature_cols:
        scores = np.array(importance_samples[f])
        mean_imp = float(np.mean(scores))
        std_imp = float(np.std(scores))
        cv = std_imp / mean_imp if mean_imp > 0 else float("inf")

        records.append({
            "feature": f,
            "mean_importance": mean_imp,
            "std_importance": std_imp,
            "cv": cv,
            "n_folds": len(scores),
        })

    result = pl.DataFrame(records).sort("cv")  # Sort by stability (low CV first)

    if emit_telemetry:
        _log_walk_forward_stability(
            n_features=len(feature_cols),
            n_splits=n_splits,
            gap=gap,
            most_stable=result.head(5)["feature"].to_list(),
        )

    return result


def filter_stable_features(
    stability_df: pl.DataFrame,
    max_cv: float = 0.5,
    min_importance: float = 0.01,
) -> list[str]:
    """Filter features by stability and minimum importance.

    Args:
        stability_df: DataFrame from compute_walk_forward_stability
        max_cv: Maximum coefficient of variation (default 0.5)
        min_importance: Minimum mean importance (default 0.01)

    Returns:
        List of stable, important features
    """
    stable = stability_df.filter(
        (pl.col("cv") <= max_cv) & (pl.col("mean_importance") >= min_importance)
    )
    return stable["feature"].to_list()


def select_top_k_stable(
    stability_df: pl.DataFrame,
    k: int = 10,
    max_cv: float = 1.0,
) -> list[str]:
    """Select top k features by importance among stable ones.

    Args:
        stability_df: DataFrame from compute_walk_forward_stability
        k: Number of features to select
        max_cv: Maximum CV to be considered stable

    Returns:
        List of top k stable features by importance
    """
    stable = stability_df.filter(pl.col("cv") <= max_cv)
    top_k = stable.sort("mean_importance", descending=True).head(k)
    return top_k["feature"].to_list()


def get_walk_forward_summary(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    target_col: str = "target",
    n_splits: int = 5,
    max_cv: float = 0.5,
    k: int = 10,
) -> dict:
    """Get summary of walk-forward stability analysis.

    Args:
        df: DataFrame with features
        feature_cols: Features to analyze
        target_col: Target column name
        n_splits: Number of CV splits
        max_cv: Maximum CV for stability filter
        k: Number of final features to select

    Returns:
        Dict with walk-forward analysis metadata
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    initial_count = len(feature_cols)
    available = filter_suppressed(feature_cols)
    after_suppression = len(available)

    try:
        stability_df = compute_walk_forward_stability(
            df,
            feature_cols=available,
            target_col=target_col,
            n_splits=n_splits,
            apply_suppression=False,
        )

        stable_features = filter_stable_features(stability_df, max_cv=max_cv)
        final_features = select_top_k_stable(stability_df, k=k, max_cv=max_cv)

    except ValueError as e:
        return {
            "phase": "WalkForward",
            "error": str(e),
            "initial_features": initial_count,
            "after_suppression": after_suppression,
        }

    return {
        "phase": "WalkForward",
        "initial_features": initial_count,
        "after_suppression": after_suppression,
        "n_splits": n_splits,
        "max_cv": max_cv,
        "stable_features_count": len(stable_features),
        "final_k": k,
        "final_features": final_features,
        "mean_cv": float(stability_df["cv"].mean()) if stability_df.height > 0 else None,
    }


def _log_walk_forward_stability(
    n_features: int,
    n_splits: int,
    gap: int,
    most_stable: list[str],
) -> None:
    """Log walk-forward stability to NDJSON telemetry."""
    try:
        from ith_python.ndjson_logger import log_ndjson_event

        log_ndjson_event(
            event_type="feature_selection",
            method="walk_forward",
            n_features=n_features,
            n_splits=n_splits,
            gap=gap,
            most_stable_5=most_stable[:5],
        )
    except ImportError:
        pass


if __name__ == "__main__":
    import sys

    print("Walk-Forward Importance Stability Module")
    print("=" * 50)
    print()
    print("Phase 4b of the principled feature selection pipeline.")
    print("Validates feature importance stability across time-series folds.")
    print()
    print("Key properties:")
    print("  - Time-series CV respects temporal order (no lookahead)")
    print("  - Optional gap between train/test prevents leakage")
    print("  - CV measures importance consistency across regimes")
    print()
    print("Usage:")
    print("  from ith_python.statistical_examination.walk_forward import (")
    print("      compute_walk_forward_stability, select_top_k_stable")
    print("  )")
    print("  stability = compute_walk_forward_stability(df, target_col='returns')")
    print("  final = select_top_k_stable(stability, k=10)")
    print()
    print("Pipeline integration:")
    print("  mRMR → dCor → PCMCI → Bootstrap + WalkForward → Final (15→10)")
    print()
    print("GitHub Issue: https://github.com/terrylica/cc-skills/issues/21")

    sys.exit(0)
