"""Block Bootstrap for ACF-robust feature importance.

Implements Phase 4a of the principled feature selection pipeline:
- Circular Block Bootstrap preserving temporal structure
- Optimal block length via Politis-White method
- Stability assessment of feature importance under resampling

GitHub Issue: https://github.com/terrylica/cc-skills/issues/21

100% Polars-native for data handling, tsbootstrap/recombinator via numpy bridge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ith_python.statistical_examination._utils import get_feature_columns
from ith_python.statistical_examination.suppression import filter_suppressed

if TYPE_CHECKING:
    from collections.abc import Sequence


def compute_optimal_block_length(
    series: np.ndarray,
    method: str = "optimal",
) -> int:
    """Compute optimal block length for bootstrap.

    Uses Politis-White method for optimal block length selection.

    Args:
        series: 1D numpy array of time series data
        method: Method for block length ("optimal" or "fixed")

    Returns:
        Optimal block length (minimum 2)
    """
    from recombinator.optimal_block_length import optimal_block_length as obl

    if method == "fixed":
        return max(2, int(np.sqrt(len(series))))

    try:
        result = obl(series)
        # result is tuple containing OptimalBlockLength namedtuple
        # OptimalBlockLength has b_star_sb (stationary) and b_star_cb (circular)
        if isinstance(result, tuple) and len(result) > 0:
            obl_result = result[0]
            if hasattr(obl_result, "b_star_cb"):
                block_len = int(obl_result.b_star_cb)
            elif hasattr(obl_result, "b_star_sb"):
                block_len = int(obl_result.b_star_sb)
            else:
                block_len = int(np.sqrt(len(series)))
        else:
            block_len = int(np.sqrt(len(series)))
        return max(2, block_len)
    except (ValueError, RuntimeError, IndexError):
        # Fallback to sqrt(n) rule
        return max(2, int(np.sqrt(len(series))))


def _circular_block_bootstrap_indices(
    n_samples: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate indices for circular block bootstrap.

    Args:
        n_samples: Total number of samples
        block_length: Length of each block
        rng: Random number generator

    Returns:
        Array of bootstrap indices
    """
    n_blocks = int(np.ceil(n_samples / block_length))
    starts = rng.integers(0, n_samples, size=n_blocks)

    indices = []
    for start in starts:
        block_indices = (np.arange(start, start + block_length)) % n_samples
        indices.extend(block_indices.tolist())

    return np.array(indices[:n_samples])


def compute_bootstrap_importance(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    target_col: str = "target",
    n_bootstrap: int = 100,
    block_length: int | None = None,
    *,
    apply_suppression: bool = True,
    emit_telemetry: bool = False,
) -> pl.DataFrame:
    """Compute feature importance with block bootstrap confidence intervals.

    Uses circular block bootstrap to preserve temporal structure,
    then computes importance stability across bootstrap samples.

    Args:
        df: DataFrame with features and target column
        feature_cols: Features to analyze (auto-detect if None)
        target_col: Name of target column
        n_bootstrap: Number of bootstrap samples (default 100)
        block_length: Block length (auto-computed if None)
        apply_suppression: If True, filter suppressed features first
        emit_telemetry: If True, log events to NDJSON

    Returns:
        DataFrame with columns: feature, mean_importance, std_importance, cv
    """
    from sklearn.ensemble import RandomForestRegressor

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
        })

    if apply_suppression:
        feature_cols = filter_suppressed(feature_cols, emit_telemetry=emit_telemetry)

    if not feature_cols:
        return pl.DataFrame(schema={
            "feature": pl.Utf8,
            "mean_importance": pl.Float64,
            "std_importance": pl.Float64,
            "cv": pl.Float64,
        })

    # Prepare data
    df_clean = df.select([*feature_cols, target_col]).drop_nulls()

    if df_clean.height < 50:
        msg = f"Insufficient data for bootstrap: {df_clean.height} rows (need >= 50)"
        raise ValueError(msg)

    X = df_clean.select(feature_cols).to_numpy()
    y = df_clean.select(target_col).to_numpy().ravel()
    n_samples = len(y)

    # Compute optimal block length if not provided
    if block_length is None:
        block_length = compute_optimal_block_length(y)

    # Initialize RNG for reproducibility
    rng = np.random.default_rng(42)

    # Collect importance scores across bootstrap samples
    importance_samples = {f: [] for f in feature_cols}

    for _ in range(n_bootstrap):
        # Generate circular block bootstrap indices
        indices = _circular_block_bootstrap_indices(n_samples, block_length, rng)

        X_boot = X[indices]
        y_boot = y[indices]

        # Fit quick RF and get feature importances
        rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=1,
        )
        rf.fit(X_boot, y_boot)

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
        })

    result = pl.DataFrame(records).sort("mean_importance", descending=True)

    if emit_telemetry:
        _log_bootstrap_importance(
            n_features=len(feature_cols),
            n_bootstrap=n_bootstrap,
            block_length=block_length,
            top_features=result.head(5)["feature"].to_list(),
        )

    return result


def filter_by_stability(
    importance_df: pl.DataFrame,
    max_cv: float = 0.5,
) -> list[str]:
    """Filter features by importance stability (low CV).

    Args:
        importance_df: DataFrame from compute_bootstrap_importance
        max_cv: Maximum coefficient of variation (default 0.5)

    Returns:
        List of stable features (CV <= max_cv)
    """
    stable = importance_df.filter(pl.col("cv") <= max_cv)
    return stable["feature"].to_list()


def get_bootstrap_summary(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    target_col: str = "target",
    n_bootstrap: int = 100,
    max_cv: float = 0.5,
) -> dict:
    """Get summary of bootstrap importance analysis.

    Args:
        df: DataFrame with features
        feature_cols: Features to analyze
        target_col: Target column name
        n_bootstrap: Number of bootstrap samples
        max_cv: Maximum CV for stability filter

    Returns:
        Dict with bootstrap analysis metadata
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    initial_count = len(feature_cols)
    available = filter_suppressed(feature_cols)
    after_suppression = len(available)

    try:
        importance_df = compute_bootstrap_importance(
            df,
            feature_cols=available,
            target_col=target_col,
            n_bootstrap=n_bootstrap,
            apply_suppression=False,
        )

        stable_features = filter_by_stability(importance_df, max_cv=max_cv)

    except ValueError as e:
        return {
            "phase": "BlockBootstrap",
            "error": str(e),
            "initial_features": initial_count,
            "after_suppression": after_suppression,
        }

    return {
        "phase": "BlockBootstrap",
        "initial_features": initial_count,
        "after_suppression": after_suppression,
        "n_bootstrap": n_bootstrap,
        "max_cv": max_cv,
        "stable_features_count": len(stable_features),
        "stable_features": stable_features,
        "mean_cv": float(importance_df["cv"].mean()) if importance_df.height > 0 else None,
    }


def _log_bootstrap_importance(
    n_features: int,
    n_bootstrap: int,
    block_length: int,
    top_features: list[str],
) -> None:
    """Log bootstrap importance to NDJSON telemetry."""
    try:
        from ith_python.ndjson_logger import log_ndjson_event

        log_ndjson_event(
            event_type="feature_selection",
            method="block_bootstrap",
            n_features=n_features,
            n_bootstrap=n_bootstrap,
            block_length=block_length,
            top_5_features=top_features[:5],
        )
    except ImportError:
        pass


if __name__ == "__main__":
    import sys

    print("Block Bootstrap Feature Importance Module")
    print("=" * 50)
    print()
    print("Phase 4a of the principled feature selection pipeline.")
    print("Computes ACF-robust feature importance with stability metrics.")
    print()
    print("Key properties:")
    print("  - Circular Block Bootstrap preserves temporal structure")
    print("  - Optimal block length via Politis-White method")
    print("  - CV (coefficient of variation) measures stability")
    print()
    print("Usage:")
    print("  from ith_python.statistical_examination.block_bootstrap import (")
    print("      compute_bootstrap_importance, filter_by_stability")
    print("  )")
    print("  importance = compute_bootstrap_importance(df, target_col='returns')")
    print("  stable = filter_by_stability(importance, max_cv=0.5)")
    print()
    print("Pipeline integration:")
    print("  mRMR → dCor → PCMCI → BlockBootstrap + WalkForward → Final")
    print()
    print("GitHub Issue: https://github.com/terrylica/cc-skills/issues/21")

    sys.exit(0)
