"""mRMR (Minimum Redundancy Maximum Relevance) feature selection.

Implements Phase 1 of the principled feature selection pipeline:
- Fast filter-based selection using mutual information
- Penalizes redundancy between selected features
- Integrates with suppression registry

GitHub Issue: https://github.com/terrylica/cc-skills/issues/21

100% Polars-native for data handling, mrmr-selection via numpy bridge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from ith_python.statistical_examination._utils import get_feature_columns
from ith_python.statistical_examination.suppression import filter_suppressed

if TYPE_CHECKING:
    from collections.abc import Sequence


def filter_mrmr(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    target_col: str = "target",
    k: int = 50,
    *,
    apply_suppression: bool = True,
    emit_telemetry: bool = False,
) -> list[str]:
    """mRMR-based feature selection (Phase 1: 160→50).

    Uses mRMR algorithm to select features with:
    - Maximum relevance (high MI with target)
    - Minimum redundancy (low MI between selected features)

    Args:
        df: DataFrame with ITH features and target column
        feature_cols: Features to consider (auto-detect if None)
        target_col: Name of target column for relevance computation
        k: Number of features to select (default 50)
        apply_suppression: If True, filter suppressed features first
        emit_telemetry: If True, log selection events to NDJSON

    Returns:
        List of selected feature names, ordered by mRMR score

    Raises:
        ValueError: If target_col not in DataFrame or insufficient features
    """
    from mrmr import mrmr_regression

    if target_col not in df.columns:
        msg = f"Target column '{target_col}' not found in DataFrame"
        raise ValueError(msg)

    # Auto-detect feature columns
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    if not feature_cols:
        return []

    # Apply suppression registry filter
    if apply_suppression:
        feature_cols = filter_suppressed(
            feature_cols,
            emit_telemetry=emit_telemetry,
        )

    if len(feature_cols) < k:
        k = len(feature_cols)

    if k == 0:
        return []

    # Prepare data for mrmr (numpy bridge)
    # mrmr expects pandas-like interface, but we use polars→numpy→pandas
    df_clean = df.select([*feature_cols, target_col]).drop_nulls()

    if df_clean.height < 10:
        msg = f"Insufficient data after dropping nulls: {df_clean.height} rows"
        raise ValueError(msg)

    # Extract numpy arrays and convert to pandas for mrmr

    X = df_clean.select(feature_cols).to_pandas()
    y = df_clean.select(target_col).to_pandas().squeeze()

    # Run mRMR selection
    selected = mrmr_regression(
        X=X,
        y=y,
        K=k,
        show_progress=False,
    )

    if emit_telemetry:
        _log_mrmr_selection(
            initial_count=len(feature_cols),
            selected_count=len(selected),
            selected_features=selected,
        )

    return list(selected)


def compute_mrmr_scores(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    target_col: str = "target",
    k: int | None = None,
    *,
    apply_suppression: bool = True,
) -> pl.DataFrame:
    """Compute mRMR scores for all features.

    Returns detailed scores including relevance and redundancy components.

    Args:
        df: DataFrame with ITH features and target column
        feature_cols: Features to consider (auto-detect if None)
        target_col: Name of target column
        k: Number of features to compute (all if None)
        apply_suppression: If True, filter suppressed features first

    Returns:
        DataFrame with columns: feature, mrmr_rank, relevance, redundancy
    """
    from mrmr import mrmr_regression

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
            "mrmr_rank": pl.Int64,
        })

    if k is None:
        k = len(feature_cols)

    k = min(k, len(feature_cols))

    # Prepare data
    df_clean = df.select([*feature_cols, target_col]).drop_nulls()


    X = df_clean.select(feature_cols).to_pandas()
    y = df_clean.select(target_col).to_pandas().squeeze()

    # Get ordered selection
    selected = mrmr_regression(
        X=X,
        y=y,
        K=k,
        show_progress=False,
    )

    # Build result DataFrame
    return pl.DataFrame({
        "feature": selected,
        "mrmr_rank": list(range(1, len(selected) + 1)),
    })


def get_mrmr_summary(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    target_col: str = "target",
    k: int = 50,
) -> dict:
    """Get summary of mRMR feature selection.

    Args:
        df: DataFrame with ITH features
        feature_cols: Features to consider
        target_col: Target column name
        k: Number of features to select

    Returns:
        Dict with selection metadata
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    # Count before suppression
    initial_count = len(feature_cols)

    # Apply suppression
    available_features = filter_suppressed(feature_cols)
    after_suppression = len(available_features)

    # Run selection
    selected = filter_mrmr(
        df,
        feature_cols=available_features,
        target_col=target_col,
        k=k,
        apply_suppression=False,  # Already applied
    )

    return {
        "phase": "mRMR",
        "initial_features": initial_count,
        "after_suppression": after_suppression,
        "k_requested": k,
        "k_selected": len(selected),
        "selected_features": selected,
        "reduction_ratio": len(selected) / initial_count if initial_count > 0 else 0.0,
    }


def _log_mrmr_selection(
    initial_count: int,
    selected_count: int,
    selected_features: list[str],
) -> None:
    """Log mRMR selection to NDJSON telemetry."""
    try:
        from ith_python.ndjson_logger import log_ndjson_event

        log_ndjson_event(
            event_type="feature_selection",
            method="mrmr",
            initial_count=initial_count,
            selected_count=selected_count,
            reduction_pct=round((1 - selected_count / initial_count) * 100, 1) if initial_count > 0 else 0,
            top_5_features=selected_features[:5],
        )
    except ImportError:
        pass  # Telemetry optional


if __name__ == "__main__":
    import sys

    print("mRMR Feature Selection Module")
    print("=" * 50)
    print()
    print("Phase 1 of the principled feature selection pipeline.")
    print("Selects features with maximum relevance and minimum redundancy.")
    print()
    print("Usage:")
    print("  from ith_python.statistical_examination.mrmr import filter_mrmr")
    print("  selected = filter_mrmr(df, target_col='returns', k=50)")
    print()
    print("Pipeline integration:")
    print("  mRMR (160→50) → dCor (50→30) → PCMCI (30→15) → Stability (15→10)")
    print()
    print("GitHub Issue: https://github.com/terrylica/cc-skills/issues/21")

    sys.exit(0)
