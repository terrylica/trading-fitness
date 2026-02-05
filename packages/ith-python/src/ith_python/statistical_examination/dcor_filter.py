"""Distance Correlation (dCor) redundancy detection.

Implements Phase 2 of the principled feature selection pipeline:
- Parameter-free nonlinear dependence measure
- dCor=0 iff statistically independent (unlike Pearson r)
- Detects ALL nonlinear relationships

GitHub Issue: https://github.com/terrylica/cc-skills/issues/21

100% Polars-native for data handling, dcor via numpy bridge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from ith_python.statistical_examination._utils import get_feature_columns
from ith_python.statistical_examination.suppression import filter_suppressed

if TYPE_CHECKING:
    from collections.abc import Sequence


def compute_dcor_matrix(
    df: pl.DataFrame,
    feature_cols: Sequence[str],
) -> pl.DataFrame:
    """Compute pairwise distance correlation matrix.

    Args:
        df: DataFrame with features
        feature_cols: Features to compute dCor for

    Returns:
        DataFrame with columns: feature_1, feature_2, dcor
    """
    import dcor

    df_clean = df.select(feature_cols).drop_nulls()

    if df_clean.height < 10:
        return pl.DataFrame(schema={
            "feature_1": pl.Utf8,
            "feature_2": pl.Utf8,
            "dcor": pl.Float64,
        })

    # Convert to numpy for dcor computation
    data = {col: df_clean.get_column(col).to_numpy() for col in feature_cols}

    results = []
    n_features = len(feature_cols)

    for i in range(n_features):
        for j in range(i + 1, n_features):
            f1, f2 = feature_cols[i], feature_cols[j]
            d = dcor.distance_correlation(data[f1], data[f2])
            results.append({
                "feature_1": f1,
                "feature_2": f2,
                "dcor": float(d),
            })

    return pl.DataFrame(results)


def filter_dcor_redundancy(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    target_col: str | None = None,
    threshold: float = 0.7,
    *,
    apply_suppression: bool = True,
    emit_telemetry: bool = False,
) -> list[str]:
    """Remove redundant features based on distance correlation (Phase 2: 50→30).

    When two features have dCor > threshold, removes one based on:
    - If target provided: keep feature with higher MI to target
    - Otherwise: keep first encountered (preserves mRMR order)

    Args:
        df: DataFrame with features
        feature_cols: Features to filter (auto-detect if None)
        target_col: Optional target for tie-breaking by relevance
        threshold: Maximum dCor before removing (default 0.7)
        apply_suppression: If True, filter suppressed features first
        emit_telemetry: If True, log events to NDJSON

    Returns:
        List of non-redundant features
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    if not feature_cols:
        return []

    if apply_suppression:
        feature_cols = filter_suppressed(feature_cols, emit_telemetry=emit_telemetry)

    if len(feature_cols) < 2:
        return list(feature_cols)

    # Compute MI scores if target provided for tie-breaking
    mi_scores: dict[str, float] = {}
    if target_col and target_col in df.columns:
        mi_scores = _compute_mi_scores(df, feature_cols, target_col)

    # Compute dCor matrix
    dcor_df = compute_dcor_matrix(df, feature_cols)

    # Build set of features to remove
    removed: set[str] = set()
    available = list(feature_cols)

    # Filter pairs above threshold
    redundant_pairs = dcor_df.filter(pl.col("dcor") > threshold)

    for row in redundant_pairs.iter_rows(named=True):
        f1, f2, d = row["feature_1"], row["feature_2"], row["dcor"]

        if f1 in removed or f2 in removed:
            continue  # Already handled

        # Decide which to remove
        if mi_scores:
            # Keep higher MI feature
            to_remove = f2 if mi_scores.get(f1, 0) >= mi_scores.get(f2, 0) else f1
        else:
            # Keep first (preserves mRMR order)
            # f1 comes before f2 in iteration, so remove f2
            to_remove = f2

        removed.add(to_remove)

        if emit_telemetry:
            _log_redundancy_removal(f1, f2, d, to_remove)

    selected = [f for f in available if f not in removed]

    if emit_telemetry:
        _log_dcor_selection(
            initial_count=len(feature_cols),
            removed_count=len(removed),
            selected_count=len(selected),
        )

    return selected


def get_redundancy_pairs(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    threshold: float = 0.7,
) -> pl.DataFrame:
    """Get all pairs of features with dCor above threshold.

    Args:
        df: DataFrame with features
        feature_cols: Features to analyze
        threshold: dCor threshold for redundancy

    Returns:
        DataFrame with columns: feature_1, feature_2, dcor (sorted by dcor desc)
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    if len(feature_cols) < 2:
        return pl.DataFrame(schema={
            "feature_1": pl.Utf8,
            "feature_2": pl.Utf8,
            "dcor": pl.Float64,
        })

    dcor_df = compute_dcor_matrix(df, feature_cols)

    return (
        dcor_df.filter(pl.col("dcor") > threshold)
        .sort("dcor", descending=True)
    )


def get_dcor_summary(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    target_col: str | None = None,
    threshold: float = 0.7,
) -> dict:
    """Get summary of dCor redundancy filtering.

    Args:
        df: DataFrame with features
        feature_cols: Features to analyze
        target_col: Optional target column
        threshold: dCor threshold

    Returns:
        Dict with filtering metadata
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    initial_count = len(feature_cols)
    available = filter_suppressed(feature_cols)
    after_suppression = len(available)

    selected = filter_dcor_redundancy(
        df,
        feature_cols=available,
        target_col=target_col,
        threshold=threshold,
        apply_suppression=False,
    )

    # Compute redundancy stats
    dcor_df = compute_dcor_matrix(df, available) if len(available) >= 2 else None
    redundant_pairs = dcor_df.filter(pl.col("dcor") > threshold).height if dcor_df is not None else 0

    return {
        "phase": "dCor",
        "initial_features": initial_count,
        "after_suppression": after_suppression,
        "redundant_pairs_found": redundant_pairs,
        "threshold": threshold,
        "features_removed": after_suppression - len(selected),
        "features_selected": len(selected),
        "selected_features": selected,
        "reduction_ratio": len(selected) / initial_count if initial_count > 0 else 0.0,
    }


def _compute_mi_scores(
    df: pl.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
) -> dict[str, float]:
    """Compute MI scores for tie-breaking."""
    try:
        from ith_python.statistical_examination.feature_importance import (
            compute_mutual_information,
        )

        mi_df = compute_mutual_information(df, feature_cols, target_col)
        return dict(zip(mi_df["feature"].to_list(), mi_df["mutual_information"].to_list(), strict=True))
    except (ImportError, ValueError, KeyError):
        # ImportError: feature_importance not available
        # ValueError/KeyError: MI computation failed (e.g., insufficient data)
        return {}


def _log_redundancy_removal(
    feature_1: str,
    feature_2: str,
    dcor: float,
    removed: str,
) -> None:
    """Log redundancy removal to NDJSON."""
    try:
        from ith_python.ndjson_logger import log_ndjson_event

        log_ndjson_event(
            event_type="dcor_redundancy",
            feature_1=feature_1,
            feature_2=feature_2,
            dcor=round(dcor, 4),
            removed=removed,
            kept=feature_1 if removed == feature_2 else feature_2,
        )
    except ImportError:
        pass


def _log_dcor_selection(
    initial_count: int,
    removed_count: int,
    selected_count: int,
) -> None:
    """Log dCor selection summary to NDJSON."""
    try:
        from ith_python.ndjson_logger import log_ndjson_event

        log_ndjson_event(
            event_type="feature_selection",
            method="dcor",
            initial_count=initial_count,
            removed_count=removed_count,
            selected_count=selected_count,
            reduction_pct=round((removed_count / initial_count) * 100, 1) if initial_count > 0 else 0,
        )
    except ImportError:
        pass


if __name__ == "__main__":
    import sys

    print("Distance Correlation (dCor) Redundancy Filter")
    print("=" * 50)
    print()
    print("Phase 2 of the principled feature selection pipeline.")
    print("Removes features with high nonlinear dependence (dCor > threshold).")
    print()
    print("Key properties:")
    print("  - Parameter-free (no kernel bandwidth, no MI estimator choice)")
    print("  - dCor=0 iff statistically independent")
    print("  - Detects ALL nonlinear relationships (unlike Pearson r)")
    print()
    print("Usage:")
    print("  from ith_python.statistical_examination.dcor_filter import filter_dcor_redundancy")
    print("  selected = filter_dcor_redundancy(df, threshold=0.7)")
    print()
    print("Pipeline integration:")
    print("  mRMR (160→50) → dCor (50→30) → PCMCI (30→15) → Stability (15→10)")
    print()
    print("GitHub Issue: https://github.com/terrylica/cc-skills/issues/21")

    sys.exit(0)
