"""Feature importance analysis for ITH features.

Computes feature importance using:
- Mutual information (sklearn)
- SHAP values with LightGBM (for interaction effects)
- Correlation with target

Uses Polars for data preparation, sklearn/shap via numpy bridge (no pandas).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ith_python.statistical_examination._utils import get_feature_columns

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


def compute_mutual_information(
    df: pl.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> pl.DataFrame:
    """Compute mutual information between features and target.

    Uses sklearn's mutual_info_regression via numpy bridge (no pandas).

    Args:
        df: DataFrame with features and target
        feature_cols: Feature column names
        target_col: Target column name
        n_neighbors: Number of neighbors for MI estimation
        random_state: Random seed for reproducibility

    Returns:
        Polars DataFrame with features sorted by MI score
    """
    from sklearn.feature_selection import mutual_info_regression

    # Validate columns exist
    missing = [c for c in [*feature_cols, target_col] if c not in df.columns]
    if missing:
        msg = f"Missing columns: {missing}"
        raise ValueError(msg)

    # Select and drop nulls - Polars native
    subset_cols = [*feature_cols, target_col]
    df_clean = df.select(subset_cols).drop_nulls()

    if len(df_clean) < 30:
        logger.warning("Insufficient samples for MI computation: %d", len(df_clean))
        return pl.DataFrame({
            "feature": feature_cols,
            "mutual_information": [np.nan] * len(feature_cols),
        })

    # Polars -> numpy (no pandas intermediate)
    X = df_clean.select(feature_cols).to_numpy()
    y = df_clean.select(target_col).to_numpy().ravel()

    # Handle any remaining NaN/inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute MI scores
    mi_scores = mutual_info_regression(
        X,
        y,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )

    # Return as Polars DataFrame
    return pl.DataFrame({
        "feature": list(feature_cols),
        "mutual_information": mi_scores.tolist(),
    }).sort("mutual_information", descending=True)


def compute_correlation_importance(
    df: pl.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
) -> pl.DataFrame:
    """Compute absolute correlation with target.

    100% Polars-native implementation.

    Args:
        df: DataFrame with features and target
        feature_cols: Feature column names
        target_col: Target column name

    Returns:
        Polars DataFrame with features sorted by |correlation|
    """
    results = []

    for col in feature_cols:
        corr = df.select(pl.corr(col, target_col)).item()
        results.append({
            "feature": col,
            "correlation": float(corr) if corr is not None else np.nan,
            "abs_correlation": abs(float(corr)) if corr is not None else np.nan,
        })

    return pl.DataFrame(results).sort("abs_correlation", descending=True)


def compute_shap_importance(
    df: pl.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    n_estimators: int = 100,
    max_samples: int = 10000,
    random_state: int = 42,
) -> dict:
    """Compute SHAP-based feature importance using LightGBM.

    Uses LightGBM TreeExplainer for fast SHAP computation.
    Polars -> numpy for model training (no pandas).

    Args:
        df: DataFrame with features and target
        feature_cols: Feature column names
        target_col: Target column name
        n_estimators: Number of LightGBM trees
        max_samples: Maximum samples for SHAP computation
        random_state: Random seed

    Returns:
        Dict with SHAP importance results
    """
    import lightgbm as lgb
    import shap

    # Validate columns
    missing = [c for c in [*feature_cols, target_col] if c not in df.columns]
    if missing:
        msg = f"Missing columns: {missing}"
        raise ValueError(msg)

    # Select and drop nulls
    subset_cols = [*feature_cols, target_col]
    df_clean = df.select(subset_cols).drop_nulls()

    if len(df_clean) < 100:
        return {
            "error": f"Insufficient samples for SHAP: {len(df_clean)}",
            "feature_importance": {},
        }

    # Polars -> numpy (direct, no pandas)
    X = df_clean.select(feature_cols).to_numpy()
    y = df_clean.select(target_col).to_numpy().ravel()

    # Subsample if too large
    if len(X) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    # Train LightGBM (accepts numpy directly)
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(X, y)

    # SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP importance
    importance = np.abs(shap_values).mean(axis=0)

    # Create importance dict
    importance_dict = dict(zip(feature_cols, importance.tolist()))

    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: -x[1])

    return {
        "n_samples": len(X),
        "n_features": len(feature_cols),
        "feature_importance": importance_dict,
        "top_features": sorted_features[:20],
        "shap_values_shape": shap_values.shape,
    }


def compute_combined_importance(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    target_col: str | None = None,
    include_shap: bool = True,
    shap_max_samples: int = 10000,
) -> dict:
    """Compute combined feature importance from multiple methods.

    Args:
        df: DataFrame with ITH features
        feature_cols: Features to analyze (auto-detect if None)
        target_col: Target column (required for MI/SHAP)
        include_shap: Whether to compute SHAP values (slower)
        shap_max_samples: Max samples for SHAP

    Returns:
        Dict with importance from all methods and combined ranking
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    results = {
        "n_features": len(feature_cols),
        "methods": [],
    }

    # Always compute correlation if target exists
    if target_col and target_col in df.columns:
        corr_df = compute_correlation_importance(df, feature_cols, target_col)
        results["correlation"] = corr_df.to_dicts()
        results["methods"].append("correlation")

        # MI
        mi_df = compute_mutual_information(df, feature_cols, target_col)
        results["mutual_information"] = mi_df.to_dicts()
        results["methods"].append("mutual_information")

        # SHAP (optional)
        if include_shap:
            shap_result = compute_shap_importance(
                df, feature_cols, target_col, max_samples=shap_max_samples
            )
            results["shap"] = shap_result
            if "error" not in shap_result:
                results["methods"].append("shap")

        # Combined ranking using rank aggregation
        rankings = _compute_rank_aggregation(results, feature_cols)
        results["combined_ranking"] = rankings
    else:
        results["error"] = "No target column provided - cannot compute importance"

    return results


def _compute_rank_aggregation(
    results: dict,
    feature_cols: Sequence[str],
) -> list[dict]:
    """Aggregate rankings from multiple methods using Borda count.

    Args:
        results: Dict with importance results from each method
        feature_cols: Feature column names

    Returns:
        List of dicts with combined rankings
    """
    n_features = len(feature_cols)
    scores = {f: 0.0 for f in feature_cols}

    # Correlation ranking
    if "correlation" in results:
        for rank, row in enumerate(results["correlation"]):
            feature = row["feature"]
            scores[feature] += (n_features - rank)  # Borda score

    # MI ranking
    if "mutual_information" in results:
        for rank, row in enumerate(results["mutual_information"]):
            feature = row["feature"]
            scores[feature] += (n_features - rank)

    # SHAP ranking
    if "shap" in results and "top_features" in results["shap"]:
        for rank, (feature, _) in enumerate(results["shap"]["top_features"]):
            if feature in scores:
                scores[feature] += (n_features - rank)

    # Sort by combined score
    sorted_features = sorted(scores.items(), key=lambda x: -x[1])

    return [
        {"feature": f, "combined_score": s, "rank": i + 1}
        for i, (f, s) in enumerate(sorted_features)
    ]


if __name__ == "__main__":
    import sys

    print("Feature importance analysis module")
    print("Run via: uv run python -m ith_python.statistical_examination.runner")
    sys.exit(0)
