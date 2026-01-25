"""Dimensionality analysis for ITH features.

Analyzes effective dimensionality of feature space using:
- PCA (Principal Component Analysis)
- Participation Ratio (effective dimensionality from statistical physics)
- VIF (Variance Inflation Factor) with Ridge regularization
- Explained variance decomposition

Uses Polars for data preparation, sklearn via numpy bridge (no pandas).

NOTE: VIF regularization added (2026-01-23):
- Standard VIF unstable for high-dimensional data (576 features)
- Ridge VIF targets condition number ~100 using Garcia et al. formula
- Pre-filter |r|>0.95 pairs before VIF calculation

Reference: docs/research/2026-01-23-statistical-methods-verification-gemini.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ith_python.statistical_examination._utils import get_feature_columns

if TYPE_CHECKING:
    from collections.abc import Sequence


# Default target condition number for Ridge VIF regularization
DEFAULT_TARGET_CONDITION = 100.0

# Correlation threshold for pre-filtering before VIF
CORRELATION_PREFILTER_THRESHOLD = 0.95


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """Compute Participation Ratio (effective dimensionality from statistical physics).

    The Participation Ratio (D_PR) measures how many dimensions effectively
    contribute to the variance. It ranges from 1 (all variance in one dimension)
    to K (variance evenly distributed across K dimensions).

    Formula: D_PR = (sum(lambda_i))^2 / sum(lambda_i^2) = Tr(C)^2 / Tr(C^2)

    Args:
        eigenvalues: Array of eigenvalues (explained variances)

    Returns:
        Participation ratio (effective dimensionality)
    """
    if len(eigenvalues) == 0 or np.sum(eigenvalues) == 0:
        return 0.0

    sum_eig = np.sum(eigenvalues)
    sum_eig_sq = np.sum(eigenvalues**2)

    if sum_eig_sq == 0:
        return 0.0

    return float(sum_eig**2 / sum_eig_sq)


def compute_vif_regularized(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    target_cond: float = DEFAULT_TARGET_CONDITION,
    max_vif_threshold: float = 10.0,
) -> tuple[pl.DataFrame, float]:
    """Ridge VIF targeting condition number ~100.

    Uses Garcia et al. (2015) matrix formulation for regularized VIF.
    Formula: VIF_ridge = diag(R_delta^-1)
    where R_delta = (R + lambda*I) / (1 + lambda)

    This is computationally efficient: O(p^3) matrix inversion vs O(p^4)
    for p separate regressions.

    Args:
        df: DataFrame with ITH features
        feature_cols: Features to analyze (auto-detect if None)
        target_cond: Target condition number (default 100)
        max_vif_threshold: Threshold above which VIF is considered high

    Returns:
        Tuple of (Polars DataFrame with VIF values, lambda used)
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    if len(feature_cols) < 2:
        return pl.DataFrame({
            "feature": list(feature_cols),
            "vif": [1.0] * len(feature_cols),
            "high_multicollinearity": [False] * len(feature_cols),
        }), 0.0

    # Polars -> numpy
    df_clean = df.select(feature_cols).drop_nulls()
    X = df_clean.to_numpy()

    # Handle non-finite values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute correlation matrix
    R = np.corrcoef(X, rowvar=False)

    # Handle NaN in correlation matrix (constant columns)
    R = np.nan_to_num(R, nan=0.0)

    # Compute lambda to achieve target condition number
    # Formula: lambda = max(0, (lambda_max - c * lambda_min) / (c - 1))
    eig = np.linalg.eigvalsh(R)
    lambda_max = eig[-1]
    lambda_min = eig[0]

    if target_cond > 1 and lambda_min > 0:
        lam = max(0.0, (lambda_max - target_cond * lambda_min) / (target_cond - 1))
    else:
        # Fallback: use small regularization
        lam = 0.01

    # Regularized correlation matrix: R_delta = (R + lambda*I) / (1 + lambda)
    n_features = R.shape[0]
    R_reg = (R + lam * np.eye(n_features)) / (1 + lam)

    # VIF from diagonal of inverse
    try:
        R_reg_inv = np.linalg.inv(R_reg)
        vif_values = np.diag(R_reg_inv).tolist()
    except np.linalg.LinAlgError:
        # Matrix still singular despite regularization
        vif_values = [np.inf] * n_features

    return pl.DataFrame({
        "feature": list(feature_cols),
        "vif": vif_values,
        "high_multicollinearity": [v > max_vif_threshold for v in vif_values],
    }).sort("vif", descending=True), float(lam)


def compute_vif(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    max_vif_threshold: float = 10.0,
    use_regularization: bool = True,
    target_cond: float = DEFAULT_TARGET_CONDITION,
) -> pl.DataFrame:
    """Variance Inflation Factor with optional Ridge regularization.

    VIF > 10 indicates high multicollinearity.
    VIF > 5 indicates moderate multicollinearity.

    Args:
        df: DataFrame with ITH features
        feature_cols: Features to analyze (auto-detect if None)
        max_vif_threshold: Threshold above which VIF is considered high
        use_regularization: If True, use Ridge VIF (recommended for >50 features)
        target_cond: Target condition number for Ridge regularization

    Returns:
        Polars DataFrame with VIF values per feature
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    if use_regularization or len(feature_cols) > 50:
        vif_df, _ = compute_vif_regularized(df, feature_cols, target_cond, max_vif_threshold)
        return vif_df

    # Standard VIF (only for small feature sets)
    if len(feature_cols) < 2:
        return pl.DataFrame({
            "feature": list(feature_cols),
            "vif": [1.0] * len(feature_cols),
            "high_multicollinearity": [False] * len(feature_cols),
        })

    # Polars -> numpy
    df_clean = df.select(feature_cols).drop_nulls()
    X = df_clean.to_numpy()

    # Handle non-finite values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

    vif_values = []
    for i in range(X.shape[1]):
        # Regress feature i on all other features
        y = X[:, i]
        X_other = np.delete(X, i, axis=1)
        X_other = np.column_stack([np.ones(len(y)), X_other])

        # OLS: solve normal equations
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_other, y, rcond=None)
            y_pred = X_other @ coeffs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
        except np.linalg.LinAlgError:
            vif = np.inf

        vif_values.append(vif)

    return pl.DataFrame({
        "feature": list(feature_cols),
        "vif": vif_values,
        "high_multicollinearity": [v > max_vif_threshold for v in vif_values],
    }).sort("vif", descending=True)


def prefilter_highly_correlated(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    correlation_threshold: float = CORRELATION_PREFILTER_THRESHOLD,
) -> tuple[list[str], list[dict]]:
    """Pre-filter features with |r| > threshold before VIF calculation.

    Features with near-perfect correlation will have extreme VIF regardless
    and can be removed deterministically. This reduces computation and
    improves numerical stability.

    Args:
        df: DataFrame with ITH features
        feature_cols: Features to analyze (auto-detect if None)
        correlation_threshold: |correlation| above this triggers removal

    Returns:
        Tuple of (filtered feature list, list of removed pairs)
    """
    if feature_cols is None:
        feature_cols = list(get_feature_columns(df))

    removed_pairs = []
    removed_features = set()

    df_clean = df.select(feature_cols).drop_nulls()

    for i, col1 in enumerate(feature_cols):
        if col1 in removed_features:
            continue
        for col2 in feature_cols[i + 1:]:
            if col2 in removed_features:
                continue

            corr = df_clean.select(pl.corr(col1, col2)).item()
            if corr is not None and abs(corr) > correlation_threshold:
                # Remove the second feature (arbitrary choice, could use variance)
                removed_features.add(col2)
                removed_pairs.append({
                    "feature1": col1,
                    "feature2": col2,
                    "correlation": float(corr),
                    "removed": col2,
                })

    filtered = [f for f in feature_cols if f not in removed_features]
    return filtered, removed_pairs


def perform_pca(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    n_components_report: int | None = None,
) -> dict:
    """PCA analysis using Polars -> numpy -> sklearn.

    Includes Participation Ratio for effective dimensionality measurement.

    Args:
        df: DataFrame with ITH features
        feature_cols: Features to analyze (auto-detect if None)
        n_components_report: Number of components to report loadings for
                            (defaults to components for 95% variance)

    Returns:
        Dict with PCA results including effective dimensionality
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    if len(feature_cols) < 2:
        return {"error": f"Need at least 2 features for PCA, got {len(feature_cols)}"}

    # Polars -> numpy (no pandas)
    df_clean = df.select(feature_cols).drop_nulls()

    if len(df_clean) < len(feature_cols):
        return {"error": f"Insufficient samples ({len(df_clean)}) for {len(feature_cols)} features"}

    X = df_clean.to_numpy()

    # Handle infinite values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize (sklearn accepts numpy)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Full PCA
    pca = PCA()
    pca.fit(X_scaled)

    # Cumulative explained variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_90 = int(np.searchsorted(cumvar, 0.90) + 1)
    n_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    n_99 = int(np.searchsorted(cumvar, 0.99) + 1)

    # Effective dimensionality metrics
    eigenvalues = pca.explained_variance_
    d_pr = participation_ratio(eigenvalues)

    # Report loadings for top components
    if n_components_report is None:
        n_components_report = min(n_95, 20)  # Cap at 20 for readability

    # Loadings as Polars DataFrame
    loadings_data = {"feature": list(feature_cols)}
    for i in range(n_components_report):
        loadings_data[f"PC{i + 1}"] = pca.components_[i].tolist()

    loadings_df = pl.DataFrame(loadings_data)

    # Top contributors per component
    top_contributors = []
    for i in range(min(n_components_report, 5)):  # Top 5 components
        component = pca.components_[i]
        abs_loadings = np.abs(component)
        top_idx = np.argsort(abs_loadings)[-5:][::-1]  # Top 5 features

        contributors = [
            {"feature": feature_cols[idx], "loading": float(component[idx])}
            for idx in top_idx
        ]
        top_contributors.append({
            "component": f"PC{i + 1}",
            "variance_explained": float(pca.explained_variance_ratio_[i]),
            "top_features": contributors,
        })

    return {
        "n_samples": len(X),
        "n_features": len(feature_cols),
        "n_components_90_variance": n_90,
        "n_components_95_variance": n_95,
        "n_components_99_variance": n_99,
        "participation_ratio": d_pr,
        "effective_dimensionality": d_pr,  # Alias for backwards compatibility
        "dimensionality_ratio_95": n_95 / len(feature_cols),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": cumvar.tolist(),
        "loadings": loadings_df.to_dicts(),
        "top_contributors_per_component": top_contributors,
    }


def identify_redundant_features(
    df: pl.DataFrame,
    feature_cols: Sequence[str] | None = None,
    vif_threshold: float = 10.0,
    correlation_threshold: float = 0.95,
    use_ridge_vif: bool = True,
) -> dict:
    """Identify redundant features using VIF and pairwise correlation.

    Recommended workflow:
    1. Pre-filter |r| > 0.95 pairs
    2. Run Ridge VIF on remaining features
    3. Remove VIF > 10

    Args:
        df: DataFrame with ITH features
        feature_cols: Features to analyze (auto-detect if None)
        vif_threshold: VIF above this indicates redundancy
        correlation_threshold: |correlation| above this indicates redundancy
        use_ridge_vif: If True, use regularized VIF (recommended)

    Returns:
        Dict with redundant features and recommendations
    """
    if feature_cols is None:
        feature_cols = list(get_feature_columns(df))

    # Step 1: Pre-filter highly correlated pairs
    filtered_features, corr_removed = prefilter_highly_correlated(
        df, feature_cols, correlation_threshold
    )

    # Step 2: Run VIF on filtered features
    if use_ridge_vif:
        vif_df, lambda_used = compute_vif_regularized(df, filtered_features, target_cond=100.0)
    else:
        vif_df = compute_vif(df, filtered_features, use_regularization=False)
        lambda_used = 0.0

    high_vif = vif_df.filter(pl.col("high_multicollinearity")).get_column("feature").to_list()

    # Combine redundant features
    redundant_from_corr = {p["removed"] for p in corr_removed}
    redundant_from_vif = set(high_vif)
    all_redundant = redundant_from_corr | redundant_from_vif

    non_redundant = [f for f in feature_cols if f not in all_redundant]

    return {
        "total_features": len(feature_cols),
        "correlation_prefilter_threshold": correlation_threshold,
        "features_removed_by_correlation": list(redundant_from_corr),
        "highly_correlated_pairs": corr_removed,
        "features_after_correlation_filter": len(filtered_features),
        "ridge_lambda_used": lambda_used,
        "high_vif_features": high_vif,
        "redundant_features": sorted(all_redundant),
        "recommended_features": non_redundant,
        "reduction_ratio": len(non_redundant) / len(feature_cols) if feature_cols else 1.0,
    }


def summarize_dimensionality(
    pca_result: dict,
    vif_df: pl.DataFrame,
) -> dict:
    """Create summary of dimensionality analysis.

    Args:
        pca_result: Result from perform_pca()
        vif_df: Result from compute_vif()

    Returns:
        Summary dict
    """
    if "error" in pca_result:
        return {"error": pca_result["error"]}

    high_vif_count = vif_df.filter(pl.col("high_multicollinearity")).height

    return {
        "n_features": pca_result["n_features"],
        "effective_dimensions": pca_result["n_components_95_variance"],
        "dimensionality_ratio": pca_result["dimensionality_ratio_95"],
        "participation_ratio": pca_result["participation_ratio"],
        "features_with_high_vif": high_vif_count,
        "interpretation": _interpret_dimensionality(
            pca_result["n_features"],
            pca_result["n_components_95_variance"],
            high_vif_count,
            pca_result["participation_ratio"],
        ),
    }


def _interpret_dimensionality(
    n_features: int,
    n_components_95: int,
    high_vif_count: int,
    d_pr: float,
) -> str:
    """Generate human-readable interpretation of dimensionality.

    Args:
        n_features: Total number of features
        n_components_95: Components needed for 95% variance
        high_vif_count: Number of features with high VIF
        d_pr: Participation Ratio (effective dimensionality)

    Returns:
        Interpretation string
    """
    ratio = n_components_95 / n_features

    if ratio < 0.2:
        dim_interp = "highly redundant (strong compression possible)"
    elif ratio < 0.4:
        dim_interp = "moderately redundant (good compression possible)"
    elif ratio < 0.6:
        dim_interp = "some redundancy (limited compression)"
    else:
        dim_interp = "low redundancy (features mostly independent)"

    vif_interp = ""
    if high_vif_count > n_features * 0.3:
        vif_interp = f" Caution: {high_vif_count} features show high multicollinearity."
    elif high_vif_count > 0:
        vif_interp = f" Note: {high_vif_count} features have elevated VIF."

    pr_interp = f" Participation ratio: {d_pr:.1f} (effective dimensions)."

    return f"Feature space is {dim_interp}. {n_components_95}/{n_features} components explain 95% variance.{pr_interp}{vif_interp}"


if __name__ == "__main__":
    import sys

    print("Dimensionality analysis module")
    print("Run via: uv run python -m ith_python.statistical_examination.runner")
    sys.exit(0)
