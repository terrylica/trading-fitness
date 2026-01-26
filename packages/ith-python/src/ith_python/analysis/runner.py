"""Analysis runner - unified interface for feature analysis.

This module provides the main entry point for running statistical analysis
on features from the FeatureStore.

Architecture: Multi-View Feature Architecture with Separation of Concerns
- Layer 3: Feature Analysis
- See: docs/plans/2026-01-25-multi-view-feature-architecture-plan.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from ith_python.storage import FeatureStore


@dataclass
class AnalysisConfig:
    """Configuration for feature analysis.

    Attributes:
        run_distribution: Run distribution analysis (Shapiro-Wilk, Beta fit)
        run_dimensionality: Run dimensionality analysis (PCA, VIF)
        run_regime: Run regime dependence analysis
        run_temporal: Run temporal analysis (ADF, ACF)
        run_cross_scale: Run cross-scale correlation analysis
        max_features: Maximum features to analyze (for speed)
        emit_telemetry: Whether to emit NDJSON telemetry
    """

    run_distribution: bool = True
    run_dimensionality: bool = True
    run_regime: bool = True
    run_temporal: bool = True
    run_cross_scale: bool = True
    max_features: int = 100
    emit_telemetry: bool = True


@dataclass
class AnalysisResults:
    """Results from feature analysis.

    Attributes:
        distribution: Distribution analysis results
        dimensionality: Dimensionality analysis results (PCA, VIF)
        regime: Regime dependence results
        temporal: Temporal analysis results (stationarity, ACF)
        cross_scale: Cross-scale correlation results
        summary: Summary statistics
    """

    distribution: dict[str, Any] = field(default_factory=dict)
    dimensionality: dict[str, Any] = field(default_factory=dict)
    regime: dict[str, Any] = field(default_factory=dict)
    temporal: dict[str, Any] = field(default_factory=dict)
    cross_scale: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "distribution": self.distribution,
            "dimensionality": self.dimensionality,
            "regime": self.regime,
            "temporal": self.temporal,
            "cross_scale": self.cross_scale,
            "summary": self.summary,
        }


def analyze_features(
    store: FeatureStore,
    config: AnalysisConfig | None = None,
    nav: Any | None = None,
) -> AnalysisResults:
    """Run statistical analysis on features from FeatureStore.

    This is the main entry point for Layer 3 analysis. It reads from the
    FeatureStore (Long Format SSoT) and runs various statistical analyses.

    Args:
        store: FeatureStore containing features in Long Format
        config: Analysis configuration (optional, uses defaults)
        nav: NAV array for regime detection (optional)

    Returns:
        AnalysisResults containing all analysis outputs
    """
    if config is None:
        config = AnalysisConfig()

    results = AnalysisResults()

    # Convert to wide format for analysis
    wide_df = store.to_wide(drop_warmup=True)

    if len(wide_df) == 0:
        results.summary = {"error": "No data after dropping warmup"}
        return results

    # Get feature columns
    feature_cols = [c for c in wide_df.columns if c.startswith("ith_")]

    # Limit features for speed
    if len(feature_cols) > config.max_features:
        feature_cols = feature_cols[: config.max_features]

    # Run distribution analysis
    if config.run_distribution:
        results.distribution = _run_distribution_analysis(wide_df, feature_cols)

    # Run dimensionality analysis
    if config.run_dimensionality:
        results.dimensionality = _run_dimensionality_analysis(wide_df, feature_cols)

    # Run regime analysis (requires NAV)
    if config.run_regime and nav is not None:
        results.regime = _run_regime_analysis(wide_df, feature_cols, nav, store)

    # Run temporal analysis
    if config.run_temporal:
        results.temporal = _run_temporal_analysis(wide_df, feature_cols)

    # Run cross-scale correlation
    if config.run_cross_scale:
        results.cross_scale = _run_cross_scale_analysis(wide_df)

    # Compile summary
    results.summary = _compile_summary(results, len(wide_df), len(feature_cols))

    return results


def _run_distribution_analysis(wide_df: Any, feature_cols: list[str]) -> dict[str, Any]:
    """Run distribution analysis with proper error handling."""
    from ith_python.statistical_examination.distribution import (
        analyze_all_distributions,
    )

    try:
        return analyze_all_distributions(wide_df, feature_cols)
    except (ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"Distribution analysis failed: {e}")
        return {"error": str(e)}


def _run_dimensionality_analysis(wide_df: Any, feature_cols: list[str]) -> dict[str, Any]:
    """Run dimensionality analysis with proper error handling."""
    from ith_python.statistical_examination.dimensionality import (
        compute_vif,
        perform_pca,
    )

    result: dict[str, Any] = {}

    try:
        result["pca"] = perform_pca(wide_df, feature_cols)
    except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as e:
        logger.warning(f"PCA analysis failed: {e}")
        result["pca"] = {"error": str(e)}

    try:
        result["vif"] = compute_vif(wide_df, feature_cols)
    except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as e:
        logger.warning(f"VIF analysis failed: {e}")
        result["vif"] = {"error": str(e)}

    return result


def _run_regime_analysis(
    wide_df: Any,
    feature_cols: list[str],
    nav: Any,
    store: FeatureStore,
) -> dict[str, Any]:
    """Run regime analysis with proper error handling."""
    from ith_python.statistical_examination.regime import (
        analyze_regime_dependence,
        detect_regime,
        summarize_regime_dependence,
    )

    try:
        min_lookback = min(store.get_lookbacks()) if store.get_lookbacks() else 20
        regimes = detect_regime(nav, lookback=min_lookback)

        # Align regimes with feature data
        if len(regimes) >= len(wide_df):
            regimes_aligned = regimes[len(regimes) - len(wide_df) :]
            regime_results = analyze_regime_dependence(
                wide_df, regimes_aligned, feature_cols
            )
            return summarize_regime_dependence(regime_results)

        return {"error": "Regime length mismatch"}
    except (ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"Regime analysis failed: {e}")
        return {"error": str(e)}


def _run_temporal_analysis(wide_df: Any, feature_cols: list[str]) -> dict[str, Any]:
    """Run temporal analysis with proper error handling."""
    from ith_python.statistical_examination.temporal import (
        analyze_temporal_structure,
    )

    try:
        return analyze_temporal_structure(wide_df, feature_cols)
    except (ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"Temporal analysis failed: {e}")
        return {"error": str(e)}


def _run_cross_scale_analysis(wide_df: Any) -> dict[str, Any]:
    """Run cross-scale correlation analysis with proper error handling."""
    from ith_python.statistical_examination.cross_scale import (
        compute_all_cross_scale_correlations,
    )

    try:
        return compute_all_cross_scale_correlations(wide_df)
    except (ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"Cross-scale analysis failed: {e}")
        return {"error": str(e)}


def _compile_summary(
    results: AnalysisResults,
    n_rows: int,
    n_features: int,
) -> dict[str, Any]:
    """Compile summary statistics from analysis results."""
    summary: dict[str, Any] = {
        "n_rows": n_rows,
        "n_features": n_features,
    }

    # Distribution summary
    if "summary" in results.distribution:
        dist_summary = results.distribution["summary"]
        summary["normality_rate"] = dist_summary.get("normality_rate", 0)
        summary["beta_fit_rate"] = dist_summary.get("beta_fit_rate", 0)

    # Dimensionality summary
    if "pca" in results.dimensionality and "error" not in results.dimensionality["pca"]:
        pca = results.dimensionality["pca"]
        summary["n_components_95"] = pca.get("n_components_95_variance")
        summary["participation_ratio"] = pca.get("participation_ratio")
        summary["dimensionality_ratio"] = pca.get("dimensionality_ratio_95")

    # Temporal summary
    if "summary" in results.temporal:
        temporal = results.temporal["summary"]
        summary["stationarity_rate"] = temporal.get("stationarity_rate", 0)

    # Regime summary
    if "invariance_rate" in results.regime:
        summary["regime_invariance_rate"] = results.regime["invariance_rate"]

    return summary


# Import numpy for LinAlgError exception handling
