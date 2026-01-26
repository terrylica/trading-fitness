"""Feature Analysis Module (Layer 3).

This module handles statistical evaluation of features from the FeatureStore.
It wraps and re-exports functionality from the statistical_examination module.

Design Principles:
- Reads from FeatureStore (Long Format SSoT)
- Emits NDJSON telemetry for Claude Code forensics
- Upgradeable independently from compute/storage layers

Architecture: Multi-View Feature Architecture with Separation of Concerns
- Layer 3: Feature Analysis
- See: docs/plans/2026-01-25-multi-view-feature-architecture-plan.md
"""

# Re-export from statistical_examination for backwards compatibility
from ith_python.statistical_examination import (
    analyze_all_distributions,
    analyze_distribution,
    analyze_regime_dependence,
    compute_all_cross_scale_correlations,
    compute_autocorrelation,
    compute_cross_scale_correlation,
    compute_stationarity,
    compute_vif,
    detect_regime,
    drop_warmup,
    extract_feature_type,
    extract_lookback,
    extract_threshold,
    filter_features,
    get_feature_columns,
    get_warmup_bars,
    perform_pca,
    run_examination,
    select_per_lookback,
    validate_ith_features,
)

# New unified analysis interface
from ith_python.analysis.runner import (
    AnalysisConfig,
    AnalysisResults,
    analyze_features,
)

__all__ = [
    # Runner
    "AnalysisConfig",
    "AnalysisResults",
    "analyze_features",
    "run_examination",
    # Distribution
    "analyze_all_distributions",
    "analyze_distribution",
    # Cross-scale
    "compute_all_cross_scale_correlations",
    "compute_cross_scale_correlation",
    # Dimensionality
    "compute_vif",
    "perform_pca",
    # Regime
    "analyze_regime_dependence",
    "detect_regime",
    # Selection
    "filter_features",
    "select_per_lookback",
    # Temporal
    "compute_autocorrelation",
    "compute_stationarity",
    # Utils
    "drop_warmup",
    "extract_feature_type",
    "extract_lookback",
    "extract_threshold",
    "get_feature_columns",
    "get_warmup_bars",
    # Validation
    "validate_ith_features",
]
