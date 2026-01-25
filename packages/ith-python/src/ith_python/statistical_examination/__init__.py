"""Statistical Examination Framework for ITH Multi-Scale Features.

This module provides comprehensive statistical analysis of ITH features across:
- 12 lookback windows: 20, 50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000 bars
- 6 range bar thresholds: 25, 50, 100, 250, 500, 1000 dbps
- 8 features per lookback: bull_ed, bear_ed, bull_eg, bear_eg, bull_cv, bear_cv, max_dd, max_ru

Data Flow:
    Rust compute_multiscale_ith() -> Arrow RecordBatch (zero-copy)
    -> Polars DataFrame (zero-copy) -> Statistical Analysis -> Parquet/NDJSON

Library Stack:
    - Arrow: Zero-copy data interchange from Rust
    - Polars: All data manipulation (NO pandas)
    - scipy.stats: Statistical tests via numpy bridge
    - sklearn/shap: ML feature analysis via numpy bridge
"""

from ith_python.statistical_examination._utils import (
    drop_warmup,
    extract_feature_type,
    extract_lookback,
    extract_threshold,
    get_feature_columns,
    get_warmup_bars,
)
from ith_python.statistical_examination.cross_scale import (
    compute_cross_scale_correlation,
    compute_all_cross_scale_correlations,
)
from ith_python.statistical_examination.dimensionality import (
    compute_vif,
    perform_pca,
)
from ith_python.statistical_examination.distribution import (
    analyze_distribution,
    analyze_all_distributions,
)
from ith_python.statistical_examination.feature_importance import (
    compute_mutual_information,
    compute_shap_importance,
)
from ith_python.statistical_examination.regime import (
    analyze_regime_dependence,
    detect_regime,
)
from ith_python.statistical_examination.runner import run_examination
from ith_python.statistical_examination.schemas import validate_ith_features
from ith_python.statistical_examination.selection import (
    filter_features,
    select_per_lookback,
)
from ith_python.statistical_examination.temporal import (
    compute_autocorrelation,
    compute_stationarity,
)

# Alias for backwards compatibility
test_stationarity = compute_stationarity
from ith_python.statistical_examination.threshold_stability import (
    compute_threshold_stability,
    compute_all_threshold_stability,
)

__all__ = [
    # Utils
    "drop_warmup",
    "extract_feature_type",
    "extract_lookback",
    "extract_threshold",
    "get_feature_columns",
    "get_warmup_bars",
    # Cross-scale
    "compute_cross_scale_correlation",
    "compute_all_cross_scale_correlations",
    # Threshold stability
    "compute_threshold_stability",
    "compute_all_threshold_stability",
    # Distribution
    "analyze_distribution",
    "analyze_all_distributions",
    # Regime
    "detect_regime",
    "analyze_regime_dependence",
    # Feature importance
    "compute_mutual_information",
    "compute_shap_importance",
    # Dimensionality
    "perform_pca",
    "compute_vif",
    # Selection
    "filter_features",
    "select_per_lookback",
    # Temporal
    "compute_autocorrelation",
    "compute_stationarity",
    "test_stationarity",  # Alias for backwards compatibility
    # Runner
    "run_examination",
    # Schemas
    "validate_ith_features",
]
