"""
trading-fitness-metrics: Time-agnostic ITH metrics for BiLSTM feature engineering.

All 9 metrics require only price data (no volume).
All outputs bounded [0, 1] for LSTM/BiLSTM input layers.

Metrics:
    Entropy:
        - permutation_entropy: Ordinal pattern complexity
        - sample_entropy: Regularity measure
        - shannon_entropy: Information content

    Risk:
        - omega_ratio: Probability-weighted gains/losses
        - ulcer_index: Drawdown risk
        - garman_klass_volatility: OHLC volatility estimator
        - kaufman_efficiency_ratio: Directional efficiency

    Fractal:
        - hurst_exponent: Market regime (trending/mean-reverting)
        - fractal_dimension: Market structure

ITH Analysis:
    - bull_ith: Long position profitability analysis
    - bear_ith: Short position profitability analysis
    - compute_rolling_ith: Rolling window ITH features (time-agnostic, bounded [0, 1])
    - compute_multiscale_ith: Multi-scale ITH features across multiple lookback windows

Multi-Scale ITH (Arrow-Native):
    - compute_multiscale_ith: Compute ITH features at multiple lookback scales
    - MultiscaleIthConfig: Configuration for multi-scale computation
    - MultiscaleIthFeatures: Container with to_arrow() for zero-copy Polars integration

Example:
    >>> import numpy as np
    >>> from trading_fitness_metrics import omega_ratio, permutation_entropy
    >>>
    >>> prices = np.array([100.0, 101.5, 99.8, 102.3, 101.0])
    >>> returns = np.diff(prices) / prices[:-1]
    >>>
    >>> # All outputs are bounded [0, 1]
    >>> omega = omega_ratio(returns, threshold=0.0)
    >>> pe = permutation_entropy(prices, m=3)
"""

from trading_fitness_metrics._core import (
    # Entropy metrics
    permutation_entropy,
    sample_entropy,
    shannon_entropy,
    # Risk metrics
    omega_ratio,
    ulcer_index,
    garman_klass_volatility,
    kaufman_efficiency_ratio,
    # Fractal metrics
    hurst_exponent,
    fractal_dimension,
    # NAV & utilities
    build_nav_from_closes,
    adaptive_windows,
    optimal_bins_freedman_diaconis,
    optimal_embedding_dimension,
    optimal_sample_entropy_tolerance,
    optimal_tmaeg,
    relative_epsilon,
    # ITH analysis
    bull_ith,
    bear_ith,
    compute_rolling_ith,
    # Multi-scale ITH (Arrow-native)
    compute_multiscale_ith,
    MultiscaleIthConfig,
    MultiscaleIthFeatures,
    # Classes
    RollingIthFeatures,
    # Batch API
    compute_all_metrics,
    # Classes
    BullIthResult,
    BearIthResult,
    GarmanKlassNormalizer,
    OnlineNormalizer,
    MetricsResult,
)

__all__ = [
    # Entropy metrics
    "permutation_entropy",
    "sample_entropy",
    "shannon_entropy",
    # Risk metrics
    "omega_ratio",
    "ulcer_index",
    "garman_klass_volatility",
    "kaufman_efficiency_ratio",
    # Fractal metrics
    "hurst_exponent",
    "fractal_dimension",
    # NAV & utilities
    "build_nav_from_closes",
    "adaptive_windows",
    "optimal_bins_freedman_diaconis",
    "optimal_embedding_dimension",
    "optimal_sample_entropy_tolerance",
    "optimal_tmaeg",
    "relative_epsilon",
    # ITH analysis
    "bull_ith",
    "bear_ith",
    "compute_rolling_ith",
    # Multi-scale ITH (Arrow-native)
    "compute_multiscale_ith",
    "MultiscaleIthConfig",
    "MultiscaleIthFeatures",
    # Classes
    "RollingIthFeatures",
    # Batch API
    "compute_all_metrics",
    # Classes
    "BullIthResult",
    "BearIthResult",
    "GarmanKlassNormalizer",
    "OnlineNormalizer",
    "MetricsResult",
]
