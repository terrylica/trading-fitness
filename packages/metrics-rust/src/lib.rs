//! Time-agnostic ITH metrics for BiLSTM feature engineering.
//!
//! All 9 metrics require only price data (no volume).
//! All outputs bounded [0, 1] for LSTM/BiLSTM input layers.
//! **Zero magic numbers**: All constants are data-adaptive or mathematically derived.
//!
//! # Metrics Overview
//!
//! | Category | Metrics |
//! |----------|---------|
//! | Entropy | Permutation, Sample, Shannon |
//! | Risk | Omega Ratio, Ulcer Index, Garman-Klass Vol, Kaufman ER |
//! | Fractal | Hurst Exponent, Fractal Dimension |
//!
//! # Example
//!
//! ```rust
//! use trading_fitness_metrics::{omega_ratio, permutation_entropy, hurst_exponent};
//!
//! let prices = vec![100.0, 101.5, 99.8, 102.3, 101.0];
//! let returns: Vec<f64> = prices.windows(2)
//!     .map(|w| (w[1] - w[0]) / w[0])
//!     .collect();
//!
//! // All outputs are bounded [0, 1]
//! let omega = omega_ratio(&returns, 0.0);
//! let pe = permutation_entropy(&prices, 3);
//! ```

pub mod adaptive;
pub mod entropy;
pub mod fractal;
pub mod ith;
pub mod nav;
pub mod risk;
pub mod types;

// Re-export public API (9 price-only metrics)
pub use entropy::{permutation_entropy, sample_entropy, shannon_entropy};
pub use fractal::{fractal_dimension, hurst_exponent};
pub use ith::{bear_ith, bull_ith};
pub use types::{BearIthResult, BullIthResult};
pub use nav::build_nav_from_closes;
pub use risk::{
    garman_klass_volatility, garman_klass_volatility_streaming, kaufman_efficiency_ratio,
    omega_ratio, ulcer_index,
};
pub use types::{IthEpoch, MetricsResult};

// Re-export adaptive utilities
pub use adaptive::{
    adaptive_windows, optimal_bins_freedman_diaconis, optimal_embedding_dimension,
    optimal_sample_entropy_tolerance, relative_epsilon, AdaptiveTolerance,
    GarmanKlassNormalizer, MinimumSamples, OnlineNormalizer,
};
