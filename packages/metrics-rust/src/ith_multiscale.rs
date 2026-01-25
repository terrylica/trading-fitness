//! Multi-scale ITH computation for LSTM feature engineering.
//!
//! Computes rolling ITH features across multiple lookback windows,
//! producing columnar outputs with standardized naming convention:
//! `ith_rb{threshold}_lb{lookback}_{feature}`
//!
//! All features are bounded [0, 1] for LSTM consumption.

use crate::ith_rolling::compute_rolling_ith;
use std::collections::HashMap;

/// Configuration for multi-scale ITH computation.
///
/// # Column Naming Convention
///
/// Output columns follow the pattern: `ith_rb{threshold}_lb{lookback}_{feature}`
/// - `threshold`: Range bar threshold in dbps (for column naming only)
/// - `lookback`: Lookback window in bars
/// - `feature`: Short feature name (e.g., `bull_ed`, `max_dd`)
///
/// # Feature Short Names
///
/// | Full Name | Short Name |
/// |-----------|------------|
/// | bull_epoch_density | bull_ed |
/// | bear_epoch_density | bear_ed |
/// | bull_excess_gain | bull_eg |
/// | bear_excess_gain | bear_eg |
/// | bull_cv | bull_cv |
/// | bear_cv | bear_cv |
/// | max_drawdown | max_dd |
/// | max_runup | max_ru |
#[derive(Debug, Clone)]
pub struct MultiscaleConfig {
    /// Range bar threshold in dbps (for column naming only, doesn't affect computation)
    pub threshold_dbps: u32,
    /// Lookback windows to compute
    pub lookbacks: Vec<usize>,
}

impl Default for MultiscaleConfig {
    fn default() -> Self {
        Self {
            threshold_dbps: 250,
            lookbacks: vec![20, 50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000],
        }
    }
}

impl MultiscaleConfig {
    /// Create a new configuration with specified threshold and lookbacks.
    pub fn new(threshold_dbps: u32, lookbacks: Vec<usize>) -> Self {
        Self {
            threshold_dbps,
            lookbacks,
        }
    }

    /// Create configuration with custom lookbacks and default threshold (250 dbps).
    pub fn with_lookbacks(lookbacks: Vec<usize>) -> Self {
        Self {
            threshold_dbps: 250,
            lookbacks,
        }
    }
}

/// Multi-scale ITH features across multiple lookback windows.
///
/// Contains a HashMap where keys are column names following the convention
/// `ith_rb{threshold}_lb{lookback}_{feature}` and values are feature arrays.
///
/// All feature values are bounded [0, 1] for LSTM consumption.
#[derive(Debug, Clone)]
pub struct MultiscaleIthFeatures {
    /// Feature columns: `ith_rb{threshold}_lb{lookback}_{feature}` -> Vec<f64>
    pub columns: HashMap<String, Vec<f64>>,
    /// Number of data points (length of each feature array)
    pub n_points: usize,
    /// Number of feature columns
    pub n_features: usize,
    /// Configuration used for computation
    pub config: MultiscaleConfig,
}

impl MultiscaleIthFeatures {
    /// Get a feature array by column name.
    pub fn get(&self, column_name: &str) -> Option<&Vec<f64>> {
        self.columns.get(column_name)
    }

    /// Get all column names.
    pub fn column_names(&self) -> Vec<String> {
        self.columns.keys().cloned().collect()
    }

    /// Check if all features are bounded [0, 1].
    pub fn all_bounded(&self) -> bool {
        self.columns.values().all(|arr| {
            arr.iter()
                .all(|&v| v.is_nan() || (v >= 0.0 && v <= 1.0))
        })
    }
}

/// Feature short name mapping.
const FEATURE_SHORT_NAMES: [(&str, &str); 8] = [
    ("bull_epoch_density", "bull_ed"),
    ("bear_epoch_density", "bear_ed"),
    ("bull_excess_gain", "bull_eg"),
    ("bear_excess_gain", "bear_eg"),
    ("bull_cv", "bull_cv"),
    ("bear_cv", "bear_cv"),
    ("max_drawdown", "max_dd"),
    ("max_runup", "max_ru"),
];

/// Compute multi-scale ITH features across multiple lookback windows.
///
/// # Arguments
/// * `nav` - NAV series (N samples, typically starting at 1.0)
/// * `config` - Multi-scale configuration specifying threshold and lookbacks
///
/// # Returns
/// `MultiscaleIthFeatures` containing all lookback × feature combinations.
/// Each feature array has length N, with the first `lookback-1` values as NaN.
///
/// # Column Naming
///
/// Output columns follow the pattern: `ith_rb{threshold}_lb{lookback}_{feature}`
///
/// # Example
///
/// ```rust
/// use trading_fitness_metrics::ith_multiscale::{compute_multiscale_ith, MultiscaleConfig};
///
/// let nav = vec![1.0, 1.01, 0.99, 1.02, 1.01, 1.03, 1.02, 1.04, 1.03, 1.05];
/// let config = MultiscaleConfig::new(250, vec![3, 5]);
/// let features = compute_multiscale_ith(&nav, &config);
///
/// // Access features by column name
/// let bull_ed_lb3 = features.get("ith_rb250_lb3_bull_ed");
/// assert!(bull_ed_lb3.is_some());
/// ```
pub fn compute_multiscale_ith(nav: &[f64], config: &MultiscaleConfig) -> MultiscaleIthFeatures {
    let mut columns = HashMap::new();
    let n = nav.len();

    for &lookback in &config.lookbacks {
        // Skip lookbacks that exceed data length
        if lookback >= n {
            continue;
        }

        // Compute rolling ITH for this lookback (auto-TMAEG)
        let features = compute_rolling_ith(nav, lookback);

        // Extract each feature with proper column naming
        for (full_name, short_name) in FEATURE_SHORT_NAMES.iter() {
            let col_name = format!(
                "ith_rb{}_lb{}_{}",
                config.threshold_dbps, lookback, short_name
            );

            let values = match *full_name {
                "bull_epoch_density" => features.bull_epoch_density.clone(),
                "bear_epoch_density" => features.bear_epoch_density.clone(),
                "bull_excess_gain" => features.bull_excess_gain.clone(),
                "bear_excess_gain" => features.bear_excess_gain.clone(),
                "bull_cv" => features.bull_cv.clone(),
                "bear_cv" => features.bear_cv.clone(),
                "max_drawdown" => features.max_drawdown.clone(),
                "max_runup" => features.max_runup.clone(),
                _ => unreachable!(),
            };

            columns.insert(col_name, values);
        }
    }

    let n_features = columns.len();
    MultiscaleIthFeatures {
        columns,
        n_points: n,
        n_features,
        config: config.clone(),
    }
}

/// Compute multi-scale ITH features with streaming callback for memory efficiency.
///
/// Instead of holding all features in memory, this function yields each feature
/// array to a callback function as it's computed.
///
/// # Arguments
/// * `nav` - NAV series (N samples)
/// * `config` - Multi-scale configuration
/// * `callback` - Function called with (column_name, feature_values) for each feature
///
/// # Example
///
/// ```rust
/// use trading_fitness_metrics::ith_multiscale::{compute_multiscale_ith_streaming, MultiscaleConfig};
///
/// let nav = vec![1.0, 1.01, 0.99, 1.02, 1.01];
/// let config = MultiscaleConfig::new(250, vec![3]);
///
/// compute_multiscale_ith_streaming(&nav, &config, |col_name, values| {
///     println!("{}: {} values", col_name, values.len());
/// });
/// ```
pub fn compute_multiscale_ith_streaming<F>(nav: &[f64], config: &MultiscaleConfig, mut callback: F)
where
    F: FnMut(&str, &[f64]),
{
    let n = nav.len();

    for &lookback in &config.lookbacks {
        if lookback >= n {
            continue;
        }

        let features = compute_rolling_ith(nav, lookback);

        // Yield each feature array
        for (full_name, short_name) in FEATURE_SHORT_NAMES.iter() {
            let col_name = format!(
                "ith_rb{}_lb{}_{}",
                config.threshold_dbps, lookback, short_name
            );

            let values: &[f64] = match *full_name {
                "bull_epoch_density" => &features.bull_epoch_density,
                "bear_epoch_density" => &features.bear_epoch_density,
                "bull_excess_gain" => &features.bull_excess_gain,
                "bear_excess_gain" => &features.bear_excess_gain,
                "bull_cv" => &features.bull_cv,
                "bear_cv" => &features.bear_cv,
                "max_drawdown" => &features.max_drawdown,
                "max_runup" => &features.max_runup,
                _ => unreachable!(),
            };

            callback(&col_name, values);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_nav(n: usize) -> Vec<f64> {
        let mut nav = vec![1.0];
        let mut rng_state: u64 = 42;
        for _ in 1..n {
            // Simple LCG for deterministic pseudo-random numbers
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5) * 0.02;
            let last = *nav.last().unwrap();
            nav.push(last * (1.0 + r));
        }
        nav
    }

    #[test]
    fn test_multiscale_config_default() {
        let config = MultiscaleConfig::default();
        assert_eq!(config.threshold_dbps, 250);
        assert_eq!(config.lookbacks.len(), 12);
        assert!(config.lookbacks.contains(&100));
        assert!(config.lookbacks.contains(&6000));
    }

    #[test]
    fn test_multiscale_config_custom() {
        let config = MultiscaleConfig::new(100, vec![10, 50, 100]);
        assert_eq!(config.threshold_dbps, 100);
        assert_eq!(config.lookbacks, vec![10, 50, 100]);
    }

    #[test]
    fn test_multiscale_column_naming() {
        let nav = sample_nav(200);
        let config = MultiscaleConfig::new(250, vec![20, 50]);
        let features = compute_multiscale_ith(&nav, &config);

        // Check column names follow convention
        assert!(features.columns.contains_key("ith_rb250_lb20_bull_ed"));
        assert!(features.columns.contains_key("ith_rb250_lb50_max_dd"));
        assert!(features.columns.contains_key("ith_rb250_lb20_bear_cv"));
    }

    #[test]
    fn test_multiscale_feature_count() {
        let nav = sample_nav(200);
        let config = MultiscaleConfig::new(250, vec![20, 50]);
        let features = compute_multiscale_ith(&nav, &config);

        // 2 lookbacks × 8 features = 16 columns
        assert_eq!(features.n_features, 16);
        assert_eq!(features.columns.len(), 16);
    }

    #[test]
    fn test_multiscale_all_bounded() {
        let nav = sample_nav(500);
        let config = MultiscaleConfig::new(250, vec![20, 50, 100]);
        let features = compute_multiscale_ith(&nav, &config);

        assert!(features.all_bounded());

        for (col_name, values) in &features.columns {
            for (i, &v) in values.iter().enumerate() {
                assert!(
                    v.is_nan() || (v >= 0.0 && v <= 1.0),
                    "{} at index {} has value {} which is not bounded",
                    col_name,
                    i,
                    v
                );
            }
        }
    }

    #[test]
    fn test_multiscale_nan_prefix() {
        let nav = sample_nav(200);
        let config = MultiscaleConfig::new(250, vec![20, 50]);
        let features = compute_multiscale_ith(&nav, &config);

        // First lookback-1 values should be NaN
        let bull_ed_20 = features.get("ith_rb250_lb20_bull_ed").unwrap();
        for i in 0..19 {
            assert!(bull_ed_20[i].is_nan(), "Index {} should be NaN", i);
        }
        assert!(!bull_ed_20[19].is_nan(), "Index 19 should not be NaN");

        let bull_ed_50 = features.get("ith_rb250_lb50_bull_ed").unwrap();
        for i in 0..49 {
            assert!(bull_ed_50[i].is_nan(), "Index {} should be NaN", i);
        }
        assert!(!bull_ed_50[49].is_nan(), "Index 49 should not be NaN");
    }

    #[test]
    fn test_multiscale_skips_large_lookbacks() {
        let nav = sample_nav(50);
        let config = MultiscaleConfig::new(250, vec![20, 100, 200]); // 100, 200 exceed data length
        let features = compute_multiscale_ith(&nav, &config);

        // Only lookback=20 should be computed (50 data points)
        assert_eq!(features.n_features, 8); // 8 features for 1 lookback
        assert!(features.columns.contains_key("ith_rb250_lb20_bull_ed"));
        assert!(!features.columns.contains_key("ith_rb250_lb100_bull_ed"));
        assert!(!features.columns.contains_key("ith_rb250_lb200_bull_ed"));
    }

    #[test]
    fn test_multiscale_streaming_callback() {
        let nav = sample_nav(100);
        let config = MultiscaleConfig::new(250, vec![20]);

        let mut count = 0;
        let mut column_names = Vec::new();

        compute_multiscale_ith_streaming(&nav, &config, |col_name, values| {
            count += 1;
            column_names.push(col_name.to_string());
            assert_eq!(values.len(), 100);
        });

        assert_eq!(count, 8); // 8 features for 1 lookback
        assert!(column_names.contains(&"ith_rb250_lb20_bull_ed".to_string()));
    }

    #[test]
    fn test_multiscale_empty_nav() {
        let nav: Vec<f64> = vec![];
        let config = MultiscaleConfig::new(250, vec![20]);
        let features = compute_multiscale_ith(&nav, &config);

        // No features computed for empty input
        assert_eq!(features.n_features, 0);
        assert_eq!(features.n_points, 0);
    }

    #[test]
    fn test_multiscale_get_method() {
        let nav = sample_nav(100);
        let config = MultiscaleConfig::new(250, vec![20]);
        let features = compute_multiscale_ith(&nav, &config);

        assert!(features.get("ith_rb250_lb20_bull_ed").is_some());
        assert!(features.get("nonexistent_column").is_none());
    }
}
