//! Type definitions for metrics results.
//!
//! All metric outputs are designed to be bounded [0, 1] for BiLSTM consumption.

use serde::{Deserialize, Serialize};

/// Result of computing all 9 price-only metrics.
///
/// All values are bounded [0, 1] for direct use in LSTM/BiLSTM models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsResult {
    // Entropy metrics
    pub permutation_entropy: f64,
    pub sample_entropy: f64,
    pub shannon_entropy: f64,

    // Risk metrics
    pub omega_ratio: f64,
    pub ulcer_index: f64,
    pub garman_klass_vol: f64,
    pub kaufman_er: f64,

    // Fractal metrics
    pub hurst_exponent: f64,
    pub fractal_dimension: f64,
}

impl MetricsResult {
    /// Check if all metrics are within valid bounds [0, 1].
    pub fn all_bounded(&self) -> bool {
        let values = [
            self.permutation_entropy,
            self.sample_entropy,
            self.shannon_entropy,
            self.omega_ratio,
            self.ulcer_index,
            self.garman_klass_vol,
            self.kaufman_er,
            self.hurst_exponent,
            self.fractal_dimension,
        ];

        values
            .iter()
            .all(|&v| v.is_finite() && v >= 0.0 && v <= 1.0)
    }

    /// Check if any metric is NaN.
    pub fn has_nan(&self) -> bool {
        let values = [
            self.permutation_entropy,
            self.sample_entropy,
            self.shannon_entropy,
            self.omega_ratio,
            self.ulcer_index,
            self.garman_klass_vol,
            self.kaufman_er,
            self.hurst_exponent,
            self.fractal_dimension,
        ];

        values.iter().any(|v| v.is_nan())
    }
}

/// Represents an ITH (Investment Time Horizon) epoch.
///
/// An epoch marks a period where the strategy exceeded the TMAEG threshold.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IthEpoch {
    /// Index in the NAV series where this epoch started.
    pub start_index: usize,
    /// Index in the NAV series where this epoch ended.
    pub end_index: usize,
    /// Excess gain at the epoch point.
    pub excess_gain: f64,
    /// Excess loss at the epoch point.
    pub excess_loss: f64,
}

/// Result of Bull ITH (long position) analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BullIthResult {
    /// Excess gains at each time point.
    pub excess_gains: Vec<f64>,
    /// Excess losses at each time point (drawdowns).
    pub excess_losses: Vec<f64>,
    /// Number of bull epochs detected.
    pub num_of_epochs: usize,
    /// Boolean array marking epoch points.
    pub epochs: Vec<bool>,
    /// Coefficient of variation of epoch intervals.
    pub intervals_cv: f64,
    /// Maximum drawdown observed.
    pub max_drawdown: f64,
}

/// Result of Bear ITH (short position) analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BearIthResult {
    /// Excess gains at each time point (from price drops).
    pub excess_gains: Vec<f64>,
    /// Excess losses at each time point (runups).
    pub excess_losses: Vec<f64>,
    /// Number of bear epochs detected.
    pub num_of_epochs: usize,
    /// Boolean array marking epoch points.
    pub epochs: Vec<bool>,
    /// Coefficient of variation of epoch intervals.
    pub intervals_cv: f64,
    /// Maximum runup observed (adverse for shorts).
    pub max_runup: f64,
}

/// OHLC (Open, High, Low, Close) price bar.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OhlcBar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

impl OhlcBar {
    /// Create a new OHLC bar with validation.
    ///
    /// Returns `None` if the bar is invalid (high < low, etc).
    pub fn new(open: f64, high: f64, low: f64, close: f64) -> Option<Self> {
        if high >= low && high >= open && high >= close && low <= open && low <= close {
            Some(Self {
                open,
                high,
                low,
                close,
            })
        } else {
            None
        }
    }

    /// Create a new OHLC bar without validation.
    ///
    /// # Safety
    ///
    /// Caller must ensure that high >= low and close is within [low, high].
    pub fn new_unchecked(open: f64, high: f64, low: f64, close: f64) -> Self {
        Self {
            open,
            high,
            low,
            close,
        }
    }

    /// Check if this bar is valid.
    pub fn is_valid(&self) -> bool {
        self.high >= self.low
            && self.high >= self.open
            && self.high >= self.close
            && self.low <= self.open
            && self.low <= self.close
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_result_all_bounded() {
        let result = MetricsResult {
            permutation_entropy: 0.5,
            sample_entropy: 0.3,
            shannon_entropy: 0.8,
            omega_ratio: 0.6,
            ulcer_index: 0.1,
            garman_klass_vol: 0.2,
            kaufman_er: 0.9,
            hurst_exponent: 0.5,
            fractal_dimension: 0.4,
        };
        assert!(result.all_bounded());
    }

    #[test]
    fn test_metrics_result_out_of_bounds() {
        let result = MetricsResult {
            permutation_entropy: 1.5, // Out of bounds
            sample_entropy: 0.3,
            shannon_entropy: 0.8,
            omega_ratio: 0.6,
            ulcer_index: 0.1,
            garman_klass_vol: 0.2,
            kaufman_er: 0.9,
            hurst_exponent: 0.5,
            fractal_dimension: 0.4,
        };
        assert!(!result.all_bounded());
    }

    #[test]
    fn test_ohlc_bar_valid() {
        let bar = OhlcBar::new(100.0, 105.0, 95.0, 102.0);
        assert!(bar.is_some());
        assert!(bar.unwrap().is_valid());
    }

    #[test]
    fn test_ohlc_bar_invalid() {
        // High less than low
        let bar = OhlcBar::new(100.0, 90.0, 95.0, 92.0);
        assert!(bar.is_none());
    }
}
