//! Risk metrics for BiLSTM feature engineering.
//!
//! All metrics produce bounded [0, 1] outputs.
//!
//! | Metric | Raw Range | Transform | Output |
//! |--------|-----------|-----------|--------|
//! | Omega Ratio | [0, ∞) | Ω/(1+Ω) | [0, 1) |
//! | Ulcer Index | [0, 1] | None | [0, 1] |
//! | Garman-Klass Vol | [0, ∞) | tanh(10x) fallback | [0, 1) |
//! | Garman-Klass Vol (streaming) | [0, ∞) | EMA sigmoid | (0, 1) |
//! | Kaufman ER | [0, 1] | None | [0, 1] |
//!
//! ## Garman-Klass Normalization
//!
//! Two variants are provided:
//! - **Stateless** (`garman_klass_volatility`): Uses `tanh(10x)` fallback for single-bar use.
//!   Suitable for batch processing where historical context is unavailable.
//! - **Streaming** (`garman_klass_volatility_streaming`): Uses `GarmanKlassNormalizer` with
//!   EMA-based z-score normalization. Adapts to changing volatility regimes and is
//!   recommended for real-time processing with sufficient warmup period.

use crate::adaptive::{relative_epsilon, GarmanKlassNormalizer};
use crate::types::OhlcBar;

// ============================================================================
// Omega Ratio
// ============================================================================

/// Calculate Omega ratio normalized to [0, 1).
///
/// Formula: Ω(t) = E[(R-t)⁺] / E[(t-R)⁻], normalized as Ω/(1+Ω)
///
/// The Omega ratio compares probability-weighted gains to losses relative
/// to a threshold. It aligns well with the ITH excess gain/loss framework.
///
/// # Arguments
///
/// * `returns` - Slice of return values
/// * `threshold` - Return threshold (typically 0.0)
///
/// # Returns
///
/// Omega ratio normalized to [0, 1), or NaN if input is empty.
///
/// # Example
///
/// ```rust
/// use trading_fitness_metrics::omega_ratio;
///
/// let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015];
/// let omega = omega_ratio(&returns, 0.0);
/// assert!(omega >= 0.0 && omega < 1.0);
/// ```
pub fn omega_ratio(returns: &[f64], threshold: f64) -> f64 {
    if returns.is_empty() {
        return f64::NAN;
    }

    let gains: f64 = returns.iter().map(|r| (r - threshold).max(0.0)).sum();
    let losses: f64 = returns.iter().map(|r| (threshold - r).max(0.0)).sum();

    let epsilon = relative_epsilon(losses);
    let omega = gains / (losses + epsilon);

    // Normalize to [0, 1)
    omega / (1.0 + omega)
}

/// Omega ratio with configurable threshold.
///
/// Default threshold is 0.0 (classic Keating & Shadwick definition).
///
/// NOTE: Rolling mean threshold was rejected as it converges to Ω≈1.0
/// for any distribution, defeating the metric's discriminating power.
pub fn omega_ratio_adaptive(returns: &[f64], threshold: Option<f64>) -> f64 {
    let threshold = threshold.unwrap_or(0.0);
    omega_ratio(returns, threshold)
}

// ============================================================================
// Ulcer Index
// ============================================================================

/// Calculate Ulcer Index.
///
/// Formula: UI = sqrt(mean(drawdown_i²))
///
/// The Ulcer Index penalizes deeper drawdowns quadratically, making it
/// more sensitive to severe drawdowns than simple max drawdown.
///
/// # Arguments
///
/// * `prices` - Slice of price values
///
/// # Returns
///
/// Ulcer Index in [0, 1], or NaN if input is empty.
///
/// # Example
///
/// ```rust
/// use trading_fitness_metrics::ulcer_index;
///
/// let prices = vec![100.0, 102.0, 98.0, 101.0, 99.0];
/// let ui = ulcer_index(&prices);
/// assert!(ui >= 0.0 && ui <= 1.0);
/// ```
pub fn ulcer_index(prices: &[f64]) -> f64 {
    if prices.is_empty() {
        return f64::NAN;
    }

    if prices.len() == 1 {
        return 0.0;
    }

    let mut running_max = prices[0];
    let mut sum_squared_dd = 0.0;

    for &price in prices {
        running_max = running_max.max(price);
        if running_max > 0.0 {
            let dd = (running_max - price) / running_max;
            sum_squared_dd += dd * dd;
        }
    }

    (sum_squared_dd / prices.len() as f64).sqrt()
}

// ============================================================================
// Garman-Klass Volatility
// ============================================================================

/// Constant for Garman-Klass formula: 2*ln(2) - 1
const GK_CONST: f64 = 2.0 * std::f64::consts::LN_2 - 1.0;

/// Calculate raw Garman-Klass volatility (not normalized).
///
/// Formula: σ_gk = sqrt(0.5 * log(H/L)² - (2*ln(2)-1) * log(C/O)²)
///
/// # Returns
///
/// Raw volatility value (unbounded), or NaN if inputs are invalid.
pub fn garman_klass_volatility_raw(open: f64, high: f64, low: f64, close: f64) -> f64 {
    if open <= 0.0 || high <= 0.0 || low <= 0.0 || close <= 0.0 || high < low {
        return f64::NAN;
    }

    let hl = (high / low).ln();
    let co = (close / open).ln();

    let variance = 0.5 * hl * hl - GK_CONST * co * co;

    if variance >= 0.0 {
        variance.sqrt()
    } else {
        0.0 // Can happen with unusual OHLC combinations
    }
}

/// Calculate Garman-Klass volatility from OHLC, normalized using tanh fallback.
///
/// This is a **stateless** function that uses `tanh(raw * 10)` for normalization.
/// The scale factor 10 is derived from typical crypto volatility ranges where
/// raw GK values are typically in [0, 0.1] for normal markets.
///
/// For streaming normalization that adapts to market regimes, use
/// [`garman_klass_volatility_streaming`] instead.
///
/// # Arguments
///
/// * `open`, `high`, `low`, `close` - OHLC price values
///
/// # Returns
///
/// Volatility normalized to [0, 1), or NaN if inputs are invalid.
///
/// # Example
///
/// ```rust
/// use trading_fitness_metrics::garman_klass_volatility;
///
/// let vol = garman_klass_volatility(100.0, 105.0, 95.0, 102.0);
/// assert!(vol >= 0.0 && vol < 1.0);
/// ```
pub fn garman_klass_volatility(open: f64, high: f64, low: f64, close: f64) -> f64 {
    let raw = garman_klass_volatility_raw(open, high, low, close);
    if raw.is_nan() {
        return f64::NAN;
    }

    // Stateless fallback normalization using tanh with scale factor.
    // Scale factor 10 maps typical crypto volatility [0, 0.1] to [0, ~0.76].
    // For streaming use with adaptive normalization, use garman_klass_volatility_streaming.
    (raw * 10.0).tanh()
}

/// Calculate Garman-Klass volatility from an OhlcBar struct.
pub fn garman_klass_volatility_from_bar(bar: &OhlcBar) -> f64 {
    garman_klass_volatility(bar.open, bar.high, bar.low, bar.close)
}

/// Calculate Garman-Klass volatility with EMA-based streaming normalization.
///
/// Uses [`GarmanKlassNormalizer`] for adaptive normalization that adjusts to
/// changing market volatility regimes. Recommended for real-time processing.
///
/// # Arguments
///
/// * `normalizer` - Mutable reference to a `GarmanKlassNormalizer`
/// * `open`, `high`, `low`, `close` - OHLC price values
///
/// # Returns
///
/// Volatility normalized to (0, 1), or NaN if inputs are invalid.
///
/// # Example
///
/// ```rust
/// use trading_fitness_metrics::{garman_klass_volatility_streaming, GarmanKlassNormalizer};
///
/// let mut normalizer = GarmanKlassNormalizer::new(100);
///
/// // First value returns 0.5 (neutral)
/// let vol1 = garman_klass_volatility_streaming(&mut normalizer, 100.0, 105.0, 95.0, 102.0);
/// assert!((vol1 - 0.5).abs() < f64::EPSILON);
///
/// // Subsequent values adapt to the data
/// let vol2 = garman_klass_volatility_streaming(&mut normalizer, 100.0, 110.0, 90.0, 105.0);
/// assert!(vol2 > 0.0 && vol2 < 1.0);
/// ```
pub fn garman_klass_volatility_streaming(
    normalizer: &mut GarmanKlassNormalizer,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
) -> f64 {
    let raw = garman_klass_volatility_raw(open, high, low, close);
    if raw.is_nan() {
        return f64::NAN;
    }
    normalizer.normalize(raw)
}

/// Calculate Garman-Klass volatility from an OhlcBar with streaming normalization.
pub fn garman_klass_volatility_streaming_from_bar(
    normalizer: &mut GarmanKlassNormalizer,
    bar: &OhlcBar,
) -> f64 {
    garman_klass_volatility_streaming(normalizer, bar.open, bar.high, bar.low, bar.close)
}

// ============================================================================
// Kaufman Efficiency Ratio
// ============================================================================

/// Calculate Kaufman Efficiency Ratio.
///
/// Formula: ER = |end - start| / Σ|changes|
///
/// Measures the efficiency of price movement - how much of the total
/// price movement contributes to the net directional change.
///
/// # Arguments
///
/// * `prices` - Slice of price values
///
/// # Returns
///
/// Efficiency ratio in [0, 1], or NaN if input is too short.
///
/// # Example
///
/// ```rust
/// use trading_fitness_metrics::kaufman_efficiency_ratio;
///
/// // Perfect trend: ER = 1.0
/// let trending = vec![100.0, 101.0, 102.0, 103.0, 104.0];
/// let er = kaufman_efficiency_ratio(&trending);
/// assert!((er - 1.0).abs() < 0.01);
///
/// // Random walk: ER closer to 0
/// let choppy = vec![100.0, 101.0, 100.0, 101.0, 100.0];
/// let er2 = kaufman_efficiency_ratio(&choppy);
/// assert!(er2 < 0.5);
/// ```
pub fn kaufman_efficiency_ratio(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return f64::NAN;
    }

    let directional = (prices.last().unwrap() - prices.first().unwrap()).abs();
    let total: f64 = prices.windows(2).map(|w| (w[1] - w[0]).abs()).sum();

    let epsilon = relative_epsilon(total);
    directional / (total + epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Omega Ratio tests
    #[test]
    fn test_omega_ratio_all_gains() {
        let returns = vec![0.01, 0.02, 0.03, 0.01, 0.02];
        let omega = omega_ratio(&returns, 0.0);
        // All gains → omega close to 1.0
        assert!(omega > 0.99);
    }

    #[test]
    fn test_omega_ratio_all_losses() {
        let returns = vec![-0.01, -0.02, -0.03, -0.01, -0.02];
        let omega = omega_ratio(&returns, 0.0);
        // All losses → omega close to 0.0
        assert!(omega < 0.01);
    }

    #[test]
    fn test_omega_ratio_balanced() {
        let returns = vec![0.01, -0.01, 0.01, -0.01];
        let omega = omega_ratio(&returns, 0.0);
        // Balanced → omega close to 0.5
        assert!((omega - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_omega_ratio_empty() {
        let omega = omega_ratio(&[], 0.0);
        assert!(omega.is_nan());
    }

    // Ulcer Index tests
    #[test]
    fn test_ulcer_index_no_drawdown() {
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let ui = ulcer_index(&prices);
        assert!((ui - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ulcer_index_with_drawdown() {
        let prices = vec![100.0, 110.0, 90.0, 100.0];
        let ui = ulcer_index(&prices);
        assert!(ui > 0.0);
    }

    #[test]
    fn test_ulcer_index_bounded() {
        let prices = vec![100.0, 50.0, 25.0, 10.0];
        let ui = ulcer_index(&prices);
        assert!(ui >= 0.0 && ui <= 1.0);
    }

    // Garman-Klass tests
    #[test]
    fn test_garman_klass_flat_bar() {
        let vol = garman_klass_volatility(100.0, 100.0, 100.0, 100.0);
        assert!((vol - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_garman_klass_bounded() {
        let vol = garman_klass_volatility(100.0, 110.0, 90.0, 105.0);
        assert!(vol >= 0.0 && vol < 1.0);
    }

    #[test]
    fn test_garman_klass_invalid_ohlc() {
        // High < Low
        let vol = garman_klass_volatility(100.0, 90.0, 110.0, 100.0);
        assert!(vol.is_nan());
    }

    #[test]
    fn test_garman_klass_streaming_initial() {
        let mut norm = GarmanKlassNormalizer::new(100);
        let vol = garman_klass_volatility_streaming(&mut norm, 100.0, 105.0, 95.0, 102.0);
        // First value should be neutral (0.5)
        assert!((vol - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_garman_klass_streaming_bounded() {
        let mut norm = GarmanKlassNormalizer::new(100);

        // Process multiple bars
        for i in 0..50 {
            let range = 1.0 + (i as f64 * 0.1);
            let vol = garman_klass_volatility_streaming(
                &mut norm,
                100.0,
                100.0 + range,
                100.0 - range,
                100.0 + range * 0.5,
            );
            // All values should be bounded (0, 1)
            assert!(vol > 0.0 && vol < 1.0, "vol {} not in (0, 1)", vol);
        }
    }

    #[test]
    fn test_garman_klass_streaming_adapts() {
        let mut norm = GarmanKlassNormalizer::new(20);

        // Low volatility regime
        for _ in 0..10 {
            garman_klass_volatility_streaming(&mut norm, 100.0, 101.0, 99.0, 100.5);
        }

        // Sudden high volatility
        let vol_spike = garman_klass_volatility_streaming(&mut norm, 100.0, 120.0, 80.0, 110.0);

        // Should be well above 0.5 since it's much higher than recent history
        assert!(vol_spike > 0.7, "Expected vol_spike > 0.7, got {}", vol_spike);
    }

    #[test]
    fn test_garman_klass_streaming_invalid_ohlc() {
        let mut norm = GarmanKlassNormalizer::new(100);
        let vol = garman_klass_volatility_streaming(&mut norm, 100.0, 90.0, 110.0, 100.0);
        assert!(vol.is_nan());
    }

    // Kaufman ER tests
    #[test]
    fn test_kaufman_er_perfect_trend() {
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let er = kaufman_efficiency_ratio(&prices);
        assert!((er - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_kaufman_er_choppy() {
        let prices = vec![100.0, 101.0, 100.0, 101.0, 100.0];
        let er = kaufman_efficiency_ratio(&prices);
        // Net movement = 0, total movement = 4 → ER close to 0
        assert!(er < 0.01);
    }

    #[test]
    fn test_kaufman_er_single_point() {
        let er = kaufman_efficiency_ratio(&[100.0]);
        assert!(er.is_nan());
    }
}
