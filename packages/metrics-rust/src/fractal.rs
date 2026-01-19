// # PROCESS-STORM-OK: Rust test functions with single-letter endings trigger false positives
//! Fractal metrics for BiLSTM feature engineering.
//!
//! All metrics produce bounded [0, 1] outputs.
//!
//! | Metric | Raw Range | Transform | Output |
//! |--------|-----------|-----------|--------|
//! | Hurst Exponent | [0, 1+] | soft_clamp | [0, 1] |
//! | Fractal Dimension | [1, 2] | D-1 | [0, 1] |

use crate::adaptive::{adaptive_windows, hurst_soft_clamp, MinimumSamples};

// ============================================================================
// Hurst Exponent (DFA Method)
// ============================================================================

/// Calculate Hurst Exponent using Detrended Fluctuation Analysis (DFA).
///
/// The Hurst exponent measures long-term memory in a time series:
/// - H < 0.5: Mean-reverting (anti-persistent)
/// - H = 0.5: Random walk (no memory)
/// - H > 0.5: Trending (persistent)
///
/// Note: DFA can produce H > 1 for non-stationary financial data.
/// Output is normalized using `hurst_soft_clamp` to [0, 1].
///
/// # Arguments
///
/// * `prices` - Slice of price values
///
/// # Returns
///
/// Hurst exponent normalized to [0, 1], or NaN if input is too short.
///
/// # Example
///
/// ```rust
/// use trading_fitness_metrics::hurst_exponent;
///
/// let prices: Vec<f64> = (0..300).map(|i| 100.0 + i as f64 * 0.1).collect();
/// let h = hurst_exponent(&prices);
/// assert!(h >= 0.0 && h <= 1.0);
/// ```
pub fn hurst_exponent(prices: &[f64]) -> f64 {
    let min_samples = MinimumSamples::hurst_exponent();
    if prices.len() < min_samples {
        return f64::NAN;
    }

    // Compute returns (first differences of log prices)
    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| {
            if w[0] > 0.0 && w[1] > 0.0 {
                (w[1] / w[0]).ln()
            } else {
                0.0
            }
        })
        .collect();

    if returns.is_empty() {
        return f64::NAN;
    }

    // Get adaptive window sizes for DFA
    let windows = adaptive_windows(returns.len(), 5);
    if windows.len() < 2 {
        return f64::NAN;
    }

    // Compute DFA for each window size
    let mut log_n = Vec::with_capacity(windows.len());
    let mut log_f = Vec::with_capacity(windows.len());

    for &window_size in &windows {
        if let Some(fluctuation) = dfa_fluctuation(&returns, window_size) {
            if fluctuation > 0.0 {
                log_n.push((window_size as f64).ln());
                log_f.push(fluctuation.ln());
            }
        }
    }

    if log_n.len() < 2 {
        return f64::NAN;
    }

    // Linear regression to find slope (Hurst exponent)
    let raw_hurst = linear_regression_slope(&log_n, &log_f);

    if raw_hurst.is_finite() {
        hurst_soft_clamp(raw_hurst)
    } else {
        f64::NAN
    }
}

/// Compute DFA fluctuation for a given window size.
fn dfa_fluctuation(data: &[f64], window_size: usize) -> Option<f64> {
    if data.len() < window_size || window_size < 4 {
        return None;
    }

    // Integrate the series (cumulative sum of mean-subtracted values)
    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    let integrated: Vec<f64> = data
        .iter()
        .scan(0.0, |acc, &x| {
            *acc += x - mean;
            Some(*acc)
        })
        .collect();

    // Split into non-overlapping windows and detrend each
    let n_windows = integrated.len() / window_size;
    if n_windows == 0 {
        return None;
    }

    let mut sum_sq_residuals = 0.0;
    let mut total_points = 0;

    for i in 0..n_windows {
        let start = i * window_size;
        let end = start + window_size;
        let segment = &integrated[start..end];

        // Linear detrend: fit y = a + b*x and compute residuals
        let residuals_sq = detrend_residuals_squared(segment);
        sum_sq_residuals += residuals_sq;
        total_points += window_size;
    }

    if total_points == 0 {
        return None;
    }

    // RMS fluctuation
    Some((sum_sq_residuals / total_points as f64).sqrt())
}

/// Compute sum of squared residuals after linear detrending.
fn detrend_residuals_squared(segment: &[f64]) -> f64 {
    let n = segment.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    // Compute linear fit y = a + b*x
    let x_mean = (n - 1.0) / 2.0;
    let y_mean: f64 = segment.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, &y) in segment.iter().enumerate() {
        let x = i as f64;
        numerator += (x - x_mean) * (y - y_mean);
        denominator += (x - x_mean) * (x - x_mean);
    }

    let slope = if denominator.abs() > f64::EPSILON {
        numerator / denominator
    } else {
        0.0
    };
    let intercept = y_mean - slope * x_mean;

    // Sum of squared residuals
    let mut sum_sq = 0.0;
    for (i, &y) in segment.iter().enumerate() {
        let predicted = intercept + slope * i as f64;
        let residual = y - predicted;
        sum_sq += residual * residual;
    }

    sum_sq
}

/// Simple linear regression to find slope.
fn linear_regression_slope(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }

    let n = x.len() as f64;
    let x_mean: f64 = x.iter().sum::<f64>() / n;
    let y_mean: f64 = y.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        numerator += (xi - x_mean) * (yi - y_mean);
        denominator += (xi - x_mean) * (xi - x_mean);
    }

    if denominator.abs() > f64::EPSILON {
        numerator / denominator
    } else {
        f64::NAN
    }
}

// ============================================================================
// Fractal Dimension (Higuchi's Method)
// ============================================================================

/// Calculate Fractal Dimension using Higuchi's method, normalized to [0, 1].
///
/// The fractal dimension measures the complexity/roughness of a time series:
/// - D ≈ 1: Smooth, trending
/// - D ≈ 1.5: Typical turbulent market
/// - D ≈ 2: Highly fragmented, noise-like
///
/// Raw output is [1, 2], normalized to [0, 1] via D - 1.
///
/// # Arguments
///
/// * `prices` - Slice of price values
/// * `k_max` - Maximum scale parameter (typically 10-50)
///
/// # Returns
///
/// Fractal dimension normalized to [0, 1], or NaN if input is too short.
///
/// # Example
///
/// ```rust
/// use trading_fitness_metrics::fractal_dimension;
///
/// let prices: Vec<f64> = (0..200).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();
/// let fd = fractal_dimension(&prices, 10);
/// assert!(fd >= 0.0 && fd <= 1.0);
/// ```
pub fn fractal_dimension(prices: &[f64], k_max: usize) -> f64 {
    if prices.len() < k_max + 1 || k_max < 2 {
        return f64::NAN;
    }

    let n = prices.len();
    let mut log_k = Vec::with_capacity(k_max);
    let mut log_l = Vec::with_capacity(k_max);

    for k in 1..=k_max {
        if let Some(length) = higuchi_curve_length(prices, k, n) {
            if length > 0.0 {
                log_k.push((k as f64).ln());
                log_l.push(length.ln());
            }
        }
    }

    if log_k.len() < 2 {
        return f64::NAN;
    }

    // Fractal dimension is negative slope of log(L) vs log(k)
    let raw_fd = -linear_regression_slope(&log_k, &log_l);

    if raw_fd.is_finite() {
        // Normalize [1, 2] to [0, 1]
        (raw_fd - 1.0).clamp(0.0, 1.0)
    } else {
        f64::NAN
    }
}

/// Compute Higuchi curve length for scale k.
fn higuchi_curve_length(data: &[f64], k: usize, n: usize) -> Option<f64> {
    if k == 0 || n < k + 1 {
        return None;
    }

    let mut total_length = 0.0;
    let mut valid_starts = 0;

    // For each starting point m in [1, k]
    for m in 1..=k {
        let mut length = 0.0;
        let num_points = (n - m) / k;

        if num_points < 1 {
            continue;
        }

        for i in 1..=num_points {
            let idx1 = m + (i - 1) * k - 1; // Convert to 0-indexed
            let idx2 = m + i * k - 1;

            if idx2 < data.len() {
                length += (data[idx2] - data[idx1]).abs();
            }
        }

        // Normalize by the number of intervals and scale
        let normalization = (n - 1) as f64 / (num_points * k) as f64 / k as f64;
        total_length += length * normalization;
        valid_starts += 1;
    }

    if valid_starts > 0 {
        Some(total_length / valid_starts as f64)
    } else {
        None
    }
}

/// Fractal dimension with adaptive k_max based on data length.
pub fn fractal_dimension_adaptive(prices: &[f64]) -> f64 {
    // k_max should be roughly sqrt(n) / 2, bounded by [5, 50]
    let k_max = ((prices.len() as f64).sqrt() / 2.0).round() as usize;
    let k_max = k_max.clamp(5, 50);
    fractal_dimension(prices, k_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hurst Exponent tests
    #[test]
    fn test_hurst_trending_series() {
        // Strictly increasing prices should have H > 0.5
        let prices: Vec<f64> = (0..300).map(|i| 100.0 + i as f64 * 0.5).collect();
        let h = hurst_exponent(&prices);
        if h.is_finite() {
            assert!(h > 0.5, "Trending series should have H > 0.5, got {}", h);
            assert!(h <= 1.0);
        }
    }

    #[test]
    fn test_hurst_bounded() {
        let prices: Vec<f64> = (0..300)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0)
            .collect();
        let h = hurst_exponent(&prices);
        if h.is_finite() {
            assert!(h >= 0.0 && h <= 1.0);
        }
    }

    #[test]
    fn test_hurst_short_input() {
        let prices = vec![100.0; 50];
        let h = hurst_exponent(&prices);
        assert!(h.is_nan());
    }

    // Fractal Dimension tests
    #[test]
    fn test_fractal_dimension_smooth() {
        // Smooth trending series should have low fractal dimension
        let prices: Vec<f64> = (0..200).map(|i| 100.0 + i as f64 * 0.1).collect();
        let fd = fractal_dimension(&prices, 10);
        if fd.is_finite() {
            assert!(fd < 0.5, "Smooth series should have low FD, got {}", fd);
        }
    }

    #[test]
    fn test_fractal_dimension_bounded() {
        let prices: Vec<f64> = (0..200)
            .map(|i| 100.0 + (i as f64 * 0.3).sin() * 5.0)
            .collect();
        let fd = fractal_dimension(&prices, 10);
        if fd.is_finite() {
            assert!(fd >= 0.0 && fd <= 1.0);
        }
    }

    #[test]
    fn test_fractal_dimension_short_input() {
        let prices = vec![100.0; 10];
        let fd = fractal_dimension(&prices, 15);
        assert!(fd.is_nan());
    }

    #[test]
    fn test_fractal_dimension_adaptive() {
        let prices: Vec<f64> = (0..500)
            .map(|i| 100.0 + (i as f64 * 0.2).sin() * 10.0)
            .collect();
        let fd = fractal_dimension_adaptive(&prices);
        if fd.is_finite() {
            assert!(fd >= 0.0 && fd <= 1.0);
        }
    }

    // Linear regression tests
    #[test]
    fn test_linear_regression_slope_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let slope = linear_regression_slope(&x, &y);
        assert!((slope - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_linear_regression_slope_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let slope = linear_regression_slope(&x, &y);
        assert!((slope - (-2.0)).abs() < 0.001);
    }
}
