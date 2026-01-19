// # PROCESS-STORM-OK: Rust test functions with single-letter endings trigger false positives
//! Entropy metrics for BiLSTM feature engineering.
//!
//! All metrics produce bounded [0, 1] outputs.
//!
//! | Metric | Raw Range | Transform | Output |
//! |--------|-----------|-----------|--------|
//! | Permutation Entropy | [0, 1] | None | [0, 1] |
//! | Sample Entropy | [0, ∞) | 1-exp(-x) | [0, 1) |
//! | Shannon Entropy | [0, log n] | /log(n) | [0, 1] |

use crate::adaptive::{
    optimal_bins_freedman_diaconis, optimal_embedding_dimension, optimal_sample_entropy_tolerance,
    MinimumSamples,
};
use std::collections::HashMap;

// ============================================================================
// Permutation Entropy
// ============================================================================

/// Calculate Permutation Entropy normalized to [0, 1].
///
/// Formula: H_perm = -Σ p_π * log(p_π) / log(m!)
///
/// Permutation entropy measures the complexity of a time series by analyzing
/// the distribution of ordinal patterns. It is scale-invariant (uses ranks).
///
/// # Arguments
///
/// * `prices` - Slice of price values
/// * `m` - Embedding dimension (pattern length), typically 3-6
///
/// # Returns
///
/// Permutation entropy in [0, 1], or NaN if input is too short.
///
/// # Example
///
/// ```rust
/// use trading_fitness_metrics::permutation_entropy;
///
/// // Need at least 60 samples for m=3 (10 × 3!)
/// let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();
/// let pe = permutation_entropy(&prices, 3);
/// assert!(pe.is_nan() || (pe >= 0.0 && pe <= 1.0));
/// ```
pub fn permutation_entropy(prices: &[f64], m: usize) -> f64 {
    let min_samples = MinimumSamples::permutation_entropy(m);
    if prices.len() < min_samples || m < 2 {
        return f64::NAN;
    }

    let n_patterns = prices.len() - m + 1;
    let mut pattern_counts: HashMap<Vec<usize>, usize> = HashMap::new();

    // Count ordinal patterns
    for i in 0..n_patterns {
        let window = &prices[i..i + m];
        let pattern = ordinal_pattern(window);
        *pattern_counts.entry(pattern).or_insert(0) += 1;
    }

    // Calculate Shannon entropy of pattern distribution
    let n = n_patterns as f64;
    let mut entropy = 0.0;

    for &count in pattern_counts.values() {
        if count > 0 {
            let p = count as f64 / n;
            entropy -= p * p.ln();
        }
    }

    // Normalize by log(m!) to get [0, 1]
    let factorial: usize = (1..=m).product();
    let max_entropy = (factorial as f64).ln();

    if max_entropy > 0.0 {
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Get the ordinal pattern (rank indices) for a window.
fn ordinal_pattern(window: &[f64]) -> Vec<usize> {
    let mut indexed: Vec<(usize, f64)> = window.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut pattern = vec![0; window.len()];
    for (rank, (original_idx, _)) in indexed.into_iter().enumerate() {
        pattern[original_idx] = rank;
    }
    pattern
}

/// Permutation entropy with adaptive embedding dimension.
///
/// Uses `optimal_embedding_dimension` to select m based on data length.
pub fn permutation_entropy_adaptive(prices: &[f64]) -> f64 {
    let m = optimal_embedding_dimension(prices, 7);
    permutation_entropy(prices, m)
}

// ============================================================================
// Sample Entropy
// ============================================================================

/// Calculate Sample Entropy normalized to [0, 1).
///
/// Formula: SampEn = -ln(A / B), normalized as 1 - exp(-SampEn)
///
/// Sample entropy measures the regularity/predictability of a time series.
/// Lower values indicate more regular patterns, higher values more randomness.
///
/// # Arguments
///
/// * `data` - Slice of values (typically returns)
/// * `m` - Pattern length (typically 1-2)
/// * `r` - Tolerance threshold (typically 0.1-0.3 × std)
///
/// # Returns
///
/// Sample entropy normalized to [0, 1), or NaN if input is too short.
///
/// # Example
///
/// ```rust
/// use trading_fitness_metrics::sample_entropy;
///
/// // Need at least 200 samples for m=2
/// let returns: Vec<f64> = (0..300).map(|i| (i as f64 * 0.1).sin() * 0.02).collect();
/// let se = sample_entropy(&returns, 2, 0.2);
/// assert!(se.is_nan() || (se >= 0.0 && se < 1.0));
/// ```
pub fn sample_entropy(data: &[f64], m: usize, r: f64) -> f64 {
    let min_samples = MinimumSamples::sample_entropy(m);
    if data.len() < min_samples || m < 1 || r <= 0.0 {
        return f64::NAN;
    }

    // Count template matches for length m and m+1
    let count_m = count_template_matches(data, m, r);
    let count_m1 = count_template_matches(data, m + 1, r);

    if count_m == 0 || count_m1 == 0 {
        return f64::NAN;
    }

    // Raw sample entropy
    let raw = -(count_m1 as f64 / count_m as f64).ln();

    // Normalize to [0, 1) using 1 - exp(-x)
    if raw.is_finite() && raw >= 0.0 {
        1.0 - (-raw).exp()
    } else {
        f64::NAN
    }
}

/// Count template matches within tolerance r.
fn count_template_matches(data: &[f64], template_len: usize, r: f64) -> usize {
    let n = data.len();
    if n < template_len + 1 {
        return 0;
    }

    let n_templates = n - template_len;
    let mut count = 0;

    for i in 0..n_templates {
        for j in (i + 1)..n_templates {
            if templates_match(data, i, j, template_len, r) {
                count += 2; // Count both (i,j) and (j,i)
            }
        }
    }

    count
}

/// Check if two templates match within tolerance.
fn templates_match(data: &[f64], i: usize, j: usize, len: usize, r: f64) -> bool {
    for k in 0..len {
        if (data[i + k] - data[j + k]).abs() > r {
            return false;
        }
    }
    true
}

/// Sample entropy with adaptive tolerance based on MAD.
///
/// Uses `optimal_sample_entropy_tolerance` for data-driven r selection.
pub fn sample_entropy_adaptive(data: &[f64], m: usize) -> f64 {
    let r = optimal_sample_entropy_tolerance(data);
    sample_entropy(data, m, r)
}

// ============================================================================
// Shannon Entropy
// ============================================================================

/// Calculate Shannon Entropy normalized to [0, 1].
///
/// Formula: H = -Σ p(x) * ln(p(x)) / ln(n_bins)
///
/// Shannon entropy measures the information content / uncertainty in the
/// distribution of values.
///
/// # Arguments
///
/// * `data` - Slice of values
/// * `n_bins` - Number of histogram bins
///
/// # Returns
///
/// Shannon entropy normalized to [0, 1], or NaN if input is invalid.
///
/// # Example
///
/// ```rust
/// use trading_fitness_metrics::shannon_entropy;
///
/// // Need at least 10 × n_bins samples
/// let returns: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin() * 0.02).collect();
/// let h = shannon_entropy(&returns, 10);
/// assert!(h.is_nan() || (h >= 0.0 && h <= 1.0));
/// ```
pub fn shannon_entropy(data: &[f64], n_bins: usize) -> f64 {
    let min_samples = MinimumSamples::shannon_entropy(n_bins);
    if data.len() < min_samples || n_bins < 2 {
        return f64::NAN;
    }

    // Find data range
    let min_val = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if !min_val.is_finite() || !max_val.is_finite() {
        return f64::NAN;
    }

    let range = max_val - min_val;
    if range < f64::EPSILON {
        // All values identical - zero entropy
        return 0.0;
    }

    // Build histogram
    let bin_width = range / n_bins as f64;
    let mut bins = vec![0usize; n_bins];

    for &x in data {
        let bin = ((x - min_val) / bin_width).floor() as usize;
        let bin = bin.min(n_bins - 1); // Handle edge case where x == max_val
        bins[bin] += 1;
    }

    // Calculate Shannon entropy
    let n = data.len() as f64;
    let mut entropy = 0.0;

    for &count in &bins {
        if count > 0 {
            let p = count as f64 / n;
            entropy -= p * p.ln();
        }
    }

    // Normalize by log(n_bins)
    let max_entropy = (n_bins as f64).ln();
    if max_entropy > 0.0 {
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Shannon entropy with adaptive bin count using Freedman-Diaconis rule.
pub fn shannon_entropy_adaptive(data: &[f64]) -> f64 {
    let n_bins = optimal_bins_freedman_diaconis(data);
    shannon_entropy(data, n_bins)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Permutation Entropy tests
    #[test]
    fn test_permutation_entropy_constant_series() {
        let prices = vec![100.0; 100];
        let pe = permutation_entropy(&prices, 3);
        assert!((pe - 0.0).abs() < 0.01 || pe.is_nan());
    }

    #[test]
    fn test_permutation_entropy_trending() {
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64).collect();
        let pe = permutation_entropy(&prices, 3);
        assert!(pe < 0.1);
    }

    #[test]
    fn test_permutation_entropy_bounded() {
        let prices: Vec<f64> = (0..200)
            .map(|i| if i % 2 == 0 { 100.0 } else { 101.0 })
            .collect();
        let pe = permutation_entropy(&prices, 3);
        assert!(pe >= 0.0 && pe <= 1.0);
    }

    #[test]
    fn test_permutation_entropy_short_input() {
        let prices = vec![100.0, 101.0];
        let pe = permutation_entropy(&prices, 3);
        assert!(pe.is_nan());
    }

    // Sample Entropy tests
    #[test]
    fn test_sample_entropy_sinusoidal() {
        let data: Vec<f64> = (0..500).map(|i| (i as f64 * 0.1).sin()).collect();
        let se = sample_entropy(&data, 2, 0.2);
        if se.is_finite() {
            assert!(se >= 0.0 && se < 1.0);
        }
    }

    #[test]
    fn test_sample_entropy_bounded() {
        let data: Vec<f64> = (0..500)
            .map(|i| ((i as f64 * 0.3).sin() + (i as f64 * 0.7).cos()) * 0.1)
            .collect();
        let se = sample_entropy(&data, 2, 0.2);
        if se.is_finite() {
            assert!(se >= 0.0 && se < 1.0);
        }
    }

    #[test]
    fn test_sample_entropy_short_input() {
        let data = vec![0.1, 0.2, 0.3];
        let se = sample_entropy(&data, 2, 0.1);
        assert!(se.is_nan());
    }

    // Shannon Entropy tests
    #[test]
    fn test_shannon_entropy_uniform() {
        let data: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
        let h = shannon_entropy(&data, 10);
        assert!(h > 0.9 && h <= 1.0);
    }

    #[test]
    fn test_shannon_entropy_constant() {
        let data = vec![50.0; 100];
        let h = shannon_entropy(&data, 10);
        assert!((h - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_shannon_entropy_bounded() {
        let data: Vec<f64> = (0..500)
            .map(|i| ((i as f64 * 0.1).sin() + 1.0) * 50.0)
            .collect();
        let h = shannon_entropy(&data, 20);
        assert!(h >= 0.0 && h <= 1.0);
    }

    #[test]
    fn test_shannon_entropy_short_input() {
        let data = vec![1.0, 2.0, 3.0];
        let h = shannon_entropy(&data, 20);
        assert!(h.is_nan());
    }

    // Adaptive function tests
    #[test]
    fn test_adaptive_functions_produce_valid_output() {
        let data: Vec<f64> = (0..500)
            .map(|i| 100.0 + (i as f64 * 0.05).sin() * 10.0)
            .collect();

        let pe = permutation_entropy_adaptive(&data);
        let se = sample_entropy_adaptive(&data, 2);
        let h = shannon_entropy_adaptive(&data);

        if pe.is_finite() {
            assert!(pe >= 0.0 && pe <= 1.0);
        }
        if se.is_finite() {
            assert!(se >= 0.0 && se < 1.0);
        }
        if h.is_finite() {
            assert!(h >= 0.0 && h <= 1.0);
        }
    }
}
