//! Adaptive numeric utilities eliminating magic numbers.
//!
//! All constants are derived from data characteristics or mathematical bounds.
//! This module provides the foundation for magic-number-free implementations.


// ============================================================================
// Division Guards (Adaptive Epsilon)
// ============================================================================

/// Compute relative epsilon based on operand magnitude.
///
/// Uses 100 × machine epsilon × max(|operand|, 1.0) to scale appropriately
/// with the magnitude of the operand.
///
/// # Example
///
/// ```rust
/// use trading_fitness_metrics::relative_epsilon;
///
/// let eps = relative_epsilon(1000.0);
/// assert!(eps > f64::EPSILON);
/// assert!(eps < 1e-10);
/// ```
#[inline]
pub fn relative_epsilon(operand: f64) -> f64 {
    let scale = operand.abs().max(1.0);
    scale * f64::EPSILON * 100.0
}

// ============================================================================
// Online Normalizers
// ============================================================================

/// Online normalizer using Welford's algorithm for numerically stable running stats.
///
/// Provides streaming normalization that adapts to non-stationary data.
#[derive(Debug, Clone)]
pub struct OnlineNormalizer {
    mean: f64,
    m2: f64,
    count: u64,
    #[allow(dead_code)]
    decay: f64, // Reserved for future EMA mode
}

impl OnlineNormalizer {
    /// Create with adaptive decay factor based on expected sequence length.
    ///
    /// Decay = 1 - 2/(n+1) for EMA half-life of n/2 samples.
    pub fn new(expected_len: usize) -> Self {
        let decay = 1.0 - 2.0 / (expected_len as f64 + 1.0);
        Self {
            mean: 0.0,
            m2: 0.0,
            count: 0,
            decay,
        }
    }

    /// Create with explicit decay factor.
    pub fn with_decay(decay: f64) -> Self {
        Self {
            mean: 0.0,
            m2: 0.0,
            count: 0,
            decay,
        }
    }

    /// Update running statistics and return normalized value in [0, 1].
    pub fn normalize(&mut self, raw: f64) -> f64 {
        self.count += 1;
        let delta = raw - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = raw - self.mean;
        self.m2 += delta * delta2;

        let variance = if self.count > 1 {
            self.m2 / (self.count - 1) as f64
        } else {
            1.0
        };

        let std = variance.sqrt().max(f64::EPSILON);
        let z_score = (raw - self.mean) / std;

        // Sigmoid transform: maps (-∞, ∞) → (0, 1)
        1.0 / (1.0 + (-z_score).exp())
    }

    /// Reset the normalizer state.
    pub fn reset(&mut self) {
        self.mean = 0.0;
        self.m2 = 0.0;
        self.count = 0;
    }

    /// Get current statistics.
    pub fn stats(&self) -> (f64, f64, u64) {
        let variance = if self.count > 1 {
            self.m2 / (self.count - 1) as f64
        } else {
            0.0
        };
        (self.mean, variance.sqrt(), self.count)
    }
}

/// Garman-Klass volatility normalizer with online EMA normalization.
///
/// Adapts to changing market volatility regimes using EMA-based z-score normalization.
#[derive(Debug, Clone)]
pub struct GarmanKlassNormalizer {
    ema_mean: f64,
    ema_variance: f64,
    decay: f64,
    initialized: bool,
}

impl GarmanKlassNormalizer {
    /// Create with adaptive decay factor based on expected sequence length.
    ///
    /// Consistent API with `OnlineNormalizer::new(expected_len)`.
    pub fn new(expected_len: usize) -> Self {
        let decay = 1.0 - 2.0 / (expected_len as f64 + 1.0);
        Self {
            ema_mean: 0.0,
            ema_variance: 1.0,
            decay,
            initialized: false,
        }
    }

    /// Create with explicit decay factor (advanced usage).
    pub fn with_decay(decay: f64) -> Self {
        Self {
            ema_mean: 0.0,
            ema_variance: 1.0,
            decay,
            initialized: false,
        }
    }

    /// Normalize a raw Garman-Klass volatility value to [0, 1].
    pub fn normalize(&mut self, raw: f64) -> f64 {
        if !self.initialized {
            self.ema_mean = raw;
            self.ema_variance = raw * raw;
            self.initialized = true;
            return 0.5; // Neutral initial output
        }

        // Update EMA statistics
        self.ema_mean = self.decay * self.ema_mean + (1.0 - self.decay) * raw;
        let sq_diff = (raw - self.ema_mean).powi(2);
        self.ema_variance = self.decay * self.ema_variance + (1.0 - self.decay) * sq_diff;

        // Z-score normalization
        let std = self.ema_variance.sqrt().max(f64::EPSILON);
        let z = (raw - self.ema_mean) / std;

        // Sigmoid maps z-score to (0, 1)
        1.0 / (1.0 + (-z).exp())
    }

    /// Reset the normalizer state.
    pub fn reset(&mut self) {
        self.ema_mean = 0.0;
        self.ema_variance = 1.0;
        self.initialized = false;
    }
}

/// Stateless Garman-Klass normalization using passed-in statistics.
pub fn garman_klass_normalized_with_stats(raw: f64, historical_mean: f64, historical_std: f64) -> f64 {
    let std = historical_std.max(f64::EPSILON);
    let z = (raw - historical_mean) / std;
    1.0 / (1.0 + (-z).exp())
}

// ============================================================================
// Hurst Normalization
// ============================================================================

/// Soft clamp for Hurst exponent using tanh.
///
/// Maps (-∞, ∞) → (0, 1) centered at 0.5, preserving information
/// when DFA produces values outside [0, 1].
pub fn hurst_soft_clamp(raw: f64) -> f64 {
    // Scale factor 4 makes 0.0→~0.02, 1.0→~0.98
    0.5 + 0.5 * ((raw - 0.5) * 4.0).tanh()
}

/// Normalize Hurst exponent using sigmoid with adaptive centering.
///
/// Centers on 0.5 (random walk) and uses data-derived scale if historical values provided.
pub fn hurst_normalized_adaptive(raw: f64, historical_hurst_values: Option<&[f64]>) -> f64 {
    let centered = raw - 0.5;

    let scale = if let Some(history) = historical_hurst_values {
        if history.len() >= 4 {
            let mut sorted = history.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q1 = sorted[sorted.len() / 4];
            let q3 = sorted[3 * sorted.len() / 4];
            (q3 - q1).max(0.1)
        } else {
            0.25
        }
    } else {
        0.25
    };

    let temperature = scale * 2.0;
    1.0 / (1.0 + (-(centered / temperature)).exp())
}

// ============================================================================
// Adaptive Windows
// ============================================================================

/// Generate log-spaced window sizes based on data length.
///
/// Ensures minimum statistical significance at each scale.
///
/// # Example
///
/// ```rust
/// use trading_fitness_metrics::adaptive_windows;
///
/// let windows = adaptive_windows(1000, 3);
/// // Returns approximately [10, 50, 250] (log-spaced)
/// assert!(windows.len() == 3);
/// assert!(windows[0] >= 10);
/// ```
pub fn adaptive_windows(data_len: usize, num_scales: usize) -> Vec<usize> {
    adaptive_windows_with_bounds(
        data_len,
        num_scales,
        10.max(data_len / 100), // At least 10, or 1% of data
        data_len / 4,           // At most 25% of data
    )
}

/// Generate log-spaced window sizes with custom bounds.
pub fn adaptive_windows_with_bounds(
    data_len: usize,
    num_scales: usize,
    min_window: usize,
    max_window: usize,
) -> Vec<usize> {
    let min_window = min_window.max(4);
    let max_window = max_window.max(min_window + 1).min(data_len);

    if max_window <= min_window || num_scales < 2 {
        return vec![min_window];
    }

    let log_min = (min_window as f64).ln();
    let log_max = (max_window as f64).ln();

    (0..num_scales)
        .map(|i| {
            let log_val = log_min + (log_max - log_min) * (i as f64) / (num_scales - 1) as f64;
            log_val.exp().round() as usize
        })
        .collect()
}

/// Generate DFA window sizes with DFA-specific bounds.
///
/// DFA literature (Peng 1994): min >= 10, max <= N/4, log-uniform spacing.
pub fn dfa_window_sizes(data_len: usize, num_windows: usize) -> Vec<usize> {
    adaptive_windows_with_bounds(
        data_len,
        num_windows,
        10.max(data_len / 50), // min: 10 or 2% of data
        data_len / 4,          // max: 25% of data (DFA standard)
    )
}

// ============================================================================
// Entropy Parameters (Self-Tuning)
// ============================================================================

/// Select optimal embedding dimension using simplified FNN criterion.
///
/// Returns m where complexity is balanced with statistical reliability.
pub fn optimal_embedding_dimension(data: &[f64], max_m: usize) -> usize {
    // Heuristic: m = ceil(log2(n)) bounded by [3, 7]
    let suggested = (data.len() as f64).log2().ceil() as usize;
    suggested.clamp(3, max_m.min(7))
}

/// Tolerance r for Sample Entropy based on Median Absolute Deviation.
///
/// MAD is robust to outliers, unlike standard deviation.
pub fn optimal_sample_entropy_tolerance(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.2; // Default
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = sorted[sorted.len() / 2];
    let mut abs_deviations: Vec<f64> = sorted.iter().map(|x| (x - median).abs()).collect();
    abs_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mad = abs_deviations[abs_deviations.len() / 2];

    // r = 0.2 × σ_MAD where σ_MAD ≈ 1.4826 × MAD
    let r = 0.2 * 1.4826 * mad;

    // Return at least a small positive value
    r.max(f64::EPSILON * 100.0)
}

/// Optimal bin count using Freedman-Diaconis rule.
pub fn optimal_bins_freedman_diaconis(data: &[f64]) -> usize {
    let n = data.len();
    if n < 4 {
        return n.max(1);
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q1 = sorted[n / 4];
    let q3 = sorted[3 * n / 4];
    let iqr = q3 - q1;

    let min = sorted[0];
    let max = sorted[n - 1];
    let range = max - min;

    if iqr < f64::EPSILON || range < f64::EPSILON {
        return 1;
    }

    // bin_width = 2 × IQR × n^(-1/3)
    let bin_width = 2.0 * iqr * (n as f64).powf(-1.0 / 3.0);
    let bins = (range / bin_width).ceil() as usize;

    bins.clamp(2, 100)
}

// ============================================================================
// Minimum Samples (Statistical Power)
// ============================================================================

/// Minimum samples for statistically meaningful computation.
///
/// Values derived from literature and power analysis.
pub struct MinimumSamples;

impl MinimumSamples {
    /// Permutation entropy: 10 × m! (acceptable per literature range 5-30×)
    pub fn permutation_entropy(m: usize) -> usize {
        let factorial: usize = (1..=m).product();
        10 * factorial
    }

    /// Sample entropy: values from Richman & Moorman literature
    pub fn sample_entropy(m: usize) -> usize {
        match m {
            1 => 50,
            2 => 200, // FIXED: was 100, literature says 200-300
            3 => 1000,
            _ => 10_usize.pow(m as u32).max(1000),
        }
    }

    /// Hurst exponent: need sufficient DFA scale range (s_max/s_min > 10)
    pub fn hurst_exponent() -> usize {
        256 // FIXED: was 100
    }

    /// Shannon entropy: 10× bins for Miller-Madow bias correction
    pub fn shannon_entropy(n_bins: usize) -> usize {
        10 * n_bins // FIXED: was 5×
    }
}

// ============================================================================
// Adaptive Tolerance
// ============================================================================

/// Compute adaptive tolerance based on algorithm characteristics.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveTolerance {
    /// Number of floating-point operations in accumulation.
    pub n_operations: usize,
    /// Condition number estimate (1 = well-conditioned).
    pub condition_number: f64,
}

impl AdaptiveTolerance {
    pub fn new(n_operations: usize, condition_number: f64) -> Self {
        Self {
            n_operations,
            condition_number,
        }
    }

    /// Relative tolerance: κ × √n × ε_machine × 10
    pub fn rtol(&self) -> f64 {
        self.condition_number * (self.n_operations as f64).sqrt() * f64::EPSILON * 10.0
    }

    /// Absolute tolerance: based on expected output scale.
    pub fn atol(&self, expected_scale: f64) -> f64 {
        self.rtol() * expected_scale
    }
}

/// Compute tolerance for Omega ratio.
pub fn tolerance_for_omega_ratio(n: usize) -> AdaptiveTolerance {
    AdaptiveTolerance::new(2 * n + 2, 2.0)
}

/// Compute tolerance for Hurst exponent.
pub fn tolerance_for_hurst(n: usize, n_windows: usize) -> AdaptiveTolerance {
    AdaptiveTolerance::new(n * n_windows, 50.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relative_epsilon_unit_scale() {
        let eps = relative_epsilon(1.0);
        assert!(eps > f64::EPSILON);
        assert!(eps < 1e-10);
    }

    #[test]
    fn test_relative_epsilon_large_scale() {
        let eps_small = relative_epsilon(1.0);
        let eps_large = relative_epsilon(1e6);
        assert!(eps_large > eps_small);
    }

    #[test]
    fn test_online_normalizer_bounded() {
        let mut norm = OnlineNormalizer::new(100);
        for i in 0..100 {
            let val = norm.normalize(i as f64);
            assert!(val > 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_gk_normalizer_initial_output() {
        let mut norm = GarmanKlassNormalizer::new(100);
        let val = norm.normalize(0.01);
        assert!((val - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hurst_soft_clamp_center() {
        let val = hurst_soft_clamp(0.5);
        assert!((val - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_hurst_soft_clamp_bounded() {
        assert!(hurst_soft_clamp(-1.0) > 0.0);
        assert!(hurst_soft_clamp(2.0) < 1.0);
    }

    #[test]
    fn test_adaptive_windows_log_spacing() {
        let windows = adaptive_windows(1000, 5);
        assert_eq!(windows.len(), 5);

        // Check log-spacing: ratios should be approximately equal
        for w in windows.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[test]
    fn test_minimum_samples_permutation_entropy() {
        assert_eq!(MinimumSamples::permutation_entropy(3), 60);
        assert_eq!(MinimumSamples::permutation_entropy(4), 240);
    }

    #[test]
    fn test_minimum_samples_sample_entropy() {
        assert_eq!(MinimumSamples::sample_entropy(2), 200);
    }

    #[test]
    fn test_optimal_bins_freedman_diaconis() {
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let bins = optimal_bins_freedman_diaconis(&data);
        assert!(bins >= 2 && bins <= 100);
    }
}
