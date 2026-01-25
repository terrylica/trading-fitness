//! Python bindings for trading-fitness-metrics using PyO3.
//!
//! This module provides zero-copy NumPy integration for all 9 price-only metrics,
//! ITH analysis functions, and adaptive utilities.

#![cfg(feature = "python")]

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::cell::RefCell;

// ============================================================================
// Entropy Metrics
// ============================================================================

/// Compute permutation entropy of a price series.
///
/// Permutation entropy measures the complexity of a time series based on ordinal
/// patterns. Output is bounded [0, 1] where 0 indicates a completely deterministic
/// series and 1 indicates maximum complexity.
///
/// Args:
///     prices: NumPy array of prices (float64)
///     m: Embedding dimension (default: 3)
///
/// Returns:
///     Permutation entropy value in [0, 1]
#[pyfunction]
#[pyo3(signature = (prices, m=3))]
fn permutation_entropy(prices: PyReadonlyArray1<f64>, m: usize) -> PyResult<f64> {
    let slice = prices
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if slice.is_empty() {
        return Err(PyValueError::new_err("prices array cannot be empty"));
    }
    if m < 2 {
        return Err(PyValueError::new_err("embedding dimension m must be >= 2"));
    }
    Ok(crate::permutation_entropy(slice, m))
}

/// Compute sample entropy of a data series.
///
/// Sample entropy measures the complexity and regularity of a time series.
/// Output is normalized to [0, 1) using the transform 1 - exp(-SampEn).
///
/// Args:
///     data: NumPy array of data points (float64)
///     m: Embedding dimension (default: 2)
///     r: Tolerance (default: computed from data using MAD)
///
/// Returns:
///     Normalized sample entropy value in [0, 1)
#[pyfunction]
#[pyo3(signature = (data, m=2, r=None))]
fn sample_entropy(data: PyReadonlyArray1<f64>, m: usize, r: Option<f64>) -> PyResult<f64> {
    let slice = data
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if slice.is_empty() {
        return Err(PyValueError::new_err("data array cannot be empty"));
    }
    let tolerance = r.unwrap_or_else(|| crate::optimal_sample_entropy_tolerance(slice));
    Ok(crate::sample_entropy(slice, m, tolerance))
}

/// Compute Shannon entropy of a data series.
///
/// Shannon entropy measures the information content of a distribution.
/// Output is normalized to [0, 1] by dividing by log(n_bins).
///
/// Args:
///     data: NumPy array of data points (float64)
///     n_bins: Number of histogram bins (default: computed using Freedman-Diaconis)
///
/// Returns:
///     Normalized Shannon entropy value in [0, 1]
#[pyfunction]
#[pyo3(signature = (data, n_bins=None))]
fn shannon_entropy(data: PyReadonlyArray1<f64>, n_bins: Option<usize>) -> PyResult<f64> {
    let slice = data
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if slice.is_empty() {
        return Err(PyValueError::new_err("data array cannot be empty"));
    }
    let bins = n_bins.unwrap_or_else(|| crate::optimal_bins_freedman_diaconis(slice));
    Ok(crate::shannon_entropy(slice, bins))
}

// ============================================================================
// Risk Metrics
// ============================================================================

/// Compute Omega ratio of returns.
///
/// Omega ratio is the probability-weighted ratio of gains over losses.
/// Output is normalized to [0, 1) using the transform Omega / (1 + Omega).
///
/// Args:
///     returns: NumPy array of returns (float64)
///     threshold: Threshold for gains/losses (default: 0.0)
///
/// Returns:
///     Normalized Omega ratio value in [0, 1)
#[pyfunction]
#[pyo3(signature = (returns, threshold=0.0))]
fn omega_ratio(returns: PyReadonlyArray1<f64>, threshold: f64) -> PyResult<f64> {
    let slice = returns
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if slice.is_empty() {
        return Err(PyValueError::new_err("returns array cannot be empty"));
    }
    Ok(crate::omega_ratio(slice, threshold))
}

/// Compute Ulcer Index of a price series.
///
/// Ulcer Index measures drawdown risk using quadratic penalty for deep drawdowns.
/// Output is bounded [0, 1].
///
/// Args:
///     prices: NumPy array of prices (float64)
///
/// Returns:
///     Ulcer Index value in [0, 1]
#[pyfunction]
fn ulcer_index(prices: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let slice = prices
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if slice.is_empty() {
        return Err(PyValueError::new_err("prices array cannot be empty"));
    }
    Ok(crate::ulcer_index(slice))
}

/// Compute Garman-Klass volatility for a single OHLC bar.
///
/// This is the stateless version using tanh normalization.
/// Output is bounded [0, 1).
///
/// Args:
///     open: Open price
///     high: High price
///     low: Low price
///     close: Close price
///
/// Returns:
///     Normalized Garman-Klass volatility in [0, 1)
#[pyfunction]
fn garman_klass_volatility(open: f64, high: f64, low: f64, close: f64) -> PyResult<f64> {
    if high < low {
        return Err(PyValueError::new_err("high must be >= low"));
    }
    if high < open || high < close {
        return Err(PyValueError::new_err("high must be >= open and close"));
    }
    if low > open || low > close {
        return Err(PyValueError::new_err("low must be <= open and close"));
    }
    Ok(crate::garman_klass_volatility(open, high, low, close))
}

/// Compute Kaufman Efficiency Ratio of a price series.
///
/// Efficiency ratio measures directional movement vs noise.
/// Output is bounded [0, 1] where 1 = perfect efficiency.
///
/// Args:
///     prices: NumPy array of prices (float64)
///
/// Returns:
///     Efficiency ratio value in [0, 1]
#[pyfunction]
fn kaufman_efficiency_ratio(prices: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let slice = prices
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if slice.len() < 2 {
        return Err(PyValueError::new_err(
            "prices array must have at least 2 elements",
        ));
    }
    Ok(crate::kaufman_efficiency_ratio(slice))
}

// ============================================================================
// Fractal Metrics
// ============================================================================

/// Compute Hurst exponent of a price series using DFA.
///
/// Hurst exponent indicates market regime:
/// - H < 0.5: Mean-reverting
/// - H = 0.5: Random walk
/// - H > 0.5: Trending/momentum
///
/// Output is soft-clamped to [0, 1].
///
/// Args:
///     prices: NumPy array of prices (float64)
///
/// Returns:
///     Hurst exponent value in [0, 1]
#[pyfunction]
fn hurst_exponent(prices: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let slice = prices
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if slice.len() < 256 {
        return Err(PyValueError::new_err(
            "prices array must have at least 256 elements for Hurst exponent",
        ));
    }
    Ok(crate::hurst_exponent(slice))
}

/// Compute fractal dimension of a price series using Higuchi's method.
///
/// Fractal dimension indicates market structure:
/// - D ≈ 1: Smooth trend
/// - D ≈ 1.5: Typical turbulent market
/// - D ≈ 2: Highly fragmented
///
/// Output is normalized to [0, 1] using D - 1.
///
/// Args:
///     prices: NumPy array of prices (float64)
///     k_max: Maximum scale parameter (default: 10)
///
/// Returns:
///     Normalized fractal dimension value in [0, 1]
#[pyfunction]
#[pyo3(signature = (prices, k_max=10))]
fn fractal_dimension(prices: PyReadonlyArray1<f64>, k_max: usize) -> PyResult<f64> {
    let slice = prices
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if slice.len() < k_max * 4 {
        return Err(PyValueError::new_err(format!(
            "prices array must have at least {} elements for k_max={}",
            k_max * 4,
            k_max
        )));
    }
    Ok(crate::fractal_dimension(slice, k_max))
}

// ============================================================================
// NAV & Utility Functions
// ============================================================================

/// Build NAV (Net Asset Value) series from close prices.
///
/// Converts close prices to returns, clamps returns to -0.99 minimum,
/// and computes cumulative product.
///
/// Args:
///     closes: NumPy array of close prices (float64)
///
/// Returns:
///     NumPy array of NAV values
#[pyfunction]
fn build_nav_from_closes<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = closes
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if slice.is_empty() {
        return Err(PyValueError::new_err("closes array cannot be empty"));
    }
    let nav = crate::build_nav_from_closes(slice);
    Ok(PyArray1::from_vec(py, nav))
}

/// Generate log-spaced window sizes for multi-scale analysis.
///
/// Args:
///     data_len: Length of data series
///     num_scales: Number of window sizes to generate
///
/// Returns:
///     List of window sizes
#[pyfunction]
fn adaptive_windows(data_len: usize, num_scales: usize) -> Vec<usize> {
    crate::adaptive_windows(data_len, num_scales)
}

/// Compute optimal number of histogram bins using Freedman-Diaconis rule.
///
/// Args:
///     data: NumPy array of data points (float64)
///
/// Returns:
///     Optimal number of bins
#[pyfunction]
fn optimal_bins_freedman_diaconis(data: PyReadonlyArray1<f64>) -> PyResult<usize> {
    let slice = data
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if slice.len() < 4 {
        return Err(PyValueError::new_err(
            "data array must have at least 4 elements",
        ));
    }
    Ok(crate::optimal_bins_freedman_diaconis(slice))
}

/// Compute optimal embedding dimension for permutation entropy.
///
/// Args:
///     data: NumPy array of data points (float64)
///     max_m: Maximum embedding dimension to consider
///
/// Returns:
///     Optimal embedding dimension
#[pyfunction]
fn optimal_embedding_dimension(data: PyReadonlyArray1<f64>, max_m: usize) -> PyResult<usize> {
    let slice = data
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if slice.is_empty() {
        return Err(PyValueError::new_err("data array cannot be empty"));
    }
    Ok(crate::optimal_embedding_dimension(slice, max_m))
}

/// Compute optimal tolerance for sample entropy using MAD.
///
/// Args:
///     data: NumPy array of data points (float64)
///
/// Returns:
///     Optimal tolerance value
#[pyfunction]
fn optimal_sample_entropy_tolerance(data: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let slice = data
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if slice.is_empty() {
        return Err(PyValueError::new_err("data array cannot be empty"));
    }
    Ok(crate::optimal_sample_entropy_tolerance(slice))
}

/// Compute relative epsilon for adaptive division guards.
///
/// Args:
///     operand: The operand magnitude
///
/// Returns:
///     Adaptive epsilon value
#[pyfunction]
fn relative_epsilon(operand: f64) -> f64 {
    crate::relative_epsilon(operand)
}

/// Compute optimal TMAEG threshold based on data volatility.
///
/// Uses MAD-based volatility estimation with sqrt(lookback) scaling to produce
/// sensible ITH epoch density (typically 10-40% of bars) regardless of bar type.
///
/// This is automatically called by `compute_rolling_ith()`, but exposed here
/// for inspection and debugging purposes.
///
/// Args:
///     nav: NumPy array of NAV values (float64)
///     lookback: Lookback window size for ITH computation
///
/// Returns:
///     Optimal TMAEG threshold (clamped to [0.001, 0.50])
///
/// Example:
///     >>> import numpy as np
///     >>> from trading_fitness_metrics import optimal_tmaeg
///     >>> nav = np.cumprod(1 + np.random.randn(500) * 0.01)
///     >>> tmaeg = optimal_tmaeg(nav, lookback=100)
///     >>> print(f"Auto-TMAEG: {tmaeg:.4f}")
#[pyfunction]
fn optimal_tmaeg(nav: PyReadonlyArray1<f64>, lookback: usize) -> PyResult<f64> {
    let slice = nav
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if slice.is_empty() {
        return Err(PyValueError::new_err("nav array cannot be empty"));
    }
    Ok(crate::optimal_tmaeg(slice, lookback))
}

// ============================================================================
// ITH Analysis
// ============================================================================

/// Python wrapper for BullIthResult.
#[pyclass(name = "BullIthResult")]
#[derive(Clone)]
pub struct PyBullIthResult {
    #[pyo3(get)]
    pub num_of_epochs: usize,
    #[pyo3(get)]
    pub max_drawdown: f64,
    #[pyo3(get)]
    pub intervals_cv: f64,
    inner: crate::BullIthResult,
}

#[pymethods]
impl PyBullIthResult {
    /// Get the excess gains array.
    fn excess_gains<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.excess_gains.clone())
    }

    /// Get the excess losses array.
    fn excess_losses<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.excess_losses.clone())
    }

    /// Get the epochs boolean array.
    fn epochs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_vec(py, self.inner.epochs.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "BullIthResult(num_of_epochs={}, max_drawdown={:.4}, intervals_cv={:.4})",
            self.num_of_epochs, self.max_drawdown, self.intervals_cv
        )
    }
}

/// Python wrapper for BearIthResult.
#[pyclass(name = "BearIthResult")]
#[derive(Clone)]
pub struct PyBearIthResult {
    #[pyo3(get)]
    pub num_of_epochs: usize,
    #[pyo3(get)]
    pub max_runup: f64,
    #[pyo3(get)]
    pub intervals_cv: f64,
    inner: crate::BearIthResult,
}

#[pymethods]
impl PyBearIthResult {
    /// Get the excess gains array.
    fn excess_gains<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.excess_gains.clone())
    }

    /// Get the excess losses array.
    fn excess_losses<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.excess_losses.clone())
    }

    /// Get the epochs boolean array.
    fn epochs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_vec(py, self.inner.epochs.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "BearIthResult(num_of_epochs={}, max_runup={:.4}, intervals_cv={:.4})",
            self.num_of_epochs, self.max_runup, self.intervals_cv
        )
    }
}

/// Perform Bull ITH (long position) analysis.
///
/// Analyzes a NAV series for bull (long) trading profitability using the
/// Investment Time Horizon methodology.
///
/// Args:
///     nav: NumPy array of NAV values (float64)
///     tmaeg: Target Maximum Acceptable Excess Gain threshold
///
/// Returns:
///     BullIthResult with epochs, excess gains/losses, and statistics
#[pyfunction]
fn bull_ith(nav: PyReadonlyArray1<f64>, tmaeg: f64) -> PyResult<PyBullIthResult> {
    let slice = nav
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if slice.len() < 2 {
        return Err(PyValueError::new_err(
            "nav array must have at least 2 elements",
        ));
    }
    if tmaeg <= 0.0 {
        return Err(PyValueError::new_err("tmaeg must be positive"));
    }
    let result = crate::bull_ith(slice, tmaeg);
    Ok(PyBullIthResult {
        num_of_epochs: result.num_of_epochs,
        max_drawdown: result.max_drawdown,
        intervals_cv: result.intervals_cv,
        inner: result,
    })
}

/// Perform Bear ITH (short position) analysis.
///
/// Analyzes a NAV series for bear (short) trading profitability using the
/// Investment Time Horizon methodology.
///
/// Args:
///     nav: NumPy array of NAV values (float64)
///     tmaeg: Target Maximum Acceptable Excess Gain threshold
///
/// Returns:
///     BearIthResult with epochs, excess gains/losses, and statistics
#[pyfunction]
fn bear_ith(nav: PyReadonlyArray1<f64>, tmaeg: f64) -> PyResult<PyBearIthResult> {
    let slice = nav
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if slice.len() < 2 {
        return Err(PyValueError::new_err(
            "nav array must have at least 2 elements",
        ));
    }
    if tmaeg <= 0.0 {
        return Err(PyValueError::new_err("tmaeg must be positive"));
    }
    let result = crate::bear_ith(slice, tmaeg);
    Ok(PyBearIthResult {
        num_of_epochs: result.num_of_epochs,
        max_runup: result.max_runup,
        intervals_cv: result.intervals_cv,
        inner: result,
    })
}

// ============================================================================
// Rolling ITH Features (Time-Agnostic, Bounded [0, 1])
// ============================================================================

/// Python wrapper for RollingIthFeatures.
///
/// All 8 feature arrays are bounded [0, 1] for LSTM consumption.
/// First `lookback - 1` values are NaN (insufficient data).
#[pyclass(name = "RollingIthFeatures")]
#[derive(Clone)]
pub struct PyRollingIthFeatures {
    inner: crate::ith_rolling::RollingIthFeatures,
}

#[pymethods]
impl PyRollingIthFeatures {
    /// Get bull epoch density array (bounded [0, 1]).
    #[getter]
    fn bull_epoch_density<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.bull_epoch_density.clone())
    }

    /// Get bear epoch density array (bounded [0, 1]).
    #[getter]
    fn bear_epoch_density<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.bear_epoch_density.clone())
    }

    /// Get bull excess gain array (bounded [0, 1]).
    #[getter]
    fn bull_excess_gain<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.bull_excess_gain.clone())
    }

    /// Get bear excess gain array (bounded [0, 1]).
    #[getter]
    fn bear_excess_gain<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.bear_excess_gain.clone())
    }

    /// Get bull coefficient of variation array (bounded [0, 1]).
    #[getter]
    fn bull_cv<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.bull_cv.clone())
    }

    /// Get bear coefficient of variation array (bounded [0, 1]).
    #[getter]
    fn bear_cv<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.bear_cv.clone())
    }

    /// Get max drawdown array (bounded [0, 1]).
    #[getter]
    fn max_drawdown<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.max_drawdown.clone())
    }

    /// Get max runup array (bounded [0, 1]).
    #[getter]
    fn max_runup<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.max_runup.clone())
    }

    /// Get the length of the feature arrays.
    fn __len__(&self) -> usize {
        self.inner.bull_epoch_density.len()
    }

    fn __repr__(&self) -> String {
        let n = self.inner.bull_epoch_density.len();
        format!("RollingIthFeatures(len={}, features=8)", n)
    }
}

/// Compute rolling ITH features over lookback windows.
///
/// This function computes Bull and Bear ITH metrics over sliding windows
/// of the NAV series, normalizing all outputs to [0, 1] for LSTM consumption.
///
/// The TMAEG threshold is automatically calculated based on the data's volatility
/// using MAD-based estimation. This ensures sensible epoch density regardless of
/// the bar type (range bars, tick bars, time bars) or instrument volatility.
///
/// All 8 feature arrays are time-agnostic (bar-based, not time-based) and
/// suitable for use with range bars, tick bars, or any bar type.
///
/// Args:
///     nav: NumPy array of NAV values (float64)
///     lookback: Number of bars to look back for each computation
///
/// Returns:
///     RollingIthFeatures with 8 bounded [0, 1] feature arrays:
///     - bull_epoch_density: Normalized bull epoch count
///     - bear_epoch_density: Normalized bear epoch count
///     - bull_excess_gain: Normalized sum of bull excess gains
///     - bear_excess_gain: Normalized sum of bear excess gains
///     - bull_cv: Normalized bull intervals coefficient of variation
///     - bear_cv: Normalized bear intervals coefficient of variation
///     - max_drawdown: Maximum drawdown in window
///     - max_runup: Maximum runup in window
///
/// Note:
///     First `lookback - 1` values are NaN (insufficient data for window).
///
/// Example:
///     >>> import numpy as np
///     >>> from trading_fitness_metrics import compute_rolling_ith
///     >>> nav = np.cumprod(1 + np.random.randn(500) * 0.01)
///     >>> features = compute_rolling_ith(nav, lookback=100)
///     >>> features.bull_epoch_density[:99]  # First 99 are NaN
///     >>> valid = features.bull_epoch_density[99:]  # All in [0, 1]
#[pyfunction]
#[pyo3(signature = (nav, lookback))]
fn compute_rolling_ith(
    nav: PyReadonlyArray1<f64>,
    lookback: usize,
) -> PyResult<PyRollingIthFeatures> {
    let slice = nav
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    if slice.is_empty() {
        return Err(PyValueError::new_err("nav array cannot be empty"));
    }
    if lookback == 0 {
        return Err(PyValueError::new_err("lookback must be positive"));
    }
    if lookback > slice.len() {
        return Err(PyValueError::new_err(
            "lookback cannot exceed nav array length",
        ));
    }

    let features = crate::ith_rolling::compute_rolling_ith(slice, lookback);
    Ok(PyRollingIthFeatures { inner: features })
}

// ============================================================================
// Stateful Normalizers
// ============================================================================

/// EMA-based Garman-Klass volatility normalizer.
///
/// Maintains state for streaming volatility normalization using EMA-based
/// z-score computation with sigmoid transform.
///
/// Args:
///     expected_len: Expected sequence length for decay factor calculation
#[pyclass(name = "GarmanKlassNormalizer", unsendable)]
pub struct PyGarmanKlassNormalizer {
    inner: RefCell<crate::GarmanKlassNormalizer>,
}

#[pymethods]
impl PyGarmanKlassNormalizer {
    #[new]
    fn new(expected_len: usize) -> Self {
        Self {
            inner: RefCell::new(crate::GarmanKlassNormalizer::new(expected_len)),
        }
    }

    /// Normalize a raw volatility value.
    ///
    /// Args:
    ///     raw: Raw Garman-Klass volatility value
    ///
    /// Returns:
    ///     Normalized value in (0, 1)
    fn normalize(&self, raw: f64) -> f64 {
        self.inner.borrow_mut().normalize(raw)
    }

    /// Reset the normalizer state.
    fn reset(&self) {
        self.inner.borrow_mut().reset();
    }

    fn __repr__(&self) -> String {
        "GarmanKlassNormalizer()".to_string()
    }
}

/// Welford-based online normalizer.
///
/// Maintains running statistics for online z-score normalization
/// with sigmoid transform.
///
/// Args:
///     expected_len: Expected sequence length for decay factor calculation
#[pyclass(name = "OnlineNormalizer", unsendable)]
pub struct PyOnlineNormalizer {
    inner: RefCell<crate::OnlineNormalizer>,
}

#[pymethods]
impl PyOnlineNormalizer {
    #[new]
    fn new(expected_len: usize) -> Self {
        Self {
            inner: RefCell::new(crate::OnlineNormalizer::new(expected_len)),
        }
    }

    /// Normalize a raw value.
    ///
    /// Args:
    ///     raw: Raw input value
    ///
    /// Returns:
    ///     Normalized value in (0, 1)
    fn normalize(&self, raw: f64) -> f64 {
        self.inner.borrow_mut().normalize(raw)
    }

    /// Reset the normalizer state.
    fn reset(&self) {
        self.inner.borrow_mut().reset();
    }

    fn __repr__(&self) -> String {
        "OnlineNormalizer()".to_string()
    }
}

// ============================================================================
// Batch API
// ============================================================================

/// Python wrapper for MetricsResult.
#[pyclass(name = "MetricsResult")]
#[derive(Clone)]
pub struct PyMetricsResult {
    #[pyo3(get)]
    pub permutation_entropy: f64,
    #[pyo3(get)]
    pub sample_entropy: f64,
    #[pyo3(get)]
    pub shannon_entropy: f64,
    #[pyo3(get)]
    pub omega_ratio: f64,
    #[pyo3(get)]
    pub ulcer_index: f64,
    #[pyo3(get)]
    pub garman_klass_vol: f64,
    #[pyo3(get)]
    pub kaufman_er: f64,
    #[pyo3(get)]
    pub hurst_exponent: f64,
    #[pyo3(get)]
    pub fractal_dimension: f64,
}

#[pymethods]
impl PyMetricsResult {
    /// Check if all metrics are bounded [0, 1].
    fn all_bounded(&self) -> bool {
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
    fn has_nan(&self) -> bool {
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

    fn __repr__(&self) -> String {
        format!(
            "MetricsResult(pe={:.4}, se={:.4}, sh={:.4}, omega={:.4}, ulcer={:.4}, gk={:.4}, er={:.4}, hurst={:.4}, fd={:.4})",
            self.permutation_entropy,
            self.sample_entropy,
            self.shannon_entropy,
            self.omega_ratio,
            self.ulcer_index,
            self.garman_klass_vol,
            self.kaufman_er,
            self.hurst_exponent,
            self.fractal_dimension
        )
    }
}

/// Compute all 9 metrics in a single call.
///
/// This batch API minimizes Python-Rust boundary crossings for better performance.
///
/// Args:
///     prices: NumPy array of prices (float64)
///     returns: NumPy array of returns (float64)
///     ohlc: Optional tuple of (open, high, low, close) for Garman-Klass
///
/// Returns:
///     MetricsResult with all 9 metrics
#[pyfunction]
#[pyo3(signature = (prices, returns, ohlc=None))]
fn compute_all_metrics(
    prices: PyReadonlyArray1<f64>,
    returns: PyReadonlyArray1<f64>,
    ohlc: Option<(f64, f64, f64, f64)>,
) -> PyResult<PyMetricsResult> {
    let price_slice = prices
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let return_slice = returns
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    if price_slice.len() < 256 {
        return Err(PyValueError::new_err(
            "prices array must have at least 256 elements for all metrics",
        ));
    }
    if return_slice.is_empty() {
        return Err(PyValueError::new_err("returns array cannot be empty"));
    }

    // Compute all metrics
    let pe = crate::permutation_entropy(price_slice, 3);
    let r_tol = crate::optimal_sample_entropy_tolerance(return_slice);
    let se = crate::sample_entropy(return_slice, 2, r_tol);
    let bins = crate::optimal_bins_freedman_diaconis(return_slice);
    let sh = crate::shannon_entropy(return_slice, bins);
    let omega = crate::omega_ratio(return_slice, 0.0);
    let ulcer = crate::ulcer_index(price_slice);
    let er = crate::kaufman_efficiency_ratio(price_slice);
    let hurst = crate::hurst_exponent(price_slice);
    let fd = crate::fractal_dimension(price_slice, 10);

    // Garman-Klass requires OHLC
    let gk = if let Some((o, h, l, c)) = ohlc {
        if h < l || h < o || h < c || l > o || l > c {
            return Err(PyValueError::new_err("invalid OHLC values"));
        }
        crate::garman_klass_volatility(o, h, l, c)
    } else {
        f64::NAN // Not available without OHLC
    };

    Ok(PyMetricsResult {
        permutation_entropy: pe,
        sample_entropy: se,
        shannon_entropy: sh,
        omega_ratio: omega,
        ulcer_index: ulcer,
        garman_klass_vol: gk,
        kaufman_er: er,
        hurst_exponent: hurst,
        fractal_dimension: fd,
    })
}

// ============================================================================
// Multi-Scale ITH Features (Arrow-Native Integration)
// ============================================================================

/// Python wrapper for MultiscaleConfig.
#[pyclass(name = "MultiscaleIthConfig")]
#[derive(Clone)]
pub struct PyMultiscaleConfig {
    inner: crate::ith_multiscale::MultiscaleConfig,
}

#[pymethods]
impl PyMultiscaleConfig {
    /// Create a new multi-scale ITH configuration.
    ///
    /// Args:
    ///     threshold_dbps: Range bar threshold in dbps (for column naming only)
    ///     lookbacks: List of lookback window sizes
    #[new]
    #[pyo3(signature = (threshold_dbps=250, lookbacks=None))]
    fn new(threshold_dbps: u32, lookbacks: Option<Vec<usize>>) -> Self {
        let lookbacks = lookbacks.unwrap_or_else(|| {
            vec![20, 50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000]
        });
        Self {
            inner: crate::ith_multiscale::MultiscaleConfig::new(threshold_dbps, lookbacks),
        }
    }

    /// Get the threshold in dbps.
    #[getter]
    fn threshold_dbps(&self) -> u32 {
        self.inner.threshold_dbps
    }

    /// Get the list of lookback windows.
    #[getter]
    fn lookbacks(&self) -> Vec<usize> {
        self.inner.lookbacks.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "MultiscaleIthConfig(threshold_dbps={}, lookbacks={:?})",
            self.inner.threshold_dbps, self.inner.lookbacks
        )
    }
}

/// Python wrapper for MultiscaleIthFeatures.
///
/// Provides both dictionary-style access and Arrow RecordBatch conversion
/// for zero-copy integration with Polars.
#[pyclass(name = "MultiscaleIthFeatures")]
#[derive(Clone)]
pub struct PyMultiscaleIthFeatures {
    inner: crate::ith_multiscale::MultiscaleIthFeatures,
}

#[pymethods]
impl PyMultiscaleIthFeatures {
    /// Convert to Arrow RecordBatch (zero-copy via PyCapsule Interface).
    ///
    /// This method provides efficient zero-copy integration with Polars:
    /// ```python
    /// import polars as pl
    /// from trading_fitness_metrics import compute_multiscale_ith
    ///
    /// features = compute_multiscale_ith(nav, config)
    /// df = pl.from_arrow(features.to_arrow())  # Zero-copy!
    /// ```
    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use arrow_array::{Float64Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};
        use pyo3_arrow::PyRecordBatch;
        use std::sync::Arc;

        // Sort column names for deterministic order
        let mut column_names: Vec<_> = self.inner.columns.keys().cloned().collect();
        column_names.sort();

        // Build Arrow schema
        let fields: Vec<Field> = column_names
            .iter()
            .map(|name| Field::new(name, DataType::Float64, true))
            .collect();
        let schema = Arc::new(Schema::new(fields));

        // Convert Vec<f64> to Arrow Float64Array (copies data once for Arrow ownership)
        let arrays: Vec<Arc<dyn arrow_array::Array>> = column_names
            .iter()
            .map(|name| {
                let values = self.inner.columns.get(name).unwrap();
                Arc::new(Float64Array::from(values.clone())) as Arc<dyn arrow_array::Array>
            })
            .collect();

        let batch = RecordBatch::try_new(schema, arrays)
            .map_err(|e| PyValueError::new_err(format!("Failed to create RecordBatch: {}", e)))?;

        // Wrap in PyRecordBatch for zero-copy transfer to Python
        let py_batch = PyRecordBatch::new(batch);
        Ok(py_batch.into_pyobject(py)?.into_any().unbind())
    }

    /// Get a feature array by column name.
    fn get<'py>(&self, py: Python<'py>, column_name: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.inner
            .columns
            .get(column_name)
            .map(|v| PyArray1::from_vec(py, v.clone()))
            .ok_or_else(|| PyValueError::new_err(format!("Column '{}' not found", column_name)))
    }

    /// Get all column names.
    fn column_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.inner.columns.keys().cloned().collect();
        names.sort();
        names
    }

    /// Get the number of data points.
    #[getter]
    fn n_points(&self) -> usize {
        self.inner.n_points
    }

    /// Get the number of feature columns.
    #[getter]
    fn n_features(&self) -> usize {
        self.inner.n_features
    }

    /// Check if all features are bounded [0, 1].
    fn all_bounded(&self) -> bool {
        self.inner.all_bounded()
    }

    /// Get the configuration used for computation.
    #[getter]
    fn config(&self) -> PyMultiscaleConfig {
        PyMultiscaleConfig {
            inner: self.inner.config.clone(),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.n_points
    }

    fn __repr__(&self) -> String {
        format!(
            "MultiscaleIthFeatures(n_points={}, n_features={})",
            self.inner.n_points, self.inner.n_features
        )
    }
}

/// Compute multi-scale ITH features across multiple lookback windows.
///
/// This function computes rolling ITH features at multiple lookback scales,
/// producing columnar outputs suitable for LSTM feature engineering.
///
/// All features are bounded [0, 1] and follow the naming convention:
/// `ith_rb{threshold}_lb{lookback}_{feature}`
///
/// Args:
///     nav: NumPy array of NAV values (float64)
///     config: MultiscaleIthConfig with threshold and lookbacks
///
/// Returns:
///     MultiscaleIthFeatures with zero-copy Arrow conversion via `to_arrow()`.
///
/// Example:
///     >>> import polars as pl
///     >>> from trading_fitness_metrics import compute_multiscale_ith, MultiscaleIthConfig
///     >>>
///     >>> nav = np.cumprod(1 + np.random.randn(5000) * 0.01)
///     >>> config = MultiscaleIthConfig(threshold_dbps=250, lookbacks=[100, 500, 1000])
///     >>> features = compute_multiscale_ith(nav, config)
///     >>>
///     >>> # Zero-copy conversion to Polars
///     >>> df = pl.from_arrow(features.to_arrow())
///     >>> print(df.columns)  # ['ith_rb250_lb100_bull_ed', ...]
#[pyfunction]
#[pyo3(signature = (nav, config=None))]
fn compute_multiscale_ith(
    nav: PyReadonlyArray1<f64>,
    config: Option<&PyMultiscaleConfig>,
) -> PyResult<PyMultiscaleIthFeatures> {
    let slice = nav
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    if slice.is_empty() {
        return Err(PyValueError::new_err("nav array cannot be empty"));
    }

    let rust_config = match config {
        Some(cfg) => cfg.inner.clone(),
        None => crate::ith_multiscale::MultiscaleConfig::default(),
    };

    let features = crate::ith_multiscale::compute_multiscale_ith(slice, &rust_config);
    Ok(PyMultiscaleIthFeatures { inner: features })
}

// ============================================================================
// Module Definition
// ============================================================================

/// Python module for trading-fitness-metrics.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Entropy metrics
    m.add_function(wrap_pyfunction!(permutation_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(sample_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(shannon_entropy, m)?)?;

    // Risk metrics
    m.add_function(wrap_pyfunction!(omega_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(ulcer_index, m)?)?;
    m.add_function(wrap_pyfunction!(garman_klass_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(kaufman_efficiency_ratio, m)?)?;

    // Fractal metrics
    m.add_function(wrap_pyfunction!(hurst_exponent, m)?)?;
    m.add_function(wrap_pyfunction!(fractal_dimension, m)?)?;

    // NAV & utilities
    m.add_function(wrap_pyfunction!(build_nav_from_closes, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_windows, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_bins_freedman_diaconis, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_embedding_dimension, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_sample_entropy_tolerance, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_tmaeg, m)?)?;
    m.add_function(wrap_pyfunction!(relative_epsilon, m)?)?;

    // ITH analysis
    m.add_function(wrap_pyfunction!(bull_ith, m)?)?;
    m.add_function(wrap_pyfunction!(bear_ith, m)?)?;

    // Rolling ITH features (time-agnostic, bounded [0, 1])
    m.add_function(wrap_pyfunction!(compute_rolling_ith, m)?)?;

    // Multi-scale ITH features (Arrow-native integration)
    m.add_function(wrap_pyfunction!(compute_multiscale_ith, m)?)?;

    // Batch API
    m.add_function(wrap_pyfunction!(compute_all_metrics, m)?)?;

    // Classes
    m.add_class::<PyBullIthResult>()?;
    m.add_class::<PyBearIthResult>()?;
    m.add_class::<PyRollingIthFeatures>()?;
    m.add_class::<PyGarmanKlassNormalizer>()?;
    m.add_class::<PyOnlineNormalizer>()?;
    m.add_class::<PyMetricsResult>()?;
    m.add_class::<PyMultiscaleConfig>()?;
    m.add_class::<PyMultiscaleIthFeatures>()?;

    Ok(())
}
