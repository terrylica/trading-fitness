//! NAV (Net Asset Value) construction utilities.
//!
//! Converts price series to NAV series with proper return clamping.

/// Build NAV series from close prices.
///
/// Computes percentage returns, clamps them to prevent -100% or worse,
/// and constructs cumulative NAV starting at 1.0.
///
/// # Arguments
///
/// * `closes` - Slice of close prices
///
/// # Returns
///
/// NAV series starting at 1.0, with length equal to input length.
///
/// # Example
///
/// ```rust
/// use trading_fitness_metrics::build_nav_from_closes;
///
/// let closes = vec![100.0, 102.0, 101.0, 105.0];
/// let nav = build_nav_from_closes(&closes);
/// assert_eq!(nav.len(), 4);
/// assert!((nav[0] - 1.0).abs() < 0.001);
/// ```
pub fn build_nav_from_closes(closes: &[f64]) -> Vec<f64> {
    if closes.is_empty() {
        return vec![];
    }

    if closes.len() == 1 {
        return vec![1.0];
    }

    let mut nav = Vec::with_capacity(closes.len());
    nav.push(1.0);

    for i in 1..closes.len() {
        let prev_close = closes[i - 1];
        let curr_close = closes[i];

        // Calculate return, guarding against division by zero
        let ret = if prev_close.abs() > f64::EPSILON {
            (curr_close - prev_close) / prev_close
        } else {
            0.0
        };

        // Clamp return to prevent NAV going negative
        // -0.99 floor prevents total wipeout
        let clamped_ret = ret.max(-0.99);

        // Apply return to previous NAV
        let prev_nav = nav.last().copied().unwrap_or(1.0);
        let new_nav = prev_nav * (1.0 + clamped_ret);

        nav.push(new_nav.max(0.0)); // Ensure non-negative
    }

    nav
}

/// Build NAV series from returns directly.
///
/// # Arguments
///
/// * `returns` - Slice of returns (e.g., 0.02 for 2%)
///
/// # Returns
///
/// NAV series starting at 1.0, with length = returns.len() + 1.
pub fn build_nav_from_returns(returns: &[f64]) -> Vec<f64> {
    if returns.is_empty() {
        return vec![1.0];
    }

    let mut nav = Vec::with_capacity(returns.len() + 1);
    nav.push(1.0);

    for &ret in returns {
        let clamped_ret = ret.max(-0.99);
        let prev_nav = nav.last().copied().unwrap_or(1.0);
        let new_nav = prev_nav * (1.0 + clamped_ret);
        nav.push(new_nav.max(0.0));
    }

    nav
}

/// Extract returns from a price series.
///
/// # Arguments
///
/// * `prices` - Slice of prices
///
/// # Returns
///
/// Vector of returns with length = prices.len() - 1.
pub fn compute_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }

    prices
        .windows(2)
        .map(|w| {
            if w[0].abs() > f64::EPSILON {
                (w[1] - w[0]) / w[0]
            } else {
                0.0
            }
        })
        .collect()
}

/// Normalize a price series to start at 1.0.
///
/// Useful for comparing multiple price series on the same scale.
///
/// # Arguments
///
/// * `prices` - Slice of prices
///
/// # Returns
///
/// Normalized price series starting at 1.0.
pub fn normalize_prices(prices: &[f64]) -> Vec<f64> {
    if prices.is_empty() {
        return vec![];
    }

    let first = prices[0];
    if first.abs() < f64::EPSILON {
        // If first price is zero, return ones
        return vec![1.0; prices.len()];
    }

    prices.iter().map(|&p| p / first).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_nav_from_closes_basic() {
        let closes = vec![100.0, 110.0, 105.0, 115.0];
        let nav = build_nav_from_closes(&closes);

        assert_eq!(nav.len(), 4);
        assert!((nav[0] - 1.0).abs() < 0.001);
        assert!((nav[1] - 1.1).abs() < 0.001); // 10% gain
    }

    #[test]
    fn test_build_nav_from_closes_empty() {
        let nav = build_nav_from_closes(&[]);
        assert!(nav.is_empty());
    }

    #[test]
    fn test_build_nav_from_closes_single() {
        let nav = build_nav_from_closes(&[100.0]);
        assert_eq!(nav.len(), 1);
        assert!((nav[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_build_nav_from_closes_large_loss_clamped() {
        // 95% loss should be clamped to 99%
        let closes = vec![100.0, 5.0];
        let nav = build_nav_from_closes(&closes);

        // Without clamping would be 0.05, with -0.99 clamp it's 0.01
        assert!(nav[1] >= 0.01);
    }

    #[test]
    fn test_build_nav_from_returns_basic() {
        let returns = vec![0.10, -0.05, 0.08];
        let nav = build_nav_from_returns(&returns);

        assert_eq!(nav.len(), 4);
        assert!((nav[0] - 1.0).abs() < 0.001);
        assert!((nav[1] - 1.1).abs() < 0.001);
    }

    #[test]
    fn test_build_nav_from_returns_empty() {
        let nav = build_nav_from_returns(&[]);
        assert_eq!(nav.len(), 1);
        assert!((nav[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_returns_basic() {
        let prices = vec![100.0, 110.0, 99.0];
        let returns = compute_returns(&prices);

        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.10).abs() < 0.001);
        assert!((returns[1] - (-0.10)).abs() < 0.001);
    }

    #[test]
    fn test_compute_returns_short() {
        assert!(compute_returns(&[]).is_empty());
        assert!(compute_returns(&[100.0]).is_empty());
    }

    #[test]
    fn test_normalize_prices_basic() {
        let prices = vec![50.0, 55.0, 45.0, 60.0];
        let normalized = normalize_prices(&prices);

        assert_eq!(normalized.len(), 4);
        assert!((normalized[0] - 1.0).abs() < 0.001);
        assert!((normalized[1] - 1.1).abs() < 0.001);
    }

    #[test]
    fn test_normalize_prices_empty() {
        let normalized = normalize_prices(&[]);
        assert!(normalized.is_empty());
    }

    #[test]
    fn test_normalize_prices_zero_start() {
        let prices = vec![0.0, 10.0, 20.0];
        let normalized = normalize_prices(&prices);
        // Should return all 1.0 when first price is zero
        assert!(normalized.iter().all(|&x| (x - 1.0).abs() < 0.001));
    }
}
