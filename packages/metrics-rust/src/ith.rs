// # PROCESS-STORM-OK: Rust test functions with single-letter endings trigger false positives
//! Investment Time Horizon (ITH) analysis for trading strategy fitness.
//!
//! ITH analysis evaluates strategy performance using TMAEG (Target Maximum
//! Acceptable Excess Gain) thresholds to count epochs where the strategy
//! exceeds performance hurdles.

use crate::types::{BearIthResult, BullIthResult};

// ============================================================================
// Bull ITH (Long Position Analysis)
// ============================================================================

/// Calculate Bull ITH (long position) analysis.
///
/// Bull ITH tracks excess gains (upside) and excess losses (drawdowns) for
/// a long-only strategy. An epoch is counted when excess gain exceeds the
/// TMAEG threshold.
///
/// # Arguments
///
/// * `nav` - Net Asset Value series (normalized prices)
/// * `tmaeg` - Target Maximum Acceptable Excess Gain threshold (e.g., 0.05 for 5%)
///
/// # Returns
///
/// `BullIthResult` containing excess gains, excess losses, epoch count, and statistics.
///
/// # Example
///
/// ```rust
/// use trading_fitness_metrics::bull_ith;
///
/// let nav = vec![1.0, 1.02, 1.01, 1.05, 1.03, 1.08];
/// let result = bull_ith(&nav, 0.05);
/// println!("Epochs: {}", result.num_of_epochs);
/// ```
pub fn bull_ith(nav: &[f64], tmaeg: f64) -> BullIthResult {
    if nav.is_empty() {
        return BullIthResult {
            excess_gains: vec![],
            excess_losses: vec![],
            num_of_epochs: 0,
            epochs: vec![],
            intervals_cv: f64::NAN,
            max_drawdown: 0.0,
        };
    }

    let n = nav.len();
    let mut excess_gains = vec![0.0; n];
    let mut excess_losses = vec![0.0; n];
    let mut epochs = vec![false; n];

    // Track running maximum (crest) for drawdown calculation
    let mut crest = nav[0];
    let mut max_drawdown = 0.0;

    // Track epoch intervals for CV calculation
    let mut epoch_indices = Vec::new();

    for i in 0..n {
        let current = nav[i];

        // Update crest (running maximum)
        if current > crest {
            crest = current;
        }

        // Excess gain: gain from the last trough (crest before drawdown)
        // For simplicity, we track gain from initial NAV
        let excess_gain = if nav[0] > 0.0 {
            (current - nav[0]) / nav[0]
        } else {
            0.0
        };

        // Excess loss: current drawdown from crest
        let excess_loss = if crest > 0.0 {
            (crest - current) / crest
        } else {
            0.0
        };

        excess_gains[i] = excess_gain.max(0.0);
        excess_losses[i] = excess_loss.max(0.0);

        // Track maximum drawdown
        if excess_loss > max_drawdown {
            max_drawdown = excess_loss;
        }

        // Mark epoch if excess gain exceeds threshold
        if excess_gain >= tmaeg {
            epochs[i] = true;
            epoch_indices.push(i);
        }
    }

    // Count distinct epochs (consecutive true values count as one)
    let num_of_epochs = count_distinct_epochs(&epochs);

    // Calculate coefficient of variation of epoch intervals
    let intervals_cv = calculate_intervals_cv(&epoch_indices);

    BullIthResult {
        excess_gains,
        excess_losses,
        num_of_epochs,
        epochs,
        intervals_cv,
        max_drawdown,
    }
}

// ============================================================================
// Bear ITH (Short Position Analysis)
// ============================================================================

/// Calculate Bear ITH (short position) analysis.
///
/// Bear ITH tracks excess gains (downside for shorts) and excess losses
/// (runups, which are adverse for shorts). An epoch is counted when the
/// price drops enough to generate excess gain exceeding the TMAEG threshold.
///
/// # Arguments
///
/// * `nav` - Net Asset Value series (normalized prices)
/// * `tmaeg` - Target Maximum Acceptable Excess Gain threshold (e.g., 0.05 for 5%)
///
/// # Returns
///
/// `BearIthResult` containing excess gains, excess losses, epoch count, and statistics.
///
/// # Example
///
/// ```rust
/// use trading_fitness_metrics::bear_ith;
///
/// let nav = vec![1.0, 0.98, 0.99, 0.95, 0.97, 0.92];
/// let result = bear_ith(&nav, 0.05);
/// println!("Bear Epochs: {}", result.num_of_epochs);
/// ```
pub fn bear_ith(nav: &[f64], tmaeg: f64) -> BearIthResult {
    if nav.is_empty() {
        return BearIthResult {
            excess_gains: vec![],
            excess_losses: vec![],
            num_of_epochs: 0,
            epochs: vec![],
            intervals_cv: f64::NAN,
            max_runup: 0.0,
        };
    }

    let n = nav.len();
    let mut excess_gains = vec![0.0; n];
    let mut excess_losses = vec![0.0; n];
    let mut epochs = vec![false; n];

    // Track running minimum (trough) for runup calculation
    let mut trough = nav[0];
    let mut max_runup = 0.0;

    // Track epoch intervals for CV calculation
    let mut epoch_indices = Vec::new();

    for i in 0..n {
        let current = nav[i];

        // Update trough (running minimum)
        if current < trough {
            trough = current;
        }

        // Excess gain for shorts: price drop from initial
        let excess_gain = if nav[0] > 0.0 {
            (nav[0] - current) / nav[0]
        } else {
            0.0
        };

        // Excess loss for shorts: runup from trough (adverse movement)
        let excess_loss = if trough > 0.0 {
            (current - trough) / trough
        } else {
            0.0
        };

        excess_gains[i] = excess_gain.max(0.0);
        excess_losses[i] = excess_loss.max(0.0);

        // Track maximum runup
        if excess_loss > max_runup {
            max_runup = excess_loss;
        }

        // Mark epoch if excess gain exceeds threshold
        if excess_gain >= tmaeg {
            epochs[i] = true;
            epoch_indices.push(i);
        }
    }

    // Count distinct epochs
    let num_of_epochs = count_distinct_epochs(&epochs);

    // Calculate coefficient of variation of epoch intervals
    let intervals_cv = calculate_intervals_cv(&epoch_indices);

    BearIthResult {
        excess_gains,
        excess_losses,
        num_of_epochs,
        epochs,
        intervals_cv,
        max_runup,
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Count distinct epochs (consecutive true values count as one epoch).
fn count_distinct_epochs(epochs: &[bool]) -> usize {
    let mut count = 0;
    let mut in_epoch = false;

    for &is_epoch in epochs {
        if is_epoch && !in_epoch {
            count += 1;
            in_epoch = true;
        } else if !is_epoch {
            in_epoch = false;
        }
    }

    count
}

/// Calculate coefficient of variation of epoch intervals.
fn calculate_intervals_cv(epoch_indices: &[usize]) -> f64 {
    if epoch_indices.len() < 2 {
        return f64::NAN;
    }

    // Calculate intervals between consecutive epochs
    let intervals: Vec<f64> = epoch_indices
        .windows(2)
        .map(|w| (w[1] - w[0]) as f64)
        .collect();

    if intervals.is_empty() {
        return f64::NAN;
    }

    let mean: f64 = intervals.iter().sum::<f64>() / intervals.len() as f64;
    if mean.abs() < f64::EPSILON {
        return f64::NAN;
    }

    let variance: f64 = intervals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
        / intervals.len() as f64;

    let std_dev = variance.sqrt();

    std_dev / mean
}

#[cfg(test)]
mod tests {
    use super::*;

    // Bull ITH tests
    #[test]
    fn test_bull_ith_empty() {
        let result = bull_ith(&[], 0.05);
        assert_eq!(result.num_of_epochs, 0);
        assert!(result.intervals_cv.is_nan());
    }

    #[test]
    fn test_bull_ith_no_epochs() {
        // Flat or declining NAV should have no epochs
        let nav = vec![1.0, 0.99, 0.98, 0.97, 0.96];
        let result = bull_ith(&nav, 0.05);
        assert_eq!(result.num_of_epochs, 0);
    }

    #[test]
    fn test_bull_ith_with_epochs() {
        // Rising NAV should have epochs
        let nav = vec![1.0, 1.02, 1.04, 1.06, 1.08, 1.10];
        let result = bull_ith(&nav, 0.05);
        assert!(result.num_of_epochs > 0);
    }

    #[test]
    fn test_bull_ith_max_drawdown() {
        let nav = vec![1.0, 1.10, 1.05, 1.15, 1.00];
        let result = bull_ith(&nav, 0.05);
        // Max drawdown from 1.15 to 1.00 = 13%
        assert!(result.max_drawdown > 0.10);
    }

    // Bear ITH tests
    #[test]
    fn test_bear_ith_empty() {
        let result = bear_ith(&[], 0.05);
        assert_eq!(result.num_of_epochs, 0);
        assert!(result.intervals_cv.is_nan());
    }

    #[test]
    fn test_bear_ith_no_epochs() {
        // Rising NAV should have no bear epochs
        let nav = vec![1.0, 1.01, 1.02, 1.03, 1.04];
        let result = bear_ith(&nav, 0.05);
        assert_eq!(result.num_of_epochs, 0);
    }

    #[test]
    fn test_bear_ith_with_epochs() {
        // Falling NAV should have bear epochs
        let nav = vec![1.0, 0.98, 0.96, 0.94, 0.92, 0.90];
        let result = bear_ith(&nav, 0.05);
        assert!(result.num_of_epochs > 0);
    }

    #[test]
    fn test_bear_ith_max_runup() {
        let nav = vec![1.0, 0.90, 0.95, 0.85, 1.00];
        let result = bear_ith(&nav, 0.05);
        // Max runup from 0.85 to 1.00 = 17.6%
        assert!(result.max_runup > 0.15);
    }

    // Helper function tests
    #[test]
    fn test_count_distinct_epochs() {
        let epochs = vec![false, true, true, false, true, false, false, true, true];
        assert_eq!(count_distinct_epochs(&epochs), 3);
    }

    #[test]
    fn test_count_distinct_epochs_none() {
        let epochs = vec![false, false, false];
        assert_eq!(count_distinct_epochs(&epochs), 0);
    }

    #[test]
    fn test_count_distinct_epochs_all() {
        let epochs = vec![true, true, true];
        assert_eq!(count_distinct_epochs(&epochs), 1);
    }

    #[test]
    fn test_intervals_cv_calculation() {
        let indices = vec![0, 10, 20, 30];
        let cv = calculate_intervals_cv(&indices);
        // All intervals are 10, so CV should be 0
        assert!((cv - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_intervals_cv_insufficient_data() {
        let indices = vec![5];
        assert!(calculate_intervals_cv(&indices).is_nan());
    }
}
