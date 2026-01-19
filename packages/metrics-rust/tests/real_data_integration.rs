// # PROCESS-STORM-OK: Rust test functions with single-letter endings trigger false positives
//! Real data integration tests for trading-fitness-metrics.
//!
//! These tests use actual market data, not synthetic data.

use trading_fitness_metrics::{
    build_nav_from_closes, bull_ith, bear_ith,
    fractal_dimension, hurst_exponent,
    kaufman_efficiency_ratio, omega_ratio, ulcer_index,
    permutation_entropy, sample_entropy, shannon_entropy,
};

/// Parse close prices from aggTrades CSV (price is column 2, 0-indexed column 1)
fn parse_aggtrades_closes(csv_content: &str) -> Vec<f64> {
    csv_content
        .lines()
        .filter_map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() >= 2 {
                fields[1].parse::<f64>().ok()
            } else {
                None
            }
        })
        .collect()
}

/// Parse NAV from trading-fitness NAV CSV (NAV is column 3, header row exists)
fn parse_nav_csv(csv_content: &str) -> Vec<f64> {
    csv_content
        .lines()
        .skip(1) // Skip header
        .filter_map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() >= 3 {
                fields[2].parse::<f64>().ok()
            } else {
                None
            }
        })
        .collect()
}

#[test]
fn test_real_btcusdt_aggtrades_metrics() {
    // Load real BTCUSDT data
    let csv_path = std::env::var("HOME").unwrap() + "/eon/rangebar-py/tests/fixtures/BTCUSDT-aggTrades-sample-10k.csv";
    let csv_content = std::fs::read_to_string(&csv_path)
        .expect("Failed to read BTCUSDT aggTrades CSV");

    let closes = parse_aggtrades_closes(&csv_content);
    assert!(closes.len() >= 1000, "Expected at least 1000 prices, got {}", closes.len());

    // Build NAV from closes
    let nav = build_nav_from_closes(&closes);
    assert_eq!(nav.len(), closes.len());

    // Compute returns for entropy/risk metrics
    let returns: Vec<f64> = closes.windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    // Test all 9 metrics produce bounded outputs

    // 1. Omega Ratio
    let omega = omega_ratio(&returns, 0.0);
    assert!(!omega.is_nan(), "Omega ratio is NaN");
    assert!(omega >= 0.0 && omega < 1.0, "Omega ratio {} not in [0, 1)", omega);

    // 2. Ulcer Index
    let ui = ulcer_index(&nav);
    assert!(!ui.is_nan(), "Ulcer index is NaN");
    assert!(ui >= 0.0 && ui <= 1.0, "Ulcer index {} not in [0, 1]", ui);

    // 3. Kaufman Efficiency Ratio
    let er = kaufman_efficiency_ratio(&nav);
    assert!(!er.is_nan(), "Kaufman ER is NaN");
    assert!(er >= 0.0 && er <= 1.0, "Kaufman ER {} not in [0, 1]", er);

    // 4. Hurst Exponent (needs at least 256 samples)
    let h = hurst_exponent(&nav);
    if !h.is_nan() {
        assert!(h >= 0.0 && h <= 1.0, "Hurst exponent {} not in [0, 1]", h);
    }

    // 5. Fractal Dimension
    let fd = fractal_dimension(&nav, 10);
    if !fd.is_nan() {
        assert!(fd >= 0.0 && fd <= 1.0, "Fractal dimension {} not in [0, 1]", fd);
    }

    // 6. Permutation Entropy (needs sufficient samples)
    let pe = permutation_entropy(&nav, 3);
    if !pe.is_nan() {
        assert!(pe >= 0.0 && pe <= 1.0, "Permutation entropy {} not in [0, 1]", pe);
    }

    // 7. Sample Entropy (needs 200+ samples for m=2)
    let se = sample_entropy(&returns, 2, 0.2);
    if !se.is_nan() {
        assert!(se >= 0.0 && se < 1.0, "Sample entropy {} not in [0, 1)", se);
    }

    // 8. Shannon Entropy
    let sh = shannon_entropy(&returns, 20);
    if !sh.is_nan() {
        assert!(sh >= 0.0 && sh <= 1.0, "Shannon entropy {} not in [0, 1]", sh);
    }

    // Print summary
    println!("\n=== BTCUSDT Real Data Metrics ===");
    println!("Data points: {}", closes.len());
    println!("Omega Ratio: {:.4}", omega);
    println!("Ulcer Index: {:.4}", ui);
    println!("Kaufman ER: {:.4}", er);
    println!("Hurst Exponent: {:.4}", h);
    println!("Fractal Dimension: {:.4}", fd);
    println!("Permutation Entropy: {:.4}", pe);
    println!("Sample Entropy: {:.4}", se);
    println!("Shannon Entropy: {:.4}", sh);
}

#[test]
fn test_real_nav_data_ith_analysis() {
    // Load real NAV data from trading-fitness
    let csv_path = std::env::var("HOME").unwrap() + "/eon/trading-fitness/data/nav_data_custom/suresh.csv";
    let csv_content = std::fs::read_to_string(&csv_path)
        .expect("Failed to read NAV CSV");

    let raw_nav = parse_nav_csv(&csv_content);
    assert!(raw_nav.len() >= 100, "Expected at least 100 NAV points, got {}", raw_nav.len());

    // Normalize NAV to start at 1.0
    let first = raw_nav[0];
    let nav: Vec<f64> = raw_nav.iter().map(|&x| x / first).collect();

    // Bull ITH analysis
    let bull_result = bull_ith(&nav, 0.05);
    assert!(!bull_result.excess_gains.is_empty());
    assert!(!bull_result.excess_losses.is_empty());
    println!("\n=== Bull ITH Analysis ===");
    println!("NAV points: {}", nav.len());
    println!("Bull epochs: {}", bull_result.num_of_epochs);
    println!("Max drawdown: {:.4}", bull_result.max_drawdown);

    // Bear ITH analysis
    let bear_result = bear_ith(&nav, 0.05);
    assert!(!bear_result.excess_gains.is_empty());
    assert!(!bear_result.excess_losses.is_empty());
    println!("\n=== Bear ITH Analysis ===");
    println!("Bear epochs: {}", bear_result.num_of_epochs);
    println!("Max runup: {:.4}", bear_result.max_runup);

    // Verify all metrics work on this NAV
    let omega = omega_ratio(&nav.windows(2).map(|w| w[1] / w[0] - 1.0).collect::<Vec<_>>(), 0.0);
    let ui = ulcer_index(&nav);
    let er = kaufman_efficiency_ratio(&nav);

    println!("\n=== NAV Metrics ===");
    println!("Omega Ratio: {:.4}", omega);
    println!("Ulcer Index: {:.4}", ui);
    println!("Kaufman ER: {:.4}", er);
}

#[test]
fn test_garman_klass_with_real_ohlc_pattern() {
    // Simulate OHLC from real price movements (approximation)
    let csv_path = std::env::var("HOME").unwrap() + "/eon/rangebar-py/tests/fixtures/BTCUSDT-aggTrades-sample-10k.csv";
    let csv_content = std::fs::read_to_string(&csv_path)
        .expect("Failed to read BTCUSDT aggTrades CSV");

    let closes = parse_aggtrades_closes(&csv_content);

    // Group into pseudo-bars of 100 trades each
    let bar_size = 100;
    let num_bars = closes.len() / bar_size;

    let mut gk_vols = Vec::new();

    for i in 0..num_bars {
        let start = i * bar_size;
        let end = (i + 1) * bar_size;
        let bar_prices = &closes[start..end];

        let open = bar_prices[0];
        let close = bar_prices[bar_size - 1];
        let high = bar_prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let low = bar_prices.iter().cloned().fold(f64::INFINITY, f64::min);

        let gk = trading_fitness_metrics::garman_klass_volatility(open, high, low, close);
        if !gk.is_nan() {
            assert!(gk >= 0.0 && gk < 1.0, "GK volatility {} not in [0, 1)", gk);
            gk_vols.push(gk);
        }
    }

    println!("\n=== Garman-Klass Volatility (Real OHLC) ===");
    println!("Number of bars: {}", gk_vols.len());
    if !gk_vols.is_empty() {
        let mean_gk: f64 = gk_vols.iter().sum::<f64>() / gk_vols.len() as f64;
        let max_gk = gk_vols.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_gk = gk_vols.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("Mean GK Vol: {:.4}", mean_gk);
        println!("Min GK Vol: {:.4}", min_gk);
        println!("Max GK Vol: {:.4}", max_gk);
    }
}
