# metrics-rust

Time-agnostic ITH metrics for BiLSTM feature engineering.

## Quick Reference

| Action            | Command                                        |
| ----------------- | ---------------------------------------------- |
| Build             | `cargo build -p trading-fitness-metrics`       |
| Test              | `cargo test -p trading-fitness-metrics`        |
| Test with nextest | `cargo nextest run -p trading-fitness-metrics` |
| Check             | `cargo check -p trading-fitness-metrics`       |

## Module Map (9 Price-Only Metrics)

| Module        | Metrics                          | Output Range |
| ------------- | -------------------------------- | ------------ |
| `entropy.rs`  | Permutation, Sample, Shannon     | [0, 1]       |
| `risk.rs`     | Omega, Ulcer, GK Vol, Kaufman ER | [0, 1]       |
| `fractal.rs`  | Hurst, Fractal Dimension         | [0, 1]       |
| `ith.rs`      | Bull ITH, Bear ITH               | epochs       |
| `nav.rs`      | NAV construction                 | prices       |
| `adaptive.rs` | Utilities (no magic numbers)     | various      |
| `types.rs`    | Result types                     | -            |

## Data Requirements

**Input**: Price series only (no volume required)

- Works with Binance aggTrades (close prices)
- Works with Exness tick data (mid prices)

## BiLSTM Integration

All metrics produce bounded [0, 1] outputs for direct use in LSTM/BiLSTM models.

### Normalization Transforms

| Metric                    | Raw Range  | Transform     | Output |
| ------------------------- | ---------- | ------------- | ------ |
| Permutation Entropy       | [0, 1]     | None          | [0, 1] |
| Sample Entropy            | [0, ∞)     | `1-exp(-x)`   | [0, 1) |
| Shannon Entropy           | [0, log n] | `/log(n)`     | [0, 1] |
| Omega Ratio               | [0, ∞)     | `Ω/(1+Ω)`     | [0, 1) |
| Ulcer Index               | [0, 1]     | None          | [0, 1] |
| Garman-Klass Vol          | [0, ∞)     | `tanh(10x)`   | [0, 1) |
| Garman-Klass Vol (stream) | [0, ∞)     | `EMA sigmoid` | (0, 1) |
| Kaufman ER                | [0, 1]     | None          | [0, 1] |
| Hurst Exponent            | [0, 1+]    | `soft_clamp`  | [0, 1] |
| Fractal Dimension         | [1, 2]     | `D-1`         | [0, 1] |

**Garman-Klass Variants**:

- `garman_klass_volatility()` - Stateless, uses `tanh(10x)` for single-bar use
- `garman_klass_volatility_streaming()` - Uses `GarmanKlassNormalizer` for real-time processing

## Minimum Sample Requirements

The library enforces minimum sample sizes for statistical validity:

| Metric                      | Minimum Samples | Rationale            |
| --------------------------- | --------------- | -------------------- |
| Permutation Entropy (m=3)   | 60              | 10 × 3!              |
| Sample Entropy (m=2)        | 200             | Literature consensus |
| Shannon Entropy (n=10 bins) | 100             | 10 × n_bins          |
| Hurst Exponent              | 256             | DFA scale range      |

## Adaptive Utilities

The `adaptive.rs` module provides magic-number-free utilities:

- `relative_epsilon(operand)` - Adaptive division guards
- `OnlineNormalizer` - Welford algorithm with sigmoid
- `GarmanKlassNormalizer` - EMA-based volatility normalization
- `hurst_soft_clamp()` - Bounded output for Hurst
- `adaptive_windows()` - Log-spaced window generation
- `MinimumSamples` - Statistical power requirements

## Example Usage

```rust
use trading_fitness_metrics::{
    omega_ratio, permutation_entropy, hurst_exponent,
    build_nav_from_closes, bull_ith,
};

// Build NAV from close prices
let closes = vec![100.0, 102.0, 101.0, 105.0, 103.0, 108.0];
let nav = build_nav_from_closes(&closes);

// Compute returns for Omega ratio
let returns: Vec<f64> = closes.windows(2)
    .map(|w| (w[1] - w[0]) / w[0])
    .collect();

// All outputs are bounded [0, 1]
let omega = omega_ratio(&returns, 0.0);

// ITH analysis
let ith_result = bull_ith(&nav, 0.05);
println!("Epochs: {}", ith_result.num_of_epochs);
```

## Architecture

```
packages/metrics-rust/
├── Cargo.toml
├── CLAUDE.md
└── src/
    ├── lib.rs          # Module declarations + re-exports
    ├── adaptive.rs     # Adaptive utilities (foundation)
    ├── entropy.rs      # Permutation, Sample, Shannon
    ├── risk.rs         # Omega, Ulcer, GK Vol, Kaufman ER
    ├── fractal.rs      # Hurst, Fractal Dimension
    ├── ith.rs          # Bull/Bear ITH analysis
    ├── nav.rs          # NAV construction
    └── types.rs        # Result types
```

## Test Coverage

76 unit tests + 16 doc tests + 3 real data integration tests covering:

- Edge cases (empty, single point, constant series)
- Boundary validation (outputs in declared ranges)
- Known values (pre-computed references)
- Property tests (monotonicity, scale invariance)
- Real market data (BTCUSDT aggTrades, custom NAV)
