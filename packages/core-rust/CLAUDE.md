# core-rust

> High-performance trading fitness calculations in Rust.

**← [Back to trading-fitness](../../CLAUDE.md)**

## Status

Implemented with ITH analysis and metrics calculations.

## Modules

| Module    | Purpose                             |
| --------- | ----------------------------------- |
| `ith`     | ITH epoch detection and analysis    |
| `metrics` | Sharpe ratio, max drawdown, returns |
| `types`   | Shared type definitions with serde  |

## Key Functions

```rust
// ITH analysis
excess_gain_excess_loss(nav: &[f64], hurdle: f64) -> ExcessGainLossResult
determine_tmaeg(nav: &[f64], method: &str, fixed_value: f64) -> f64

// Metrics
sharpe_ratio(returns: &[f64], periods_per_year: f64, risk_free_rate: f64) -> f64
max_drawdown(nav_values: &[f64]) -> f64
total_return(nav_values: &[f64]) -> f64
pnl_from_nav(nav_values: &[f64]) -> Vec<f64>
calculate_fitness_metrics(nav_values: &[f64], periods_per_year: f64) -> FitnessMetrics
```

## Dependencies

- serde (serialization for JSON interop)
- tracing, tracing-subscriber (structured logging)

## Quick Start

```bash
cargo check   # Verify compilation
cargo test    # Run tests (14 tests)
cargo build --release  # Build optimized binary
```

## Future: Python Bindings

Can be exposed to Python via PyO3 for performance-critical paths.
