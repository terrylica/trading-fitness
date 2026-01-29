# metrics-rust

> Time-agnostic ITH metrics for BiLSTM feature engineering with PyO3 Python bindings.

**← [Back to trading-fitness](../../CLAUDE.md)**

## Quick Reference

| Action           | Command                                        |
| ---------------- | ---------------------------------------------- |
| Build (Rust)     | `cargo build -p trading-fitness-metrics`       |
| Test (Rust)      | `cargo nextest run -p trading-fitness-metrics` |
| Build (Python)   | `maturin build --features python`              |
| Develop (Python) | `maturin develop --features python`            |

---

## Module Map

| Module              | Metrics                          | Output |
| ------------------- | -------------------------------- | ------ |
| `entropy.rs`        | Permutation, Sample, Shannon     | [0, 1] |
| `risk.rs`           | Omega, Ulcer, GK Vol, Kaufman ER | [0, 1] |
| `fractal.rs`        | Hurst, Fractal Dimension         | [0, 1] |
| `ith.rs`            | Bull ITH, Bear ITH               | epochs |
| `ith_rolling.rs`    | Rolling window ITH features      | [0, 1] |
| `ith_multiscale.rs` | Multi-scale ITH (Arrow-native)   | [0, 1] |
| `python.rs`         | PyO3 bindings                    | -      |

**All metrics produce bounded [0, 1] outputs** suitable for LSTM/BiLSTM consumption.

---

## Critical: TMAEG Auto-Calculation

The `threshold_dbps` parameter in `MultiscaleIthConfig` is **for column naming only**. The actual TMAEG is auto-calculated:

```
tmaeg = 3.0 × MAD_std × sqrt(lookback), clamped to [0.001, 0.50]
```

To get different feature values, fetch range bars at different thresholds (different NAV series) BEFORE computing ITH.

---

## Python API (Quick)

```python
from trading_fitness_metrics import (
    compute_rolling_ith, optimal_tmaeg,
    bull_ith, bear_ith,
    compute_all_metrics,
    MultiscaleIthConfig, compute_multiscale_ith,
)

# Rolling ITH features (8 bounded [0,1] outputs)
features = compute_rolling_ith(nav, lookback=100)

# Multi-scale ITH (Arrow RecordBatch)
config = MultiscaleIthConfig(lookbacks=[20, 50, 100], threshold_dbps=25)
batch = compute_multiscale_ith(nav, config)
```

### 8 Features per Lookback

| Feature   | Range  | Description                |
| --------- | ------ | -------------------------- |
| `bull_ed` | [0, 1] | Bull epoch density         |
| `bear_ed` | [0, 1] | Bear epoch density         |
| `bull_eg` | [0, 1] | Bull excess gain (tanh)    |
| `bear_eg` | [0, 1] | Bear excess gain (tanh)    |
| `bull_cv` | [0, 1] | Bull intervals CV          |
| `bear_cv` | [0, 1] | Bear intervals CV          |
| `max_dd`  | [0, 1] | Maximum drawdown in window |
| `max_ru`  | [0, 1] | Maximum runup in window    |

---

## Column Naming Convention

```
ith_rb{threshold}_lb{lookback}_{feature}

Example: ith_rb25_lb100_bull_ed
  - rb25: Range bar 25 dbps (identification only)
  - lb100: 100-bar lookback
  - bull_ed: Bull epoch density
```

---

## Architecture

```
packages/metrics-rust/
├── Cargo.toml
├── pyproject.toml          # Maturin build config
├── src/
│   ├── lib.rs              # Re-exports
│   ├── adaptive.rs         # Magic-number-free utilities
│   ├── entropy.rs          # Entropy metrics
│   ├── risk.rs             # Risk metrics
│   ├── fractal.rs          # Fractal metrics
│   ├── ith.rs              # Core ITH analysis
│   ├── ith_rolling.rs      # Rolling window features
│   ├── ith_multiscale.rs   # Multi-scale (Arrow)
│   └── python.rs           # PyO3 bindings
└── python/
    └── trading_fitness_metrics/
        ├── __init__.py
        └── __init__.pyi    # Type stubs
```

---

## Related Documentation

| Document                                           | Purpose                |
| -------------------------------------------------- | ---------------------- |
| [docs/ITH.md](../../docs/ITH.md)                   | ITH methodology        |
| [docs/forensic/E2E.md](../../docs/forensic/E2E.md) | Sparse wide table docs |
| [VALIDATION_REPORT.md](VALIDATION_REPORT.md)       | Test coverage report   |
| [ith-python](../ith-python/CLAUDE.md)              | Python consumer        |
