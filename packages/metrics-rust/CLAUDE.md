# metrics-rust

> Time-agnostic ITH metrics for BiLSTM feature engineering.

**← [Back to trading-fitness](../../CLAUDE.md)**

## Quick Reference

| Action           | Command                                        |
| ---------------- | ---------------------------------------------- |
| Build (Rust)     | `cargo build -p trading-fitness-metrics`       |
| Test (Rust)      | `cargo nextest run -p trading-fitness-metrics` |
| Build (Python)   | `maturin build --features python`              |
| Develop (Python) | `maturin develop --features python`            |

## Module Map (9 Price-Only Metrics + Rolling/Multi-Scale ITH)

| Module              | Metrics                          | Output Range |
| ------------------- | -------------------------------- | ------------ |
| `entropy.rs`        | Permutation, Sample, Shannon     | [0, 1]       |
| `risk.rs`           | Omega, Ulcer, GK Vol, Kaufman ER | [0, 1]       |
| `fractal.rs`        | Hurst, Fractal Dimension         | [0, 1]       |
| `ith.rs`            | Bull ITH, Bear ITH               | epochs       |
| `ith_rolling.rs`    | Rolling window ITH features      | [0, 1]       |
| `ith_multiscale.rs` | Multi-scale ITH (Arrow-native)   | [0, 1]       |
| `ith_normalize.rs`  | Bounded normalization utilities  | [0, 1]       |
| `nav.rs`            | NAV construction                 | prices       |
| `adaptive.rs`       | Utilities (no magic numbers)     | various      |
| `python.rs`         | PyO3 bindings                    | -            |

## Data Requirements

**Input**: Price series only (no volume required)

- Works with Binance aggTrades (close prices)
- Works with Exness tick data (mid prices)

---

## Python Bindings (PyO3)

The library provides zero-copy Python bindings via PyO3 and maturin.

### Installation

```bash
cd packages/metrics-rust

# Development install
maturin develop --features python

# Build wheel for distribution
maturin build --features python --release
# Wheel: target/wheels/trading_fitness_metrics-*.whl
```

### Python API

```python
import numpy as np
from trading_fitness_metrics import (
    # Entropy metrics
    permutation_entropy, sample_entropy, shannon_entropy,
    # Risk metrics
    omega_ratio, ulcer_index, garman_klass_volatility, kaufman_efficiency_ratio,
    # Fractal metrics
    hurst_exponent, fractal_dimension,
    # NAV & utilities
    build_nav_from_closes, adaptive_windows,
    optimal_bins_freedman_diaconis, optimal_embedding_dimension,
    optimal_sample_entropy_tolerance, relative_epsilon,
    # ITH analysis
    bull_ith, bear_ith,
    # Batch API
    compute_all_metrics,
    # Classes
    BullIthResult, BearIthResult, MetricsResult,
    GarmanKlassNormalizer, OnlineNormalizer,
)
```

### Function Reference

#### Entropy Metrics

| Function              | Signature                      | Output |
| --------------------- | ------------------------------ | ------ |
| `permutation_entropy` | `(prices, m=3) -> float`       | [0, 1] |
| `sample_entropy`      | `(data, m=2, r=None) -> float` | [0, 1) |
| `shannon_entropy`     | `(data, n_bins=None) -> float` | [0, 1] |

#### Risk Metrics

| Function                   | Signature                           | Output |
| -------------------------- | ----------------------------------- | ------ |
| `omega_ratio`              | `(returns, threshold=0.0) -> float` | [0, 1) |
| `ulcer_index`              | `(prices) -> float`                 | [0, 1] |
| `garman_klass_volatility`  | `(open, high, low, close) -> float` | [0, 1) |
| `kaufman_efficiency_ratio` | `(prices) -> float`                 | [0, 1] |

#### Fractal Metrics

| Function            | Signature                     | Output |
| ------------------- | ----------------------------- | ------ |
| `hurst_exponent`    | `(prices) -> float`           | [0, 1] |
| `fractal_dimension` | `(prices, k_max=10) -> float` | [0, 1] |

#### ITH Analysis

| Function   | Signature                       | Returns                 |
| ---------- | ------------------------------- | ----------------------- |
| `bull_ith` | `(nav, tmaeg) -> BullIthResult` | Long position analysis  |
| `bear_ith` | `(nav, tmaeg) -> BearIthResult` | Short position analysis |

#### Rolling ITH Features (Time-Agnostic)

| Function              | Signature                               | Returns                  |
| --------------------- | --------------------------------------- | ------------------------ |
| `compute_rolling_ith` | `(nav, lookback) -> RollingIthFeatures` | 8 bounded [0,1] features |
| `optimal_tmaeg`       | `(nav, lookback) -> float`              | Auto-calculated TMAEG    |

#### Batch API

| Function              | Signature                                       | Returns       |
| --------------------- | ----------------------------------------------- | ------------- |
| `compute_all_metrics` | `(prices, returns, ohlc=None) -> MetricsResult` | All 9 metrics |

### Classes

#### `BullIthResult` / `BearIthResult`

```python
result = bull_ith(nav, tmaeg=0.05)
result.num_of_epochs      # int: Number of epochs detected
result.max_drawdown       # float: Maximum drawdown (bull) / max_runup (bear)
result.intervals_cv       # float: Coefficient of variation
result.excess_gains()     # NDArray[float64]: Excess gains per bar
result.excess_losses()    # NDArray[float64]: Excess losses per bar
result.epochs()           # NDArray[bool_]: Boolean mask of epochs
```

#### `MetricsResult`

```python
metrics = compute_all_metrics(prices, returns, ohlc=(o, h, l, c))
metrics.permutation_entropy   # float
metrics.sample_entropy        # float
metrics.shannon_entropy       # float
metrics.omega_ratio           # float
metrics.ulcer_index           # float
metrics.garman_klass_vol      # float
metrics.kaufman_er            # float
metrics.hurst_exponent        # float
metrics.fractal_dimension     # float
metrics.all_bounded()         # bool: All in [0, 1]
metrics.has_nan()             # bool: Any NaN values
```

#### `RollingIthFeatures`

```python
from trading_fitness_metrics import compute_rolling_ith, optimal_tmaeg

# Compute rolling features over 100-bar lookback windows
# TMAEG is automatically calculated from data volatility
features = compute_rolling_ith(nav, lookback=100)

# To inspect the auto-calculated TMAEG:
auto_tmaeg = optimal_tmaeg(nav, lookback=100)

# 8 feature arrays, all bounded [0, 1]
features.bull_epoch_density    # NDArray: Normalized bull epoch count
features.bear_epoch_density    # NDArray: Normalized bear epoch count
features.bull_excess_gain      # NDArray: Normalized sum of bull excess gains
features.bear_excess_gain      # NDArray: Normalized sum of bear excess gains
features.bull_cv               # NDArray: Normalized bull intervals CV
features.bear_cv               # NDArray: Normalized bear intervals CV
features.max_drawdown          # NDArray: Max drawdown in window
features.max_runup             # NDArray: Max runup in window

# First lookback-1 values are NaN (insufficient data)
assert np.all(np.isnan(features.bull_epoch_density[:99]))
```

#### `GarmanKlassNormalizer` / `OnlineNormalizer`

```python
# Stateful normalizers for streaming data
normalizer = GarmanKlassNormalizer(expected_len=100)
for bar in ohlc_bars:
    raw_vol = garman_klass_volatility(bar.o, bar.h, bar.l, bar.c)
    normalized = normalizer.normalize(raw_vol)  # (0, 1)
normalizer.reset()  # Reset state
```

### Python Example

```python
import numpy as np
from trading_fitness_metrics import (
    build_nav_from_closes, omega_ratio, hurst_exponent,
    bull_ith, compute_all_metrics
)

# Sample price data
prices = np.array([100.0, 102.5, 101.0, 105.0, 103.5, 108.0, 106.0], dtype=np.float64)
returns = np.diff(prices) / prices[:-1]

# Individual metrics (all outputs [0, 1])
omega = omega_ratio(returns, threshold=0.0)
print(f"Omega ratio: {omega:.4f}")

# For Hurst, need 256+ samples
long_prices = np.cumsum(np.random.randn(300)) + 1000
hurst = hurst_exponent(long_prices)
print(f"Hurst exponent: {hurst:.4f}")

# ITH analysis
nav = build_nav_from_closes(prices)
result = bull_ith(nav, tmaeg=0.05)
print(f"Bull epochs: {result.num_of_epochs}")

# Batch API (requires 256+ prices)
metrics = compute_all_metrics(long_prices, np.diff(long_prices)/long_prices[:-1])
print(f"All bounded: {metrics.all_bounded()}")
```

---

## Rolling ITH Features (Time-Agnostic)

The `compute_rolling_ith()` function computes ITH features over sliding windows,
producing 8 bounded [0, 1] outputs suitable for LSTM/BiLSTM consumption.

### Auto-TMAEG Calculation

The TMAEG threshold is **automatically calculated** based on the data's volatility
using MAD (Median Absolute Deviation) based estimation. This ensures sensible
epoch density regardless of the bar type or instrument volatility.

```
tmaeg = 3.0 × MAD_std × sqrt(lookback), clamped to [0.001, 0.50]
where MAD_std = 1.4826 × MAD(returns)
```

To inspect the calculated TMAEG, use `optimal_tmaeg(nav, lookback)`.

### Feature Normalization

| Feature       | Raw Range | Transform                         | Output |
| ------------- | --------- | --------------------------------- | ------ |
| epoch_density | [0, ∞)    | `min(epochs / (lookback/100), 1)` | [0, 1] |
| excess_gain   | [0, ∞)    | `tanh(x * 5.0)`                   | [0, 1) |
| intervals_cv  | [0, ∞)    | `sigmoid((cv - 0.5) * 4.0)`       | (0, 1) |
| max_drawdown  | [0, 1]    | None (already bounded)            | [0, 1] |
| max_runup     | [0, 1]    | None (already bounded)            | [0, 1] |

### Time-Agnostic Design

Rolling ITH features are designed for use with any bar type:

- **Range bars**: Fixed price movement per bar
- **Tick bars**: Fixed number of trades per bar
- **Volume bars**: Fixed volume per bar
- **Time bars**: Traditional OHLC (still works)

### Downstream Usage Example

```python
# In consuming project (e.g., rangebar-py)
from rangebar import get_range_bars
from trading_fitness_metrics import compute_rolling_ith, optimal_tmaeg

# Get range bars (time-agnostic)
bars = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30", threshold=250)

# Build NAV from close prices
nav = (bars['Close'].pct_change().fillna(0) + 1).cumprod().values

# Compute rolling ITH features (all bounded [0, 1])
# TMAEG is automatically calculated from data volatility
features = compute_rolling_ith(nav, lookback=100)

# To inspect the auto-calculated TMAEG:
auto_tmaeg = optimal_tmaeg(nav, lookback=100)
print(f"Auto-TMAEG: {auto_tmaeg:.4f}")

# Add as DataFrame columns for LSTM training
bars['bull_epoch_density'] = features.bull_epoch_density
bars['bear_epoch_density'] = features.bear_epoch_density
bars['max_drawdown'] = features.max_drawdown

# Ready for LSTM - all features bounded [0, 1]
X = bars[['bull_epoch_density', 'bear_epoch_density', 'max_drawdown']].values
```

---

## BiLSTM Integration

All metrics produce bounded [0, 1] outputs for direct use in LSTM/BiLSTM models.

### Normalization Transforms

| Metric                | Raw Range  | Transform     | Output |
| --------------------- | ---------- | ------------- | ------ |
| Permutation Entropy   | [0, 1]     | None          | [0, 1] |
| Sample Entropy        | [0, ∞)     | `1-exp(-x)`   | [0, 1) |
| Shannon Entropy       | [0, log n] | `/log(n)`     | [0, 1] |
| Omega Ratio           | [0, ∞)     | `Ω/(1+Ω)`     | [0, 1) |
| Ulcer Index           | [0, 1]     | None          | [0, 1] |
| Garman-Klass Vol      | [0, ∞)     | `tanh(10x)`   | [0, 1) |
| Garman-Klass (stream) | [0, ∞)     | `EMA sigmoid` | (0, 1) |
| Kaufman ER            | [0, 1]     | None          | [0, 1] |
| Hurst Exponent        | [0, 1+]    | `soft_clamp`  | [0, 1] |
| Fractal Dimension     | [1, 2]     | `D-1`         | [0, 1] |

## Minimum Sample Requirements

| Metric                    | Minimum | Rationale            |
| ------------------------- | ------- | -------------------- |
| Permutation Entropy (m=3) | 60      | 10 × 3!              |
| Sample Entropy (m=2)      | 200     | Literature consensus |
| Shannon Entropy (n=10)    | 100     | 10 × n_bins          |
| Hurst Exponent            | 256     | DFA scale range      |

## Adaptive Utilities

The `adaptive.rs` module provides magic-number-free utilities:

- `optimal_tmaeg(nav, lookback)` - Auto-calculate TMAEG from data volatility
- `relative_epsilon(operand)` - Adaptive division guards
- `OnlineNormalizer` - Welford algorithm with sigmoid
- `GarmanKlassNormalizer` - EMA-based volatility normalization
- `hurst_soft_clamp()` - Bounded output for Hurst
- `adaptive_windows()` - Log-spaced window generation
- `MinimumSamples` - Statistical power requirements

## Rust Example

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
├── pyproject.toml          # Maturin build config
├── CLAUDE.md
├── src/
│   ├── lib.rs              # Module declarations + re-exports
│   ├── adaptive.rs         # Adaptive utilities (foundation)
│   ├── entropy.rs          # Permutation, Sample, Shannon
│   ├── risk.rs             # Omega, Ulcer, GK Vol, Kaufman ER
│   ├── fractal.rs          # Hurst, Fractal Dimension
│   ├── ith.rs              # Bull/Bear ITH analysis
│   ├── ith_rolling.rs      # Rolling window ITH features
│   ├── ith_normalize.rs    # Bounded normalization utilities
│   ├── nav.rs              # NAV construction
│   ├── types.rs            # Result types
│   └── python.rs           # PyO3 bindings
└── python/
    └── trading_fitness_metrics/
        ├── __init__.py     # Re-exports from _core
        ├── __init__.pyi    # Type stubs
        └── py.typed        # PEP 561 marker
```

## Test Coverage

134 tests covering:

- Edge cases (empty, single point, constant series)
- Boundary validation (outputs in declared ranges)
- Known values (pre-computed references)
- Property tests (monotonicity, scale invariance)
- Real market data (BTCUSDT aggTrades, custom NAV)

## Related Documentation

- **Validation Report**: [VALIDATION_REPORT.md](VALIDATION_REPORT.md)
- **Root Overview**: [← trading-fitness](../../CLAUDE.md)
- **ITH Concept**: [docs/ITH.md](../../docs/ITH.md)
