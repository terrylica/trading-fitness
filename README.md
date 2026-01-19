# Trading Fitness

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](packages/ith-python)
[![Rust](https://img.shields.io/badge/Rust-stable-DEA584?logo=rust&logoColor=white)](packages/core-rust)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6?logo=typescript&logoColor=white)](packages/core-bun)
[![Tests](https://img.shields.io/badge/tests-146%20passing-brightgreen)](.)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Polyglot monorepo for trading strategy fitness analysis using **ITH (Investment Time Horizon)** methodology.

## Overview

ITH analysis evaluates trading strategy fitness by measuring how consistently a strategy generates excess returns above a drawdown-adjusted threshold (TMAEG). This provides a more nuanced view of strategy quality than simple Sharpe ratios.

```
data/nav_data_custom/*.csv  ──▶  [Analysis Engine]  ──▶  artifacts/results.html
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
              Python+Numba          Rust            TypeScript
               (primary)         (compute)          (APIs)
```

## Quick Start

```bash
# Prerequisites: mise (https://mise.jdx.dev)
brew install mise

# Clone and setup
git clone https://github.com/terrylica/trading-fitness.git
cd trading-fitness
mise install

# Run ITH analysis
mise run analyze

# View results
open artifacts/results.html
```

## Packages

| Package                                 | Language       | Tests | Purpose                                    |
| --------------------------------------- | -------------- | ----- | ------------------------------------------ |
| [`ith-python`](packages/ith-python)     | Python + Numba | 100   | Primary ITH analysis with JIT acceleration |
| [`core-rust`](packages/core-rust)       | Rust           | 14    | Native performance-critical computations   |
| [`core-bun`](packages/core-bun)         | TypeScript/Bun | 32    | Async I/O, APIs, web integrations          |
| [`shared-types`](packages/shared-types) | JSON Schema    | —     | Cross-language type definitions            |

## Performance

Benchmarked on 1M data points (Apple M-series):

| Implementation | ITH Analysis | Total  | vs Baseline |
| -------------- | ------------ | ------ | ----------- |
| Python + Numba | 5.5 ms       | 7.7 ms | baseline    |
| Rust (native)  | 4.0 ms       | 7.3 ms | 1.1x faster |
| Bun/TypeScript | 10.3 ms      | 24 ms  | 3x slower   |

> Numba JIT compiles to LLVM, making Python competitive with native Rust for numerical workloads.

## Input Format

Place CSV files with `Date` and `NAV` columns in `data/nav_data_custom/`:

```csv
Date,NAV
2024-01-01,100.00
2024-01-02,100.50
2024-01-03,99.80
```

## Tasks

```bash
mise run analyze          # Run ITH analysis
mise run test             # Run all tests (146 total)
mise run lint             # Lint all packages
mise run affected         # List affected packages
mise run generate-types   # Generate types from JSON Schema

# Release workflow
mise run release:preflight  # Validate prerequisites (clean tree, GH_TOKEN, main branch)
mise run release:version    # Run semantic-release (updates CHANGELOG, tags)
mise run release:full       # Complete workflow (preflight + version)
```

## Development

```bash
# Python package
cd packages/ith-python && uv run pytest

# Rust package
cd packages/core-rust && cargo test

# TypeScript package
cd packages/core-bun && bun test
```

## Architecture

```
trading-fitness/
├── .claude/skills/       # Claude Code automation
├── packages/
│   ├── ith-python/       # Primary analysis engine
│   ├── core-rust/        # Native compute library
│   ├── core-bun/         # TypeScript/Bun package
│   └── shared-types/     # JSON Schema definitions
├── rules/                # ast-grep lint rules
├── scripts/              # Automation & benchmarks
└── data/                 # Input NAV data
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — System design and data flow
- [ITH Methodology](docs/ITH.md) — Investment Time Horizon explained
- [Logging Contract](docs/LOGGING.md) — NDJSON structured logging

## ITH Algorithm Specification

### Core Concepts

ITH (Investment Time Horizon) analysis evaluates strategy fitness for both **long** (bull) and **short** (bear) positions using mathematically symmetric algorithms.

| Concept                | Bull (Long)                                   | Bear (Short)                                   |
| ---------------------- | --------------------------------------------- | ---------------------------------------------- |
| **Threshold**          | TMAEG (Target Maximum Acceptable Excess Gain) | TMAER (Target Maximum Acceptable Excess Runup) |
| **Adverse Movement**   | Drawdown (price ↓)                            | Runup (price ↑)                                |
| **Favorable Movement** | Rally (price ↑)                               | Decline (price ↓)                              |
| **Reference Point**    | `endorsing_crest` (confirmed peak)            | `endorsing_trough` (confirmed valley)          |
| **Default Hurdle**     | 0.05 (5%) or `max_drawdown(nav)`              | 0.05 (5%) or `max_runup(nav)`                  |

### Mathematical Formulas

#### Drawdown and Runup (Bounded \[0, 1))

```
Bull Max Drawdown:  max_dd  = max(1 - nav / cummax(nav))     # Loss from peak
Bear Max Runup:     max_ru  = max(1 - cummin(nav) / nav)     # Gain from trough (adverse for shorts)
```

#### Excess Gain/Loss Calculations

```
Bull Excess Gain:   excess_gain  = nav[i] / endorsing_crest - 1      # Unbounded [0, ∞)
Bull Excess Loss:   excess_loss  = 1 - nav[i] / endorsing_crest      # Bounded [0, 1)

Bear Excess Gain:   excess_gain  = endorsing_trough / nav[i] - 1     # Unbounded [0, ∞)
Bear Excess Loss:   excess_loss  = 1 - endorsing_trough / nav[i]     # Bounded [0, 1)
```

#### Epoch Detection Condition

```
epoch[i] = (excess_gains[i] > excess_losses[i]) AND (excess_gains[i] > hurdle)
```

### Algorithm Pseudocode

```python
# Bull ITH Algorithm (bear inverts crest↔trough, max↔min, >↔<)
# Reference: bull_ith_numba.py:36-143

def bull_excess_gain_excess_loss(nav: array, hurdle: float) -> Result:
    n = len(nav)
    excess_gains = zeros(n)
    excess_losses = zeros(n)
    bull_epochs = zeros(n, bool)

    # State variables
    endorsing_crest = nav[0]      # Confirmed HIGH we measure FROM
    endorsing_nadir = nav[0]      # Confirmed LOW after last crest
    candidate_crest = nav[0]      # Potential new HIGH (favorable)
    candidate_nadir = nav[0]      # Potential new LOW (adverse drawdown)
    excess_gain = 0.0
    excess_loss = 0.0

    for i in range(1, n):
        equity = nav[i]

        # Track new HIGH (favorable for longs)
        if equity > candidate_crest:
            excess_gain = equity / endorsing_crest - 1
            candidate_crest = equity

        # Track new LOW (adverse drawdown)
        if equity < candidate_nadir:
            excess_loss = 1 - equity / endorsing_crest
            candidate_nadir = equity

        # Reset condition: lock in new reference when profitable
        if (excess_gain > abs(excess_loss) AND
            excess_gain > hurdle AND
            candidate_crest >= endorsing_crest):

            endorsing_crest = candidate_crest
            endorsing_nadir = equity
            candidate_nadir = equity
            excess_gain = 0.0
            excess_loss = 0.0

        excess_gains[i] = excess_gain
        excess_losses[i] = excess_loss

        # Epoch detection
        bull_epochs[i] = (excess_gains[i] > excess_losses[i] AND
                         excess_gains[i] > hurdle)

    # Calculate interval CV (consistency metric)
    epoch_indices = where(bull_epochs)
    intervals = diff(epoch_indices)
    intervals_cv = std(intervals) / mean(intervals) if len(intervals) > 0 else NaN

    return Result(excess_gains, excess_losses, count(bull_epochs),
                  bull_epochs, intervals_cv)
```

### Variable Name Mapping

| Variable            | Type            | Bull Meaning                     | Bear Meaning                      |
| ------------------- | --------------- | -------------------------------- | --------------------------------- |
| `hurdle`            | `float`         | TMAEG threshold                  | TMAER threshold                   |
| `endorsing_crest`   | `float`         | Confirmed peak (HIGH)            | —                                 |
| `endorsing_trough`  | `float`         | —                                | Confirmed valley (LOW)            |
| `candidate_crest`   | `float`         | Potential new high               | —                                 |
| `candidate_trough`  | `float`         | —                                | Potential new low                 |
| `candidate_nadir`   | `float`         | Potential drawdown low           | —                                 |
| `candidate_peak`    | `float`         | —                                | Potential runup high              |
| `excess_gains`      | `ndarray`       | Per-point gains from rallies     | Per-point gains from declines     |
| `excess_losses`     | `ndarray`       | Per-point losses from drawdowns  | Per-point losses from runups      |
| `bull_epochs`       | `ndarray[bool]` | Points where long exceeds hurdle | —                                 |
| `bear_epochs`       | `ndarray[bool]` | —                                | Points where short exceeds hurdle |
| `bull_intervals_cv` | `float`         | CV of epoch spacing (long)       | —                                 |
| `bear_intervals_cv` | `float`         | —                                | CV of epoch spacing (short)       |

### Result Structure

```python
class BullExcessGainLossResult(NamedTuple):
    excess_gains: ndarray       # Gains at each point
    excess_losses: ndarray      # Losses at each point
    num_of_bull_epochs: int     # Total epoch count
    bull_epochs: ndarray[bool]  # Boolean mask of epochs
    bull_intervals_cv: float    # Coefficient of variation

class BearExcessGainLossResult(NamedTuple):
    excess_gains: ndarray       # Gains at each point (from declines)
    excess_losses: ndarray      # Losses at each point (from runups)
    num_of_bear_epochs: int     # Total epoch count
    bear_epochs: ndarray[bool]  # Boolean mask of epochs
    bear_intervals_cv: float    # Coefficient of variation
```

### Fitness Qualification Criteria

| Metric                | Bull Bounds            | Bear Bounds            | Purpose                 |
| --------------------- | ---------------------- | ---------------------- | ----------------------- |
| Sharpe Ratio          | 0.5 < SR < 9.9         | -9.9 < SR < -0.5       | Risk-adjusted returns   |
| Epoch Count           | > `ceil(points/168)`   | > `ceil(points/168)`   | Minimum opportunities   |
| Aggregate CV          | 0.0 < CV < 0.70        | 0.0 < CV < 0.70        | Epoch consistency       |
| P2E (Points to Epoch) | `points / epoch_count` | `points / epoch_count` | Average epoch frequency |

### Symmetry Proof

The Bull and Bear algorithms are **exact mathematical inverses**:

```
Bull: tracks HIGHS, gains from UP,   losses from DOWN
Bear: tracks LOWS,  gains from DOWN, losses from UP

Bull excess_gain  = nav / crest - 1     ←→  Bear excess_gain  = trough / nav - 1
Bull excess_loss  = 1 - nav / crest     ←→  Bear excess_loss  = 1 - trough / nav
Bull running_max  = cummax(nav)         ←→  Bear running_min  = cummin(nav)
Bull new_high     = nav > candidate     ←→  Bear new_low      = nav < candidate
```

## Universal Applicability

The ITH algorithm is **timeframe-agnostic** — it operates purely on sequential data points, making it suitable for feature engineering across any data frequency:

| Data Type    | Example             | Sharpe `periods_per_year` |
| ------------ | ------------------- | ------------------------- |
| Daily equity | SPY daily closes    | 252                       |
| Daily crypto | BTC daily closes    | 365                       |
| Hourly       | 4H candles          | 2190 (365×6)              |
| Range bars   | 50-point range bars | Estimate bars/year        |
| Tick data    | Trade ticks         | Estimate ticks/year       |

### Time-Agnostic API

```python
from ith_python.ith import sharpe_ratio, SyntheticNavParams

# Explicit periods (recommended for any frequency)
sr = sharpe_ratio(returns, periods_per_year=252)  # Daily equity
sr = sharpe_ratio(returns, periods_per_year=500)  # Custom range bars

# Point-based synthetic data generation
params = SyntheticNavParams(n_points=1000)  # Generate 1000 points
```

The core algorithm computes:

- **Excess gains/losses**: Ratios relative to reference points (time-independent)
- **Epoch detection**: Boolean condition on gains vs losses (time-independent)
- **Interval CV**: Point-count spacing between epochs (time-independent)
- **P2E (Points to Epoch)**: Average points between epochs (time-independent)

## License

MIT
