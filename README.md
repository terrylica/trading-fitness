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

## Key Concepts

- **TMAEG** (Target Maximum Acceptable Excess Gain): Drawdown-based threshold for ITH epochs
- **ITH Epochs**: Periods where strategy exceeds the TMAEG threshold
- **ITH Intervals CV**: Coefficient of variation measuring epoch consistency

## License

MIT
