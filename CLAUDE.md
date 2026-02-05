# Trading Fitness

> Polyglot monorepo for time-agnostic trading strategy fitness analysis using ITH (Investment Time Horizon) methodology.

## Navigation

| Section                                     | Purpose                     |
| ------------------------------------------- | --------------------------- |
| [Quick Reference](#quick-reference)         | Common commands             |
| [Package Map](#package-map)                 | Monorepo structure          |
| [Documentation](#documentation)             | Deep-dive docs              |
| [Key Concepts](#key-concepts)               | ITH, terminology, warmup    |
| [Development](#development)                 | Build, test, release        |
| [Data Infrastructure](#data-infrastructure) | Bigblack storage (critical) |

---

## Quick Reference

| Action                | Command                             |
| --------------------- | ----------------------------------- |
| Full pipeline         | `mise run forensic:full-pipeline`   |
| Data preflight        | `mise run preflight:rangebar-cache` |
| Precompute range bars | `mise run data:precompute-parallel` |
| Run tests             | `mise run test`                     |
| Pre-release check     | `mise run validate:pre-release`     |
| Symmetric dogfooding  | `mise run validate:symmetric`       |

---

## Package Map

| Package                                         | Language    | Status      | Purpose                           |
| ----------------------------------------------- | ----------- | ----------- | --------------------------------- |
| [ith-python](packages/ith-python/CLAUDE.md)     | Python      | **PRIMARY** | ITH fitness analysis              |
| [metrics-rust](packages/metrics-rust/CLAUDE.md) | Rust + PyO3 | Active      | Multi-scale ITH + Python bindings |
| [core-rust](packages/core-rust/CLAUDE.md)       | Rust        | Active      | Performance-critical compute      |
| [core-bun](packages/core-bun/CLAUDE.md)         | Bun/TS      | Active      | Async I/O, APIs                   |
| [shared-types](packages/shared-types/CLAUDE.md) | Multi       | Active      | JSON Schema type definitions      |

### Services (Future)

| Service                                                   | Status      |
| --------------------------------------------------------- | ----------- |
| [data-ingestion](services/data-ingestion/CLAUDE.md)       | Placeholder |
| [strategy-engine](services/strategy-engine/CLAUDE.md)     | Placeholder |
| [execution-gateway](services/execution-gateway/CLAUDE.md) | Placeholder |

---

## Documentation

### Core Methodology

| Document                                               | Purpose                            |
| ------------------------------------------------------ | ---------------------------------- |
| [docs/ITH.md](docs/ITH.md)                             | ITH algorithm and fitness criteria |
| [docs/features/REGISTRY.md](docs/features/REGISTRY.md) | Feature definitions (SSoT)         |

### Infrastructure

| Document                                                   | Purpose                       |
| ---------------------------------------------------------- | ----------------------------- |
| [docs/infrastructure/DATA.md](docs/infrastructure/DATA.md) | Bigblack data storage (SSoT)  |
| [config/forensic.toml](config/forensic.toml)               | Pipeline configuration (SSoT) |

### Forensic Analysis

| Document                                     | Purpose                    |
| -------------------------------------------- | -------------------------- |
| [docs/forensic/E2E.md](docs/forensic/E2E.md) | E2E pipeline documentation |
| [docs/LOGGING.md](docs/LOGGING.md)           | NDJSON telemetry contract  |

### Architecture & Planning

| Document                                                                                                                       | Purpose                       |
| ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------- |
| [docs/plans/2026-01-25-multi-view-feature-architecture-plan.md](docs/plans/2026-01-25-multi-view-feature-architecture-plan.md) | 3-layer architecture          |
| [docs/adr/](docs/adr/)                                                                                                         | Architecture Decision Records |

### Research & Compliance

| Document                         | Purpose                     |
| -------------------------------- | --------------------------- |
| [docs/SRED.md](docs/SRED.md)     | CRA tax credit tracking     |
| [docs/research/](docs/research/) | Statistical method research |

---

## Key Concepts

### ITH (Investment Time Horizon)

ITH is a **time-agnostic** fitness metric that counts threshold crossings rather than calendar time.

```
NAV Series → Rolling Window → Count Epochs → Normalize [0,1] → BiLSTM Features
```

**Key insight**: TMAEG (threshold) is **auto-calculated** from data volatility. The `threshold_dbps` parameter is for column naming only.

**Deep Dive**: [docs/ITH.md](docs/ITH.md)

### Terminology

| Term       | Definition                                                |
| ---------- | --------------------------------------------------------- |
| **ITH**    | Investment Time Horizon - epoch count metric              |
| **TMAEG**  | Target Maximum Acceptable Excess Gain (auto-calculated)   |
| **NAV**    | Net Asset Value - normalized price series starting at 1.0 |
| **Epoch**  | Period where excess gain exceeds TMAEG                    |
| **dbps**   | Decimal basis points (1 dbps = 0.0001 = 0.01%)            |
| **Warmup** | Initial bars with NaN (max_lookback - 1 bars)             |

**Full glossary**: [docs/features/REGISTRY.md](docs/features/REGISTRY.md)

### Feature Selection & Statistical Methods

Methods for principled feature selection under high autocorrelation (ACF=0.94) and OOD robustness requirements.

| Term                                               | Acronym   | Definition                                                                                                                                            |
| -------------------------------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Autocorrelation Function**                       | ACF       | Correlation of a time series with lagged versions of itself. ACF=0.94 means extreme temporal dependence.                                              |
| **Effective Sample Size**                          | ESS       | True independent information in autocorrelated data: `N_eff ≈ N × (1-ACF)/(1+ACF)`. For ACF=0.94, N_eff ≈ 0.03×N.                                     |
| **Out-of-Distribution**                            | OOD       | Data from different regime than training (e.g., bull vs bear market). OOD robustness = generalization across regimes.                                 |
| **False Discovery Rate**                           | FDR       | Expected proportion of false positives among selected features. FDR=0.2 means 20% of selections may be spurious.                                      |
| **Peter-Clark Momentary Conditional Independence** | PCMCI     | Gold standard causal discovery for time series. Explicitly handles autocorrelation via MCI test. Package: `tigramite`.                                |
| **Hilbert-Schmidt Independence Criterion**         | HSIC      | Kernel-based dependence measure capturing all nonlinear relationships. HSIC Lasso: feature selection with redundancy penalty. Package: `pyHSICLasso`. |
| **Minimum Redundancy Maximum Relevance**           | mRMR      | Fast filter selecting features with high target relevance and low inter-feature redundancy. Package: `mrmr-selection`.                                |
| **Distance Correlation**                           | dCor      | Parameter-free nonlinear dependence measure. dCor=0 iff independent (unlike Pearson r). Package: `dcor`.                                              |
| **Time Series Knockoffs Inference**                | TSKI      | FDR-controlled feature selection using block sampling for temporal data. Package: `knockpy`.                                                          |
| **Invariant Risk Minimization**                    | IRM       | Framework selecting features with stable relationships across environments/regimes.                                                                   |
| **Sparse Invariant Risk Minimization**             | SparseIRM | IRM with sparsity constraint for simultaneous feature selection and OOD robustness.                                                                   |
| **Scattering Transform**                           | -         | Multi-scale wavelet transform extracting features at ALL temporal scales automatically. Sidesteps lookback selection. Package: `kymatio`.             |
| **Random Convolutional Kernel Transform**          | ROCKET    | Generates thousands of random convolutional features with varying receptive fields. Package: `sktime`.                                                |
| **Sequential Feature Detachment**                  | SFD       | Pruning algorithm for ROCKET selecting minimal predictive kernel set. Package: `detach_rocket`.                                                       |
| **Block Bootstrap**                                | -         | Resampling preserving temporal structure by shuffling contiguous blocks. Required for ACF>0.5 data. Package: `tsbootstrap`.                           |
| **Circular Block Bootstrap**                       | CBB       | Block bootstrap variant wrapping around series end for stability selection.                                                                           |
| **Walk-Forward Importance Stability**              | -         | Validation method: compute feature importance per fold, select features with low CV across folds.                                                     |
| **TimeSeriesSplit**                                | -         | Cross-validation respecting temporal order. NEVER use random k-fold on time series.                                                                   |
| **Participation Ratio**                            | PR        | Effective dimensionality from PCA eigenvalues: `PR = (Σλ)² / Σλ²`. PR=10 means 10 effective dimensions.                                               |
| **Variance Inflation Factor**                      | VIF       | Multicollinearity measure. VIF>10 indicates high redundancy. Use Ridge-regularized VIF for stability.                                                 |
| **Feature Suppression Registry**                   | -         | Config pattern: keep features but mark as suppressed (don't use) with reason and date. Like deprecation for features.                                 |
| **Anchor Regression**                              | -         | OOD method using exogenous anchors (regime labels) for distributional robustness. Single hyperparameter γ.                                            |
| **Change Point Detection**                         | CPD       | Algorithms (PELT, BinSeg) detecting regime boundaries for environment definition. Package: `ruptures`.                                                |

**Research**: [docs/research/external/](docs/research/external/) | GitHub Issue: [cc-skills#21](https://github.com/terrylica/cc-skills/issues/21)

### Warmup Handling

Each lookback window requires `(lookback - 1)` bars before producing valid values.

| Lookback | First Valid Bar |
| -------- | --------------- |
| lb20     | bar_index = 19  |
| lb100    | bar_index = 99  |
| lb500    | bar_index = 499 |

**Preflight**: `mise run preflight:warmup`

---

## Development

### Quick Start

```bash
# Python (ith-python)
cd packages/ith-python && uv sync && uv run pytest

# Rust (metrics-rust)
cargo nextest run -p trading-fitness-metrics
mise run develop:metrics-rust    # Build + install Python bindings
```

### mise Task Orchestration

Uses [jdx/mise](https://mise.jdx.dev/) for DAG-based task orchestration.

```
mise.toml              # Root: [tools] + [env] + orchestration
.mise.local.toml       # Secrets (gitignored)
packages/*/mise.toml   # Package-specific tasks
```

**Key patterns**:

- `depends = [...]` for task dependencies
- `sources`/`outputs` for incremental builds
- `mise watch <task>` for live reload

### Git Workflow

Uses git-town for branch automation:

```bash
git town hack     # Create feature branch
git town sync     # Sync with main
git town ship     # Create PR and merge
```

---

## Data Infrastructure

> **CRITICAL**: Bigblack is the sole data storage. Local machines should NOT cache range bar data.

```
┌─────────────────────┐         ┌─────────────────────────────────────┐
│   Local (macOS)     │  SSH    │   Bigblack (Linux GPU Workstation)  │
│   Code + Analysis   │ ──────▶ │   ClickHouse + Tick Cache           │
└─────────────────────┘         └─────────────────────────────────────┘
```

| Component     | Location                      | Size      |
| ------------- | ----------------------------- | --------- |
| ClickHouse DB | `bigblack:rangebar_cache`     | 86M+ bars |
| Tick Cache    | `bigblack:~/.cache/rangebar/` | Parquet   |

### Timestamp Precision by Year (GOTCHA)

Binance tick data changed format between years:

| Year      | dtype            | Conversion to ms    |
| --------- | ---------------- | ------------------- |
| 2022-2023 | `datetime64[ms]` | Use as-is           |
| 2024+     | `datetime64[ns]` | Divide by 1,000,000 |

**Symptom if wrong**: Timestamps show year 52000+ or 1970.

```python
# Always check dtype before converting
ts_dtype = str(df["timestamp"].dtype)
if "ns" in ts_dtype:
    df["timestamp_ms"] = df["timestamp"].astype("int64") // 1_000_000
else:
    df["timestamp_ms"] = df["timestamp"].astype("int64")
```

### Ouroboros Mode (rangebar-py)

Range bars use `ouroboros="year"` for reproducible construction with yearly reset boundaries (Jan 1 00:00 UTC). Bars crossing year boundaries are marked as orphans.

```bash
# Precompute with Ouroboros year mode
uv run python scripts/precompute_ouroboros_year.py --workers 4
```

**ClickHouse table**: `rangebar_cache.range_bars_ouroboros_year`

**Exchange session columns** (opt-in via `include_exchange_sessions=True`):
`exchange_session_sydney`, `exchange_session_tokyo`, `exchange_session_london`, `exchange_session_newyork`

**Schema migrations**: `scripts/migrations/` (run on bigblack with `clickhouse-client --multiquery`)

### Common Commands

```bash
# Check data status
ssh bigblack "clickhouse-client --query 'SELECT symbol, count() FROM rangebar_cache.range_bars GROUP BY 1'"

# Sync code to bigblack
rsync -avz --exclude='.venv' --exclude='artifacts' --exclude='.git' . bigblack:~/eon/trading-fitness/
```

**Deep Dive**: [docs/infrastructure/DATA.md](docs/infrastructure/DATA.md)

---

## Directory Structure

```
trading-fitness/
├── packages/
│   ├── ith-python/        # PRIMARY: ITH fitness analysis
│   ├── metrics-rust/      # Rust ITH + PyO3 bindings
│   ├── core-rust/         # Performance-critical Rust
│   ├── core-bun/          # Async I/O (Bun/TS)
│   └── shared-types/      # JSON Schema definitions
├── services/              # Future: data-ingestion, strategy-engine
├── docs/                  # Documentation hub
│   ├── ITH.md             # Core methodology
│   ├── features/          # Feature registry
│   ├── forensic/          # E2E pipeline docs
│   ├── infrastructure/    # Data architecture
│   ├── plans/             # Implementation plans
│   └── adr/               # Architecture decisions
├── config/                # Configuration files
├── scripts/               # Utility scripts
├── artifacts/             # Generated outputs (gitignored)
└── logs/                  # NDJSON telemetry (gitignored)
```

---

_Polyglot Monorepo - Time-Agnostic Trading Strategy Fitness Analysis_
