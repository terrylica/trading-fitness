# Trading Fitness

> Polyglot monorepo for time-agnostic trading strategy fitness analysis using ITH (Investment Time Horizon) methodology.

## Quick Reference

| Action            | Command                      |
| ----------------- | ---------------------------- |
| Full E2E pipeline | `mise run forensic:pipeline` |
| Run ITH analysis  | `mise run analyze`           |
| Run tests         | `mise run test`              |
| Preflight checks  | `mise run preflight:warmup`  |

## Package Map

| Package                                         | Language      | Purpose                         | Docs                                 |
| ----------------------------------------------- | ------------- | ------------------------------- | ------------------------------------ |
| [ith-python](packages/ith-python/CLAUDE.md)     | Python        | ITH fitness analysis (PRIMARY)  | [→](packages/ith-python/CLAUDE.md)   |
| [metrics-rust](packages/metrics-rust/CLAUDE.md) | Rust + Python | Multi-scale ITH + PyO3 bindings | [→](packages/metrics-rust/CLAUDE.md) |
| [core-rust](packages/core-rust/CLAUDE.md)       | Rust          | Performance-critical compute    | [→](packages/core-rust/CLAUDE.md)    |
| [core-bun](packages/core-bun/CLAUDE.md)         | Bun/TS        | Async I/O, APIs                 | [→](packages/core-bun/CLAUDE.md)     |
| [shared-types](packages/shared-types/CLAUDE.md) | Multi         | JSON Schema type definitions    | [→](packages/shared-types/CLAUDE.md) |

## Documentation Hub

| Topic               | Location                                                                                                                       | Purpose                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------- |
| ITH Methodology     | [docs/ITH.md](docs/ITH.md)                                                                                                     | Core algorithm and fitness criteria      |
| Feature Registry    | [docs/features/REGISTRY.md](docs/features/REGISTRY.md)                                                                         | All extractable features (SSoT)          |
| Logging Contract    | [docs/LOGGING.md](docs/LOGGING.md)                                                                                             | NDJSON telemetry format                  |
| Forensic Analysis   | [docs/forensic/E2E.md](docs/forensic/E2E.md)                                                                                   | E2E pipeline and artifact interpretation |
| Architecture Plan   | [docs/plans/2026-01-25-multi-view-feature-architecture-plan.md](docs/plans/2026-01-25-multi-view-feature-architecture-plan.md) | 3-layer separation of concerns           |
| SR&ED Documentation | [docs/SRED.md](docs/SRED.md)                                                                                                   | CRA tax credit tracking                  |

---

## Architecture: Multi-View Feature Pipeline

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Layer 1: Compute │───▶│ Layer 2: Storage │───▶│ Layer 3: Analysis│
│   features/      │    │   storage/       │    │   analysis/      │
│                  │    │                  │    │                  │
│ Rust multiscale  │    │ Long Format SSoT │    │ Statistical eval │
│ ITH features     │    │ + View Generators│    │ + NDJSON emit    │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

| Layer | Module                | Upgrade Command             |
| ----- | --------------------- | --------------------------- |
| 1     | `ith_python.features` | `mise run upgrade:features` |
| 2     | `ith_python.storage`  | `mise run upgrade:views`    |
| 3     | `ith_python.analysis` | `mise run upgrade:analysis` |

**Key classes**: `FeatureConfig`, `FeatureStore`, `AnalysisConfig`, `AnalysisResults`

**Deep Dive**: [packages/ith-python/CLAUDE.md](packages/ith-python/CLAUDE.md)

---

## ITH Concept (Essential)

ITH (Investment Time Horizon) is a **time-agnostic** fitness metric that counts threshold crossings rather than calendar time.

```
NAV Series → Rolling Window → Count Epochs → Normalize [0,1] → BiLSTM Features
```

**Key insight**: TMAEG (threshold) is **auto-calculated** from data volatility using MAD-based estimation. The `threshold_dbps` parameter is for **column naming only**, not computation.

**Deep Dive**: [docs/ITH.md](docs/ITH.md)

---

## Terminology

| Term       | Definition                                                | Range        |
| ---------- | --------------------------------------------------------- | ------------ |
| **ITH**    | Investment Time Horizon - epoch count metric              | epochs       |
| **TMAEG**  | Target Maximum Acceptable Excess Gain (auto-calculated)   | [0.001, 0.5] |
| **NAV**    | Net Asset Value - normalized price series starting at 1.0 | ratio        |
| **Epoch**  | Period where excess gain exceeds TMAEG, triggering reset  | count        |
| **dbps**   | Decimal basis points (1 dbps = 0.0001 = 0.01%)            | unit         |
| **Warmup** | Initial bars with NaN values (max_lookback - 1 bars)      | bars         |

### Feature Short Names (8 per lookback)

| Full Name          | Short   | Range | Description                   |
| ------------------ | ------- | ----- | ----------------------------- |
| bull_epoch_density | bull_ed | [0,1] | Normalized bull epoch count   |
| bear_epoch_density | bear_ed | [0,1] | Normalized bear epoch count   |
| bull_excess_gain   | bull_eg | [0,1) | tanh-normalized excess gain   |
| bear_excess_gain   | bear_eg | [0,1) | tanh-normalized excess gain   |
| bull_cv            | bull_cv | (0,1) | Intervals coefficient of var. |
| bear_cv            | bear_cv | (0,1) | Intervals coefficient of var. |
| max_drawdown       | max_dd  | [0,1] | Maximum drawdown in window    |
| max_runup          | max_ru  | [0,1] | Maximum runup in window       |

---

## Warmup Handling

Each lookback window requires `(lookback - 1)` bars of history before producing valid values.

| Lookback | Warmup Bars | First Valid Bar |
| -------- | ----------- | --------------- |
| lb20     | 19          | bar_index = 19  |
| lb50     | 49          | bar_index = 49  |
| lb100    | 99          | bar_index = 99  |
| lb200    | 199         | bar_index = 199 |
| lb500    | 499         | bar_index = 499 |

**Preflight check**: `mise run preflight:warmup` validates data sufficiency.

**In code**:

```python
from ith_python.storage import validate_warmup, get_warmup_bars

is_valid, info = validate_warmup(n_bars=2000, lookbacks=[20,50,100,200,500])
# info["warmup_bars"] = 499, info["valid_bars"] = 1501
```

---

## Development

### Python (ith-python)

```bash
cd packages/ith-python
uv sync                           # Install dependencies
uv run pytest                     # Run tests
```

### Rust + Python (metrics-rust)

```bash
cargo nextest run -p trading-fitness-metrics   # Rust tests
mise run develop:metrics-rust                  # Build + install into venv
```

### Forensic Pipeline

```bash
mise run forensic:pipeline         # Full E2E (new architecture)
mise run forensic:e2e              # Legacy E2E with ClickHouse data
mise run forensic:hypothesis-audit # Audit hypothesis test results
```

---

## Configuration

| File           | Purpose                                         |
| -------------- | ----------------------------------------------- |
| `mise.toml`    | Runtime versions, tasks, `UV_PYTHON=python3.13` |
| `.mcp.json`    | MCP servers (mise, code-search, ast-grep)       |
| `sgconfig.yml` | ast-grep rules configuration                    |

---

## Directory Structure

```
trading-fitness/
├── packages/
│   ├── ith-python/          # Primary Python package
│   │   └── src/ith_python/
│   │       ├── features/          # Layer 1: Feature computation
│   │       ├── storage/           # Layer 2: FeatureStore + views
│   │       ├── analysis/          # Layer 3: Statistical analysis
│   │       ├── telemetry/         # Provenance tracking
│   │       └── statistical_examination/  # ML readiness (legacy)
│   ├── metrics-rust/        # Rust ITH + PyO3 bindings
│   ├── core-rust/           # Performance-critical Rust
│   ├── core-bun/            # Async I/O (Bun/TS)
│   └── shared-types/        # JSON Schema definitions
├── docs/
│   ├── ITH.md               # Core methodology
│   ├── LOGGING.md           # Telemetry contract
│   ├── features/REGISTRY.md # Feature SSoT
│   ├── forensic/E2E.md      # Pipeline documentation
│   └── plans/               # Implementation plans
├── artifacts/               # Generated outputs (gitignored)
└── logs/ndjson/             # NDJSON telemetry logs
```

---

## Git Workflow

Uses git-town for branch automation:

```bash
git town hack     # Create feature branch
git town sync     # Sync with main
git town ship     # Create PR and merge
```

---

_Polyglot Monorepo - Time-Agnostic Trading Strategy Fitness Analysis_
