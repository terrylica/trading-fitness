# Trading Fitness

> Polyglot monorepo for time-agnostic trading strategy fitness analysis using ITH (Investment Time Horizon) methodology.

## Quick Reference

| Action                | Command                             |
| --------------------- | ----------------------------------- |
| Full pipeline (new)   | `mise run forensic:full-pipeline`   |
| Legacy E2E pipeline   | `mise run forensic:pipeline`        |
| Data preflight        | `mise run preflight:rangebar-cache` |
| Precompute range bars | `mise run data:precompute-parallel` |
| Check data status     | `mise run data:cache-status`        |
| Run tests             | `mise run test`                     |
| **Pre-release check** | `mise run validate:pre-release`     |
| Symmetric dogfooding  | `mise run validate:symmetric`       |

---

## Data Infrastructure (CRITICAL)

**Bigblack is the sole data storage.** Local machines should NOT cache range bar data.

### Architecture

```
┌─────────────────────┐         ┌─────────────────────────────────────┐
│   Local (macOS)     │         │   Bigblack (Linux GPU Workstation)  │
│                     │  SSH    │                                     │
│  Code + Analysis    │ ──────▶ │  ClickHouse: rangebar_cache.range_bars
│  (no data caching)  │         │  Tick Cache: ~/.cache/rangebar/ticks/
└─────────────────────┘         └─────────────────────────────────────┘
```

| Component     | Location                      | Purpose                       |
| ------------- | ----------------------------- | ----------------------------- |
| ClickHouse DB | `bigblack:rangebar_cache`     | Range bar storage (86M+ bars) |
| Tick Cache    | `bigblack:~/.cache/rangebar/` | Parquet tick files (Binance)  |
| Code + Config | Local + `bigblack:~/eon/`     | Synced via rsync              |

### Data Commands

```bash
# Check bigblack data status
ssh bigblack "clickhouse-client --query 'SELECT symbol, threshold_decimal_bps, count() FROM rangebar_cache.range_bars GROUP BY 1,2 ORDER BY 1,2'"

# Run precompute on bigblack (parallel, 8 workers)
ssh bigblack "cd ~/eon/trading-fitness/packages/ith-python && ~/.local/bin/uv run python ../../scripts/precompute_historical_parallel.py --workers 8"

# Sync code to bigblack (excludes data/cache)
rsync -avz --exclude='.venv' --exclude='artifacts' --exclude='logs' --exclude='.git' . bigblack:~/eon/trading-fitness/
```

### Data Coverage (as of 2026-01-26)

| Symbol  | Thresholds (dbps) | Coverage  | Bars  |
| ------- | ----------------- | --------- | ----- |
| BTCUSDT | 50, 100           | 4.1 years | 6.4M  |
| ETHUSDT | 25                | 4.1 years | 16.6M |
| SOLUSDT | 25, 250           | 4.1 years | 35.8M |
| BNBUSDT | 50, 250           | 4.1 years | 4.9M  |

**Reference**: [docs/infrastructure/DATA.md](docs/infrastructure/DATA.md)

---

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
| Data Infrastructure | [docs/infrastructure/DATA.md](docs/infrastructure/DATA.md)                                                                     | Bigblack storage architecture (SSoT)     |
| Forensic Config     | [config/forensic.toml](config/forensic.toml)                                                                                   | Data pipeline configuration (SSoT)       |
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

## mise: DAG-Based Task Orchestration

Uses [jdx/mise](https://mise.jdx.dev/) with **DAG-based task orchestration**.

### Task Graph Patterns

| Pattern     | Example                               | Use Case           |
| ----------- | ------------------------------------- | ------------------ |
| Sequential  | `depends = ["lint", "test"]`          | Pipeline stages    |
| Parallel    | `depends = ["test:unit", "test:e2e"]` | Independent checks |
| Conditional | `run = "if [ -f x ]; then ...`        | Optional steps     |

### Environment Hierarchy

```
mise.toml          # Hub: [tools] + [env] + orchestration tasks
.mise.local.toml   # Secrets (gitignored): GH_TOKEN, API keys
packages/*/mise.toml  # Spoke: domain-specific execution tasks
```

### Secrets Isolation (CRITICAL)

**Never commit credentials to `mise.toml`** - use `.mise.local.toml`:

```toml
# .mise.local.toml (gitignored)
[env]
GH_TOKEN = "{{ read_file(path=env.HOME ~ '/.claude/.secrets/gh-token-terrylica') | trim }}"
GITHUB_TOKEN = "{{ env.GH_TOKEN }}"
```

### Key Patterns

- **Incremental**: `sources`/`outputs` arrays skip unchanged tasks
- **Watch mode**: `mise watch <task>` with `sources` for live reload
- **Profiles**: `mise.{env}.toml` for dev/staging/prod configs
- **Hooks**: `[hooks.enter]` for directory-activated setup

### Configuration Files

| File           | Purpose                                   |
| -------------- | ----------------------------------------- |
| `mise.toml`    | Root orchestration + delegation           |
| `.mcp.json`    | MCP servers (mise, code-search, ast-grep) |
| `sgconfig.yml` | ast-grep rules configuration              |

**Reference**: [mise Tasks](https://mise.jdx.dev/tasks/) | [mise Configuration](https://mise.jdx.dev/configuration.html)

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
