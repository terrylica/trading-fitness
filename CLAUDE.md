# Trading Fitness

> Full polyglot monorepo for trading strategy fitness analysis.

## Quick Reference

| Action           | Command             |
| ---------------- | ------------------- |
| Run ITH analysis | `mise run analyze`  |
| Run tests        | `mise run test`     |
| Lint             | `mise run lint`     |
| List affected    | `mise run affected` |

## Package Map

| Package                                         | Language      | Purpose                           | Tests | Docs                                 |
| ----------------------------------------------- | ------------- | --------------------------------- | ----- | ------------------------------------ |
| [ith-python](packages/ith-python/CLAUDE.md)     | Python        | ITH fitness analysis (PRIMARY)    | 100   | [→](packages/ith-python/CLAUDE.md)   |
| [metrics-rust](packages/metrics-rust/CLAUDE.md) | Rust + Python | BiLSTM metrics with PyO3 bindings | 95    | [→](packages/metrics-rust/CLAUDE.md) |
| [core-rust](packages/core-rust/CLAUDE.md)       | Rust          | Performance-critical compute      | 14    | [→](packages/core-rust/CLAUDE.md)    |
| [core-bun](packages/core-bun/CLAUDE.md)         | Bun/TS        | Async I/O, APIs, metrics          | 32    | [→](packages/core-bun/CLAUDE.md)     |
| [shared-types](packages/shared-types/CLAUDE.md) | Multi         | JSON Schema type definitions      | -     | [→](packages/shared-types/CLAUDE.md) |

## Data Flow

```
data/nav_data_custom/*.csv  -->  [ith-python]  -->  artifacts/synth_ithes/
                                      |
                                      v
                              artifacts/results.html
```

## Directory Structure

```
trading-fitness/
├── .claude/
│   └── skills/              # Claude Code skill modules (Python, Rust, Bun)
├── packages/
│   ├── ith-python/          # Primary Python analysis package
│   ├── metrics-rust/        # BiLSTM metrics with Python bindings (PyO3)
│   ├── core-rust/           # Rust performance-critical code
│   ├── core-bun/            # Bun/TS async I/O, APIs
│   └── shared-types/        # Cross-language schemas
├── services/                # Future deployable services
├── data/                    # Input data (TRACKED)
│   └── nav_data_custom/     # Custom NAV CSV files
├── artifacts/               # Generated outputs (GITIGNORED)
│   └── synth_ithes/         # Analysis results
├── logs/                    # JSONL logs (GITIGNORED)
├── rules/                   # ast-grep rule directories
├── scripts/                 # Automation (benchmarks, code generation)
└── docs/                    # Documentation
```

## ITH (Investment Time Horizon) Concept

ITH analysis evaluates trading strategy fitness using TMAEG (Target Maximum Acceptable Excess Gain) thresholds:

- **TMAEG**: Drawdown-based hurdle for counting ITH epochs
- **ITH Epochs**: Time periods where strategy exceeds performance thresholds
- **Fitness Criteria**: Minimum epoch count, Sharpe ratio bounds, coefficient of variation

**Deep Dive**: [docs/ITH.md](docs/ITH.md)

## Terminology

| Term                 | Definition                                                                                  | Unit/Range                |
| -------------------- | ------------------------------------------------------------------------------------------- | ------------------------- |
| **ITH**              | Investment Time Horizon - time-agnostic fitness metric counting threshold crossings         | epochs                    |
| **TMAEG**            | Target Maximum Acceptable Excess Gain - drawdown-based hurdle for epoch counting            | decimal (e.g., 0.05 = 5%) |
| **MCOT**             | Minimum Cost of Trade - floor value for TMAEG representing transaction costs                | dbps                      |
| **Maximum Drawdown** | Worst decline from peak: `1 - (trough/peak)`. Adverse movement for long positions.          | [0, 1]                    |
| **Maximum Runup**    | Worst rise from trough: `1 - (trough/current)`. Adverse movement for short positions.       | [0, 1]                    |
| **Bull ITH**         | ITH for long positions. TMAEG = max(Max Drawdown, MCOT). Epochs count excess gains.         | epochs                    |
| **Bear ITH**         | ITH for short positions. TMAEG = max(Max Runup, MCOT). Epochs count excess gains (inverse). | epochs                    |
| **NAV**              | Net Asset Value - normalized price series starting at 1.0                                   | ratio                     |
| **Epoch**            | Period where cumulative excess gain exceeds TMAEG threshold, triggering reset               | count                     |
| **CV**               | Coefficient of Variation - `std(intervals) / mean(intervals)`. Measures epoch regularity.   | [0, ∞)                    |
| **dbps**             | Decimal basis points. 1 dbps = 0.01% = 0.0001. Used for MCOT and thresholds.                | 1 dbps = 0.0001           |

## Documentation

| Topic                                                                       | Location    | Purpose                              |
| --------------------------------------------------------------------------- | ----------- | ------------------------------------ |
| [Architecture](docs/ARCHITECTURE.md)                                        | docs/       | System design, tech stack, data flow |
| [ITH Methodology](docs/ITH.md)                                              | docs/       | Core algorithm and fitness criteria  |
| [Logging Contract](docs/LOGGING.md)                                         | docs/       | NDJSON format, structured logging    |
| [SR&ED Tracking](docs/SRED.md)                                              | docs/       | Tax credit evidence and claims       |
| [Observability Plan](docs/plans/2026-01-25-observability-telemetry-plan.md) | docs/plans/ | Telemetry enhancement roadmap        |
| [Session Resume](RESUME.md)                                                 | root        | Continue work from last session      |

## MCP Servers

Configured in `.mcp.json`:

| Server        | Purpose                                   |
| ------------- | ----------------------------------------- |
| `mise`        | Task runner and environment management    |
| `code-search` | Semantic code search via ck               |
| `ast-grep`    | Structural code search and transformation |

## Development

### Python (ith-python)

```bash
cd packages/ith-python
uv sync                      # Install dependencies
uv run python -m ith_python.ith  # Run analysis
uv run pytest                # Run tests
uv run ruff check --fix      # Lint
```

### Rust (core-rust)

```bash
cd packages/core-rust
cargo check                  # Verify compilation
cargo test                   # Run tests
```

### Rust + Python (metrics-rust)

```bash
cd packages/metrics-rust
cargo test                         # Rust tests
maturin build --features python    # Build Python wheel
# Install: uv pip install target/wheels/*.whl
```

**Python Usage**: See [metrics-rust/CLAUDE.md](packages/metrics-rust/CLAUDE.md) for API reference.

### Bun (core-bun)

```bash
cd packages/core-bun
bun run index.ts             # Run entry point
bun test                     # Run tests
```

## Configuration

- **mise.toml**: Runtime versions (SSoT), tasks, environment
- **sgconfig.yml**: ast-grep rules configuration
- **.mcp.json**: MCP server definitions

## Git Workflow

This repository uses git-town for branch workflow automation.

**Configuration**:

```bash
git config git-town.main-branch main
git config git-town.push-new-branches true
git config git-town.sync-feature-strategy rebase
```

**Workflow**:

| Task                  | Command           |
| --------------------- | ----------------- |
| Create feature branch | `git town hack`   |
| Sync with main        | `git town sync`   |
| Create PR and merge   | `git town ship`   |
| Switch branches       | `git town switch` |

**Environment**: Uses `read_file()` pattern in mise.toml for GH_TOKEN (prevents process storms).

## SR&ED Documentation

This project tracks SR&ED (Scientific Research & Experimental Development) eligible work for CRA tax credits.

**Deep Dive**: [docs/SRED.md](docs/SRED.md)

| Commit Types                               | Labels                                 |
| ------------------------------------------ | -------------------------------------- |
| `experiment:`, `research:`, `uncertainty:` | `sred:uncertainty`, `sred:advancement` |
| `advancement:`, `hypothesis:`, `analysis:` | `sred:experiment`, `sred:research`     |
| `iteration:`, `benchmark:`                 | `sred:eligible`                        |

## Migration History

Consolidated from:

- `~/scripts/personal/fitness/custom-fitness/ith/` (ith.py, results.html)
- `~/eon/tl-ml-feature-set/experiments/ith_numba.py`
- `~/Documents/ith-fitness/` (artifacts)
- `~/Library/Application Support/ith-fitness/` (cached analysis)

---

_Polyglot Monorepo - Trading Strategy Fitness Analysis_
