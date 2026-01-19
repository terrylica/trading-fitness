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

| Package        | Language | Purpose                        | Tests |
| -------------- | -------- | ------------------------------ | ----- |
| `ith-python`   | Python   | ITH fitness analysis (PRIMARY) | 40    |
| `core-rust`    | Rust     | Performance-critical compute   | 14    |
| `core-bun`     | Bun/TS   | Async I/O, APIs, metrics       | 32    |
| `shared-types` | Multi    | JSON Schema type definitions   | -     |

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
│   └── skills/              # Claude Code skill modules
├── packages/
│   ├── ith-python/          # Primary Python analysis package
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

**Commit Types**: `experiment:`, `research:`, `uncertainty:`, `advancement:`, `hypothesis:`, `analysis:`, `iteration:`, `benchmark:`

**Documentation**: [docs/SRED.md](docs/SRED.md)

**Labels**: `sred:uncertainty`, `sred:advancement`, `sred:experiment`, `sred:research`, `sred:eligible`

## Migration History

Consolidated from:

- `~/scripts/personal/fitness/custom-fitness/ith/` (ith.py, results.html)
- `~/eon/tl-ml-feature-set/experiments/ith_numba.py`
- `~/Documents/ith-fitness/` (artifacts)
- `~/Library/Application Support/ith-fitness/` (cached analysis)

---

_Polyglot Monorepo - Trading Strategy Fitness Analysis_
