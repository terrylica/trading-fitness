# Meta-Prompt: Autonomous Polyglot Monorepo Bootstrap

> **Role**: You are a Principal Software Architect specializing in AI-native monorepo design.
> **Mission**: Construct a production-grade polyglot monorepo from scratch, optimized for agentic workflows with Claude Code CLI.
> **Constraint**: The human will not touch any code. You execute everything autonomously, verifying at each phase.

---

## Phase 0: Pre-Flight Verification

Before creating any files, verify the environment:

```bash
# Check required tools exist
command -v mise && mise --version
command -v git && git --version
command -v cargo && cargo --version
command -v uv && uv --version
command -v bun && bun --version
```

If any tool is missing, install via mise:

```bash
mise use -g rust@latest python@3.12 node@lts bun@latest uv@latest
```

Create project root and initialize git:

```bash
mkdir -p ~/projects/hft-monorepo && cd ~/projects/hft-monorepo
git init
```

---

## Phase 1: Foundational Structure

Create the canonical directory structure for a polyglot HFT monorepo:

```
hft-monorepo/
├── CLAUDE.md                    # Hub: Link Farm root (this file)
├── mise.toml                    # Orchestrator: tools + tasks
├── sgconfig.yml                 # ast-grep rules configuration
├── .mise/                       # Mise local config
├── .mcp.json                    # MCP server configuration
├── targets.json                 # Affected-target manifest
├── rules/                       # ast-grep rule directories
│   ├── general/                 # Cross-language patterns
│   ├── python/                  # Python-specific rules
│   ├── rust/                    # Rust-specific rules
│   └── typescript/              # TypeScript-specific rules
├── docs/                        # Deep documentation (spoke)
│   ├── ARCHITECTURE.md
│   ├── LOGGING.md
│   ├── TESTING.md
│   └── WORKFLOWS.md
├── skills/                      # Claude Code skill modules (spoke)
│   ├── python/
│   │   └── SKILL.md
│   ├── rust/
│   │   └── SKILL.md
│   └── bun/
│       └── SKILL.md
├── packages/                    # Polyglot packages
│   ├── core-python/             # Python: shared utilities
│   │   ├── CLAUDE.md            # Child hub
│   │   ├── pyproject.toml
│   │   └── src/
│   ├── core-rust/               # Rust: performance-critical
│   │   ├── CLAUDE.md            # Child hub
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── core-bun/                # Bun: async I/O, APIs
│   │   ├── CLAUDE.md            # Child hub
│   │   ├── package.json
│   │   └── src/
│   └── shared-types/            # Cross-language type definitions
│       ├── CLAUDE.md
│       └── schemas/
├── services/                    # Deployable services
│   ├── data-ingestion/
│   │   └── CLAUDE.md
│   ├── strategy-engine/
│   │   └── CLAUDE.md
│   └── execution-gateway/
│       └── CLAUDE.md
├── scripts/                     # Build/deploy automation
│   └── affected.sh
└── logs/                        # Local log output (gitignored)
```

Execute creation:

```bash
mkdir -p docs skills/{python,rust,bun} packages/{core-python/src,core-rust/src,core-bun/src,shared-types/schemas} services/{data-ingestion,strategy-engine,execution-gateway} scripts logs rules/{general,python,rust,typescript}
touch .gitignore
```

---

## Phase 2: Root CLAUDE.md — The Hub

Create the root `CLAUDE.md` as the Link Farm hub with Progressive Disclosure:

````markdown
# HFT Polyglot Monorepo

> **Navigation**: This file is the single entry point. Each section links to deeper documentation. Child directories contain their own `CLAUDE.md` files that Claude loads on-demand.

## Quick Reference

| Action         | Command                                       |
| -------------- | --------------------------------------------- |
| Build affected | `mise run build:affected`                     |
| Test affected  | `mise run test:affected`                      |
| Lint all       | `mise run lint`                               |
| Search code    | Use `ck` MCP tool: `semantic_search("query")` |

## Architecture Overview

**Stack**: Python (uv) · Rust (cargo) · Bun · Mise (orchestrator)
**Pattern**: Polyglot monorepo with independent semantic versioning
**AI Interface**: Claude Code CLI via MCP servers

→ Deep dive: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Package Map

| Package        | Language | Purpose                       | Entry                                                              |
| -------------- | -------- | ----------------------------- | ------------------------------------------------------------------ |
| `core-python`  | Python   | Shared utilities, data models | [packages/core-python/CLAUDE.md](packages/core-python/CLAUDE.md)   |
| `core-rust`    | Rust     | Performance-critical compute  | [packages/core-rust/CLAUDE.md](packages/core-rust/CLAUDE.md)       |
| `core-bun`     | Bun/TS   | Async I/O, HTTP APIs          | [packages/core-bun/CLAUDE.md](packages/core-bun/CLAUDE.md)         |
| `shared-types` | Multi    | Cross-language schemas        | [packages/shared-types/CLAUDE.md](packages/shared-types/CLAUDE.md) |

## Services

| Service             | Owner Package | Purpose           |
| ------------------- | ------------- | ----------------- |
| `data-ingestion`    | core-python   | Market data feeds |
| `strategy-engine`   | core-rust     | Signal generation |
| `execution-gateway` | core-bun      | Order routing     |

→ Service docs in respective `services/*/CLAUDE.md`

## Logging Contract

All packages emit **NDJSON to `logs/*.jsonl`** with schema:

```json
{
  "ts": "ISO8601",
  "level": "INFO",
  "msg": "...",
  "component": "pkg",
  "env": "dev",
  "pid": 123,
  "trace_id": "uuid"
}
```
````

→ Deep dive: [docs/LOGGING.md](docs/LOGGING.md)

## Workflow Protocol

When modifying code in this repo:

1. **Explore** — Read the relevant `CLAUDE.md` in the target directory
2. **Search** — Use `semantic_search` MCP tool to find related code
3. **Affected** — Run `mise run affected` to identify impacted targets
4. **Plan** — State approach before editing (ultrathink if complex)
5. **Implement** — Make changes, running `mise run lint` after each file
6. **Test** — Run `mise run test:affected` before committing
7. **Verify** — Confirm logs emit correctly to `logs/`

→ Deep dive: [docs/WORKFLOWS.md](docs/WORKFLOWS.md)

## Skills Reference

Language-specific patterns and idioms:

- [skills/python/SKILL.md](skills/python/SKILL.md) — uv, loguru, async patterns
- [skills/rust/SKILL.md](skills/rust/SKILL.md) — cargo, tracing, error handling
- [skills/bun/SKILL.md](skills/bun/SKILL.md) — bun, pino, Zod validation

## MCP Tools Available

| Server  | Tools                                         | Use For                 |
| ------- | --------------------------------------------- | ----------------------- |
| `mise`  | `mise_tools`, `mise_tasks`, `mise_env`        | Project context         |
| `ck`    | `semantic_search`, `hybrid_search`, `reindex` | Code search             |
| `shell` | Execute allowed commands                      | Git, affected detection |

## Do NOT

- Skip reading child `CLAUDE.md` before editing that package
- Add features beyond what was requested
- Log secrets, PII, or credentials
- Break the NDJSON schema contract
- Commit without running affected tests

````

---

## Phase 3: Documentation Spokes (docs/)

### docs/ARCHITECTURE.md

```markdown
# Architecture Deep Dive

← Back to [CLAUDE.md](../CLAUDE.md)

## Design Principles

1. **Language-Native Tooling**: Each language uses its idiomatic package manager
   - Python: `uv` (fast, lockfile-based)
   - Rust: `cargo` (workspaces)
   - Bun: `bun` (all-in-one runtime)

2. **Mise as Orchestrator**: Single tool for:
   - Tool version management (replaces asdf/nvm/pyenv)
   - Task running (replaces make/just)
   - Environment variables (replaces direnv)

3. **Affected-Target Detection**: Git-based analysis determines which packages need rebuild/test after changes

4. **MCP-First AI Interface**: Claude Code interacts via:
   - `mise mcp` — project context
   - `ck --serve` — semantic code search
   - Shell commands — git operations

## Dependency Graph

````

shared-types (schemas)
↓
┌────┴────┐
↓ ↓
core-python core-rust
↓ ↓
└────┬────┘
↓
core-bun (orchestrates)
↓
┌────┼────┐
↓ ↓ ↓
data strategy execution

````

## Cross-Language Communication

- **Schemas**: JSON Schema definitions in `shared-types/schemas/`
- **IPC**: Unix sockets for local, gRPC for networked
- **Serialization**: MessagePack for binary, JSON for human-readable

## Build Order

Mise enforces correct ordering via task dependencies:
```toml
[tasks."build:all"]
depends = ["build:shared-types", "build:core-python", "build:core-rust", "build:core-bun"]
````

````

### docs/LOGGING.md

```markdown
# Logging Contract

← Back to [CLAUDE.md](../CLAUDE.md)

## Core Schema (REQUIRED)

Every log line MUST be valid NDJSON with these fields:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `ts` | ISO8601 string | UTC timestamp | `"2026-01-15T10:30:00.123Z"` |
| `level` | string | Log level | `"DEBUG"`, `"INFO"`, `"WARN"`, `"ERROR"` |
| `msg` | string | Human-readable message | `"Order submitted"` |
| `component` | string | Logger/module name | `"strategy.signals"` |
| `env` | string | Environment | `"dev"`, `"staging"`, `"prod"` |
| `pid` | integer | Process ID | `12345` |

## Optional Fields (RECOMMENDED)

| Field | Type | Use Case |
|-------|------|----------|
| `trace_id` | UUID string | Distributed tracing correlation |
| `request_id` | UUID string | HTTP request correlation |
| `tid` | integer | Thread ID (for threaded runtimes) |
| `duration_ms` | float | Operation timing |
| `error` | object | Structured error `{type, message, stack}` |
| `context` | object | Additional structured data |

## Output Location

Use `platformdirs` (or equivalent) to write logs:

| Platform | Log Directory |
|----------|---------------|
| Linux | `~/.local/state/{app}/logs/` or `./logs/` in dev |
| macOS | `~/Library/Logs/{app}/` or `./logs/` in dev |
| Windows | `%LOCALAPPDATA%/{app}/Logs/` or `./logs/` in dev |

For local development, default to `{repo_root}/logs/{component}.jsonl`

## Rotation & Retention

| Setting | Default | Notes |
|---------|---------|-------|
| Max file size | 10 MB | Rotate when exceeded |
| Max files | 5 | Delete oldest |
| Compression | gzip | Optional, for archived logs |
| Retention | 7 days | For dev; 30+ for prod |

## Language Implementations

### Python (loguru)
```python
from loguru import logger
import sys

logger.remove()  # Remove default handler
logger.add(
    "logs/core-python.jsonl",
    format="{time:YYYY-MM-DDTHH:mm:ss.SSS}Z|{level}|{message}|{extra}",
    serialize=True,  # NDJSON output
    rotation="10 MB",
    retention="7 days",
    compression="gz"
)
````

### Rust (tracing + tracing-subscriber)

```rust
use tracing_subscriber::{fmt, prelude::*};

tracing_subscriber::registry()
    .with(fmt::layer().json().with_file(true).with_line_number(true))
    .init();
```

### Bun/Node (pino)

```typescript
import pino from "pino";

const logger = pino({
  level: "info",
  timestamp: pino.stdTimeFunctions.isoTime,
  formatters: {
    level: (label) => ({ level: label.toUpperCase() }),
  },
  transport: {
    target: "pino/file",
    options: { destination: "./logs/core-bun.jsonl" },
  },
});
```

## Anti-Patterns

❌ **Never log**:

- Passwords, API keys, tokens
- PII (emails, names, addresses)
- Full request/response bodies (summarize instead)
- Stack traces in production INFO logs (use ERROR level)

❌ **Never allow logging to crash the app**:

```python
# WRONG
logger.info(f"User: {user}")  # Crashes if user is None

# RIGHT
logger.info("User action", user_id=getattr(user, 'id', 'unknown'))
```

````

### docs/TESTING.md

```markdown
# Testing Strategy

← Back to [CLAUDE.md](../CLAUDE.md)

## Affected-Only Testing

Run tests only for packages changed since `origin/main`:

```bash
mise run test:affected
````

This executes `scripts/affected.sh` which:

1. Gets changed files via `git diff --name-only origin/main`
2. Maps files to packages via `targets.json`
3. Includes transitive dependents
4. Runs `mise run test:{package}` for each

## Test Commands by Language

| Package     | Command         | Framework |
| ----------- | --------------- | --------- |
| core-python | `uv run pytest` | pytest    |
| core-rust   | `cargo test`    | built-in  |
| core-bun    | `bun test`      | bun:test  |

## Coverage Requirements

| Type           | Threshold | Enforcement |
| -------------- | --------- | ----------- |
| Line           | 80%       | CI gate     |
| Branch         | 70%       | CI gate     |
| Critical paths | 100%      | Code review |

## Test Naming

```
test_{unit_under_test}_{scenario}_{expected_outcome}
```

Example: `test_order_submit_insufficient_balance_returns_error`

````

### docs/WORKFLOWS.md

```markdown
# Development Workflows

← Back to [CLAUDE.md](../CLAUDE.md)

## Standard Change Workflow

````

┌─────────────────────────────────────────────────────────────┐
│ 1. EXPLORE │
│ Read CLAUDE.md in target directory │
│ Use semantic_search MCP to find related code │
└─────────────────────────┬───────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ 2. PLAN │
│ For complex changes: ultrathink │
│ State approach before any edits │
│ Identify affected packages via: mise run affected │
└─────────────────────────┬───────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ 3. IMPLEMENT │
│ Edit files, one logical change at a time │
│ Run mise run lint after each file │
│ Follow language skill guide │
└─────────────────────────┬───────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ 4. VERIFY │
│ Run mise run test:affected │
│ Check logs emit to logs/\*.jsonl │
│ Confirm NDJSON schema compliance │
└─────────────────────────┬───────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ 5. COMMIT │
│ Conventional commit format │
│ Reference affected packages in scope │
└─────────────────────────────────────────────────────────────┘

```

## Commit Message Format

```

<type>(<scope>): <description>

[optional body]

[optional footer]

```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`
Scope: Package name (`core-python`, `core-rust`, etc.)

Example:
```

feat(core-rust): add order validation module

- Implements balance check before submission
- Adds structured error types
- Logs validation failures to NDJSON

Affects: core-rust, execution-gateway

```

```

---

## Phase 4: Skills Spokes (skills/)

### skills/python/SKILL.md

````markdown
# Python Skill Guide

← Back to [CLAUDE.md](../../CLAUDE.md)

## Toolchain

| Tool     | Purpose              | Config           |
| -------- | -------------------- | ---------------- |
| `uv`     | Package management   | `pyproject.toml` |
| `ruff`   | Linting + formatting | `ruff.toml`      |
| `pytest` | Testing              | `pytest.ini`     |
| `loguru` | Logging              | In-code config   |

## Project Setup Pattern

```toml
# pyproject.toml
[project]
name = "core-python"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "loguru>=0.7",
    "platformdirs>=4.0",
    "pydantic>=2.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
]
```
````

## Code Style

- Type hints: **mandatory** on all public functions
- Async: **prefer** for I/O operations
- Docstrings: Google style
- Max line length: 100

## Logging Setup

```python
# src/core_python/logging.py
from loguru import logger
from platformdirs import user_log_dir
from pathlib import Path
import os

def setup_logging(component: str, env: str = "dev") -> None:
    """Configure NDJSON logging per repo contract."""

    # Determine log directory
    if env == "dev":
        log_dir = Path(__file__).parents[3] / "logs"
    else:
        log_dir = Path(user_log_dir("hft-monorepo"))

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{component}.jsonl"

    # Remove default and add NDJSON handler
    logger.remove()
    logger.add(
        log_file,
        serialize=True,
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        enqueue=True,  # Thread-safe
    )

    # Bind constant fields
    logger.configure(extra={
        "component": component,
        "env": env,
        "pid": os.getpid(),
    })
```

## Error Handling Pattern

```python
from typing import TypeVar, Generic
from dataclasses import dataclass

T = TypeVar("T")

@dataclass
class Result(Generic[T]):
    value: T | None
    error: str | None

    @property
    def is_ok(self) -> bool:
        return self.error is None

    @classmethod
    def ok(cls, value: T) -> "Result[T]":
        return cls(value=value, error=None)

    @classmethod
    def err(cls, error: str) -> "Result[T]":
        return cls(value=None, error=error)
```

## Common Commands

```bash
uv sync                    # Install dependencies
uv run pytest              # Run tests
uv run ruff check --fix    # Lint and auto-fix
uv run ruff format         # Format code
```

````

### skills/rust/SKILL.md

```markdown
# Rust Skill Guide

← Back to [CLAUDE.md](../../CLAUDE.md)

## Toolchain

| Tool | Purpose | Config |
|------|---------|--------|
| `cargo` | Build + package | `Cargo.toml` |
| `clippy` | Linting | `.clippy.toml` |
| `rustfmt` | Formatting | `rustfmt.toml` |
| `tracing` | Logging | In-code config |

## Workspace Setup

```toml
# Cargo.toml (workspace root)
[workspace]
members = ["packages/core-rust"]
resolver = "2"

[workspace.dependencies]
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "1.0"
anyhow = "1.0"
tokio = { version = "1.0", features = ["full"] }
````

## Code Style

- Error handling: `thiserror` for libraries, `anyhow` for binaries
- Async runtime: `tokio`
- Serialization: `serde` with JSON
- Max line length: 100

## Logging Setup

```rust
// src/logging.rs
use tracing_subscriber::{
    fmt::{self, format::JsonFields},
    prelude::*,
    EnvFilter,
};
use std::fs::OpenOptions;

pub fn setup_logging(component: &str, env: &str) -> anyhow::Result<()> {
    let log_dir = if env == "dev" {
        std::env::current_dir()?.join("logs")
    } else {
        dirs::state_dir()
            .unwrap_or_else(|| std::env::current_dir().unwrap())
            .join("hft-monorepo/logs")
    };

    std::fs::create_dir_all(&log_dir)?;
    let log_file = log_dir.join(format!("{}.jsonl", component));

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_file)?;

    let subscriber = tracing_subscriber::registry()
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .with(
            fmt::layer()
                .json()
                .with_writer(file)
                .with_current_span(false)
                .flatten_event(true)
        );

    tracing::subscriber::set_global_default(subscriber)?;
    Ok(())
}
```

## Error Handling Pattern

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OrderError {
    #[error("insufficient balance: required {required}, available {available}")]
    InsufficientBalance { required: f64, available: f64 },

    #[error("invalid symbol: {0}")]
    InvalidSymbol(String),

    #[error("network error: {0}")]
    Network(#[from] std::io::Error),
}

// Usage with tracing
fn submit_order(order: Order) -> Result<OrderId, OrderError> {
    tracing::info!(order_id = %order.id, symbol = %order.symbol, "submitting order");

    if order.amount > balance {
        tracing::warn!(required = order.amount, available = balance, "insufficient balance");
        return Err(OrderError::InsufficientBalance {
            required: order.amount,
            available: balance,
        });
    }

    Ok(order.id)
}
```

## Common Commands

```bash
cargo build --release       # Build optimized
cargo test                  # Run tests
cargo clippy --all-targets  # Lint
cargo fmt                   # Format
```

````

### skills/bun/SKILL.md

```markdown
# Bun Skill Guide

← Back to [CLAUDE.md](../../CLAUDE.md)

## Toolchain

| Tool | Purpose | Config |
|------|---------|--------|
| `bun` | Runtime + package | `package.json`, `bunfig.toml` |
| `biome` | Lint + format | `biome.json` |
| `bun:test` | Testing | Built-in |
| `pino` | Logging | In-code config |

## Project Setup

```json
{
  "name": "core-bun",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "bun --watch src/index.ts",
    "build": "bun build src/index.ts --outdir dist --target bun",
    "test": "bun test",
    "lint": "biome check --apply .",
    "format": "biome format --write ."
  },
  "dependencies": {
    "pino": "^9.0.0",
    "zod": "^3.23.0"
  },
  "devDependencies": {
    "@biomejs/biome": "^1.8.0",
    "@types/bun": "latest"
  }
}
````

## Code Style

- Validation: **Zod** for runtime type checking
- Modules: ES modules only (no CommonJS)
- Async: Native async/await
- Max line length: 100

## Logging Setup

```typescript
// src/logging.ts
import pino from "pino";
import { mkdirSync, existsSync } from "fs";
import { join, dirname } from "path";

export function setupLogging(component: string, env: string = "dev") {
  const logDir =
    env === "dev"
      ? join(dirname(dirname(import.meta.dir)), "logs")
      : join(process.env.HOME || "/tmp", ".local/state/hft-monorepo/logs");

  if (!existsSync(logDir)) {
    mkdirSync(logDir, { recursive: true });
  }

  const logFile = join(logDir, `${component}.jsonl`);

  return pino({
    level: env === "dev" ? "debug" : "info",
    timestamp: () => `,"ts":"${new Date().toISOString()}"`,
    formatters: {
      level: (label) => ({ level: label.toUpperCase() }),
    },
    base: {
      component,
      env,
      pid: process.pid,
    },
    transport: {
      target: "pino/file",
      options: { destination: logFile },
    },
  });
}

// Usage
const logger = setupLogging("core-bun");
logger.info({ orderId: "123", symbol: "BTC-USD" }, "Order received");
```

## Validation Pattern

```typescript
import { z } from "zod";

export const OrderSchema = z.object({
  id: z.string().uuid(),
  symbol: z.string().min(1),
  side: z.enum(["buy", "sell"]),
  amount: z.number().positive(),
  price: z.number().positive().optional(),
  type: z.enum(["market", "limit"]),
});

export type Order = z.infer<typeof OrderSchema>;

// Usage with logging
function validateOrder(input: unknown, logger: pino.Logger): Order | null {
  const result = OrderSchema.safeParse(input);

  if (!result.success) {
    logger.warn({ errors: result.error.flatten() }, "Order validation failed");
    return null;
  }

  return result.data;
}
```

## Common Commands

```bash
bun install           # Install dependencies
bun test              # Run tests
bun run lint          # Lint and fix
bun run build         # Build for production
```

````

---

## Phase 5: Child CLAUDE.md Files (Package-Level)

### packages/core-python/CLAUDE.md

```markdown
# core-python

← Back to [../../CLAUDE.md](../../CLAUDE.md) | Skill guide: [../../skills/python/SKILL.md](../../skills/python/SKILL.md)

## Purpose

Shared Python utilities for the HFT monorepo:
- Data models (Pydantic)
- Market data parsers
- Common async utilities
- Logging configuration

## Structure

````

core-python/
├── CLAUDE.md # This file
├── pyproject.toml # uv config
├── src/
│ └── core_python/
│ ├── **init**.py
│ ├── logging.py # NDJSON setup
│ ├── models/ # Pydantic models
│ └── utils/ # Shared utilities
└── tests/

````

## Dependencies

- **Upstream**: `shared-types` (JSON schemas)
- **Downstream**: `data-ingestion`, `strategy-engine`

## Key Files

| File | Purpose |
|------|---------|
| `logging.py` | Loguru NDJSON configuration |
| `models/order.py` | Order data model |
| `models/market.py` | Market data structures |

## Commands

```bash
cd packages/core-python
uv sync
uv run pytest
uv run ruff check --fix
````

## Logging

Logs emit to: `{repo_root}/logs/core-python.jsonl`

Initialize in entry points:

```python
from core_python.logging import setup_logging
setup_logging("core-python")
```

````

### packages/core-rust/CLAUDE.md

```markdown
# core-rust

← Back to [../../CLAUDE.md](../../CLAUDE.md) | Skill guide: [../../skills/rust/SKILL.md](../../skills/rust/SKILL.md)

## Purpose

Performance-critical Rust components:
- Order matching engine
- Risk calculations
- Signal processing
- Low-latency data structures

## Structure

````

core-rust/
├── CLAUDE.md # This file
├── Cargo.toml
├── src/
│ ├── lib.rs
│ ├── logging.rs # tracing NDJSON setup
│ ├── matching/ # Order matching
│ ├── risk/ # Risk calculations
│ └── signals/ # Signal processing
└── tests/

````

## Dependencies

- **Upstream**: `shared-types` (JSON schemas)
- **Downstream**: `strategy-engine`

## Key Modules

| Module | Purpose |
|--------|---------|
| `matching::engine` | Order book and matching |
| `risk::position` | Position tracking |
| `signals::indicators` | Technical indicators |

## Commands

```bash
cd packages/core-rust
cargo build --release
cargo test
cargo clippy --all-targets
cargo fmt
````

## Logging

Logs emit to: `{repo_root}/logs/core-rust.jsonl`

Initialize in main/lib:

```rust
use core_rust::logging::setup_logging;
setup_logging("core-rust", "dev").expect("logging init");
```

## Performance Notes

- Use `#[inline]` for hot-path functions
- Prefer stack allocation where possible
- Benchmark with `criterion` before optimizing

````

### packages/core-bun/CLAUDE.md

```markdown
# core-bun

← Back to [../../CLAUDE.md](../../CLAUDE.md) | Skill guide: [../../skills/bun/SKILL.md](../../skills/bun/SKILL.md)

## Purpose

Async I/O and HTTP APIs:
- REST/WebSocket servers
- External API clients
- Event orchestration
- Real-time data streaming

## Structure

````

core-bun/
├── CLAUDE.md # This file
├── package.json
├── bunfig.toml
├── src/
│ ├── index.ts
│ ├── logging.ts # pino NDJSON setup
│ ├── api/ # HTTP handlers
│ ├── ws/ # WebSocket handlers
│ └── clients/ # External API clients
└── tests/

````

## Dependencies

- **Upstream**: `shared-types`, `core-python`, `core-rust`
- **Downstream**: `execution-gateway`

## Key Files

| File | Purpose |
|------|---------|
| `api/orders.ts` | Order submission endpoints |
| `ws/market.ts` | Market data WebSocket |
| `clients/exchange.ts` | Exchange API client |

## Commands

```bash
cd packages/core-bun
bun install
bun test
bun run lint
bun run dev
````

## Logging

Logs emit to: `{repo_root}/logs/core-bun.jsonl`

Initialize in entry:

```typescript
import { setupLogging } from "./logging";
const logger = setupLogging("core-bun");
```

````

### packages/shared-types/CLAUDE.md

```markdown
# shared-types

← Back to [../../CLAUDE.md](../../CLAUDE.md)

## Purpose

Cross-language type definitions:
- JSON Schema source of truth
- Generated types for each language
- API contract definitions

## Structure

````

shared-types/
├── CLAUDE.md
├── schemas/
│ ├── order.schema.json
│ ├── market.schema.json
│ └── error.schema.json
└── generated/ # Auto-generated, do not edit
├── python/
├── rust/
└── typescript/

````

## Schema Contract

All schemas follow JSON Schema Draft 2020-12.

Required fields for every schema:
- `$id`: Unique identifier
- `$schema`: Draft version
- `title`: Human-readable name
- `type`: Root type

## Code Generation

```bash
mise run generate:types
````

This generates:

- Python: Pydantic models via `datamodel-codegen`
- Rust: Structs via `typify`
- TypeScript: Zod schemas via `json-schema-to-zod`

## Modification Workflow

1. Edit schema in `schemas/`
2. Run `mise run generate:types`
3. Commit both schema and generated files
4. Update downstream packages

````

---

## Phase 6: Configuration Files

### mise.toml

```toml
[env]
MISE_EXPERIMENTAL = "1"
LOG_DIR = "{{config_root}}/logs"
ENV = "dev"

[tools]
python = "3.12"
rust = "1.82"
node = "22"
bun = "1.1"
uv = "0.5"

[tasks.lint]
description = "Lint all packages"
run = """
cd packages/core-python && uv run ruff check --fix
cd packages/core-rust && cargo clippy --all-targets
cd packages/core-bun && bun run lint
"""

[tasks.test]
description = "Test all packages"
run = """
cd packages/core-python && uv run pytest
cd packages/core-rust && cargo test
cd packages/core-bun && bun test
"""

[tasks.affected]
description = "List packages affected by git changes"
run = "bash scripts/affected.sh list"

[tasks."test:affected"]
description = "Test only affected packages"
run = "bash scripts/affected.sh test"

[tasks."build:affected"]
description = "Build only affected packages"
run = "bash scripts/affected.sh build"

[tasks."generate:types"]
description = "Generate types from JSON schemas"
run = """
cd packages/shared-types
# Add generation commands per language
echo "Type generation not yet implemented"
"""

[tasks.reindex]
description = "Reindex codebase for semantic search"
run = "ck index --all"
````

### .mcp.json

```json
{
  "mcpServers": {
    "mise": {
      "command": "mise",
      "args": ["mcp"],
      "env": {
        "MISE_EXPERIMENTAL": "1"
      }
    },
    "code-search": {
      "command": "ck",
      "args": ["--serve"],
      "cwd": "."
    },
    "shell": {
      "command": "uvx",
      "args": ["mcp-shell-server"],
      "env": {
        "ALLOW_COMMANDS": "mise,git,jq,cargo,uv,bun,cat,ls,grep,head,tail,find"
      }
    },
    "ast-grep": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/ast-grep/ast-grep-mcp",
        "ast-grep-server"
      ],
      "cwd": "."
    }
  }
}
```

**ast-grep MCP Tools Reference**:

| Tool                   | Purpose                                |
| ---------------------- | -------------------------------------- |
| `dump_syntax_tree`     | Visualize AST structure of code        |
| `test_match_code_rule` | Validate YAML rules before applying    |
| `find_code`            | Simple pattern-based structural search |
| `find_code_by_rule`    | Complex YAML rule-based search         |

**Usage**: Ask Claude to "use ast-grep to find..." for structural code searches.

### sgconfig.yml

```yaml
# ast-grep rule configuration
# Structural code search and transformation

ruleDirs:
  - rules/general
  - rules/python
  - rules/rust
  - rules/typescript

testConfigs:
  - testDir: tests/rules

utilDirs:
  - utils

languageGlobs:
  typescript: ["*.ts", "*.tsx"]
  javascript: ["*.js", "*.jsx", "*.mjs"]
  python: ["*.py", "*.pyi"]
  html: ["*.vue", "*.astro", "*.svelte"]
```

### targets.json

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "description": "Monorepo target manifest for affected detection",
  "targets": {
    "shared-types": {
      "path": "packages/shared-types",
      "language": "multi",
      "deps": [],
      "commands": {
        "build": "mise run generate:types",
        "test": "echo 'no tests for schemas'"
      }
    },
    "core-python": {
      "path": "packages/core-python",
      "language": "python",
      "deps": ["shared-types"],
      "commands": {
        "build": "cd packages/core-python && uv sync",
        "test": "cd packages/core-python && uv run pytest",
        "lint": "cd packages/core-python && uv run ruff check --fix"
      }
    },
    "core-rust": {
      "path": "packages/core-rust",
      "language": "rust",
      "deps": ["shared-types"],
      "commands": {
        "build": "cd packages/core-rust && cargo build",
        "test": "cd packages/core-rust && cargo test",
        "lint": "cd packages/core-rust && cargo clippy --all-targets"
      }
    },
    "core-bun": {
      "path": "packages/core-bun",
      "language": "bun",
      "deps": ["shared-types", "core-python", "core-rust"],
      "commands": {
        "build": "cd packages/core-bun && bun run build",
        "test": "cd packages/core-bun && bun test",
        "lint": "cd packages/core-bun && bun run lint"
      }
    },
    "data-ingestion": {
      "path": "services/data-ingestion",
      "language": "python",
      "deps": ["core-python"],
      "commands": {
        "build": "cd services/data-ingestion && uv sync",
        "test": "cd services/data-ingestion && uv run pytest"
      }
    },
    "strategy-engine": {
      "path": "services/strategy-engine",
      "language": "rust",
      "deps": ["core-rust", "core-python"],
      "commands": {
        "build": "cd services/strategy-engine && cargo build --release",
        "test": "cd services/strategy-engine && cargo test"
      }
    },
    "execution-gateway": {
      "path": "services/execution-gateway",
      "language": "bun",
      "deps": ["core-bun"],
      "commands": {
        "build": "cd services/execution-gateway && bun run build",
        "test": "cd services/execution-gateway && bun test"
      }
    }
  }
}
```

### scripts/affected.sh

```bash
#!/usr/bin/env bash
set -euo pipefail

# Affected target detection for polyglot monorepo
# Usage: ./scripts/affected.sh [list|test|build|lint]

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGETS_FILE="$REPO_ROOT/targets.json"
ACTION="${1:-list}"

# Get changed files relative to origin/main
get_changed_files() {
    git diff --name-only origin/main 2>/dev/null || git diff --name-only HEAD~1
}

# Map a file path to its owning target
file_to_target() {
    local file="$1"
    jq -r --arg file "$file" '
        .targets | to_entries[] |
        select(.value.path as $p | $file | startswith($p)) |
        .key
    ' "$TARGETS_FILE"
}

# Get all targets that depend on a given target (transitively)
get_dependents() {
    local target="$1"
    jq -r --arg target "$target" '
        .targets | to_entries[] |
        select(.value.deps | index($target)) |
        .key
    ' "$TARGETS_FILE"
}

# Collect all affected targets
get_affected_targets() {
    local -A affected=()

    while IFS= read -r file; do
        local target
        target=$(file_to_target "$file")
        if [[ -n "$target" ]]; then
            affected["$target"]=1
            # Add transitive dependents
            while IFS= read -r dep; do
                [[ -n "$dep" ]] && affected["$dep"]=1
            done < <(get_dependents "$target")
        fi
    done < <(get_changed_files)

    printf '%s\n' "${!affected[@]}" | sort
}

# Execute command for affected targets
run_for_affected() {
    local cmd="$1"
    while IFS= read -r target; do
        [[ -z "$target" ]] && continue
        local command
        command=$(jq -r --arg t "$target" --arg c "$cmd" '.targets[$t].commands[$c] // empty' "$TARGETS_FILE")
        if [[ -n "$command" ]]; then
            echo "▶ [$target] $cmd"
            eval "$command" || echo "⚠ [$target] $cmd failed"
        fi
    done < <(get_affected_targets)
}

case "$ACTION" in
    list)
        echo "Affected targets:"
        get_affected_targets | while read -r t; do echo "  - $t"; done
        ;;
    test|build|lint)
        run_for_affected "$ACTION"
        ;;
    *)
        echo "Usage: $0 [list|test|build|lint]"
        exit 1
        ;;
esac
```

### .gitignore

```gitignore
# Logs
logs/
*.jsonl

# Dependencies
node_modules/
target/
.venv/
__pycache__/
*.pyc

# Build outputs
dist/
build/
*.egg-info/

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db

# Mise
.mise.local.toml

# Secrets (never commit)
.env.local
*.key
*.pem
```

---

## Phase 7: Verification Checklist

After creating all files, verify the setup:

```bash
# 1. Directory structure
find . -name "CLAUDE.md" -o -name "SKILL.md" | head -20

# 2. Mise configuration
mise doctor
mise tasks

# 3. Affected detection
chmod +x scripts/affected.sh
./scripts/affected.sh list

# 4. MCP configuration
cat .mcp.json | jq .

# 5. Log directory
mkdir -p logs
ls -la logs/

# 6. Git status
git status
git add -A
git commit -m "chore: initial monorepo scaffold with CLAUDE.md hub-and-spoke"
```

---

## Phase 8: Post-Bootstrap Tasks

Once the scaffold is complete, initialize each package:

### Python Package

```bash
cd packages/core-python
uv init
uv add loguru platformdirs pydantic
uv add --dev pytest pytest-asyncio ruff
```

### Rust Package

```bash
cd packages/core-rust
cargo init --lib
# Add dependencies to Cargo.toml per skill guide
```

### Bun Package

```bash
cd packages/core-bun
bun init -y
bun add pino zod
bun add -d @biomejs/biome @types/bun
```

### Install Code Search

```bash
cargo install ck-search
ck index --all  # Index the codebase
```

---

## Success Criteria

The bootstrap is complete when:

- [ ] All `CLAUDE.md` files exist and link correctly
- [ ] `mise tasks` shows all defined tasks
- [ ] `./scripts/affected.sh list` runs without error
- [ ] `.mcp.json` is valid JSON
- [ ] Each package has initialized tooling
- [ ] `logs/` directory exists (gitignored)
- [ ] Initial commit is made

---

## Maintenance Protocol

When adding new packages:

1. Create directory under `packages/` or `services/`
2. Add `CLAUDE.md` following existing pattern
3. Update `targets.json` with new target entry
4. Link from root `CLAUDE.md` package map
5. Run `mise run reindex` for code search
