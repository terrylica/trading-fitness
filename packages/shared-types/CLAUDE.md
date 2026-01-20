# shared-types

> Cross-language type definitions for trading-fitness packages.

**← [Back to trading-fitness](../../CLAUDE.md)**

## Status

Schemas defined. Ready for code generation when needed.

## Purpose

- Define shared data structures across Python, Rust, and TypeScript
- JSON Schema (Draft 2020-12) for validation
- Code generation for type-safe interfaces

## Structure

```
shared-types/
└── schemas/
    ├── nav-record.json       # NAV data record format
    ├── ith-result.json       # ITH analysis result
    └── fitness-metrics.json  # Strategy fitness metrics
```

## Schema Reference

| Schema                 | Purpose                                   |
| ---------------------- | ----------------------------------------- |
| `nav-record.json`      | NAV data record (date, nav, optional pnl) |
| `ith-result.json`      | Full ITH analysis result with metadata    |
| `fitness-metrics.json` | Strategy fitness metrics (SR, MDD, etc.)  |

## Usage

Schemas will be consumed by:

- **Python**: pydantic models via `datamodel-codegen`
- **Rust**: serde structs via `typify`
- **TypeScript**: zod schemas via `json-schema-to-zod`

## Code Generation

Use mise tasks to generate types from schemas:

```bash
mise run generate-types           # Generate all targets
mise run generate-types:python    # Generate Python pydantic models
mise run generate-types:typescript # Generate TypeScript zod schemas
mise run generate-types:rust      # Generate Rust serde structs
```

Or run the script directly:

```bash
./scripts/generate-types.sh all
./scripts/generate-types.sh python
./scripts/generate-types.sh typescript
./scripts/generate-types.sh rust
```

### Prerequisites

- **Python**: `uv pip install datamodel-code-generator`
- **TypeScript**: `bun add -D json-schema-to-zod`
- **Rust**: `cargo install typify-cli`
