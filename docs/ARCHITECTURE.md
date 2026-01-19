# Architecture

## Overview

trading-fitness is a polyglot monorepo for trading strategy fitness analysis.

## Package Structure

```
packages/
├── ith-python/    # PRIMARY: ITH fitness analysis (Python + Numba)
├── core-rust/     # Performance-critical code (native)
├── core-bun/      # Async I/O, APIs (TypeScript/Bun)
└── shared-types/  # Cross-language type definitions (JSON Schema)
```

## Data Flow

1. Input: CSV files in `data/nav_data_custom/` (Date, NAV columns)
2. Processing: `ith-python` analyzes NAV data for ITH fitness
3. Output: HTML visualizations and CSV results in `artifacts/`

## Technology Stack

| Layer           | Technology | Purpose                                     |
| --------------- | ---------- | ------------------------------------------- |
| Runtime mgmt    | mise       | Version control for Python, Rust, Node, Bun |
| Python pkg      | uv         | Fast Python package management              |
| Code search     | ast-grep   | Structural code search                      |
| Semantic search | ck         | Code knowledge search                       |
