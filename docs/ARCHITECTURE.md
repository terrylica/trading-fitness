# Architecture

## Overview

trading-fitness is a polyglot monorepo for trading strategy fitness analysis.

## Package Structure

```
packages/
├── ith-python/    # PRIMARY: ITH fitness analysis
├── core-rust/     # PLACEHOLDER: performance-critical code
├── core-bun/      # PLACEHOLDER: async I/O, APIs
└── shared-types/  # Cross-language type definitions
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
