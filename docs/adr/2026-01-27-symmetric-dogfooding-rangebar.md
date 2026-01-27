# ADR: Symmetric Dogfooding with rangebar-py

**Date**: 2026-01-27
**Status**: Accepted
**Pattern**: quality-tools:symmetric-dogfooding

## Context

trading-fitness depends on rangebar-py for range bar data. To ensure integration stability, we implement symmetric dogfooding where trading-fitness validates its ITH metrics against real range bars from rangebar-py.

## Decision

1. **Pin rangebar-py to explicit tags** (not `branch = "main"`)
2. **Create integration tests** in `packages/ith-python/tests/integration/`
3. **Add mise task** `validate:symmetric` for pre-release validation
4. **Document pre-release protocol** in CLAUDE.md

## Integration Surface

```
trading-fitness                      rangebar-py
─────────────────                    ───────────
IMPORTS:                             EXPORTS:
- rangebar.get_range_bars()          - Range bar construction
- rangebar.RangeBarCache             - ClickHouse cache layer
- Real Binance market data           - Microstructure features

VALIDATES:
- ITH metrics on real range bars
- Feature bounds [0,1]
- Warmup handling
```

## Pre-Release Protocol

Before releasing trading-fitness:

1. Run `mise run validate:symmetric`
2. Verify ITH features work on latest rangebar-py tag
3. Update version pin if needed

## Consequences

- Breaking changes in rangebar-py are caught before trading-fitness release
- Version pins prevent surprise breaks from upstream changes
- Integration tests provide regression safety
