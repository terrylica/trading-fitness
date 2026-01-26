# ith-python

> Primary Python package for ITH (Investment Time Horizon) fitness analysis.

**← [Back to trading-fitness](../../CLAUDE.md)**

## Quick Start

```bash
uv sync                           # Install dependencies
uv run python -m ith_python.ith   # Run analysis
```

## Architecture: Multi-View Feature Architecture

This package implements a 3-layer separation of concerns:

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Layer 1: Compute │───▶│ Layer 2: Storage │───▶│ Layer 3: Analysis│
│   features/      │    │   storage/       │    │   analysis/      │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

| Layer | Module                                  | Purpose                            | Upgrade Path                |
| ----- | --------------------------------------- | ---------------------------------- | --------------------------- |
| 1     | [`features/`](src/ith_python/features/) | ITH feature computation            | `mise run upgrade:features` |
| 2     | [`storage/`](src/ith_python/storage/)   | Long Format SSoT + view generators | `mise run upgrade:views`    |
| 3     | [`analysis/`](src/ith_python/analysis/) | Statistical evaluation + telemetry | `mise run upgrade:analysis` |

**Plan**: [docs/plans/2026-01-25-multi-view-feature-architecture-plan.md](../../docs/plans/2026-01-25-multi-view-feature-architecture-plan.md)

## Module Structure

| Module                     | Purpose                                      |
| -------------------------- | -------------------------------------------- |
| `features/`                | Layer 1: Feature computation (wraps Rust)    |
| `storage/`                 | Layer 2: FeatureStore + view generators      |
| `analysis/`                | Layer 3: Statistical analysis orchestrator   |
| `ith.py`                   | Main Bull ITH analysis script                |
| `bear_ith.py`              | Bear ITH analysis for short positions        |
| `bull_ith_numba.py`        | Numba-accelerated Bull (long) calculations   |
| `bear_ith_numba.py`        | Numba-accelerated Bear (short) calculations  |
| `ndjson_logger.py`         | NDJSON structured logging with provenance    |
| `telemetry/`               | Reproducibility and forensic telemetry       |
| `statistical_examination/` | Legacy analysis (deprecated → use analysis/) |
| `validate_edge_cases.py`   | Visual validation PNG generation             |
| `paths.py`                 | Repository-local path configuration          |

## Submodules

| Submodule                                                                    | Purpose                                     | Docs                                                  |
| ---------------------------------------------------------------------------- | ------------------------------------------- | ----------------------------------------------------- |
| [features/](src/ith_python/features/)                                        | Feature computation (Layer 1)               | See Architecture above                                |
| [storage/](src/ith_python/storage/)                                          | FeatureStore with view generators (Layer 2) | See Architecture above                                |
| [analysis/](src/ith_python/analysis/)                                        | Statistical analysis (Layer 3)              | See Architecture above                                |
| [statistical_examination/](src/ith_python/statistical_examination/CLAUDE.md) | Feature redundancy, stability, ML readiness | [→](src/ith_python/statistical_examination/CLAUDE.md) |
| telemetry/                                                                   | Provenance tracking, event logging          | See below                                             |

### Telemetry Module

The `telemetry/` module provides scientific reproducibility infrastructure:

- **provenance.py**: `ProvenanceContext`, `fingerprint_array()`, `capture_random_state()`
- **events.py**: `log_data_load()`, `log_algorithm_init()`, `log_epoch_detected()`

Enable epoch telemetry:

```python
from ith_python.bull_ith_numba import bull_excess_gain_excess_loss
result = bull_excess_gain_excess_loss(nav, hurdle=0.05, emit_telemetry=True)
```

## Key Concepts

- **TMAEG**: Target Maximum Acceptable Excess Gain (drawdown threshold for longs)
- **TMAER**: Target Maximum Acceptable Excess Runup (runup threshold for shorts)
- **Bull ITH Epochs**: Periods where long positions exceed TMAEG threshold
- **Bear ITH Epochs**: Periods where short positions exceed TMAER threshold
- **Fitness Criteria**: Epoch count, Sharpe ratio, coefficient of variation
- **Symmetry**: Bull and Bear algorithms are mathematical inverses

## Dependencies

Core: pandas, numpy, plotly, scipy, numba, loguru, rich, kaleido
Dev: pytest, ruff

## Related Documentation

- **Root Overview**: [← trading-fitness](../../CLAUDE.md)
- **ITH Methodology**: [docs/ITH.md](../../docs/ITH.md)
- **E2E Forensic Pipeline**: [docs/forensic/E2E.md](../../docs/forensic/E2E.md)
- **BiLSTM Metrics**: [metrics-rust](../metrics-rust/CLAUDE.md) (Python bindings available)
- **Rust ITH**: [core-rust](../core-rust/CLAUDE.md)
