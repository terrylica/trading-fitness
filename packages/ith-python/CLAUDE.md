# ith-python

> Primary Python package for ITH (Investment Time Horizon) fitness analysis.

**← [Back to trading-fitness](../../CLAUDE.md)**

## Quick Start

```bash
uv sync                           # Install dependencies
uv run python -m ith_python.ith   # Run analysis
```

## Module Structure

| Module                     | Purpose                                     |
| -------------------------- | ------------------------------------------- |
| `ith.py`                   | Main Bull ITH analysis script               |
| `bear_ith.py`              | Bear ITH analysis for short positions       |
| `bull_ith_numba.py`        | Numba-accelerated Bull (long) calculations  |
| `bear_ith_numba.py`        | Numba-accelerated Bear (short) calculations |
| `ndjson_logger.py`         | NDJSON structured logging with provenance   |
| `telemetry/`               | Reproducibility and forensic telemetry      |
| `statistical_examination/` | Multi-scale feature analysis framework      |
| `validate_edge_cases.py`   | Visual validation PNG generation            |
| `paths.py`                 | Repository-local path configuration         |

## Submodules

| Submodule                                                                    | Purpose                                     | Docs                                                  |
| ---------------------------------------------------------------------------- | ------------------------------------------- | ----------------------------------------------------- |
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
- **BiLSTM Metrics**: [metrics-rust](../metrics-rust/CLAUDE.md) (Python bindings available)
- **Rust ITH**: [core-rust](../core-rust/CLAUDE.md)
