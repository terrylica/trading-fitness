# ith-python

> Primary Python package for ITH (Investment Time Horizon) fitness analysis.

**← [Back to trading-fitness](../../CLAUDE.md)**

## Quick Start

```bash
uv sync                           # Install dependencies
uv run pytest                     # Run tests
uv run python -m ith_python.ith   # Run Bull ITH analysis
```

---

## Architecture: 3-Layer Feature Pipeline

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Layer 1: Compute │───▶│ Layer 2: Storage │───▶│ Layer 3: Analysis│
│   features/      │    │   storage/       │    │   analysis/      │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

| Layer | Module      | Purpose                      | Upgrade Command             |
| ----- | ----------- | ---------------------------- | --------------------------- |
| 1     | `features/` | ITH feature computation      | `mise run upgrade:features` |
| 2     | `storage/`  | Long Format SSoT + views     | `mise run upgrade:views`    |
| 3     | `analysis/` | Statistical eval + telemetry | `mise run upgrade:analysis` |

**Plan**: [docs/plans/2026-01-25-multi-view-feature-architecture-plan.md](../../docs/plans/2026-01-25-multi-view-feature-architecture-plan.md)

---

## Module Map

| Module                     | Purpose                                    |
| -------------------------- | ------------------------------------------ |
| `features/`                | Layer 1: Feature computation (wraps Rust)  |
| `storage/`                 | Layer 2: FeatureStore + view generators    |
| `analysis/`                | Layer 3: Statistical analysis orchestrator |
| `ith.py`                   | Main Bull ITH analysis script              |
| `bear_ith.py`              | Bear ITH analysis for short positions      |
| `bull_ith_numba.py`        | Numba-accelerated Bull calculations        |
| `bear_ith_numba.py`        | Numba-accelerated Bear calculations        |
| `telemetry/`               | Provenance tracking, event logging         |
| `statistical_examination/` | Statistical methods (SSoT for analysis/)   |
| `ndjson_logger.py`         | NDJSON structured logging                  |

### Submodule Documentation

| Submodule                  | CLAUDE.md                                             |
| -------------------------- | ----------------------------------------------------- |
| `features/`                | [→](src/ith_python/features/CLAUDE.md)                |
| `storage/`                 | [→](src/ith_python/storage/CLAUDE.md)                 |
| `telemetry/`               | [→](src/ith_python/telemetry/CLAUDE.md)               |
| `statistical_examination/` | [→](src/ith_python/statistical_examination/CLAUDE.md) |
| `provenance/`              | [→](src/ith_python/provenance/CLAUDE.md) _(Part B)_   |

---

## Key Concepts

| Concept      | Description                                         |
| ------------ | --------------------------------------------------- |
| **TMAEG**    | Target Maximum Acceptable Excess Gain (for longs)   |
| **TMAER**    | Target Maximum Acceptable Excess Runup (for shorts) |
| **Bull ITH** | Periods where long positions exceed TMAEG           |
| **Bear ITH** | Periods where short positions exceed TMAER          |
| **Symmetry** | Bull and Bear algorithms are mathematical inverses  |

**Deep Dive**: [docs/ITH.md](../../docs/ITH.md)

---

## Telemetry

The `telemetry/` module provides scientific reproducibility:

- **provenance.py**: `ProvenanceContext`, `fingerprint_array()`, `capture_random_state()`
- **events.py**: `log_data_load()`, `log_algorithm_init()`, `log_epoch_detected()`

```python
from ith_python.bull_ith_numba import bull_excess_gain_excess_loss
result = bull_excess_gain_excess_loss(nav, hurdle=0.05, emit_telemetry=True)
```

**Contract**: [docs/LOGGING.md](../../docs/LOGGING.md)

---

## Dependencies

**Core**: pandas, numpy, plotly, scipy, numba, loguru, rich, kaleido

**Dev**: pytest, ruff, polars, pandera

**Optional**: rangebar (symmetric dogfooding), scikit-learn, shap, lightgbm

---

## Related Documentation

| Document                                           | Purpose                    |
| -------------------------------------------------- | -------------------------- |
| [docs/ITH.md](../../docs/ITH.md)                   | ITH methodology            |
| [docs/forensic/E2E.md](../../docs/forensic/E2E.md) | E2E pipeline               |
| [docs/LOGGING.md](../../docs/LOGGING.md)           | Telemetry contract         |
| [metrics-rust](../metrics-rust/CLAUDE.md)          | Rust ITH + Python bindings |
