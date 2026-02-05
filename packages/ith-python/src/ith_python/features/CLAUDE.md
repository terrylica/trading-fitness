# features/

> Layer 1: ITH Feature Computation (wraps Rust metrics).

**← [Back to ith-python](../../../CLAUDE.md)**

## Purpose

Compute ITH features from NAV series using Rust bindings (metrics-rust).

---

## Module Map

| Module       | Purpose                                       |
| ------------ | --------------------------------------------- |
| `compute.py` | Feature computation pipeline                  |
| `config.py`  | Feature configuration (lookbacks, thresholds) |

---

## Quick Start

```python
from ith_python.features import compute_features, FeatureConfig

config = FeatureConfig(
    lookbacks=[20, 50, 100, 200, 500],
    threshold_dbps=25,
)
features_df = compute_features(nav_series, config)
```

---

## Feature Naming Convention

```
ith_rb{threshold}_lb{lookback}_{feature}

Example: ith_rb25_lb100_bull_ed
  - rb25: Range bar 25 dbps
  - lb100: 100-bar lookback
  - bull_ed: Bull epoch density
```

---

## 8 Features per Lookback

| Feature   | Range  | Description                |
| --------- | ------ | -------------------------- |
| `bull_ed` | [0, 1] | Bull epoch density         |
| `bear_ed` | [0, 1] | Bear epoch density         |
| `bull_eg` | [0, 1] | Bull excess gain (tanh)    |
| `bear_eg` | [0, 1] | Bear excess gain (tanh)    |
| `bull_cv` | [0, 1] | Bull intervals CV          |
| `bear_cv` | [0, 1] | Bear intervals CV          |
| `max_dd`  | [0, 1] | Maximum drawdown in window |
| `max_ru`  | [0, 1] | Maximum runup in window    |

---

## Related Documentation

| Document                                                              | Purpose                  |
| --------------------------------------------------------------------- | ------------------------ |
| [ith-python CLAUDE.md](../../../CLAUDE.md)                            | Parent package           |
| [metrics-rust CLAUDE.md](../../../../metrics-rust/CLAUDE.md)          | Rust implementation      |
| [docs/features/REGISTRY.md](../../../../../docs/features/REGISTRY.md) | Feature definitions SSoT |
| [storage/](../storage/)                                               | Layer 2 (storage)        |
