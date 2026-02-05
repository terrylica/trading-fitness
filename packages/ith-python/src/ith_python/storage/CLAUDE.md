# storage/

> Layer 2: Long Format SSoT + View Generators for ITH features.

**← [Back to ith-python](../../../CLAUDE.md)**

## Purpose

Central storage layer that:

1. Maintains features in **Long Format** (row-per-feature) as SSoT
2. Generates views (Sparse Wide, Dense Narrow) on demand
3. Validates data with Polars schemas

---

## Module Map

| Module       | Purpose                                     |
| ------------ | ------------------------------------------- |
| `store.py`   | FeatureStore class - main interface         |
| `views.py`   | View generators (sparse_wide, dense_narrow) |
| `schemas.py` | Polars schema definitions                   |

---

## Quick Start

```python
from ith_python.storage import FeatureStore

store = FeatureStore(long_format_df)
sparse_wide = store.to_sparse_wide()
dense_narrow = store.to_dense_narrow(lookbacks=[20, 50, 100])
```

---

## Long Format (SSoT)

The canonical storage format is **Long Format** - one row per feature value:

| Column      | Type    | Description                                   |
| ----------- | ------- | --------------------------------------------- |
| `bar_index` | Int64   | Bar position in sequence                      |
| `feature`   | String  | Feature name (e.g., `ith_rb25_lb100_bull_ed`) |
| `value`     | Float64 | Feature value [0, 1]                          |
| `lookback`  | Int64   | Lookback window size                          |

---

## View Formats

| View         | Shape              | Use Case                      |
| ------------ | ------------------ | ----------------------------- |
| Sparse Wide  | (bars × features)  | ML training, full feature set |
| Dense Narrow | (bars × lookbacks) | Single feature analysis       |

---

## Related Documentation

| Document                                                                                                  | Purpose            |
| --------------------------------------------------------------------------------------------------------- | ------------------ |
| [ith-python CLAUDE.md](../../../CLAUDE.md)                                                                | Parent package     |
| [3-layer architecture plan](../../../../../docs/plans/2026-01-25-multi-view-feature-architecture-plan.md) | Architecture       |
| [features/](../features/)                                                                                 | Layer 1 (compute)  |
| [analysis/](../analysis/)                                                                                 | Layer 3 (analysis) |
