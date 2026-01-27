# E2E Forensic Analysis Pipeline

> Comprehensive ITH feature validation with NDJSON telemetry.

**[Back to CLAUDE.md](../../CLAUDE.md)**

## Quick Start

```bash
mise run forensic:e2e              # Full pipeline with ClickHouse data
mise run forensic:hypothesis-audit # Audit hypothesis test results
mise run forensic:report           # Generate report from latest run
```

---

## Pipeline Overview

### New Architecture (Multi-View)

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Layer 1: Compute │───▶│ Layer 2: Storage │───▶│ Layer 3: Analysis│
│   features/      │    │   storage/       │    │   analysis/      │
│                  │    │                  │    │                  │
│ Rust multiscale  │    │ Long Format SSoT │    │ Statistical eval │
│ ITH features     │    │ + View Generators│    │ + NDJSON emit    │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

**Full Pipeline**: `mise run forensic:pipeline`

**Independent Upgrades**:

- `mise run upgrade:features` - Rebuild features only
- `mise run upgrade:views` - Regenerate views only
- `mise run upgrade:analysis` - Re-run analysis only

**Plan**: [docs/plans/2026-01-25-multi-view-feature-architecture-plan.md](../plans/2026-01-25-multi-view-feature-architecture-plan.md)

### Legacy Pipeline

```
ClickHouse → rangebar-py → NAV → Rust compute_multiscale_ith → Polars → Parquet
                                        ↓
                          Statistical Examination (Python)
                                        ↓
                          NDJSON Telemetry + summary.json
```

**Legacy**: `mise run forensic:e2e`

---

## Data Structure: Long Format SSoT (New)

The new architecture uses a **Long Format** as the Single Source of Truth:

```
┌───────────┬─────────┬────────────┬─────────┬─────────────┬───────┬───────┐
│ bar_index │ symbol  │ threshold  │ lookback│ feature     │ value │ valid │
│ UInt32    │ Cat     │ UInt16     │ UInt16  │ Cat         │ Float │ Bool  │
└───────────┴─────────┴────────────┴─────────┴─────────────┴───────┴───────┘
```

**Benefits**:

- Zero structural sparsity (vs 75% in wide format)
- View generators for different use cases
- Independent upgradeable layers

**View Generators** (from `FeatureStore`):

| Method            | Output                 | Use Case               |
| ----------------- | ---------------------- | ---------------------- |
| `to_wide()`       | Wide DataFrame         | ML training            |
| `to_nested()`     | Nested JSON            | Semantic queries, APIs |
| `to_dense()`      | Single-threshold dense | Per-threshold analysis |
| `to_clickhouse()` | Arrow RecordBatch      | Analytics DB insertion |

**Example**:

```python
from ith_python.storage import FeatureStore

store = FeatureStore.from_parquet("artifacts/ssot/features_long.parquet")
wide_df = store.to_wide(threshold=25, drop_warmup=True)
```

---

## Data Structure: Sparse Wide Table (Legacy)

The E2E pipeline produces a **sparse wide table** with diagonal structure:

```
┌─────────────────┬────────────┬───────────────┬───────────────┬─────────────┐
│ threshold_dbps  │ symbol     │ ith_rb25_*    │ ith_rb50_*    │ ith_rb100_* │
├─────────────────┼────────────┼───────────────┼───────────────┼─────────────┤
│ 25              │ BTCUSDT    │ ✓ VALUES      │ null          │ null        │
│ 25              │ ETHUSDT    │ ✓ VALUES      │ null          │ null        │
│ 50              │ BTCUSDT    │ null          │ ✓ VALUES      │ null        │
│ 50              │ ETHUSDT    │ null          │ ✓ VALUES      │ null        │
│ 100             │ BTCUSDT    │ null          │ null          │ ✓ VALUES    │
│ 100             │ ETHUSDT    │ null          │ null          │ ✓ VALUES    │
└─────────────────┴────────────┴───────────────┴───────────────┴─────────────┘
```

### Why 75% Sparsity?

This is **by design**, not a bug:

1. Each threshold produces 40 columns (8 metrics × 5 lookbacks)
2. With 4 thresholds, total columns = 160
3. Each row only has values for its OWN threshold's columns
4. Sparsity = 3/4 = 75% (expected)

### Column Naming Convention

```
ith_rb{threshold}_lb{lookback}_{feature}

Example: ith_rb25_lb100_bull_ed
  - rb25: Range bar 25 dbps (for identification only)
  - lb100: 100-bar lookback window
  - bull_ed: Bull epoch density feature
```

### Critical Understanding

The `threshold_dbps` in the column name is for **identification only**. The actual TMAEG used in computation is **auto-calculated** from data volatility using:

```
tmaeg = 3.0 × MAD_std × sqrt(lookback), clamped to [0.001, 0.50]
```

This means all thresholds produce **identical feature values** when computed from the same NAV. The differentiation comes from fetching range bars at different thresholds (which produce different NAV series).

---

## Artifacts

### features.parquet

| Field             | Type    | Description                            |
| ----------------- | ------- | -------------------------------------- |
| symbol            | String  | Trading pair (BTCUSDT, ETHUSDT)        |
| threshold_dbps    | Int32   | Range bar threshold for identification |
| ith*rb*\_lb\*\*\* | Float64 | 160 feature columns (40 per threshold) |

**Example dimensions**: 16,000 rows × 162 columns (2 symbols × 4 thresholds × 2000 bars)

### summary.json

```json
{
  "timestamp": "2026-01-26T01:08:28.687765+00:00",
  "trace_id": "ad8a725cc026464c",
  "provenance": [
    {
      "symbol": "BTCUSDT",
      "n_bars": 2000,
      "nav_hash": "fb09feda73dfe320",
      "nav_range": [0.916, 1.020]
    }
  ],
  "data": {
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "thresholds": [25, 50, 100, 250],
    "lookbacks": [20, 50, 100, 200, 500],
    "total_rows": 16000,
    "total_feature_cols": 160
  },
  "distribution": { ... },
  "regime": { ... },
  "pca": { ... },
  "temporal": { ... }
}
```

---

## Telemetry Events

The pipeline emits NDJSON events to `logs/ndjson/`:

### Event Types

| Event Type        | Count | Typical Decision       |
| ----------------- | ----- | ---------------------- |
| hypothesis_result | 279   | Various (see below)    |
| data.load         | 2     | Per symbol             |
| algorithm.init    | 8     | Per threshold × symbol |

### Hypothesis Test Results

| Test Name          | Count | Decision                          |
| ------------------ | ----- | --------------------------------- |
| shapiro_wilk       | 208   | 100% non_normal                   |
| mann_whitney_u     | 50    | 100% regime_invariant             |
| ridge_vif          | 9     | 100% acceptable_multicollinearity |
| pca_dimensionality | 7     | 100% low_redundancy               |
| temporal_structure | 5     | 100% mixed_stationarity           |

---

## Warmup NaN Analysis

Each lookback window requires (lookback - 1) bars of history before producing valid values:

| Lookback | NaN Count | Expected (2 symbols × (lb-1)) |
| -------- | --------- | ----------------------------- |
| lb20     | 38        | 38 ✓                          |
| lb50     | 98        | 98 ✓                          |
| lb100    | 198       | 198 ✓                         |
| lb200    | 398       | 398 ✓                         |
| lb500    | 998       | 998 ✓                         |

---

## Feature Statistics (lb100 @ any threshold)

All features are bounded [0, 1]:

| Feature | Mean   | Std    | Range           |
| ------- | ------ | ------ | --------------- |
| bull_ed | 0.0069 | 0.0005 | [0.0067, 0.011] |
| bear_ed | 0.0074 | 0.0012 | [0.0067, 0.018] |
| max_dd  | 0.0141 | 0.0066 | [0.004, 0.038]  |
| max_ru  | 0.0101 | 0.0040 | [0.003, 0.025]  |
| bull_cv | 0.1368 | 0.0878 | [0.12, 0.92]    |
| bear_cv | 0.2009 | 0.1836 | [0.12, 0.96]    |

---

## Inter-Feature Correlations

Key correlation findings:

| Pair              | Correlation | Interpretation                         |
| ----------------- | ----------- | -------------------------------------- |
| bear_ed ↔ max_dd  | r=0.77      | High redundancy - drop one             |
| bull_ed ↔ bull_cv | r=0.77      | High redundancy - drop one             |
| bull_ed ↔ bear_ed | r=-0.25     | Weak negative (asymmetric)             |
| max_dd ↔ max_ru   | r=-0.41     | Moderate negative (inverse)            |
| BTC ↔ ETH         | r≈0.03      | Independent (good for diversification) |

---

## Usage for ML Training

### Filter by Threshold

```python
import polars as pl

df = pl.read_parquet('artifacts/forensic/e2e-*/features.parquet')

# Get dense data for threshold=25
df_25 = df.filter(pl.col('threshold_dbps') == 25)
feature_cols = [c for c in df.columns if c.startswith('ith_rb25_')]
X = df_25.select(feature_cols).to_numpy()
```

### Handle NaN Warmup

```python
# Option 1: Drop warmup rows
df_valid = df_25.drop_nulls(subset=feature_cols)

# Option 2: Use NaN-aware models (XGBoost, LightGBM)

# Option 3: Mask for variable-length sequences
mask = ~df_25.select(feature_cols).to_numpy().isnan()
```

---

## Troubleshooting

### "75% of data is NULL"

This is expected. See [Data Structure](#data-structure-sparse-wide-table) above.

### "All thresholds have identical values"

This is correct if all thresholds were computed from the **same NAV**. The `threshold_dbps` parameter in `MultiscaleIthConfig` is for column naming only.

To get different feature values, fetch range bars at different thresholds BEFORE computing ITH.

### "Regime length mismatch"

The regime detection uses a lookback window that may not align with the feature DataFrame length. Check the NAV length vs DataFrame length in the pipeline.

---

## Related Documentation

- [CLAUDE.md](../../CLAUDE.md) - Project overview
- [ITH.md](../ITH.md) - Core methodology
- [statistical_examination/CLAUDE.md](../../packages/ith-python/src/ith_python/statistical_examination/CLAUDE.md) - ML readiness analysis

---

## Forensic Audit Reports

| Report                                                               | Date       | Focus                                     |
| -------------------------------------------------------------------- | ---------- | ----------------------------------------- |
| [COMPREHENSIVE_AUDIT_20260125.md](./COMPREHENSIVE_AUDIT_20260125.md) | 2026-01-25 | Full artifact and telemetry inventory     |
| [SYMMETRIC_AUDIT_20260125.md](./SYMMETRIC_AUDIT_20260125.md)         | 2026-01-25 | Symmetric feature selection (24 features) |
| [ANALYSIS_REPORT_20260125.md](./ANALYSIS_REPORT_20260125.md)         | 2026-01-25 | Initial analysis with pipeline fixes      |
