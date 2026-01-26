# Multi-View Feature Architecture with Separation of Concerns

## Goal

Redesign the E2E forensic pipeline with:
1. **Robust, Versatile Data Storage** - Long format SSoT with multiple view generators
2. **Clean Separation of Concerns** - Feature computation, storage, analysis as independent modules
3. **Mise Orchestration** - Independent upgrades via `mise run` tasks
4. **Claude Code Forensics** - NDJSON telemetry enables AI-assisted post-hoc analysis

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              mise.toml Orchestration                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │ Feature Compute  │    │  Feature Store   │    │ Feature Analysis │       │
│  │    (Rust/Py)     │───▶│   (Long SSoT)    │───▶│    (Python)      │       │
│  │                  │    │                  │    │                  │       │
│  │ metrics-rust/    │    │ artifacts/ssot/  │    │ statistical_     │       │
│  │ ith_multiscale   │    │ features.parquet │    │ examination/     │       │
│  └──────────────────┘    └────────┬─────────┘    └────────┬─────────┘       │
│                                   │                       │                  │
│                          ┌────────▼─────────┐    ┌────────▼─────────┐       │
│                          │   View Generators │    │    Telemetry     │       │
│                          │                  │    │    (NDJSON)      │       │
│                          │ • to_wide()      │    │                  │       │
│                          │ • to_nested()    │    │ logs/ndjson/     │       │
│                          │ • to_clickhouse()│    │ • hypothesis     │       │
│                          └──────────────────┘    │ • decisions      │       │
│                                                  │ • provenance     │       │
│                                                  └──────────────────┘       │
│                                                           │                  │
│                                                  ┌────────▼─────────┐       │
│                                                  │  Claude Code     │       │
│                                                  │  Forensic Audit  │       │
│                                                  └──────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Separation

### Layer 1: Feature Computation (Upgradeable Independently)

**Location**: `packages/metrics-rust/` + `packages/ith-python/src/ith_python/features/`

**Responsibility**: Compute raw ITH features from NAV arrays

**Interface Contract**:
```python
# Input: NAV array + config
# Output: Dict[feature_name, np.ndarray]

def compute_features(
    nav: np.ndarray,
    config: FeatureConfig,
) -> dict[str, np.ndarray]:
    """Compute ITH features. Returns {feature_name: values}."""
```

**Upgrade Path**:
- Add new features → new functions, same interface
- Change normalization → modify existing functions
- No storage/analysis changes required

---

### Layer 2: Feature Storage (Long Format SSoT)

**Location**: `packages/ith-python/src/ith_python/storage/`

**Responsibility**: Canonical storage + view generation

**Schema (Long Format)**:
```
┌───────────┬─────────┬────────────┬─────────┬─────────────┬───────┬───────┐
│ bar_index │ symbol  │ threshold  │ lookback│ feature     │ value │ valid │
│ UInt32    │ Cat     │ UInt16     │ UInt16  │ Cat         │ Float │ Bool  │
└───────────┴─────────┴────────────┴─────────┴─────────────┴───────┴───────┘
```

**View Generators**:
```python
class FeatureStore:
    """SSoT with view generators."""

    def __init__(self, path: Path):
        self.df = pl.read_parquet(path)

    # View A: Wide format for ML
    def to_wide(self, threshold: int = None) -> pl.DataFrame: ...

    # View B: Nested JSON for semantic queries
    def to_nested(self) -> list[dict]: ...

    # View C: ClickHouse batch for analytics DB
    def to_clickhouse(self) -> pa.RecordBatch: ...

    # View D: Per-threshold dense (no sparsity)
    def to_dense(self, threshold: int) -> pl.DataFrame: ...
```

**Upgrade Path**:
- Add new views → new methods
- Change schema → migration function
- No compute/analysis changes required

---

### Layer 3: Feature Analysis (Upgradeable Independently)

**Location**: `packages/ith-python/src/ith_python/analysis/`

**Responsibility**: Statistical evaluation of features

**Interface Contract**:
```python
# Input: FeatureStore (reads from SSoT)
# Output: AnalysisResults + NDJSON telemetry

def analyze_features(
    store: FeatureStore,
    config: AnalysisConfig,
) -> AnalysisResults:
    """Run statistical analysis. Emits NDJSON telemetry."""
```

**Sub-Modules** (each upgradeable independently):
```
analysis/
├── __init__.py           # Public API
├── distribution.py       # Shapiro-Wilk, Beta fit, AD test
├── dimensionality.py     # PCA, VIF, Participation Ratio
├── regime.py             # Mann-Whitney, Cliff's Delta
├── temporal.py           # ADF, ACF, stationarity
├── cross_scale.py        # Spearman across lookbacks
├── stability.py          # CV across thresholds
└── selection.py          # Feature selection pipeline
```

**Upgrade Path**:
- Add new statistical tests → new module
- Improve existing tests → modify module
- No compute/storage changes required

---

### Layer 4: Telemetry (NDJSON for Claude Code)

**Location**: `logs/ndjson/`

**Responsibility**: Machine-readable audit trail for AI forensics

**Event Schema**:
```json
{
  "ts": "2026-01-25T14:30:52Z",
  "event_type": "hypothesis_result",
  "trace_id": "abc123",
  "module": "distribution",
  "test_name": "shapiro_wilk",
  "feature": "bull_ed",
  "lookback": 100,
  "threshold": 25,
  "statistic": 0.847,
  "p_value": 0.001,
  "decision": "non_normal"
}
```

**Claude Code Integration**:
```bash
# AI can query telemetry
jq 'select(.decision == "non_normal")' logs/ndjson/analysis.jsonl

# Or use Claude Code to analyze patterns
claude "analyze the hypothesis test failures in logs/ndjson/"
```

---

## Mise Task Orchestration

### Task Structure

```toml
# mise.toml

# ============ Layer 1: Feature Computation ============
[tasks."features:compute"]
description = "Compute ITH features from range bar data"
depends = ["develop:metrics-rust"]
run = """
cd packages/ith-python
UV_PYTHON=python3.13 uv run python -m ith_python.features.compute \
    --symbols BTCUSDT,ETHUSDT \
    --thresholds 25,50,100,250 \
    --lookbacks 20,50,100,200,500 \
    --output ../../artifacts/ssot/features_long.parquet
"""

# ============ Layer 2: View Generation ============
[tasks."views:wide"]
description = "Generate wide format views from SSoT"
depends = ["features:compute"]
run = """
cd packages/ith-python
UV_PYTHON=python3.13 uv run python -m ith_python.storage.views \
    --input ../../artifacts/ssot/features_long.parquet \
    --output-dir ../../artifacts/views/wide \
    --format wide
"""

[tasks."views:nested"]
description = "Generate nested JSON views from SSoT"
depends = ["features:compute"]
run = """
cd packages/ith-python
UV_PYTHON=python3.13 uv run python -m ith_python.storage.views \
    --input ../../artifacts/ssot/features_long.parquet \
    --output ../../artifacts/views/nested/features.jsonl \
    --format nested
"""

[tasks."views:all"]
description = "Generate all view formats"
depends = ["views:wide", "views:nested"]

# ============ Layer 3: Analysis ============
[tasks."analysis:distribution"]
description = "Run distribution analysis (Shapiro-Wilk, Beta fit)"
depends = ["features:compute"]
run = """
cd packages/ith-python
UV_PYTHON=python3.13 uv run python -m ith_python.analysis.distribution \
    --input ../../artifacts/ssot/features_long.parquet \
    --output ../../artifacts/analysis/distribution.json \
    --telemetry ../../logs/ndjson/distribution.jsonl
"""

[tasks."analysis:dimensionality"]
description = "Run dimensionality analysis (PCA, VIF)"
depends = ["features:compute"]
run = """
cd packages/ith-python
UV_PYTHON=python3.13 uv run python -m ith_python.analysis.dimensionality \
    --input ../../artifacts/ssot/features_long.parquet \
    --output ../../artifacts/analysis/dimensionality.json \
    --telemetry ../../logs/ndjson/dimensionality.jsonl
"""

[tasks."analysis:all"]
description = "Run all statistical analyses"
depends = ["analysis:distribution", "analysis:dimensionality", "analysis:regime", "analysis:temporal"]

# ============ Layer 4: Full Pipeline ============
[tasks."forensic:e2e"]
description = "Full E2E pipeline: compute → store → analyze"
depends = ["features:compute", "views:all", "analysis:all"]
run = """
echo "✅ E2E Pipeline Complete"
echo "   SSoT: artifacts/ssot/features_long.parquet"
echo "   Views: artifacts/views/"
echo "   Analysis: artifacts/analysis/"
echo "   Telemetry: logs/ndjson/"
"""

# ============ Independent Upgrades ============
[tasks."upgrade:features"]
description = "Rebuild features only (after feature code changes)"
run = "mise run features:compute"

[tasks."upgrade:analysis"]
description = "Re-run analysis only (after analysis code changes)"
depends = ["analysis:all"]

[tasks."upgrade:views"]
description = "Regenerate views only (after view code changes)"
depends = ["views:all"]
```

### Upgrade Workflows

```bash
# After changing feature computation code
mise run upgrade:features
mise run upgrade:analysis  # Re-analyze with new features

# After changing analysis code (e.g., new statistical test)
mise run upgrade:analysis  # Only re-analyze, reuse existing features

# After adding new view format
mise run upgrade:views     # Only regenerate views

# Full rebuild
mise run forensic:e2e
```

---

## Directory Structure (New)

```
packages/ith-python/src/ith_python/
├── features/                    # Layer 1: Feature Computation
│   ├── __init__.py
│   ├── compute.py               # Main computation entry point
│   ├── config.py                # FeatureConfig dataclass
│   └── normalizers.py           # Normalization functions
│
├── storage/                     # Layer 2: Feature Storage
│   ├── __init__.py
│   ├── store.py                 # FeatureStore class
│   ├── views.py                 # View generators (wide, nested, etc.)
│   ├── schemas.py               # Pandera validation schemas
│   └── migrations.py            # Schema migration utilities
│
├── analysis/                    # Layer 3: Statistical Analysis
│   ├── __init__.py
│   ├── runner.py                # Analysis orchestrator
│   ├── distribution.py          # Distribution tests
│   ├── dimensionality.py        # PCA, VIF
│   ├── regime.py                # Regime dependence
│   ├── temporal.py              # Stationarity, ACF
│   ├── cross_scale.py           # Cross-lookback correlation
│   ├── stability.py             # Cross-threshold stability
│   └── selection.py             # Feature selection
│
├── telemetry/                   # Layer 4: Observability
│   ├── __init__.py
│   ├── events.py                # Event type definitions
│   ├── provenance.py            # Data lineage tracking
│   └── emitter.py               # NDJSON emission
│
└── statistical_examination/     # DEPRECATED (migrate to analysis/)
    └── ...
```

---

## Long Format SSoT Schema

### Parquet Schema

```python
LONG_SCHEMA = {
    # Primary Keys
    "bar_index": pl.UInt32,
    "symbol": pl.Categorical,
    "threshold_dbps": pl.UInt16,
    "lookback": pl.UInt16,
    "feature": pl.Categorical,  # bull_ed, bear_ed, etc.

    # Value
    "value": pl.Float64,

    # Metadata
    "valid": pl.Boolean,         # True if not in warmup period
    "computed_at": pl.Datetime,
    "nav_hash": pl.Utf8,         # Provenance: hash of source NAV
}
```

### Row Count Calculation

```
rows = n_bars × n_symbols × n_thresholds × n_lookbacks × n_features
     = 2000 × 2 × 4 × 5 × 8
     = 640,000 rows

vs. current sparse wide: 16,000 rows × 162 columns = 2.6M cells (75% null)
```

### Compression Efficiency

Long format with Parquet:
- Categorical columns (symbol, feature): Excellent dictionary compression
- threshold_dbps, lookback: Few unique values → RLE compression
- value: Float64 with good distribution
- Estimated size: ~15-25 MB for 640K rows (vs. ~20 MB sparse wide)

---

## View Generation Examples

### Wide Format (for ML)

```python
def to_wide(
    self,
    threshold: int | None = None,
    symbol: str | None = None,
    drop_warmup: bool = True,
) -> pl.DataFrame:
    """Generate wide format suitable for ML training."""
    df = self.df

    if threshold:
        df = df.filter(pl.col("threshold_dbps") == threshold)
    if symbol:
        df = df.filter(pl.col("symbol") == symbol)

    # Pivot: rows=bar_index, cols=feature combinations
    wide = df.pivot(
        on=["threshold_dbps", "lookback", "feature"],
        index=["bar_index", "symbol"],
        values="value",
    )

    # Rename to convention: ith_rb{t}_lb{lb}_{f}
    wide = wide.rename({
        col: f"ith_rb{col[0]}_lb{col[1]}_{col[2]}"
        for col in wide.columns if isinstance(col, tuple)
    })

    if drop_warmup:
        # Drop rows with any null (warmup period)
        wide = wide.drop_nulls()

    return wide
```

### Nested Format (for Semantic Queries)

```python
def to_nested(self) -> list[dict]:
    """Generate nested JSON for semantic queries and API responses."""
    results = []

    for (bar_idx, symbol), group in self.df.group_by(["bar_index", "symbol"]):
        record = {
            "bar_index": bar_idx,
            "symbol": symbol,
            "features": {}
        }

        for threshold in group["threshold_dbps"].unique():
            t_key = f"rb{threshold}"
            record["features"][t_key] = {}

            t_group = group.filter(pl.col("threshold_dbps") == threshold)
            for lookback in t_group["lookback"].unique():
                lb_key = f"lb{lookback}"
                lb_group = t_group.filter(pl.col("lookback") == lookback)

                record["features"][t_key][lb_key] = {
                    row["feature"]: row["value"]
                    for row in lb_group.iter_rows(named=True)
                    if row["valid"]
                }

        results.append(record)

    return results
```

---

## Migration Path

### Phase 1: Create New Module Structure

```bash
# Create new directories
mkdir -p packages/ith-python/src/ith_python/{features,storage,analysis}

# Move/refactor existing code
# statistical_examination/*.py → analysis/*.py
# telemetry/*.py → telemetry/ (already exists)
```

### Phase 2: Implement FeatureStore

```python
# packages/ith-python/src/ith_python/storage/store.py
class FeatureStore:
    """Central feature storage with view generators."""

    @classmethod
    def from_parquet(cls, path: Path) -> "FeatureStore": ...

    @classmethod
    def from_computation(
        cls,
        nav_data: dict[str, np.ndarray],  # symbol -> NAV
        config: FeatureConfig,
    ) -> "FeatureStore": ...

    def to_wide(self, **kwargs) -> pl.DataFrame: ...
    def to_nested(self) -> list[dict]: ...
    def to_clickhouse(self) -> pa.RecordBatch: ...

    def save(self, path: Path) -> None: ...
```

### Phase 3: Update Mise Tasks

Add new tasks alongside existing ones (parallel path), then deprecate old tasks.

### Phase 4: Update Documentation

- Update CLAUDE.md files with new architecture
- Update docs/forensic/E2E.md with Long format documentation
- Add migration guide for downstream consumers

---

## Critical Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `packages/ith-python/src/ith_python/features/__init__.py` | CREATE | Feature computation module |
| `packages/ith-python/src/ith_python/features/compute.py` | CREATE | Main computation entry |
| `packages/ith-python/src/ith_python/storage/__init__.py` | CREATE | Storage module |
| `packages/ith-python/src/ith_python/storage/store.py` | CREATE | FeatureStore class |
| `packages/ith-python/src/ith_python/storage/views.py` | CREATE | View generators |
| `packages/ith-python/src/ith_python/analysis/__init__.py` | CREATE | Analysis module |
| `packages/ith-python/src/ith_python/analysis/runner.py` | CREATE | Analysis orchestrator |
| `mise.toml` | MODIFY | Add new task structure |
| `docs/forensic/E2E.md` | MODIFY | Document new architecture |
| `CLAUDE.md` | MODIFY | Update with new module structure |

---

## Verification Plan

### Unit Tests

```bash
# Test feature computation
UV_PYTHON=python3.13 uv run pytest tests/test_features/ -v

# Test storage/views
UV_PYTHON=python3.13 uv run pytest tests/test_storage/ -v

# Test analysis modules
UV_PYTHON=python3.13 uv run pytest tests/test_analysis/ -v
```

### Integration Tests

```bash
# Full pipeline
mise run forensic:e2e

# Verify SSoT exists and is valid
UV_PYTHON=python3.13 uv run python -c "
import polars as pl
df = pl.read_parquet('artifacts/ssot/features_long.parquet')
print(f'Rows: {len(df):,}')
print(f'Columns: {df.columns}')
print(f'Null rate: {df.null_count().sum() / (len(df) * len(df.columns)):.2%}')
"

# Verify views
ls -la artifacts/views/wide/*.parquet
ls -la artifacts/views/nested/*.jsonl

# Verify telemetry
jq -s 'length' logs/ndjson/*.jsonl
```

### Upgrade Tests

```bash
# Test independent upgrade paths
mise run upgrade:features && mise run upgrade:analysis
mise run upgrade:analysis  # Should work without recomputing features
mise run upgrade:views     # Should work without recomputing anything
```

---

## Success Criteria

1. **SSoT Established**: Single `features_long.parquet` is source of truth
2. **Zero Sparsity**: Long format has no structural nulls (only warmup nulls)
3. **Multiple Views**: Wide, Nested, ClickHouse views generated from SSoT
4. **Independent Upgrades**: Can upgrade compute/storage/analysis separately
5. **Mise Orchestration**: All operations via `mise run` tasks
6. **Claude Code Ready**: NDJSON telemetry enables AI forensic analysis
7. **No Regressions**: Existing tests pass, same feature values produced

---

## Out of Scope (Deferred)

- ClickHouse materialized views (use Parquet first)
- Real-time streaming features (batch-only for now)
- Multi-language analysis modules (Python-only for now)
- Feature versioning/lineage tracking (basic provenance only)
