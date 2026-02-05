# Session Resume Point

> Quick context for continuing work on trading-fitness.

## Last Session: 2026-01-29

### Completed Work

**rangebar-py Upgrade to Latest** <!-- SSoT-OK: external version references in session log -->

Full upgrade cycle with workaround removal.

| Item | Status |
| ---- | ------ |
| Version pin in pyproject.toml | Updated to latest tag |
| `uv lock` + `uv sync` | Passed |
| Test suite (328 tests) | Passed (5 timeout on integration tests — expected without bigblack) |
| ADR documenting API diff | Updated: `docs/adr/2026-01-29-rangebar-py-upgrade.md` |
| ClickHouse migration SQL | Written: `scripts/migrations/001_add_exchange_session_columns.sql` |
| Precompute script simplified | Removed OOM/TypeError workarounds, uses upstream per-segment loading |
| Resumable workaround script | Deleted: `scripts/precompute_ouroboros_resumable.py` (no longer needed) |
| GitHub issues filed | #48 (JSONL tracing), #49 (resource mgmt), #50 (bool TypeError), #51 (OOM) |
| Statistical examination modules | Verified: 12 modules + 2 analysis modules intact |
| CLAUDE.md updated | Exchange session columns + migration path documented |

**Key upstream fixes:**

- Issue #50: Bool→int cast in CH cache write (was causing silent worker crashes)
- Issue #51: Per-segment tick loading — 70GB → 3GB peak memory for Ouroboros year
- Issue #49: Pre-flight memory estimation, MEM guards, `max_memory_gb` parameter
- Issue #46: Bar open deferral fix — ALL cached bars potentially stale

**Workarounds removed:**

- Bool→int cast block in precompute script
- `--sequential` subprocess isolation mode
- `TypeError` in except clause
- Entire `precompute_ouroboros_resumable.py` script

### Pending (requires bigblack access)

- Kill old resumable recompute process on bigblack
- Sync updated code to bigblack, `uv lock && uv sync`
- Truncate `range_bars_ouroboros_year` (stale partial data from workaround runs)
- Restart recompute: `precompute_ouroboros_year.py --workers 2`
- Assess legacy `range_bars` table (268M bars) staleness
- Run statistical examination on recomputed data
- Assess exchange session columns as ITH features

---

## Previous Session: 2026-01-28

### Completed Work

**CLAUDE.md Hub-and-Spoke Reorganization**

Reorganized project memory following Link Farm + Hub-and-Spoke with Progressive Disclosure pattern:

| File                              | Changes                                                            |
| --------------------------------- | ------------------------------------------------------------------ |
| Root `CLAUDE.md`                  | Reorganized as hub with navigation, package map, Ouroboros support |
| `docs/infrastructure/DATA.md`     | DRY - links to root for timestamp precision gotcha                 |
| `packages/ith-python/CLAUDE.md`   | Made concise with 3-layer architecture diagram                     |
| `packages/metrics-rust/CLAUDE.md` | Shortened from ~475 to ~124 lines                                  |
| `services/*/CLAUDE.md`            | Simplified placeholders                                            |

**Ouroboros Support Documented**

- Added Ouroboros mode section to Data Infrastructure
- ClickHouse table: `range_bars_ouroboros_year`
- Precompute script: `scripts/precompute_ouroboros_year.py`
- Currently running on bigblack with ~24M bars processed

**Timestamp Precision by Year Gotcha**

Documented critical data handling issue:

- 2022-2023: `datetime64[ms]` - use as-is
- 2024+: `datetime64[ns]` - divide by 1,000,000
- Symptom: timestamps show year 52000+ or 1970 if wrong

### Ouroboros Precompute Status (bigblack)

| Symbol  | 2022 | 2023 | 2024 | 2025 | 2026 |
| ------- | ---- | ---- | ---- | ---- | ---- |
| BNBUSDT | 635K | 265K | 395K | 380K | 8K   |
| BTCUSDT | 46K  | 10K  | 28K  | 7K   | 6K   |
| ETHUSDT | 7.7M | 2.1M | 3.4M | -    | -    |
| SOLUSDT | 1.8M | 1M   | 933K | 775K | 20K  |

### To Continue

```bash
# Check Ouroboros precompute progress
ssh bigblack "clickhouse-client --query 'SELECT symbol, toYear(toDateTime(timestamp_ms/1000)) as year, count() FROM rangebar_cache.range_bars_ouroboros_year GROUP BY 1,2 ORDER BY 1,2'"

# Run statistical examination (after precompute completes)
mise run forensic:full-pipeline
```

---

## Previous Session: 2026-01-26

### Completed Work

**Multi-View Feature Architecture Implementation**

Implemented 3-layer separation of concerns for feature pipeline:

| Layer | Module      | Purpose                              | Status |
| ----- | ----------- | ------------------------------------ | ------ |
| 1     | `features/` | ITH feature computation (wraps Rust) | DONE   |
| 2     | `storage/`  | Long Format SSoT + view generators   | DONE   |
| 3     | `analysis/` | Statistical evaluation orchestrator  | DONE   |

**Key Components Created:**

| Component                            | Description                                                                                |
| ------------------------------------ | ------------------------------------------------------------------------------------------ |
| `FeatureStore`                       | Central storage with view generators (`to_wide`, `to_nested`, `to_dense`, `to_clickhouse`) |
| `FeatureConfig`                      | Configuration dataclass for ITH computation                                                |
| `AnalysisConfig` / `AnalysisResults` | Unified analysis interface                                                                 |
| `validate_warmup()`                  | Preflight check for data sufficiency                                                       |
| `preflight:warmup`                   | Mise task for warmup validation                                                            |

**Warmup Handling Fixed:**

- `to_wide(drop_warmup=True)` now correctly drops NaN values (not just null)
- Preflight validation ensures sufficient data after warmup
- 499 warmup bars for lb500, leaving 1501 valid rows from 2000 input

### Mise Tasks Added

```bash
mise run preflight:warmup     # Validate warmup requirements
mise run features:compute     # Layer 1 (depends on preflight)
mise run views:all            # Layer 2 view generation
mise run analysis:all         # Layer 3 statistical analysis
mise run forensic:pipeline    # Full E2E pipeline
mise run upgrade:features     # Independent feature upgrade
mise run upgrade:views        # Independent view upgrade
mise run upgrade:analysis     # Independent analysis upgrade
```

### Key Files Created/Modified

```
packages/ith-python/src/ith_python/
├── features/                # NEW - Layer 1
│   ├── __init__.py
│   ├── config.py            # FeatureConfig dataclass
│   └── compute.py           # Wraps Rust compute_multiscale_ith
│
├── storage/                 # NEW - Layer 2
│   ├── __init__.py
│   ├── schemas.py           # Long format schema, validation
│   ├── store.py             # FeatureStore class
│   └── views.py             # View generators + warmup validation
│
├── analysis/                # NEW - Layer 3
│   ├── __init__.py          # Re-exports from statistical_examination
│   └── runner.py            # Unified analysis interface

docs/plans/
└── 2026-01-25-multi-view-feature-architecture-plan.md  # Architecture plan

mise.toml                    # Added preflight + layer tasks
```

### E2E Test Results (2000 bars BTCUSDT)

| Metric               | Value                |
| -------------------- | -------------------- |
| Input bars           | 2000                 |
| Warmup bars          | 499                  |
| Valid bars           | 1501                 |
| Features             | 40 (8 × 5 lookbacks) |
| Normality rate       | 0%                   |
| PCA components (95%) | 26 of 40             |
| Stationarity rate    | 79%                  |

### Plan Reference

- **Architecture Plan**: [docs/plans/2026-01-25-multi-view-feature-architecture-plan.md](docs/plans/2026-01-25-multi-view-feature-architecture-plan.md)
- **Observability Plan**: [docs/plans/2026-01-25-observability-telemetry-plan.md](docs/plans/2026-01-25-observability-telemetry-plan.md)

### To Continue

```bash
# Run full pipeline
mise run forensic:pipeline

# Or run individual layers
mise run preflight:warmup
mise run features:compute
mise run analysis:all
```

---

## Previous Sessions

### 2026-01-25: Feature Registry & ClaSPy Documentation

- Created `docs/features/REGISTRY.md` - SSoT for all extractable features
- Created `docs/features/CLASPY.md` - ClaSPy integration guide
- Forked ClaSPy to `~/fork-tools/claspy/`

### 2026-01-25: Observability Telemetry Phases 1-2

- Created telemetry module with provenance tracking
- Extended ndjson_logger with scientific reproducibility fields
- Added epoch_detected telemetry to Bull/Bear ITH

### 2026-01-23: Statistical Methods Rectification

- Fixed Friedman test (removed - independence violation)
- Fixed Beta fit (AD test instead of KS)
- Fixed Cohen's d (weighted pooled SD)
- Added Cliff's Delta effect size
- Added Participation Ratio for PCA
- Added Ridge VIF for stability

### 2026-01-22: Statistical Examination Framework

- Created cross_scale, threshold_stability, distribution, regime modules
- Created dimensionality, selection, temporal modules
- Added runner.py CLI orchestration
