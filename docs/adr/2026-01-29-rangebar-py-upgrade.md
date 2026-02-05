# rangebar-py Upgrade: v11.0.0 → v11.3.0 <!-- SSoT-OK: external version refs -->

## Status

Accepted (updated 2026-01-29)

## Context

trading-fitness pins rangebar-py via git tag in `packages/ith-python/pyproject.toml`.
The upstream rangebar-py has released multiple versions since our original pin,
introducing breaking changes to bar construction semantics, new columns, and
critical OOM/TypeError fixes.

## API Diff <!-- SSoT-OK: external rangebar-py version references for documentation -->

### New Parameters in `get_range_bars()`

| Parameter                      | Type   | Default | Version | Impact                |
| ------------------------------ | ------ | ------- | ------- | --------------------- |
| `include_exchange_sessions`    | `bool` | `False` | v11.2.0 | 4 new session columns |
| `prevent_same_timestamp_close` | `bool` | `True`  | v11.1.0 | Bar open deferral fix |
| `verify_checksum`              | `bool` | `True`  | v11.1.0 | Staleness detection   |
| `max_memory_gb`                | `int`  | None    | v11.3.0 | Memory budget limit   |

### New Exchange Session Columns (when `include_exchange_sessions=True`)

- `exchange_session_sydney`
- `exchange_session_tokyo`
- `exchange_session_london`
- `exchange_session_newyork`

### Bar Open Deferral Fix (Issue #46)

Previous behavior: when a trade breached the threshold, the new bar opened at the
same timestamp as the breaching trade. This caused ambiguity in bar boundaries.

New behavior (`prevent_same_timestamp_close=True`, default): the new bar open is
deferred to the next distinct timestamp after the breach.

**Impact**: ALL previously cached bars are potentially stale. Bar counts and
timestamps will differ from pre-fix data.

### Staleness Detection (Issue #39)

Schema version constants enable automatic detection of stale cached data:

- `SCHEMA_VERSION_OHLCV_ONLY`: Pre-microstructure data
- `SCHEMA_VERSION_MICROSTRUCTURE`: Added 15 microstructure columns
- `SCHEMA_VERSION_OUROBOROS`: Added ouroboros mode

### New Exports

- `MICROSTRUCTURE_COLUMNS`, `EXCHANGE_SESSION_COLUMNS`, `ALL_OPTIONAL_COLUMNS`
- `normalize_arrow_dtypes()`, `normalize_temporal_precision()`
- `process_trades_polars()` (Issue #45)

### OOM-Safe Cache Regeneration (Issue #47)

New script for memory-safe cache regeneration on large datasets.

### v11.3.0 Fixes (Issues #49, #50, #51)

#### Per-Segment Tick Loading (Issue #51)

Previous behavior: `get_range_bars(ouroboros="year")` loaded ALL tick data for
the full year into memory at once (~70GB for BTCUSDT 2024).

New behavior: Ticks are loaded per-ouroboros-segment (monthly), with bar state
carried forward across segment boundaries. Peak memory drops from ~70GB to ~3GB.

#### Bool→Int Cast in CH Cache Write (Issue #50)

Previous behavior: Exchange session columns (`numpy.bool_` dtype) caused
`TypeError: object of type 'numpy.bool' has no len()` when `clickhouse_connect`
tried to serialize them to `Nullable(String)` columns.

New behavior: rangebar-py casts bool columns to `int` before ClickHouse insert.

#### Pre-Flight Memory Estimation and MEM Guards (Issue #49)

- 10 MEM guards (MEM-001 through MEM-010) for memory-sensitive operations
- Pre-flight estimation returns "safe" / "streaming_recommended" / "will_oom"
- `max_memory_gb` parameter for memory budgeting
- Memory snapshots in diagnostic hook payloads

## Decision

Upgrade to v11.3.0 tag. Remove all local workarounds for OOM and TypeError
issues that are now fixed upstream. Simplify the precompute script.

## Consequences

- All `range_bars_ouroboros_year` data must be recomputed
- Legacy `range_bars` table (268M bars) staleness must be assessed
- ClickHouse exchange session columns use `Nullable(UInt8)` (not `Nullable(String)`)
- `scripts/precompute_ouroboros_resumable.py` deleted (no longer needed)
- `scripts/precompute_ouroboros_year.py` simplified: removed bool→int cast,
  sequential subprocess mode, TypeError handling
- Test fixtures may need exchange session column additions

## Workarounds Removed

| Workaround                                 | Reason                                   | Upstream Fix               |
| ------------------------------------------ | ---------------------------------------- | -------------------------- |
| Bool→int cast in precompute script         | Exchange session `numpy.bool_` TypeError | Issue #50 (commit 24c5676) |
| `precompute_ouroboros_resumable.py` script | Full-year tick loading caused OOM        | Issue #51 (commit 6134fd8) |
| `--sequential` subprocess isolation mode   | Memory isolation per job                 | Per-segment loading in #51 |
| `TypeError` in except clause               | Silent worker crashes on CH cache write  | Issue #50                  |
