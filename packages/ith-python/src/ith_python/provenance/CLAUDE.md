# provenance/

> Anchor columns and validation for incremental feature addition to ClickHouse range bars.

**← [Back to ith-python](../../../../CLAUDE.md)**

## Purpose

Enable adding new features to existing range bars WITHOUT full table recomputation by:

1. Fingerprinting source tick data (`source_tick_*` columns)
2. Tracking feature computation versions (`feature_computation_versions_json`)
3. Strict validation before incremental updates

---

## Module Map

| Module                            | ClickHouse Columns                  | Purpose                        |
| --------------------------------- | ----------------------------------- | ------------------------------ |
| `source_tick_fingerprint.py`      | `source_tick_*`                     | xxHash64 fingerprint of ticks  |
| `feature_computation_versions.py` | `feature_computation_versions_json` | JSON version tracking          |
| `alignment_validator.py`          | All `source_tick_*`                 | Validate bar→tick traceability |

---

## ClickHouse Column Groups

### Group 1: Segment Identity (`ouroboros_segment_*`)

| Column                                 | Type   | Purpose                      |
| -------------------------------------- | ------ | ---------------------------- |
| `ouroboros_segment_id`                 | String | Segment ID (e.g., "2024_01") |
| `ouroboros_segment_start_timestamp_ms` | Int64  | Segment start (ms)           |
| `ouroboros_segment_end_timestamp_ms`   | Int64  | Segment end (ms)             |

### Group 2: Tick Data Provenance (`source_tick_*`)

| Column                           | Type   | Purpose                    |
| -------------------------------- | ------ | -------------------------- |
| `source_tick_xxhash64_checksum`  | UInt64 | xxHash64 of tick DataFrame |
| `source_tick_row_count`          | UInt32 | Number of ticks            |
| `source_tick_first_timestamp_ms` | Int64  | First tick timestamp (ms)  |
| `source_tick_last_timestamp_ms`  | Int64  | Last tick timestamp (ms)   |

### Group 3: Feature Version Tracking (`feature_computation_*`)

| Column                              | Type   | Purpose                     |
| ----------------------------------- | ------ | --------------------------- |
| `feature_computation_versions_json` | String | JSON map of feature→version |

### Group 4: Bar Position (`bar_position_*`)

| Column                          | Type   | Purpose                   |
| ------------------------------- | ------ | ------------------------- |
| `bar_position_index_in_segment` | UInt32 | 0-based index in segment  |
| `bar_position_is_segment_first` | UInt8  | 1 if first bar in segment |
| `bar_position_is_segment_last`  | UInt8  | 1 if last bar in segment  |

---

## Column Discovery (for AI Agents)

```sql
-- Find all provenance columns by prefix
SELECT name, type FROM system.columns
WHERE table = 'range_bars_ouroboros_year'
  AND (name LIKE 'source_tick_%'
       OR name LIKE 'ouroboros_segment_%'
       OR name LIKE 'feature_computation_%'
       OR name LIKE 'bar_position_%');
```

---

## Quick Start

```python
from ith_python.provenance import (
    compute_source_tick_fingerprint,
    FeatureComputationVersions,
    validate_source_tick_alignment,
    SourceTickAlignmentError,
)

# Fingerprint tick data
fingerprint = compute_source_tick_fingerprint(tick_df)
# Returns: {"xxhash64_checksum": ..., "row_count": ..., ...}

# Track feature versions
versions = FeatureComputationVersions({"range_bar": "11.6.1", "ith": None})
if versions.needs_backfill("ith", "1.0"):
    # Compute ITH features...
    versions.mark_computed("ith", "1.0")

# Validate before incremental update
try:
    validate_source_tick_alignment(bar_anchors, tick_df, strict=True)
except SourceTickAlignmentError as e:
    print(f"Cannot add features: {e}")
```

---

## Validation Workflow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Load Existing   │────▶│ Validate Anchor  │────▶│ Add New         │
│ Bar from CH     │     │ vs Tick Data     │     │ Features        │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   source_tick_*          SourceTickAlignment      feature_computation_
   columns                Error if mismatch        versions_json updated
```

---

## Related Documentation

| Document                                                                                             | Purpose                    |
| ---------------------------------------------------------------------------------------------------- | -------------------------- |
| [Migration SQL](../../../../../../scripts/migrations/003_add_incremental_feature_anchor_columns.sql) | ClickHouse schema          |
| [DATA.md](../../../../../../docs/infrastructure/DATA.md)                                             | Bigblack data architecture |
| [telemetry/](../telemetry/CLAUDE.md)                                                                 | Related: ProvenanceContext |
| [ith-python CLAUDE.md](../../../../CLAUDE.md)                                                        | Parent package             |
