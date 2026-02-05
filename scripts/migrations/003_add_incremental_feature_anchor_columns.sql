-- Migration: Add anchor columns for incremental feature addition
-- Context: Enable adding new features without full table recomputation
-- Run on: bigblack (ClickHouse)
-- Usage: clickhouse-client --multiquery < 003_add_incremental_feature_anchor_columns.sql
--
-- Column naming convention:
--   ouroboros_segment_*     - Segment identity (Ouroboros boundaries)
--   source_tick_*           - Tick data provenance (traceability)
--   feature_computation_*   - Feature version tracking
--   bar_position_*          - Bar location within segment

-- Table: range_bars_ouroboros_year (primary target)
ALTER TABLE rangebar_cache.range_bars_ouroboros_year
    -- Segment identity
    ADD COLUMN IF NOT EXISTS ouroboros_segment_id String DEFAULT '' AFTER is_orphan,
    ADD COLUMN IF NOT EXISTS ouroboros_segment_start_timestamp_ms Int64 DEFAULT 0 AFTER ouroboros_segment_id,
    ADD COLUMN IF NOT EXISTS ouroboros_segment_end_timestamp_ms Int64 DEFAULT 0 AFTER ouroboros_segment_start_timestamp_ms,
    -- Tick data provenance
    ADD COLUMN IF NOT EXISTS source_tick_xxhash64_checksum UInt64 DEFAULT 0 AFTER ouroboros_segment_end_timestamp_ms,
    ADD COLUMN IF NOT EXISTS source_tick_row_count UInt32 DEFAULT 0 AFTER source_tick_xxhash64_checksum,
    ADD COLUMN IF NOT EXISTS source_tick_first_timestamp_ms Int64 DEFAULT 0 AFTER source_tick_row_count,
    ADD COLUMN IF NOT EXISTS source_tick_last_timestamp_ms Int64 DEFAULT 0 AFTER source_tick_first_timestamp_ms,
    -- Feature version tracking
    ADD COLUMN IF NOT EXISTS feature_computation_versions_json String DEFAULT '{}' AFTER source_tick_last_timestamp_ms,
    -- Bar position in segment
    ADD COLUMN IF NOT EXISTS bar_position_index_in_segment UInt32 DEFAULT 0 AFTER feature_computation_versions_json,
    ADD COLUMN IF NOT EXISTS bar_position_is_segment_first UInt8 DEFAULT 0 AFTER bar_position_index_in_segment,
    ADD COLUMN IF NOT EXISTS bar_position_is_segment_last UInt8 DEFAULT 0 AFTER bar_position_is_segment_first;

-- Verify: Segment identity columns
SELECT 'ouroboros_segment_* columns:' AS info;
SELECT name, type, default_expression FROM system.columns
WHERE database = 'rangebar_cache'
  AND table = 'range_bars_ouroboros_year'
  AND name LIKE 'ouroboros_segment_%'
ORDER BY position;

-- Verify: Source tick provenance columns
SELECT 'source_tick_* columns:' AS info;
SELECT name, type, default_expression FROM system.columns
WHERE database = 'rangebar_cache'
  AND table = 'range_bars_ouroboros_year'
  AND name LIKE 'source_tick_%'
ORDER BY position;

-- Verify: Feature computation columns
SELECT 'feature_computation_* columns:' AS info;
SELECT name, type, default_expression FROM system.columns
WHERE database = 'rangebar_cache'
  AND table = 'range_bars_ouroboros_year'
  AND name LIKE 'feature_computation_%'
ORDER BY position;

-- Verify: Bar position columns
SELECT 'bar_position_* columns:' AS info;
SELECT name, type, default_expression FROM system.columns
WHERE database = 'rangebar_cache'
  AND table = 'range_bars_ouroboros_year'
  AND name LIKE 'bar_position_%'
ORDER BY position;
