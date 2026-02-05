-- Migration: Add exchange session columns to range bar tables
-- Context: rangebar-py v11.2.0 added 4 exchange session columns (bool dtype)
-- Run on: bigblack (ClickHouse)
-- Usage: clickhouse-client --multiquery < 001_add_exchange_session_columns.sql

-- Table 1: range_bars (legacy)
ALTER TABLE rangebar_cache.range_bars
    ADD COLUMN IF NOT EXISTS exchange_session_sydney Nullable(UInt8) AFTER aggregation_density,
    ADD COLUMN IF NOT EXISTS exchange_session_tokyo Nullable(UInt8) AFTER exchange_session_sydney,
    ADD COLUMN IF NOT EXISTS exchange_session_london Nullable(UInt8) AFTER exchange_session_tokyo,
    ADD COLUMN IF NOT EXISTS exchange_session_newyork Nullable(UInt8) AFTER exchange_session_london;

-- Table 2: range_bars_ouroboros_year (Ouroboros mode, recompute target)
ALTER TABLE rangebar_cache.range_bars_ouroboros_year
    ADD COLUMN IF NOT EXISTS exchange_session_sydney Nullable(UInt8) AFTER aggregation_density,
    ADD COLUMN IF NOT EXISTS exchange_session_tokyo Nullable(UInt8) AFTER exchange_session_sydney,
    ADD COLUMN IF NOT EXISTS exchange_session_london Nullable(UInt8) AFTER exchange_session_tokyo,
    ADD COLUMN IF NOT EXISTS exchange_session_newyork Nullable(UInt8) AFTER exchange_session_london;

-- Verify
SELECT 'range_bars columns:' AS info;
SELECT name, type FROM system.columns
WHERE database = 'rangebar_cache' AND table = 'range_bars' AND name LIKE 'exchange_%'
ORDER BY position;

SELECT 'range_bars_ouroboros_year columns:' AS info;
SELECT name, type FROM system.columns
WHERE database = 'rangebar_cache' AND table = 'range_bars_ouroboros_year' AND name LIKE 'exchange_%'
ORDER BY position;
