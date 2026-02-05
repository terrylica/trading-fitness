-- Migration: Add inter-bar feature columns to range bar tables
-- Context: rangebar-py v11.6.0 added 16 inter-bar microstructure features (Issue #59)
-- Run on: bigblack (ClickHouse)
-- Usage: clickhouse-client --multiquery < 002_add_inter_bar_columns.sql
--
-- Inter-bar features are computed from a lookback window of trades BEFORE each bar opens.
-- Tier 1 Core (7): trade_count, ofi, duration_us, intensity, vwap_raw, vwap_position, count_imbalance
-- Tier 2 Statistical (5): kyle_lambda, burstiness, volume_skew, volume_kurt, price_range
-- Tier 3 Advanced (4): kaufman_er, garman_klass_vol, hurst, permutation_entropy

-- Table 1: range_bars (legacy)
ALTER TABLE rangebar_cache.range_bars
    ADD COLUMN IF NOT EXISTS lookback_trade_count Float64 DEFAULT 0 AFTER exchange_session_newyork,
    ADD COLUMN IF NOT EXISTS lookback_ofi Float64 DEFAULT 0 AFTER lookback_trade_count,
    ADD COLUMN IF NOT EXISTS lookback_duration_us Float64 DEFAULT 0 AFTER lookback_ofi,
    ADD COLUMN IF NOT EXISTS lookback_intensity Float64 DEFAULT 0 AFTER lookback_duration_us,
    ADD COLUMN IF NOT EXISTS lookback_vwap_raw Float64 DEFAULT 0 AFTER lookback_intensity,
    ADD COLUMN IF NOT EXISTS lookback_vwap_position Float64 DEFAULT 0 AFTER lookback_vwap_raw,
    ADD COLUMN IF NOT EXISTS lookback_count_imbalance Float64 DEFAULT 0 AFTER lookback_vwap_position,
    ADD COLUMN IF NOT EXISTS lookback_kyle_lambda Float64 DEFAULT 0 AFTER lookback_count_imbalance,
    ADD COLUMN IF NOT EXISTS lookback_burstiness Float64 DEFAULT 0 AFTER lookback_kyle_lambda,
    ADD COLUMN IF NOT EXISTS lookback_volume_skew Float64 DEFAULT 0 AFTER lookback_burstiness,
    ADD COLUMN IF NOT EXISTS lookback_volume_kurt Float64 DEFAULT 0 AFTER lookback_volume_skew,
    ADD COLUMN IF NOT EXISTS lookback_price_range Float64 DEFAULT 0 AFTER lookback_volume_kurt,
    ADD COLUMN IF NOT EXISTS lookback_kaufman_er Float64 DEFAULT 0 AFTER lookback_price_range,
    ADD COLUMN IF NOT EXISTS lookback_garman_klass_vol Float64 DEFAULT 0 AFTER lookback_kaufman_er,
    ADD COLUMN IF NOT EXISTS lookback_hurst Float64 DEFAULT 0 AFTER lookback_garman_klass_vol,
    ADD COLUMN IF NOT EXISTS lookback_permutation_entropy Float64 DEFAULT 0 AFTER lookback_hurst;

-- Table 2: range_bars_ouroboros_year (Ouroboros mode, recompute target)
ALTER TABLE rangebar_cache.range_bars_ouroboros_year
    ADD COLUMN IF NOT EXISTS lookback_trade_count Float64 DEFAULT 0 AFTER exchange_session_newyork,
    ADD COLUMN IF NOT EXISTS lookback_ofi Float64 DEFAULT 0 AFTER lookback_trade_count,
    ADD COLUMN IF NOT EXISTS lookback_duration_us Float64 DEFAULT 0 AFTER lookback_ofi,
    ADD COLUMN IF NOT EXISTS lookback_intensity Float64 DEFAULT 0 AFTER lookback_duration_us,
    ADD COLUMN IF NOT EXISTS lookback_vwap_raw Float64 DEFAULT 0 AFTER lookback_intensity,
    ADD COLUMN IF NOT EXISTS lookback_vwap_position Float64 DEFAULT 0 AFTER lookback_vwap_raw,
    ADD COLUMN IF NOT EXISTS lookback_count_imbalance Float64 DEFAULT 0 AFTER lookback_vwap_position,
    ADD COLUMN IF NOT EXISTS lookback_kyle_lambda Float64 DEFAULT 0 AFTER lookback_count_imbalance,
    ADD COLUMN IF NOT EXISTS lookback_burstiness Float64 DEFAULT 0 AFTER lookback_kyle_lambda,
    ADD COLUMN IF NOT EXISTS lookback_volume_skew Float64 DEFAULT 0 AFTER lookback_burstiness,
    ADD COLUMN IF NOT EXISTS lookback_volume_kurt Float64 DEFAULT 0 AFTER lookback_volume_skew,
    ADD COLUMN IF NOT EXISTS lookback_price_range Float64 DEFAULT 0 AFTER lookback_volume_kurt,
    ADD COLUMN IF NOT EXISTS lookback_kaufman_er Float64 DEFAULT 0 AFTER lookback_price_range,
    ADD COLUMN IF NOT EXISTS lookback_garman_klass_vol Float64 DEFAULT 0 AFTER lookback_kaufman_er,
    ADD COLUMN IF NOT EXISTS lookback_hurst Float64 DEFAULT 0 AFTER lookback_garman_klass_vol,
    ADD COLUMN IF NOT EXISTS lookback_permutation_entropy Float64 DEFAULT 0 AFTER lookback_hurst;

-- Verify
SELECT 'range_bars lookback columns:' AS info;
SELECT name, type FROM system.columns
WHERE database = 'rangebar_cache' AND table = 'range_bars' AND name LIKE 'lookback_%'
ORDER BY position;

SELECT 'range_bars_ouroboros_year lookback columns:' AS info;
SELECT name, type FROM system.columns
WHERE database = 'rangebar_cache' AND table = 'range_bars_ouroboros_year' AND name LIKE 'lookback_%'
ORDER BY position;
