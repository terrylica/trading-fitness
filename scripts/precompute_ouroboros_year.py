#!/usr/bin/env python3
"""
Precompute Ouroboros year-based range bars into ClickHouse.

Builds range bars with yearly reset boundaries (Jan 1 00:00 UTC).
Bars crossing year boundaries are marked as orphans.

IMPORTANT: Run with RANGEBAR_NO_MEMORY_GUARD=1 to prevent RLIMIT_AS errors:
    RANGEBAR_NO_MEMORY_GUARD=1 uv run python ../../scripts/precompute_ouroboros_year.py

Usage:
    # On bigblack:
    cd ~/eon/trading-fitness/packages/ith-python
    RANGEBAR_NO_MEMORY_GUARD=1 uv run python ../../scripts/precompute_ouroboros_year.py [--workers N] [--dry-run]

Target table: rangebar_cache.range_bars_ouroboros_year

Anchor columns (incremental feature support):
    - ouroboros_segment_* : Segment identity (e.g., "2024_01" for year 2024)
    - source_tick_* : Tick data fingerprint for validation
    - feature_computation_versions_json : Track which features are computed
    - bar_position_* : Bar position within segment

ADR: docs/adr/2026-01-29-rangebar-py-upgrade.md
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

# =============================================================================
# Memory Configuration (before importing rangebar)
# =============================================================================
# Disable auto memory guard - we'll handle per-worker limits manually.
# Root cause: rangebar's auto_memory_guard() sets RLIMIT_AS at import time.
# With multiprocessing fork, child processes inherit this limit and can't
# allocate more than the total limit / n_workers effectively.
# By disabling the auto guard, Polars/Rust allocator works normally.
os.environ["RANGEBAR_NO_MEMORY_GUARD"] = "1"

# =============================================================================
# Configuration
# =============================================================================

# Symbols and thresholds to precompute
# NOTE: Only thresholds >= 1000 dbps are economically viable (overcome trading costs)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
THRESHOLDS = [1000, 2500, 5000, 10000]  # decimal basis points (1000 dbps = 10%)

# Date range: 4+ years (2022-01-01 to present)
# Binance data availability starts ~2019 for major pairs
START_DATE = "2022-01-01"
END_DATE = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")

# Ouroboros mode
OUROBOROS_MODE = "year"

# Inter-bar feature lookback (v11.6.0) - number of trades before each bar
INTER_BAR_LOOKBACK = 100

# ClickHouse target table
TARGET_TABLE = "rangebar_cache.range_bars_ouroboros_year"


@dataclass
class JobResult:
    """Result from a precompute job."""

    symbol: str
    threshold: int
    year: int
    bars: int
    orphans: int
    elapsed: float
    status: str
    error: str | None = None


def get_year_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Split date range into per-year chunks for Ouroboros processing."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    ranges = []
    current_year = start.year

    while current_year <= end.year:
        year_start = max(start, datetime(current_year, 1, 1))
        year_end = min(end, datetime(current_year, 12, 31))

        ranges.append((
            year_start.strftime("%Y-%m-%d"),
            year_end.strftime("%Y-%m-%d"),
        ))
        current_year += 1

    return ranges


def run_precompute_job(args: tuple[str, int, str, str, bool]) -> JobResult:
    """Run a single precompute job for one symbol/threshold/year."""
    symbol, threshold, start_date, end_date, dry_run = args
    worker_id = os.getpid()
    year = int(start_date[:4])

    import time
    start_time = time.time()

    print(f"[PID {worker_id}] Starting {symbol} @ {threshold}dbps for {year}", flush=True)

    if dry_run:
        print(f"[PID {worker_id}] DRY RUN: Would process {symbol} @ {threshold}dbps ({start_date} to {end_date})", flush=True)
        return JobResult(
            symbol=symbol,
            threshold=threshold,
            year=year,
            bars=0,
            orphans=0,
            elapsed=0.0,
            status="DRY_RUN",
        )

    try:
        import hashlib

        import clickhouse_connect
        from rangebar import get_range_bars, populate_cache_resumable

        from ith_python.provenance import (
            FEATURE_VERSIONS,
            FeatureComputationVersions,
        )

        # NOTE: Memory guard disabled via RANGEBAR_NO_MEMORY_GUARD=1 at top of script.
        # RLIMIT_AS is incompatible with Polars/Rust mmap-based allocations.
        # Let the system OOM killer handle extreme cases instead.

        # v12.2.0+: Long-range date protection requires cache population first
        # For date ranges > 30 days, we must populate the cache before get_range_bars()
        print(f"[PID {worker_id}] Populating cache for {symbol} @ {threshold}dbps ({start_date} to {end_date})", flush=True)
        populate_cache_resumable(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            threshold_decimal_bps=threshold,
        )

        # Fetch range bars with Ouroboros year mode
        # This forces fresh construction with year boundaries
        # Note: get_range_bars returns a pandas DataFrame
        df = get_range_bars(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            threshold_decimal_bps=threshold,
            ouroboros=OUROBOROS_MODE,
            include_orphaned_bars=True,  # Include orphans so we can mark them
            use_cache=True,  # Use tick + range bar cache
            fetch_if_missing=False,  # v12.2.0+: Cache already populated above
            include_microstructure=True,
            include_exchange_sessions=True,  # v11.2.0: 4 session columns
            inter_bar_lookback_count=INTER_BAR_LOOKBACK,  # v11.6.0: 16 inter-bar features
        )

        if df is None or len(df) == 0:
            print(f"[PID {worker_id}] No data for {symbol} @ {threshold}dbps in {year}", flush=True)
            return JobResult(
                symbol=symbol,
                threshold=threshold,
                year=year,
                bars=0,
                orphans=0,
                elapsed=time.time() - start_time,
                status="NO_DATA",
            )

        # Count orphans if column exists
        orphan_count = 0
        if "is_orphan" in df.columns:
            orphan_count = int(df["is_orphan"].sum())

        # Prepare data for ClickHouse insertion (pandas DataFrame)
        # Reset index to get timestamp as column (rangebar uses DatetimeIndex)
        df = df.reset_index()
        df = df.rename(columns={"index": "timestamp"})

        # Convert timestamp to milliseconds
        # Note: rangebar returns datetime64[ms, UTC] for older data (2022-2023)
        # but datetime64[ns, UTC] for newer data (2024+)
        ts_dtype = str(df["timestamp"].dtype)
        if "ns" in ts_dtype:
            # Nanoseconds → divide by 1,000,000 to get milliseconds
            df["timestamp_ms"] = df["timestamp"].astype("int64") // 1_000_000
        else:
            # Milliseconds or microseconds → already close to ms
            df["timestamp_ms"] = df["timestamp"].astype("int64")

        # Normalize column names (rangebar uses Title case, ClickHouse uses lowercase)
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        # Add required columns from job parameters
        df["symbol"] = symbol
        df["threshold_decimal_bps"] = threshold

        # Add ouroboros metadata columns
        df["ouroboros_mode"] = OUROBOROS_MODE
        df["is_orphan"] = 0 if "is_orphan" not in df.columns else df["is_orphan"]

        # Note: ouroboros_boundary uses ClickHouse DEFAULT

        # Add cache metadata
        cache_key = hashlib.sha256(
            f"{symbol}:{threshold}:{start_date}:{end_date}:{OUROBOROS_MODE}".encode()
        ).hexdigest()[:16]

        df["cache_key"] = cache_key
        df["rangebar_version"] = "12.5.2"
        df["source_start_ts"] = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        df["source_end_ts"] = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        # =================================================================
        # Anchor columns for incremental feature addition
        # =================================================================
        # Segment identity - use year as segment ID (e.g., "2024")
        segment_id = str(year)
        segment_start_ms = int(datetime(year, 1, 1).timestamp() * 1000)
        segment_end_ms = int(datetime(year, 12, 31, 23, 59, 59).timestamp() * 1000)

        df["ouroboros_segment_id"] = segment_id
        df["ouroboros_segment_start_timestamp_ms"] = segment_start_ms
        df["ouroboros_segment_end_timestamp_ms"] = segment_end_ms

        # Bar position within segment
        df["bar_position_index_in_segment"] = range(len(df))
        df["bar_position_is_segment_first"] = [1] + [0] * (len(df) - 1) if len(df) > 0 else []
        df["bar_position_is_segment_last"] = [0] * (len(df) - 1) + [1] if len(df) > 0 else []

        # Source tick fingerprint - requires loading tick data
        # Note: For now, set placeholder values. Full fingerprinting requires
        # access to raw tick data which rangebar caches internally.
        # TODO: Add tick fingerprint computation when rangebar exposes tick data
        df["source_tick_xxhash64_checksum"] = 0  # Placeholder
        df["source_tick_row_count"] = 0  # Placeholder
        df["source_tick_first_timestamp_ms"] = df["timestamp_ms"].iloc[0] if len(df) > 0 else 0
        df["source_tick_last_timestamp_ms"] = df["timestamp_ms"].iloc[-1] if len(df) > 0 else 0

        # Feature computation versions - track what's computed
        versions = FeatureComputationVersions(versions=FEATURE_VERSIONS.copy())
        df["feature_computation_versions_json"] = versions.to_json()

        # Drop the original timestamp column (not needed for insert)
        df = df.drop(columns=["timestamp"])

        # Insert into ClickHouse
        client = clickhouse_connect.get_client(
            host="localhost",
            port=8123,
            database="rangebar_cache",
        )

        # df is already pandas from rangebar.get_range_bars()
        # Get column names that exist in target table
        table_columns = [
            "symbol", "threshold_decimal_bps", "timestamp_ms",
            "open", "high", "low", "close", "volume",
            "vwap", "buy_volume", "sell_volume",
            "individual_trade_count", "agg_record_count",
            "duration_us", "ofi", "vwap_close_deviation",
            "price_impact", "kyle_lambda_proxy", "trade_intensity",
            "volume_per_trade", "aggression_ratio", "aggregation_density",
            "turnover_imbalance",
            "exchange_session_sydney", "exchange_session_tokyo",
            "exchange_session_london", "exchange_session_newyork",
            # v11.6.0: 16 inter-bar features (lookback window before bar opens)
            "lookback_trade_count", "lookback_ofi", "lookback_duration_us",
            "lookback_intensity", "lookback_vwap_raw", "lookback_vwap_position",
            "lookback_count_imbalance", "lookback_kyle_lambda", "lookback_burstiness",
            "lookback_volume_skew", "lookback_volume_kurt", "lookback_price_range",
            "lookback_kaufman_er", "lookback_garman_klass_vol", "lookback_hurst",
            "lookback_permutation_entropy",
            "ouroboros_mode", "is_orphan",
            # Anchor columns for incremental feature addition
            "ouroboros_segment_id", "ouroboros_segment_start_timestamp_ms",
            "ouroboros_segment_end_timestamp_ms",
            "source_tick_xxhash64_checksum", "source_tick_row_count",
            "source_tick_first_timestamp_ms", "source_tick_last_timestamp_ms",
            "feature_computation_versions_json",
            "bar_position_index_in_segment", "bar_position_is_segment_first",
            "bar_position_is_segment_last",
            "cache_key", "rangebar_version",  # ouroboros_boundary uses DEFAULT
            "source_start_ts", "source_end_ts",
        ]

        # Filter to columns that exist in dataframe
        insert_columns = [c for c in table_columns if c in df.columns]
        df_insert = df[insert_columns].copy()

        # Fill NaN/None values for Float64 columns (ClickHouse doesn't accept None)
        # lookback_* columns can be None for bars without enough preceding trades
        float_cols = [c for c in df_insert.columns if c.startswith("lookback_")]
        for col in float_cols:
            df_insert[col] = df_insert[col].fillna(0.0)

        client.insert_df(
            table="range_bars_ouroboros_year",
            df=df_insert,
        )

        elapsed = time.time() - start_time
        print(f"[PID {worker_id}] DONE {symbol} @ {threshold}dbps {year}: {len(df):,} bars ({orphan_count} orphans) in {elapsed:.1f}s", flush=True)

        return JobResult(
            symbol=symbol,
            threshold=threshold,
            year=year,
            bars=len(df),
            orphans=orphan_count,
            elapsed=elapsed,
            status="OK",
        )

    except (OSError, ValueError, RuntimeError, ImportError) as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        print(f"[PID {worker_id}] ERROR {symbol} @ {threshold}dbps {year}: {error_msg}", flush=True)
        return JobResult(
            symbol=symbol,
            threshold=threshold,
            year=year,
            bars=0,
            orphans=0,
            elapsed=elapsed,
            status="ERROR",
            error=error_msg,
        )


def main():
    parser = argparse.ArgumentParser(description="Precompute Ouroboros year-based range bars")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols (default: all)")
    parser.add_argument("--thresholds", type=str, help="Comma-separated thresholds (default: all)")
    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else SYMBOLS
    thresholds = [int(t) for t in args.thresholds.split(",")] if args.thresholds else THRESHOLDS

    print("=" * 70)
    print("  Ouroboros Year Range Bar Precompute")
    print("=" * 70)
    print(f"  Symbols: {symbols}")
    print(f"  Thresholds: {thresholds} dbps")
    print(f"  Date range: {START_DATE} to {END_DATE}")
    print(f"  Ouroboros mode: {OUROBOROS_MODE}")
    print(f"  Target table: {TARGET_TABLE}")
    print(f"  Workers: {args.workers}")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 70)
    print()

    # Build job list: one job per symbol/threshold/year
    year_ranges = get_year_ranges(START_DATE, END_DATE)
    jobs = []

    for symbol in symbols:
        for threshold in thresholds:
            for start, end in year_ranges:
                jobs.append((symbol, threshold, start, end, args.dry_run))

    print(f"Total jobs: {len(jobs)}")
    print(f"  {len(symbols)} symbols x {len(thresholds)} thresholds x {len(year_ranges)} years")
    print()

    if args.dry_run:
        print("DRY RUN - showing first 10 jobs:")
        for job in jobs[:10]:
            symbol, threshold, start, end, _ = job
            print(f"  {symbol} @ {threshold}dbps: {start} to {end}")
        if len(jobs) > 10:
            print(f"  ... and {len(jobs) - 10} more")
        return

    import time
    overall_start = time.time()

    # v11.6.0: per-segment tick loading prevents OOM (~3GB per segment vs ~70GB before)
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(run_precompute_job, jobs)

    overall_elapsed = time.time() - overall_start

    # Summary
    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)

    total_bars = sum(r.bars for r in results)
    total_orphans = sum(r.orphans for r in results)
    ok_count = sum(1 for r in results if r.status == "OK")
    error_count = sum(1 for r in results if r.status == "ERROR")
    no_data_count = sum(1 for r in results if r.status == "NO_DATA")

    print(f"  Total bars inserted: {total_bars:,}")
    print(f"  Total orphan bars: {total_orphans:,}")
    print(f"  Jobs OK: {ok_count}")
    print(f"  Jobs NO_DATA: {no_data_count}")
    print(f"  Jobs ERROR: {error_count}")
    print(f"  Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}m)")
    print()

    if error_count > 0:
        print("Errors:")
        for r in results:
            if r.status == "ERROR":
                print(f"  {r.symbol} @ {r.threshold}dbps {r.year}: {r.error}")
        print()

    # Per-symbol/threshold breakdown
    print("Per-symbol breakdown:")
    for symbol in symbols:
        symbol_bars = sum(r.bars for r in results if r.symbol == symbol)
        symbol_orphans = sum(r.orphans for r in results if r.symbol == symbol)
        print(f"  {symbol}: {symbol_bars:,} bars ({symbol_orphans:,} orphans)")

    print()
    print("=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == "__main__":
    # Import polars here for type checking
    main()
