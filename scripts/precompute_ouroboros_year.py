#!/usr/bin/env python3
"""
Precompute Ouroboros year-based range bars into ClickHouse.

Builds range bars with yearly reset boundaries (Jan 1 00:00 UTC).
Bars crossing year boundaries are marked as orphans.

Usage:
    # On bigblack:
    cd ~/eon/trading-fitness/packages/ith-python
    uv run python ../../scripts/precompute_ouroboros_year.py [--workers N] [--dry-run]

Target table: rangebar_cache.range_bars_ouroboros_year
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# =============================================================================
# Configuration
# =============================================================================

# Symbols and thresholds to precompute
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
THRESHOLDS = [25, 50, 100, 250]  # decimal basis points

# Date range: 4+ years (2022-01-01 to present)
# Binance data availability starts ~2019 for major pairs
START_DATE = "2022-01-01"
END_DATE = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")

# Ouroboros mode
OUROBOROS_MODE = "year"

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
        from rangebar import get_range_bars

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
            use_cache=True,  # Use tick cache
            fetch_if_missing=True,
            include_microstructure=True,
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
        df["rangebar_version"] = "11.0.0"
        df["source_start_ts"] = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        df["source_end_ts"] = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

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
            "turnover_imbalance", "ouroboros_mode", "is_orphan",
            "cache_key", "rangebar_version",  # ouroboros_boundary uses DEFAULT
            "source_start_ts", "source_end_ts",
        ]

        # Filter to columns that exist in dataframe
        insert_columns = [c for c in table_columns if c in df.columns]
        df_insert = df[insert_columns]

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
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
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

    # Run jobs in parallel
    import time
    overall_start = time.time()

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
