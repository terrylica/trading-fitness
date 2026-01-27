#!/usr/bin/env python3
"""
Parallel precompute historical range bar data from Binance into ClickHouse.

Uses multiprocessing to run multiple symbol/threshold combinations concurrently.

Usage:
    python scripts/precompute_historical_parallel.py [--workers N]

Expected runtime: ~5-10 hours with parallel processing (vs ~60 hours sequential).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta

from rangebar import precompute_range_bars


# Configuration
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
THRESHOLDS = [25, 50, 100, 250]  # decimal basis points (added 25dbps)

# Date range: 3+ years of history
END_DATE = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
START_DATE = "2022-01-01"

# Binance data availability
SYMBOL_START_DATES = {
    "BTCUSDT": "2019-09-08",
    "ETHUSDT": "2019-09-08",
    "SOLUSDT": "2020-08-11",
    "BNBUSDT": "2019-09-08",
}


@dataclass
class JobResult:
    """Result from a precompute job."""

    symbol: str
    threshold: int
    bars: int
    elapsed: float
    status: str
    error: str | None = None


def run_precompute_job(args: tuple[str, int, str, str]) -> JobResult:
    """Run a single precompute job (called in worker process)."""
    symbol, threshold, start_date, end_date = args
    worker_id = os.getpid()

    print(f"[PID {worker_id}] Starting {symbol} @ {threshold}dbps ({start_date} to {end_date})", flush=True)

    try:
        result = precompute_range_bars(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            threshold_decimal_bps=threshold,
            source="binance",
            market="spot",
            progress_callback=None,  # Disable per-month progress in parallel mode
            validate_on_complete="warn",
            invalidate_existing="overlap",  # Invalidate only bars in date range, fill gaps
        )
        print(f"[PID {worker_id}] DONE {symbol} @ {threshold}dbps: {result.total_bars:,} bars in {result.elapsed_seconds:.1f}s", flush=True)
        return JobResult(
            symbol=symbol,
            threshold=threshold,
            bars=result.total_bars,
            elapsed=result.elapsed_seconds,
            status="OK",
        )
    except (OSError, ValueError, RuntimeError, ConnectionError) as e:
        print(f"[PID {worker_id}] FAILED {symbol} @ {threshold}dbps: {e}", flush=True)
        return JobResult(
            symbol=symbol,
            threshold=threshold,
            bars=0,
            elapsed=0,
            status="FAILED",
            error=str(e),
        )


def run_symbol_jobs(args: tuple[str, list[int], str, str]) -> list[JobResult]:
    """Run all thresholds for a single symbol sequentially (avoids tick cache race condition)."""
    symbol, thresholds, start_date, end_date = args
    results = []
    for threshold in thresholds:
        result = run_precompute_job((symbol, threshold, start_date, end_date))
        results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel historical range bar precomputation")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, mp.cpu_count() - 2),  # Leave 2 cores for system
        help="Number of parallel workers (default: min(8, cpu_count-2))",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(SYMBOLS),
        help=f"Comma-separated symbols (default: {','.join(SYMBOLS)})",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=",".join(map(str, THRESHOLDS)),
        help=f"Comma-separated thresholds in dbps (default: {','.join(map(str, THRESHOLDS))})",
    )
    args = parser.parse_args()

    symbols = args.symbols.split(",")
    thresholds = [int(t) for t in args.thresholds.split(",")]
    n_workers = args.workers

    print("=" * 80)
    print("PARALLEL HISTORICAL RANGE BAR PRECOMPUTATION")
    print("=" * 80)
    print()
    print(f"Workers: {n_workers} (of {mp.cpu_count()} CPUs)")
    print(f"Symbols: {symbols}")
    print(f"Thresholds: {thresholds} dbps")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print()

    # Build job list - one job per symbol (thresholds run sequentially within symbol)
    # This avoids race conditions on the tick cache (shared per symbol)
    symbol_jobs: list[tuple[str, list[int], str, str]] = []
    for symbol in symbols:
        start_date = max(START_DATE, SYMBOL_START_DATES.get(symbol, START_DATE))
        symbol_jobs.append((symbol, thresholds, start_date, END_DATE))

    total_jobs = len(symbols) * len(thresholds)
    print(f"Total jobs: {total_jobs} ({len(symbols)} symbols x {len(thresholds)} thresholds)")
    print(f"Parallelization: {min(n_workers, len(symbols))} symbols in parallel, thresholds sequential per symbol")
    print()
    print("Starting parallel execution...")
    print("-" * 80)

    # Run symbols in parallel (max workers = number of symbols to avoid tick cache race)
    start_time = datetime.now()
    effective_workers = min(n_workers, len(symbols))
    with mp.Pool(processes=effective_workers) as pool:
        nested_results = pool.map(run_symbol_jobs, symbol_jobs)
    # Flatten results
    results = [r for symbol_results in nested_results for r in symbol_results]
    total_elapsed = (datetime.now() - start_time).total_seconds()

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Symbol':<10} {'Threshold':>10} {'Bars':>15} {'Time':>10} {'Status':<20}")
    print("-" * 70)

    total_bars = 0
    failed = 0
    for r in results:
        status_str = r.status if r.status == "OK" else f"{r.status}: {r.error[:30]}..."
        time_str = f"{r.elapsed:.1f}s" if r.elapsed > 0 else "-"
        print(f"{r.symbol:<10} {r.threshold:>10}dbps {r.bars:>15,} {time_str:>10} {status_str:<20}")
        total_bars += r.bars
        if r.status != "OK":
            failed += 1

    print("-" * 70)
    print(f"Total bars: {total_bars:,}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"Successful: {len(results) - failed}/{len(results)}")
    if failed > 0:
        print(f"Failed: {failed}")
    print()

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
