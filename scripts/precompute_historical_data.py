#!/usr/bin/env python3
"""
Precompute historical range bar data from Binance into ClickHouse.

This script fetches trade data from Binance and computes range bars
for multiple symbols and thresholds, storing results in ClickHouse.

Usage:
    python scripts/precompute_historical_data.py

Expected runtime: Several hours for full historical data.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta

from rangebar import PrecomputeProgress, precompute_range_bars

# Configuration
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
THRESHOLDS = [50, 100, 250]  # decimal basis points

# Date range: 3 years of history
END_DATE = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")  # Binance has 1-2 day lag
START_DATE = "2022-01-01"  # ~3 years of data

# Binance data availability (approximate)
SYMBOL_START_DATES = {
    "BTCUSDT": "2019-09-08",  # Binance Spot inception
    "ETHUSDT": "2019-09-08",
    "SOLUSDT": "2020-08-11",  # SOL listing date
    "BNBUSDT": "2019-09-08",
}


def progress_callback(p: PrecomputeProgress) -> None:
    """Print progress updates."""
    pct = (p.months_completed / p.months_total * 100) if p.months_total > 0 else 0
    print(f"  [{p.phase}] {p.current_month}: {pct:.0f}% ({p.bars_generated:,} bars)")


def main() -> int:
    print("=" * 80)
    print("HISTORICAL RANGE BAR PRECOMPUTATION")
    print("=" * 80)
    print()
    print(f"Symbols: {SYMBOLS}")
    print(f"Thresholds: {THRESHOLDS} dbps")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print()

    total_combinations = len(SYMBOLS) * len(THRESHOLDS)
    completed = 0
    failed = 0
    results = []

    for symbol in SYMBOLS:
        # Use symbol-specific start date if available
        start_date = max(START_DATE, SYMBOL_START_DATES.get(symbol, START_DATE))

        for threshold in THRESHOLDS:
            completed += 1
            print(f"[{completed}/{total_combinations}] {symbol} @ {threshold}dbps")
            print(f"    Date range: {start_date} to {END_DATE}")

            try:
                result = precompute_range_bars(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=END_DATE,
                    threshold_decimal_bps=threshold,
                    source="binance",
                    market="spot",
                    progress_callback=progress_callback,
                    validate_on_complete="warn",  # Don't fail on gaps
                    invalidate_existing="smart",  # Only recompute if needed
                )
                results.append((symbol, threshold, result.total_bars, "OK"))
                print(f"    SUCCESS: {result.total_bars:,} bars in {result.elapsed_seconds:.1f}s")
            except (OSError, ValueError, RuntimeError) as e:
                failed += 1
                results.append((symbol, threshold, 0, str(e)))
                print(f"    FAILED: {e}")

            print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Symbol':<10} {'Threshold':>10} {'Bars':>15} {'Status':<30}")
    print("-" * 70)
    for symbol, threshold, bars, status in results:
        print(f"{symbol:<10} {threshold:>10}dbps {bars:>15,} {status:<30}")
    print("-" * 70)
    print(f"Total combinations: {total_combinations}")
    print(f"Successful: {total_combinations - failed}")
    print(f"Failed: {failed}")
    print()

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
