"""Audit ITH features with REAL Binance aggTrades data per range bar.

Constructs range bars from raw aggregate trade data, then computes
ITH metrics for each range bar based on the trades within it.

This validates that ITH features produce sensible values at the
granularity of individual range bars using real market data.

Run with:
    uv run python -m ith_python.audit_real_aggtrades
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any

from trading_fitness_metrics import compute_rolling_ith, optimal_tmaeg

# Try to import rangebar for range bar construction
try:
    from rangebar import RangeBarProcessor
    HAS_RANGEBAR = True
except ImportError:
    HAS_RANGEBAR = False


# =============================================================================
# Configuration
# =============================================================================

AUDIT_DIR = Path(__file__).parent.parent.parent / "artifacts" / "feature_audit_aggtrades"

FEATURE_NAMES = [
    "bull_epoch_density",
    "bear_epoch_density",
    "bull_excess_gain",
    "bear_excess_gain",
    "bull_cv",
    "bear_cv",
    "max_drawdown",
    "max_runup",
]

# Sample aggTrades file from rangebar-py
SAMPLE_AGGTRADES = Path.home() / "eon" / "rangebar-py" / "sample_binance_aggTrades.csv"

# Range bar thresholds to test (in decimal basis points)
# 1 dbps = 0.01% = 0.0001
THRESHOLD_CONFIGS = [
    {"threshold_dbps": 10, "lookback": 10, "desc": "10 dbps (0.1%)"},
    {"threshold_dbps": 25, "lookback": 20, "desc": "25 dbps (0.25%)"},
    {"threshold_dbps": 50, "lookback": 20, "desc": "50 dbps (0.5%)"},
    {"threshold_dbps": 100, "lookback": 20, "desc": "100 dbps (1%)"},
]


# =============================================================================
# Range Bar Construction (Pure Python fallback)
# =============================================================================

def construct_range_bars_python(df: pd.DataFrame, threshold_dbps: int) -> pd.DataFrame:
    """Construct range bars from aggTrades using pure Python.

    Args:
        df: DataFrame with 'price', 'quantity', 'timestamp' columns
        threshold_dbps: Range bar threshold in decimal basis points
                       (10 dbps = 0.1% = 0.001)

    Returns:
        DataFrame with OHLCV range bars
    """
    threshold_pct = threshold_dbps / 10000.0  # Convert to decimal

    prices = df["price"].values
    quantities = df["quantity"].values
    timestamps = df["timestamp"].values

    bars = []
    bar_open = prices[0]
    bar_high = prices[0]
    bar_low = prices[0]
    bar_close = prices[0]
    bar_volume = quantities[0]
    bar_start_ts = timestamps[0]
    bar_trades = 1

    for i in range(1, len(prices)):
        price = prices[i]
        qty = quantities[i]
        ts = timestamps[i]

        # Update current bar
        bar_high = max(bar_high, price)
        bar_low = min(bar_low, price)
        bar_close = price
        bar_volume += qty
        bar_trades += 1

        # Check if range threshold breached
        range_pct = (bar_high - bar_low) / bar_open

        if range_pct >= threshold_pct:
            # Close this bar
            bars.append({
                "open": bar_open,
                "high": bar_high,
                "low": bar_low,
                "close": bar_close,
                "volume": bar_volume,
                "timestamp_start": bar_start_ts,
                "timestamp_end": ts,
                "n_trades": bar_trades,
            })

            # Start new bar
            bar_open = price
            bar_high = price
            bar_low = price
            bar_close = price
            bar_volume = 0.0
            bar_start_ts = ts
            bar_trades = 0

    # Don't forget the last incomplete bar
    if bar_trades > 0:
        bars.append({
            "open": bar_open,
            "high": bar_high,
            "low": bar_low,
            "close": bar_close,
            "volume": bar_volume,
            "timestamp_start": bar_start_ts,
            "timestamp_end": timestamps[-1],
            "n_trades": bar_trades,
        })

    return pd.DataFrame(bars)


def construct_range_bars(df: pd.DataFrame, threshold_dbps: int) -> pd.DataFrame:
    """Construct range bars from aggTrades.

    Uses rangebar-py if available, otherwise falls back to pure Python.
    """
    if HAS_RANGEBAR:
        print(f"  Using rangebar-py RangeBarProcessor")
        processor = RangeBarProcessor(threshold_decimal_bps=threshold_dbps)

        # Convert to list of dicts for rangebar-py
        trades = df[["agg_trade_id", "price", "quantity", "timestamp", "is_buyer_maker"]].to_dict("records")
        bars_data = processor.process_trades(trades)

        if not bars_data:
            return pd.DataFrame()

        return pd.DataFrame(bars_data)
    else:
        print(f"  Using pure Python range bar construction")
        return construct_range_bars_python(df, threshold_dbps)


# =============================================================================
# Data Loading
# =============================================================================

def load_aggtrades(filepath: Path) -> pd.DataFrame:
    """Load Binance aggTrades CSV."""
    print(f"  Loading aggTrades from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} trades")
    print(f"  Price range: {df['price'].min():.2f} - {df['price'].max():.2f}")
    return df


def build_nav_from_bars(bars_df: pd.DataFrame) -> np.ndarray:
    """Build NAV series from range bar close prices."""
    closes = bars_df["close"].values.astype(np.float64)
    nav = closes / closes[0]
    return nav


# =============================================================================
# Audit Functions
# =============================================================================

def compute_statistics(values: np.ndarray, name: str) -> dict[str, Any]:
    """Compute comprehensive statistics for a feature."""
    valid = values[~np.isnan(values)]

    if len(valid) == 0:
        return {"count": 0, "name": name}

    return {
        "name": name,
        "count": int(len(valid)),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "median": float(np.median(valid)),
        "p5": float(np.percentile(valid, 5)),
        "p25": float(np.percentile(valid, 25)),
        "p75": float(np.percentile(valid, 75)),
        "p95": float(np.percentile(valid, 95)),
        "zeros_pct": float((valid == 0).sum() / len(valid) * 100),
        "unique_values": int(len(np.unique(valid))),
        # CV-specific: check for the "no epochs" signal (~0.119)
        "at_0119_pct": float(np.sum(np.abs(valid - 0.119) < 0.01) / len(valid) * 100),
    }


def audit_threshold(df: pd.DataFrame, config: dict) -> dict[str, Any]:
    """Run audit for a single threshold configuration."""
    threshold = config["threshold_dbps"]
    lookback = config["lookback"]
    desc = config["desc"]

    print(f"\n{'='*60}")
    print(f"Auditing: {desc}")
    print(f"{'='*60}")

    # Construct range bars from trades
    bars_df = construct_range_bars(df, threshold)
    n_bars = len(bars_df)

    print(f"  Constructed {n_bars} range bars from {len(df)} trades")

    if n_bars <= lookback:
        print(f"  WARNING: Only {n_bars} bars, need >{lookback}. Skipping.")
        return {"status": "skipped", "reason": f"insufficient bars ({n_bars})"}

    # Build NAV and compute features
    nav = build_nav_from_bars(bars_df)

    print(f"  NAV range: {nav.min():.6f} - {nav.max():.6f}")

    features = compute_rolling_ith(nav, lookback=lookback)
    tmaeg = optimal_tmaeg(nav, lookback=lookback)

    print(f"  Auto-TMAEG: {tmaeg:.6f} ({tmaeg*100:.4f}%)")
    print(f"  Valid windows: {n_bars - lookback + 1}")

    # Compute statistics for each feature
    stats = {}
    for name in FEATURE_NAMES:
        arr = getattr(features, name)
        valid_arr = arr[lookback-1:]  # Skip warmup
        stats[name] = compute_statistics(valid_arr, name)

    # Print summary
    print(f"\n  {'Feature':<22} {'Min':>8} {'Max':>8} {'Mean':>8} {'Unique':>8} {'@0.119%':>8}")
    print(f"  {'-'*70}")

    for name, s in stats.items():
        if s["count"] == 0:
            print(f"  {name:<22} {'N/A':>8}")
        else:
            print(f"  {name:<22} {s['min']:>8.4f} {s['max']:>8.4f} "
                  f"{s['mean']:>8.4f} {s['unique_values']:>8} {s['at_0119_pct']:>7.1f}%")

    # Check epoch counts
    bull_has_epochs = np.sum(features.bull_epoch_density[lookback-1:] > 0)
    bear_has_epochs = np.sum(features.bear_epoch_density[lookback-1:] > 0)
    total_windows = n_bars - lookback + 1

    print(f"\n  Windows with bull epochs: {bull_has_epochs}/{total_windows} ({100*bull_has_epochs/total_windows:.1f}%)")
    print(f"  Windows with bear epochs: {bear_has_epochs}/{total_windows} ({100*bear_has_epochs/total_windows:.1f}%)")

    # Check CV distribution (key metric we're investigating)
    bull_cv = features.bull_cv[lookback-1:]
    bear_cv = features.bear_cv[lookback-1:]

    bull_cv_unique = len(np.unique(bull_cv[~np.isnan(bull_cv)]))
    bear_cv_unique = len(np.unique(bear_cv[~np.isnan(bear_cv)]))

    print(f"\n  bull_cv unique values: {bull_cv_unique}")
    print(f"  bear_cv unique values: {bear_cv_unique}")

    if bull_cv_unique > 1:
        unique_vals = np.unique(bull_cv[~np.isnan(bull_cv)])
        print(f"  bull_cv sample values: {unique_vals[:10]}")
    if bear_cv_unique > 1:
        unique_vals = np.unique(bear_cv[~np.isnan(bear_cv)])
        print(f"  bear_cv sample values: {unique_vals[:10]}")

    # Save artifacts
    config_name = f"threshold_{threshold}dbps"

    # Save CSV with per-bar features
    csv_data = {
        "bar_idx": list(range(n_bars)),
        "open": bars_df["open"].tolist(),
        "high": bars_df["high"].tolist(),
        "low": bars_df["low"].tolist(),
        "close": bars_df["close"].tolist(),
        "nav": nav.tolist(),
    }
    for name in FEATURE_NAMES:
        csv_data[name] = getattr(features, name).tolist()

    csv_df = pd.DataFrame(csv_data)
    csv_path = AUDIT_DIR / f"{config_name}_features.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path.name}")

    # Save stats JSON
    result = {
        "config": {
            "threshold_dbps": threshold,
            "lookback": lookback,
            "desc": desc,
        },
        "n_trades": len(df),
        "n_bars": n_bars,
        "tmaeg": tmaeg,
        "windows_with_bull_epochs": int(bull_has_epochs),
        "windows_with_bear_epochs": int(bear_has_epochs),
        "bull_cv_unique_values": bull_cv_unique,
        "bear_cv_unique_values": bear_cv_unique,
        "statistics": stats,
    }

    stats_path = AUDIT_DIR / f"{config_name}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {stats_path.name}")

    return result


def main():
    """Run audit on aggTrades data with various range bar thresholds."""
    print("="*70)
    print("ITH FEATURE AUDIT - REAL BINANCE AGGTRADES -> RANGE BARS")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"rangebar-py available: {HAS_RANGEBAR}")

    # Create output directory
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # Load real data
    if not SAMPLE_AGGTRADES.exists():
        raise FileNotFoundError(f"Sample aggTrades not found: {SAMPLE_AGGTRADES}")

    df = load_aggtrades(SAMPLE_AGGTRADES)

    # Run all configs
    results = []
    for config in THRESHOLD_CONFIGS:
        result = audit_threshold(df, config)
        results.append(result)

    # Save master summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_source": str(SAMPLE_AGGTRADES),
        "n_trades": len(df),
        "rangebar_py_available": HAS_RANGEBAR,
        "configs": [c["desc"] for c in THRESHOLD_CONFIGS],
        "results": results,
    }

    summary_path = AUDIT_DIR / "AUDIT_SUMMARY.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for config, result in zip(THRESHOLD_CONFIGS, results):
        if "status" in result and result["status"] in ("skipped", "error"):
            print(f"  {config['desc']}: {result.get('status', 'unknown')}")
        else:
            tmaeg = result.get("tmaeg", 0)
            n_bars = result.get("n_bars", 0)
            bull = result.get("windows_with_bull_epochs", 0)
            bear = result.get("windows_with_bear_epochs", 0)
            bull_cv = result.get("bull_cv_unique_values", 0)
            bear_cv = result.get("bear_cv_unique_values", 0)
            print(f"  {config['desc']}: bars={n_bars}, TMAEG={tmaeg:.4f}, "
                  f"epochs=(bull={bull}, bear={bear}), CV_unique=(bull={bull_cv}, bear={bear_cv})")

    print(f"\nArtifacts saved to: {AUDIT_DIR}")


if __name__ == "__main__":
    main()
