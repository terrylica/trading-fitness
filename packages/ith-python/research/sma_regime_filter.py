#!/usr/bin/env python3
"""SMA Crossover Regime Filter — Research Notebook.

Hypothesis: Filtering ITH fitness signals by SMA crossover direction
improves strategy performance. Bull ITH epochs should be stronger
when SMA(fast) > SMA(slow) (bullish regime), and vice versa.

Three-way comparison:
  1. Unfiltered (baseline) — all bars
  2. Aligned — bull features in bullish regime, bear in bearish
  3. Anti-aligned (control) — opposite direction (should be worse)

Usage:
    RANGEBAR_CH_HOSTS=littleblack RANGEBAR_CRYPTO_MIN_THRESHOLD=1 \
      uv run python research/sma_regime_filter.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table
from scipy import stats as sp_stats

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
THRESHOLD_DBPS = 1000
LOOKBACKS = [20, 50, 100, 200, 500]
MA_PAIRS = [(10, 30), (20, 50), (50, 200)]

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "research" / "sma_regime_filter"

console = Console()

# ---------------------------------------------------------------------------
# SMA Computation
# ---------------------------------------------------------------------------


def sma(close: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """Simple moving average with NaN warmup."""
    out = np.full_like(close, np.nan, dtype=np.float64)
    cumsum = np.cumsum(close)
    out[period - 1 :] = (
        cumsum[period - 1 :] - np.concatenate([[0.0], cumsum[: -period]])
    ) / period
    return out


def sma_regime(
    close: NDArray[np.float64], fast: int, slow: int
) -> NDArray[np.str_]:
    """Classify each bar: 'bullish' when SMA(fast) > SMA(slow), else 'bearish'.

    First (slow - 1) bars are labeled 'warmup'.
    """
    sma_fast = sma(close, fast)
    sma_slow = sma(close, slow)
    regimes = np.where(sma_fast > sma_slow, "bullish", "bearish")
    regimes[: slow - 1] = "warmup"
    return regimes


# ---------------------------------------------------------------------------
# Effect size (reuse from statistical_examination.regime)
# ---------------------------------------------------------------------------


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's Delta via Mann-Whitney U."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0.0
    U, _ = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(1 - 2 * U / (n1 * n2))


def interpret_delta(delta: float) -> str:
    abs_d = abs(delta)
    if abs_d < 0.05:
        return "negligible"
    if abs_d < 0.15:
        return "small_tradable"
    if abs_d < 0.30:
        return "medium_strong"
    return "large_suspicious"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_range_bars(symbol: str) -> pl.DataFrame:
    """Load range bar Close prices from ClickHouse directly.

    Bypasses rangebar-py's 30-day limit by querying the cache table directly.
    The data for 1000dbps is already populated with lookback columns.
    """
    import clickhouse_connect

    ch_host = os.environ.get("RANGEBAR_CH_HOSTS", "172.25.236.1")
    console.print(f"[bold]Loading {symbol} @ {THRESHOLD_DBPS}dbps from {ch_host}...[/bold]")

    client = clickhouse_connect.get_client(host=ch_host)
    result = client.query(
        "SELECT close FROM rangebar_cache.range_bars FINAL "
        "WHERE symbol = %(symbol)s AND threshold_decimal_bps = %(threshold)s "
        "ORDER BY timestamp_ms ASC",
        parameters={"symbol": symbol, "threshold": THRESHOLD_DBPS},
    )

    close = np.array([row[0] for row in result.result_rows], dtype=np.float64)
    client.close()

    console.print(f"  {symbol}: {len(close)} bars loaded")
    return pl.DataFrame({"close": close})


# ---------------------------------------------------------------------------
# ITH Feature Computation
# ---------------------------------------------------------------------------


def compute_ith_features(close: NDArray[np.float64]) -> pl.DataFrame:
    """Compute 40 ITH features from close prices."""
    from ith_python.features import FeatureConfig, compute_features

    nav = close / close[0]  # Normalize to NAV starting at 1.0
    config = FeatureConfig(lookbacks=LOOKBACKS, threshold_dbps=THRESHOLD_DBPS)
    return compute_features(nav, config, emit_telemetry=False)


# ---------------------------------------------------------------------------
# Three-Way Comparison
# ---------------------------------------------------------------------------


BULL_FEATURES = ["bull_ed", "bull_eg", "bull_cv", "max_ru"]
BEAR_FEATURES = ["bear_ed", "bear_eg", "bear_cv", "max_dd"]


def run_comparison(
    features_df: pl.DataFrame,
    regimes: NDArray[np.str_],
    ma_label: str,
) -> list[dict]:
    """Compare aligned vs unfiltered vs anti-aligned for all features."""
    results = []

    # Add regime column
    df = features_df.with_columns(pl.Series("regime", regimes))

    # Filter out warmup bars for both ITH and SMA
    valid_mask = pl.col("regime") != "warmup"

    for lb in LOOKBACKS:
        for feat_base in ["bull_ed", "bear_ed", "bull_eg", "bear_eg",
                          "bull_cv", "bear_cv", "max_dd", "max_ru"]:
            col = f"ith_rb{THRESHOLD_DBPS}_lb{lb}_{feat_base}"
            if col not in df.columns:
                continue

            # Determine direction
            is_bull = feat_base.startswith("bull") or feat_base == "max_ru"

            # Also filter out ITH warmup (NaN values stored as float NaN)
            not_nan_mask = pl.col(col).is_not_nan()

            # Unfiltered (baseline): all valid bars
            unfiltered = (
                df.filter(valid_mask & not_nan_mask)
                .get_column(col)
                .to_numpy()
            )

            if len(unfiltered) < 30:
                continue

            # Aligned: bull features in bullish, bear features in bearish
            aligned_regime = "bullish" if is_bull else "bearish"
            aligned = (
                df.filter(valid_mask & not_nan_mask & (pl.col("regime") == aligned_regime))
                .get_column(col)
                .to_numpy()
            )

            # Anti-aligned (control): opposite
            anti_regime = "bearish" if is_bull else "bullish"
            anti_aligned = (
                df.filter(valid_mask & not_nan_mask & (pl.col("regime") == anti_regime))
                .get_column(col)
                .to_numpy()
            )

            if len(aligned) < 30 or len(anti_aligned) < 30:
                continue

            # Statistical tests: aligned vs unfiltered
            delta_aligned = cliffs_delta(aligned, unfiltered)

            # Statistical tests: aligned vs anti-aligned
            delta_vs_anti = cliffs_delta(aligned, anti_aligned)
            _, p_vs_anti = sp_stats.mannwhitneyu(
                aligned, anti_aligned, alternative="two-sided"
            )

            results.append({
                "ma_pair": ma_label,
                "lookback": lb,
                "feature": feat_base,
                "column": col,
                "direction": "bull" if is_bull else "bear",
                "n_unfiltered": len(unfiltered),
                "n_aligned": len(aligned),
                "n_anti_aligned": len(anti_aligned),
                "mean_unfiltered": float(np.mean(unfiltered)),
                "mean_aligned": float(np.mean(aligned)),
                "mean_anti_aligned": float(np.mean(anti_aligned)),
                "delta_aligned_vs_unfiltered": delta_aligned,
                "delta_aligned_vs_anti": delta_vs_anti,
                "p_aligned_vs_anti": p_vs_anti,
                "effect_vs_unfiltered": interpret_delta(delta_aligned),
                "effect_vs_anti": interpret_delta(delta_vs_anti),
                "hypothesis_supported": (
                    # For ed/eg/ru: aligned mean should be HIGHER
                    # For cv/dd: aligned mean should be LOWER (better regularity / less drawdown)
                    (feat_base in ("bull_ed", "bear_ed", "bull_eg", "bear_eg", "max_ru")
                     and np.mean(aligned) > np.mean(unfiltered))
                    or
                    (feat_base in ("bull_cv", "bear_cv", "max_dd")
                     and np.mean(aligned) < np.mean(unfiltered))
                ),
            })

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def build_charts(all_results: dict[str, list[dict]]) -> str:
    """Build interactive plotly HTML charts."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Collect all results across symbols
    rows = []
    for symbol, results in all_results.items():
        for r in results:
            rows.append({**r, "symbol": symbol})

    if not rows:
        return "<p>No results to chart</p>"

    df = pl.DataFrame(rows)

    # Chart 1: Cliff's Delta heatmap (aligned vs unfiltered) per MA pair
    fig = make_subplots(
        rows=len(MA_PAIRS),
        cols=len(SYMBOLS),
        subplot_titles=[
            f"{sym} — SMA({f},{s})"
            for f, s in MA_PAIRS
            for sym in SYMBOLS
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    for i, (fast, slow) in enumerate(MA_PAIRS):
        ma_label = f"SMA({fast},{slow})"
        for j, symbol in enumerate(SYMBOLS):
            subset = df.filter(
                (pl.col("ma_pair") == ma_label)
                & (pl.col("symbol") == symbol)
                & pl.col("feature").is_in(["bull_ed", "bear_ed", "bull_eg", "bear_eg"])
            )

            if len(subset) == 0:
                continue

            features = subset.get_column("feature").unique().sort().to_list()
            lookbacks_list = sorted(subset.get_column("lookback").unique().to_list())

            z = []
            for feat in features:
                row_vals = []
                for lb in lookbacks_list:
                    val = subset.filter(
                        (pl.col("feature") == feat) & (pl.col("lookback") == lb)
                    ).get_column("delta_aligned_vs_unfiltered")
                    row_vals.append(float(val[0]) if len(val) > 0 else 0.0)
                z.append(row_vals)

            fig.add_trace(
                go.Heatmap(
                    z=z,
                    x=[f"lb{lb}" for lb in lookbacks_list],
                    y=features,
                    colorscale="RdBu",
                    zmid=0,
                    zmin=-0.3,
                    zmax=0.3,
                    showscale=(i == 0 and j == 0),
                    colorbar=dict(title="Cliff's δ"),
                ),
                row=i + 1,
                col=j + 1,
            )

    fig.update_layout(
        title="SMA Regime Filter: Cliff's Delta (Aligned vs Unfiltered)",
        height=300 * len(MA_PAIRS) + 100,
        width=500 * len(SYMBOLS) + 100,
    )

    # Chart 2: Mean comparison bar chart for key features
    fig2 = make_subplots(
        rows=len(SYMBOLS),
        cols=len(MA_PAIRS),
        subplot_titles=[
            f"{sym} — SMA({f},{s})"
            for sym in SYMBOLS
            for f, s in MA_PAIRS
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )

    for i, symbol in enumerate(SYMBOLS):
        for j, (fast, slow) in enumerate(MA_PAIRS):
            ma_label = f"SMA({fast},{slow})"
            subset = df.filter(
                (pl.col("ma_pair") == ma_label)
                & (pl.col("symbol") == symbol)
                & (pl.col("lookback") == 100)  # Representative lookback
                & pl.col("feature").is_in(["bull_ed", "bear_ed", "bull_eg", "bear_eg"])
            )

            if len(subset) == 0:
                continue

            features = subset.get_column("feature").to_list()
            means_unf = subset.get_column("mean_unfiltered").to_list()
            means_ali = subset.get_column("mean_aligned").to_list()
            means_anti = subset.get_column("mean_anti_aligned").to_list()

            fig2.add_trace(
                go.Bar(name="Unfiltered", x=features, y=means_unf,
                       marker_color="#636EFA", showlegend=(i == 0 and j == 0)),
                row=i + 1, col=j + 1,
            )
            fig2.add_trace(
                go.Bar(name="Aligned", x=features, y=means_ali,
                       marker_color="#00CC96", showlegend=(i == 0 and j == 0)),
                row=i + 1, col=j + 1,
            )
            fig2.add_trace(
                go.Bar(name="Anti-aligned", x=features, y=means_anti,
                       marker_color="#EF553B", showlegend=(i == 0 and j == 0)),
                row=i + 1, col=j + 1,
            )

    fig2.update_layout(
        title="Mean Feature Values at lb100: Unfiltered vs Aligned vs Anti-aligned",
        barmode="group",
        height=350 * len(SYMBOLS) + 100,
        width=400 * len(MA_PAIRS) + 100,
    )

    html = (
        "<html><head><title>SMA Regime Filter Research</title></head><body>"
        + fig.to_html(full_html=False, include_plotlyjs="cdn")
        + "<hr/>"
        + fig2.to_html(full_html=False, include_plotlyjs=False)
        + "</body></html>"
    )
    return html


# ---------------------------------------------------------------------------
# Console Summary
# ---------------------------------------------------------------------------


def print_summary(all_results: dict[str, list[dict]]) -> None:
    """Print Rich summary tables."""
    for symbol, results in all_results.items():
        console.print(f"\n[bold cyan]═══ {symbol} ═══[/bold cyan]")

        for fast, slow in MA_PAIRS:
            ma_label = f"SMA({fast},{slow})"
            ma_results = [r for r in results if r["ma_pair"] == ma_label]

            if not ma_results:
                continue

            table = Table(title=f"{ma_label}")
            table.add_column("Feature", style="bold")
            table.add_column("LB", justify="right")
            table.add_column("μ Unfilt", justify="right")
            table.add_column("μ Aligned", justify="right")
            table.add_column("μ Anti", justify="right")
            table.add_column("δ vs Unfilt", justify="right")
            table.add_column("δ vs Anti", justify="right")
            table.add_column("p-value", justify="right")
            table.add_column("Effect", style="dim")
            table.add_column("✓", justify="center")

            for r in sorted(ma_results, key=lambda x: (x["feature"], x["lookback"])):
                # Color-code delta
                d_unfilt = r["delta_aligned_vs_unfiltered"]
                d_anti = r["delta_aligned_vs_anti"]
                supported = "✅" if r["hypothesis_supported"] else "❌"

                table.add_row(
                    r["feature"],
                    str(r["lookback"]),
                    f"{r['mean_unfiltered']:.4f}",
                    f"{r['mean_aligned']:.4f}",
                    f"{r['mean_anti_aligned']:.4f}",
                    f"{d_unfilt:+.4f}",
                    f"{d_anti:+.4f}",
                    f"{r['p_aligned_vs_anti']:.2e}",
                    r["effect_vs_anti"],
                    supported,
                )

            console.print(table)

        # Summary stats
        supported = sum(1 for r in results if r["hypothesis_supported"])
        total = len(results)
        console.print(
            f"\n  Hypothesis supported: [bold]{supported}/{total}[/bold] "
            f"({supported/total*100:.0f}%)"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    console.print("[bold green]SMA Crossover Regime Filter Research[/bold green]")
    console.print(f"MA Pairs: {MA_PAIRS}")
    console.print(f"Symbols: {SYMBOLS}")
    console.print(f"Lookbacks: {LOOKBACKS}")
    console.print()

    all_results: dict[str, list[dict]] = {}

    for symbol in SYMBOLS:
        # Load data
        df = load_range_bars(symbol)
        close = df.get_column("close").to_numpy()

        # Compute ITH features
        features_df = compute_ith_features(close)

        # Run comparison for each MA pair
        symbol_results = []
        for fast, slow in MA_PAIRS:
            ma_label = f"SMA({fast},{slow})"
            console.print(f"  Testing {symbol} with {ma_label}...")

            regimes = sma_regime(close, fast, slow)

            # Ensure regime array aligns with features
            results = run_comparison(features_df, regimes, ma_label)
            symbol_results.extend(results)

        all_results[symbol] = symbol_results

    # Print console summary
    print_summary(all_results)

    # Save artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON summary
    summary_path = ARTIFACTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    console.print(f"\n[dim]Saved: {summary_path}[/dim]")

    # HTML charts
    html = build_charts(all_results)
    charts_path = ARTIFACTS_DIR / "charts.html"
    charts_path.write_text(html)
    console.print(f"[dim]Saved: {charts_path}[/dim]")

    # Overall verdict
    total_supported = sum(
        1 for results in all_results.values()
        for r in results if r["hypothesis_supported"]
    )
    total_tests = sum(len(r) for r in all_results.values())
    pct = total_supported / total_tests * 100 if total_tests > 0 else 0

    console.print(f"\n[bold]Overall: {total_supported}/{total_tests} ({pct:.0f}%) hypothesis supported[/bold]")

    if pct > 60:
        console.print("[bold green]Verdict: SMA regime filter shows promise — worth integrating.[/bold green]")
    elif pct > 40:
        console.print("[bold yellow]Verdict: Mixed results — further investigation needed.[/bold yellow]")
    else:
        console.print("[bold red]Verdict: SMA regime filter does not improve ITH signals.[/bold red]")


if __name__ == "__main__":
    main()
