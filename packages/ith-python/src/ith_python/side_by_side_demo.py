"""Side-by-side demo: Run BOTH Numba and Rust on identical NAV data.

This generates HTML output for BOTH implementations using the exact same
NAV data, proving they produce identical results.

Uses the ORIGINAL NAV generation method from:
/Users/terryli/eon/legal-docs-source/candidates/MingXu/ith.py

Key difference from rust_ith_demo.py:
- Original uses ADDITIVE returns: np.cumsum(walk)
- rust_ith_demo uses MULTIPLICATIVE returns: np.cumprod(1 + returns)

Usage:
    uv run python -m ith_python.side_by_side_demo
"""

from __future__ import annotations

import webbrowser
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from rich.console import Console
from rich.table import Table
from scipy import stats

# Import Numba implementations
from ith_python.bull_ith_numba import bull_excess_gain_excess_loss as bull_ith_numba
from ith_python.bear_ith_numba import bear_excess_gain_excess_loss as bear_ith_numba

# Import Rust implementations
from trading_fitness_metrics import (
    bull_ith as bull_ith_rust,
    bear_ith as bear_ith_rust,
)

from ith_python.paths import get_artifacts_dir, ensure_dirs

console = Console()


class SyntheticNavParams(NamedTuple):
    """Parameters for synthetic NAV generation - EXACT same as original ith.py."""
    start_date: str = '2020-01-30'
    end_date: str = '2023-07-25'
    avg_daily_return: float = 0.00010123
    daily_return_volatility: float = 0.009
    df: int = 5  # t-distribution degrees of freedom
    drawdown_prob: float = 0.05
    drawdown_magnitude_low: float = 0.001
    drawdown_magnitude_high: float = 0.003
    drawdown_recovery_prob: float = 0.02


def generate_synthetic_nav_original(params: SyntheticNavParams, seed: int = 42) -> np.ndarray:
    """Generate synthetic NAV - EXACT same as original ith.py.

    Uses ADDITIVE returns (np.cumsum) instead of multiplicative (np.cumprod).
    This is the key difference from rust_ith_demo.py.
    """
    np.random.seed(seed)
    dates = pd.date_range(params.start_date, params.end_date)
    n_points = len(dates)

    # Generate t-distributed returns
    walk = stats.t.rvs(
        params.df,
        loc=params.avg_daily_return,
        scale=params.daily_return_volatility,
        size=n_points,
    )

    # ADDITIVE cumsum (original method)
    walk = np.cumsum(walk)

    # Apply drawdown events
    drawdown = False
    for i in range(n_points):
        if drawdown:
            walk[i] -= np.random.uniform(
                params.drawdown_magnitude_low,
                params.drawdown_magnitude_high,
            )
            if np.random.rand() < params.drawdown_recovery_prob:
                drawdown = False
        elif np.random.rand() < params.drawdown_prob:
            drawdown = True

    # Normalize so series starts at 1
    walk = walk - walk[0] + 1
    return walk


def max_drawdown(nav_values: np.ndarray) -> float:
    """Calculate maximum drawdown - EXACT same as original ith.py."""
    return float(np.max(1 - nav_values / np.maximum.accumulate(nav_values)))


def create_comparison_plot(
    nav: np.ndarray,
    numba_result,
    rust_result,
    title: str,
    is_bull: bool = True,
) -> go.Figure:
    """Create side-by-side comparison plot."""
    x = np.arange(len(nav))

    # Get arrays
    numba_eg = numba_result.excess_gains
    numba_el = numba_result.excess_losses
    numba_epochs = numba_result.bull_epochs if is_bull else numba_result.bear_epochs

    rust_eg = rust_result.excess_gains()
    rust_el = rust_result.excess_losses()
    rust_epochs = rust_result.epochs()

    # Find epoch indices
    numba_epoch_idx = np.where(numba_epochs)[0]
    rust_epoch_idx = np.where(rust_epochs)[0]

    fig = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        subplot_titles=(
            "Numba - NAV", "Rust - NAV",
            "Numba - Excess Gains", "Rust - Excess Gains",
            "Numba - Excess Losses", "Rust - Excess Losses",
        ),
    )

    # NAV traces (same for both)
    for col in [1, 2]:
        fig.add_trace(
            go.Scatter(x=x, y=nav, mode="lines", name="NAV", line=dict(color="cyan")),
            row=1, col=col,
        )

    # Numba epochs on NAV
    if len(numba_epoch_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=numba_epoch_idx, y=nav[numba_epoch_idx],
                mode="markers", name=f"Numba Epochs ({len(numba_epoch_idx)})",
                marker=dict(color="gold", size=10, symbol="star"),
            ),
            row=1, col=1,
        )

    # Rust epochs on NAV
    if len(rust_epoch_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=rust_epoch_idx, y=nav[rust_epoch_idx],
                mode="markers", name=f"Rust Epochs ({len(rust_epoch_idx)})",
                marker=dict(color="gold", size=10, symbol="star"),
            ),
            row=1, col=2,
        )

    # Excess Gains
    fig.add_trace(
        go.Scatter(x=x, y=numba_eg, mode="lines", name="Numba EG", line=dict(color="green")),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=rust_eg, mode="lines", name="Rust EG", line=dict(color="green")),
        row=2, col=2,
    )

    # Excess Losses
    fig.add_trace(
        go.Scatter(x=x, y=numba_el, mode="lines", name="Numba EL", line=dict(color="red")),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=rust_el, mode="lines", name="Rust EL", line=dict(color="red")),
        row=3, col=2,
    )

    # Check if identical
    eg_match = np.allclose(numba_eg, rust_eg, rtol=1e-12)
    el_match = np.allclose(numba_el, rust_el, rtol=1e-12)
    epochs_match = np.array_equal(numba_epochs, rust_epochs)

    match_status = "✓ IDENTICAL" if (eg_match and el_match and epochs_match) else "✗ DIFFERENT"

    fig.update_layout(
        title=dict(
            text=f"{title}<br>Numba vs Rust: {match_status} | Epochs: {len(numba_epoch_idx)} vs {len(rust_epoch_idx)}",
            font=dict(size=14),
        ),
        height=900,
        paper_bgcolor="DarkSlateGrey",
        plot_bgcolor="Black",
        font=dict(color="White", family="Monospace"),
        showlegend=True,
    )

    fig.update_xaxes(gridcolor="dimgray")
    fig.update_yaxes(gridcolor="dimgray")

    return fig


def main():
    """Run side-by-side comparison demo using ORIGINAL ith.py NAV generation."""
    ensure_dirs()
    output_dir = get_artifacts_dir() / "side_by_side_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]Side-by-Side Demo: Numba vs Rust on Identical NAV Data[/bold cyan]")
    console.print("[bold yellow]Using ORIGINAL ith.py NAV generation (np.cumsum)[/bold yellow]")
    console.print("=" * 60)

    params = SyntheticNavParams()
    results = []

    for i in range(5):
        seed = 42 + i
        console.print(f"\n[yellow]Processing NAV #{i+1} (seed={seed})...[/yellow]")

        # Generate NAV using ORIGINAL method (ADDITIVE np.cumsum)
        nav = generate_synthetic_nav_original(params, seed=seed)

        # Use FIXED TMAEG of 0.05 for meaningful comparison (original ith.py default)
        # Dynamic TMAEG produces 30-70% values which result in very few epochs
        tmaeg = 0.05
        mdd = max_drawdown(nav)
        console.print(f"  NAV points: {len(nav)}, Max Drawdown: {mdd:.4f}, TMAEG: {tmaeg}")

        # Run BOTH implementations on SAME NAV
        bull_numba = bull_ith_numba(nav, tmaeg)
        bull_rust = bull_ith_rust(nav, tmaeg)
        bear_numba = bear_ith_numba(nav, tmaeg)
        bear_rust = bear_ith_rust(nav, tmaeg)

        # Compare results
        bull_eg_match = np.allclose(bull_numba.excess_gains, bull_rust.excess_gains(), rtol=1e-12)
        bull_el_match = np.allclose(bull_numba.excess_losses, bull_rust.excess_losses(), rtol=1e-12)
        bull_epochs_match = np.array_equal(bull_numba.bull_epochs, bull_rust.epochs())

        bear_eg_match = np.allclose(bear_numba.excess_gains, bear_rust.excess_gains(), rtol=1e-12)
        bear_el_match = np.allclose(bear_numba.excess_losses, bear_rust.excess_losses(), rtol=1e-12)
        bear_epochs_match = np.array_equal(bear_numba.bear_epochs, bear_rust.epochs())

        bull_match = bull_eg_match and bull_el_match and bull_epochs_match
        bear_match = bear_eg_match and bear_el_match and bear_epochs_match

        console.print(f"  Bull: Numba={bull_numba.num_of_bull_epochs} epochs, Rust={bull_rust.num_of_epochs} epochs, Match={bull_match}")
        console.print(f"  Bear: Numba={bear_numba.num_of_bear_epochs} epochs, Rust={bear_rust.num_of_epochs} epochs, Match={bear_match}")

        # Create comparison plots
        bull_fig = create_comparison_plot(nav, bull_numba, bull_rust, f"Bull ITH #{i+1} (seed={seed})", is_bull=True)
        bear_fig = create_comparison_plot(nav, bear_numba, bear_rust, f"Bear ITH #{i+1} (seed={seed})", is_bull=False)

        # Save HTML
        bull_html = output_dir / f"comparison_bull_{i+1}.html"
        bear_html = output_dir / f"comparison_bear_{i+1}.html"
        bull_fig.write_html(str(bull_html), include_plotlyjs="cdn")
        bear_fig.write_html(str(bear_html), include_plotlyjs="cdn")

        results.append({
            "ID": i + 1,
            "Seed": seed,
            "TMAEG": f"{tmaeg:.4f}",
            "Bull Numba": bull_numba.num_of_bull_epochs,
            "Bull Rust": bull_rust.num_of_epochs,
            "Bull Match": "✓" if bull_match else "✗",
            "Bear Numba": bear_numba.num_of_bear_epochs,
            "Bear Rust": bear_rust.num_of_epochs,
            "Bear Match": "✓" if bear_match else "✗",
        })

    # Print summary table
    console.print("\n" + "=" * 60)
    table = Table(title="Side-by-Side Comparison Summary")
    table.add_column("ID", style="cyan")
    table.add_column("Seed")
    table.add_column("TMAEG", justify="right")
    table.add_column("Bull Numba", justify="right")
    table.add_column("Bull Rust", justify="right")
    table.add_column("Bull Match", justify="center")
    table.add_column("Bear Numba", justify="right")
    table.add_column("Bear Rust", justify="right")
    table.add_column("Bear Match", justify="center")

    for r in results:
        table.add_row(
            str(r["ID"]),
            str(r["Seed"]),
            str(r["TMAEG"]),
            str(r["Bull Numba"]),
            str(r["Bull Rust"]),
            r["Bull Match"],
            str(r["Bear Numba"]),
            str(r["Bear Rust"]),
            r["Bear Match"],
        )

    console.print(table)

    # Create summary HTML
    df = pd.DataFrame(results)
    html_table = df.to_html(classes="table", index=False, escape=False)

    summary_html = f"""
    <html>
    <head>
    <title>Side-by-Side Comparison: Numba vs Rust ITH</title>
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.css">
    <style>
        body {{ font-family: monospace; padding: 20px; background-color: #1a1a1a; color: #e0e0e0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ padding: 10px; border: 1px solid #444; text-align: center; }}
        th {{ background-color: #333; }}
        .info {{ background-color: #2a2a2a; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        a {{ color: #6af; }}
    </style>
    </head>
    <body>
    <h1>Side-by-Side Comparison: Numba vs Rust ITH</h1>
    <div class="info">
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>NAV Generation:</strong> ORIGINAL ith.py method (ADDITIVE np.cumsum)</p>
        <p><strong>Date Range:</strong> {params.start_date} to {params.end_date}</p>
        <p><strong>TMAEG:</strong> Fixed 0.05 (original default)</p>
        <p><strong>Same NAV data used for BOTH implementations</strong></p>
    </div>
    {html_table}
    <h2>Comparison Charts</h2>
    <ul>
    {"".join(f'<li><a href="comparison_bull_{i+1}.html">Bull #{i+1}</a> | <a href="comparison_bear_{i+1}.html">Bear #{i+1}</a></li>' for i in range(5))}
    </ul>
    </body>
    </html>
    """

    summary_path = output_dir / "index.html"
    summary_path.write_text(summary_html)

    console.print(f"\n[green]Output saved to: {output_dir}[/green]")
    console.print(f"Open {summary_path} in browser")

    webbrowser.open(f"file://{summary_path.resolve()}")


if __name__ == "__main__":
    main()
