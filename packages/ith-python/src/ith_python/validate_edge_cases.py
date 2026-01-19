"""Validation PNG generator for Bear ITH edge cases.

This module generates side-by-side charts showing long and short epochs
for visual inspection during algorithm validation.

SR&ED: Visual validation tooling for Bear ITH experimental development.
SRED-Type: experimental-development
SRED-Claim: BEAR-ITH
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add tests to path for importing edge cases
tests_dir = Path(__file__).parent.parent.parent / "tests"
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from ith_python.bear_ith_numba import bear_excess_gain_excess_loss
from ith_python.bull_ith_numba import bull_excess_gain_excess_loss


def generate_validation_png(
    case: dict,
    output_path: Path,
    width: int = 1600,
    height: int = 1200,
) -> None:
    """Generate a validation PNG showing long and short epochs with excess gains/losses curves.

    Creates a 4-row subplot (like the existing results.html pattern):
    - Row 1: NAV with LONG epoch markers (gold)
    - Row 2: Long algorithm excess gains (green) & losses (red) curves
    - Row 3: NAV with SHORT epoch markers (cyan)
    - Row 4: Short algorithm excess gains (green) & losses (red) curves

    Args:
        case: Edge case dictionary with 'nav', 'name', 'description', etc.
        output_path: Path to write the PNG file
        width: Image width in pixels
        height: Image height in pixels
    """
    nav = case["nav"]
    tmaeg = case.get("tmaeg", 0.05)
    tmaer = case.get("tmaer", 0.05)

    # Run both algorithms
    long_result = bull_excess_gain_excess_loss(nav, tmaeg)
    short_result = bear_excess_gain_excess_loss(nav, tmaer)

    long_epochs = list(np.where(long_result.bull_epochs)[0])
    short_epochs = list(np.where(short_result.bear_epochs)[0])

    # Create 4-row subplots
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=[
            f"LONG: NAV with Epochs ({len(long_epochs)}) - Expected: {case.get('expected_bull_epochs', [])}",
            f"LONG: Excess Gains & Losses (TMAEG={tmaeg})",
            f"SHORT: NAV with Epochs ({len(short_epochs)}) - Expected: {case.get('expected_bear_epochs', [])}",
            f"SHORT: Excess Gains & Losses (TMAER={tmaer})",
        ],
        vertical_spacing=0.08,
        row_heights=[0.25, 0.25, 0.25, 0.25],
    )

    x = list(range(len(nav)))

    # ====== LONG ALGORITHM (Rows 1-2) ======

    # Row 1: NAV with long epoch markers
    fig.add_trace(
        go.Scatter(
            x=x,
            y=nav,
            name="NAV",
            line=dict(color="white", width=1.5),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    if long_epochs:
        fig.add_trace(
            go.Scatter(
                x=long_epochs,
                y=nav[long_epochs],
                mode="markers",
                marker=dict(color="gold", size=14, symbol="circle"),
                name="Long Epochs",
            ),
            row=1,
            col=1,
        )

    # Row 2: Long excess gains/losses curves (full time series)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=long_result.excess_gains,
            name="Long Excess Gains",
            line=dict(color="limegreen", width=2),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=long_result.excess_losses,
            name="Long Excess Losses",
            line=dict(color="red", width=2),
        ),
        row=2,
        col=1,
    )
    # Add threshold line
    fig.add_hline(y=tmaeg, line_dash="dash", line_color="yellow",
                  annotation_text=f"TMAEG={tmaeg}", row=2, col=1)
    # Mark epochs on gains curve
    if long_epochs:
        fig.add_trace(
            go.Scatter(
                x=long_epochs,
                y=long_result.excess_gains[long_epochs],
                mode="markers",
                marker=dict(color="gold", size=10, symbol="circle"),
                name="Long Epoch (on gains)",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # ====== SHORT ALGORITHM (Rows 3-4) ======

    # Row 3: NAV with short epoch markers
    fig.add_trace(
        go.Scatter(
            x=x,
            y=nav,
            name="NAV",
            line=dict(color="white", width=1.5),
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    if short_epochs:
        fig.add_trace(
            go.Scatter(
                x=short_epochs,
                y=nav[short_epochs],
                mode="markers",
                marker=dict(color="cyan", size=14, symbol="circle"),
                name="Short Epochs",
            ),
            row=3,
            col=1,
        )

    # Row 4: Short excess gains/losses curves (full time series)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=short_result.excess_gains,
            name="Short Excess Gains",
            line=dict(color="limegreen", width=2),
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=short_result.excess_losses,
            name="Short Excess Losses",
            line=dict(color="red", width=2),
        ),
        row=4,
        col=1,
    )
    # Add threshold line
    fig.add_hline(y=tmaer, line_dash="dash", line_color="yellow",
                  annotation_text=f"TMAER={tmaer}", row=4, col=1)
    # Mark epochs on gains curve
    if short_epochs:
        fig.add_trace(
            go.Scatter(
                x=short_epochs,
                y=short_result.excess_gains[short_epochs],
                mode="markers",
                marker=dict(color="cyan", size=10, symbol="circle"),
                name="Short Epoch (on gains)",
                showlegend=False,
            ),
            row=4,
            col=1,
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Edge Case: {case['name']}<br><sub>{case.get('description', '')}</sub>",
            font=dict(size=18),
        ),
        paper_bgcolor="DarkSlateGrey",
        plot_bgcolor="Black",
        font=dict(family="Monospace", color="White"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        width=width,
        height=height,
    )

    # Update axes
    fig.update_xaxes(title_text="", gridcolor="gray", row=1, col=1)
    fig.update_xaxes(title_text="", gridcolor="gray", row=2, col=1)
    fig.update_xaxes(title_text="", gridcolor="gray", row=3, col=1)
    fig.update_xaxes(title_text="Day Index", gridcolor="gray", row=4, col=1)
    fig.update_yaxes(title_text="NAV", gridcolor="gray", row=1, col=1)
    fig.update_yaxes(title_text="Excess G/L", gridcolor="gray", row=2, col=1)
    fig.update_yaxes(title_text="NAV", gridcolor="gray", row=3, col=1)
    fig.update_yaxes(title_text="Excess G/L", gridcolor="gray", row=4, col=1)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path))
    print(f"Generated: {output_path}")


def generate_all_validation_pngs(output_dir: Path) -> list[Path]:
    """Generate validation PNGs for all edge cases.

    Args:
        output_dir: Directory to write PNG files

    Returns:
        List of generated file paths
    """
    from fixtures.edge_cases import ALL_EDGE_CASES

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    for case in ALL_EDGE_CASES:
        output_path = output_dir / f"edge_case_{case['name']}.png"
        generate_validation_png(case, output_path)
        generated_files.append(output_path)

    return generated_files


def main():
    """CLI entry point for validation PNG generation."""
    parser = argparse.ArgumentParser(
        description="Generate validation PNGs for Bear ITH edge cases"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("artifacts/validation"),
        help="Output directory for PNG files",
    )
    parser.add_argument(
        "--case",
        "-c",
        type=str,
        help="Generate only for specific case name",
    )

    args = parser.parse_args()

    from fixtures.edge_cases import ALL_EDGE_CASES

    if args.case:
        # Generate for specific case
        for case in ALL_EDGE_CASES:
            if case["name"] == args.case:
                output_path = args.output / f"edge_case_{case['name']}.png"
                generate_validation_png(case, output_path)
                return
        print(f"Case '{args.case}' not found")
        sys.exit(1)
    else:
        # Generate all
        generated = generate_all_validation_pngs(args.output)
        print(f"\nGenerated {len(generated)} validation PNGs in {args.output}")
