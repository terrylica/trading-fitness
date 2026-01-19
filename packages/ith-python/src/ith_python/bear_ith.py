"""
Bear ITH (Investment Time Horizon) Fitness Analysis Script

This module provides bear market (short position) ITH analysis,
mirroring the bull ITH analysis for short position profitability.

HOW TO ANALYZE YOUR OWN TRADING DATA:
====================================

1. PREPARE YOUR CSV FILE:
   - Required columns: "Date", "NAV"
   - Optional column: "PnL" (will be calculated if missing)
   - Date format: Any standard format (YYYY-MM-DD recommended)
   - NAV: Net Asset Value or cumulative returns starting from 1.0

2. DROP YOUR FILE:
   - Run this script once to create the folder structure
   - Place your CSV file in: data/nav_data_custom/
   - Multiple CSV files are supported

3. RUN THE ANALYSIS:
   - Execute: python -m ith_python.bear_ith
   - Results saved as interactive HTML charts
   - Summary report opens automatically in your browser

4. INTERPRET RESULTS:
   - Bear epochs: Points where SHORT positions would have profited
   - TMAER: Target Maximum Acceptable Excess Runup (adverse for shorts)
   - Green "Qualified" count: Datasets meeting Bear ITH fitness criteria
"""

from __future__ import annotations

import glob
import os
import shutil
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from loguru import logger
from numba import njit
from plotly.graph_objs.layout import XAxis, YAxis
from plotly.subplots import make_subplots
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.traceback import install
from scipy import stats

from ith_python.bear_ith_numba import bear_excess_gain_excess_loss
from ith_python.paths import (
    ensure_dirs,
    get_artifacts_dir,
    get_custom_nav_dir,
    get_log_dir,
    get_synth_bear_ithes_dir,
)

# === Module initialization ===

# Install rich traceback for better error display
install(show_locals=True)

# Initialize console for critical user messages only
console = Console()

# Initialize project directories using repository-local paths
APP_NAME = "bear-ith-fitness"

# Get repository-local directories
data_dir = get_artifacts_dir()
logs_dir = get_log_dir()

# Create directories idempotently
ensure_dirs()

# Show user where data will be stored
console.print("[bold cyan]Bear ITH Project directories initialized:[/bold cyan]")
console.print(f"   Artifacts: {data_dir}")
console.print(f"   Logs: {logs_dir}")

# Configure Loguru - set appropriate log levels
logger.remove()

# Add console handler with WARNING level (minimal console output)
logger.add(
    sys.stderr,
    level="WARNING",
    format="<level>{level}</level>: <level>{message}</level>",
)

# Add file handler with DEBUG level for detailed logging
logger.add(
    logs_dir / f"bear_ith_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level="DEBUG",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
)


# === Configuration Classes ===


class BearIthConfig(NamedTuple):
    """Configuration for Bear ITH (Short Position) analysis."""

    delete_everything: bool = False
    output_dir: Path = get_synth_bear_ithes_dir()
    TMAER_dynamically_determined_by: str = "mru"  # max runup
    TMAER: float = 0.05
    date_initiate: str = "2020-01-30"
    date_conclude: str = "2023-07-25"
    bear_epochs_lower_bound: int = 10
    bear_epochs_upper_bound: int = 100000
    sr_lower_bound: float = -9.9  # Negative for bear markets
    sr_upper_bound: float = -0.5  # Negative for bear markets
    aggcv_low_bound: float = 0
    aggcv_up_bound: float = 0.70
    qualified_results: int = 0
    required_qualified_results: int = 15


class BearSyntheticNavParams(NamedTuple):
    """Parameters for generating synthetic bear market NAV data.

    Uses stronger negative drift than bull's positive drift because
    multiplicative returns require larger drift to overcome volatility.
    """

    start_date: str = "2020-01-30"
    end_date: str = "2023-07-25"
    # Stronger negative drift for reliable bear market (-0.1% daily)
    avg_daily_return: float = -0.001
    # Slightly lower volatility to ensure consistent decline
    daily_return_volatility: float = 0.008
    # Same df as bull (5) for symmetric distribution
    df: int = 5
    # Same probability as bull's drawdown_prob for symmetric behavior
    rally_prob: float = 0.05
    rally_magnitude_low: float = 0.001
    rally_magnitude_high: float = 0.003
    rally_recovery_prob: float = 0.05


class ProcessingParams(NamedTuple):
    trading_year_days_crypto: int = 365
    trading_year_days_other: int = 252
    retries: int = 20
    delay: int = 3


class PlotConfig(NamedTuple):
    margin: dict = {"l": 50, "r": 50, "b": 50, "t": 50, "pad": 4}
    paper_bgcolor: str = "DarkSlateGrey"
    plot_bgcolor: str = "Black"
    legend_font_family: str = "Courier New"
    legend_font_size: int = 12
    legend_font_color: str = "White"
    legend_bgcolor: str = "DarkSlateGrey"
    legend_bordercolor: str = "White"
    legend_borderwidth: int = 2
    global_font_family: str = "Monospace"
    global_font_size: int = 12
    global_font_color: str = "White"
    annotation_font_size: int = 16
    annotation_font_color: str = "White"
    gridcolor: str = "dimgray"
    xaxis_tickmode: str = "array"
    xaxis_gridwidth: float = 0.5
    yaxis_gridwidth: float = 0.5
    auto_open_index_html: bool = True
    auto_open_nav_html: bool = False
    auto_open_live_view: bool = False


# Use the configuration
config = BearIthConfig()
synthetic_nav_params = BearSyntheticNavParams()
processing_params = ProcessingParams()
plot_config = PlotConfig()


# ===== HELPER FUNCTIONS =====


def setup_directories(config: BearIthConfig) -> tuple[Path, Path]:
    """Setup output directories with consistent logic."""
    if config.delete_everything:
        if config.output_dir.exists():
            shutil.rmtree(config.output_dir)
            console.print(
                "🚀 [bold red]Sayonara, files![/bold red] Your files have left the chat"
            )
            logger.warning(f"Deleted directory: {config.output_dir}")
        else:
            console.print("🤔 [yellow]Hmm, nothing to delete here.[/yellow]")
            logger.info("No existing directory to delete")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    nav_dir = config.output_dir / "nav_data_synthetic"
    nav_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directories: {config.output_dir} and {nav_dir}")
    return config.output_dir, nav_dir


def create_progress_bar(console: Console) -> Progress:
    """Create a standardized progress bar configuration."""
    return Progress(
        SpinnerColumn(),
        TimeElapsedColumn(),
        BarColumn(bar_width=25),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn(
            "[red]Disqualified:{task.fields[disqualified]}[/red] [green]Qualified:{task.fields[qualified]}[/green]"
        ),
        console=console,
        transient=False,
        refresh_per_second=4,
    )


def load_and_validate_csv(csv_file: Path) -> pd.DataFrame:
    """Load and validate CSV file with consistent error handling."""
    try:
        nav_data = pd.read_csv(csv_file, index_col="Date", parse_dates=True)

        if nav_data.empty:
            raise ValueError(f"Empty CSV file: {csv_file}")

        if "NAV" not in nav_data.columns:
            raise ValueError(f"No NAV column in {csv_file}")

        if "PnL" not in nav_data.columns:
            logger.info(f"Calculating PnL for {csv_file}")
            nav_data = pnl_from_nav(nav_data).nav_data

        return nav_data

    except pd.errors.EmptyDataError:
        raise ValueError(f"Empty or corrupt CSV file: {csv_file}")
    except Exception as e:
        raise ValueError(f"Error processing {csv_file}: {e}") from e


def print_processing_summary(title: str, total: int, processed: int, disqualified: int):
    """Print processing summary with consistent formatting."""
    if total > 0:
        console.print(f"📈 [bold blue]{title}[/bold blue]")
        console.print(
            f"   Total: {total} | Processed: {processed} | Disqualified: {disqualified}"
        )
    logger.info(
        f"{title}: Total={total}, Processed={processed}, Disqualified={disqualified}"
    )


# Setup directories using helper function
output_dir, nav_dir = setup_directories(config)

# Set constants
date_duration = (
    pd.to_datetime(config.date_conclude) - pd.to_datetime(config.date_initiate)
).days
bear_epochs_lower_bound = int(np.floor(date_duration / 28 / 6))
logger.debug(f"{bear_epochs_lower_bound=}")
bear_epochs_upper_bound = config.bear_epochs_upper_bound
sr_lower_bound = config.sr_lower_bound
sr_upper_bound = config.sr_upper_bound
aggcv_low_bound = config.aggcv_low_bound
aggcv_up_bound = config.aggcv_up_bound
qualified_results = config.qualified_results


def generate_synthetic_bear_nav(params: BearSyntheticNavParams):
    """Generate synthetic NAV data with bear market characteristics.

    Uses MULTIPLICATIVE returns (cumprod) to guarantee NAV stays positive.
    This mirrors the bull generator but with negative drift.
    """
    dates = pd.date_range(params.start_date, params.end_date)

    # Generate daily returns using t-distribution
    daily_returns = stats.t.rvs(
        params.df,
        loc=params.avg_daily_return,
        scale=params.daily_return_volatility,
        size=len(dates),
    )

    # Add dead cat bounces (rallies in bear market)
    rally = False
    for i in range(len(dates)):
        if rally:
            # Rally adds positive return (price goes up temporarily)
            daily_returns[i] += np.random.uniform(
                params.rally_magnitude_low, params.rally_magnitude_high
            )
            if np.random.rand() < params.rally_recovery_prob:
                rally = False
        elif np.random.rand() < params.rally_prob:
            rally = True

    # Use MULTIPLICATIVE returns: NAV = cumprod(1 + returns)
    # This guarantees NAV stays positive (unlike additive cumsum)
    # Clamp returns to prevent NAV going negative (returns > -100%)
    daily_returns = np.clip(daily_returns, -0.99, None)
    walk = np.cumprod(1 + daily_returns)

    nav = pd.DataFrame(data=walk, index=dates, columns=["NAV"])
    nav.index.name = "Date"
    nav["PnL"] = nav["NAV"].diff()
    nav["PnL"] = nav["PnL"].fillna(nav["NAV"].iloc[0] - 1)
    return nav


@njit
def _sharpe_ratio_helper(
    returns: np.ndarray, nperiods: float, rf: float = 0.0, annualize: bool = True
) -> float:
    """Numba-accelerated Sharpe ratio calculation."""
    # Filter out NaN values manually (np.isnan not supported in older numba)
    valid_count = 0
    for r in returns:
        if not np.isnan(r):
            valid_count += 1

    if valid_count < 2:
        return np.nan

    # Calculate mean
    total = 0.0
    for r in returns:
        if not np.isnan(r):
            total += r
    mean_returns = total / valid_count

    # Calculate std dev (sample std)
    var_sum = 0.0
    for r in returns:
        if not np.isnan(r):
            var_sum += (r - mean_returns) ** 2
    std_dev = np.sqrt(var_sum / (valid_count - 1))

    if std_dev == 0:
        return np.nan

    mean_diff = mean_returns - rf
    if annualize:
        return np.sqrt(nperiods) * (mean_diff / std_dev)
    return mean_diff / std_dev


def sharpe_ratio_numba(
    returns: np.ndarray,
    granularity: str,
    market_type: str = "crypto",
    rf: float = 0.0,
    annualize: bool = True,
) -> float:
    """Calculate Sharpe ratio with Numba acceleration."""
    trading_year_days = (
        processing_params.trading_year_days_crypto
        if market_type == "crypto"
        else processing_params.trading_year_days_other
    )
    if "d" in granularity:
        nperiods = trading_year_days / int(granularity.replace("d", ""))
    elif "m" in granularity:
        nperiods = (trading_year_days * 24 * 60) / int(granularity.replace("m", ""))
    else:
        raise ValueError(
            "Invalid granularity format. Use '1d', '2d', ..., '1m', '2m', ..."
        )
    return _sharpe_ratio_helper(returns, nperiods, rf, annualize)


class MaxRunupResult(NamedTuple):
    max_runup: float


def max_runup(nav_values) -> MaxRunupResult:
    """Calculate maximum runup (adverse for shorts).

    Uses bounded formula: runup = 1 - (running_min / nav)
    This is symmetric with max_drawdown: drawdown = 1 - (nav / running_max)
    Both formulas are bounded [0, 1).
    """
    running_min = np.minimum.accumulate(nav_values)
    # Bounded formula: runups in [0, 1) - symmetric with max_drawdown
    runups = 1 - running_min / nav_values
    return MaxRunupResult(max_runup=float(np.max(runups)))


class PnLResult(NamedTuple):
    nav_data: pd.DataFrame


def pnl_from_nav(nav_data) -> PnLResult:
    """Calculate PnL from NAV as a fractional percentage."""
    try:
        nav_copy = nav_data.copy()
        nav_copy.loc[:, "PnL"] = nav_copy["NAV"].diff() / nav_copy["NAV"].shift(1)
        nav_copy.loc[nav_copy.index[0], "PnL"] = 0
        return PnLResult(nav_data=nav_copy)
    except Exception as e:
        logger.error(f"Error calculating PnL: {e}")
        raise


def save_files(fig, filename, output_dir, nav_data, uid, source_file=None):
    """Save HTML and CSV files with optional source file name in output filename."""
    try:
        output_dir = Path(output_dir).resolve()
        if not output_dir.exists():
            logger.warning(f"Creating missing output directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)

        if source_file:
            source_name = Path(source_file).stem
            filename = filename.replace(".html", f"_{source_name}.html")

        filename = filename.replace("_nav_data_synthetic.html", ".html")

        html_path = output_dir / filename
        csv_path = output_dir / filename.replace(".html", ".csv")

        logger.debug(f"Saving files to: HTML={html_path}, CSV={csv_path}")

        if not os.access(str(output_dir), os.W_OK):
            logger.error(f"No write permission for directory: {output_dir}")
            return False

        try:
            fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=False)
        except Exception as e:  # noqa: BLE001 - logs and returns False
            logger.error(f"Error saving HTML file: {e}")
            return False

        try:
            nav_data.to_csv(str(csv_path))
        except Exception as e:  # noqa: BLE001 - logs and returns False
            logger.error(f"Error saving CSV file: {e}")
            return False

        if not html_path.exists() or html_path.stat().st_size == 0:
            logger.error(f"HTML file not created or empty: {html_path}")
            return False

        if not csv_path.exists() or csv_path.stat().st_size == 0:
            logger.error(f"CSV file not created or empty: {csv_path}")
            return False

        logger.success(f"Successfully saved files to {output_dir}")
        return True

    except Exception as e:  # noqa: BLE001 - outer catch-all, logs and returns False
        logger.error(f"Error in save_files: {e}")
        return False


def get_first_non_zero_digits(num, digit_count):
    """Get first non-zero digits from a number for UID generation."""
    non_zero_digits = "".join([i for i in str(num) if i not in ["0", ".", "-"]][:12])
    uid = (non_zero_digits + "0" * digit_count)[:digit_count]
    return uid


class ProcessNavDataResult(NamedTuple):
    qualified_results: int
    sharpe_ratio: float
    num_of_bear_epochs: int
    filename: str
    uid: str
    fig: go.Figure
    recent_3_max_days: float


def determine_tmaer(nav_data, method):
    """Determine TMAER (Target Maximum Acceptable Excess Runup)."""
    if method == "mru":
        return max_runup(nav_data["NAV"].values).max_runup
    elif method == "fixed":
        return config.TMAER
    return None


class PreflightResult(NamedTuple):
    """Result of preflight direction validation."""

    trend_direction: str  # "bullish", "bearish", or "neutral"
    cumulative_return: float  # Total return from start to end
    up_days_pct: float  # Percentage of up days
    suitable_for_bear: bool  # Whether data is suitable for bear analysis
    warning_message: str | None  # Warning if direction mismatch


def preflight_direction_check(nav_data: pd.DataFrame) -> PreflightResult:
    """Validate NAV trend direction before running bear analysis.

    Bear ITH analysis is designed for DECLINING NAV (bear markets).
    Running bear analysis on rising NAV will produce 0 epochs.

    Args:
        nav_data: DataFrame with 'NAV' column

    Returns:
        PreflightResult with trend analysis and suitability assessment
    """
    nav = nav_data["NAV"].values
    pnl = np.diff(nav)

    # Calculate metrics
    cumulative_return = (nav[-1] / nav[0]) - 1 if nav[0] != 0 else 0
    up_days = np.sum(pnl > 0)
    total_days = len(pnl)
    up_days_pct = up_days / total_days if total_days > 0 else 0.5

    # Classify trend
    if cumulative_return > 0.02 and up_days_pct > 0.55:
        trend = "bullish"
        suitable = False
        warning = (
            f"WARNING: Bullish trend detected (return={cumulative_return:.1%}, "
            f"up_days={up_days_pct:.1%}). Bear analysis expects declining NAV. "
            "Expect 0 bear epochs."
        )
    elif cumulative_return < -0.02 and up_days_pct < 0.45:
        trend = "bearish"
        suitable = True
        warning = None
    else:
        trend = "neutral"
        suitable = True  # Neutral can still have some bear epochs
        warning = (
            f"Neutral/choppy market detected (return={cumulative_return:.1%}). "
            "Bear epochs may be minimal."
        )

    return PreflightResult(
        trend_direction=trend,
        cumulative_return=cumulative_return,
        up_days_pct=up_days_pct,
        suitable_for_bear=suitable,
        warning_message=warning,
    )


def bear_excess_gain_excess_loss_wrapper(hurdle, nav):
    """Wrapper for bear epoch calculation that handles DataFrame input."""
    original_df = (
        nav.copy() if isinstance(nav, pd.DataFrame) and "NAV" in nav.columns else None
    )
    nav_values = nav["NAV"].values if original_df is not None else nav.values
    result = bear_excess_gain_excess_loss(nav_values, hurdle)
    if original_df is not None:
        original_df["Excess Gains"] = result.excess_gains
        original_df["Excess Losses"] = result.excess_losses
        original_df["BearEpochs"] = result.bear_epochs
        return original_df
    else:
        return result


# Initialize a list to store results
results = []


def process_nav_data(
    nav_data,
    output_dir,
    qualified_results,
    nav_dir,
    TMAER,
    bypass_thresholds=False,
    data_source="Synthetic",
) -> ProcessNavDataResult:
    """Process NAV data and generate bear epoch analysis."""

    uid_part1 = get_first_non_zero_digits(nav_data["NAV"].iloc[0], 6)
    uid_part2 = get_first_non_zero_digits(nav_data["NAV"].iloc[1], 6)
    uid = uid_part1 + uid_part2
    logger.debug(f"{uid=}")

    filename = None
    bear_durations = None
    excess_losses_at_bear_epochs = None
    fig = None

    days_elapsed = (nav_data.index[-1] - nav_data.index[0]).days

    sharpe_ratio = sharpe_ratio_numba(nav_data["PnL"].dropna().values, "1d")
    calculated_nav = bear_excess_gain_excess_loss_wrapper(TMAER, nav_data)

    if isinstance(calculated_nav, pd.DataFrame):
        bear_epochs_idx = calculated_nav[calculated_nav["BearEpochs"]].index
        num_of_bear_epochs = len(bear_epochs_idx)
    else:
        bear_epochs_idx = nav_data.index[calculated_nav.bear_epochs]
        num_of_bear_epochs = calculated_nav.num_of_bear_epochs

    logger.debug(f"Threshold check details: bypass_thresholds={bypass_thresholds}")
    logger.debug(f"SR check: {sr_lower_bound} < {sharpe_ratio} < {sr_upper_bound}")
    logger.debug(
        f"Bear epochs check: {bear_epochs_lower_bound} < {num_of_bear_epochs} < {bear_epochs_upper_bound}"
    )

    # For bear markets, we check for negative Sharpe ratios OR bypass
    sr_condition = bypass_thresholds or (
        sr_lower_bound < sharpe_ratio < sr_upper_bound
    ) or (sharpe_ratio < 0)  # Bear markets often have negative SR

    if sr_condition and (
        bypass_thresholds
        or bear_epochs_lower_bound < num_of_bear_epochs < bear_epochs_upper_bound
    ):
        if bypass_thresholds:
            logger.debug("Bypassing thresholds for custom CSV.")
        else:
            logger.debug("Thresholds met for synthetic data.")

        logger.debug(f"Found {num_of_bear_epochs=}, {sharpe_ratio=}")

        bear_epoch_dates = calculated_nav[calculated_nav["BearEpochs"]].index

        timeline_dates = bear_epoch_dates.insert(0, calculated_nav.index[0])
        if not calculated_nav["BearEpochs"].iloc[-1]:
            timeline_dates = timeline_dates.append(pd.Index([calculated_nav.index[-1]]))

        logger.debug(f"timeline_dates: {timeline_dates}")

        bear_epoch_ct = len(bear_epoch_dates)

        timeline_indices = [
            calculated_nav.index.get_loc(date) for date in timeline_dates
        ]
        bear_durations_indices = np.diff(timeline_indices)

        date_diffs = [
            (timeline_dates[i + 1] - timeline_dates[i]).days
            for i in range(len(timeline_dates) - 1)
        ]
        bear_durations_days = np.array(date_diffs)

        logger.debug(f"bear_durations_indices: {bear_durations_indices}")
        logger.debug(f"bear_durations_days: {bear_durations_days}")

        days_taken_to_bear_epoch = days_elapsed / bear_epoch_ct if bear_epoch_ct > 0 else np.nan

        recent_3_max_days = (
            np.max(bear_durations_days[-3:])
            if len(bear_durations_days) >= 3
            else np.max(bear_durations_days) if len(bear_durations_days) > 0 else np.nan
        )

        bear_cv = (
            np.std(bear_durations_indices) / np.mean(bear_durations_indices)
            if len(bear_durations_indices) > 0
            else np.nan
        )

        excess_losses_at_bear_epochs = calculated_nav[calculated_nav["BearEpochs"]][
            "Excess Losses"
        ]
        excess_losses_at_bear_epochs = excess_losses_at_bear_epochs[
            excess_losses_at_bear_epochs != 0
        ]
        last_excess_loss = calculated_nav["Excess Losses"].iloc[-1]
        if not calculated_nav["BearEpochs"].iloc[-1]:
            excess_losses_at_bear_epochs = pd.concat(
                [
                    excess_losses_at_bear_epochs,
                    pd.Series([last_excess_loss], index=[calculated_nav.index[-1]]),
                ]
            )
        if excess_losses_at_bear_epochs.empty:
            el_cv = np.nan
        else:
            el_cv = np.std(excess_losses_at_bear_epochs) / np.mean(excess_losses_at_bear_epochs)

        aggcv = max(el_cv, bear_cv) if not np.isnan(el_cv) and not np.isnan(bear_cv) else np.nan
        logger.debug(f"{aggcv=}")

        logger.debug(f"AGGCV check: {aggcv_low_bound} < {aggcv} < {aggcv_up_bound}")

        if bypass_thresholds or (not np.isnan(aggcv) and aggcv_low_bound < aggcv < aggcv_up_bound):
            if bypass_thresholds:
                source_name = Path(nav_dir).stem if nav_dir else "unknown"
                filename = (
                    f"bear_nav_"
                    f"EL_{el_cv:.5f}_"
                    f"BearCV_{bear_cv:.5f}_"
                    f"TMAER_{TMAER:.5f}_"
                    f"BearEpochs_{bear_epoch_ct}_"
                    f"D2BE_{days_taken_to_bear_epoch:.2f}_"
                    f"SR_{sharpe_ratio:.4f}_"
                    f"UID_{uid}.html"
                )
            else:
                filename = (
                    f"EL_{el_cv:.5f}_"
                    f"BearCV_{bear_cv:.5f}_"
                    f"TMAER_{TMAER:.5f}_"
                    f"BearEpochs_{bear_epoch_ct}_"
                    f"D2BE_{days_taken_to_bear_epoch:.2f}_"
                    f"SR_{sharpe_ratio:.4f}_"
                    f"UID_{uid}.html"
                )

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.01,
                subplot_titles=("NAV", "Excess Gains & Losses"),
            )

            bear_epoch_indices = calculated_nav[calculated_nav["BearEpochs"]].index
            num_of_bear_epochs_local = len(bear_epoch_indices)
            bear_epochs_dir = output_dir / f"Bear_Epochs_{num_of_bear_epochs_local}"
            bear_epochs_dir.mkdir(parents=True, exist_ok=True)
            crossover_epochs = calculated_nav.loc[bear_epoch_indices]

            fig.add_trace(
                go.Scatter(
                    x=crossover_epochs.index,
                    y=crossover_epochs["NAV"],
                    mode="markers",
                    name="Bear Epochs on NAV",
                    marker=dict(color="cyan", size=20),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=crossover_epochs.index,
                    y=crossover_epochs["Excess Gains"],
                    mode="markers",
                    name="Bear Epochs on Excess Gains",
                    marker=dict(color="magenta", size=20),
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=calculated_nav.index,
                    y=calculated_nav["NAV"],
                    mode="lines",
                    name="NAV",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=calculated_nav.index,
                    y=calculated_nav["Excess Gains"],
                    mode="lines",
                    name="Excess Gains",
                    line=dict(color="green"),
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=calculated_nav.index,
                    y=calculated_nav["Excess Losses"],
                    mode="lines",
                    name="Excess Losses",
                    line=dict(color="red"),
                ),
                row=2,
                col=1,
            )

            fig.update_layout(
                title=f"{num_of_bear_epochs_local} Bear Epochs -- {filename}",
                autosize=True,
                margin=plot_config.margin,
                paper_bgcolor=plot_config.paper_bgcolor,
                plot_bgcolor=plot_config.plot_bgcolor,
                legend=dict(
                    x=0.01,
                    y=0.98,
                    traceorder="normal",
                    font=dict(
                        family=plot_config.legend_font_family,
                        size=plot_config.legend_font_size,
                        color=plot_config.legend_font_color,
                    ),
                    bgcolor=plot_config.legend_bgcolor,
                    bordercolor=plot_config.legend_bordercolor,
                    borderwidth=plot_config.legend_borderwidth,
                ),
                font=dict(
                    family=plot_config.global_font_family,
                    size=plot_config.global_font_size,
                    color=plot_config.global_font_color,
                ),
                annotations=[
                    dict(
                        x=0.5,
                        y=0.95,
                        xref="paper",
                        yref="paper",
                        text="NAV<br>",
                        showarrow=False,
                        font=dict(
                            size=plot_config.annotation_font_size,
                            color=plot_config.annotation_font_color,
                        ),
                    ),
                    dict(
                        x=0.5,
                        y=0.45,
                        xref="paper",
                        yref="paper",
                        text="Excess Gains & Losses<br>",
                        showarrow=False,
                        font=dict(
                            size=plot_config.annotation_font_size,
                            color=plot_config.annotation_font_color,
                        ),
                    ),
                ],
            )

            fig.update_yaxes(
                gridcolor=plot_config.gridcolor, type="linear", row=1, col=1
            )
            fig.update_yaxes(gridcolor=plot_config.gridcolor, row=2, col=1)
            fig.update_xaxes(gridcolor=plot_config.gridcolor, row=1, col=1)
            fig.update_xaxes(gridcolor=plot_config.gridcolor, row=2, col=1)

            monthly_ticks = pd.date_range(
                nav_data.index.min(), nav_data.index.max(), freq="MS"
            )

            custom_xaxis = XAxis(
                tickmode=plot_config.xaxis_tickmode,
                tickvals=monthly_ticks,
                showgrid=True,
                gridwidth=plot_config.xaxis_gridwidth,
            )

            custom_yaxis = YAxis(
                showgrid=True,
                gridwidth=plot_config.yaxis_gridwidth,
            )
            fig.update_layout(xaxis=custom_xaxis.to_plotly_json())
            fig.update_layout(yaxis=custom_yaxis.to_plotly_json())

            if plot_config.auto_open_live_view:
                fig.show()

            save_files(
                fig,
                filename,
                output_dir,
                nav_data,
                uid,
                source_file=None if not bypass_thresholds else nav_dir,
            )

            if not bypass_thresholds:
                qualified_results += 1

            logger.debug(
                f"Generated {TMAER=}, {sharpe_ratio=}, {num_of_bear_epochs=}, {el_cv=},{bear_cv=}, {aggcv=}, {days_taken_to_bear_epoch=}"
            )

        if filename:
            results.append(
                [
                    TMAER,
                    sharpe_ratio,
                    num_of_bear_epochs,
                    el_cv,
                    bear_cv,
                    aggcv,
                    days_taken_to_bear_epoch,
                    recent_3_max_days,
                    filename,
                    data_source,
                ]
            )

        if plot_config.auto_open_nav_html and filename:
            file_path = str(output_dir / filename)
            webbrowser.open(f"file://{os.path.abspath(file_path)}")

    else:
        pass

    if bear_durations is not None:
        logger.debug(f"bear_durations in process_nav_data: {bear_durations}")

    logger.debug(
        f"excess_losses_at_bear_epochs in process_nav_data: {excess_losses_at_bear_epochs}"
    )

    return ProcessNavDataResult(
        qualified_results=qualified_results,
        sharpe_ratio=sharpe_ratio,
        num_of_bear_epochs=num_of_bear_epochs,
        filename=filename,
        uid=uid,
        fig=fig,
        recent_3_max_days=(
            recent_3_max_days if "recent_3_max_days" in locals() else np.nan
        ),
    )


def log_results(results):
    """Generate HTML results table for bear analysis."""
    headers = [
        "TMAER",
        "Sharpe<br/>Ratio",
        "Bear<br/>Epochs<br/>Count",
        "EL CV",
        "Bear CV",
        "AGG CV",
        "Avg Days<br/>Per Epoch",
        "Recent 3<br/>Max Days",
        "Filename",
        "Data Source",
    ]
    df = pd.DataFrame(results, columns=headers)

    df = df[df["Filename"].notna()]

    df["AGG CV Rank"] = df["AGG CV"].rank(method="min").astype(int)
    df["Sharpe<br/>Ratio Rank"] = (
        df["Sharpe<br/>Ratio"].rank(method="min", ascending=True).astype(int)
    )  # Ascending for bear (more negative = better for shorts)
    df["Recent 3<br/>Max Days Rank"] = (
        df["Recent 3<br/>Max Days"].rank(method="min", ascending=True).astype(int)
    )

    df["Link"] = df["Filename"].apply(
        lambda x: f'<a href="synth_bear_ithes/{x}" target="_blank">Open</a>'
    )

    columns = [
        col for col in df.columns if col not in ["Filename", "Data Source", "Link"]
    ]
    columns.extend(["Data Source", "Link"])
    df = df[columns]

    recent_3_rank_col_index = df.columns.get_loc("Recent 3<br/>Max Days Rank")

    html_table = df.to_html(
        classes="table table-striped",
        index=False,
        table_id="results_table",
        escape=False,
    )

    html_output = f"""
    <html>
    <head>
    <title>Bear ITH Fitness Analysis Results</title>
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.css">
    <link rel="stylesheet" href="https://classless.de/classless.css">
    <style>
        body {{ font-family: Osaka, 'SF Mono', Monaco, Menlo, Consolas, 'Cascadia Code', 'JetBrains Mono', 'Fira Code', 'Source Code Pro', 'Liberation Mono', 'DejaVu Sans Mono', 'Courier New', monospace !important; margin: 0 !important; padding: 20px !important; }}
        .container {{ max-width: 100% !important; margin: 0 auto !important; padding: 0 15px !important; }}
        table {{ width: 100% !important; border-collapse: collapse !important; margin: 0 auto !important; font-family: inherit !important; }}
        th, td {{ padding: 8px 12px !important; border-bottom: 1px solid #ddd !important; font-family: inherit !important; }}
        th {{ background-color: #2a4a5a !important; font-weight: 600 !important; text-align: right !important; line-height: 1.2 !important; color: #fff !important; }}
        td {{ text-align: right !important; }}
        th:nth-last-child(2), td:nth-last-child(2) {{ text-align: left !important; }}
        th:nth-last-child(1), td:nth-last-child(1) {{ text-align: left !important; }}
        th {{ vertical-align: middle !important; }}
        .dataTables_wrapper {{ width: 100% !important; font-family: inherit !important; }}
        h1 {{ font-family: inherit !important; color: #2a4a5a !important; }}
    </style>
    <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.5.1.js"></script>
    <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.js"></script>
    <script>
    $(document).ready(function() {{
        $('#results_table').DataTable({{
            "pageLength": 200,
            "lengthMenu": [[200, 500, 1000, -1], [200, 500, 1000, "All"]],
            "order": [[ {recent_3_rank_col_index}, "asc" ]]
        }});
    }});
    </script>
    </head>
    <body>
    <div class="container">
        <h1>Bear ITH Fitness Analysis Results</h1>
        {html_table}
    </div>
    </body>
    </html>
    """

    html_file_path = get_artifacts_dir() / "bear_results.html"
    with open(html_file_path, "w") as file:
        file.write(html_output)

    logger.success("Results have been written to bear_results.html")
    console.print(f"📊 [bold green]Results saved to:[/bold green] {html_file_path}")

    if plot_config.auto_open_index_html:
        webbrowser.open(f"file://{html_file_path.resolve()}")
        console.print("🌐 [blue]Opening results in browser...[/blue]")


def process_csv_batch(
    csv_files: list[Path],
    output_dir: Path,
    nav_dir: Path,
    config: BearIthConfig,
    bypass_thresholds: bool = False,
    data_source: str = "Synthetic",
) -> tuple[int, int, int]:
    """Process a batch of CSV files with consistent logic."""
    processed_files = 0
    disqualified_files = 0
    qualified_results = config.qualified_results

    for csv_file in csv_files:
        try:
            logger.info(f"Processing CSV: {csv_file.name}")
            nav_data = load_and_validate_csv(csv_file)

            TMAER = determine_tmaer(nav_data, config.TMAER_dynamically_determined_by)
            if pd.isna(TMAER):
                logger.error(f"Invalid TMAER calculated for {csv_file}")
                disqualified_files += 1
                continue

            result = process_nav_data(
                nav_data,
                output_dir,
                qualified_results,
                nav_dir,
                TMAER,
                bypass_thresholds,
                data_source,
            )
            qualified_results = result.qualified_results

            if result.filename is not None and result.fig is not None:
                should_save_files = data_source != "Custom"
                if should_save_files:
                    if save_files(
                        result.fig,
                        result.filename,
                        nav_dir,
                        nav_data,
                        result.uid,
                        source_file=csv_file if bypass_thresholds else None,
                    ):
                        processed_files += 1
                    else:
                        disqualified_files += 1
                else:
                    processed_files += 1
            else:
                logger.warning(f"No output generated for {csv_file}")
                disqualified_files += 1

        except Exception as e:  # noqa: BLE001 - logs and continues batch processing
            logger.error(f"Error processing {csv_file}: {e}")
            disqualified_files += 1
            continue

    return processed_files, disqualified_files, qualified_results


# Initialize counters
qualified_results = 0
counter = 0

# Try to load existing data
existing_csv_files = glob.glob(str(nav_dir / "*.csv"))
logger.debug(f"{existing_csv_files=}")

# Process existing CSV files
if existing_csv_files:
    existing_csv_paths = [Path(f) for f in existing_csv_files]
    processed, disqualified, qualified_results = process_csv_batch(
        existing_csv_paths,
        config.output_dir,
        nav_dir,
        config,
        bypass_thresholds=False,
        data_source="Synthetic",
    )
    print_processing_summary(
        "Existing CSV Processing Complete",
        len(existing_csv_files),
        processed,
        disqualified,
    )

# Generate new NAV data if necessary
console.print("🔄 [bold yellow]Starting synthetic bear data generation...[/bold yellow]")
with create_progress_bar(console) as progress:
    task = progress.add_task(
        "Generating synthetic bear data...",
        total=config.required_qualified_results,
        disqualified=0,
        qualified=qualified_results,
    )

    while qualified_results < config.required_qualified_results:
        counter += 1
        synthetic_nav = generate_synthetic_bear_nav(synthetic_nav_params)
        TMAER = determine_tmaer(synthetic_nav, config.TMAER_dynamically_determined_by)
        result = process_nav_data(
            synthetic_nav,
            config.output_dir,
            qualified_results,
            nav_dir,
            TMAER,
            bypass_thresholds=False,
            data_source="Generated",
        )
        qualified_results = result.qualified_results

        if result.filename is not None:
            logger.debug(
                f"Processing synthetic bear data {counter:4}: {result.filename=}, {result.uid=}, {TMAER=}, {result.sharpe_ratio=}, newly generated."
            )
            if result.fig is not None:
                save_files(
                    result.fig, result.filename, nav_dir, synthetic_nav, result.uid
                )
            progress.update(task, advance=1, qualified=qualified_results)
        else:
            progress.update(
                task, disqualified=progress.tasks[0].fields["disqualified"] + 1
            )

        if qualified_results >= config.required_qualified_results:
            console.print(
                f"🎯 [bold green]Target reached![/bold green] Found {config.required_qualified_results} qualified bear results"
            )
            logger.info(
                f"Required number of qualified results ({config.required_qualified_results}) reached."
            )
            break

console.print("")


def get_date_range_from_csvs(csv_folder: Path) -> tuple[str, str]:
    """Get the earliest start date and latest end date from all CSVs in folder."""
    if not csv_folder.exists():
        logger.warning(
            f"Custom CSV folder not found: {csv_folder}\n"
            "The folder will be created automatically."
        )
        return synthetic_nav_params.start_date, synthetic_nav_params.end_date

    csv_files = list(csv_folder.glob("*.csv"))
    if not csv_files:
        logger.info(
            f"No custom CSV files found in {csv_folder} - using default date range"
        )
        return synthetic_nav_params.start_date, synthetic_nav_params.end_date

    all_dates = []
    processed_files = 0
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, parse_dates=["Date"])
            if df.empty:
                logger.warning(f"Empty CSV file: {csv_file}")
                continue
            if "Date" not in df.columns:
                logger.error(f"No Date column in {csv_file}")
                continue
            all_dates.extend(df["Date"].tolist())
            processed_files += 1
        except pd.errors.EmptyDataError:
            logger.error(f"Empty or corrupt CSV file: {csv_file}")
            continue
        except Exception as e:  # noqa: BLE001 - logs and continues
            logger.error(f"Failed to process {csv_file}: {e}")
            continue

    if not all_dates:
        logger.warning("No valid dates found in any CSV files")
        return synthetic_nav_params.start_date, synthetic_nav_params.end_date

    if processed_files < len(csv_files):
        logger.warning(
            f"Only processed {processed_files} out of {len(csv_files)} CSV files"
        )

    min_date = min(all_dates).strftime("%Y-%m-%d")
    max_date = max(all_dates).strftime("%Y-%m-%d")
    logger.info(f"Using date range from custom CSVs: {min_date} to {max_date}")
    return min_date, max_date


custom_csv_folder = get_custom_nav_dir()

console.print(f"📁 [bold cyan]Custom CSV folder ready:[/bold cyan] {custom_csv_folder}")
if not any(custom_csv_folder.glob("*.csv")):
    console.print(
        "💡 [yellow]Tip:[/yellow] Place your trading CSV files (with Date, NAV columns) in the folder above to analyze your own performance data!"
    )

start_date, end_date = get_date_range_from_csvs(custom_csv_folder)

synthetic_nav_params = synthetic_nav_params._replace(
    start_date=start_date, end_date=end_date
)

# Process custom CSV files if they exist
custom_csvs = list(custom_csv_folder.glob("*.csv"))
if custom_csvs:
    console.print(
        f"📈 [bold blue]Processing {len(custom_csvs)} custom CSV file(s) for Bear analysis...[/bold blue]"
    )
    processed_files, disqualified_files, qualified_results = process_csv_batch(
        custom_csvs,
        config.output_dir,
        nav_dir,
        config,
        bypass_thresholds=True,
        data_source="Custom",
    )
    print_processing_summary(
        "Custom CSV Bear Processing Complete",
        len(custom_csvs),
        processed_files,
        disqualified_files,
    )
else:
    console.print(
        "📂 [dim]No custom CSV files to process - continuing with synthetic bear data generation[/dim]"
    )

# Log the results in a tabulated format
log_results(results)
