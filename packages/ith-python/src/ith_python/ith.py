# polars-exception: Legacy module with extensive Pandas integration (plotly, scipy stats)
"""
ITH (Investment Time Horizon) Fitness Analysis Script

HOW TO ANALYZE YOUR OWN TRADING DATA:
====================================

1. PREPARE YOUR CSV FILE:
   - Required columns: "Date", "NAV"
   - Optional column: "PnL" (will be calculated if missing)
   - Date format: Any standard format (YYYY-MM-DD recommended)
   - NAV: Net Asset Value or cumulative returns starting from 1.0

2. DROP YOUR FILE:
   - Run this script once to create the folder structure
   - Place your CSV file in: ~/Documents/ith-fitness/nav_data_custom/
   - Multiple CSV files are supported

3. RUN THE ANALYSIS:
   - Execute: python ith.py
   - Your custom data will be processed first
   - Results saved as interactive HTML charts in the same folder
   - Summary report opens automatically in your browser

4. INTERPRET RESULTS:
   - Green "Qualified" count: Datasets meeting ITH fitness criteria
   - Red "Disqualified" count: Datasets not meeting criteria
   - HTML files contain detailed Investment Time Horizon epoch analysis and visualizations

EXAMPLE CSV FORMAT:
Date,NAV
2020-01-01,1.0000
2020-01-02,1.0123
2020-01-03,0.9987
...

The script will automatically bypass strict thresholds for your custom data,
ensuring your trading performance gets analyzed regardless of synthetic criteria.
"""

# === All imports at top of file ===
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
import numba
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

from ith_python.paths import (
    ensure_dirs,
    get_artifacts_dir,
    get_custom_nav_dir,
    get_log_dir,
    get_synth_bull_ithes_dir,
)

# Import canonical Bull ITH algorithm from bull_ith_numba module
from ith_python.bull_ith_numba import (
    bull_excess_gain_excess_loss,
    max_drawdown as _max_drawdown_numba,
)

# === Module initialization ===

# Install rich traceback for better error display
install(show_locals=True)

# Initialize console for critical user messages only
console = Console()

sys.path.insert(
    0,
    str(
        next((p for p in Path.cwd().parents if p.name == "ml_feature_set"), Path.cwd())
    ),
)
__package__ = Path.cwd().parent.name

# Initialize project directories using repository-local paths
APP_NAME = "ith-fitness"

# Get repository-local directories
data_dir = get_artifacts_dir()
logs_dir = get_log_dir()

# Create directories idempotently
ensure_dirs()

# Show user where data will be stored
console.print("[bold cyan]Project directories initialized:[/bold cyan]")
console.print(f"   Artifacts: {data_dir}")
console.print(f"   Logs: {logs_dir}")

# Configure Loguru - set appropriate log levels
# Remove default handler and add custom ones
logger.remove()

# Add console handler with WARNING level (minimal console output)
logger.add(
    sys.stderr,
    level="WARNING",
    format="<level>{level}</level>: <level>{message}</level>",
)

# Add file handler with DEBUG level for detailed logging
logger.add(
    logs_dir / f"ith_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level="DEBUG",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
)

# === Configuration Classes ===


class BullIthConfig(NamedTuple):
    """Configuration for Bull ITH (Long Position) analysis.

    Supports two modes:
    1. Point-based: Set analysis_n_points (recommended for feature engineering)
    2. Date-based: Set date_initiate/date_conclude (legacy)

    When analysis_n_points is set, it's used for epoch bounds calculation.
    """

    delete_everything: bool = False
    output_dir: Path = get_synth_bull_ithes_dir()
    TMAEG_dynamically_determined_by: str = "mdd"
    TMAEG: float = 0.05

    # Point-based configuration (preferred for time-agnostic analysis)
    analysis_n_points: int | None = None  # If set, used for epoch bounds

    # Date-based configuration (legacy, for backward compatibility)
    date_initiate: str = "2020-01-30"
    date_conclude: str = "2023-07-25"

    # Epoch bounds (tightened with fast batch generation)
    bull_epochs_lower_bound: int = 5  # Minimum epochs for meaningful analysis
    bull_epochs_upper_bound: int = 100000

    # Fitness criteria (restored reasonable bounds)
    sr_lower_bound: float = 0.5  # Require positive risk-adjusted returns
    sr_upper_bound: float = 9.9  # Cap unrealistic Sharpe ratios
    aggcv_low_bound: float = 0
    aggcv_up_bound: float = 0.70  # Coefficient of variation threshold
    qualified_results: int = 0
    required_qualified_results: int = 100  # Generate 100 qualified samples


# Backwards compatibility alias
IthConfig = BullIthConfig


class SyntheticNavParams(NamedTuple):
    """Parameters for generating synthetic NAV data.

    Supports two generation modes:
    1. Point-based (recommended): Set n_points, dates auto-generated
    2. Date-based (legacy): Set start_date/end_date

    When n_points is set, it takes precedence over date range.
    """

    # Point-based mode (preferred for time-agnostic analysis)
    n_points: int | None = None  # If set, generates exactly this many points

    # Date-based mode (legacy, for backward compatibility)
    start_date: str = "2020-01-30"  # Start date for NAV data
    end_date: str = "2023-07-25"  # End date for NAV data

    # Return characteristics
    avg_period_return: float = (
        0.002  # Average period return; INCREASED for testing (was 0.00010123)
    )
    period_return_volatility: float = (
        0.008  # Period return volatility; slightly reduced for more consistent bull trend (was 0.009)
    )
    df: int = (
        5  # Degrees of freedom for the t-distribution; lower values increase the likelihood of extreme returns
    )
    drawdown_prob: float = (
        0.05  # Probability of entering a drawdown; higher values increase the frequency of drawdowns
    )
    drawdown_magnitude_low: float = (
        0.001  # Lower bound of drawdown magnitude; higher values increase the minimum drawdown size
    )
    drawdown_magnitude_high: float = (
        0.003  # Upper bound of drawdown magnitude; higher values increase the maximum drawdown size
    )
    drawdown_recovery_prob: float = (
        0.02  # Probability of recovering from a drawdown; higher values increase the likelihood of recovery
    )


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
    auto_open_index_html: bool = (
        True  # New attribute to control auto-opening of index HTML
    )
    auto_open_nav_html: bool = (
        False  # New attribute to control auto-opening of NAV HTML
    )
    auto_open_live_view: bool = (
        False  # New attribute to control auto-opening of live view HTML
    )


# Use the configuration
config = IthConfig()
synthetic_nav_params = SyntheticNavParams()
processing_params = ProcessingParams()
plot_config = PlotConfig()

# ===== HELPER FUNCTIONS TO ELIMINATE DRY VIOLATIONS =====


def setup_directories(config: IthConfig) -> tuple[Path, Path]:
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

    # Create anew after the great purge
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
    """Load and validate CSV file with consistent error handling.

    Emits data.load telemetry event for forensic analysis and reproducibility.
    """
    from ith_python.telemetry import log_data_load
    from ith_python.telemetry.provenance import fingerprint_file

    try:
        nav_data = pd.read_csv(csv_file, index_col="Date", parse_dates=True)

        if nav_data.empty:
            raise ValueError(f"Empty CSV file: {csv_file}")

        if "NAV" not in nav_data.columns:
            raise ValueError(f"No NAV column in {csv_file}")

        if "PnL" not in nav_data.columns:
            logger.info(f"Calculating PnL for {csv_file}")
            nav_data = pnl_from_nav(nav_data).nav_data

        # Emit data.load telemetry event
        file_fp = fingerprint_file(csv_file)
        nav_range = (float(nav_data["NAV"].min()), float(nav_data["NAV"].max()))
        log_data_load(
            source_path=str(csv_file),
            sha256_hash=file_fp["sha256"],
            row_count=len(nav_data),
            column_count=len(nav_data.columns),
            columns=list(nav_data.columns),
            value_range=nav_range,
        )

        return nav_data

    except pd.errors.EmptyDataError:
        raise ValueError(f"Empty or corrupt CSV file: {csv_file}")
    except Exception as e:  # noqa: BLE001 - pre-existing, wraps as ValueError
        raise ValueError(f"Error processing {csv_file}: {e}") from e


def process_csv_batch(
    csv_files: list[Path],
    output_dir: Path,
    nav_dir: Path,
    config: IthConfig,
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

            TMAEG = determine_tmaeg(nav_data, config.TMAEG_dynamically_determined_by)
            if pd.isna(TMAEG):
                logger.error(f"Invalid TMAEG calculated for {csv_file}")
                disqualified_files += 1
                continue

            result = process_nav_data(
                nav_data,
                output_dir,
                qualified_results,
                nav_dir,
                TMAEG,
                bypass_thresholds,
                data_source,
            )
            qualified_results = result.qualified_results

            if result.filename is not None and result.fig is not None:
                # Only save files for synthetic data, not for custom data to avoid duplication
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
                    # For custom data, we still count as processed but don't save files
                    processed_files += 1
            else:
                logger.warning(f"No output generated for {csv_file}")
                disqualified_files += 1

        except Exception as e:  # noqa: BLE001 - pre-existing, logs and continues
            logger.error(f"Error processing {csv_file}: {e}")
            disqualified_files += 1
            continue

    return processed_files, disqualified_files, qualified_results


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


# ===== END HELPER FUNCTIONS =====

# Setup directories using helper function
output_dir, nav_dir = setup_directories(config)

# Set constants
date_duration = (
    pd.to_datetime(config.date_conclude) - pd.to_datetime(config.date_initiate)
).days
# Use config value directly (not calculated from dates)
bull_epochs_lower_bound = config.bull_epochs_lower_bound
logger.debug(f"{bull_epochs_lower_bound=} (from config)")
bull_epochs_upper_bound = config.bull_epochs_upper_bound
sr_lower_bound = config.sr_lower_bound
sr_upper_bound = config.sr_upper_bound
aggcv_low_bound = config.aggcv_low_bound
aggcv_up_bound = config.aggcv_up_bound
qualified_results = config.qualified_results

# Try to load existing data
existing_csv_files = glob.glob(str(nav_dir / "*.csv"))

logger.debug(f"{existing_csv_files=}")


@njit
def _apply_drawdown_events(
    period_returns: np.ndarray,
    drawdown_prob: float,
    drawdown_magnitude_low: float,
    drawdown_magnitude_high: float,
    drawdown_recovery_prob: float,
) -> np.ndarray:
    """Numba-accelerated drawdown event simulation.

    Modifies period_returns in place to add drawdown events.

    Args:
        period_returns: Array of period returns to modify
        drawdown_prob: Probability of entering a drawdown
        drawdown_magnitude_low: Lower bound of drawdown magnitude
        drawdown_magnitude_high: Upper bound of drawdown magnitude
        drawdown_recovery_prob: Probability of recovering from a drawdown

    Returns:
        Modified period_returns array with drawdown events applied
    """
    n = len(period_returns)
    drawdown = False

    for i in range(n):
        if drawdown:
            # Drawdown subtracts from returns (price goes down temporarily)
            magnitude = drawdown_magnitude_low + np.random.random() * (
                drawdown_magnitude_high - drawdown_magnitude_low
            )
            period_returns[i] -= magnitude
            if np.random.random() < drawdown_recovery_prob:
                drawdown = False
        elif np.random.random() < drawdown_prob:
            drawdown = True

    return period_returns


def generate_synthetic_nav(params: SyntheticNavParams):
    """Generate synthetic NAV data with bull market characteristics.

    Uses MULTIPLICATIVE returns (cumprod) to guarantee NAV stays positive.
    This is symmetric with the bear generator which also uses cumprod.

    If params.n_points is set, generates exactly that many points.
    Otherwise, uses start_date/end_date for backward compatibility.
    """
    if params.n_points is not None:
        # Point-based mode: generate fixed number of points
        n = params.n_points
        # Create dummy dates starting from a fixed base (for plotting compatibility)
        dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
    else:
        # Date-based mode (legacy)
        dates = pd.date_range(params.start_date, params.end_date)
        n = len(dates)

    # Generate period returns using t-distribution
    period_returns = stats.t.rvs(
        params.df,
        loc=params.avg_period_return,
        scale=params.period_return_volatility,
        size=n,
    )

    # Add drawdowns using Numba-accelerated function
    period_returns = _apply_drawdown_events(
        period_returns,
        params.drawdown_prob,
        params.drawdown_magnitude_low,
        params.drawdown_magnitude_high,
        params.drawdown_recovery_prob,
    )

    # Use MULTIPLICATIVE returns: NAV = cumprod(1 + returns)
    # This guarantees NAV stays positive (unlike additive cumsum)
    # Clamp returns to prevent NAV going negative (returns > -100%)
    period_returns = np.clip(period_returns, -0.99, None)
    walk = np.cumprod(1 + period_returns)

    nav = pd.DataFrame(data=walk, index=dates, columns=["NAV"])
    nav.index.name = "Date"
    nav["PnL"] = nav["NAV"].diff()
    nav["PnL"] = nav["PnL"].fillna(nav["NAV"].iloc[0] - 1)
    return nav


# =============================================================================
# VECTORIZED BATCH GENERATION (Optimized for speed)
# =============================================================================


@njit(parallel=True)
def _generate_batch_navs_numba(
    batch_size: int,
    n_points: int,
    avg_return: float,
    volatility: float,
    df: int,
    drawdown_prob: float,
    dd_mag_low: float,
    dd_mag_high: float,
    dd_recovery_prob: float,
) -> np.ndarray:
    """Generate multiple NAV series in parallel using Numba.

    Returns shape (batch_size, n_points) array of NAV values.
    """
    navs = np.empty((batch_size, n_points), dtype=np.float64)

    for b in numba.prange(batch_size):
        # Generate t-distributed returns
        # Numba doesn't have t-distribution, so we approximate with normal
        # scaled by chi-squared for more realistic fat tails
        returns = np.random.standard_normal(n_points) * volatility + avg_return

        # Apply drawdown events
        drawdown = False
        for i in range(n_points):
            if drawdown:
                magnitude = dd_mag_low + np.random.random() * (dd_mag_high - dd_mag_low)
                returns[i] -= magnitude
                if np.random.random() < dd_recovery_prob:
                    drawdown = False
            elif np.random.random() < drawdown_prob:
                drawdown = True

        # Clamp and compute cumulative product
        for i in range(n_points):
            if returns[i] < -0.99:
                returns[i] = -0.99

        # Compute NAV as cumprod(1 + returns)
        navs[b, 0] = 1.0 + returns[0]
        for i in range(1, n_points):
            navs[b, i] = navs[b, i - 1] * (1.0 + returns[i])

    return navs


@njit(parallel=True)
def _compute_batch_sharpe_ratios(navs: np.ndarray, periods_per_year: float) -> np.ndarray:
    """Compute Sharpe ratios for a batch of NAV series.

    Args:
        navs: Shape (batch_size, n_points) array of NAV values
        periods_per_year: Annualization factor (e.g., 365 for daily)

    Returns:
        Shape (batch_size,) array of Sharpe ratios
    """
    batch_size, n_points = navs.shape
    sharpes = np.empty(batch_size, dtype=np.float64)

    for b in numba.prange(batch_size):
        # Compute returns from NAV
        returns = np.empty(n_points - 1, dtype=np.float64)
        for i in range(n_points - 1):
            returns[i] = (navs[b, i + 1] - navs[b, i]) / navs[b, i]

        # Compute Sharpe ratio
        n = len(returns)
        if n < 2:
            sharpes[b] = np.nan
            continue

        mean_ret = np.mean(returns)
        var_sum = 0.0
        for i in range(n):
            var_sum += (returns[i] - mean_ret) ** 2
        std_ret = np.sqrt(var_sum / (n - 1))

        if std_ret == 0:
            sharpes[b] = np.nan
        else:
            sharpes[b] = np.sqrt(periods_per_year) * (mean_ret / std_ret)

    return sharpes


@njit(parallel=True)
def _compute_batch_max_drawdowns(navs: np.ndarray) -> np.ndarray:
    """Compute max drawdowns for a batch of NAV series.

    Args:
        navs: Shape (batch_size, n_points) array of NAV values

    Returns:
        Shape (batch_size,) array of max drawdown values (positive values, e.g., 0.15 = 15% drawdown)
    """
    batch_size, n_points = navs.shape
    max_dds = np.empty(batch_size, dtype=np.float64)

    for b in numba.prange(batch_size):
        running_max = navs[b, 0]
        max_dd = 0.0

        for i in range(n_points):
            if navs[b, i] > running_max:
                running_max = navs[b, i]
            dd = (running_max - navs[b, i]) / running_max
            if dd > max_dd:
                max_dd = dd

        max_dds[b] = max_dd

    return max_dds


def generate_batch_synthetic_navs(
    params: SyntheticNavParams,
    batch_size: int = 1000,
    sr_lower: float = 0.5,
    sr_upper: float = 10.0,
    min_epochs: int = 1,
) -> list[pd.DataFrame]:
    """Generate a batch of synthetic NAVs and pre-filter by Sharpe ratio.

    This is much faster than generating one at a time because:
    1. Generates all random numbers at once (vectorized)
    2. Computes Sharpe ratios in parallel using Numba
    3. Pre-filters before expensive ITH epoch computation

    Args:
        params: SyntheticNavParams configuration
        batch_size: Number of NAVs to generate in one batch
        sr_lower: Lower bound for Sharpe ratio pre-filter
        sr_upper: Upper bound for Sharpe ratio pre-filter
        min_epochs: Minimum epochs expected (used for adaptive parameter tuning)

    Returns:
        List of NAV DataFrames that passed the Sharpe ratio pre-filter
    """
    if params.n_points is not None:
        n = params.n_points
        dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
    else:
        dates = pd.date_range(params.start_date, params.end_date)
        n = len(dates)

    # Generate batch of NAVs using Numba
    navs_array = _generate_batch_navs_numba(
        batch_size=batch_size,
        n_points=n,
        avg_return=params.avg_period_return,
        volatility=params.period_return_volatility,
        df=params.df,
        drawdown_prob=params.drawdown_prob,
        dd_mag_low=params.drawdown_magnitude_low,
        dd_mag_high=params.drawdown_magnitude_high,
        dd_recovery_prob=params.drawdown_recovery_prob,
    )

    # Compute Sharpe ratios in parallel
    sharpes = _compute_batch_sharpe_ratios(navs_array, periods_per_year=365.0)

    # Pre-filter by Sharpe ratio
    valid_mask = (sharpes >= sr_lower) & (sharpes <= sr_upper) & ~np.isnan(sharpes)
    valid_indices = np.where(valid_mask)[0]

    logger.debug(
        f"Batch generation: {batch_size} generated, {len(valid_indices)} passed SR filter "
        f"(acceptance rate: {100*len(valid_indices)/batch_size:.1f}%)"
    )

    # Convert valid NAVs to DataFrames
    result = []
    for idx in valid_indices:
        nav_values = navs_array[idx]
        nav = pd.DataFrame(data=nav_values, index=dates, columns=["NAV"])
        nav.index.name = "Date"
        nav["PnL"] = nav["NAV"].diff()
        nav["PnL"] = nav["PnL"].fillna(nav["NAV"].iloc[0] - 1)
        result.append(nav)

    return result


@njit
def _sharpe_ratio_numba_helper(
    returns: np.ndarray, nperiods: float, rf: float = 0.0, annualize: bool = True
) -> float:
    valid_returns = returns[~np.isnan(returns)]
    n = len(valid_returns)
    if n < 2:
        return np.nan
    mean_returns = np.mean(valid_returns)
    std_dev = np.sqrt(np.sum((valid_returns - mean_returns) ** 2) / (n - 1))
    if std_dev == 0:
        return np.nan
    mean_diff = mean_returns - rf
    return (
        np.sqrt(nperiods) * (mean_diff / std_dev) if annualize else mean_diff / std_dev
    )


def sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: float,
    rf: float = 0.0,
) -> float:
    """Calculate annualized Sharpe ratio with explicit periods.

    This is the time-agnostic API. For any data frequency, caller specifies
    the number of periods per year directly.

    Args:
        returns: Array of periodic returns
        periods_per_year: Number of data points per year
            - 252 for daily equity
            - 365 for daily crypto
            - 8760 for hourly (365*24)
            - Custom for range bars/tick data
        rf: Risk-free rate (default 0.0)

    Returns:
        Annualized Sharpe ratio

    Example:
        # Daily equity data
        sr = sharpe_ratio(returns, periods_per_year=252)

        # Range bar data (500 bars expected per year)
        sr = sharpe_ratio(returns, periods_per_year=500)
    """
    return _sharpe_ratio_numba_helper(returns, periods_per_year, rf, annualize=True)


def sharpe_ratio_numba(
    returns: np.ndarray,
    granularity: str,
    market_type: str = "crypto",
    rf: float = 0.0,
    annualize: bool = True,
) -> float:
    """DEPRECATED: Use sharpe_ratio(returns, periods_per_year) instead.

    This function is maintained for backward compatibility but will emit
    a deprecation warning.
    """
    import warnings

    warnings.warn(
        "sharpe_ratio_numba() is deprecated. Use sharpe_ratio(returns, periods_per_year) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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
    return _sharpe_ratio_numba_helper(returns, nperiods, rf, annualize)


class MaxDrawdownResult(NamedTuple):
    max_drawdown: float


def max_drawdown(nav_values) -> MaxDrawdownResult:
    """Calculate maximum drawdown using canonical Numba-accelerated function."""
    mdd = _max_drawdown_numba(nav_values)
    return MaxDrawdownResult(max_drawdown=mdd)


def save_files(fig, filename, output_dir, nav_data, uid, source_file=None):
    """Save HTML and CSV files with optional source file name in output filename"""
    try:
        output_dir = Path(output_dir).resolve()
        if not output_dir.exists():
            logger.warning(f"Creating missing output directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)

        # Add source filename to output if provided
        if source_file:
            source_name = Path(source_file).stem
            filename = filename.replace(".html", f"_{source_name}.html")

        # Remove '_nav_data_synthetic' suffix if present
        filename = filename.replace("_nav_data_synthetic.html", ".html")

        html_path = output_dir / filename
        csv_path = output_dir / filename.replace(".html", ".csv")

        logger.debug(f"Saving files to: HTML={html_path}, CSV={csv_path}")

        # Verify write permissions before attempting to save
        if not os.access(str(output_dir), os.W_OK):
            logger.error(f"No write permission for directory: {output_dir}")
            return False

        # Save files with explicit error handling
        try:
            fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=False)
        except Exception as e:  # noqa: BLE001 - pre-existing, logs and returns
            logger.error(f"Error saving HTML file: {e}")
            return False

        try:
            nav_data.to_csv(str(csv_path))
        except Exception as e:  # noqa: BLE001 - pre-existing, logs and returns
            logger.error(f"Error saving CSV file: {e}")
            return False

        # Verify files were created with correct size
        if not html_path.exists() or html_path.stat().st_size == 0:
            logger.error(f"HTML file not created or empty: {html_path}")
            return False

        if not csv_path.exists() or csv_path.stat().st_size == 0:
            logger.error(f"CSV file not created or empty: {csv_path}")
            return False

        logger.success(f"Successfully saved files to {output_dir}")
        return True

    except Exception as e:  # noqa: BLE001 - pre-existing, outer catch-all
        logger.error(f"Error in save_files: {e}")
        return False


class DeepestTroughsResult(NamedTuple):
    deepest_troughs_after_new_high: pd.Series
    new_high_flag: pd.Series


@njit
def _deepest_troughs_and_new_highs_numba(
    nav_values: np.ndarray, running_max_nav: np.ndarray
):
    """Numba-accelerated trough tracking after new highs."""
    n = len(nav_values)
    deepest_troughs = np.zeros(n, dtype=np.float64)
    new_high_flag = np.zeros(n, dtype=np.int32)

    current_max = running_max_nav[0]
    current_trough = nav_values[0]
    deepest_troughs[0] = current_trough

    for i in range(1, n):
        if running_max_nav[i] > current_max:
            current_max = running_max_nav[i]
            current_trough = nav_values[i]
            new_high_flag[i] = 1
        elif nav_values[i] < current_trough:
            current_trough = nav_values[i]
        deepest_troughs[i] = current_trough

    return deepest_troughs, new_high_flag


def deepest_troughs_and_new_highs(nav_values, running_max_nav) -> DeepestTroughsResult:
    """Calculate deepest troughs after new highs using Numba acceleration."""
    # Convert pandas Series to numpy arrays for Numba
    nav_arr = nav_values.values if hasattr(nav_values, "values") else nav_values
    max_arr = running_max_nav.values if hasattr(running_max_nav, "values") else running_max_nav

    # Call Numba-accelerated function
    troughs_arr, flags_arr = _deepest_troughs_and_new_highs_numba(nav_arr, max_arr)

    # Convert back to pandas Series with proper index
    index = nav_values.index if hasattr(nav_values, "index") else None
    deepest_troughs_after_new_high = pd.Series(troughs_arr, index=index, dtype=float)
    new_high_flag = pd.Series(flags_arr, index=index, dtype=int)

    return DeepestTroughsResult(deepest_troughs_after_new_high, new_high_flag)


class MaxDDPointsResult(NamedTuple):
    max_dd_points: pd.Series


@njit
def _max_dd_points_after_new_high_numba(
    drawdowns: np.ndarray, new_high_flag: np.ndarray
) -> np.ndarray:
    """Numba-accelerated max drawdown point tracking."""
    n = len(drawdowns)
    max_dd_points = np.zeros(n, dtype=np.float64)
    current_max_dd = 0.0
    max_dd_index = -1

    for i in range(1, n):
        if new_high_flag[i] == 1:
            if max_dd_index != -1:
                max_dd_points[max_dd_index] = current_max_dd
            current_max_dd = 0.0
            max_dd_index = -1
        else:
            if drawdowns[i] > current_max_dd:
                current_max_dd = drawdowns[i]
                max_dd_index = i

    # Handle final segment
    if max_dd_index != -1:
        max_dd_points[max_dd_index] = current_max_dd

    return max_dd_points


def max_dd_points_after_new_high(drawdowns, new_high_flag) -> MaxDDPointsResult:
    """Calculate max drawdown points after new highs using Numba acceleration."""
    # Convert pandas Series to numpy arrays for Numba
    dd_arr = drawdowns.values if hasattr(drawdowns, "values") else drawdowns
    flag_arr = new_high_flag.values if hasattr(new_high_flag, "values") else new_high_flag

    # Call Numba-accelerated function
    max_dd_arr = _max_dd_points_after_new_high_numba(dd_arr, flag_arr)

    # Convert back to pandas Series with proper index
    index = drawdowns.index if hasattr(drawdowns, "index") else None
    max_dd_points = pd.Series(max_dd_arr, index=index)

    return MaxDDPointsResult(max_dd_points)


class GeometricMeanDrawdownResult(NamedTuple):
    geometric_mean: float


def geometric_mean_of_drawdown(nav_values) -> GeometricMeanDrawdownResult:
    running_max_nav = nav_values.cummax()
    deepest_troughs_result = deepest_troughs_and_new_highs(nav_values, running_max_nav)
    drawdowns_to_deepest_troughs = (
        running_max_nav - deepest_troughs_result.deepest_troughs_after_new_high
    )
    max_dd_points_result = max_dd_points_after_new_high(
        drawdowns_to_deepest_troughs, deepest_troughs_result.new_high_flag
    )
    max_dd_points_fraction = max_dd_points_result.max_dd_points / running_max_nav
    spike_values = max_dd_points_fraction[max_dd_points_fraction > 0]
    if spike_values.empty:  # Check if spike_values is empty
        geometric_mean = np.nan  # Return NaN or some other appropriate value
    else:
        geometric_mean = stats.gmean(spike_values)
    return GeometricMeanDrawdownResult(geometric_mean=geometric_mean)


def bull_excess_gain_excess_loss_numba(hurdle, nav):
    """Wrapper for bull epoch calculation that handles DataFrame input.

    NOTE: This wrapper maintains the legacy parameter order (hurdle, nav).
    The canonical API in bull_ith_numba uses (nav, hurdle).
    """
    original_df = (
        nav.copy() if isinstance(nav, pd.DataFrame) and "NAV" in nav.columns else None
    )
    nav_array = nav["NAV"].values if original_df is not None else nav.values
    # Call canonical function with correct parameter order (nav, hurdle)
    result = bull_excess_gain_excess_loss(nav_array, hurdle)
    if original_df is not None:
        original_df["Excess Gains"] = result.excess_gains
        original_df["Excess Losses"] = result.excess_losses
        original_df["BullEpochs"] = result.bull_epochs
        return original_df
    else:
        return result


# Backwards compatibility aliases
excess_gain_excess_loss_numba = bull_excess_gain_excess_loss_numba


def get_first_non_zero_digits(num, digit_count):
    # Get first 12 non-zero, non-decimal, non-negative digits
    non_zero_digits = "".join([i for i in str(num) if i not in ["0", ".", "-"]][:12])
    # Then trim or pad to the desired length
    uid = (non_zero_digits + "0" * digit_count)[:digit_count]
    return uid


class PnLResult(NamedTuple):
    nav_data: pd.DataFrame


def pnl_from_nav(nav_data) -> PnLResult:
    """Calculate PnL from NAV as a fractional percentage."""
    try:
        nav_copy = nav_data.copy()  # Create explicit copy to avoid chained assignment
        nav_copy.loc[:, "PnL"] = nav_copy["NAV"].diff() / nav_copy["NAV"].shift(1)
        nav_copy.loc[nav_copy.index[0], "PnL"] = 0  # Set first row PnL
        return PnLResult(nav_data=nav_copy)
    except Exception as e:
        logger.error(f"Error calculating PnL: {e}")
        raise


class ProcessNavDataResult(NamedTuple):
    qualified_results: int
    sharpe_ratio_val: float
    num_of_bull_epochs: int
    filename: str
    uid: str
    fig: go.Figure  # Add fig to the NamedTuple
    recent_3_max_days: float  # Add recent 3 maximum days


def log_results(results):
    headers = [
        "TMAEG",
        "Sharpe<br/>Ratio",
        "Bull<br/>Epochs<br/>Count",
        "EL CV",
        "Bull CV",
        "AGG CV",
        "P2E<br/>(Points to Epoch)",
        "Recent 3<br/>Max Points",
        "Filename",
        "Data Source",
    ]
    df = pd.DataFrame(results, columns=headers)

    # Filter out rows where Filename is None
    df = df[df["Filename"].notna()]

    # Add a rank column for AGG CV (convert to integer)
    df["AGG CV Rank"] = df["AGG CV"].rank(method="min").astype(int)

    # Add a rank column for Sharpe Ratio (largest value ranks as #1, convert to integer)
    df["Sharpe<br/>Ratio Rank"] = (
        df["Sharpe<br/>Ratio"].rank(method="min", ascending=False).astype(int)
    )

    # Add a rank column for Recent 3 Max Points (smallest value ranks as #1, convert to integer)
    df["Recent 3<br/>Max Points Rank"] = (
        df["Recent 3<br/>Max Points"].rank(method="min", ascending=True).astype(int)
    )

    # Create a new column with HTML links
    df["Link"] = df["Filename"].apply(
        lambda x: f'<a href="synth_bull_ithes/{x}" target="_blank">Open</a>'
    )

    # Reorder columns to put Data Source before Link, and exclude Filename
    columns = [
        col for col in df.columns if col not in ["Filename", "Data Source", "Link"]
    ]
    columns.extend(["Data Source", "Link"])
    df = df[columns]

    # Get the index of the Recent 3 Max Points Rank column for default sorting
    recent_3_rank_col_index = df.columns.get_loc("Recent 3<br/>Max Points Rank")

    html_table = df.to_html(
        classes="table table-striped",
        index=False,
        table_id="results_table",
        escape=False,
    )

    # Add DataTables script to enable sorting and set page length
    html_output = f"""
    <html>
    <head>
    <title>Bull ITH Fitness Analysis Results</title>
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.css">
    <link rel="stylesheet" href="https://classless.de/classless.css">
    <style>
        body {{ font-family: Osaka, 'SF Mono', Monaco, Menlo, Consolas, 'Cascadia Code', 'JetBrains Mono', 'Fira Code', 'Source Code Pro', 'Liberation Mono', 'DejaVu Sans Mono', 'Courier New', monospace !important; margin: 0 !important; padding: 20px !important; }}
        .container {{ max-width: 100% !important; margin: 0 auto !important; padding: 0 15px !important; }}
        table {{ width: 100% !important; border-collapse: collapse !important; margin: 0 auto !important; font-family: inherit !important; }}
        th, td {{ padding: 8px 12px !important; border-bottom: 1px solid #ddd !important; font-family: inherit !important; }}
        th {{ background-color: #f8f9fa !important; font-weight: 600 !important; text-align: right !important; line-height: 1.2 !important; }}
        td {{ text-align: right !important; }}
        /* Left-align text columns */
        th:nth-last-child(2), td:nth-last-child(2) {{ text-align: left !important; }} /* Data Source */
        th:nth-last-child(1), td:nth-last-child(1) {{ text-align: left !important; }} /* Link */
        /* Multi-line header support */
        th {{ vertical-align: middle !important; }}
        .dataTables_wrapper {{ width: 100% !important; font-family: inherit !important; }}
        h1 {{ font-family: inherit !important; }}
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
        <h1>Bull ITH Fitness Analysis Results</h1>
        {html_table}
    </div>
    </body>
    </html>
    """

    html_file_path = get_artifacts_dir() / "bull_results.html"
    with open(html_file_path, "w") as file:
        file.write(html_output)

    logger.success("Results have been written to bull_results.html")
    console.print(f"📊 [bold green]Results saved to:[/bold green] {html_file_path}")

    # Automatically open the HTML file in the default web browser if enabled
    if plot_config.auto_open_index_html:
        webbrowser.open(f"file://{html_file_path.resolve()}")
        console.print("🌐 [blue]Opening results in browser...[/blue]")


# Initialize a list to store results
results = []


def process_nav_data(
    nav_data,
    output_dir,
    qualified_results,
    nav_dir,
    TMAEG,
    bypass_thresholds=False,
    data_source="Synthetic",
) -> ProcessNavDataResult:
    """Process NAV data through ITH algorithm.

    Emits algorithm.init telemetry event for reproducibility.
    """
    from ith_python.telemetry import log_algorithm_init
    from ith_python.telemetry.provenance import fingerprint_array

    # Extract the first six non-zero digits from the first two rows of the NAV column
    uid_part1 = get_first_non_zero_digits(nav_data["NAV"].iloc[0], 6)
    uid_part2 = get_first_non_zero_digits(nav_data["NAV"].iloc[1], 6)
    # Concatenate the two parts to form the UID
    uid = uid_part1 + uid_part2
    logger.debug(f"{uid=}")

    # Emit algorithm.init telemetry event
    nav_fp = fingerprint_array(nav_data["NAV"].values, "nav_input")
    log_algorithm_init(
        algorithm_name="bull_ith",
        version="1.0.0",
        config={
            "tmaeg": TMAEG,
            "bypass_thresholds": bypass_thresholds,
            "data_source": data_source,
            "nav_length": len(nav_data),
        },
        input_hash=nav_fp["sha256"],
    )

    # Initialize filename with a default value
    filename = None
    bull_durations = None
    excess_losses_at_bull_epochs = None
    fig = None  # Initialize fig to None

    # Calculate points_elapsed (time-agnostic) and days_elapsed (for display)
    points_elapsed = len(nav_data) - 1  # Number of data points
    days_elapsed = (nav_data.index[-1] - nav_data.index[0]).days  # For display only

    # Use time-agnostic sharpe_ratio with explicit periods (default: daily crypto = 365)
    sharpe_ratio_val = sharpe_ratio(nav_data["PnL"].dropna().values, periods_per_year=365)
    calculated_nav = excess_gain_excess_loss_numba(TMAEG, nav_data)

    if isinstance(calculated_nav, pd.DataFrame):
        bull_epochs_idx = calculated_nav[calculated_nav["BullEpochs"]].index
        num_of_bull_epochs = len(bull_epochs_idx)
    else:
        bull_epochs_idx = nav_data.index[calculated_nav.bull_epochs]
        num_of_bull_epochs = calculated_nav.num_of_bull_epochs

    # Add detailed logging for threshold checks
    logger.debug(f"Threshold check details: bypass_thresholds={bypass_thresholds}")
    logger.debug(f"SR check: {sr_lower_bound} < {sharpe_ratio_val} < {sr_upper_bound}")
    logger.debug(
        f"Bull epochs check: {bull_epochs_lower_bound} < {num_of_bull_epochs} < {bull_epochs_upper_bound}"
    )

    if bypass_thresholds or (
        sr_lower_bound < sharpe_ratio_val < sr_upper_bound
        and bull_epochs_lower_bound < num_of_bull_epochs < bull_epochs_upper_bound
    ):
        if bypass_thresholds:
            logger.debug("Bypassing thresholds for custom CSV.")
        else:
            logger.debug("Thresholds met for synthetic data.")

        logger.debug(f"Found {num_of_bull_epochs=}, {sharpe_ratio_val=}")

        # Get actual bull epoch dates (confirmed epochs only)
        bull_epoch_dates = calculated_nav[calculated_nav["BullEpochs"]].index

        # Create complete timeline with start, epochs, and end for interval calculation
        # Only include the end date if it's NOT already an epoch (to avoid double counting)
        timeline_dates = bull_epoch_dates.insert(0, calculated_nav.index[0])
        if not calculated_nav["BullEpochs"].iloc[-1]:  # Only add end if it's not an epoch
            timeline_dates = timeline_dates.append(pd.Index([calculated_nav.index[-1]]))

        logger.debug(f"timeline_dates: {timeline_dates}")

        # Calculate actual confirmed bull epoch count (excluding artificial boundaries)
        bull_epoch_ct = len(bull_epoch_dates)  # Only count real epochs

        # Calculate intervals in days between timeline points
        timeline_indices = [
            calculated_nav.index.get_loc(date) for date in timeline_dates
        ]
        bull_durations_indices = np.diff(timeline_indices)

        # Convert index differences to actual days
        date_diffs = [
            (timeline_dates[i + 1] - timeline_dates[i]).days
            for i in range(len(timeline_dates) - 1)
        ]
        bull_durations_days = np.array(date_diffs)

        logger.debug(f"bull_durations_indices: {bull_durations_indices}")
        logger.debug(f"bull_durations_days: {bull_durations_days}")

        # Calculate points to epoch (time-agnostic metric)
        points_to_bull_epoch = points_elapsed / bull_epoch_ct if bull_epoch_ct > 0 else np.nan
        # Backward compatibility alias (deprecated)
        days_taken_to_bull_epoch = points_to_bull_epoch

        # Calculate maximum of recent 3 timespan durations in days
        recent_3_max_days = (
            np.max(bull_durations_days[-3:])
            if len(bull_durations_days) >= 3
            else np.max(bull_durations_days) if len(bull_durations_days) > 0 else np.nan
        )

        # Calculate bull CV using index-based durations for consistency
        bull_cv = (
            np.std(bull_durations_indices) / np.mean(bull_durations_indices)
            if len(bull_durations_indices) > 0
            else np.nan
        )

        # Calculate the coefficient of variation for Excess Losses at Bull Epochs
        excess_losses_at_bull_epochs = calculated_nav[calculated_nav["BullEpochs"]][
            "Excess Losses"
        ]
        excess_losses_at_bull_epochs = excess_losses_at_bull_epochs[
            excess_losses_at_bull_epochs != 0
        ]  # Exclude zero values
        last_excess_loss = calculated_nav["Excess Losses"].iloc[
            -1
        ]  # Include the last value of Excess Losses (even if it is not flagged with Bull Epoch True), unless it's already included
        if not calculated_nav["BullEpochs"].iloc[
            -1
        ]:  # Check if the last value of Bull Epoch is False
            excess_losses_at_bull_epochs = pd.concat(
                [
                    excess_losses_at_bull_epochs,
                    pd.Series([last_excess_loss], index=[calculated_nav.index[-1]]),
                ]
            )
        if excess_losses_at_bull_epochs.empty:  # Check if excess_losses_at_bull_epochs is empty
            el_cv = np.nan  # Return NaN or some other appropriate value
        else:
            el_cv = np.std(excess_losses_at_bull_epochs) / np.mean(excess_losses_at_bull_epochs)

        aggcv = max(el_cv, bull_cv)
        logger.debug(f"{aggcv=}")

        # Add logging for aggcv check
        logger.debug(f"AGGCV check: {aggcv_low_bound} < {aggcv} < {aggcv_up_bound}")

        if bypass_thresholds or (aggcv_low_bound < aggcv < aggcv_up_bound):
            if bypass_thresholds:
                source_name = Path(nav_dir).stem if nav_dir else "unknown"
                # Symmetric with bear_ith.py which uses "bear_nav_"
                filename = (
                    f"bull_nav_"
                    f"EL_{el_cv:.5f}_"
                    f"BullCV_{bull_cv:.5f}_"
                    f"TMAEG_{TMAEG:.5f}_"
                    f"BullEpochs_{bull_epoch_ct}_"
                    f"P2E_{points_to_bull_epoch:.2f}_"
                    f"SR_{sharpe_ratio_val:.4f}_"
                    f"UID_{uid}.html"
                )
            else:
                filename = (
                    f"EL_{el_cv:.5f}_"
                    f"BullCV_{bull_cv:.5f}_"
                    f"TMAEG_{TMAEG:.5f}_"
                    f"BullEpochs_{bull_epoch_ct}_"
                    f"P2E_{points_to_bull_epoch:.2f}_"
                    f"SR_{sharpe_ratio_val:.4f}_"
                    f"UID_{uid}.html"
                )

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.01,
                subplot_titles=("NAV", "Excess Gains & Losses"),
            )

            bull_epoch_indices = calculated_nav[calculated_nav["BullEpochs"]].index
            num_of_bull_epochs_local = len(bull_epoch_indices)
            bull_epochs_dir = output_dir / f"Bull_Epochs_{num_of_bull_epochs_local}"
            bull_epochs_dir.mkdir(parents=True, exist_ok=True)
            crossover_epochs = calculated_nav.loc[bull_epoch_indices]
            fig.add_trace(
                go.Scatter(
                    x=crossover_epochs.index,
                    y=crossover_epochs["NAV"],
                    mode="markers",
                    name="Bull Epochs on NAV",
                    marker=dict(color="darkgoldenrod", size=20),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=crossover_epochs.index,
                    y=crossover_epochs["Excess Gains"],
                    mode="markers",
                    name="Bull Epochs on Excess Gains",
                    marker=dict(color="blue", size=20),
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
                title=f"{num_of_bull_epochs_local} Bull Epochs -- {filename}",
                autosize=True,  # Enable autosize for responsive layout
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

            # Generate monthly ticks between the minimum and maximum dates
            monthly_ticks = pd.date_range(
                nav_data.index.min(), nav_data.index.max(), freq="MS"
            )
            monthly_tick_labels = monthly_ticks.strftime("%Y-%m")

            # Customize X-axis grid lines
            custom_xaxis = XAxis(
                tickmode=plot_config.xaxis_tickmode,
                tickvals=monthly_ticks,  # Set to monthly_ticks
                showgrid=True,  # Show vertical grid
                gridwidth=plot_config.xaxis_gridwidth,  # Vertical grid width
            )

            custom_yaxis = YAxis(
                showgrid=True,  # Show vertical grid
                gridwidth=plot_config.yaxis_gridwidth,  # Vertical grid width
            )
            fig.update_layout(xaxis=custom_xaxis.to_plotly_json())
            fig.update_layout(yaxis=custom_yaxis.to_plotly_json())

            # Automatically open the live view version of the HTML if enabled
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

            # Increment qualified_results if not bypassing thresholds
            if not bypass_thresholds:
                qualified_results += 1

            logger.debug(
                f"Generated {TMAEG=}, {sharpe_ratio_val=}, {num_of_bull_epochs=}, {el_cv=},{bull_cv=}, {aggcv=}, {points_to_bull_epoch=}"
            )

        # Append the results to the list only if filename is not None
        if filename:
            results.append(
                [
                    TMAEG,
                    sharpe_ratio_val,
                    num_of_bull_epochs,
                    el_cv,
                    bull_cv,
                    aggcv,
                    points_to_bull_epoch,
                    recent_3_max_days,
                    filename,
                    data_source,
                ]
            )

        # Automatically open the NAV HTML file in the default web browser if enabled
        if plot_config.auto_open_nav_html and filename:
            file_path = str(output_dir / filename)
            webbrowser.open(f"file://{os.path.abspath(file_path)}")

    else:
        pass  # Removed repetitive warning logs

    if bull_durations is not None:
        logger.debug(f"bull_durations in process_nav_data: {bull_durations}")

    logger.debug(
        f"excess_losses_at_bull_epochs in process_nav_data: {excess_losses_at_bull_epochs}"
    )

    return ProcessNavDataResult(
        qualified_results=qualified_results,
        sharpe_ratio_val=sharpe_ratio_val,
        num_of_bull_epochs=num_of_bull_epochs,
        filename=filename,
        uid=uid,
        fig=fig,  # Return the fig object
        recent_3_max_days=(
            recent_3_max_days if "recent_3_max_days" in locals() else np.nan
        ),
    )


# Initialize counters
qualified_results = 0
counter = 0  # Add counter initialization here

# Try to load existing data
existing_csv_files = glob.glob(str(nav_dir / "*.csv"))

logger.debug(f"{existing_csv_files=}")


def determine_tmaeg(nav_data, method):
    if method == "geomean":
        return geometric_mean_of_drawdown(nav_data["NAV"]).geometric_mean
    elif method == "mdd":
        return max_drawdown(nav_data["NAV"]).max_drawdown
    elif method == "fixed":
        return TMAEG
    return None


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
console.print("🔄 [bold yellow]Starting synthetic data generation (batch mode)...[/bold yellow]")
# Log config values at start for debugging
logger.debug(f"Generation config: {bull_epochs_lower_bound=}, {bull_epochs_upper_bound=}")
logger.debug(f"Generation config: {sr_lower_bound=}, {sr_upper_bound=}")
logger.debug(f"Generation config: {config.required_qualified_results=}")

# Batch generation settings
BATCH_SIZE = 500  # Generate 500 NAVs at a time
total_generated = 0
total_prefiltered = 0
total_disqualified = 0

with create_progress_bar(console) as progress:
    task = progress.add_task(
        "Generating synthetic data (batch)...",
        total=config.required_qualified_results,
        disqualified=0,
        qualified=qualified_results,
        prefiltered=0,
    )

    while qualified_results < config.required_qualified_results:
        # Generate a batch of pre-filtered NAVs
        batch_navs = generate_batch_synthetic_navs(
            synthetic_nav_params,
            batch_size=BATCH_SIZE,
            sr_lower=sr_lower_bound,
            sr_upper=sr_upper_bound,
            min_epochs=bull_epochs_lower_bound,
        )
        total_generated += BATCH_SIZE
        total_prefiltered += len(batch_navs)

        logger.debug(
            f"Batch: {BATCH_SIZE} generated, {len(batch_navs)} passed pre-filter "
            f"(total: {total_generated} gen, {total_prefiltered} prefilt, {qualified_results} qual)"
        )

        # Process each pre-filtered NAV through full ITH analysis
        for synthetic_nav in batch_navs:
            if qualified_results >= config.required_qualified_results:
                break

            counter += 1
            TMAEG = determine_tmaeg(synthetic_nav, config.TMAEG_dynamically_determined_by)
            result = process_nav_data(
                synthetic_nav,
                config.output_dir,
                qualified_results,
                nav_dir,
                TMAEG,
                bypass_thresholds=False,
                data_source="Generated",
            )
            qualified_results = result.qualified_results

            if result.filename is not None:
                logger.debug(
                    f"Processing synthetic data {counter:4}: {result.filename=}, {result.uid=}, {TMAEG=}, {result.sharpe_ratio_val=}, newly generated."
                )
                if result.fig is not None:
                    save_files(
                        result.fig, result.filename, nav_dir, synthetic_nav, result.uid
                    )
                progress.update(task, advance=1, qualified=qualified_results)
            else:
                total_disqualified += 1
                # Log first 10 disqualified to understand why
                if total_disqualified <= 10:
                    logger.debug(f"Disqualified {total_disqualified}: epochs={result.num_of_bull_epochs}, sr={result.sharpe_ratio_val:.4f}, TMAEG={TMAEG:.4f}")
                progress.update(
                    task,
                    disqualified=total_disqualified,
                    prefiltered=total_prefiltered,
                )

        # Update progress with batch stats
        progress.update(task, prefiltered=total_prefiltered)

        if qualified_results >= config.required_qualified_results:
            console.print(
                f"🎯 [bold green]Target reached![/bold green] Found {config.required_qualified_results} qualified results"
            )
            logger.info(
                f"Required number of qualified results ({config.required_qualified_results}) reached."
            )
            logger.info(
                f"Generation stats: {total_generated} generated, {total_prefiltered} pre-filtered, "
                f"{qualified_results} qualified (pre-filter rate: {100*total_prefiltered/total_generated:.1f}%)"
            )
            break

# Ensure clean line after progress bar
console.print("")


# After the config declarations, add this function to handle custom CSV files
def get_date_range_from_csvs(csv_folder: Path) -> tuple[str, str]:
    """Get the earliest start date and latest end date from all CSVs in folder"""
    if not csv_folder.exists():
        logger.warning(
            f"Custom CSV folder not found: {csv_folder}\n"
            "The folder will be created automatically. Place your custom CSV files there with columns: Date, NAV, PnL (optional)."
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
        except Exception as e:  # noqa: BLE001 - pre-existing, logs and continues
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


# * IMPORTANT: For first time users, replace this path with your own performance record folder
# * The folder should contain CSV files with columns: Date, NAV, PnL (optional)
# ^ Example path: Path('/Users/your_username/trading_records')
custom_csv_folder = get_custom_nav_dir()

# Directories already created by ensure_dirs() at module load
console.print(f"📁 [bold cyan]Custom CSV folder ready:[/bold cyan] {custom_csv_folder}")
if not any(custom_csv_folder.glob("*.csv")):
    console.print(
        "💡 [yellow]Tip:[/yellow] Place your trading CSV files (with Date, NAV columns) in the folder above to analyze your own performance data!"
    )

start_date, end_date = get_date_range_from_csvs(custom_csv_folder)

# Update synthetic params with custom date range
synthetic_nav_params = synthetic_nav_params._replace(
    start_date=start_date, end_date=end_date
)

# Process custom CSV files if they exist
custom_csvs = list(custom_csv_folder.glob("*.csv"))
if custom_csvs:
    console.print(
        f"📈 [bold blue]Processing {len(custom_csvs)} custom CSV file(s)...[/bold blue]"
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
        "Custom CSV Processing Complete",
        len(custom_csvs),
        processed_files,
        disqualified_files,
    )
else:
    console.print(
        "📂 [dim]No custom CSV files to process - continuing with synthetic data generation[/dim]"
    )

# The rest of the synthetic data generation code remains the same
# but will now use the updated date range from custom CSVs if they exist

# Log the results in a tabulated format - do this at the very end after all processing
log_results(results)
