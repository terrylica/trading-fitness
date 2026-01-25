"""Numba-accelerated Bull ITH (Long Position) calculations.

This module provides JIT-compiled functions for calculating excess gain/loss
and Bull ITH (Investment Time Horizon) epochs for LONG positions.

The algorithm tracks:
- Long positions gain from price UP (new highs)
- Drawdown (price DOWN) is adverse for longs

SR&ED: Refactored from ith_numba.py for symmetric bull/bear naming.
SRED-Type: experimental-development
SRED-Claim: BULL-ITH
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from numba import njit
from scipy import stats


class BullExcessGainLossResult(NamedTuple):
    """Result of bull excess gain/loss calculation for LONG positions."""

    excess_gains: np.ndarray  # Gains from price rally (favorable for longs)
    excess_losses: np.ndarray  # Losses from price drawdown (adverse for longs)
    num_of_bull_epochs: int
    bull_epochs: np.ndarray  # Boolean array marking bull epoch points
    bull_intervals_cv: float


@njit
def _bull_excess_gain_excess_loss_numba(nav, hurdle):
    """Calculate bull excess gains/losses for LONG positions using Numba.

    This is the Bull (Long) Algorithm:
    - endorsing_crest: Confirmed HIGH we track performance FROM
    - candidate_nadir: Potential new LOW (drawdown = adverse for longs)
    - excess_gain: Profit when price goes UP above crest
    - excess_loss: Loss when price drops below crest (drawdown)
    - Bull epoch: excess_gain > excess_loss AND excess_gain > TMAEG AND new_high

    Args:
        nav: NAV values array
        hurdle: TMAEG threshold (Target Maximum Acceptable Excess Gain)

    Returns:
        Tuple of (excess_gains, excess_losses, num_of_bull_epochs, bull_epochs, cv)
    """
    n = len(nav)

    # Pre-allocate arrays
    excess_gains = np.zeros(n, dtype=np.float64)
    excess_losses = np.zeros(n, dtype=np.float64)
    bull_epochs = np.zeros(n, dtype=np.bool_)

    excess_gain = 0.0
    excess_loss = 0.0
    endorsing_crest = nav[0]
    endorsing_nadir = nav[0]
    candidate_crest = nav[0]
    candidate_nadir = nav[0]

    for i in range(1, n):
        equity = nav[i - 1]
        next_equity = nav[i]

        # Track new HIGHS (favorable for longs)
        if next_equity > candidate_crest:
            if endorsing_crest != 0 and next_equity != 0:
                # Excess gain = profit from price rally
                excess_gain = next_equity / endorsing_crest - 1
            else:
                excess_gain = 0.0
            candidate_crest = next_equity

        # Track new LOWS (drawdown = adverse for longs)
        if next_equity < candidate_nadir:
            # Excess loss = drawdown hurts longs
            if endorsing_crest != 0:
                excess_loss = 1 - next_equity / endorsing_crest
            else:
                excess_loss = 0.0
            candidate_nadir = next_equity

        reset_condition = (
            excess_gain > abs(excess_loss)
            and excess_gain > hurdle
            and candidate_crest >= endorsing_crest
        )

        if reset_condition:
            endorsing_crest = candidate_crest
            endorsing_nadir = equity
            candidate_nadir = equity
        else:
            endorsing_nadir = min(endorsing_nadir, equity)

        excess_gains[i] = excess_gain
        excess_losses[i] = excess_loss

        if reset_condition:
            excess_gain = 0.0
            excess_loss = 0.0

        # Check bull epoch condition
        bull_epoch_condition = (
            excess_gains[i] > excess_losses[i] and excess_gains[i] > hurdle
        )
        bull_epochs[i] = bull_epoch_condition

    # Count bull epochs
    num_of_bull_epochs = 0
    for i in range(n):
        if bull_epochs[i]:
            num_of_bull_epochs += 1

    # Calculate bull intervals CV
    epoch_indices = np.zeros(num_of_bull_epochs + 1, dtype=np.int64)
    epoch_indices[0] = 0
    idx = 1
    for i in range(n):
        if bull_epochs[i]:
            epoch_indices[idx] = i
            idx += 1

    if num_of_bull_epochs > 0:
        bull_intervals = np.diff(epoch_indices[: num_of_bull_epochs + 1])
        if len(bull_intervals) > 0:
            mean_interval = np.mean(bull_intervals)
            if mean_interval > 0:
                bull_intervals_cv = np.std(bull_intervals) / mean_interval
            else:
                bull_intervals_cv = np.nan
        else:
            bull_intervals_cv = np.nan
    else:
        bull_intervals_cv = np.nan

    return (excess_gains, excess_losses, num_of_bull_epochs, bull_epochs, bull_intervals_cv)


def bull_excess_gain_excess_loss(
    nav: np.ndarray,
    hurdle: float,
    emit_telemetry: bool = False,
    timestamps: np.ndarray | None = None,
) -> BullExcessGainLossResult:
    """Calculate bull excess gains/losses with typed result.

    This is the wrapper function that calls the Numba-compiled core
    and returns a typed NamedTuple result.

    Args:
        nav: NAV values array
        hurdle: TMAEG threshold (Target Maximum Acceptable Excess Gain)
        emit_telemetry: If True, emit epoch_detected events for each epoch
        timestamps: Optional array of timestamps for telemetry (ISO format strings)

    Returns:
        BullExcessGainLossResult with all calculation outputs
    """
    excess_gains, excess_losses, num_of_bull_epochs, bull_epochs, cv = (
        _bull_excess_gain_excess_loss_numba(nav, hurdle)
    )

    # Emit telemetry for each detected epoch (optional)
    if emit_telemetry and num_of_bull_epochs > 0:
        from ith_python.telemetry import log_epoch_detected

        epoch_index = 0
        # Track state for epoch context
        endorsing_crest = nav[0]
        candidate_nadir = nav[0]

        for i in range(len(nav)):
            if bull_epochs[i]:
                epoch_index += 1
                # Get timestamp if available
                ts = None
                if timestamps is not None and i < len(timestamps):
                    ts = str(timestamps[i])

                log_epoch_detected(
                    epoch_index=epoch_index,
                    bar_index=i,
                    excess_gain=float(excess_gains[i]),
                    excess_loss=float(excess_losses[i]),
                    endorsing_crest=float(endorsing_crest),
                    candidate_nadir=float(candidate_nadir),
                    tmaeg_threshold=hurdle,
                    position_type="bull",
                    timestamp=ts,
                    nav_at_epoch=float(nav[i]),
                )
                # Update tracked state after epoch
                endorsing_crest = nav[i]
                candidate_nadir = nav[i]
            else:
                # Track candidate nadir for context
                if nav[i] < candidate_nadir:
                    candidate_nadir = nav[i]

    return BullExcessGainLossResult(
        excess_gains=excess_gains,
        excess_losses=excess_losses,
        num_of_bull_epochs=num_of_bull_epochs,
        bull_epochs=bull_epochs,
        bull_intervals_cv=cv,
    )


def max_drawdown(nav_values: np.ndarray) -> float:
    """Calculate maximum drawdown.

    Maximum drawdown = maximum drop from any peak.
    This is ADVERSE for long positions (price going down hurts longs).

    Args:
        nav_values: NAV values array

    Returns:
        Maximum drawdown as a decimal (e.g., 0.20 for 20% drawdown)
    """
    running_max = np.maximum.accumulate(nav_values)
    drawdowns = 1 - (nav_values / running_max)
    return float(np.max(drawdowns))


def generate_synthetic_nav(
    start_date: str = "2020-01-01",
    end_date: str = "2023-01-01",
    avg_period_return: float = 0.0001,
    period_return_volatility: float = 0.01,
    df: int = 5,
    n_points: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic NAV data using t-distribution returns.

    Supports two generation modes:
    1. Point-based (recommended): Set n_points, dates auto-generated
    2. Date-based (legacy): Set start_date/end_date

    Args:
        start_date: Start date for the NAV series (legacy mode).
        end_date: End date for the NAV series (legacy mode).
        avg_period_return: Average period return.
        period_return_volatility: Period return volatility.
        df: Degrees of freedom for the t-distribution.
        n_points: If set, generates exactly this many points.

    Returns:
        DataFrame with Date index and NAV column.
    """
    if n_points is not None:
        dates = pd.date_range(start="2020-01-01", periods=n_points, freq="D")
    else:
        dates = pd.date_range(start_date, end_date)

    walk = stats.t.rvs(
        df, loc=avg_period_return, scale=period_return_volatility, size=len(dates)
    )
    walk = np.cumsum(walk)
    walk = walk - walk[0] + 1  # Normalize the series so that it starts with 1
    nav = pd.DataFrame(data=walk, index=dates, columns=["NAV"])
    nav.index.name = "Date"
    return nav
