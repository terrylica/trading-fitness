"""Numba-accelerated Bear ITH (Short Position) calculations.

This module provides JIT-compiled functions for calculating excess gain/loss
and bear market epochs for SHORT positions.

The algorithm is the INVERSE of the long (bull) algorithm:
- Long: gains from price UP, adverse from drawdown (price DOWN)
- Short: gains from price DOWN, adverse from runup (price UP)

SR&ED: Experimental development of Bear ITH algorithm.
SRED-Type: experimental-development
SRED-Claim: BEAR-ITH
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numba import njit


class BearExcessGainLossResult(NamedTuple):
    """Result of bear excess gain/loss calculation for SHORT positions."""

    excess_gains: np.ndarray  # Gains from price decline (favorable for shorts)
    excess_losses: np.ndarray  # Losses from price runup (adverse for shorts)
    num_of_bear_epochs: int
    bear_epochs: np.ndarray  # Boolean array marking bear epoch points
    bear_intervals_cv: float


@njit
def _bear_excess_gain_excess_loss_numba(nav, hurdle):
    """Calculate bear excess gains/losses for SHORT positions using Numba.

    This is the INVERSE of the long algorithm:

    Long (Bull) Algorithm:
    - endorsing_crest: Confirmed high we track performance FROM
    - candidate_nadir: Potential new low (drawdown = adverse)
    - excess_gain: Profit when price goes UP above crest
    - excess_loss: Loss when price drops below crest (drawdown)
    - Long epoch: excess_gain > excess_loss AND excess_gain > TMAEG AND new_high

    Bear (Short) Algorithm (INVERTED):
    - endorsing_trough: Confirmed LOW we track performance FROM (short entry)
    - candidate_peak: Potential new HIGH (runup = adverse for shorts)
    - excess_gain: Profit when price goes DOWN below trough
    - excess_loss: Loss when price goes UP above trough (runup)
    - Bear epoch: excess_gain > excess_loss AND excess_gain > TMAER AND new_low

    Args:
        nav: NAV values array
        hurdle: TMAER threshold (Target Maximum Acceptable Excess Runup)

    Returns:
        Tuple of (excess_gains, excess_losses, num_of_bear_epochs, bear_epochs, cv)
    """
    n = len(nav)

    # Pre-allocate arrays
    excess_gains = np.zeros(n, dtype=np.float64)
    excess_losses = np.zeros(n, dtype=np.float64)
    bear_epochs = np.zeros(n, dtype=np.bool_)

    excess_gain = 0.0
    excess_loss = 0.0

    # INVERTED: Track troughs (lows) instead of crests (highs)
    endorsing_trough = nav[0]  # Confirmed low we track FROM
    endorsing_peak = nav[0]  # Highest point since endorsing_trough
    candidate_trough = nav[0]  # Potential new low (favorable for shorts)
    candidate_peak = nav[0]  # Potential new high (runup = adverse)

    for i in range(1, n):
        equity = nav[i - 1]
        next_equity = nav[i]

        # INVERTED: Track new LOWS (favorable for shorts)
        if next_equity < candidate_trough:
            if endorsing_trough != 0 and next_equity != 0:
                # Excess gain = profit from price decline
                # SYMMETRIC with bull: (trough/new) - 1 is UNBOUNDED as price → 0
                # (vs old formula: 1 - new/trough which was bounded at 1.0)
                excess_gain = endorsing_trough / next_equity - 1
            else:
                excess_gain = 0.0
            candidate_trough = next_equity

        # INVERTED: Track new HIGHS (runup = adverse for shorts)
        if next_equity > candidate_peak:
            # Excess loss = runup hurts shorts
            # SYMMETRIC with bull: 1 - (trough/new) is BOUNDED at 1.0
            # (vs old formula: new/trough - 1 which was unbounded)
            if next_equity != 0:
                excess_loss = 1 - endorsing_trough / next_equity
            else:
                excess_loss = 0.0
            candidate_peak = next_equity

        # INVERTED reset condition: gains exceed losses, exceed hurdle, AND new low
        reset_condition = (
            excess_gain > abs(excess_loss)
            and excess_gain > hurdle
            and candidate_trough <= endorsing_trough  # New low (inverted from new high)
        )

        if reset_condition:
            endorsing_trough = candidate_trough
            endorsing_peak = equity
            candidate_peak = equity
        else:
            endorsing_peak = max(endorsing_peak, equity)

        excess_gains[i] = excess_gain
        excess_losses[i] = excess_loss

        if reset_condition:
            excess_gain = 0.0
            excess_loss = 0.0

        # Check bear epoch condition
        bear_epoch_condition = (
            excess_gains[i] > excess_losses[i] and excess_gains[i] > hurdle
        )
        bear_epochs[i] = bear_epoch_condition

    # Count bear epochs
    num_of_bear_epochs = 0
    for i in range(n):
        if bear_epochs[i]:
            num_of_bear_epochs += 1

    # Calculate bear intervals CV
    epoch_indices = np.zeros(num_of_bear_epochs + 1, dtype=np.int64)
    epoch_indices[0] = 0
    idx = 1
    for i in range(n):
        if bear_epochs[i]:
            epoch_indices[idx] = i
            idx += 1

    if num_of_bear_epochs > 0:
        bear_intervals = np.diff(epoch_indices[: num_of_bear_epochs + 1])
        if len(bear_intervals) > 0:
            mean_interval = np.mean(bear_intervals)
            if mean_interval > 0:
                bear_intervals_cv = np.std(bear_intervals) / mean_interval
            else:
                bear_intervals_cv = np.nan
        else:
            bear_intervals_cv = np.nan
    else:
        bear_intervals_cv = np.nan

    return (excess_gains, excess_losses, num_of_bear_epochs, bear_epochs, bear_intervals_cv)


def bear_excess_gain_excess_loss(
    nav: np.ndarray,
    hurdle: float,
    emit_telemetry: bool = False,
    timestamps: np.ndarray | None = None,
) -> BearExcessGainLossResult:
    """Calculate bear excess gains/losses with typed result.

    This is the wrapper function that calls the Numba-compiled core
    and returns a typed NamedTuple result.

    Args:
        nav: NAV values array
        hurdle: TMAER threshold (Target Maximum Acceptable Excess Runup)
        emit_telemetry: If True, emit epoch_detected events for each epoch
        timestamps: Optional array of timestamps for telemetry (ISO format strings)

    Returns:
        BearExcessGainLossResult with all calculation outputs
    """
    excess_gains, excess_losses, num_of_bear_epochs, bear_epochs, cv = (
        _bear_excess_gain_excess_loss_numba(nav, hurdle)
    )

    # Emit telemetry for each detected epoch (optional)
    if emit_telemetry and num_of_bear_epochs > 0:
        from ith_python.telemetry import log_epoch_detected

        epoch_index = 0
        # Track state for epoch context
        endorsing_nadir = nav[0]
        candidate_crest = nav[0]

        for i in range(len(nav)):
            if bear_epochs[i]:
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
                    endorsing_crest=float(candidate_crest),  # For bear, crest is tracked differently
                    candidate_nadir=float(endorsing_nadir),
                    tmaeg_threshold=hurdle,
                    position_type="bear",
                    timestamp=ts,
                    nav_at_epoch=float(nav[i]),
                )
                # Update tracked state after epoch
                endorsing_nadir = nav[i]
                candidate_crest = nav[i]
            else:
                # Track candidate crest for context
                if nav[i] > candidate_crest:
                    candidate_crest = nav[i]

    return BearExcessGainLossResult(
        excess_gains=excess_gains,
        excess_losses=excess_losses,
        num_of_bear_epochs=num_of_bear_epochs,
        bear_epochs=bear_epochs,
        bear_intervals_cv=cv,
    )


def max_runup(nav_values: np.ndarray) -> float:
    """Calculate maximum runup (symmetric inverse of max_drawdown).

    Maximum runup = maximum rally from any trough.
    This is ADVERSE for short positions (price going up hurts shorts).

    Uses symmetric formula to max_drawdown for bounded [0, 1) output:
    - max_drawdown = max(1 - nav / running_max)  # bounded [0, 1)
    - max_runup    = max(1 - running_min / nav)  # bounded [0, 1)

    Args:
        nav_values: NAV values array

    Returns:
        Maximum runup as a decimal (e.g., 0.20 for 20% runup), bounded [0, 1)
    """
    running_min = np.minimum.accumulate(nav_values)
    # Symmetric formula: 1 - running_min / nav (bounded like max_drawdown)
    runups = 1 - running_min / nav_values
    return float(np.max(runups))
