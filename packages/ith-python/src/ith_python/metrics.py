"""Shared statistical metrics with Numba acceleration.

This module provides common statistical functions used across
the ITH (Investment Time Horizon) analysis codebase.

SR&ED: Support work for code consolidation.
SRED-Type: support-work
SRED-Claim: BEAR-ITH
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit
def calculate_cv(values: np.ndarray) -> float:
    """Calculate coefficient of variation (std/mean) with Numba.

    The coefficient of variation is the ratio of the standard deviation
    to the mean. It provides a standardized measure of dispersion.

    Args:
        values: Array of numerical values

    Returns:
        CV value (std/mean), or np.nan if:
        - Array is empty
        - Mean is zero or negative
        - Array has fewer than 2 elements
    """
    n = len(values)
    if n == 0:
        return np.nan

    if n < 2:
        return np.nan

    # Calculate mean
    total = 0.0
    for v in values:
        total += v
    mean_val = total / n

    if mean_val <= 0:
        return np.nan

    # Calculate standard deviation
    var_sum = 0.0
    for v in values:
        var_sum += (v - mean_val) ** 2
    std_val = np.sqrt(var_sum / n)  # Population std

    return std_val / mean_val


@njit
def calculate_cv_sample(values: np.ndarray) -> float:
    """Calculate coefficient of variation using sample std (n-1).

    Same as calculate_cv but uses sample standard deviation (dividing by n-1)
    instead of population standard deviation.

    Args:
        values: Array of numerical values

    Returns:
        CV value (sample_std/mean), or np.nan if invalid
    """
    n = len(values)
    if n < 2:
        return np.nan

    # Calculate mean
    total = 0.0
    for v in values:
        total += v
    mean_val = total / n

    if mean_val <= 0:
        return np.nan

    # Calculate sample standard deviation (n-1)
    var_sum = 0.0
    for v in values:
        var_sum += (v - mean_val) ** 2
    std_val = np.sqrt(var_sum / (n - 1))

    return std_val / mean_val
