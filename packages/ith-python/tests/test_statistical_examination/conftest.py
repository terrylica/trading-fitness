"""Pytest fixtures for statistical examination tests."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def synthetic_nav() -> np.ndarray:
    """Generate synthetic NAV series for testing."""
    np.random.seed(42)
    n = 5000
    returns = np.random.randn(n) * 0.01
    nav = np.cumprod(1 + returns)
    return nav / nav[0]  # Normalize to start at 1.0


@pytest.fixture
def synthetic_nav_short() -> np.ndarray:
    """Generate short synthetic NAV for quick tests."""
    np.random.seed(42)
    n = 500
    returns = np.random.randn(n) * 0.01
    nav = np.cumprod(1 + returns)
    return nav / nav[0]


@pytest.fixture
def sample_ith_features_df() -> pl.DataFrame:
    """Create sample ITH features DataFrame for testing.

    Mimics structure from compute_multiscale_ith output.
    """
    np.random.seed(42)
    n = 1000

    # Generate features with realistic structure
    data = {"bar_index": list(range(n))}

    thresholds = [100, 250]
    lookbacks = [20, 50, 100]
    feature_types = ["bull_ed", "bear_ed", "bull_eg", "bear_eg", "bull_cv", "bear_cv", "max_dd", "max_ru"]

    for threshold in thresholds:
        for lookback in lookbacks:
            for ft in feature_types:
                col_name = f"ith_rb{threshold}_lb{lookback}_{ft}"
                # Generate values in [0, 1] with NaN for warmup
                values = np.random.beta(2, 5, n)  # Realistic skewed distribution
                values[: lookback - 1] = np.nan  # Warmup period
                data[col_name] = values.tolist()

    return pl.DataFrame(data)


@pytest.fixture
def sample_features_with_target(sample_ith_features_df: pl.DataFrame) -> pl.DataFrame:
    """Add a target column to sample features."""
    np.random.seed(42)
    n = len(sample_ith_features_df)

    # Create target that has some correlation with features
    base_target = np.random.randn(n) * 0.02

    # Add correlation with some features
    if "ith_rb100_lb50_bull_ed" in sample_ith_features_df.columns:
        feature_vals = sample_ith_features_df.get_column("ith_rb100_lb50_bull_ed").to_numpy()
        feature_vals = np.nan_to_num(feature_vals, nan=0.5)
        base_target += feature_vals * 0.01

    return sample_ith_features_df.with_columns(pl.Series("forward_return", base_target))
