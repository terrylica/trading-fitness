"""Tests for numba-accelerated Bear ITH calculations.

SR&ED: Validation testing for Bear ITH experimental development.
SRED-Type: experimental-development
SRED-Claim: BEAR-ITH
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from ith_python.bear_ith_numba import (
    BearExcessGainLossResult,
    bear_excess_gain_excess_loss,
    max_runup,
)


class BearSyntheticNavParams(NamedTuple):
    """Parameters for generating synthetic bear market NAV data.

    Local definition to avoid fragile conftest imports.
    Supports both point-based and date-based modes.
    """

    n_points: int | None = None  # If set, generates exactly this many points
    start_date: str = "2020-01-01"
    end_date: str = "2020-06-30"
    avg_period_return: float = -0.001
    period_return_volatility: float = 0.008
    df: int = 5
    rally_prob: float = 0.05
    rally_magnitude_low: float = 0.001
    rally_magnitude_high: float = 0.003
    rally_recovery_prob: float = 0.05


class TestBearExcessGainLossResult:
    """Tests for BearExcessGainLossResult NamedTuple."""

    def test_result_has_expected_fields(self):
        """BearExcessGainLossResult should have all expected fields."""
        result = BearExcessGainLossResult(
            excess_gains=np.array([0.0, 0.1]),
            excess_losses=np.array([0.0, 0.05]),
            num_of_bear_epochs=1,
            bear_epochs=np.array([False, True]),
            bear_intervals_cv=0.5,
        )

        assert hasattr(result, "excess_gains")
        assert hasattr(result, "excess_losses")
        assert hasattr(result, "num_of_bear_epochs")
        assert hasattr(result, "bear_epochs")
        assert hasattr(result, "bear_intervals_cv")

    def test_result_is_immutable(self):
        """BearExcessGainLossResult should be immutable (NamedTuple)."""
        result = BearExcessGainLossResult(
            excess_gains=np.array([0.0]),
            excess_losses=np.array([0.0]),
            num_of_bear_epochs=0,
            bear_epochs=np.array([False]),
            bear_intervals_cv=np.nan,
        )

        with pytest.raises(AttributeError):
            result.num_of_bear_epochs = 5


class TestBearExcessGainExcessLoss:
    """Tests for the core bear excess gain/loss calculation."""

    def test_basic_calculation(self, sample_nav_array: np.ndarray):
        """Basic test that the function runs and returns expected types."""
        hurdle = 0.05
        result = bear_excess_gain_excess_loss(sample_nav_array, hurdle)

        assert isinstance(result, BearExcessGainLossResult)
        assert len(result.excess_gains) == len(sample_nav_array)
        assert len(result.excess_losses) == len(sample_nav_array)
        assert len(result.bear_epochs) == len(sample_nav_array)
        assert isinstance(result.num_of_bear_epochs, (int, np.integer))

    def test_flat_nav_no_epochs(self):
        """Flat NAV should produce zero bear epochs."""
        flat_nav = np.ones(100)
        hurdle = 0.05
        result = bear_excess_gain_excess_loss(flat_nav, hurdle)

        assert result.num_of_bear_epochs == 0

    def test_strong_downtrend_produces_epochs(self):
        """Strong downtrend should produce bear epochs (shorts profit)."""
        # Create strong downward trend
        nav = np.cumprod(1 + np.ones(100) * -0.02)  # 2% daily losses
        hurdle = 0.05
        result = bear_excess_gain_excess_loss(nav, hurdle)

        assert result.num_of_bear_epochs > 0

    def test_strong_uptrend_no_epochs(self):
        """Strong uptrend should produce zero bear epochs (runup adverse)."""
        # Create strong upward trend
        nav = np.cumprod(1 + np.ones(100) * 0.02)  # 2% daily gains
        hurdle = 0.05
        result = bear_excess_gain_excess_loss(nav, hurdle)

        assert result.num_of_bear_epochs == 0

    def test_excess_gains_non_negative(self, sample_nav_array: np.ndarray):
        """Excess gains should be non-negative."""
        hurdle = 0.05
        result = bear_excess_gain_excess_loss(sample_nav_array, hurdle)

        assert np.all(result.excess_gains >= 0)

    def test_different_hurdles(self):
        """Higher hurdle should produce fewer or equal bear epochs."""
        # Create downtrend for bear epochs
        nav = np.cumprod(1 + np.ones(100) * -0.01)

        result_low = bear_excess_gain_excess_loss(nav, 0.01)
        result_high = bear_excess_gain_excess_loss(nav, 0.10)

        assert result_high.num_of_bear_epochs <= result_low.num_of_bear_epochs


class TestGenerateSyntheticBearNav:
    """Tests for synthetic bear NAV generation using fixtures."""

    def test_generates_dataframe(self, generate_synthetic_bear_nav_func):
        """Should return a pandas DataFrame."""
        params = BearSyntheticNavParams(
            start_date="2020-01-01",
            end_date="2020-01-31",
        )
        nav = generate_synthetic_bear_nav_func(params)
        assert isinstance(nav, pd.DataFrame)

    def test_has_nav_column(self, generate_synthetic_bear_nav_func):
        """Generated data should have NAV column."""
        params = BearSyntheticNavParams(
            start_date="2020-01-01",
            end_date="2020-01-31",
        )
        nav = generate_synthetic_bear_nav_func(params)
        assert "NAV" in nav.columns

    def test_has_pnl_column(self, generate_synthetic_bear_nav_func):
        """Generated data should have PnL column."""
        params = BearSyntheticNavParams(
            start_date="2020-01-01",
            end_date="2020-01-31",
        )
        nav = generate_synthetic_bear_nav_func(params)
        assert "PnL" in nav.columns

    def test_has_date_index(self, generate_synthetic_bear_nav_func):
        """Generated data should have Date as index."""
        params = BearSyntheticNavParams(
            start_date="2020-01-01",
            end_date="2020-01-31",
        )
        nav = generate_synthetic_bear_nav_func(params)
        assert nav.index.name == "Date"
        assert isinstance(nav.index, pd.DatetimeIndex)

    def test_nav_values_positive(self, generate_synthetic_bear_nav_func):
        """NAV values should always be positive (multiplicative returns)."""
        params = BearSyntheticNavParams(
            start_date="2020-01-01",
            end_date="2020-06-30",  # Longer period to test stability
        )
        nav = generate_synthetic_bear_nav_func(params)
        assert (nav["NAV"] > 0).all(), "NAV should never go negative"

    def test_bear_trend_generally_declining(self, generate_synthetic_bear_nav_func):
        """Bear synthetic NAV should generally decline over time."""
        params = BearSyntheticNavParams(
            start_date="2020-01-01",
            end_date="2020-12-31",
        )
        nav = generate_synthetic_bear_nav_func(params)

        # With negative drift, NAV should generally decline
        # (though individual runs may vary due to randomness)
        start_nav = nav["NAV"].iloc[0]
        end_nav = nav["NAV"].iloc[-1]

        # Loose assertion: end should typically be lower than start
        # This may occasionally fail due to randomness, which is acceptable
        assert end_nav <= start_nav * 1.2, "Bear NAV should not rally strongly"


class TestMaxRunup:
    """Tests for max_runup calculation."""

    def test_pure_downtrend_zero_runup(self):
        """Pure downtrend should have zero runup."""
        nav = np.array([100.0, 90.0, 80.0, 70.0])
        result = max_runup(nav)
        assert result == 0.0

    def test_pure_uptrend_positive_runup(self):
        """Pure uptrend should have positive runup."""
        nav = np.array([100.0, 110.0, 120.0, 130.0])
        result = max_runup(nav)
        # Runup = 1 - (100/130) = 0.2308
        expected = 1 - 100 / 130
        assert abs(result - expected) < 0.01

    def test_runup_is_bounded(self):
        """Max runup should be bounded [0, 1) with symmetric formula."""
        # Extreme runup case
        nav = np.array([1.0, 10.0, 100.0])
        result = max_runup(nav)
        assert 0 <= result < 1.0, f"Runup should be bounded, got {result}"

    def test_runup_recovery_preserves_max(self):
        """Decline after runup should not affect max runup."""
        nav = np.array([100.0, 80.0, 120.0, 90.0])  # Rally from 80 to 120
        result = max_runup(nav)
        # Max runup from trough 80 to peak 120: 1 - 80/120 = 0.333
        expected = 1 - 80 / 120
        assert abs(result - expected) < 0.01
