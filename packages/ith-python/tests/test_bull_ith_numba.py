"""Tests for numba-accelerated Bull ITH calculations."""

import numpy as np
import pandas as pd
import pytest

from ith_python.bull_ith_numba import (
    BullExcessGainLossResult,
    bull_excess_gain_excess_loss,
    generate_synthetic_nav,
    max_drawdown,
)


class TestBullExcessGainLossResult:
    """Tests for BullExcessGainLossResult NamedTuple."""

    def test_result_has_expected_fields(self):
        """BullExcessGainLossResult should have all expected fields."""
        result = BullExcessGainLossResult(
            excess_gains=np.array([0.0, 0.1]),
            excess_losses=np.array([0.0, 0.05]),
            num_of_bull_epochs=1,
            bull_epochs=np.array([False, True]),
            bull_intervals_cv=0.5,
        )

        assert hasattr(result, "excess_gains")
        assert hasattr(result, "excess_losses")
        assert hasattr(result, "num_of_bull_epochs")
        assert hasattr(result, "bull_epochs")
        assert hasattr(result, "bull_intervals_cv")

    def test_result_is_immutable(self, sample_nav_array: np.ndarray):
        """BullExcessGainLossResult should be immutable (NamedTuple)."""
        result = bull_excess_gain_excess_loss(sample_nav_array, 0.05)

        with pytest.raises(AttributeError):
            result.num_of_bull_epochs = 999


class TestBullExcessGainExcessLoss:
    """Tests for the core bull excess gain/loss calculation."""

    def test_basic_calculation(self, sample_nav_array: np.ndarray):
        """Basic test that the function runs and returns expected types."""
        hurdle = 0.05
        result = bull_excess_gain_excess_loss(sample_nav_array, hurdle)

        assert isinstance(result, BullExcessGainLossResult)
        assert len(result.excess_gains) == len(sample_nav_array)
        assert len(result.excess_losses) == len(sample_nav_array)
        assert len(result.bull_epochs) == len(sample_nav_array)
        assert isinstance(result.num_of_bull_epochs, (int, np.integer))

    def test_flat_nav_no_epochs(self):
        """Flat NAV should produce zero bull epochs."""
        flat_nav = np.ones(100)
        hurdle = 0.05
        result = bull_excess_gain_excess_loss(flat_nav, hurdle)

        assert result.num_of_bull_epochs == 0

    def test_strong_uptrend_produces_epochs(self):
        """Strong uptrend should produce bull epochs."""
        # Create strong upward trend
        nav = np.cumprod(1 + np.ones(100) * 0.02)  # 2% daily gains
        hurdle = 0.05
        result = bull_excess_gain_excess_loss(nav, hurdle)

        assert result.num_of_bull_epochs > 0

    def test_excess_gains_non_negative(self, sample_nav_array: np.ndarray):
        """Excess gains should be non-negative."""
        hurdle = 0.05
        result = bull_excess_gain_excess_loss(sample_nav_array, hurdle)

        assert np.all(result.excess_gains >= 0)

    def test_different_hurdles(self, sample_nav_array: np.ndarray):
        """Higher hurdle should produce fewer or equal bull epochs."""
        result_low = bull_excess_gain_excess_loss(sample_nav_array, 0.01)
        result_high = bull_excess_gain_excess_loss(sample_nav_array, 0.10)

        assert result_high.num_of_bull_epochs <= result_low.num_of_bull_epochs


class TestGenerateSyntheticNav:
    """Tests for synthetic NAV generation."""

    def test_generates_dataframe(self):
        """Should return a pandas DataFrame."""
        nav = generate_synthetic_nav()
        assert isinstance(nav, pd.DataFrame)

    def test_has_nav_column(self):
        """Generated data should have NAV column."""
        nav = generate_synthetic_nav()
        assert "NAV" in nav.columns

    def test_has_date_index(self):
        """Generated data should have Date as index."""
        nav = generate_synthetic_nav()
        assert nav.index.name == "Date"
        assert isinstance(nav.index, pd.DatetimeIndex)

    def test_starts_at_one(self):
        """NAV should start at approximately 1.0."""
        nav = generate_synthetic_nav()
        assert abs(nav["NAV"].iloc[0] - 1.0) < 0.01

    def test_custom_date_range(self):
        """Should respect custom date range."""
        nav = generate_synthetic_nav(
            start_date="2021-01-01",
            end_date="2021-12-31",
        )
        assert nav.index[0] == pd.Timestamp("2021-01-01")
        assert nav.index[-1] == pd.Timestamp("2021-12-31")

    def test_nav_values_positive(self):
        """NAV values should be positive."""
        nav = generate_synthetic_nav()
        assert (nav["NAV"] > 0).all()


class TestMaxDrawdown:
    """Tests for max_drawdown() function."""

    def test_pure_uptrend_zero_drawdown(self):
        """Uptrend should have near-zero drawdown."""
        nav = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        result = max_drawdown(nav)
        assert result == 0.0

    def test_pure_downtrend_positive_drawdown(self):
        """Downtrend should have positive drawdown."""
        nav = np.array([100.0, 95.0, 90.0, 85.0, 80.0])
        result = max_drawdown(nav)
        assert result == pytest.approx(0.20, rel=0.01)

    def test_recovery_preserves_max(self):
        """Max drawdown should be preserved after recovery."""
        nav = np.array([100.0, 90.0, 80.0, 90.0, 100.0, 110.0])
        result = max_drawdown(nav)
        assert result == pytest.approx(0.20, rel=0.01)

    def test_drawdown_bounded(self):
        """Drawdown should be in [0, 1) range."""
        np.random.seed(42)
        nav = np.cumprod(1 + np.random.randn(100) * 0.02) * 100
        result = max_drawdown(nav)
        assert 0 <= result < 1
