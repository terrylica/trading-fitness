"""Tests for main ITH analysis functions."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Import only pure functions that can be tested without side effects
# Note: ith.py has module-level side effects (creates directories, configures logging)
# so we test specific functions that can be imported safely


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_sharpe_ratio_positive_returns(self):
        """Positive returns should yield positive Sharpe ratio."""
        from ith_python.ith import sharpe_ratio_numba

        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.02] * 20)  # 100 samples
        sr = sharpe_ratio_numba(returns, "1d", "crypto")
        assert sr > 0

    def test_sharpe_ratio_negative_returns(self):
        """Negative returns should yield negative Sharpe ratio."""
        from ith_python.ith import sharpe_ratio_numba

        returns = np.array([-0.01, -0.02, -0.01, -0.015, -0.02] * 20)
        sr = sharpe_ratio_numba(returns, "1d", "crypto")
        assert sr < 0

    def test_sharpe_ratio_low_volatility(self):
        """Very low volatility produces extreme Sharpe ratios."""
        from ith_python.ith import sharpe_ratio_numba

        # Constant returns produce near-zero volatility
        # Due to floating-point precision, std != exactly 0
        returns = np.array([0.01] * 100)
        sr = sharpe_ratio_numba(returns, "1d", "crypto")
        # Result is either NaN (if std==0) or very large (if std near 0)
        assert np.isnan(sr) or abs(sr) > 1e10

    def test_sharpe_ratio_granularity_daily(self):
        """Should handle daily granularity."""
        from ith_python.ith import sharpe_ratio_numba

        returns = np.random.randn(100) * 0.01 + 0.001
        sr = sharpe_ratio_numba(returns, "1d", "crypto")
        assert isinstance(sr, (float, np.floating))

    def test_sharpe_ratio_granularity_minute(self):
        """Should handle minute granularity."""
        from ith_python.ith import sharpe_ratio_numba

        returns = np.random.randn(100) * 0.001
        sr = sharpe_ratio_numba(returns, "1m", "crypto")
        assert isinstance(sr, (float, np.floating))

    def test_sharpe_ratio_invalid_granularity(self):
        """Invalid granularity should raise ValueError."""
        from ith_python.ith import sharpe_ratio_numba

        returns = np.random.randn(100) * 0.01
        with pytest.raises(ValueError):
            sharpe_ratio_numba(returns, "invalid", "crypto")

    def test_sharpe_ratio_market_types(self):
        """Different market types should use different trading days."""
        from ith_python.ith import sharpe_ratio_numba

        returns = np.random.randn(100) * 0.01 + 0.001
        sr_crypto = sharpe_ratio_numba(returns, "1d", "crypto")  # 365 days
        sr_other = sharpe_ratio_numba(returns, "1d", "stocks")  # 252 days

        # Crypto should have higher annualized SR due to more trading days
        # (assuming positive returns)
        if sr_crypto > 0 and sr_other > 0:
            assert sr_crypto > sr_other


class TestSharpeRatioTimeAgnostic:
    """Tests for time-agnostic sharpe_ratio() function."""

    def test_sharpe_ratio_explicit_periods(self):
        """Test Sharpe with explicit periods_per_year."""
        from ith_python.ith import sharpe_ratio

        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.02] * 20)
        sr = sharpe_ratio(returns, periods_per_year=252)
        assert sr > 0
        assert isinstance(sr, (float, np.floating))

    def test_sharpe_ratio_custom_periods(self):
        """Test Sharpe with custom period count (e.g., range bars)."""
        from ith_python.ith import sharpe_ratio

        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.02] * 20)
        sr_252 = sharpe_ratio(returns, periods_per_year=252)
        sr_500 = sharpe_ratio(returns, periods_per_year=500)
        # More periods = higher annualized Sharpe (for positive returns)
        assert sr_500 > sr_252

    def test_sharpe_ratio_different_frequencies(self):
        """Test Sharpe ratio scales correctly with different frequencies."""
        from ith_python.ith import sharpe_ratio

        returns = np.array([0.01] * 100 + [0.02] * 100)  # Positive returns
        sr_daily = sharpe_ratio(returns, periods_per_year=252)
        sr_hourly = sharpe_ratio(returns, periods_per_year=252 * 24)

        # Hourly should scale up by sqrt(24) approximately
        assert sr_hourly > sr_daily

    def test_sharpe_ratio_deprecation_warning(self):
        """Verify deprecation warning for old API."""
        import warnings
        from ith_python.ith import sharpe_ratio_numba

        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.02] * 20)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sharpe_ratio_numba(returns, "1d", "crypto")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_sharpe_ratio_with_risk_free_rate(self):
        """Test Sharpe ratio with non-zero risk-free rate."""
        from ith_python.ith import sharpe_ratio

        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.02] * 20)
        sr_no_rf = sharpe_ratio(returns, periods_per_year=252, rf=0.0)
        sr_with_rf = sharpe_ratio(returns, periods_per_year=252, rf=0.001)
        # With positive risk-free rate, Sharpe should be lower
        assert sr_with_rf < sr_no_rf


class TestMaxDrawdown:
    """Tests for max drawdown calculation."""

    def test_max_drawdown_uptrend(self):
        """Pure uptrend should have zero drawdown."""
        from ith_python.ith import max_drawdown

        nav = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        result = max_drawdown(nav)
        assert result.max_drawdown == 0.0

    def test_max_drawdown_downtrend(self):
        """Downtrend should have positive drawdown."""
        from ith_python.ith import max_drawdown

        nav = np.array([1.0, 0.9, 0.8, 0.7])
        result = max_drawdown(nav)
        assert result.max_drawdown > 0
        assert result.max_drawdown == pytest.approx(0.3, rel=0.01)  # 30% drawdown

    def test_max_drawdown_recovery(self):
        """Recovery should not affect max drawdown."""
        from ith_python.ith import max_drawdown

        nav = np.array([1.0, 1.1, 0.9, 1.2])  # Dips then recovers
        result = max_drawdown(nav)
        expected = 1 - 0.9 / 1.1  # Max drawdown from peak of 1.1 to trough of 0.9
        assert result.max_drawdown == pytest.approx(expected, rel=0.01)


class TestBullIthConfig:
    """Tests for BullIthConfig defaults."""

    def test_default_config_values(self):
        """BullIthConfig should have sensible defaults."""
        from ith_python.ith import BullIthConfig

        config = BullIthConfig()

        assert config.delete_everything is False
        assert config.TMAEG == 0.05
        assert config.bull_epochs_lower_bound == 5  # Updated for batch generation optimization
        assert config.sr_lower_bound == 0.5
        assert config.sr_upper_bound == 9.9
        assert config.analysis_n_points is None  # Default is date-based mode

    def test_config_is_immutable(self):
        """NamedTuple config should be immutable."""
        from ith_python.ith import BullIthConfig

        config = BullIthConfig()
        with pytest.raises(AttributeError):
            config.TMAEG = 0.10

    def test_backwards_compatibility_alias(self):
        """IthConfig alias should work for backwards compatibility."""
        from ith_python.ith import IthConfig, BullIthConfig

        # IthConfig should be an alias for BullIthConfig
        assert IthConfig is BullIthConfig


class TestSyntheticNavParams:
    """Tests for SyntheticNavParams configuration."""

    def test_default_params(self):
        """SyntheticNavParams should have expected defaults."""
        from ith_python.ith import SyntheticNavParams

        params = SyntheticNavParams()

        assert params.start_date == "2020-01-30"
        assert params.end_date == "2023-07-25"
        assert params.avg_period_return > 0
        assert params.period_return_volatility > 0
        assert 0 < params.drawdown_prob < 1
        assert params.n_points is None  # Default is date-based mode


class TestGenerateSyntheticNav:
    """Tests for synthetic NAV generation in main module."""

    def test_generates_with_pnl(self):
        """Generated NAV should include PnL column."""
        from ith_python.ith import generate_synthetic_nav, SyntheticNavParams

        params = SyntheticNavParams(
            start_date="2020-01-01",
            end_date="2020-01-31",
        )
        nav = generate_synthetic_nav(params)

        assert "NAV" in nav.columns
        assert "PnL" in nav.columns

    def test_pnl_matches_nav_diff(self):
        """PnL should match NAV differences (except first value)."""
        from ith_python.ith import generate_synthetic_nav, SyntheticNavParams

        params = SyntheticNavParams(
            start_date="2020-01-01",
            end_date="2020-01-10",
        )
        nav = generate_synthetic_nav(params)

        # PnL from index 1 onwards should match NAV diff
        expected_pnl = nav["NAV"].diff().iloc[1:]
        actual_pnl = nav["PnL"].iloc[1:]
        np.testing.assert_array_almost_equal(expected_pnl.values, actual_pnl.values)

    def test_point_based_generation(self):
        """Test point-based synthetic NAV generation (time-agnostic)."""
        from ith_python.ith import generate_synthetic_nav, SyntheticNavParams

        params = SyntheticNavParams(n_points=500)
        nav = generate_synthetic_nav(params)

        assert len(nav) == 500
        assert "NAV" in nav.columns
        assert "PnL" in nav.columns
        assert (nav["NAV"] > 0).all()  # NAV always positive

    def test_point_based_overrides_dates(self):
        """When n_points is set, it should override date range."""
        from ith_python.ith import generate_synthetic_nav, SyntheticNavParams

        # Set dates that would give ~365 days, but n_points=100 should override
        params = SyntheticNavParams(
            n_points=100,
            start_date="2020-01-01",
            end_date="2020-12-31",
        )
        nav = generate_synthetic_nav(params)

        assert len(nav) == 100  # n_points takes precedence


class TestLoadAndValidateCsv:
    """Tests for CSV loading and validation."""

    def test_load_valid_csv(self, sample_csv_file: Path):
        """Should successfully load valid CSV."""
        from ith_python.ith import load_and_validate_csv

        nav_data = load_and_validate_csv(sample_csv_file)

        assert isinstance(nav_data, pd.DataFrame)
        assert "NAV" in nav_data.columns
        assert "PnL" in nav_data.columns

    def test_load_csv_without_pnl(self, temp_dir: Path):
        """Should calculate PnL if missing."""
        from ith_python.ith import load_and_validate_csv

        # Create CSV without PnL column
        csv_path = temp_dir / "no_pnl.csv"
        df = pd.DataFrame(
            {"NAV": [1.0, 1.01, 1.02]},
            index=pd.date_range("2020-01-01", periods=3),
        )
        df.index.name = "Date"
        df.to_csv(csv_path)

        nav_data = load_and_validate_csv(csv_path)
        assert "PnL" in nav_data.columns

    def test_load_invalid_csv_no_nav(self, invalid_csv_file: Path):
        """Should raise ValueError for CSV without NAV column."""
        from ith_python.ith import load_and_validate_csv

        with pytest.raises(ValueError, match="No NAV column"):
            load_and_validate_csv(invalid_csv_file)

    def test_load_empty_csv(self, empty_csv_file: Path):
        """Should raise ValueError for empty CSV."""
        from ith_python.ith import load_and_validate_csv

        with pytest.raises(ValueError):
            load_and_validate_csv(empty_csv_file)
