"""Validate ITH features against real range bar data from rangebar-py.

This module implements the "Symmetric Dogfooding" pattern:
- trading-fitness exports ITH metrics (PyO3)
- rangebar-py exports range bar construction
- This test validates ITH features work correctly with real range bar data
- rangebar-py (separately) validates its features work with trading-fitness metrics

Related Patterns:
- Dogfooding: using your own product
- Consumer-Driven Contract Testing: validating consumer expectations
- Cross-repo testing: polyrepo integration testing

Note: Tests that require rangebar are skipped if it's not installed.
Edge case and API tests always run.
"""

import numpy as np
import pytest

# Check if rangebar is available (for conditional skipping)
try:
    import rangebar  # noqa: F401 - used to check availability
    HAS_RANGEBAR = True
except ImportError:
    HAS_RANGEBAR = False

requires_rangebar = pytest.mark.skipif(
    not HAS_RANGEBAR,
    reason="rangebar not installed (install with: uv sync --extra validation)"
)


@requires_rangebar
class TestRollingIthWithRealRangeBars:
    """Validate ITH features using REAL range bar data from rangebar-py.

    This is the core of Symmetric Dogfooding - we use the actual consumer
    package (rangebar) to generate real data, not synthetic mocks.
    """

    @pytest.fixture
    def real_btc_range_bars(self):
        """Fetch real BTCUSDT range bars from rangebar-py.

        Uses rangebar's actual data construction with Binance aggTrades.
        This is the "real consumer data" the symmetric dogfooding pattern requires.

        FAILS LOUDLY if data cannot be fetched - no synthetic fallback.
        Symmetric dogfooding requires REAL data validation.
        """
        from rangebar import get_n_range_bars

        # Use rangebar's get_n_range_bars to fetch real data
        # This fetches from Binance aggTrades or local ClickHouse cache
        try:
            bars = get_n_range_bars(
                symbol="BTCUSDT",
                n_bars=500,
                threshold_decimal_bps=1000,  # 1000 dbps minimum for crypto
            )
        except ConnectionError as exc:
            pytest.fail(
                f"SYMMETRIC DOGFOODING FAILED: Cannot connect to data source.\n"
                f"Error: {exc}\n"
                f"Check: Binance API access or ClickHouse cache availability."
            )
        except TimeoutError as exc:
            pytest.fail(
                f"SYMMETRIC DOGFOODING FAILED: Data fetch timed out.\n"
                f"Error: {exc}\n"
                f"Check: Network connectivity or increase timeout."
            )
        except OSError as exc:
            pytest.fail(
                f"SYMMETRIC DOGFOODING FAILED: OS-level error during data fetch.\n"
                f"Error: {exc}\n"
                f"Check: DNS resolution, firewall, or disk access."
            )

        if bars is None or len(bars) == 0:
            pytest.fail(
                "SYMMETRIC DOGFOODING FAILED: No data returned from rangebar.\n"
                "Check: Symbol validity, date range, or ClickHouse cache state."
            )

        return bars

    @pytest.fixture
    def real_nav_from_rangebar(self, real_btc_range_bars):
        """Convert rangebar output to NAV for ITH computation."""
        closes = real_btc_range_bars["Close"].to_numpy()
        returns = np.diff(closes) / closes[:-1]
        nav = np.concatenate([[1.0], np.cumprod(1 + returns)])
        return nav

    def test_ith_bounded_on_real_rangebar_data(self, real_nav_from_rangebar):
        """All ITH features must be bounded [0, 1] on REAL range bar data.

        This is the key symmetric dogfooding test - validates our features
        work correctly with the actual consumer's data construction.
        """
        from trading_fitness_metrics import compute_rolling_ith

        # Use dynamic lookback based on available data (max 50% of data length)
        nav_len = len(real_nav_from_rangebar)
        lookback = min(100, max(10, nav_len // 2))

        if nav_len < 20:
            pytest.skip(f"Insufficient data for test: {nav_len} bars (need >= 20)")

        features = compute_rolling_ith(real_nav_from_rangebar, lookback=lookback)

        feature_names = [
            "bull_epoch_density", "bear_epoch_density",
            "bull_excess_gain", "bear_excess_gain",
            "bull_cv", "bear_cv",
            "max_drawdown", "max_runup",
        ]

        for name in feature_names:
            arr = getattr(features, name)
            valid = arr[~np.isnan(arr)]
            assert np.all(
                (valid >= 0) & (valid <= 1)
            ), f"{name} not bounded [0, 1]: min={valid.min()}, max={valid.max()}"


@requires_rangebar
class TestRollingIthWithSyntheticRangeBars:
    """Validate ITH features using synthetic range bar-like data.

    These tests use synthetic data that mimics range bar characteristics.
    They run even when real rangebar data is unavailable.
    """

    @pytest.fixture
    def synthetic_range_bar_nav(self):
        """Generate synthetic NAV data that simulates range bar characteristics.

        Range bars have specific properties:
        - Uniform price movement per bar (by definition)
        - Non-uniform time spacing
        - Volume clustering during volatility

        We simulate this by generating returns with varying volatility.
        """
        np.random.seed(42)
        n_bars = 500

        # Simulate range bar returns (bounded by range threshold)
        # In real range bars, each bar moves exactly the threshold amount
        threshold = 0.0025  # 25 bps
        returns = np.random.choice([-threshold, threshold], size=n_bars)

        # Add some noise to make it more realistic
        returns += np.random.randn(n_bars) * threshold * 0.1

        # Convert to NAV
        nav = np.cumprod(1 + returns)
        return nav

    def test_rolling_ith_bounded_on_synthetic_rangebar_data(self, synthetic_range_bar_nav):
        """All ITH features must be bounded [0, 1] on range bar-like data."""
        from trading_fitness_metrics import compute_rolling_ith

        # Note: TMAEG is auto-calculated from data volatility
        features = compute_rolling_ith(synthetic_range_bar_nav, lookback=100)

        # Validate all 8 feature arrays are bounded
        feature_names = [
            "bull_epoch_density",
            "bear_epoch_density",
            "bull_excess_gain",
            "bear_excess_gain",
            "bull_cv",
            "bear_cv",
            "max_drawdown",
            "max_runup",
        ]

        for name in feature_names:
            arr = getattr(features, name)
            valid = arr[~np.isnan(arr)]
            assert np.all(
                (valid >= 0) & (valid <= 1)
            ), f"{name} not bounded [0, 1]: min={valid.min()}, max={valid.max()}"

    def test_rolling_ith_nan_prefix(self, synthetic_range_bar_nav):
        """First lookback-1 values should be NaN."""
        from trading_fitness_metrics import compute_rolling_ith

        lookback = 100
        features = compute_rolling_ith(synthetic_range_bar_nav, lookback=lookback)

        # First lookback-1 values should be NaN
        assert np.all(np.isnan(features.bull_epoch_density[: lookback - 1]))
        assert np.all(np.isnan(features.bear_epoch_density[: lookback - 1]))

        # Value at lookback-1 should be valid
        assert not np.isnan(features.bull_epoch_density[lookback - 1])
        assert not np.isnan(features.bear_epoch_density[lookback - 1])

    def test_rolling_ith_with_different_lookbacks(self, synthetic_range_bar_nav):
        """ITH features should work with different lookback periods."""
        from trading_fitness_metrics import compute_rolling_ith

        lookbacks = [20, 50, 100, 200]

        for lookback in lookbacks:
            features = compute_rolling_ith(
                synthetic_range_bar_nav, lookback=lookback
            )

            # Check that we get valid features
            valid_bull = features.bull_epoch_density[lookback - 1 :]
            valid_bear = features.bear_epoch_density[lookback - 1 :]

            assert len(valid_bull) == len(synthetic_range_bar_nav) - lookback + 1
            assert np.all(~np.isnan(valid_bull)), f"NaN in valid region for lookback={lookback}"
            assert np.all(~np.isnan(valid_bear)), f"NaN in valid region for lookback={lookback}"

    def test_rolling_ith_with_varying_volatility(self, synthetic_range_bar_nav):
        """ITH features should adapt to data volatility (auto-TMAEG).

        The auto-TMAEG calculation adapts to the underlying data's volatility,
        ensuring sensible epoch density regardless of the bar type or instrument.
        """
        from trading_fitness_metrics import compute_rolling_ith, optimal_tmaeg

        features = compute_rolling_ith(synthetic_range_bar_nav, lookback=100)

        # All valid values should be bounded
        valid = features.bull_epoch_density[99:]
        assert np.all(
            (valid >= 0) & (valid <= 1)
        ), "Unbounded values with auto-TMAEG"

        # Verify we can inspect the auto-calculated TMAEG
        auto_tmaeg = optimal_tmaeg(synthetic_range_bar_nav, lookback=100)
        assert 0.0001 <= auto_tmaeg <= 0.50, f"Auto-TMAEG {auto_tmaeg} outside valid range"

    def test_rolling_ith_epoch_density_behavior(self, synthetic_range_bar_nav):
        """Shorter lookback should result in different epoch density patterns.

        With auto-TMAEG, the threshold adapts to data volatility.
        Shorter lookbacks have smaller windows, so TMAEG scales accordingly.
        """
        from trading_fitness_metrics import compute_rolling_ith, optimal_tmaeg

        features_short = compute_rolling_ith(
            synthetic_range_bar_nav, lookback=50
        )  # Shorter lookback
        features_long = compute_rolling_ith(
            synthetic_range_bar_nav, lookback=200
        )  # Longer lookback

        # TMAEG should be different for different lookbacks
        tmaeg_short = optimal_tmaeg(synthetic_range_bar_nav, lookback=50)
        tmaeg_long = optimal_tmaeg(synthetic_range_bar_nav, lookback=200)

        # Longer lookback = larger window = higher TMAEG (due to sqrt scaling)
        assert tmaeg_long > tmaeg_short, "TMAEG should scale with sqrt(lookback)"

        # Both should produce valid bounded features
        valid_short = features_short.bull_epoch_density[49:]
        valid_long = features_long.bull_epoch_density[199:]

        assert np.all((valid_short >= 0) & (valid_short <= 1))
        assert np.all((valid_long >= 0) & (valid_long <= 1))


class TestRollingIthEdgeCases:
    """Test edge cases for rolling ITH computation."""

    def test_minimum_nav_length(self):
        """Should handle minimum valid NAV length."""
        from trading_fitness_metrics import compute_rolling_ith

        nav = np.array([1.0, 1.01, 1.02])
        features = compute_rolling_ith(nav, lookback=2)

        # First value should be NaN, rest should be valid
        assert np.isnan(features.bull_epoch_density[0])
        assert not np.isnan(features.bull_epoch_density[1])
        assert not np.isnan(features.bull_epoch_density[2])

    def test_constant_nav(self):
        """Should handle constant NAV (no movement)."""
        from trading_fitness_metrics import compute_rolling_ith

        nav = np.ones(100)
        features = compute_rolling_ith(nav, lookback=20)

        # Should produce valid (though potentially zero) features
        valid = features.bull_epoch_density[19:]
        assert np.all(~np.isnan(valid))
        assert np.all((valid >= 0) & (valid <= 1))

        # With no movement, drawdown and runup should be 0
        valid_dd = features.max_drawdown[19:]
        valid_ru = features.max_runup[19:]
        assert np.allclose(valid_dd, 0.0)
        assert np.allclose(valid_ru, 0.0)

    def test_pure_uptrend(self):
        """Should have low drawdown in pure uptrend."""
        from trading_fitness_metrics import compute_rolling_ith

        nav = np.cumprod(np.ones(100) * 1.01)  # 1% gain each bar
        features = compute_rolling_ith(nav, lookback=20)

        # Max drawdown should be very low or zero
        valid_dd = features.max_drawdown[19:]
        assert np.all(valid_dd < 0.01), "Unexpected drawdown in pure uptrend"

    def test_pure_downtrend(self):
        """Should have low runup in pure downtrend."""
        from trading_fitness_metrics import compute_rolling_ith

        nav = np.cumprod(np.ones(100) * 0.99)  # 1% loss each bar
        features = compute_rolling_ith(nav, lookback=20)

        # Max runup should be very low or zero
        valid_ru = features.max_runup[19:]
        assert np.all(valid_ru < 0.01), "Unexpected runup in pure downtrend"


class TestRollingIthPythonAPI:
    """Test the Python API ergonomics."""

    def test_feature_array_length(self):
        """Feature arrays should match input length."""
        from trading_fitness_metrics import compute_rolling_ith

        n = 500
        nav = np.cumprod(1 + np.random.randn(n) * 0.01)
        features = compute_rolling_ith(nav, lookback=100)

        assert len(features) == n
        assert len(features.bull_epoch_density) == n
        assert len(features.bear_epoch_density) == n
        assert len(features.bull_excess_gain) == n
        assert len(features.bear_excess_gain) == n
        assert len(features.bull_cv) == n
        assert len(features.bear_cv) == n
        assert len(features.max_drawdown) == n
        assert len(features.max_runup) == n

    def test_invalid_inputs(self):
        """Should raise ValueError for invalid inputs."""
        from trading_fitness_metrics import compute_rolling_ith

        nav = np.array([1.0, 1.01, 1.02])

        # Empty nav
        with pytest.raises(ValueError, match="empty"):
            compute_rolling_ith(np.array([]), lookback=2)

        # Zero lookback
        with pytest.raises(ValueError, match="positive"):
            compute_rolling_ith(nav, lookback=0)

        # Lookback too large
        with pytest.raises(ValueError, match="exceed"):
            compute_rolling_ith(nav, lookback=10)

    def test_dataframe_integration(self):
        """Features should integrate smoothly with pandas DataFrames."""
        import pandas as pd
        from trading_fitness_metrics import compute_rolling_ith

        # Simulate range bar-like data
        np.random.seed(42)
        n = 200
        nav = np.cumprod(1 + np.random.randn(n) * 0.01)

        features = compute_rolling_ith(nav, lookback=50)

        # Add features to DataFrame
        df = pd.DataFrame({"nav": nav})
        df["bull_epoch_density"] = features.bull_epoch_density
        df["bear_epoch_density"] = features.bear_epoch_density
        df["max_drawdown"] = features.max_drawdown

        # Verify DataFrame operations work
        assert df["bull_epoch_density"].isna().sum() == 49  # First 49 are NaN
        assert df["bull_epoch_density"].dropna().between(0, 1).all()
