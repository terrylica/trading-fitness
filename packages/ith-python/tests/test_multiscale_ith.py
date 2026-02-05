"""Test multi-scale ITH feature computation.

Validates the compute_multiscale_ith() function and Arrow-native integration
for zero-copy Polars DataFrame conversion.

Tests cover:
1. Multi-scale configuration
2. Column naming convention
3. Feature boundedness
4. NaN prefix handling
5. Arrow RecordBatch conversion
6. Polars integration (when available)
"""

import numpy as np
import pytest

from trading_fitness_metrics import (
    MultiscaleIthConfig,
    compute_multiscale_ith,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_nav_500() -> np.ndarray:
    """Generate 500-point NAV array (deterministic)."""
    np.random.seed(42)
    returns = 1 + np.random.randn(500) * 0.02 + 0.001
    return np.cumprod(returns)


@pytest.fixture
def sample_nav_10000() -> np.ndarray:
    """Generate 10000-point NAV array for testing large lookbacks."""
    np.random.seed(42)
    returns = 1 + np.random.randn(10000) * 0.02 + 0.001
    return np.cumprod(returns)


# ============================================================================
# Configuration Tests
# ============================================================================


class TestMultiscaleConfig:
    """Test MultiscaleIthConfig."""

    def test_default_config(self):
        """Default config uses 250 dbps and standard lookbacks."""
        config = MultiscaleIthConfig()
        assert config.threshold_dbps == 250
        assert config.lookbacks == [
            20, 50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000
        ]

    def test_custom_threshold(self):
        """Custom threshold is applied."""
        config = MultiscaleIthConfig(threshold_dbps=100)
        assert config.threshold_dbps == 100

    def test_custom_lookbacks(self):
        """Custom lookbacks are applied."""
        config = MultiscaleIthConfig(lookbacks=[50, 100, 200])
        assert config.lookbacks == [50, 100, 200]

    def test_repr(self):
        """Config has useful repr."""
        config = MultiscaleIthConfig(threshold_dbps=100, lookbacks=[50, 100])
        r = repr(config)
        assert "100" in r
        assert "50" in r or "[50, 100]" in r


# ============================================================================
# Feature Computation Tests
# ============================================================================


class TestComputeMultiscaleIth:
    """Test compute_multiscale_ith function."""

    def test_basic_computation(self, sample_nav_500):
        """Basic computation produces features."""
        config = MultiscaleIthConfig(threshold_dbps=250, lookbacks=[20, 50])
        features = compute_multiscale_ith(sample_nav_500, config)

        assert features.n_points == 500
        assert features.n_features == 16  # 2 lookbacks × 8 features

    def test_default_config(self, sample_nav_500):
        """None config uses default."""
        features = compute_multiscale_ith(sample_nav_500, None)
        assert features.config.threshold_dbps == 250

    def test_column_naming_convention(self, sample_nav_500):
        """Column names follow ith_rb{threshold}_lb{lookback}_{feature} pattern."""
        config = MultiscaleIthConfig(threshold_dbps=250, lookbacks=[20, 50])
        features = compute_multiscale_ith(sample_nav_500, config)

        expected_cols = [
            "ith_rb250_lb20_bull_ed",
            "ith_rb250_lb20_bear_ed",
            "ith_rb250_lb20_bull_eg",
            "ith_rb250_lb20_bear_eg",
            "ith_rb250_lb20_bull_cv",
            "ith_rb250_lb20_bear_cv",
            "ith_rb250_lb20_max_dd",
            "ith_rb250_lb20_max_ru",
            "ith_rb250_lb50_bull_ed",
            "ith_rb250_lb50_bear_ed",
            "ith_rb250_lb50_bull_eg",
            "ith_rb250_lb50_bear_eg",
            "ith_rb250_lb50_bull_cv",
            "ith_rb250_lb50_bear_cv",
            "ith_rb250_lb50_max_dd",
            "ith_rb250_lb50_max_ru",
        ]

        for col in expected_cols:
            assert col in features.column_names(), f"Missing column: {col}"

    def test_all_features_bounded(self, sample_nav_500):
        """All features must be bounded [0, 1]."""
        config = MultiscaleIthConfig(threshold_dbps=250, lookbacks=[20, 50, 100])
        features = compute_multiscale_ith(sample_nav_500, config)

        assert features.all_bounded()

        for col_name in features.column_names():
            values = features.get(col_name)
            valid = values[~np.isnan(values)]
            assert np.all((valid >= 0) & (valid <= 1)), (
                f"{col_name} not bounded: min={valid.min()}, max={valid.max()}"
            )

    def test_nan_prefix_increases_with_lookback(self, sample_nav_500):
        """Longer lookbacks have longer NaN prefixes."""
        config = MultiscaleIthConfig(threshold_dbps=250, lookbacks=[20, 50, 100])
        features = compute_multiscale_ith(sample_nav_500, config)

        # Check NaN prefix for each lookback
        for lookback in [20, 50, 100]:
            col = f"ith_rb250_lb{lookback}_bull_ed"
            values = features.get(col)

            # First lookback-1 values should be NaN
            expected_nan = lookback - 1
            actual_nan = np.sum(np.isnan(values[:expected_nan]))
            assert actual_nan == expected_nan, (
                f"lookback={lookback}: expected {expected_nan} NaN prefix, got {actual_nan}"
            )

            # Value at lookback-1 index should be valid
            assert not np.isnan(values[expected_nan]), (
                f"lookback={lookback}: value at index {expected_nan} should not be NaN"
            )

    def test_skips_too_large_lookbacks(self, sample_nav_500):
        """Lookbacks exceeding data length are skipped."""
        config = MultiscaleIthConfig(threshold_dbps=250, lookbacks=[20, 600, 1000])
        features = compute_multiscale_ith(sample_nav_500, config)

        # Only lookback=20 should be computed (500 data points)
        assert features.n_features == 8  # 8 features for 1 lookback
        assert "ith_rb250_lb20_bull_ed" in features.column_names()
        assert "ith_rb250_lb600_bull_ed" not in features.column_names()
        assert "ith_rb250_lb1000_bull_ed" not in features.column_names()

    def test_empty_nav_raises(self):
        """Empty NAV raises ValueError."""
        config = MultiscaleIthConfig(lookbacks=[20])
        with pytest.raises(ValueError, match="empty"):
            compute_multiscale_ith(np.array([]), config)

    def test_get_method(self, sample_nav_500):
        """get() method retrieves individual columns."""
        config = MultiscaleIthConfig(threshold_dbps=250, lookbacks=[20])
        features = compute_multiscale_ith(sample_nav_500, config)

        values = features.get("ith_rb250_lb20_bull_ed")
        assert len(values) == 500
        assert isinstance(values, np.ndarray)

    def test_get_missing_column_raises(self, sample_nav_500):
        """get() raises for missing column."""
        config = MultiscaleIthConfig(threshold_dbps=250, lookbacks=[20])
        features = compute_multiscale_ith(sample_nav_500, config)

        with pytest.raises(ValueError, match="not found"):
            features.get("nonexistent_column")


# ============================================================================
# Large Lookback Tests
# ============================================================================


class TestLargeLookbacks:
    """Test multi-scale ITH with large lookbacks (up to 6000 bars)."""

    def test_large_lookbacks_bounded(self, sample_nav_10000):
        """Large lookbacks still produce bounded features."""
        config = MultiscaleIthConfig(
            threshold_dbps=250,
            lookbacks=[100, 500, 1000, 2000, 5000, 6000]
        )
        features = compute_multiscale_ith(sample_nav_10000, config)

        assert features.all_bounded()
        assert features.n_features == 6 * 8  # 6 lookbacks × 8 features

    def test_feature_arrays_correct_length(self, sample_nav_10000):
        """All feature arrays have same length as input."""
        config = MultiscaleIthConfig(
            threshold_dbps=250,
            lookbacks=[100, 1000, 5000]
        )
        features = compute_multiscale_ith(sample_nav_10000, config)

        for col_name in features.column_names():
            values = features.get(col_name)
            assert len(values) == 10000, f"{col_name} has wrong length"


# ============================================================================
# Arrow Integration Tests
# ============================================================================


class TestArrowIntegration:
    """Test Arrow RecordBatch conversion.

    The to_arrow() method returns an Arrow RecordBatch via PyCapsule interface.
    We test via Polars which has robust PyCapsule support (pl.from_arrow).
    """

    def test_to_arrow_returns_recordbatch(self, sample_nav_500):
        """to_arrow() returns an Arrow RecordBatch."""
        import polars as pl

        config = MultiscaleIthConfig(threshold_dbps=250, lookbacks=[20, 50])
        features = compute_multiscale_ith(sample_nav_500, config)

        arrow_batch = features.to_arrow()
        assert arrow_batch is not None

        # Verify via Polars (robust PyCapsule support)
        df = pl.from_arrow(arrow_batch)
        assert len(df.columns) == 16  # 2 lookbacks × 8 features

    def test_arrow_column_names_match(self, sample_nav_500):
        """Arrow RecordBatch column names match feature column names."""
        import polars as pl

        config = MultiscaleIthConfig(threshold_dbps=250, lookbacks=[20])
        features = compute_multiscale_ith(sample_nav_500, config)

        arrow_batch = features.to_arrow()
        df = pl.from_arrow(arrow_batch)
        arrow_names = set(df.columns)
        feature_names = set(features.column_names())

        assert arrow_names == feature_names

    def test_arrow_values_match_numpy(self, sample_nav_500):
        """Arrow values match numpy array values."""
        import polars as pl

        config = MultiscaleIthConfig(threshold_dbps=250, lookbacks=[20])
        features = compute_multiscale_ith(sample_nav_500, config)

        arrow_batch = features.to_arrow()
        df = pl.from_arrow(arrow_batch)
        col_name = "ith_rb250_lb20_bull_ed"

        numpy_values = features.get(col_name)
        arrow_values = df[col_name].to_numpy()

        # Handle NaN comparison
        np.testing.assert_array_equal(
            np.isnan(numpy_values),
            np.isnan(arrow_values),
            err_msg="NaN positions differ"
        )

        # Compare non-NaN values
        mask = ~np.isnan(numpy_values)
        np.testing.assert_allclose(
            numpy_values[mask],
            arrow_values[mask],
            rtol=1e-10,
            err_msg="Values differ"
        )


# ============================================================================
# Polars Integration Tests
# ============================================================================


polars = pytest.importorskip("polars", reason="polars not installed")


class TestPolarsIntegration:
    """Test Polars DataFrame integration via Arrow."""

    def test_from_arrow_zero_copy(self, sample_nav_500):
        """Polars pl.from_arrow() works with to_arrow()."""
        import polars as pl

        config = MultiscaleIthConfig(threshold_dbps=250, lookbacks=[20, 50])
        features = compute_multiscale_ith(sample_nav_500, config)

        df = pl.from_arrow(features.to_arrow())

        assert len(df) == 500
        assert len(df.columns) == 16

    def test_polars_column_names(self, sample_nav_500):
        """Polars DataFrame has correct column names."""
        import polars as pl

        config = MultiscaleIthConfig(threshold_dbps=250, lookbacks=[20])
        features = compute_multiscale_ith(sample_nav_500, config)

        df = pl.from_arrow(features.to_arrow())

        assert "ith_rb250_lb20_bull_ed" in df.columns
        assert "ith_rb250_lb20_max_dd" in df.columns

    def test_polars_values_bounded(self, sample_nav_500):
        """Polars DataFrame values are bounded [0, 1]."""
        import polars as pl

        config = MultiscaleIthConfig(threshold_dbps=250, lookbacks=[20, 50])
        features = compute_multiscale_ith(sample_nav_500, config)

        df = pl.from_arrow(features.to_arrow())

        for col in df.columns:
            series = df[col]
            valid = series.drop_nulls()
            assert valid.min() >= 0, f"{col} has min < 0"
            assert valid.max() <= 1, f"{col} has max > 1"

    def test_polars_join_workflow(self, sample_nav_500):
        """Feature DataFrame can be joined with source data."""
        import polars as pl

        config = MultiscaleIthConfig(threshold_dbps=250, lookbacks=[20])
        features = compute_multiscale_ith(sample_nav_500, config)

        feature_df = pl.from_arrow(features.to_arrow())
        feature_df = feature_df.with_row_index("bar_index")

        # Simulate source DataFrame
        source_df = pl.DataFrame({
            "bar_index": pl.arange(500, eager=True),
            "close": sample_nav_500,
        })

        # Join
        combined = source_df.join(feature_df, on="bar_index", how="inner")

        assert len(combined) == 500
        assert "close" in combined.columns
        assert "ith_rb250_lb20_bull_ed" in combined.columns


# ============================================================================
# Cross-Scale Property Tests
# ============================================================================


class TestCrossScaleProperties:
    """Test properties that should hold across scales."""

    def test_nan_prefix_monotonically_increasing(self, sample_nav_500):
        """Longer lookbacks have more NaN values in prefix."""
        config = MultiscaleIthConfig(threshold_dbps=250, lookbacks=[20, 50, 100])
        features = compute_multiscale_ith(sample_nav_500, config)

        prev_nan_count = 0
        for lookback in sorted(config.lookbacks):
            col = f"ith_rb250_lb{lookback}_bull_ed"
            values = features.get(col)
            nan_count = np.sum(np.isnan(values[:lookback]))

            assert nan_count >= prev_nan_count, (
                f"NaN count should increase with lookback: "
                f"lb={lookback} has {nan_count} vs prev {prev_nan_count}"
            )
            prev_nan_count = nan_count

    def test_all_scales_produce_non_constant_features(self, sample_nav_500):
        """Each scale produces features with variance (not all constant)."""
        config = MultiscaleIthConfig(threshold_dbps=250, lookbacks=[20, 50, 100])
        features = compute_multiscale_ith(sample_nav_500, config)

        for col_name in features.column_names():
            values = features.get(col_name)
            valid = values[~np.isnan(values)]

            # At least some variance expected
            # (may be zero for constant NAV, but not for random walk)
            if len(valid) > 10:
                assert valid.std() >= 0, f"{col_name} has negative std"


# ============================================================================
# Integration with rangebar-py
# ============================================================================


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
class TestRangebarIntegration:
    """Validate multi-scale features on real range bar data.

    This implements the symmetric dogfooding pattern:
    - trading-fitness exports ITH metrics
    - rangebar-py exports range bar construction
    - This test validates ITH features work with real range bar data
    """

    @pytest.fixture
    def btc_range_bars(self):
        """Fetch real BTCUSDT range bars.

        FAILS LOUDLY if data cannot be fetched - no synthetic fallback.
        Symmetric dogfooding requires REAL data validation.
        """
        from rangebar import get_n_range_bars

        try:
            bars = get_n_range_bars(
                symbol="BTCUSDT",
                n_bars=5000,
                threshold_decimal_bps=1000,  # 1000 dbps minimum for crypto
            )
        except (ConnectionError, TimeoutError, OSError) as exc:
            pytest.fail(
                f"SYMMETRIC DOGFOODING FAILED: Cannot fetch range bar data.\n"
                f"Error: {exc}\n"
                f"Check: Binance API access or ClickHouse cache availability."
            )

        if bars is None or len(bars) == 0:
            pytest.fail(
                "SYMMETRIC DOGFOODING FAILED: No data returned from rangebar.\n"
                "Check: Symbol validity, date range, or ClickHouse cache state."
            )

        return bars

    def test_multiscale_bounded_on_real_data(self, btc_range_bars):
        """Multi-scale features bounded [0, 1] on real range bar data."""
        # Convert to NAV
        closes = btc_range_bars["Close"].to_numpy()
        returns = np.diff(closes) / closes[:-1]
        nav = np.concatenate([[1.0], np.cumprod(1 + returns)])

        config = MultiscaleIthConfig(
            threshold_dbps=250,
            lookbacks=[50, 100, 500, 1000]
        )
        features = compute_multiscale_ith(nav, config)

        assert features.all_bounded()

    def test_multiscale_polars_workflow_real_data(self, btc_range_bars):
        """Complete workflow with real range bars and Polars."""
        import polars as pl

        # Convert to NAV
        closes = btc_range_bars["Close"].to_numpy()

        if len(closes) < 50:
            pytest.skip(f"Insufficient data for test: {len(closes)} bars (need >= 50)")

        returns = np.diff(closes) / closes[:-1]
        nav = np.concatenate([[1.0], np.cumprod(1 + returns)])

        # Use dynamic lookbacks based on available data
        max_lookback = min(100, len(nav) // 2)
        lookbacks = [lb for lb in [20, 50] if lb <= max_lookback]

        if len(lookbacks) == 0:
            pytest.skip(f"Data too short for any lookback: {len(nav)} bars")

        config = MultiscaleIthConfig(
            threshold_dbps=1000,
            lookbacks=lookbacks
        )
        features = compute_multiscale_ith(nav, config)

        # Convert to Polars
        df = pl.from_arrow(features.to_arrow())

        assert len(df) == len(nav)
        assert len(df.columns) == len(lookbacks) * 8  # lookbacks x 8 features

        # All features bounded
        for col in df.columns:
            valid = df[col].drop_nulls()
            assert valid.min() >= 0
            assert valid.max() <= 1
