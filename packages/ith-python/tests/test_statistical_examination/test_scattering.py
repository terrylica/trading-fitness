"""Tests for Scattering Transform feature extraction.

GitHub Issue: https://github.com/terrylica/cc-skills/issues/21
"""

import numpy as np
import polars as pl
import pytest

from ith_python.statistical_examination.scattering import (
    extract_scattering_features,
    extract_scattering_time_series,
    get_scattering_summary,
)


@pytest.fixture
def price_series() -> pl.DataFrame:
    """Create a simple price series for testing."""
    np.random.seed(42)
    n = 256  # Power of 2 for clean scattering

    # Random walk price series
    returns = np.random.randn(n) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))

    return pl.DataFrame({
        "bar_index": list(range(n)),
        "Close": prices.tolist(),
    })


@pytest.fixture
def long_price_series() -> pl.DataFrame:
    """Create a longer price series for time series extraction."""
    np.random.seed(42)
    n = 1024

    returns = np.random.randn(n) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))

    return pl.DataFrame({
        "bar_index": list(range(n)),
        "Close": prices.tolist(),
    })


class TestExtractScatteringFeatures:
    """Tests for extract_scattering_features function."""

    def test_basic_extraction(self, price_series: pl.DataFrame):
        """Basic scattering feature extraction works."""
        result = extract_scattering_features(price_series, price_col="Close", J=4, Q=4)

        assert isinstance(result, pl.DataFrame)
        assert result.height == 1  # Single row of features

        # Should have scattering path features
        path_cols = [c for c in result.columns if c.startswith("scat_path")]
        assert len(path_cols) > 0

        # Should have metadata columns
        assert "scat_n_paths" in result.columns
        assert "scat_J" in result.columns
        assert "scat_Q" in result.columns

    def test_missing_price_column_raises(self, price_series: pl.DataFrame):
        """Should raise ValueError for missing price column."""
        with pytest.raises(ValueError, match="not found"):
            extract_scattering_features(price_series, price_col="NonExistent")

    def test_insufficient_data_raises(self):
        """Should raise ValueError for insufficient data."""
        small_df = pl.DataFrame({
            "Close": [1.0, 2.0, 3.0],  # Only 3 samples
        })

        with pytest.raises(ValueError, match="Insufficient data"):
            extract_scattering_features(small_df)

    def test_normalization_option(self, price_series: pl.DataFrame):
        """Normalization option affects output."""
        result_norm = extract_scattering_features(
            price_series, normalize=True
        )
        result_no_norm = extract_scattering_features(
            price_series, normalize=False
        )

        # Results should differ with normalization
        # Get first path mean for comparison
        norm_val = result_norm["scat_path0_mean"].item()
        no_norm_val = result_no_norm["scat_path0_mean"].item()

        assert norm_val != no_norm_val

    def test_j_parameter_affects_output(self, price_series: pl.DataFrame):
        """Different J values produce different number of paths."""
        result_j4 = extract_scattering_features(price_series, J=4, Q=4)
        result_j6 = extract_scattering_features(price_series, J=6, Q=4)

        n_paths_j4 = result_j4["scat_n_paths"].item()
        n_paths_j6 = result_j6["scat_n_paths"].item()

        # More octaves = more paths
        assert n_paths_j6 >= n_paths_j4

    def test_q_parameter_affects_output(self, price_series: pl.DataFrame):
        """Different Q values produce different number of paths."""
        result_q1 = extract_scattering_features(price_series, J=4, Q=1)
        result_q4 = extract_scattering_features(price_series, J=4, Q=4)

        n_paths_q1 = result_q1["scat_n_paths"].item()
        n_paths_q4 = result_q4["scat_n_paths"].item()

        # More wavelets per octave = more paths
        assert n_paths_q4 >= n_paths_q1

    def test_feature_statistics_present(self, price_series: pl.DataFrame):
        """Each path should have mean, std, max, min statistics."""
        result = extract_scattering_features(price_series, J=4, Q=4)

        # Check path 0 has all statistics
        assert "scat_path0_mean" in result.columns
        assert "scat_path0_std" in result.columns
        assert "scat_path0_max" in result.columns
        assert "scat_path0_min" in result.columns

    def test_handles_non_power_of_two(self):
        """Should handle non-power-of-2 length series."""
        np.random.seed(42)
        n = 200  # Not power of 2

        df = pl.DataFrame({
            "Close": np.random.randn(n).tolist(),
        })

        result = extract_scattering_features(df, J=4, Q=4)

        assert result.height == 1
        # T should be padded to nearest power of 2
        assert result["scat_T"].item() >= n


class TestExtractScatteringTimeSeries:
    """Tests for extract_scattering_time_series function."""

    def test_basic_time_series(self, long_price_series: pl.DataFrame):
        """Rolling window extraction produces multiple rows."""
        result = extract_scattering_time_series(
            long_price_series,
            price_col="Close",
            J=4,
            Q=4,
            window_size=256,
            stride=64,
        )

        assert isinstance(result, pl.DataFrame)
        assert result.height > 1  # Multiple windows
        assert "bar_index" in result.columns

    def test_bar_index_alignment(self, long_price_series: pl.DataFrame):
        """Bar indices should align with window ends."""
        result = extract_scattering_time_series(
            long_price_series,
            window_size=256,
            stride=128,
        )

        # First window ends at bar 255 (0-indexed)
        assert result["bar_index"][0] == 255

    def test_stride_affects_output_count(self, long_price_series: pl.DataFrame):
        """Smaller stride = more windows."""
        result_64 = extract_scattering_time_series(
            long_price_series, window_size=256, stride=64
        )
        result_128 = extract_scattering_time_series(
            long_price_series, window_size=256, stride=128
        )

        assert result_64.height > result_128.height

    def test_insufficient_data_raises(self):
        """Should raise for insufficient data."""
        small_df = pl.DataFrame({
            "Close": np.random.randn(100).tolist(),
        })

        with pytest.raises(ValueError, match="Insufficient data"):
            extract_scattering_time_series(small_df, window_size=256)

    def test_handles_nans_in_windows(self, long_price_series: pl.DataFrame):
        """Windows with NaNs should be skipped."""
        # Add NaNs to some rows
        df = long_price_series.with_columns(
            pl.when(pl.col("bar_index") < 50)
            .then(None)
            .otherwise(pl.col("Close"))
            .alias("Close")
        )

        result = extract_scattering_time_series(
            df, window_size=256, stride=64
        )

        # Should have fewer windows due to NaN skipping
        assert result.height > 0

    def test_creates_bar_index_if_missing(self):
        """Should create bar_index if not present."""
        np.random.seed(42)
        df = pl.DataFrame({
            "Close": np.random.randn(512).tolist(),
        })

        result = extract_scattering_time_series(df, window_size=256, stride=128)

        assert "bar_index" in result.columns

    def test_feature_naming(self, long_price_series: pl.DataFrame):
        """Features should have compact naming for time series."""
        result = extract_scattering_time_series(
            long_price_series, window_size=256, stride=128
        )

        # Time series uses scat_p{n} (compact) not scat_path{n}_mean
        scat_cols = [c for c in result.columns if c.startswith("scat_p")]
        assert len(scat_cols) > 0


class TestGetScatteringSummary:
    """Tests for get_scattering_summary function."""

    def test_returns_dict(self, price_series: pl.DataFrame):
        """Should return dictionary with summary."""
        summary = get_scattering_summary(price_series, J=4, Q=4)

        assert isinstance(summary, dict)
        assert summary["method"] == "ScatteringTransform"
        assert "J" in summary
        assert "Q" in summary
        assert "n_paths" in summary
        assert "n_features" in summary

    def test_handles_error_gracefully(self):
        """Should return error in summary for invalid data."""
        small_df = pl.DataFrame({"Close": [1.0, 2.0]})

        summary = get_scattering_summary(small_df)

        assert "error" in summary
        assert "Insufficient" in summary["error"]


class TestScatteringIntegration:
    """Integration tests for scattering transform."""

    def test_output_compatible_with_polars_join(self, price_series: pl.DataFrame):
        """Time series output can be joined with original data."""
        # Create longer series for time series extraction
        np.random.seed(42)
        n = 512
        df = pl.DataFrame({
            "bar_index": list(range(n)),
            "Close": (100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))).tolist(),
        })

        scat_features = extract_scattering_time_series(
            df, window_size=256, stride=64
        )

        # Should be joinable
        joined = df.join(scat_features, on="bar_index", how="left")

        assert joined.height == n
        assert "scat_p0" in joined.columns

    def test_deterministic_output(self, price_series: pl.DataFrame):
        """Same input should produce same output."""
        result1 = extract_scattering_features(price_series, J=4, Q=4)
        result2 = extract_scattering_features(price_series, J=4, Q=4)

        # Compare first path mean
        assert result1["scat_path0_mean"].item() == result2["scat_path0_mean"].item()

    def test_different_price_columns(self):
        """Should work with different price column names."""
        np.random.seed(42)
        df = pl.DataFrame({
            "open": np.random.randn(256).tolist(),
            "high": np.random.randn(256).tolist(),
            "low": np.random.randn(256).tolist(),
            "close": np.random.randn(256).tolist(),
        })

        result = extract_scattering_features(df, price_col="close", J=4, Q=4)

        assert result.height == 1
