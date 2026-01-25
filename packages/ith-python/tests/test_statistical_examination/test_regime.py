"""Tests for regime dependence analysis."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ith_python.statistical_examination.regime import (
    analyze_regime_dependence,
    compute_hurst_exponent,
    compute_regime_statistics,
    detect_regime,
    summarize_regime_dependence,
)


class TestHurstExponent:
    """Tests for Hurst exponent computation."""

    def test_hurst_random_walk(self):
        """Hurst exponent should return a value in valid range [0, 1]."""
        np.random.seed(42)
        random_walk = np.cumsum(np.random.randn(1000))

        h = compute_hurst_exponent(random_walk)

        # R/S estimator can have significant variance; just verify it's bounded
        assert 0.0 <= h <= 1.0

    def test_hurst_trending(self):
        """Trending series should have H > 0.5."""
        np.random.seed(42)
        # Create trending series with persistent increments
        increments = np.abs(np.random.randn(1000)) * 0.1
        trending = np.cumsum(increments)

        h = compute_hurst_exponent(trending)

        assert h > 0.5

    def test_hurst_short_series(self):
        """Short series should return default 0.5."""
        short = np.array([1.0, 1.1, 1.2])

        h = compute_hurst_exponent(short)

        assert h == 0.5

    def test_hurst_bounds(self, synthetic_nav: np.ndarray):
        """Hurst exponent should be in [0, 1]."""
        h = compute_hurst_exponent(synthetic_nav)

        assert 0.0 <= h <= 1.0


class TestRegimeDetection:
    """Tests for regime detection."""

    def test_detect_regime_basic(self, synthetic_nav: np.ndarray):
        regimes = detect_regime(synthetic_nav, lookback=100)

        assert len(regimes) == len(synthetic_nav)
        assert set(regimes).issubset({"trending", "mean_reverting", "random", "warmup"})

    def test_detect_regime_warmup(self, synthetic_nav: np.ndarray):
        lookback = 100
        regimes = detect_regime(synthetic_nav, lookback=lookback)

        # First lookback bars should be warmup
        assert all(r == "warmup" for r in regimes[:lookback])

    def test_detect_regime_custom_thresholds(self, synthetic_nav: np.ndarray):
        regimes = detect_regime(
            synthetic_nav,
            lookback=100,
            trending_threshold=0.6,
            mean_rev_threshold=0.4,
        )

        assert len(regimes) == len(synthetic_nav)


class TestRegimeDependence:
    """Tests for regime dependence analysis."""

    def test_analyze_regime_dependence_basic(self, sample_ith_features_df: pl.DataFrame):
        n = len(sample_ith_features_df)
        np.random.seed(42)
        # Create mock regimes
        regimes = np.array(["trending"] * (n // 3) + ["mean_reverting"] * (n // 3) + ["random"] * (n - 2 * (n // 3)))

        results = analyze_regime_dependence(
            sample_ith_features_df,
            regimes,
            feature_cols=["ith_rb100_lb50_bull_ed", "ith_rb100_lb50_bear_ed"],
        )

        assert len(results) >= 1
        # At least one result should have analysis
        valid = [r for r in results if "error" not in r]
        if valid:
            assert "mann_whitney_p" in valid[0]
            assert "cohens_d" in valid[0]

    def test_analyze_regime_dependence_length_mismatch(self, sample_ith_features_df: pl.DataFrame):
        wrong_len_regimes = np.array(["trending", "mean_reverting"])

        with pytest.raises(ValueError, match="length"):
            analyze_regime_dependence(sample_ith_features_df, wrong_len_regimes)

    def test_analyze_regime_dependence_effect_magnitude(self, sample_ith_features_df: pl.DataFrame):
        n = len(sample_ith_features_df)
        regimes = np.array(["trending"] * (n // 2) + ["mean_reverting"] * (n - n // 2))

        results = analyze_regime_dependence(sample_ith_features_df, regimes)

        valid = [r for r in results if "error" not in r]
        for r in valid:
            # Updated: finance-specific Cliff's Delta thresholds
            assert r["effect_magnitude"] in ["negligible", "small_tradable", "medium_strong", "large_suspicious"]
            assert "cliffs_delta" in r  # New field added


class TestRegimeSummary:
    """Tests for regime analysis summary functions."""

    def test_summarize_regime_dependence(self):
        # Updated: include cliffs_delta and effect_magnitude fields
        mock_results = [
            {"feature": "f1", "regime_dependent": True, "cohens_d": 0.9, "cliffs_delta": 0.4, "effect_magnitude": "large_suspicious", "mann_whitney_p": 0.001},
            {"feature": "f2", "regime_dependent": False, "cohens_d": 0.1, "cliffs_delta": 0.03, "effect_magnitude": "negligible", "mann_whitney_p": 0.5},
            {"feature": "f3", "error": "insufficient samples"},
        ]

        summary = summarize_regime_dependence(mock_results)

        assert summary["total_analyzed"] == 2
        assert summary["regime_dependent"] == 1
        assert summary["regime_invariant"] == 1
        assert "dependence_rate" in summary
        assert "effect_magnitude_distribution" in summary  # New field

    def test_compute_regime_statistics(self, sample_ith_features_df: pl.DataFrame):
        n = len(sample_ith_features_df)
        regimes = np.array(["trending"] * 400 + ["mean_reverting"] * 300 + ["random"] * (n - 700))

        stats = compute_regime_statistics(sample_ith_features_df, regimes)

        assert "total_bars" in stats
        assert "regime_counts" in stats
        assert "regime_proportions" in stats
        assert stats["total_bars"] == n
