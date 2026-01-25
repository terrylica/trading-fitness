"""Tests for distribution analysis."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from ith_python.statistical_examination.distribution import (
    analyze_all_distributions,
    analyze_distribution,
    compute_outlier_stats,
)


class TestDistributionAnalysis:
    """Tests for single feature distribution analysis."""

    def test_analyze_distribution_basic(self, sample_ith_features_df: pl.DataFrame):
        col = "ith_rb100_lb50_bull_ed"
        result = analyze_distribution(sample_ith_features_df, col)

        assert result["feature"] == col
        assert "mean" in result
        assert "std" in result
        assert "skewness" in result
        assert "kurtosis" in result
        assert "normality_test" in result
        assert "beta_fit" in result

    def test_analyze_distribution_quantiles(self, sample_ith_features_df: pl.DataFrame):
        result = analyze_distribution(sample_ith_features_df, "ith_rb100_lb50_bull_ed")

        assert "q01" in result
        assert "q50" in result
        assert "q99" in result
        # Quantiles should be ordered (allowing for NaN due to warmup)
        if result["q01"] is not None and result["q50"] is not None and result["q99"] is not None:
            if not (np.isnan(result["q01"]) or np.isnan(result["q50"]) or np.isnan(result["q99"])):
                assert result["q01"] <= result["q50"] <= result["q99"]

    def test_analyze_distribution_shape(self, sample_ith_features_df: pl.DataFrame):
        result = analyze_distribution(sample_ith_features_df, "ith_rb100_lb50_bull_ed")

        assert "distribution_shape" in result
        shape = result["distribution_shape"]
        assert "shape" in shape
        assert shape["shape"] in ["symmetric", "right_skewed", "left_skewed", "uniform", "u_shaped", "bell_shaped", "unknown"]

    def test_analyze_distribution_missing_column(self, sample_ith_features_df: pl.DataFrame):
        result = analyze_distribution(sample_ith_features_df, "nonexistent_column")

        assert "error" in result

    def test_analyze_distribution_beta_fit(self, sample_ith_features_df: pl.DataFrame):
        result = analyze_distribution(sample_ith_features_df, "ith_rb100_lb50_bull_ed")

        beta_fit = result.get("beta_fit", {})
        if "error" not in beta_fit:
            assert "alpha" in beta_fit
            assert "beta" in beta_fit
            assert "fits_well" in beta_fit


class TestAllDistributions:
    """Tests for analyzing all distributions."""

    def test_analyze_all_distributions(self, sample_ith_features_df: pl.DataFrame):
        result = analyze_all_distributions(sample_ith_features_df)

        assert "results" in result
        assert "summary" in result
        assert len(result["results"]) > 0

    def test_analyze_all_with_filters(self, sample_ith_features_df: pl.DataFrame):
        result = analyze_all_distributions(
            sample_ith_features_df,
            threshold=100,
            lookback=50,
        )

        # Should only analyze features matching filters
        for r in result["results"]:
            if "error" not in r:
                assert "_rb100_" in r["feature"]
                assert "_lb50_" in r["feature"]

    def test_summary_statistics(self, sample_ith_features_df: pl.DataFrame):
        result = analyze_all_distributions(sample_ith_features_df)

        summary = result["summary"]
        assert "total_analyzed" in summary
        assert "gaussianity_distribution" in summary  # Updated: W as continuous metric
        assert "beta_fits_well" in summary
        assert "shapes" in summary


class TestOutlierStats:
    """Tests for outlier statistics computation."""

    def test_compute_outlier_stats(self, sample_ith_features_df: pl.DataFrame):
        result = compute_outlier_stats(sample_ith_features_df)

        assert isinstance(result, pl.DataFrame)
        assert "feature" in result.columns
        assert "vif" not in result.columns  # Wrong test - fix column name
        assert "q1" in result.columns
        assert "q3" in result.columns
        assert "n_outliers" in result.columns
        assert "outlier_rate" in result.columns

    def test_outlier_stats_bounds(self, sample_ith_features_df: pl.DataFrame):
        result = compute_outlier_stats(sample_ith_features_df)

        for row in result.iter_rows(named=True):
            assert row["q1"] <= row["q3"]
            assert row["lower_bound"] < row["upper_bound"]
            assert 0 <= row["outlier_rate"] <= 1
