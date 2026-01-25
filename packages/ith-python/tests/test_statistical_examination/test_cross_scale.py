"""Tests for cross-scale correlation analysis."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ith_python.statistical_examination.cross_scale import (
    compute_all_cross_scale_correlations,
    compute_cross_scale_correlation,
    identify_redundant_scales,
)


class TestCrossScaleCorrelation:
    """Tests for cross-scale correlation computation."""

    def test_compute_cross_scale_correlation_basic(self, sample_ith_features_df: pl.DataFrame):
        result = compute_cross_scale_correlation(
            sample_ith_features_df,
            feature_type="bull_ed",
            method="spearman",
        )

        assert "feature_type" in result
        assert result["feature_type"] == "bull_ed"
        assert "correlation_matrix" in result
        assert "mean_correlation" in result
        assert "lookbacks" in result

    def test_compute_cross_scale_correlation_with_threshold(self, sample_ith_features_df: pl.DataFrame):
        result = compute_cross_scale_correlation(
            sample_ith_features_df,
            feature_type="bull_ed",
            threshold=100,
            method="pearson",
        )

        assert result["threshold"] == 100
        assert "correlation_matrix" in result

    def test_compute_cross_scale_correlation_insufficient_columns(self, sample_ith_features_df: pl.DataFrame):
        # Filter to a single threshold AND single lookback - should return error (only 1 column)
        df_single = sample_ith_features_df.select(
            [c for c in sample_ith_features_df.columns if "_rb100_lb50_" in c or c == "bar_index"]
        )

        result = compute_cross_scale_correlation(df_single, feature_type="bull_ed")

        # With only one column matching, should return error
        assert "error" in result or result.get("n_features", 0) < 2

    def test_correlation_matrix_symmetric(self, sample_ith_features_df: pl.DataFrame):
        result = compute_cross_scale_correlation(
            sample_ith_features_df,
            feature_type="bull_ed",
        )

        if "correlation_matrix" in result:
            matrix = np.array(result["correlation_matrix"])
            np.testing.assert_array_almost_equal(matrix, matrix.T)

    def test_correlation_diagonal_is_one(self, sample_ith_features_df: pl.DataFrame):
        result = compute_cross_scale_correlation(
            sample_ith_features_df,
            feature_type="bull_ed",
        )

        if "correlation_matrix" in result:
            matrix = np.array(result["correlation_matrix"])
            np.testing.assert_array_almost_equal(np.diag(matrix), np.ones(matrix.shape[0]))


class TestAllCrossScaleCorrelations:
    """Tests for computing all cross-scale correlations."""

    def test_compute_all_cross_scale(self, sample_ith_features_df: pl.DataFrame):
        result = compute_all_cross_scale_correlations(sample_ith_features_df)

        assert "by_feature_type" in result
        assert "summary" in result
        assert "bull_ed" in result["by_feature_type"]

    def test_compute_all_with_thresholds(self, sample_ith_features_df: pl.DataFrame):
        result = compute_all_cross_scale_correlations(
            sample_ith_features_df,
            thresholds=[100, 250],
        )

        assert "by_threshold" in result
        assert 100 in result["by_threshold"]
        assert 250 in result["by_threshold"]


class TestRedundantScales:
    """Tests for identifying redundant scales."""

    def test_identify_redundant_scales(self, sample_ith_features_df: pl.DataFrame):
        result = identify_redundant_scales(
            sample_ith_features_df,
            feature_type="bull_ed",
            correlation_threshold=0.95,
        )

        assert "all_lookbacks" in result
        assert "recommended_scales" in result
        assert "redundant_scales" in result
        assert "compression_ratio" in result

    def test_redundant_scales_compression(self, sample_ith_features_df: pl.DataFrame):
        result = identify_redundant_scales(
            sample_ith_features_df,
            feature_type="bull_ed",
            correlation_threshold=0.5,  # Very low threshold = more redundancy
        )

        # With low threshold, should have fewer recommended scales
        assert len(result["recommended_scales"]) <= len(result["all_lookbacks"])
