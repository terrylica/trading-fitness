"""Tests for dimensionality analysis."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ith_python.statistical_examination.dimensionality import (
    compute_vif,
    identify_redundant_features,
    perform_pca,
    summarize_dimensionality,
)


class TestPCA:
    """Tests for PCA analysis."""

    def test_perform_pca_basic(self, sample_ith_features_df: pl.DataFrame):
        result = perform_pca(sample_ith_features_df)

        assert "n_samples" in result
        assert "n_features" in result
        assert "n_components_95_variance" in result
        assert "explained_variance_ratio" in result
        assert "loadings" in result

    def test_pca_variance_explained(self, sample_ith_features_df: pl.DataFrame):
        result = perform_pca(sample_ith_features_df)

        # 95% variance components should be less than total features
        assert result["n_components_95_variance"] <= result["n_features"]
        # 90% components should be less than 95%
        assert result["n_components_90_variance"] <= result["n_components_95_variance"]

    def test_pca_cumulative_variance(self, sample_ith_features_df: pl.DataFrame):
        result = perform_pca(sample_ith_features_df)

        cumvar = result["cumulative_variance"]
        # Should be monotonically increasing
        assert all(cumvar[i] <= cumvar[i + 1] for i in range(len(cumvar) - 1))
        # Should end at 1.0
        assert abs(cumvar[-1] - 1.0) < 0.001

    def test_pca_top_contributors(self, sample_ith_features_df: pl.DataFrame):
        result = perform_pca(sample_ith_features_df)

        if "top_contributors_per_component" in result:
            contributors = result["top_contributors_per_component"]
            assert len(contributors) > 0
            for comp in contributors:
                assert "component" in comp
                assert "variance_explained" in comp
                assert "top_features" in comp

    def test_pca_effective_dimensionality(self, sample_ith_features_df: pl.DataFrame):
        result = perform_pca(sample_ith_features_df)

        eff_dim = result["effective_dimensionality"]
        # Participation ratio should be >= 1 and <= n_features
        assert 1 <= eff_dim <= result["n_features"]


class TestVIF:
    """Tests for VIF computation."""

    def test_compute_vif_basic(self, sample_ith_features_df: pl.DataFrame):
        result = compute_vif(sample_ith_features_df)

        assert isinstance(result, pl.DataFrame)
        assert "feature" in result.columns
        assert "vif" in result.columns
        assert "high_multicollinearity" in result.columns

    def test_vif_sorted_descending(self, sample_ith_features_df: pl.DataFrame):
        result = compute_vif(sample_ith_features_df)

        vif_values = result["vif"].to_list()
        # Should be sorted descending
        assert vif_values == sorted(vif_values, reverse=True)

    def test_vif_positive(self, sample_ith_features_df: pl.DataFrame):
        result = compute_vif(sample_ith_features_df)

        for vif in result["vif"].to_list():
            assert vif >= 1.0  # VIF is always >= 1

    def test_vif_threshold(self, sample_ith_features_df: pl.DataFrame):
        threshold = 5.0
        result = compute_vif(sample_ith_features_df, max_vif_threshold=threshold)

        for row in result.iter_rows(named=True):
            if row["vif"] > threshold:
                assert row["high_multicollinearity"]
            else:
                assert not row["high_multicollinearity"]


class TestRedundantFeatures:
    """Tests for redundant feature identification."""

    def test_identify_redundant_features(self, sample_ith_features_df: pl.DataFrame):
        result = identify_redundant_features(sample_ith_features_df)

        assert "total_features" in result
        assert "high_vif_features" in result
        assert "highly_correlated_pairs" in result
        assert "redundant_features" in result
        assert "recommended_features" in result
        assert "reduction_ratio" in result

    def test_redundant_features_no_overlap(self, sample_ith_features_df: pl.DataFrame):
        result = identify_redundant_features(sample_ith_features_df)

        redundant = set(result["redundant_features"])
        recommended = set(result["recommended_features"])

        # No overlap between redundant and recommended
        assert len(redundant & recommended) == 0

    def test_redundant_features_complete(self, sample_ith_features_df: pl.DataFrame):
        result = identify_redundant_features(sample_ith_features_df)

        redundant = set(result["redundant_features"])
        recommended = set(result["recommended_features"])

        # Should cover all features
        assert result["total_features"] == len(redundant) + len(recommended)


class TestDimensionalitySummary:
    """Tests for dimensionality summary."""

    def test_summarize_dimensionality(self, sample_ith_features_df: pl.DataFrame):
        pca_result = perform_pca(sample_ith_features_df)
        vif_df = compute_vif(sample_ith_features_df)

        summary = summarize_dimensionality(pca_result, vif_df)

        assert "n_features" in summary
        assert "effective_dimensions" in summary
        assert "dimensionality_ratio" in summary
        assert "interpretation" in summary

    def test_summary_interpretation(self, sample_ith_features_df: pl.DataFrame):
        pca_result = perform_pca(sample_ith_features_df)
        vif_df = compute_vif(sample_ith_features_df)

        summary = summarize_dimensionality(pca_result, vif_df)

        # Interpretation should be non-empty string
        assert isinstance(summary["interpretation"], str)
        assert len(summary["interpretation"]) > 0
