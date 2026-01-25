"""Tests for feature importance analysis."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ith_python.statistical_examination.feature_importance import (
    compute_combined_importance,
    compute_correlation_importance,
    compute_mutual_information,
)


class TestMutualInformation:
    """Tests for mutual information computation."""

    def test_compute_mi_basic(self, sample_features_with_target: pl.DataFrame):
        feature_cols = [c for c in sample_features_with_target.columns if c.startswith("ith_rb")][:10]

        result = compute_mutual_information(
            sample_features_with_target,
            feature_cols,
            "forward_return",
        )

        assert isinstance(result, pl.DataFrame)
        assert "feature" in result.columns
        assert "mutual_information" in result.columns
        assert len(result) == len(feature_cols)

    def test_mi_sorted_descending(self, sample_features_with_target: pl.DataFrame):
        feature_cols = [c for c in sample_features_with_target.columns if c.startswith("ith_rb")][:10]

        result = compute_mutual_information(
            sample_features_with_target,
            feature_cols,
            "forward_return",
        )

        mi_values = result["mutual_information"].to_list()
        assert mi_values == sorted(mi_values, reverse=True)

    def test_mi_non_negative(self, sample_features_with_target: pl.DataFrame):
        feature_cols = [c for c in sample_features_with_target.columns if c.startswith("ith_rb")][:10]

        result = compute_mutual_information(
            sample_features_with_target,
            feature_cols,
            "forward_return",
        )

        for mi in result["mutual_information"].to_list():
            assert mi >= 0

    def test_mi_missing_column_error(self, sample_features_with_target: pl.DataFrame):
        with pytest.raises(ValueError, match="Missing columns"):
            compute_mutual_information(
                sample_features_with_target,
                ["nonexistent_feature"],
                "forward_return",
            )


class TestCorrelationImportance:
    """Tests for correlation-based importance."""

    def test_compute_correlation_basic(self, sample_features_with_target: pl.DataFrame):
        feature_cols = [c for c in sample_features_with_target.columns if c.startswith("ith_rb")][:10]

        result = compute_correlation_importance(
            sample_features_with_target,
            feature_cols,
            "forward_return",
        )

        assert isinstance(result, pl.DataFrame)
        assert "feature" in result.columns
        assert "correlation" in result.columns
        assert "abs_correlation" in result.columns

    def test_correlation_sorted_by_abs(self, sample_features_with_target: pl.DataFrame):
        feature_cols = [c for c in sample_features_with_target.columns if c.startswith("ith_rb")][:10]

        result = compute_correlation_importance(
            sample_features_with_target,
            feature_cols,
            "forward_return",
        )

        abs_corr = result["abs_correlation"].to_list()
        assert abs_corr == sorted(abs_corr, reverse=True)

    def test_correlation_bounds(self, sample_features_with_target: pl.DataFrame):
        feature_cols = [c for c in sample_features_with_target.columns if c.startswith("ith_rb")][:10]

        result = compute_correlation_importance(
            sample_features_with_target,
            feature_cols,
            "forward_return",
        )

        for row in result.iter_rows(named=True):
            if row["correlation"] is not None and not np.isnan(row["correlation"]):
                assert -1 <= row["correlation"] <= 1


class TestCombinedImportance:
    """Tests for combined importance computation."""

    def test_combined_importance_basic(self, sample_features_with_target: pl.DataFrame):
        feature_cols = [c for c in sample_features_with_target.columns if c.startswith("ith_rb")][:10]

        result = compute_combined_importance(
            sample_features_with_target,
            feature_cols,
            target_col="forward_return",
            include_shap=False,  # Skip SHAP for faster tests
        )

        assert "n_features" in result
        assert "methods" in result
        assert "correlation" in result["methods"]
        assert "mutual_information" in result["methods"]

    def test_combined_importance_ranking(self, sample_features_with_target: pl.DataFrame):
        feature_cols = [c for c in sample_features_with_target.columns if c.startswith("ith_rb")][:10]

        result = compute_combined_importance(
            sample_features_with_target,
            feature_cols,
            target_col="forward_return",
            include_shap=False,
        )

        if "combined_ranking" in result:
            rankings = result["combined_ranking"]
            assert len(rankings) == len(feature_cols)
            # Check ranks are sequential
            ranks = [r["rank"] for r in rankings]
            assert ranks == list(range(1, len(feature_cols) + 1))

    def test_combined_importance_no_target(self, sample_ith_features_df: pl.DataFrame):
        result = compute_combined_importance(
            sample_ith_features_df,
            target_col=None,
        )

        assert "error" in result
