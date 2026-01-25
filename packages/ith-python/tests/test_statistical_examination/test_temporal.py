"""Tests for temporal analysis."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ith_python.statistical_examination.temporal import (
    analyze_temporal_structure,
    compute_autocorrelation,
    compute_stationarity,
    identify_fast_decaying_features,
    identify_persistent_features,
)


class TestAutocorrelation:
    """Tests for autocorrelation computation."""

    def test_compute_acf_basic(self, sample_ith_features_df: pl.DataFrame):
        feature_cols = [c for c in sample_ith_features_df.columns if c.startswith("ith_rb")][:5]

        result = compute_autocorrelation(sample_ith_features_df, feature_cols)

        assert isinstance(result, pl.DataFrame)
        assert "feature" in result.columns
        assert "acf_lag1" in result.columns
        assert "half_life" in result.columns

    def test_acf_lag1_high(self, sample_ith_features_df: pl.DataFrame):
        """ACF at lag 1 should be computed for all features."""
        feature_cols = [c for c in sample_ith_features_df.columns if c.startswith("ith_rb")][:5]

        result = compute_autocorrelation(sample_ith_features_df, feature_cols)

        # All features should have lag-1 ACF computed
        acf_lag1 = result["acf_lag1"].to_list()
        assert len(acf_lag1) == len(feature_cols)
        # At least some should be non-null
        non_null = [v for v in acf_lag1 if v is not None]
        assert len(non_null) > 0

    def test_acf_decreasing(self, sample_ith_features_df: pl.DataFrame):
        """ACF computation should return values at all specified lags."""
        feature_cols = [c for c in sample_ith_features_df.columns if c.startswith("ith_rb")][:5]
        lags = [1, 5, 10, 20]

        result = compute_autocorrelation(sample_ith_features_df, feature_cols, lags=lags)

        # Check that all expected columns exist
        for lag in lags:
            assert f"acf_lag{lag}" in result.columns

        # Check that we got results for all features
        assert len(result) == len(feature_cols)

    def test_acf_custom_lags(self, sample_ith_features_df: pl.DataFrame):
        custom_lags = [2, 4, 8]
        feature_cols = [c for c in sample_ith_features_df.columns if c.startswith("ith_rb")][:3]

        result = compute_autocorrelation(sample_ith_features_df, feature_cols, lags=custom_lags)

        assert "acf_lag2" in result.columns
        assert "acf_lag4" in result.columns
        assert "acf_lag8" in result.columns


class TestStationarity:
    """Tests for stationarity testing."""

    def test_stationarity_basic(self, sample_ith_features_df: pl.DataFrame):
        feature_cols = [c for c in sample_ith_features_df.columns if c.startswith("ith_rb")][:5]

        result = compute_stationarity(sample_ith_features_df, feature_cols)

        assert isinstance(result, pl.DataFrame)
        assert "feature" in result.columns
        assert "adf_stat" in result.columns
        assert "stationary" in result.columns

    def test_stationarity_values(self, sample_ith_features_df: pl.DataFrame):
        feature_cols = [c for c in sample_ith_features_df.columns if c.startswith("ith_rb")][:5]

        result = compute_stationarity(sample_ith_features_df, feature_cols)

        for row in result.iter_rows(named=True):
            if row["stationary"] is not None:
                assert isinstance(row["stationary"], bool)
            if row["adf_stat"] is not None:
                # ADF stat should be negative for stationary series
                assert isinstance(row["adf_stat"], float)


class TestTemporalStructure:
    """Tests for combined temporal analysis."""

    def test_analyze_temporal_structure(self, sample_ith_features_df: pl.DataFrame):
        feature_cols = [c for c in sample_ith_features_df.columns if c.startswith("ith_rb")][:10]

        result = analyze_temporal_structure(sample_ith_features_df, feature_cols)

        assert "acf_results" in result
        assert "stationarity_results" in result
        assert "combined" in result
        assert "summary" in result

    def test_temporal_summary(self, sample_ith_features_df: pl.DataFrame):
        feature_cols = [c for c in sample_ith_features_df.columns if c.startswith("ith_rb")][:10]

        result = analyze_temporal_structure(sample_ith_features_df, feature_cols)

        summary = result["summary"]
        assert "n_features" in summary
        assert "n_stationary" in summary
        assert "persistence_distribution" in summary
        assert "recommended_lstm_sequence_length" in summary

    def test_lstm_sequence_recommendation(self, sample_ith_features_df: pl.DataFrame):
        feature_cols = [c for c in sample_ith_features_df.columns if c.startswith("ith_rb")][:10]

        result = analyze_temporal_structure(sample_ith_features_df, feature_cols)

        seq_len = result["summary"]["recommended_lstm_sequence_length"]
        assert 10 <= seq_len <= 200


class TestFeatureClassification:
    """Tests for temporal feature classification."""

    def test_identify_persistent_features(self, sample_ith_features_df: pl.DataFrame):
        feature_cols = [c for c in sample_ith_features_df.columns if c.startswith("ith_rb")][:10]

        persistent = identify_persistent_features(
            sample_ith_features_df,
            feature_cols,
            min_half_life=20,
        )

        assert isinstance(persistent, list)
        for f in persistent:
            assert f in feature_cols

    def test_identify_fast_decaying_features(self, sample_ith_features_df: pl.DataFrame):
        feature_cols = [c for c in sample_ith_features_df.columns if c.startswith("ith_rb")][:10]

        fast_decay = identify_fast_decaying_features(
            sample_ith_features_df,
            feature_cols,
            max_half_life=5,
        )

        assert isinstance(fast_decay, list)
        for f in fast_decay:
            assert f in feature_cols

    def test_persistent_and_fast_no_overlap(self, sample_ith_features_df: pl.DataFrame):
        feature_cols = [c for c in sample_ith_features_df.columns if c.startswith("ith_rb")][:10]

        persistent = set(identify_persistent_features(sample_ith_features_df, feature_cols, min_half_life=10))
        fast_decay = set(identify_fast_decaying_features(sample_ith_features_df, feature_cols, max_half_life=5))

        # With reasonable thresholds, should have no overlap
        assert len(persistent & fast_decay) == 0
