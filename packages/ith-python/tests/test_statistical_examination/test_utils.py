"""Tests for statistical examination utility functions."""

from __future__ import annotations

import polars as pl
import pytest

from ith_python.statistical_examination._utils import (
    columns_by_feature_type,
    columns_by_lookback,
    drop_warmup,
    extract_feature_type,
    extract_lookback,
    extract_threshold,
    get_all_feature_types,
    get_all_lookbacks,
    get_all_thresholds,
    get_feature_columns,
    get_warmup_bars,
)


class TestColumnParsing:
    """Tests for column name parsing functions."""

    def test_extract_threshold(self):
        assert extract_threshold("ith_rb250_lb100_bull_ed") == 250
        assert extract_threshold("ith_rb50_lb20_bear_cv") == 50
        assert extract_threshold("invalid_col") is None
        assert extract_threshold("bar_index") is None

    def test_extract_lookback(self):
        assert extract_lookback("ith_rb250_lb100_bull_ed") == 100
        assert extract_lookback("ith_rb50_lb20_bear_cv") == 20
        assert extract_lookback("invalid_col") is None

    def test_extract_feature_type(self):
        assert extract_feature_type("ith_rb250_lb100_bull_ed") == "bull_ed"
        assert extract_feature_type("ith_rb50_lb20_bear_cv") == "bear_cv"
        assert extract_feature_type("ith_rb100_lb50_max_dd") == "max_dd"
        assert extract_feature_type("invalid_col") is None


class TestWarmupHandling:
    """Tests for warmup period handling."""

    def test_get_warmup_bars(self):
        assert get_warmup_bars([20, 50, 100]) == 99
        assert get_warmup_bars([20]) == 19
        assert get_warmup_bars([100, 200, 500]) == 499

    def test_drop_warmup(self, sample_ith_features_df: pl.DataFrame):
        lookbacks = [20, 50, 100]
        original_len = len(sample_ith_features_df)
        warmup = get_warmup_bars(lookbacks)

        cleaned = drop_warmup(sample_ith_features_df, lookbacks)

        assert len(cleaned) == original_len - warmup


class TestColumnFiltering:
    """Tests for column filtering and grouping."""

    def test_get_feature_columns(self, sample_ith_features_df: pl.DataFrame):
        cols = get_feature_columns(sample_ith_features_df)

        assert len(cols) > 0
        assert "bar_index" not in cols
        assert all(c.startswith("ith_rb") for c in cols)

    def test_get_feature_columns_with_threshold_filter(self, sample_ith_features_df: pl.DataFrame):
        cols = get_feature_columns(sample_ith_features_df, threshold=100)

        assert len(cols) > 0
        assert all("_rb100_" in c for c in cols)
        assert not any("_rb250_" in c for c in cols)

    def test_get_feature_columns_with_lookback_filter(self, sample_ith_features_df: pl.DataFrame):
        cols = get_feature_columns(sample_ith_features_df, lookback=50)

        assert len(cols) > 0
        assert all("_lb50_" in c for c in cols)

    def test_get_feature_columns_with_feature_type_filter(self, sample_ith_features_df: pl.DataFrame):
        cols = get_feature_columns(sample_ith_features_df, feature_type="bull_ed")

        assert len(cols) > 0
        assert all(c.endswith("_bull_ed") for c in cols)

    def test_get_all_thresholds(self, sample_ith_features_df: pl.DataFrame):
        thresholds = get_all_thresholds(sample_ith_features_df)

        assert thresholds == [100, 250]

    def test_get_all_lookbacks(self, sample_ith_features_df: pl.DataFrame):
        lookbacks = get_all_lookbacks(sample_ith_features_df)

        assert lookbacks == [20, 50, 100]

    def test_get_all_feature_types(self, sample_ith_features_df: pl.DataFrame):
        types = get_all_feature_types(sample_ith_features_df)

        assert "bull_ed" in types
        assert "bear_ed" in types
        assert "max_dd" in types

    def test_columns_by_feature_type(self, sample_ith_features_df: pl.DataFrame):
        grouped = columns_by_feature_type(sample_ith_features_df)

        assert "bull_ed" in grouped
        assert len(grouped["bull_ed"]) > 0
        assert all(c.endswith("_bull_ed") for c in grouped["bull_ed"])

    def test_columns_by_lookback(self, sample_ith_features_df: pl.DataFrame):
        grouped = columns_by_lookback(sample_ith_features_df)

        assert 50 in grouped
        assert len(grouped[50]) > 0
        assert all("_lb50_" in c for c in grouped[50])
