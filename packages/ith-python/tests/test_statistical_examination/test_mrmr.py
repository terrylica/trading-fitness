"""Tests for mRMR feature selection module.

Phase 1 of the principled feature selection pipeline.
GitHub Issue: https://github.com/terrylica/cc-skills/issues/21
"""

import numpy as np
import polars as pl
import pytest

from ith_python.statistical_examination.mrmr import (
    compute_mrmr_scores,
    filter_mrmr,
    get_mrmr_summary,
)


@pytest.fixture
def sample_features_df() -> pl.DataFrame:
    """Create sample features DataFrame for mRMR testing.

    Uses rb1000 threshold to avoid suppression filtering.
    """
    np.random.seed(42)
    n = 500

    data = {"bar_index": list(range(n))}

    # Generate 20 features with rb1000 (not suppressed)
    lookbacks = [50, 100, 200, 500]
    feature_types = ["bull_ed", "bear_ed", "bull_eg", "bear_eg", "bull_cv"]

    for lookback in lookbacks:
        for ft in feature_types:
            col_name = f"ith_rb1000_lb{lookback}_{ft}"
            # Generate values in [0, 1] with varying correlations
            values = np.random.beta(2, 5, n)
            values[: lookback - 1] = np.nan  # Warmup period
            data[col_name] = values.tolist()

    # Add target column with correlation to some features
    target = np.random.randn(n) * 0.02
    # Add correlation with first feature
    first_feature_vals = np.array(data["ith_rb1000_lb50_bull_ed"])
    first_feature_vals = np.nan_to_num(first_feature_vals, nan=0.5)
    target += first_feature_vals * 0.1

    data["forward_return"] = target.tolist()

    return pl.DataFrame(data)


class TestFilterMrmr:
    """Test filter_mrmr function."""

    def test_selects_k_features(self, sample_features_df: pl.DataFrame):
        """Should select exactly k features."""
        selected = filter_mrmr(
            sample_features_df,
            target_col="forward_return",
            k=5,
        )

        assert len(selected) == 5
        assert all(isinstance(f, str) for f in selected)

    def test_returns_fewer_if_insufficient_features(self, sample_features_df: pl.DataFrame):
        """Should return all available if k > available features."""
        # Request more than available
        selected = filter_mrmr(
            sample_features_df,
            target_col="forward_return",
            k=100,
        )

        # Should return at most 20 (4 lookbacks * 5 types)
        assert len(selected) <= 20

    def test_raises_on_missing_target(self, sample_features_df: pl.DataFrame):
        """Should raise ValueError if target column missing."""
        with pytest.raises(ValueError, match="not found"):
            filter_mrmr(
                sample_features_df,
                target_col="nonexistent_target",
                k=5,
            )

    def test_respects_suppression(self):
        """Should filter out suppressed features before selection."""
        np.random.seed(42)
        n = 500

        # Create features with mixed thresholds
        data = {"bar_index": list(range(n))}

        # Add rb25 features (should be suppressed)
        for ft in ["bull_ed", "bear_ed"]:
            col_name = f"ith_rb25_lb100_{ft}"
            data[col_name] = np.random.beta(2, 5, n).tolist()

        # Add rb1000 features (should be available)
        for ft in ["bull_ed", "bear_ed"]:
            col_name = f"ith_rb1000_lb100_{ft}"
            data[col_name] = np.random.beta(2, 5, n).tolist()

        data["target"] = np.random.randn(n).tolist()

        df = pl.DataFrame(data)

        selected = filter_mrmr(
            df,
            target_col="target",
            k=10,
            apply_suppression=True,
        )

        # Only rb1000 features should be selected
        assert all("rb1000" in f for f in selected)
        assert not any("rb25" in f for f in selected)

    def test_can_disable_suppression(self):
        """Should include suppressed features when apply_suppression=False."""
        np.random.seed(42)
        n = 500

        data = {"bar_index": list(range(n))}

        # Only add rb25 features (normally suppressed)
        for ft in ["bull_ed", "bear_ed"]:
            col_name = f"ith_rb25_lb100_{ft}"
            data[col_name] = np.random.beta(2, 5, n).tolist()

        data["target"] = np.random.randn(n).tolist()

        df = pl.DataFrame(data)

        # With suppression disabled, should select from rb25
        selected = filter_mrmr(
            df,
            target_col="target",
            k=2,
            apply_suppression=False,
        )

        assert len(selected) == 2
        assert all("rb25" in f for f in selected)

    def test_empty_features_returns_empty(self):
        """Should return empty list if no features available."""
        df = pl.DataFrame({
            "bar_index": [0, 1, 2],
            "target": [0.1, 0.2, 0.3],
        })

        selected = filter_mrmr(df, target_col="target", k=5)
        assert selected == []


class TestComputeMrmrScores:
    """Test compute_mrmr_scores function."""

    def test_returns_ranked_features(self, sample_features_df: pl.DataFrame):
        """Should return features with mRMR ranks."""
        scores = compute_mrmr_scores(
            sample_features_df,
            target_col="forward_return",
            k=5,
        )

        assert "feature" in scores.columns
        assert "mrmr_rank" in scores.columns
        assert scores.height == 5
        assert scores["mrmr_rank"].to_list() == [1, 2, 3, 4, 5]

    def test_all_features_when_k_none(self, sample_features_df: pl.DataFrame):
        """Should compute scores for all features when k=None."""
        scores = compute_mrmr_scores(
            sample_features_df,
            target_col="forward_return",
            k=None,
        )

        # Should have features ranked in order
        # Note: actual count may be less than 20 due to warmup NaN handling
        assert scores.height > 0
        assert scores["mrmr_rank"].to_list() == list(range(1, scores.height + 1))


class TestGetMrmrSummary:
    """Test get_mrmr_summary function."""

    def test_returns_summary_dict(self, sample_features_df: pl.DataFrame):
        """Should return dict with selection metadata."""
        summary = get_mrmr_summary(
            sample_features_df,
            target_col="forward_return",
            k=5,
        )

        assert summary["phase"] == "mRMR"
        assert summary["initial_features"] == 20
        assert summary["k_requested"] == 5
        assert summary["k_selected"] == 5
        assert len(summary["selected_features"]) == 5
        assert 0 < summary["reduction_ratio"] < 1

    def test_shows_suppression_effect(self):
        """Summary should show impact of suppression."""
        np.random.seed(42)
        n = 500

        data = {"bar_index": list(range(n))}

        # Add 5 rb25 features (suppressed) and 5 rb1000 features (available)
        for i in range(5):
            data[f"ith_rb25_lb100_type{i}"] = np.random.beta(2, 5, n).tolist()
            data[f"ith_rb1000_lb100_type{i}"] = np.random.beta(2, 5, n).tolist()

        data["target"] = np.random.randn(n).tolist()

        df = pl.DataFrame(data)

        summary = get_mrmr_summary(df, target_col="target", k=3)

        assert summary["initial_features"] == 10
        assert summary["after_suppression"] == 5  # Only rb1000 available
        assert summary["k_selected"] == 3


class TestIntegrationWithPipeline:
    """Integration tests showing mRMR as Phase 1 of pipeline."""

    def test_mrmr_reduces_feature_count(self, sample_features_df: pl.DataFrame):
        """mRMR should reduce 160→50 style reduction."""
        # This test uses 20 features, reduce to 10 (50% reduction)
        selected = filter_mrmr(
            sample_features_df,
            target_col="forward_return",
            k=10,
        )

        assert len(selected) == 10

        # Verify selected features are valid ITH features
        for f in selected:
            assert f.startswith("ith_rb")

    def test_output_format_for_next_phase(self, sample_features_df: pl.DataFrame):
        """Output should be list of strings suitable for next phase (dCor)."""
        selected = filter_mrmr(
            sample_features_df,
            target_col="forward_return",
            k=10,
        )

        # Should be a list of strings
        assert isinstance(selected, list)
        assert all(isinstance(f, str) for f in selected)

        # Can be used to slice DataFrame
        subset_df = sample_features_df.select(selected)
        assert subset_df.width == 10
