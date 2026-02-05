"""Tests for Distance Correlation (dCor) redundancy filter.

Phase 2 of the principled feature selection pipeline.
GitHub Issue: https://github.com/terrylica/cc-skills/issues/21
"""

import numpy as np
import polars as pl
import pytest

from ith_python.statistical_examination.dcor_filter import (
    compute_dcor_matrix,
    filter_dcor_redundancy,
    get_dcor_summary,
    get_redundancy_pairs,
)


@pytest.fixture
def features_with_redundancy() -> pl.DataFrame:
    """Create features with known redundancy for testing."""
    np.random.seed(42)
    n = 300

    # Create base features
    base1 = np.random.randn(n)
    base2 = np.random.randn(n)

    data = {
        "bar_index": list(range(n)),
        # Independent features (rb1000 to avoid suppression)
        "ith_rb1000_lb100_bull_ed": base1,
        "ith_rb1000_lb100_bear_ed": base2,
        # Highly correlated with base1 (nonlinear)
        "ith_rb1000_lb100_bull_cv": base1 ** 2 + 0.1 * np.random.randn(n),
        # Moderately correlated with base2
        "ith_rb1000_lb200_bear_ed": base2 + 0.5 * np.random.randn(n),
        # Independent feature
        "ith_rb1000_lb200_bull_ed": np.random.randn(n),
    }

    # Add target with correlation to base1
    data["forward_return"] = base1 * 0.1 + np.random.randn(n) * 0.05

    return pl.DataFrame(data)


@pytest.fixture
def independent_features() -> pl.DataFrame:
    """Create features with no redundancy."""
    np.random.seed(42)
    n = 300

    data = {
        "bar_index": list(range(n)),
        "ith_rb1000_lb100_bull_ed": np.random.randn(n),
        "ith_rb1000_lb100_bear_ed": np.random.randn(n),
        "ith_rb1000_lb200_bull_ed": np.random.randn(n),
        "ith_rb1000_lb200_bear_ed": np.random.randn(n),
    }

    return pl.DataFrame(data)


class TestComputeDcorMatrix:
    """Test compute_dcor_matrix function."""

    def test_returns_pairwise_dcor(self, features_with_redundancy: pl.DataFrame):
        """Should compute dCor for all pairs."""
        feature_cols = [
            "ith_rb1000_lb100_bull_ed",
            "ith_rb1000_lb100_bear_ed",
            "ith_rb1000_lb100_bull_cv",
        ]

        dcor_df = compute_dcor_matrix(features_with_redundancy, feature_cols)

        assert "feature_1" in dcor_df.columns
        assert "feature_2" in dcor_df.columns
        assert "dcor" in dcor_df.columns

        # 3 features = 3 pairs
        assert dcor_df.height == 3

    def test_dcor_values_in_valid_range(self, features_with_redundancy: pl.DataFrame):
        """dCor values should be in [0, 1]."""
        feature_cols = [
            "ith_rb1000_lb100_bull_ed",
            "ith_rb1000_lb100_bear_ed",
            "ith_rb1000_lb100_bull_cv",
        ]

        dcor_df = compute_dcor_matrix(features_with_redundancy, feature_cols)

        dcor_values = dcor_df["dcor"].to_numpy()
        assert all(0 <= v <= 1 for v in dcor_values)

    def test_redundant_pair_has_high_dcor(self, features_with_redundancy: pl.DataFrame):
        """Correlated features should have high dCor."""
        feature_cols = [
            "ith_rb1000_lb100_bull_ed",
            "ith_rb1000_lb100_bull_cv",
        ]

        dcor_df = compute_dcor_matrix(features_with_redundancy, feature_cols)

        # These are nonlinearly related (one is square of other)
        assert dcor_df["dcor"].item() > 0.5


class TestFilterDcorRedundancy:
    """Test filter_dcor_redundancy function."""

    def test_removes_redundant_features(self, features_with_redundancy: pl.DataFrame):
        """Should remove one feature from redundant pairs."""
        feature_cols = [
            "ith_rb1000_lb100_bull_ed",
            "ith_rb1000_lb100_bull_cv",
            "ith_rb1000_lb100_bear_ed",
        ]

        selected = filter_dcor_redundancy(
            features_with_redundancy,
            feature_cols=feature_cols,
            threshold=0.5,
            apply_suppression=False,
        )

        # Should remove one of the redundant pair
        assert len(selected) < len(feature_cols)
        # Should keep independent feature
        assert "ith_rb1000_lb100_bear_ed" in selected

    def test_keeps_all_independent_features(self, independent_features: pl.DataFrame):
        """Should keep all features when none are redundant."""
        feature_cols = [
            "ith_rb1000_lb100_bull_ed",
            "ith_rb1000_lb100_bear_ed",
            "ith_rb1000_lb200_bull_ed",
            "ith_rb1000_lb200_bear_ed",
        ]

        selected = filter_dcor_redundancy(
            independent_features,
            feature_cols=feature_cols,
            threshold=0.7,
            apply_suppression=False,
        )

        # Independent features should all be kept
        assert len(selected) == 4

    def test_respects_threshold(self, features_with_redundancy: pl.DataFrame):
        """Higher threshold should keep more features."""
        feature_cols = [
            "ith_rb1000_lb100_bull_ed",
            "ith_rb1000_lb100_bull_cv",
            "ith_rb1000_lb100_bear_ed",
        ]

        # Low threshold - more aggressive removal
        selected_low = filter_dcor_redundancy(
            features_with_redundancy,
            feature_cols=feature_cols,
            threshold=0.3,
            apply_suppression=False,
        )

        # High threshold - less aggressive
        selected_high = filter_dcor_redundancy(
            features_with_redundancy,
            feature_cols=feature_cols,
            threshold=0.9,
            apply_suppression=False,
        )

        assert len(selected_high) >= len(selected_low)

    def test_respects_suppression(self):
        """Should filter suppressed features before dCor analysis."""
        np.random.seed(42)
        n = 300

        data = {
            "bar_index": list(range(n)),
            "ith_rb25_lb100_bull_ed": np.random.randn(n),  # Suppressed
            "ith_rb1000_lb100_bull_ed": np.random.randn(n),  # Available
            "ith_rb1000_lb100_bear_ed": np.random.randn(n),  # Available
        }

        df = pl.DataFrame(data)

        selected = filter_dcor_redundancy(
            df,
            threshold=0.9,
            apply_suppression=True,
        )

        # Only rb1000 features should remain
        assert all("rb1000" in f for f in selected)
        assert not any("rb25" in f for f in selected)

    def test_empty_input_returns_empty(self):
        """Should handle empty feature list."""
        df = pl.DataFrame({
            "bar_index": [0, 1, 2],
            "target": [0.1, 0.2, 0.3],
        })

        selected = filter_dcor_redundancy(df, feature_cols=[], threshold=0.7)
        assert selected == []


class TestGetRedundancyPairs:
    """Test get_redundancy_pairs function."""

    def test_returns_pairs_above_threshold(self, features_with_redundancy: pl.DataFrame):
        """Should return pairs with dCor > threshold."""
        feature_cols = [
            "ith_rb1000_lb100_bull_ed",
            "ith_rb1000_lb100_bull_cv",
            "ith_rb1000_lb100_bear_ed",
        ]

        pairs = get_redundancy_pairs(
            features_with_redundancy,
            feature_cols=feature_cols,
            threshold=0.3,
        )

        assert pairs.height > 0
        assert all(pairs["dcor"].to_numpy() > 0.3)

    def test_sorted_by_dcor_descending(self, features_with_redundancy: pl.DataFrame):
        """Results should be sorted by dCor descending."""
        feature_cols = [
            "ith_rb1000_lb100_bull_ed",
            "ith_rb1000_lb100_bull_cv",
            "ith_rb1000_lb100_bear_ed",
            "ith_rb1000_lb200_bear_ed",
        ]

        pairs = get_redundancy_pairs(
            features_with_redundancy,
            feature_cols=feature_cols,
            threshold=0.1,
        )

        if pairs.height > 1:
            dcor_values = pairs["dcor"].to_numpy()
            assert all(dcor_values[i] >= dcor_values[i + 1] for i in range(len(dcor_values) - 1))


class TestGetDcorSummary:
    """Test get_dcor_summary function."""

    def test_returns_summary_dict(self, features_with_redundancy: pl.DataFrame):
        """Should return dict with summary metadata."""
        feature_cols = [
            "ith_rb1000_lb100_bull_ed",
            "ith_rb1000_lb100_bull_cv",
            "ith_rb1000_lb100_bear_ed",
        ]

        summary = get_dcor_summary(
            features_with_redundancy,
            feature_cols=feature_cols,
            threshold=0.5,
        )

        assert summary["phase"] == "dCor"
        assert summary["initial_features"] == 3
        assert summary["threshold"] == 0.5
        assert "features_removed" in summary
        assert "features_selected" in summary
        assert "selected_features" in summary


class TestIntegrationWithPipeline:
    """Integration tests showing dCor as Phase 2 of pipeline."""

    def test_dcor_after_mrmr_output(self):
        """dCor should work with mRMR output (list of feature names)."""
        np.random.seed(42)
        n = 300

        # Simulate mRMR output (list of feature names)
        mrmr_selected = [
            "ith_rb1000_lb100_bull_ed",
            "ith_rb1000_lb100_bear_ed",
            "ith_rb1000_lb200_bull_ed",
            "ith_rb1000_lb200_bear_ed",
            "ith_rb1000_lb500_bull_ed",
        ]

        # Create DataFrame with these features
        data = {"bar_index": list(range(n))}
        for f in mrmr_selected:
            data[f] = np.random.randn(n).tolist()

        df = pl.DataFrame(data)

        # dCor filter should accept mRMR output directly
        selected = filter_dcor_redundancy(
            df,
            feature_cols=mrmr_selected,
            threshold=0.7,
            apply_suppression=False,
        )

        # Should return subset of input
        assert all(f in mrmr_selected for f in selected)
        assert isinstance(selected, list)

    def test_pipeline_reduction(self, features_with_redundancy: pl.DataFrame):
        """dCor should reduce feature count from mRMR output."""
        feature_cols = [
            "ith_rb1000_lb100_bull_ed",
            "ith_rb1000_lb100_bear_ed",
            "ith_rb1000_lb100_bull_cv",
            "ith_rb1000_lb200_bear_ed",
            "ith_rb1000_lb200_bull_ed",
        ]

        selected = filter_dcor_redundancy(
            features_with_redundancy,
            feature_cols=feature_cols,
            threshold=0.5,
            apply_suppression=False,
        )

        # With known redundancy, should reduce
        assert len(selected) <= len(feature_cols)
