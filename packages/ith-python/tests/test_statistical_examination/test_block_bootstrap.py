"""Tests for Block Bootstrap feature importance.

Phase 4a of the principled feature selection pipeline.
GitHub Issue: https://github.com/terrylica/cc-skills/issues/21
"""

import numpy as np
import polars as pl
import pytest

from ith_python.statistical_examination.block_bootstrap import (
    compute_bootstrap_importance,
    compute_optimal_block_length,
    filter_by_stability,
    get_bootstrap_summary,
)


@pytest.fixture
def features_with_importance() -> pl.DataFrame:
    """Create features with known importance structure."""
    np.random.seed(42)
    n = 200

    # Important feature (high correlation with target)
    important_feature = np.random.randn(n)

    # Less important features
    noise1 = np.random.randn(n)
    noise2 = np.random.randn(n)

    data = {
        "bar_index": list(range(n)),
        "ith_rb1000_lb100_bull_ed": important_feature,
        "ith_rb1000_lb100_bear_ed": noise1,
        "ith_rb1000_lb200_bull_ed": noise2,
    }

    # Target has strong relationship with important_feature
    data["forward_return"] = (important_feature * 0.8 + np.random.randn(n) * 0.2).tolist()

    return pl.DataFrame(data)


class TestComputeOptimalBlockLength:
    """Test compute_optimal_block_length function."""

    def test_returns_positive_integer(self):
        """Should return positive integer block length."""
        series = np.random.randn(200)
        block_len = compute_optimal_block_length(series)

        assert isinstance(block_len, int)
        assert block_len >= 2

    def test_minimum_block_length(self):
        """Should enforce minimum block length of 2."""
        series = np.random.randn(10)  # Short series
        block_len = compute_optimal_block_length(series)

        assert block_len >= 2

    def test_fixed_method(self):
        """Fixed method should use sqrt(n) rule."""
        n = 100
        series = np.random.randn(n)
        block_len = compute_optimal_block_length(series, method="fixed")

        # sqrt(100) = 10
        assert block_len == max(2, int(np.sqrt(n)))


class TestComputeBootstrapImportance:
    """Test compute_bootstrap_importance function."""

    def test_returns_importance_metrics(self, features_with_importance: pl.DataFrame):
        """Should return importance with statistics."""
        feature_cols = [
            "ith_rb1000_lb100_bull_ed",
            "ith_rb1000_lb100_bear_ed",
        ]

        importance = compute_bootstrap_importance(
            features_with_importance,
            feature_cols=feature_cols,
            target_col="forward_return",
            n_bootstrap=10,  # Small for test speed
            apply_suppression=False,
        )

        assert "feature" in importance.columns
        assert "mean_importance" in importance.columns
        assert "std_importance" in importance.columns
        assert "cv" in importance.columns
        assert importance.height == 2

    def test_sorted_by_importance(self, features_with_importance: pl.DataFrame):
        """Results should be sorted by mean importance descending."""
        feature_cols = [
            "ith_rb1000_lb100_bull_ed",
            "ith_rb1000_lb100_bear_ed",
            "ith_rb1000_lb200_bull_ed",
        ]

        importance = compute_bootstrap_importance(
            features_with_importance,
            feature_cols=feature_cols,
            target_col="forward_return",
            n_bootstrap=10,
            apply_suppression=False,
        )

        imp_values = importance["mean_importance"].to_numpy()
        assert all(imp_values[i] >= imp_values[i + 1] for i in range(len(imp_values) - 1))

    def test_raises_on_missing_target(self, features_with_importance: pl.DataFrame):
        """Should raise ValueError if target column missing."""
        with pytest.raises(ValueError, match="not found"):
            compute_bootstrap_importance(
                features_with_importance,
                target_col="nonexistent_target",
            )

    def test_raises_on_insufficient_data(self):
        """Should raise ValueError if data too short."""
        df = pl.DataFrame({
            "bar_index": list(range(20)),
            "ith_rb1000_lb100_bull_ed": np.random.randn(20).tolist(),
            "forward_return": np.random.randn(20).tolist(),
        })

        with pytest.raises(ValueError, match="Insufficient data"):
            compute_bootstrap_importance(df, target_col="forward_return")

    def test_respects_suppression(self):
        """Should filter suppressed features before analysis."""
        np.random.seed(42)
        n = 100

        data = {
            "bar_index": list(range(n)),
            "ith_rb25_lb100_bull_ed": np.random.randn(n).tolist(),  # Suppressed
            "ith_rb1000_lb100_bull_ed": np.random.randn(n).tolist(),  # Available
            "forward_return": np.random.randn(n).tolist(),
        }

        df = pl.DataFrame(data)

        importance = compute_bootstrap_importance(
            df,
            target_col="forward_return",
            n_bootstrap=10,
            apply_suppression=True,
        )

        # Only rb1000 features should be analyzed
        features = importance["feature"].to_list()
        assert not any("rb25" in f for f in features)

    def test_empty_features_returns_empty(self):
        """Should return empty DataFrame if no features available."""
        df = pl.DataFrame({
            "bar_index": list(range(100)),
            "forward_return": np.random.randn(100).tolist(),
        })

        importance = compute_bootstrap_importance(
            df,
            feature_cols=[],
            target_col="forward_return",
        )

        assert importance.height == 0


class TestFilterByStability:
    """Test filter_by_stability function."""

    def test_filters_by_cv(self):
        """Should filter features by CV threshold."""
        importance_df = pl.DataFrame({
            "feature": ["f1", "f2", "f3"],
            "mean_importance": [0.5, 0.3, 0.2],
            "std_importance": [0.1, 0.3, 0.05],
            "cv": [0.2, 1.0, 0.25],  # f1 and f3 are stable
        })

        stable = filter_by_stability(importance_df, max_cv=0.5)

        assert "f1" in stable
        assert "f3" in stable
        assert "f2" not in stable

    def test_respects_threshold(self):
        """Higher threshold should keep more features."""
        importance_df = pl.DataFrame({
            "feature": ["f1", "f2", "f3"],
            "mean_importance": [0.5, 0.3, 0.2],
            "std_importance": [0.1, 0.3, 0.5],
            "cv": [0.2, 1.0, 2.5],
        })

        stable_strict = filter_by_stability(importance_df, max_cv=0.3)
        stable_lenient = filter_by_stability(importance_df, max_cv=1.5)

        assert len(stable_lenient) >= len(stable_strict)


class TestGetBootstrapSummary:
    """Test get_bootstrap_summary function."""

    def test_returns_summary_dict(self, features_with_importance: pl.DataFrame):
        """Should return dict with summary metadata."""
        feature_cols = [
            "ith_rb1000_lb100_bull_ed",
            "ith_rb1000_lb100_bear_ed",
        ]

        summary = get_bootstrap_summary(
            features_with_importance,
            feature_cols=feature_cols,
            target_col="forward_return",
            n_bootstrap=10,
            max_cv=0.5,
        )

        assert summary["phase"] == "BlockBootstrap"
        assert summary["initial_features"] == 2
        assert summary["n_bootstrap"] == 10
        assert summary["max_cv"] == 0.5
        assert "stable_features_count" in summary
        assert "stable_features" in summary

    def test_handles_insufficient_data(self):
        """Should return error dict for insufficient data."""
        df = pl.DataFrame({
            "bar_index": list(range(20)),
            "ith_rb1000_lb100_bull_ed": np.random.randn(20).tolist(),
            "forward_return": np.random.randn(20).tolist(),
        })

        summary = get_bootstrap_summary(df, target_col="forward_return")

        assert "error" in summary
        assert "Insufficient" in summary["error"]


class TestIntegrationWithPipeline:
    """Integration tests showing Block Bootstrap in pipeline."""

    def test_accepts_pcmci_output(self, features_with_importance: pl.DataFrame):
        """Block Bootstrap should work with PCMCI output."""
        # Simulate PCMCI output
        pcmci_selected = [
            "ith_rb1000_lb100_bull_ed",
            "ith_rb1000_lb100_bear_ed",
        ]

        importance = compute_bootstrap_importance(
            features_with_importance,
            feature_cols=pcmci_selected,
            target_col="forward_return",
            n_bootstrap=10,
            apply_suppression=False,
        )

        # Should have importance for all PCMCI features
        features = importance["feature"].to_list()
        assert all(f in pcmci_selected for f in features)
