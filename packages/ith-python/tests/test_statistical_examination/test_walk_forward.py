"""Tests for Walk-Forward Importance Stability.

Phase 4b of the principled feature selection pipeline.
GitHub Issue: https://github.com/terrylica/cc-skills/issues/21
"""

import numpy as np
import polars as pl
import pytest

from ith_python.statistical_examination.walk_forward import (
    compute_walk_forward_stability,
    filter_stable_features,
    get_walk_forward_summary,
    select_top_k_stable,
)


@pytest.fixture
def features_with_stability() -> pl.DataFrame:
    """Create features with varying stability across time."""
    np.random.seed(42)
    n = 300  # Need enough for 5 splits

    # Stable feature (consistent importance across time)
    stable_feature = np.random.randn(n)

    # Unstable feature (importance varies with time)
    unstable_feature = np.zeros(n)
    unstable_feature[: n // 2] = np.random.randn(n // 2) * 2
    unstable_feature[n // 2 :] = np.random.randn(n - n // 2) * 0.1

    # Another stable feature
    stable_feature2 = np.random.randn(n) * 0.5

    data = {
        "bar_index": list(range(n)),
        "ith_rb1000_lb100_bull_ed": stable_feature,
        "ith_rb1000_lb100_bear_ed": unstable_feature,
        "ith_rb1000_lb200_bull_ed": stable_feature2,
    }

    # Target has relationship with stable feature
    data["forward_return"] = (stable_feature * 0.5 + np.random.randn(n) * 0.3).tolist()

    return pl.DataFrame(data)


class TestComputeWalkForwardStability:
    """Test compute_walk_forward_stability function."""

    def test_returns_stability_metrics(self, features_with_stability: pl.DataFrame):
        """Should return stability metrics."""
        feature_cols = [
            "ith_rb1000_lb100_bull_ed",
            "ith_rb1000_lb100_bear_ed",
        ]

        stability = compute_walk_forward_stability(
            features_with_stability,
            feature_cols=feature_cols,
            target_col="forward_return",
            n_splits=3,
            apply_suppression=False,
        )

        assert "feature" in stability.columns
        assert "mean_importance" in stability.columns
        assert "std_importance" in stability.columns
        assert "cv" in stability.columns
        assert "n_folds" in stability.columns
        assert stability.height == 2

    def test_sorted_by_cv(self, features_with_stability: pl.DataFrame):
        """Results should be sorted by CV ascending (most stable first)."""
        feature_cols = [
            "ith_rb1000_lb100_bull_ed",
            "ith_rb1000_lb100_bear_ed",
            "ith_rb1000_lb200_bull_ed",
        ]

        stability = compute_walk_forward_stability(
            features_with_stability,
            feature_cols=feature_cols,
            target_col="forward_return",
            n_splits=3,
            apply_suppression=False,
        )

        cv_values = stability["cv"].to_numpy()
        assert all(cv_values[i] <= cv_values[i + 1] for i in range(len(cv_values) - 1))

    def test_raises_on_missing_target(self, features_with_stability: pl.DataFrame):
        """Should raise ValueError if target column missing."""
        with pytest.raises(ValueError, match="not found"):
            compute_walk_forward_stability(
                features_with_stability,
                target_col="nonexistent_target",
            )

    def test_raises_on_insufficient_data(self):
        """Should raise ValueError if data too short for splits."""
        df = pl.DataFrame({
            "bar_index": list(range(50)),
            "ith_rb1000_lb100_bull_ed": np.random.randn(50).tolist(),
            "forward_return": np.random.randn(50).tolist(),
        })

        with pytest.raises(ValueError, match="Insufficient data"):
            compute_walk_forward_stability(
                df,
                target_col="forward_return",
                n_splits=5,
            )

    def test_respects_suppression(self):
        """Should filter suppressed features before analysis."""
        np.random.seed(42)
        n = 200

        data = {
            "bar_index": list(range(n)),
            "ith_rb25_lb100_bull_ed": np.random.randn(n).tolist(),  # Suppressed
            "ith_rb1000_lb100_bull_ed": np.random.randn(n).tolist(),  # Available
            "forward_return": np.random.randn(n).tolist(),
        }

        df = pl.DataFrame(data)

        stability = compute_walk_forward_stability(
            df,
            target_col="forward_return",
            n_splits=3,
            apply_suppression=True,
        )

        # Only rb1000 features should be analyzed
        features = stability["feature"].to_list()
        assert not any("rb25" in f for f in features)

    def test_empty_features_returns_empty(self):
        """Should return empty DataFrame if no features available."""
        df = pl.DataFrame({
            "bar_index": list(range(200)),
            "forward_return": np.random.randn(200).tolist(),
        })

        stability = compute_walk_forward_stability(
            df,
            feature_cols=[],
            target_col="forward_return",
        )

        assert stability.height == 0


class TestFilterStableFeatures:
    """Test filter_stable_features function."""

    def test_filters_by_cv_and_importance(self):
        """Should filter features by CV and importance thresholds."""
        stability_df = pl.DataFrame({
            "feature": ["f1", "f2", "f3", "f4"],
            "mean_importance": [0.5, 0.3, 0.02, 0.4],
            "std_importance": [0.1, 0.3, 0.005, 0.08],
            "cv": [0.2, 1.0, 0.25, 0.2],
            "n_folds": [5, 5, 5, 5],
        })

        stable = filter_stable_features(
            stability_df,
            max_cv=0.5,
            min_importance=0.05,
        )

        # f1 and f4: low CV, high importance
        # f2: high CV
        # f3: low importance
        assert "f1" in stable
        assert "f4" in stable
        assert "f2" not in stable
        assert "f3" not in stable


class TestSelectTopKStable:
    """Test select_top_k_stable function."""

    def test_selects_top_k(self):
        """Should select top k features by importance among stable."""
        stability_df = pl.DataFrame({
            "feature": ["f1", "f2", "f3", "f4"],
            "mean_importance": [0.5, 0.3, 0.4, 0.6],
            "std_importance": [0.1, 0.3, 0.1, 0.12],
            "cv": [0.2, 1.0, 0.25, 0.2],  # f2 is unstable
            "n_folds": [5, 5, 5, 5],
        })

        top_2 = select_top_k_stable(stability_df, k=2, max_cv=0.5)

        # Should get f4 (0.6) and f1 (0.5), not f2 (unstable)
        assert len(top_2) == 2
        assert "f4" in top_2
        assert "f1" in top_2
        assert "f2" not in top_2

    def test_respects_k_limit(self):
        """Should return at most k features."""
        stability_df = pl.DataFrame({
            "feature": ["f1", "f2", "f3", "f4", "f5"],
            "mean_importance": [0.5, 0.4, 0.3, 0.2, 0.1],
            "std_importance": [0.1, 0.08, 0.06, 0.04, 0.02],
            "cv": [0.2, 0.2, 0.2, 0.2, 0.2],
            "n_folds": [5, 5, 5, 5, 5],
        })

        top_3 = select_top_k_stable(stability_df, k=3)

        assert len(top_3) == 3


class TestGetWalkForwardSummary:
    """Test get_walk_forward_summary function."""

    def test_returns_summary_dict(self, features_with_stability: pl.DataFrame):
        """Should return dict with summary metadata."""
        feature_cols = [
            "ith_rb1000_lb100_bull_ed",
            "ith_rb1000_lb100_bear_ed",
        ]

        summary = get_walk_forward_summary(
            features_with_stability,
            feature_cols=feature_cols,
            target_col="forward_return",
            n_splits=3,
            max_cv=0.5,
            k=2,
        )

        assert summary["phase"] == "WalkForward"
        assert summary["initial_features"] == 2
        assert summary["n_splits"] == 3
        assert summary["max_cv"] == 0.5
        assert summary["final_k"] == 2
        assert "final_features" in summary
        assert "stable_features_count" in summary

    def test_handles_insufficient_data(self):
        """Should return error dict for insufficient data."""
        df = pl.DataFrame({
            "bar_index": list(range(50)),
            "ith_rb1000_lb100_bull_ed": np.random.randn(50).tolist(),
            "forward_return": np.random.randn(50).tolist(),
        })

        summary = get_walk_forward_summary(df, target_col="forward_return", n_splits=5)

        assert "error" in summary
        assert "Insufficient" in summary["error"]


class TestIntegrationWithPipeline:
    """Integration tests showing Walk-Forward in pipeline."""

    def test_accepts_bootstrap_output(self, features_with_stability: pl.DataFrame):
        """Walk-Forward should work after Block Bootstrap."""
        # Simulate Block Bootstrap output (list of features)
        bootstrap_selected = [
            "ith_rb1000_lb100_bull_ed",
            "ith_rb1000_lb100_bear_ed",
            "ith_rb1000_lb200_bull_ed",
        ]

        stability = compute_walk_forward_stability(
            features_with_stability,
            feature_cols=bootstrap_selected,
            target_col="forward_return",
            n_splits=3,
            apply_suppression=False,
        )

        # Should analyze all bootstrap features
        features = stability["feature"].to_list()
        assert all(f in bootstrap_selected for f in features)

    def test_final_selection(self, features_with_stability: pl.DataFrame):
        """Complete Phase 4 selection flow."""
        # Start with all features
        all_features = [
            "ith_rb1000_lb100_bull_ed",
            "ith_rb1000_lb100_bear_ed",
            "ith_rb1000_lb200_bull_ed",
        ]

        # Compute stability
        stability = compute_walk_forward_stability(
            features_with_stability,
            feature_cols=all_features,
            target_col="forward_return",
            n_splits=3,
            apply_suppression=False,
        )

        # Select final features
        final = select_top_k_stable(stability, k=2, max_cv=1.0)

        # Should return list of strings
        assert isinstance(final, list)
        assert len(final) <= 2
        assert all(isinstance(f, str) for f in final)
