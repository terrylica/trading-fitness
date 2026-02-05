"""End-to-end integration tests for the feature selection pipeline.

Full pipeline: mRMR (160→50) → dCor (50→30) → PCMCI (30→15) → Stability (15→10)
GitHub Issue: https://github.com/terrylica/cc-skills/issues/21
"""

import numpy as np
import polars as pl
import pytest

from ith_python.statistical_examination.block_bootstrap import (
    compute_bootstrap_importance,
    filter_by_stability,
)
from ith_python.statistical_examination.dcor_filter import filter_dcor_redundancy
from ith_python.statistical_examination.mrmr import filter_mrmr
from ith_python.statistical_examination.pcmci_filter import filter_pcmci
from ith_python.statistical_examination.suppression import filter_suppressed
from ith_python.statistical_examination.walk_forward import (
    compute_walk_forward_stability,
    select_top_k_stable,
)


@pytest.fixture
def large_feature_dataset() -> pl.DataFrame:
    """Create a realistic feature dataset with 160+ features.

    Simulates the ITH feature matrix with:
    - Multiple thresholds (rb1000 only, per SUPPRESSION_REGISTRY.md)
    - Multiple lookbacks (lb50, lb100, lb200, lb500)
    - Multiple feature types (bull_ed, bear_ed, bull_eg, bear_eg, bull_cv, bear_cv, ...)
    - Target column with realistic relationships

    Returns DataFrame with ~160 features and 500 rows.
    """
    np.random.seed(42)
    n = 500  # Rows

    data = {"bar_index": list(range(n))}

    # Generate features: 4 thresholds x 8 lookbacks x 5 types = 160 features
    # Only use rb1000 to respect suppression registry
    lookbacks = [20, 50, 100, 200, 500, 1000, 1500, 2000]
    feature_types = ["bull_ed", "bear_ed", "bull_eg", "bear_eg", "bull_cv"]

    # Create features with known structure:
    # - Some truly predictive features
    # - Some redundant features
    # - Some noise features
    base_signal = np.random.randn(n)

    for i, lb in enumerate(lookbacks):
        for j, ft in enumerate(feature_types):
            col_name = f"ith_rb1000_lb{lb}_{ft}"

            # Warmup period
            warmup_len = lb - 1 if lb < n else n - 10

            if i < 2 and j < 2:
                # Predictive features (correlated with base_signal)
                values = base_signal * 0.5 + np.random.randn(n) * 0.3
                values = (values - values.min()) / (values.max() - values.min())  # Normalize to [0,1]
            elif i < 4 and j < 3:
                # Redundant features (correlated with predictive)
                values = base_signal * 0.3 + np.random.randn(n) * 0.5
                values = (values - values.min()) / (values.max() - values.min())
            else:
                # Noise features
                values = np.random.beta(2, 5, n)

            values[:warmup_len] = np.nan
            data[col_name] = values.tolist()

    # Target: correlated with base_signal
    target = base_signal * 0.1 + np.random.randn(n) * 0.5
    data["forward_return"] = target.tolist()

    return pl.DataFrame(data)


class TestFullPipelineIntegration:
    """End-to-end tests for the complete feature selection pipeline."""

    def test_suppression_filters_unstable_lookbacks(self, large_feature_dataset: pl.DataFrame):
        """Phase 0: Suppression should filter out unstable lookbacks (lb20)."""
        feature_cols = [c for c in large_feature_dataset.columns if c.startswith("ith_")]

        # rb1000 passes threshold check, but lb20 is suppressed per SUPPRESSION_REGISTRY.md
        filtered = filter_suppressed(feature_cols, emit_telemetry=False)

        # lb20 features should be filtered out (5 features per lookback)
        # Original: 8 lookbacks * 5 types = 40, after removing lb20: 7 * 5 = 35
        assert len(filtered) == 35
        assert all("rb1000" in f for f in filtered)
        # lb20 is suppressed (ith_*_lb20_* pattern)
        assert not any("_lb20_" in f for f in filtered)

    def test_mrmr_reduces_to_k(self, large_feature_dataset: pl.DataFrame):
        """Phase 1: mRMR should reduce features to k (or fewer if not enough)."""
        selected = filter_mrmr(
            large_feature_dataset,
            target_col="forward_return",
            k=30,  # We have 40 features (8 lookbacks * 5 types)
            apply_suppression=False,
        )

        assert len(selected) == 30
        assert all(isinstance(f, str) for f in selected)
        assert all(f.startswith("ith_") for f in selected)

    def test_dcor_reduces_redundancy(self, large_feature_dataset: pl.DataFrame):
        """Phase 2: dCor should reduce redundant features."""
        # Start with 50 features from mRMR
        mrmr_selected = filter_mrmr(
            large_feature_dataset,
            target_col="forward_return",
            k=50,
            apply_suppression=False,
        )

        # Apply dCor filter
        dcor_selected = filter_dcor_redundancy(
            large_feature_dataset,
            feature_cols=mrmr_selected,
            threshold=0.9,  # More lenient for test
            apply_suppression=False,
        )

        # Should reduce features
        assert len(dcor_selected) <= len(mrmr_selected)
        assert all(f in mrmr_selected for f in dcor_selected)

    def test_pcmci_identifies_causal_features(self):
        """Phase 3: PCMCI should identify causally relevant features."""
        # Create a dedicated dataset with NO NaNs for PCMCI testing
        np.random.seed(42)
        n = 300

        # Simple features without warmup
        base_signal = np.random.randn(n)
        data = {
            "bar_index": list(range(n)),
            "ith_rb1000_lb100_f1": ((base_signal + np.random.randn(n) * 0.5) * 0.5 + 0.5).tolist(),
            "ith_rb1000_lb100_f2": (np.random.beta(2, 5, n)).tolist(),
            "ith_rb1000_lb100_f3": (np.random.beta(2, 5, n)).tolist(),
            "forward_return": (base_signal * 0.1 + np.random.randn(n) * 0.5).tolist(),
        }
        df = pl.DataFrame(data)

        initial_features = ["ith_rb1000_lb100_f1", "ith_rb1000_lb100_f2", "ith_rb1000_lb100_f3"]

        pcmci_selected = filter_pcmci(
            df,
            feature_cols=initial_features,
            target_col="forward_return",
            alpha=0.2,  # More lenient for test
            tau_max=2,
            apply_suppression=False,
        )

        # Should return subset
        assert isinstance(pcmci_selected, list)
        assert len(pcmci_selected) <= len(initial_features)

    def test_bootstrap_stability(self, large_feature_dataset: pl.DataFrame):
        """Phase 4a: Bootstrap should assess importance stability."""
        initial_features = [c for c in large_feature_dataset.columns if c.startswith("ith_")][:10]

        importance = compute_bootstrap_importance(
            large_feature_dataset,
            feature_cols=initial_features,
            target_col="forward_return",
            n_bootstrap=20,  # Small for test
            apply_suppression=False,
        )

        assert "feature" in importance.columns
        assert "mean_importance" in importance.columns
        assert "cv" in importance.columns
        assert importance.height == len(initial_features)

        # Get stable features
        stable = filter_by_stability(importance, max_cv=1.0)
        assert isinstance(stable, list)

    def test_walk_forward_stability(self, large_feature_dataset: pl.DataFrame):
        """Phase 4b: Walk-forward should validate temporal stability."""
        initial_features = [c for c in large_feature_dataset.columns if c.startswith("ith_")][:10]

        stability = compute_walk_forward_stability(
            large_feature_dataset,
            feature_cols=initial_features,
            target_col="forward_return",
            n_splits=3,  # Small for test
            apply_suppression=False,
        )

        assert "feature" in stability.columns
        assert "cv" in stability.columns
        assert "mean_importance" in stability.columns
        assert stability.height == len(initial_features)

        # Select top stable
        top_k = select_top_k_stable(stability, k=5, max_cv=2.0)
        assert len(top_k) <= 5

    def test_full_pipeline_flow(self, large_feature_dataset: pl.DataFrame):
        """Integration: Full pipeline mRMR → dCor → Bootstrap → WalkForward."""
        # Phase 0: Suppression
        all_features = [c for c in large_feature_dataset.columns if c.startswith("ith_")]
        after_suppression = filter_suppressed(all_features, emit_telemetry=False)

        # Phase 1: mRMR (40 → 20 for faster test)
        after_mrmr = filter_mrmr(
            large_feature_dataset,
            feature_cols=after_suppression,
            target_col="forward_return",
            k=20,
            apply_suppression=False,
        )

        assert len(after_mrmr) == 20
        print(f"After mRMR: {len(after_mrmr)} features")

        # Phase 2: dCor (20 → ~15)
        after_dcor = filter_dcor_redundancy(
            large_feature_dataset,
            feature_cols=after_mrmr,
            threshold=0.85,
            apply_suppression=False,
        )

        assert len(after_dcor) <= len(after_mrmr)
        print(f"After dCor: {len(after_dcor)} features")

        # Phase 4a: Bootstrap stability (skip PCMCI for speed)
        importance = compute_bootstrap_importance(
            large_feature_dataset,
            feature_cols=after_dcor[:10],  # Limit for speed
            target_col="forward_return",
            n_bootstrap=10,
            apply_suppression=False,
        )

        stable_bootstrap = filter_by_stability(importance, max_cv=1.5)
        print(f"After Bootstrap: {len(stable_bootstrap)} stable features")

        # Phase 4b: Walk-forward stability
        stability = compute_walk_forward_stability(
            large_feature_dataset,
            feature_cols=stable_bootstrap if stable_bootstrap else after_dcor[:5],
            target_col="forward_return",
            n_splits=3,
            apply_suppression=False,
        )

        final_features = select_top_k_stable(stability, k=5, max_cv=2.0)
        print(f"Final selection: {len(final_features)} features")

        # Assertions
        assert isinstance(final_features, list)
        assert len(final_features) <= 5
        assert all(isinstance(f, str) for f in final_features)

    def test_pipeline_respects_suppression_throughout(self):
        """All pipeline phases should respect suppression filtering."""
        np.random.seed(42)
        n = 300

        # Create mixed threshold features
        data = {"bar_index": list(range(n))}

        # rb25 features (suppressed)
        for ft in ["bull_ed", "bear_ed"]:
            data[f"ith_rb25_lb100_{ft}"] = np.random.beta(2, 5, n).tolist()

        # rb1000 features (allowed)
        for ft in ["bull_ed", "bear_ed", "bull_cv"]:
            data[f"ith_rb1000_lb100_{ft}"] = np.random.beta(2, 5, n).tolist()

        data["forward_return"] = np.random.randn(n).tolist()
        df = pl.DataFrame(data)

        # Run mRMR with suppression enabled
        selected = filter_mrmr(
            df,
            target_col="forward_return",
            k=5,
            apply_suppression=True,
        )

        # Should only contain rb1000 features
        assert all("rb1000" in f for f in selected)
        assert not any("rb25" in f for f in selected)

    def test_pipeline_output_format(self):
        """Pipeline outputs should be compatible with downstream consumers."""
        # Create clean dataset without NaNs
        np.random.seed(42)
        n = 300

        data = {"bar_index": list(range(n))}
        for i in range(10):
            data[f"ith_rb1000_lb100_f{i}"] = np.random.beta(2, 5, n).tolist()
        data["forward_return"] = np.random.randn(n).tolist()
        df = pl.DataFrame(data)

        feature_cols = [c for c in df.columns if c.startswith("ith_")]

        # mRMR returns list of feature names
        mrmr_out = filter_mrmr(
            df,
            feature_cols=feature_cols,
            target_col="forward_return",
            k=8,
            apply_suppression=False,
        )
        assert isinstance(mrmr_out, list)
        assert len(mrmr_out) == 8

        # dCor returns list of feature names
        dcor_out = filter_dcor_redundancy(
            df,
            feature_cols=mrmr_out,
            threshold=0.95,
            apply_suppression=False,
        )
        assert isinstance(dcor_out, list)

        # PCMCI returns list of feature names
        pcmci_out = filter_pcmci(
            df,
            feature_cols=dcor_out[:5],
            target_col="forward_return",
            alpha=0.2,
            apply_suppression=False,
        )
        assert isinstance(pcmci_out, list)

        # Walk-forward returns list of feature names
        stability_df = compute_walk_forward_stability(
            df,
            feature_cols=dcor_out[:5],
            target_col="forward_return",
            n_splits=3,
            apply_suppression=False,
        )
        final_out = select_top_k_stable(stability_df, k=3)
        assert isinstance(final_out, list)

        # All outputs can be chained
        print(f"Pipeline: {len(mrmr_out)} -> {len(dcor_out)} -> {len(pcmci_out)} -> {len(final_out)}")


class TestPipelineEdgeCases:
    """Edge case tests for pipeline robustness."""

    def test_handles_empty_feature_list(self):
        """Pipeline should handle empty feature lists gracefully."""
        df = pl.DataFrame({
            "bar_index": list(range(100)),
            "forward_return": np.random.randn(100).tolist(),
        })

        # mRMR with empty features
        selected = filter_mrmr(
            df,
            feature_cols=[],
            target_col="forward_return",
            k=5,
        )
        assert selected == []

    def test_handles_single_feature(self):
        """Pipeline should work with single feature."""
        np.random.seed(42)
        n = 200

        df = pl.DataFrame({
            "bar_index": list(range(n)),
            "ith_rb1000_lb100_bull_ed": np.random.beta(2, 5, n).tolist(),
            "forward_return": np.random.randn(n).tolist(),
        })

        selected = filter_mrmr(
            df,
            target_col="forward_return",
            k=5,
            apply_suppression=False,
        )
        assert len(selected) == 1

    def test_reproducibility_with_seed(self):
        """Pipeline should produce reproducible results with same seed."""
        np.random.seed(42)
        n = 200

        data = {"bar_index": list(range(n))}
        for i in range(10):
            data[f"ith_rb1000_lb100_f{i}"] = np.random.beta(2, 5, n).tolist()
        data["forward_return"] = np.random.randn(n).tolist()
        df = pl.DataFrame(data)

        # Run twice
        result1 = filter_mrmr(df, target_col="forward_return", k=5, apply_suppression=False)
        result2 = filter_mrmr(df, target_col="forward_return", k=5, apply_suppression=False)

        # Should be identical
        assert result1 == result2
