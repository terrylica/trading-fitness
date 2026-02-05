"""Tests for feature suppression registry.

GitHub Issue: https://github.com/terrylica/cc-skills/issues/21
"""

from ith_python.statistical_examination.suppression import (
    DEFAULT_SUPPRESSED_PATTERNS,
    filter_suppressed,
    get_suppression_reason,
    get_suppression_summary,
    is_suppressed,
    load_suppressed_patterns,
)


class TestIsSuppressed:
    """Test is_suppressed function."""

    def test_redundant_rb25_suppressed(self):
        """Features with rb25 threshold should be suppressed."""
        assert is_suppressed("ith_rb25_lb100_bull_ed") is True
        assert is_suppressed("ith_rb25_lb500_bear_cv") is True

    def test_redundant_rb50_suppressed(self):
        """Features with rb50 threshold should be suppressed."""
        assert is_suppressed("ith_rb50_lb100_bull_ed") is True

    def test_rb1000_not_suppressed(self):
        """Features with rb1000 threshold should NOT be suppressed."""
        assert is_suppressed("ith_rb1000_lb100_bull_ed") is False
        assert is_suppressed("ith_rb1000_lb500_bear_cv") is False

    def test_unstable_lb20_suppressed(self):
        """Features with lb20 lookback should be suppressed as unstable."""
        # Note: rb1000_lb20 matches ith_*_lb20_* pattern
        assert is_suppressed("ith_rb1000_lb20_bull_ed") is True

    def test_intermediate_debug_suppressed(self):
        """Intermediate/debug features should be suppressed."""
        assert is_suppressed("some_intermediate_value") is True
        assert is_suppressed("my_intermediate_debug_feature") is True

    def test_non_ith_feature_not_suppressed(self):
        """Non-ITH features that don't match patterns should pass through."""
        assert is_suppressed("clasp_n_segments") is False
        assert is_suppressed("sharpe_ratio") is False


class TestGetSuppressionReason:
    """Test get_suppression_reason function."""

    def test_returns_dict_for_suppressed_feature(self):
        """Should return dict with suppression details."""
        reason = get_suppression_reason("ith_rb25_lb100_bull_ed")

        assert reason is not None
        assert reason["feature"] == "ith_rb25_lb100_bull_ed"
        assert reason["pattern_matched"] == "ith_rb25_lb*_*"
        assert reason["category"] == "redundant"
        assert "rb1000" in reason["reason"]

    def test_returns_none_for_available_feature(self):
        """Should return None for non-suppressed features."""
        reason = get_suppression_reason("ith_rb1000_lb100_bull_ed")
        assert reason is None


class TestFilterSuppressed:
    """Test filter_suppressed function."""

    def test_filters_out_suppressed_features(self):
        """Should remove suppressed features from list."""
        features = [
            "ith_rb25_lb100_bull_ed",  # suppressed (redundant)
            "ith_rb1000_lb100_bull_ed",  # available
            "ith_rb1000_lb500_bear_cv",  # available
            "ith_rb50_lb20_bull_ed",  # suppressed (both redundant and unstable)
            "some_intermediate_debug",  # suppressed (debug)
        ]

        available = filter_suppressed(features)

        assert "ith_rb1000_lb100_bull_ed" in available
        assert "ith_rb1000_lb500_bear_cv" in available
        assert "ith_rb25_lb100_bull_ed" not in available
        assert "ith_rb50_lb20_bull_ed" not in available
        assert "some_intermediate_debug" not in available
        assert len(available) == 2

    def test_empty_list_returns_empty(self):
        """Should handle empty input."""
        assert filter_suppressed([]) == []

    def test_all_suppressed_returns_empty(self):
        """Should return empty if all features are suppressed."""
        features = ["ith_rb25_lb100_bull_ed", "ith_rb50_lb200_bear_cv"]
        assert filter_suppressed(features) == []


class TestGetSuppressionSummary:
    """Test get_suppression_summary function."""

    def test_counts_by_category(self):
        """Should count suppressions by category."""
        features = [
            "ith_rb25_lb100_bull_ed",  # redundant
            "ith_rb1000_lb100_bull_ed",  # available
            "ith_rb1000_lb20_bear_cv",  # unstable
            "some_intermediate_value",  # debug
        ]

        summary = get_suppression_summary(features)

        assert summary["total_features"] == 4
        assert summary["suppressed_count"] == 3
        assert summary["available_count"] == 1
        assert summary["by_category"]["redundant"] == 1
        assert summary["by_category"]["unstable"] == 1
        assert summary["by_category"]["debug"] == 1


class TestLoadSuppressedPatterns:
    """Test load_suppressed_patterns function."""

    def test_returns_default_patterns_when_no_path(self):
        """Should return default patterns when no path provided."""
        patterns = load_suppressed_patterns(None)
        assert patterns == DEFAULT_SUPPRESSED_PATTERNS
        assert len(patterns) >= 7  # At least 7 default patterns

    def test_patterns_have_required_keys(self):
        """All patterns should have pattern, category, reason keys."""
        patterns = load_suppressed_patterns()
        for p in patterns:
            assert "pattern" in p
            assert "category" in p
            assert "reason" in p


class TestIntegrationWithSelection:
    """Integration tests showing suppression + selection pipeline."""

    def test_suppression_before_selection(self):
        """Demonstrates intended usage: suppress before select."""
        # Simulate all features from rb25, rb50, rb1000
        all_features = [
            "ith_rb25_lb100_bull_ed",
            "ith_rb25_lb100_bear_cv",
            "ith_rb50_lb100_bull_ed",
            "ith_rb50_lb100_bear_cv",
            "ith_rb1000_lb100_bull_ed",
            "ith_rb1000_lb100_bear_cv",
            "ith_rb1000_lb500_bull_ed",
            "ith_rb1000_lb500_bear_cv",
        ]

        # Apply suppression (filter out redundant thresholds)
        available = filter_suppressed(all_features)

        # Only rb1000 features should remain
        assert all("rb1000" in f for f in available)
        assert len(available) == 4

        # lb20 features should also be removed (unstable)
        all_features_with_lb20 = [
            *all_features,
            "ith_rb1000_lb20_bull_ed",
            "ith_rb1000_lb20_bear_cv",
        ]
        available_without_lb20 = filter_suppressed(all_features_with_lb20)
        assert all("lb20" not in f for f in available_without_lb20)
