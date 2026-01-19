"""Tests for shared statistical metrics module.

SR&ED: Validation testing for shared metrics consolidation.
SRED-Type: support-work
SRED-Claim: BEAR-ITH
"""

import numpy as np
import pytest

from ith_python.metrics import calculate_cv, calculate_cv_sample


class TestCalculateCV:
    """Tests for calculate_cv (population std)."""

    def test_basic_positive_values(self):
        """CV should work with basic positive values."""
        values = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        result = calculate_cv(values)
        # Mean = 14, Population std = sqrt(8) ≈ 2.83, CV ≈ 0.202
        expected = np.std(values) / np.mean(values)
        assert abs(result - expected) < 0.001

    def test_uniform_values_zero_cv(self):
        """Uniform values should have zero CV."""
        values = np.array([5.0, 5.0, 5.0, 5.0])
        result = calculate_cv(values)
        assert result == 0.0

    def test_empty_array_returns_nan(self):
        """Empty array should return NaN."""
        values = np.array([])
        result = calculate_cv(values)
        assert np.isnan(result)

    def test_single_element_returns_nan(self):
        """Single element should return NaN (need n>=2 for std)."""
        values = np.array([5.0])
        result = calculate_cv(values)
        assert np.isnan(result)

    def test_zero_mean_returns_nan(self):
        """Zero mean should return NaN (division by zero)."""
        values = np.array([-5.0, 5.0])  # Mean = 0
        result = calculate_cv(values)
        assert np.isnan(result)

    def test_negative_mean_returns_nan(self):
        """Negative mean should return NaN (CV undefined for negative mean)."""
        values = np.array([-10.0, -5.0, -15.0])
        result = calculate_cv(values)
        assert np.isnan(result)

    def test_high_dispersion(self):
        """High dispersion values should have high CV."""
        values = np.array([1.0, 100.0])
        result = calculate_cv(values)
        assert result > 0.5  # High dispersion

    def test_low_dispersion(self):
        """Low dispersion values should have low CV."""
        values = np.array([99.0, 100.0, 101.0])
        result = calculate_cv(values)
        assert result < 0.02  # Low dispersion

    def test_matches_numpy_calculation(self):
        """CV should match numpy std/mean calculation."""
        np.random.seed(42)
        values = np.random.rand(100) * 50 + 10  # Random positive values
        result = calculate_cv(values)
        expected = np.std(values) / np.mean(values)  # Population std
        assert abs(result - expected) < 0.0001


class TestCalculateCVSample:
    """Tests for calculate_cv_sample (sample std, n-1)."""

    def test_basic_positive_values(self):
        """Sample CV should work with basic positive values."""
        values = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        result = calculate_cv_sample(values)
        # Sample std uses n-1 in denominator
        expected = np.std(values, ddof=1) / np.mean(values)
        assert abs(result - expected) < 0.001

    def test_sample_cv_larger_than_population(self):
        """Sample CV should be larger than population CV for small n."""
        values = np.array([10.0, 20.0, 30.0])
        pop_cv = calculate_cv(values)
        sample_cv = calculate_cv_sample(values)
        assert sample_cv > pop_cv

    def test_empty_array_returns_nan(self):
        """Empty array should return NaN."""
        values = np.array([])
        result = calculate_cv_sample(values)
        assert np.isnan(result)

    def test_single_element_returns_nan(self):
        """Single element should return NaN."""
        values = np.array([5.0])
        result = calculate_cv_sample(values)
        assert np.isnan(result)

    def test_two_elements_works(self):
        """Two elements is minimum for sample CV."""
        values = np.array([10.0, 20.0])
        result = calculate_cv_sample(values)
        expected = np.std(values, ddof=1) / np.mean(values)
        assert abs(result - expected) < 0.001

    def test_negative_mean_returns_nan(self):
        """Negative mean should return NaN."""
        values = np.array([-10.0, -5.0, -15.0])
        result = calculate_cv_sample(values)
        assert np.isnan(result)

    def test_matches_numpy_sample_calculation(self):
        """Sample CV should match numpy std(ddof=1)/mean calculation."""
        np.random.seed(123)
        values = np.random.rand(50) * 100 + 20
        result = calculate_cv_sample(values)
        expected = np.std(values, ddof=1) / np.mean(values)
        assert abs(result - expected) < 0.0001


class TestCVEdgeCases:
    """Edge case tests for both CV functions."""

    def test_large_array_performance(self):
        """CV should handle large arrays efficiently."""
        np.random.seed(999)
        values = np.random.rand(100_000) * 100 + 50
        result = calculate_cv(values)
        assert not np.isnan(result)
        assert 0 < result < 1

    def test_very_small_values(self):
        """CV should work with very small positive values."""
        values = np.array([1e-10, 2e-10, 3e-10])
        result = calculate_cv(values)
        assert not np.isnan(result)
        assert result > 0

    def test_very_large_values(self):
        """CV should work with very large values."""
        values = np.array([1e10, 2e10, 3e10])
        result = calculate_cv(values)
        assert not np.isnan(result)
        assert result > 0

    def test_mixed_magnitude_values(self):
        """CV should work with mixed magnitude values."""
        values = np.array([0.001, 1.0, 1000.0])
        result = calculate_cv(values)
        assert not np.isnan(result)
        assert result > 1  # High dispersion expected
