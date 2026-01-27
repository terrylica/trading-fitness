"""Tests for Anchored Walk-Forward Optimization (AWFO) module."""

from __future__ import annotations

import numpy as np
import pytest

from ith_python.walk_forward import (
    AnchoredWalkForward,
    AWFOResult,
    WalkForwardWindow,
    evaluate_awfo,
    plot_awfo_windows,
)


class TestWalkForwardWindow:
    """Tests for WalkForwardWindow dataclass."""

    def test_train_size(self):
        window = WalkForwardWindow(
            train_start=0, train_end=100, test_start=100, test_end=150, anchor_id=0, window_id=0
        )
        assert window.train_size == 100

    def test_test_size(self):
        window = WalkForwardWindow(
            train_start=0, train_end=100, test_start=100, test_end=150, anchor_id=0, window_id=0
        )
        assert window.test_size == 50

    def test_get_train_slice(self):
        window = WalkForwardWindow(
            train_start=10, train_end=20, test_start=20, test_end=25, anchor_id=0, window_id=0
        )
        data = np.arange(100)
        train = window.get_train_slice(data)
        assert len(train) == 10
        assert train[0] == 10
        assert train[-1] == 19

    def test_get_test_slice(self):
        window = WalkForwardWindow(
            train_start=10, train_end=20, test_start=20, test_end=25, anchor_id=0, window_id=0
        )
        data = np.arange(100)
        test = window.get_test_slice(data)
        assert len(test) == 5
        assert test[0] == 20
        assert test[-1] == 24


class TestAnchoredWalkForward:
    """Tests for AnchoredWalkForward configuration and window generation."""

    def test_basic_configuration(self):
        awfo = AnchoredWalkForward(
            n_samples=1000,
            train_size=200,
            test_size=50,
            n_anchors=3,
            anchor_spacing=50,
        )
        assert awfo.n_samples == 1000
        assert awfo.train_size == 200
        assert awfo.test_size == 50
        assert awfo.n_anchors == 3

    def test_default_step_size(self):
        awfo = AnchoredWalkForward(
            n_samples=1000,
            train_size=200,
            test_size=50,
        )
        assert awfo.step_size == 50  # Defaults to test_size

    def test_auto_anchor_spacing(self):
        awfo = AnchoredWalkForward(
            n_samples=1000,
            train_size=200,
            test_size=50,
            n_anchors=5,
            anchor_spacing=None,
        )
        # Should auto-calculate based on first quarter
        assert awfo.anchor_spacing is not None
        assert awfo.anchor_spacing > 0

    def test_generates_windows(self):
        awfo = AnchoredWalkForward(
            n_samples=1000,
            train_size=200,
            test_size=50,
            n_anchors=3,
            anchor_spacing=50,
        )
        windows = awfo.generate_windows()
        assert len(windows) > 0
        assert all(isinstance(w, WalkForwardWindow) for w in windows)

    def test_windows_per_anchor(self):
        awfo = AnchoredWalkForward(
            n_samples=1000,
            train_size=200,
            test_size=50,
            n_anchors=3,
            anchor_spacing=50,
        )
        for anchor_id in range(3):
            anchor_windows = awfo.get_anchor_windows(anchor_id)
            assert len(anchor_windows) > 0
            assert all(w.anchor_id == anchor_id for w in anchor_windows)

    def test_anchored_mode_expanding_train(self):
        """In anchored mode, training window should expand over time."""
        awfo = AnchoredWalkForward(
            n_samples=1000,
            train_size=200,
            test_size=50,
            n_anchors=1,
            anchor_spacing=0,
            anchored=True,
        )
        windows = awfo.get_anchor_windows(0)
        assert len(windows) >= 2

        # Each subsequent window should have larger train size
        for i in range(1, len(windows)):
            assert windows[i].train_size > windows[i - 1].train_size
            # All should start at same anchor point
            assert windows[i].train_start == windows[0].train_start

    def test_rolling_mode_fixed_train(self):
        """In rolling mode, training window should stay fixed size."""
        awfo = AnchoredWalkForward(
            n_samples=1000,
            train_size=200,
            test_size=50,
            n_anchors=1,
            anchor_spacing=0,
            anchored=False,
        )
        windows = awfo.get_anchor_windows(0)
        assert len(windows) >= 2

        # All windows should have same train size
        for w in windows:
            assert w.train_size == 200

    def test_validation_insufficient_samples(self):
        with pytest.raises(ValueError, match="Need at least"):
            AnchoredWalkForward(
                n_samples=100,
                train_size=200,
                test_size=50,
            )

    def test_validation_anchor_exceeds_data(self):
        with pytest.raises(ValueError, match="Anchor configuration exceeds"):
            AnchoredWalkForward(
                n_samples=500,
                train_size=200,
                test_size=50,
                n_anchors=10,
                anchor_spacing=100,  # 10 * 100 = 1000 > 500
            )

    def test_iteration(self):
        awfo = AnchoredWalkForward(
            n_samples=1000,
            train_size=200,
            test_size=50,
            n_anchors=2,
        )
        count = sum(1 for _ in awfo)
        assert count == len(awfo)

    def test_summary(self):
        awfo = AnchoredWalkForward(
            n_samples=1000,
            train_size=200,
            test_size=50,
            n_anchors=3,
            anchor_spacing=50,
        )
        summary = awfo.summary()
        assert summary["n_samples"] == 1000
        assert summary["n_anchors"] == 3
        assert summary["train_size"] == 200
        assert summary["test_size"] == 50
        assert len(summary["windows_per_anchor"]) == 3


class TestEvaluateAWFO:
    """Tests for the evaluate_awfo function."""

    def test_basic_evaluation(self):
        data = np.random.randn(1000)
        awfo = AnchoredWalkForward(
            n_samples=len(data),
            train_size=100,
            test_size=25,
            n_anchors=3,
            anchor_spacing=50,
        )

        def simple_mean(train, test):
            return float(np.mean(test))

        result = evaluate_awfo(data, awfo, simple_mean)

        assert isinstance(result, AWFOResult)
        assert result.n_anchors == 3
        assert result.n_windows_total > 0
        assert len(result.per_anchor_means) == 3
        assert len(result.per_anchor_stds) == 3

    def test_mean_aggregation(self):
        data = np.ones(1000)
        awfo = AnchoredWalkForward(
            n_samples=len(data),
            train_size=100,
            test_size=25,
            n_anchors=2,
        )

        def return_one(train, test):
            return 1.0

        result = evaluate_awfo(data, awfo, return_one, aggregate="mean")
        assert result.aggregate_mean == pytest.approx(1.0)

    def test_median_aggregation(self):
        data = np.ones(1000)
        awfo = AnchoredWalkForward(
            n_samples=len(data),
            train_size=100,
            test_size=25,
            n_anchors=2,
        )

        def return_one(train, test):
            return 1.0

        result = evaluate_awfo(data, awfo, return_one, aggregate="median")
        assert result.aggregate_mean == pytest.approx(1.0)  # Mean of all 1s is 1

    def test_handles_none_values(self):
        data = np.random.randn(1000)
        awfo = AnchoredWalkForward(
            n_samples=len(data),
            train_size=100,
            test_size=25,
            n_anchors=2,
        )

        call_count = [0]

        def sometimes_none(train, test):
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                return None
            return float(np.mean(test))

        result = evaluate_awfo(data, awfo, sometimes_none)
        # Should still produce valid results, just with fewer windows
        assert result.n_windows_total > 0

    def test_handles_nan_values(self):
        data = np.random.randn(1000)
        awfo = AnchoredWalkForward(
            n_samples=len(data),
            train_size=100,
            test_size=25,
            n_anchors=2,
        )

        call_count = [0]

        def sometimes_nan(train, test):
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                return float("nan")
            return float(np.mean(test))

        result = evaluate_awfo(data, awfo, sometimes_nan)
        # Should filter out NaN values
        assert result.n_windows_total > 0
        assert not np.isnan(result.aggregate_mean)

    def test_no_valid_evaluations_raises(self):
        data = np.random.randn(1000)
        awfo = AnchoredWalkForward(
            n_samples=len(data),
            train_size=100,
            test_size=25,
            n_anchors=2,
        )

        def always_none(train, test):
            return None

        with pytest.raises(ValueError, match="No valid evaluations"):
            evaluate_awfo(data, awfo, always_none)


class TestPlotAWFOWindows:
    """Tests for ASCII visualization."""

    def test_generates_output(self):
        awfo = AnchoredWalkForward(
            n_samples=1000,
            train_size=200,
            test_size=50,
            n_anchors=3,
        )
        output = plot_awfo_windows(awfo)
        assert isinstance(output, str)
        assert "Anchored Walk-Forward Windows" in output
        assert "Anchor 0" in output
        assert "Legend" in output

    def test_shows_train_and_test(self):
        awfo = AnchoredWalkForward(
            n_samples=1000,
            train_size=200,
            test_size=50,
            n_anchors=2,
        )
        output = plot_awfo_windows(awfo)
        # Legend should show train and test markers
        assert "training" in output.lower()
        assert "test" in output.lower()
