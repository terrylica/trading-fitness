"""Anchored Walk-Forward Optimization (AWFO) with Multiple Origins.

This module implements walk-forward analysis with multiple anchors (starting points)
to reduce sensitivity to the choice of start date.

Reference: Pardo's "Design, Testing, and Optimization of Trading Systems"
and academic work on robust backtesting methodologies.

The key insight: Traditional WFO results can vary significantly based on where
you start. AWFO uses multiple origins and aggregates results for robustness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class WalkForwardWindow:
    """A single walk-forward train/test window."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int
    anchor_id: int  # Which anchor this window belongs to
    window_id: int  # Window index within this anchor

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start

    def get_train_slice(self, data: NDArray) -> NDArray:
        """Extract training data from array."""
        return data[self.train_start : self.train_end]

    def get_test_slice(self, data: NDArray) -> NDArray:
        """Extract test data from array."""
        return data[self.test_start : self.test_end]


@dataclass
class AnchoredWalkForward:
    """Anchored Walk-Forward Optimization configuration and window generator.

    Generates multiple walk-forward sequences, each starting from a different
    anchor point. This addresses the "start date sensitivity" problem in
    traditional walk-forward optimization.

    Example:
        >>> awfo = AnchoredWalkForward(
        ...     n_samples=1000,
        ...     train_size=200,
        ...     test_size=50,
        ...     n_anchors=5,
        ...     anchor_spacing=50,
        ... )
        >>> for window in awfo.generate_windows():
        ...     train = window.get_train_slice(data)
        ...     test = window.get_test_slice(data)
        ...     # Train on train, evaluate on test
    """

    n_samples: int  # Total number of samples in the dataset
    train_size: int  # Size of training window
    test_size: int  # Size of test window
    n_anchors: int = 5  # Number of different starting points
    anchor_spacing: int | None = None  # Spacing between anchors (auto if None)
    step_size: int | None = None  # Step between windows (defaults to test_size)
    anchored: bool = True  # If True, training window expands; if False, slides

    # Derived attributes
    _windows: list[WalkForwardWindow] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Validate configuration and generate windows."""
        if self.step_size is None:
            self.step_size = self.test_size

        if self.anchor_spacing is None:
            # Distribute anchors evenly across the first quarter of data
            max_anchor_offset = self.n_samples // 4
            self.anchor_spacing = max(1, max_anchor_offset // self.n_anchors)

        self._validate()
        self._generate_windows()

    def _validate(self):
        """Validate configuration."""
        min_required = self.train_size + self.test_size
        if self.n_samples < min_required:
            msg = f"Need at least {min_required} samples, got {self.n_samples}"
            raise ValueError(msg)

        max_anchor_start = (self.n_anchors - 1) * self.anchor_spacing
        if max_anchor_start + self.train_size + self.test_size > self.n_samples:
            msg = f"Anchor configuration exceeds data: {max_anchor_start + self.train_size + self.test_size} > {self.n_samples}"
            raise ValueError(msg)

    def _generate_windows(self):
        """Generate all walk-forward windows across all anchors."""
        self._windows = []

        for anchor_id in range(self.n_anchors):
            anchor_start = anchor_id * self.anchor_spacing
            window_id = 0

            if self.anchored:
                # Anchored: training always starts at anchor, grows over time
                train_start = anchor_start
                pos = anchor_start + self.train_size

                while pos + self.test_size <= self.n_samples:
                    train_end = pos
                    test_start = pos
                    test_end = pos + self.test_size

                    self._windows.append(
                        WalkForwardWindow(
                            train_start=train_start,
                            train_end=train_end,
                            test_start=test_start,
                            test_end=test_end,
                            anchor_id=anchor_id,
                            window_id=window_id,
                        )
                    )

                    pos += self.step_size
                    window_id += 1
            else:
                # Rolling: fixed-size training window slides forward
                pos = anchor_start

                while pos + self.train_size + self.test_size <= self.n_samples:
                    train_start = pos
                    train_end = pos + self.train_size
                    test_start = train_end
                    test_end = test_start + self.test_size

                    self._windows.append(
                        WalkForwardWindow(
                            train_start=train_start,
                            train_end=train_end,
                            test_start=test_start,
                            test_end=test_end,
                            anchor_id=anchor_id,
                            window_id=window_id,
                        )
                    )

                    pos += self.step_size
                    window_id += 1

    def generate_windows(self) -> list[WalkForwardWindow]:
        """Return all generated windows."""
        return self._windows

    def get_anchor_windows(self, anchor_id: int) -> list[WalkForwardWindow]:
        """Get all windows for a specific anchor."""
        return [w for w in self._windows if w.anchor_id == anchor_id]

    def __len__(self) -> int:
        return len(self._windows)

    def __iter__(self):
        return iter(self._windows)

    def summary(self) -> dict:
        """Return summary statistics about the window configuration."""
        windows_per_anchor = [
            len(self.get_anchor_windows(i)) for i in range(self.n_anchors)
        ]
        return {
            "n_samples": self.n_samples,
            "n_anchors": self.n_anchors,
            "n_windows_total": len(self._windows),
            "windows_per_anchor": windows_per_anchor,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "step_size": self.step_size,
            "anchor_spacing": self.anchor_spacing,
            "anchored": self.anchored,
        }


@dataclass
class AWFOResult:
    """Results from an AWFO evaluation."""

    metric_name: str
    aggregate_mean: float
    aggregate_std: float
    per_anchor_means: list[float]
    per_anchor_stds: list[float]
    per_window_values: list[float]
    n_anchors: int
    n_windows_total: int


def evaluate_awfo(
    data: NDArray,
    awfo: AnchoredWalkForward,
    eval_fn: callable,
    aggregate: Literal["mean", "median"] = "mean",
) -> AWFOResult:
    """Evaluate a metric across all AWFO windows.

    Args:
        data: The data array to evaluate
        awfo: Configured AnchoredWalkForward instance
        eval_fn: Function that takes (train_data, test_data) and returns a metric
        aggregate: How to aggregate results ("mean" or "median")

    Returns:
        AWFOResult with aggregated statistics
    """
    per_window_values = []
    per_anchor_values = {i: [] for i in range(awfo.n_anchors)}

    for window in awfo:
        train_data = window.get_train_slice(data)
        test_data = window.get_test_slice(data)

        value = eval_fn(train_data, test_data)
        if value is not None and not np.isnan(value):
            per_window_values.append(value)
            per_anchor_values[window.anchor_id].append(value)

    if not per_window_values:
        raise ValueError("No valid evaluations across any windows")

    # Aggregate per-anchor
    per_anchor_means = []
    per_anchor_stds = []
    for anchor_id in range(awfo.n_anchors):
        vals = per_anchor_values[anchor_id]
        if vals:
            per_anchor_means.append(float(np.mean(vals)))
            per_anchor_stds.append(float(np.std(vals)))
        else:
            per_anchor_means.append(np.nan)
            per_anchor_stds.append(np.nan)

    # Overall aggregate
    if aggregate == "mean":
        agg_mean = float(np.mean(per_window_values))
    else:
        agg_mean = float(np.median(per_window_values))

    agg_std = float(np.std(per_window_values))

    return AWFOResult(
        metric_name=eval_fn.__name__ if hasattr(eval_fn, "__name__") else "metric",
        aggregate_mean=agg_mean,
        aggregate_std=agg_std,
        per_anchor_means=per_anchor_means,
        per_anchor_stds=per_anchor_stds,
        per_window_values=per_window_values,
        n_anchors=awfo.n_anchors,
        n_windows_total=len(per_window_values),
    )


# Visualization helpers
def plot_awfo_windows(awfo: AnchoredWalkForward) -> str:
    """Generate ASCII visualization of AWFO windows.

    Returns ASCII art showing the window structure.
    """
    lines = []
    lines.append("Anchored Walk-Forward Windows")
    lines.append("=" * 60)

    for anchor_id in range(awfo.n_anchors):
        windows = awfo.get_anchor_windows(anchor_id)
        if not windows:
            continue

        lines.append(f"\nAnchor {anchor_id} (start={windows[0].train_start}):")

        # Scale to 60 chars
        scale = 60 / awfo.n_samples

        for w in windows[:3]:  # Show first 3 windows per anchor
            bar = [" "] * 60
            train_s = int(w.train_start * scale)
            train_e = int(w.train_end * scale)
            test_s = int(w.test_start * scale)
            test_e = int(w.test_end * scale)

            for i in range(train_s, min(train_e, 60)):
                bar[i] = "="
            for i in range(test_s, min(test_e, 60)):
                bar[i] = "#"

            lines.append(f"  W{w.window_id}: |{''.join(bar)}|")

        if len(windows) > 3:
            lines.append(f"  ... ({len(windows) - 3} more windows)")

    lines.append("\nLegend: = training, # test")
    return "\n".join(lines)
