"""Feature computation configuration.

This module defines the configuration for ITH feature computation.

Architecture: Multi-View Feature Architecture with Separation of Concerns
- Layer 1: Feature Computation (this module)
- See: docs/plans/2026-01-25-multi-view-feature-architecture-plan.md
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FeatureConfig:
    """Configuration for ITH feature computation.

    Attributes:
        lookbacks: Lookback windows in bars (e.g., [20, 50, 100, 200, 500])
        threshold_dbps: Range bar threshold in decimal bps (for column naming only)

    Note:
        The threshold_dbps is ONLY used for column naming (e.g., ith_rb25_lb100_bull_ed).
        Actual TMAEG is auto-calculated from data volatility in the Rust implementation.
    """

    lookbacks: list[int] = field(
        default_factory=lambda: [20, 50, 100, 200, 500]
    )
    threshold_dbps: int = 25

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.lookbacks:
            msg = "lookbacks cannot be empty"
            raise ValueError(msg)
        if any(lb <= 0 for lb in self.lookbacks):
            msg = "all lookbacks must be positive"
            raise ValueError(msg)
        if self.threshold_dbps <= 0:
            msg = "threshold_dbps must be positive"
            raise ValueError(msg)

    @property
    def n_lookbacks(self) -> int:
        """Number of lookback windows."""
        return len(self.lookbacks)

    @property
    def n_features_per_lookback(self) -> int:
        """Number of features computed per lookback (constant = 8)."""
        return 8

    @property
    def total_features(self) -> int:
        """Total number of feature columns."""
        return self.n_lookbacks * self.n_features_per_lookback
