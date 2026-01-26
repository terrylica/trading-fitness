"""Feature computation entry point.

This module provides the main interface for computing ITH features from NAV arrays.
It wraps the Rust compute_multiscale_ith() function.

Architecture: Multi-View Feature Architecture with Separation of Concerns
- Layer 1: Feature Computation (this module)
- See: docs/plans/2026-01-25-multi-view-feature-architecture-plan.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ith_python.features.config import FeatureConfig
from ith_python.telemetry import log_algorithm_init
from ith_python.telemetry.provenance import fingerprint_array

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_features(
    nav: NDArray[np.float64],
    config: FeatureConfig,
    *,
    emit_telemetry: bool = True,
) -> pl.DataFrame:
    """Compute ITH features from NAV array.

    This is the main entry point for feature computation. It wraps the Rust
    compute_multiscale_ith() function and returns a Polars DataFrame.

    Args:
        nav: NAV series (N samples, typically starting at 1.0)
        config: Feature computation configuration
        emit_telemetry: Whether to emit NDJSON telemetry events

    Returns:
        Polars DataFrame with columns:
        - bar_index: UInt32 index (0 to N-1)
        - ith_rb{threshold}_lb{lookback}_{feature}: Float64 feature values

    Example:
        >>> nav = np.cumprod(1 + np.random.randn(1000) * 0.01)
        >>> config = FeatureConfig(lookbacks=[20, 50, 100], threshold_dbps=25)
        >>> df = compute_features(nav, config)
        >>> df.columns
        ['bar_index', 'ith_rb25_lb20_bull_ed', 'ith_rb25_lb20_bear_ed', ...]
    """
    # Import here to avoid circular imports and allow graceful degradation
    try:
        from trading_fitness_metrics import (
            MultiscaleIthConfig,
            compute_multiscale_ith,
        )
    except ImportError as e:
        msg = (
            "trading_fitness_metrics not installed. "
            "Run: mise run develop:metrics-rust"
        )
        raise ImportError(msg) from e

    # Emit telemetry
    if emit_telemetry:
        nav_fp = fingerprint_array(nav, "nav_input")
        log_algorithm_init(
            algorithm_name="compute_multiscale_ith",
            version="1.0",
            config={
                "threshold_dbps": config.threshold_dbps,
                "lookbacks": config.lookbacks,
            },
            input_hash=nav_fp["sha256"],
        )

    # Create Rust config
    rust_config = MultiscaleIthConfig(
        threshold_dbps=config.threshold_dbps,
        lookbacks=config.lookbacks,
    )

    # Compute features using Rust implementation
    features = compute_multiscale_ith(nav, rust_config)

    # Convert Arrow RecordBatch to Polars (zero-copy)
    arrow_batch = features.to_arrow()
    df = pl.from_arrow(arrow_batch)

    # Add bar_index column
    df = df.with_row_index("bar_index")

    return df


def compute_features_for_threshold(
    nav: NDArray[np.float64],
    threshold_dbps: int,
    lookbacks: list[int] | None = None,
    *,
    emit_telemetry: bool = True,
) -> pl.DataFrame:
    """Convenience function to compute features for a single threshold.

    Args:
        nav: NAV series (N samples)
        threshold_dbps: Range bar threshold in decimal bps
        lookbacks: Lookback windows (default: [20, 50, 100, 200, 500])
        emit_telemetry: Whether to emit NDJSON telemetry events

    Returns:
        Polars DataFrame with feature columns
    """
    if lookbacks is None:
        lookbacks = [20, 50, 100, 200, 500]

    config = FeatureConfig(
        lookbacks=lookbacks,
        threshold_dbps=threshold_dbps,
    )

    return compute_features(nav, config, emit_telemetry=emit_telemetry)
