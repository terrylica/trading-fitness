"""Feature Computation Module (Layer 1).

This module handles the computation of ITH features from NAV arrays.
It wraps the Rust compute_multiscale_ith() function and provides a Python-friendly API.

Design Principles:
- Pure computation, no storage concerns
- Returns Dict[feature_name, np.ndarray] for flexibility
- Upgradeable independently from storage/analysis layers
"""

from ith_python.features.compute import (
    compute_features,
    compute_features_for_threshold,
)
from ith_python.features.config import FeatureConfig

__all__ = [
    "compute_features",
    "compute_features_for_threshold",
    "FeatureConfig",
]
