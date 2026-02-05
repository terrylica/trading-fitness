"""Provenance module for incremental feature addition to ClickHouse range bars.

This module provides:
- compute_source_tick_fingerprint: xxHash64 fingerprint of tick data
- FeatureComputationVersions: Track which features are computed on each bar
- validate_source_tick_alignment: Strict validation for incremental updates
- SourceTickAlignmentError: Exception for alignment failures

ClickHouse column mapping:
    source_tick_*                    <- source_tick_fingerprint.py
    feature_computation_versions_json <- feature_computation_versions.py
    Validation uses all source_tick_* <- alignment_validator.py
"""

from ith_python.provenance.source_tick_fingerprint import (
    compute_source_tick_fingerprint,
)

from ith_python.provenance.feature_computation_versions import (
    FeatureComputationVersions,
    FEATURE_VERSIONS,
)

from ith_python.provenance.alignment_validator import (
    SourceTickAlignmentError,
    validate_source_tick_alignment,
    validate_segment_continuity,
)

__all__ = [
    # Fingerprinting
    "compute_source_tick_fingerprint",
    # Version tracking
    "FeatureComputationVersions",
    "FEATURE_VERSIONS",
    # Validation
    "SourceTickAlignmentError",
    "validate_source_tick_alignment",
    "validate_segment_continuity",
]
