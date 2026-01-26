"""Feature Storage Module (Layer 2).

This module handles canonical storage (Long Format SSoT) and view generation.

Design Principles:
- Long format is the SSoT (single source of truth)
- Multiple views generated on-demand from SSoT
- Upgradeable independently from compute/analysis layers

Architecture: Multi-View Feature Architecture with Separation of Concerns
- Layer 2: Feature Storage (this module)
- See: docs/plans/2026-01-25-multi-view-feature-architecture-plan.md
"""

from ith_python.storage.schemas import LONG_SCHEMA, validate_long_format
from ith_python.storage.store import FeatureStore
from ith_python.storage.views import (
    get_warmup_bars,
    to_clickhouse,
    to_dense,
    to_nested,
    to_wide,
    validate_warmup,
)

__all__ = [
    "FeatureStore",
    "LONG_SCHEMA",
    "get_warmup_bars",
    "to_clickhouse",
    "to_dense",
    "to_nested",
    "to_wide",
    "validate_long_format",
    "validate_warmup",
]
