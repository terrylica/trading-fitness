"""Configuration Module for Forensic Analysis Pipeline.

Loads and validates config/forensic.toml for the data preparation layer.

Architecture: Multi-View Feature Architecture with DAG-based orchestration
Reference: docs/plans/2026-01-25-multi-view-feature-architecture-plan.md
"""

from ith_python.config.forensic import (
    ForensicConfig,
    load_forensic_config,
    validate_forensic_config,
)

__all__ = [
    "ForensicConfig",
    "load_forensic_config",
    "validate_forensic_config",
]
