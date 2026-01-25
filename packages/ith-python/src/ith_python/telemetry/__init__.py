"""Telemetry module for ITH analysis reproducibility and forensics.

This module provides:
- ProvenanceContext: Track data lineage through analysis pipeline
- fingerprint_array: Generate reproducibility fingerprints for arrays
- capture_random_state: Capture numpy random state for reproducibility
- Event types for structured telemetry logging
"""

from ith_python.telemetry.provenance import (
    ProvenanceContext,
    capture_random_state,
    fingerprint_array,
    get_git_sha,
)

from ith_python.telemetry.events import (
    DataLoadEvent,
    AlgorithmInitEvent,
    EpochDetectedEvent,
    HypothesisResultEvent,
    log_data_load,
    log_algorithm_init,
    log_epoch_detected,
    log_hypothesis_result,
)

__all__ = [
    # Provenance
    "ProvenanceContext",
    "fingerprint_array",
    "capture_random_state",
    "get_git_sha",
    # Events
    "DataLoadEvent",
    "AlgorithmInitEvent",
    "EpochDetectedEvent",
    "HypothesisResultEvent",
    "log_data_load",
    "log_algorithm_init",
    "log_epoch_detected",
    "log_hypothesis_result",
]
