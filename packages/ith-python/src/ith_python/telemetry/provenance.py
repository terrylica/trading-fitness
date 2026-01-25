"""Provenance tracking for scientific reproducibility.

This module provides tools for tracking data lineage through the analysis pipeline,
enabling any ITH analysis to be reproduced from logs alone.

Key components:
- ProvenanceContext: Tracks session, experiment, input hashes, and random seeds
- fingerprint_array: Generates SHA256 fingerprints for numpy arrays
- capture_random_state: Captures numpy random state for reproducibility
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def get_git_sha(short: bool = True) -> str:
    """Get current git SHA for provenance tracking.

    Args:
        short: If True, return 8-character short SHA. Otherwise, full SHA.

    Returns:
        Git SHA string, or "unknown" if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        sha = result.stdout.strip()
        return sha[:8] if short else sha
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def fingerprint_array(arr: np.ndarray, name: str) -> dict[str, Any]:
    """Generate reproducibility fingerprint for numpy array.

    Creates a comprehensive fingerprint including:
    - Full SHA256 hash of array bytes
    - Shape and dtype metadata
    - Value range for sanity checking
    - Checksum of first 10 elements for quick comparison

    Args:
        arr: Numpy array to fingerprint
        name: Descriptive name for the array (e.g., "nav_input", "features")

    Returns:
        Dictionary with fingerprint data suitable for JSON serialization
    """
    arr_bytes = arr.tobytes()
    return {
        "name": name,
        "sha256": hashlib.sha256(arr_bytes).hexdigest(),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "range": [float(np.nanmin(arr)), float(np.nanmax(arr))],
        "checksum_first_10": hashlib.sha256(arr[:10].tobytes()).hexdigest()[:16],
        "n_nans": int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
    }


def fingerprint_file(path: Path | str) -> dict[str, Any]:
    """Generate fingerprint for input file.

    Args:
        path: Path to file

    Returns:
        Dictionary with file fingerprint data
    """
    path = Path(path)
    with open(path, "rb") as f:
        content = f.read()
    return {
        "path": str(path),
        "filename": path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def capture_random_state() -> dict[str, Any]:
    """Capture numpy random state for reproducibility.

    Returns:
        Dictionary with random state information for logging
    """
    state = np.random.get_state()
    # state is a tuple: (str, ndarray, int, int, float)
    return {
        "numpy_generator": state[0],  # "MT19937"
        "numpy_seed_first": int(state[1][0]),  # First element of state array
        "numpy_state_hash": hashlib.sha256(state[1].tobytes()).hexdigest()[:16],
        "numpy_pos": int(state[2]),
    }


def set_reproducible_seed(seed: int) -> dict[str, Any]:
    """Set numpy random seed and return state for logging.

    Args:
        seed: Random seed to set

    Returns:
        Dictionary with seed and resulting state
    """
    np.random.seed(seed)
    state = capture_random_state()
    state["seed_set"] = seed
    return state


@dataclass
class ProvenanceContext:
    """Track data lineage through analysis pipeline.

    This context object accumulates provenance information throughout an
    analysis run, enabling complete reproducibility from logs.

    Attributes:
        session_id: Unique identifier for this analysis session
        experiment_id: Optional experiment identifier for grouping runs
        input_hashes: Map of input name to SHA256 hash
        random_seeds: Map of component name to random seed used
        git_sha: Git commit SHA at time of analysis
        config_hash: Hash of configuration used
        start_time: UTC timestamp when analysis started
    """

    session_id: str = field(default_factory=lambda: f"sess_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    experiment_id: str | None = None
    input_hashes: dict[str, str] = field(default_factory=dict)
    random_seeds: dict[str, int] = field(default_factory=dict)
    git_sha: str = field(default_factory=get_git_sha)
    config_hash: str | None = None
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_input_hash(self, name: str, hash_value: str) -> None:
        """Record input data hash for provenance."""
        self.input_hashes[name] = hash_value

    def add_random_seed(self, component: str, seed: int) -> None:
        """Record random seed used by component."""
        self.random_seeds[component] = seed

    def set_config_hash(self, config: dict[str, Any]) -> None:
        """Compute and store hash of configuration."""
        import json
        config_str = json.dumps(config, sort_keys=True, default=str)
        self.config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "experiment_id": self.experiment_id,
            "input_hashes": self.input_hashes,
            "random_seeds": self.random_seeds,
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "start_time": self.start_time,
        }
