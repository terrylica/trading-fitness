"""Repository-local path configuration.

Replaces platformdirs for project-local artifact storage.
All paths are relative to the repository root.
"""
from pathlib import Path

# Navigate from packages/ith-python/src/ith_python/ to repo root
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent


def get_data_dir() -> Path:
    """Return the data directory (tracked in git)."""
    return REPO_ROOT / "data"


def get_artifacts_dir() -> Path:
    """Return the artifacts directory (gitignored)."""
    return REPO_ROOT / "artifacts"


def get_log_dir() -> Path:
    """Return the logs directory (gitignored)."""
    return REPO_ROOT / "logs"


def get_custom_nav_dir() -> Path:
    """Return the custom NAV data directory."""
    return get_data_dir() / "nav_data_custom"


def get_synth_ithes_dir() -> Path:
    """Return the synthetic ITH output directory."""
    return get_artifacts_dir() / "synth_ithes"


def ensure_dirs() -> None:
    """Create all required directories if they don't exist."""
    for d in [
        get_data_dir(),
        get_artifacts_dir(),
        get_log_dir(),
        get_custom_nav_dir(),
        get_synth_ithes_dir(),
    ]:
        d.mkdir(parents=True, exist_ok=True)
