#!/usr/bin/env python
"""CLI entry point for Bear ITH edge case validation PNG generation.

This script generates visual validation PNGs for comparing Bull (long) and
Bear (short) epoch detection across predefined edge cases.

Usage:
    python scripts/validate_edge_cases.py --output artifacts/validation
    python scripts/validate_edge_cases.py --case pure_decline

SR&ED: Visual validation tooling for Bear ITH experimental development.
SRED-Type: experimental-development
SRED-Claim: BEAR-ITH
"""

import sys
from pathlib import Path

# Add package and tests to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "packages" / "ith-python" / "src"))
sys.path.insert(0, str(repo_root / "packages" / "ith-python" / "tests"))

from ith_python.validate_edge_cases import main

if __name__ == "__main__":
    main()
