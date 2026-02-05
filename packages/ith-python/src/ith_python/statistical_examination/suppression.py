"""Feature suppression registry integration.

Loads suppressed feature patterns from docs/features/SUPPRESSION_REGISTRY.md
and provides filtering functions for the selection pipeline.

Features are suppressed (not deleted) when:
- Known redundant with selected features
- Domain-specific exclusion
- Debugging artifacts
- Regime-specific validity

GitHub Issue: https://github.com/terrylica/cc-skills/issues/21
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


# Default patterns extracted from SUPPRESSION_REGISTRY.md
# These are hardcoded for reliability; the markdown is the SSoT for documentation
DEFAULT_SUPPRESSED_PATTERNS: list[dict] = [
    {
        "pattern": "ith_rb25_lb*_*",
        "category": "redundant",
        "reason": "Highly correlated with rb1000 equivalents",
    },
    {
        "pattern": "ith_rb50_lb*_*",
        "category": "redundant",
        "reason": "Highly correlated with rb1000 equivalents",
    },
    {
        "pattern": "ith_rb100_lb*_*",
        "category": "redundant",
        "reason": "Highly correlated with rb1000 equivalents",
    },
    {
        "pattern": "ith_rb250_lb*_*",
        "category": "redundant",
        "reason": "Highly correlated with rb1000 equivalents",
    },
    {
        "pattern": "ith_rb500_lb*_*",
        "category": "redundant",
        "reason": "Highly correlated with rb1000 equivalents",
    },
    {
        "pattern": "ith_*_lb20_*",
        "category": "unstable",
        "reason": "High variance in walk-forward validation",
    },
    {
        "pattern": "*_intermediate_*",
        "category": "debug",
        "reason": "Diagnostic features, not for ML",
    },
]


def load_suppressed_patterns(
    registry_path: Path | None = None,
) -> list[dict]:
    """Load suppressed feature patterns.

    Args:
        registry_path: Path to SUPPRESSION_REGISTRY.md (uses default patterns if None)

    Returns:
        List of dicts with 'pattern', 'category', 'reason' keys
    """
    if registry_path is None:
        return DEFAULT_SUPPRESSED_PATTERNS.copy()

    # Parse markdown table if custom path provided
    return _parse_registry_markdown(registry_path)


def _parse_registry_markdown(registry_path: Path) -> list[dict]:
    """Parse suppressed patterns from markdown table.

    Extracts patterns from the "Suppressed Features" table in the registry.
    """
    patterns = []

    if not registry_path.exists():
        return DEFAULT_SUPPRESSED_PATTERNS.copy()

    content = registry_path.read_text()

    # Find the Suppressed Features table
    # Pattern: | `pattern` | category | date | reason | superseded_by |
    table_pattern = re.compile(
        r"\|\s*`([^`]+)`\s*\|\s*(\w+)\s*\|\s*[^|]+\|\s*([^|]+)\|",
        re.MULTILINE,
    )

    for match in table_pattern.finditer(content):
        pattern_str = match.group(1).strip()
        category = match.group(2).strip()
        reason = match.group(3).strip()

        # Skip header row
        if pattern_str.lower() == "feature pattern":
            continue

        patterns.append({
            "pattern": pattern_str,
            "category": category,
            "reason": reason,
        })

    return patterns if patterns else DEFAULT_SUPPRESSED_PATTERNS.copy()


def is_suppressed(
    feature_name: str,
    patterns: list[dict] | None = None,
) -> bool:
    """Check if a feature matches any suppression pattern.

    Args:
        feature_name: Name of the feature to check
        patterns: Suppression patterns (uses default if None)

    Returns:
        True if feature should be suppressed
    """
    if patterns is None:
        patterns = DEFAULT_SUPPRESSED_PATTERNS

    return any(fnmatch.fnmatch(feature_name, p["pattern"]) for p in patterns)


def get_suppression_reason(
    feature_name: str,
    patterns: list[dict] | None = None,
) -> dict | None:
    """Get suppression details for a feature.

    Args:
        feature_name: Name of the feature to check
        patterns: Suppression patterns (uses default if None)

    Returns:
        Dict with pattern, category, reason if suppressed; None otherwise
    """
    if patterns is None:
        patterns = DEFAULT_SUPPRESSED_PATTERNS

    for p in patterns:
        if fnmatch.fnmatch(feature_name, p["pattern"]):
            return {
                "feature": feature_name,
                "pattern_matched": p["pattern"],
                "category": p["category"],
                "reason": p["reason"],
            }

    return None


def filter_suppressed(
    feature_cols: Sequence[str],
    patterns: list[dict] | None = None,
    emit_telemetry: bool = False,
) -> list[str]:
    """Filter out suppressed features from a list.

    Args:
        feature_cols: List of feature names to filter
        patterns: Suppression patterns (uses default if None)
        emit_telemetry: If True, log each suppression to NDJSON

    Returns:
        List of non-suppressed feature names
    """
    if patterns is None:
        patterns = DEFAULT_SUPPRESSED_PATTERNS

    available = []
    suppressed_count = 0

    for feature in feature_cols:
        reason = get_suppression_reason(feature, patterns)
        if reason is None:
            available.append(feature)
        else:
            suppressed_count += 1
            if emit_telemetry:
                _log_suppression(reason)

    return available


def _log_suppression(reason: dict) -> None:
    """Log feature suppression to NDJSON telemetry."""
    try:
        from ith_python.ndjson_logger import log_ndjson_event

        log_ndjson_event(
            event_type="feature_suppression",
            **reason,
        )
    except ImportError:
        pass  # Telemetry optional


def get_suppression_summary(
    feature_cols: Sequence[str],
    patterns: list[dict] | None = None,
) -> dict:
    """Get summary of suppression for a feature set.

    Args:
        feature_cols: List of all feature names
        patterns: Suppression patterns (uses default if None)

    Returns:
        Dict with counts by category
    """
    if patterns is None:
        patterns = DEFAULT_SUPPRESSED_PATTERNS

    summary = {
        "total_features": len(feature_cols),
        "suppressed_count": 0,
        "available_count": 0,
        "by_category": {},
    }

    for feature in feature_cols:
        reason = get_suppression_reason(feature, patterns)
        if reason is None:
            summary["available_count"] += 1
        else:
            summary["suppressed_count"] += 1
            category = reason["category"]
            summary["by_category"][category] = summary["by_category"].get(category, 0) + 1

    return summary


if __name__ == "__main__":
    # Demo: show suppression summary for typical ITH features
    demo_features = [
        "ith_rb25_lb100_bull_ed",
        "ith_rb1000_lb100_bull_ed",
        "ith_rb1000_lb500_bear_cv",
        "ith_rb50_lb20_bull_ed",
        "some_intermediate_debug_feature",
    ]

    print("Feature Suppression Demo")
    print("=" * 50)

    for f in demo_features:
        reason = get_suppression_reason(f)
        status = "SUPPRESSED" if reason else "AVAILABLE"
        print(f"{f}: {status}")
        if reason:
            print(f"  -> {reason['category']}: {reason['reason']}")

    print()
    summary = get_suppression_summary(demo_features)
    print(f"Summary: {summary['available_count']}/{summary['total_features']} available")
    print(f"By category: {summary['by_category']}")
