"""Feature computation version tracking for incremental updates.

ClickHouse column: feature_computation_versions_json
Format: {"range_bar": "11.6.1", "exchange_sessions": "11.2.0", "ith": null}

A feature version of null means the feature has not been computed yet.
"""

import json
from dataclasses import dataclass, field
from typing import Self


# Current versions (update when feature computation changes)
FEATURE_VERSIONS: dict[str, str | None] = {
    "range_bar": "11.6.1",  # rangebar-py version
    "exchange_sessions": "11.2.0",  # When exchange_session_* columns added
    "inter_bar": "11.6.0",  # When lookback_* columns added
    "ith_features": None,  # ITH features not yet computed on range bars
}


@dataclass
class FeatureComputationVersions:
    """Track which features have been computed on a bar/segment.

    Example:
        >>> versions = FeatureComputationVersions({"range_bar": "11.6.1", "ith": None})
        >>> versions.has_feature("range_bar")
        True
        >>> versions.has_feature("ith")
        False
        >>> versions.needs_backfill("ith", "1.0")
        True
    """

    versions: dict[str, str | None] = field(default_factory=dict)

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Parse from JSON string (ClickHouse column)."""
        if not json_str or json_str == "{}":
            return cls(versions={})
        data = json.loads(json_str)
        return cls(versions=data)

    def to_json(self) -> str:
        """Serialize to JSON string for ClickHouse."""
        return json.dumps(self.versions, sort_keys=True)

    def has_feature(self, feature: str, min_version: str | None = None) -> bool:
        """Check if feature is present with optional minimum version.

        Args:
            feature: Feature name (e.g., "range_bar", "ith_features")
            min_version: Optional minimum version (semver comparison)

        Returns:
            True if feature is computed (not None) and meets min_version
        """
        if feature not in self.versions:
            return False
        version = self.versions[feature]
        if version is None:
            return False
        if min_version is None:
            return True
        # Simple semver comparison (lexicographic works for X.Y.Z)
        return version >= min_version

    def needs_backfill(self, feature: str, current_version: str) -> bool:
        """Check if feature needs backfill (missing or outdated).

        Args:
            feature: Feature name
            current_version: Current version to compare against

        Returns:
            True if feature should be recomputed
        """
        if feature not in self.versions:
            return True
        version = self.versions[feature]
        if version is None:
            return True
        return version < current_version

    def mark_computed(self, feature: str, version: str) -> None:
        """Mark feature as computed with version."""
        self.versions[feature] = version

    def mark_not_computed(self, feature: str) -> None:
        """Mark feature as not computed (set to None)."""
        self.versions[feature] = None
