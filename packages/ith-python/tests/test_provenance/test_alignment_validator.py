"""Tests for provenance alignment validation.

Tests validate that:
1. Source tick fingerprints are computed correctly
2. Feature computation versions track feature state
3. Alignment validation catches mismatches
4. Segment continuity validation works
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from ith_python.provenance import (
    FEATURE_VERSIONS,
    FeatureComputationVersions,
    SourceTickAlignmentError,
    compute_source_tick_fingerprint,
    validate_segment_continuity,
    validate_source_tick_alignment,
)


class TestSourceTickFingerprint:
    """Tests for compute_source_tick_fingerprint()."""

    def test_fingerprint_basic(self):
        """Basic fingerprint computation."""
        df = pl.DataFrame({
            "timestamp": [1704067200000, 1704067201000, 1704067202000],
            "price": [100.0, 101.0, 102.0],
            "volume": [10, 20, 30],
        })

        fp = compute_source_tick_fingerprint(df)

        assert "xxhash64_checksum" in fp
        assert "row_count" in fp
        assert "first_timestamp_ms" in fp
        assert "last_timestamp_ms" in fp

        assert fp["row_count"] == 3
        assert fp["first_timestamp_ms"] == 1704067200000
        assert fp["last_timestamp_ms"] == 1704067202000
        assert isinstance(fp["xxhash64_checksum"], int)

    def test_fingerprint_empty_dataframe(self):
        """Empty DataFrame returns zeros."""
        df = pl.DataFrame({
            "timestamp": [],
            "price": [],
        }).cast({"timestamp": pl.Int64, "price": pl.Float64})

        fp = compute_source_tick_fingerprint(df)

        assert fp["xxhash64_checksum"] == 0
        assert fp["row_count"] == 0
        assert fp["first_timestamp_ms"] == 0
        assert fp["last_timestamp_ms"] == 0

    def test_fingerprint_deterministic(self):
        """Same data produces same fingerprint."""
        df1 = pl.DataFrame({
            "timestamp": [1, 2, 3],
            "price": [100.0, 101.0, 102.0],
        })
        df2 = pl.DataFrame({
            "timestamp": [1, 2, 3],
            "price": [100.0, 101.0, 102.0],
        })

        fp1 = compute_source_tick_fingerprint(df1)
        fp2 = compute_source_tick_fingerprint(df2)

        assert fp1["xxhash64_checksum"] == fp2["xxhash64_checksum"]

    def test_fingerprint_different_data(self):
        """Different data produces different fingerprint."""
        df1 = pl.DataFrame({
            "timestamp": [1, 2, 3],
            "price": [100.0, 101.0, 102.0],
        })
        df2 = pl.DataFrame({
            "timestamp": [1, 2, 3],
            "price": [100.0, 101.0, 999.0],  # Different price
        })

        fp1 = compute_source_tick_fingerprint(df1)
        fp2 = compute_source_tick_fingerprint(df2)

        assert fp1["xxhash64_checksum"] != fp2["xxhash64_checksum"]


class TestFeatureComputationVersions:
    """Tests for FeatureComputationVersions dataclass."""

    def test_from_json_empty(self):
        """Parse empty JSON."""
        versions = FeatureComputationVersions.from_json("{}")
        assert versions.versions == {}

    def test_from_json_with_data(self):
        """Parse JSON with feature versions."""
        json_str = '{"range_bar": "11.6.1", "ith": null}'
        versions = FeatureComputationVersions.from_json(json_str)

        assert versions.versions["range_bar"] == "11.6.1"
        assert versions.versions["ith"] is None

    def test_to_json(self):
        """Serialize to JSON."""
        versions = FeatureComputationVersions({"range_bar": "11.6.1", "ith": None})
        json_str = versions.to_json()

        # Parse back to verify
        data = json.loads(json_str)
        assert data["range_bar"] == "11.6.1"
        assert data["ith"] is None

    def test_has_feature_present(self):
        """Check feature presence."""
        versions = FeatureComputationVersions({"range_bar": "11.6.1", "ith": None})

        assert versions.has_feature("range_bar") is True
        assert versions.has_feature("ith") is False  # None means not computed
        assert versions.has_feature("unknown") is False

    def test_has_feature_with_min_version(self):
        """Check feature with minimum version."""
        versions = FeatureComputationVersions({"range_bar": "11.6.1"})

        assert versions.has_feature("range_bar", min_version="11.0.0") is True
        assert versions.has_feature("range_bar", min_version="11.6.0") is True
        assert versions.has_feature("range_bar", min_version="11.6.1") is True
        assert versions.has_feature("range_bar", min_version="11.7.0") is False
        assert versions.has_feature("range_bar", min_version="12.0.0") is False

    def test_needs_backfill_missing(self):
        """Feature needs backfill if missing."""
        versions = FeatureComputationVersions({})

        assert versions.needs_backfill("ith", "1.0") is True

    def test_needs_backfill_null(self):
        """Feature needs backfill if null."""
        versions = FeatureComputationVersions({"ith": None})

        assert versions.needs_backfill("ith", "1.0") is True

    def test_needs_backfill_outdated(self):
        """Feature needs backfill if outdated."""
        versions = FeatureComputationVersions({"ith": "0.9"})

        assert versions.needs_backfill("ith", "1.0") is True

    def test_needs_backfill_current(self):
        """Feature does not need backfill if current."""
        versions = FeatureComputationVersions({"ith": "1.0"})

        assert versions.needs_backfill("ith", "1.0") is False
        assert versions.needs_backfill("ith", "0.9") is False

    def test_mark_computed(self):
        """Mark feature as computed."""
        versions = FeatureComputationVersions({})
        versions.mark_computed("ith", "1.0")

        assert versions.versions["ith"] == "1.0"
        assert versions.has_feature("ith") is True

    def test_mark_not_computed(self):
        """Mark feature as not computed."""
        versions = FeatureComputationVersions({"ith": "1.0"})
        versions.mark_not_computed("ith")

        assert versions.versions["ith"] is None
        assert versions.has_feature("ith") is False

    def test_feature_versions_constant(self):
        """FEATURE_VERSIONS constant has expected keys."""
        assert "range_bar" in FEATURE_VERSIONS
        assert "exchange_sessions" in FEATURE_VERSIONS
        assert "inter_bar" in FEATURE_VERSIONS
        assert "ith_features" in FEATURE_VERSIONS


class TestValidateSourceTickAlignment:
    """Tests for validate_source_tick_alignment()."""

    def test_valid_alignment(self):
        """Valid alignment passes."""
        df = pl.DataFrame({
            "timestamp": [1000, 2000, 3000],
            "price": [100.0, 101.0, 102.0],
        })
        fp = compute_source_tick_fingerprint(df)

        is_valid, msg = validate_source_tick_alignment(fp, df, strict=False)

        assert is_valid is True
        assert msg == ""

    def test_invalid_checksum_non_strict(self):
        """Invalid checksum in non-strict mode returns False."""
        df = pl.DataFrame({
            "timestamp": [1000, 2000, 3000],
            "price": [100.0, 101.0, 102.0],
        })
        fp = compute_source_tick_fingerprint(df)
        fp["xxhash64_checksum"] = 99999  # Wrong checksum

        is_valid, msg = validate_source_tick_alignment(fp, df, strict=False)

        assert is_valid is False
        assert "xxhash64_checksum" in msg
        assert "mismatch" in msg

    def test_invalid_checksum_strict(self):
        """Invalid checksum in strict mode raises error."""
        df = pl.DataFrame({
            "timestamp": [1000, 2000, 3000],
            "price": [100.0, 101.0, 102.0],
        })
        fp = compute_source_tick_fingerprint(df)
        fp["xxhash64_checksum"] = 99999
        fp["bar_id"] = "test_bar"

        with pytest.raises(SourceTickAlignmentError) as exc_info:
            validate_source_tick_alignment(fp, df, strict=True)

        assert exc_info.value.field == "xxhash64_checksum"
        assert exc_info.value.bar_id == "test_bar"

    def test_legacy_data_skipped(self):
        """Legacy data with zero/None anchors is skipped."""
        df = pl.DataFrame({
            "timestamp": [1000, 2000, 3000],
            "price": [100.0, 101.0, 102.0],
        })
        # Legacy anchors with zeros
        anchors = {
            "xxhash64_checksum": 0,
            "row_count": 0,
            "first_timestamp_ms": 0,
            "last_timestamp_ms": 0,
        }

        is_valid, msg = validate_source_tick_alignment(anchors, df, strict=True)

        assert is_valid is True
        assert msg == ""


class TestValidateSegmentContinuity:
    """Tests for validate_segment_continuity()."""

    def test_valid_segment(self):
        """Valid continuous segment passes."""
        bars = pl.DataFrame({
            "ouroboros_segment_id": ["2024_01", "2024_01", "2024_01"],
            "bar_position_index_in_segment": [0, 1, 2],
            "bar_position_is_segment_first": [1, 0, 0],
            "bar_position_is_segment_last": [0, 0, 1],
        })

        is_valid, errors = validate_segment_continuity(bars, "2024_01")

        assert is_valid is True
        assert errors == []

    def test_missing_columns(self):
        """Missing columns reported."""
        bars = pl.DataFrame({
            "ouroboros_segment_id": ["2024_01"],
        })

        is_valid, errors = validate_segment_continuity(bars, "2024_01")

        assert is_valid is False
        assert any("Missing required columns" in e for e in errors)

    def test_wrong_segment_id(self):
        """Wrong segment ID reported."""
        bars = pl.DataFrame({
            "ouroboros_segment_id": ["2024_02", "2024_02"],
            "bar_position_index_in_segment": [0, 1],
            "bar_position_is_segment_first": [1, 0],
            "bar_position_is_segment_last": [0, 1],
        })

        is_valid, errors = validate_segment_continuity(bars, "2024_01")

        assert is_valid is False
        assert any("Segment ID mismatch" in e for e in errors)

    def test_missing_first_bar(self):
        """Missing first bar marker reported."""
        bars = pl.DataFrame({
            "ouroboros_segment_id": ["2024_01", "2024_01"],
            "bar_position_index_in_segment": [0, 1],
            "bar_position_is_segment_first": [0, 0],  # No first marker
            "bar_position_is_segment_last": [0, 1],
        })

        is_valid, errors = validate_segment_continuity(bars, "2024_01")

        assert is_valid is False
        assert any("first bar" in e for e in errors)

    def test_missing_last_bar(self):
        """Missing last bar marker reported."""
        bars = pl.DataFrame({
            "ouroboros_segment_id": ["2024_01", "2024_01"],
            "bar_position_index_in_segment": [0, 1],
            "bar_position_is_segment_first": [1, 0],
            "bar_position_is_segment_last": [0, 0],  # No last marker
        })

        is_valid, errors = validate_segment_continuity(bars, "2024_01")

        assert is_valid is False
        assert any("last bar" in e for e in errors)
