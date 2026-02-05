"""Strict validation that bars can be traced to source tick data.

Uses source_tick_* columns to validate alignment before incremental updates.
Raises SourceTickAlignmentError when bar cannot be traced to its source tick data.

Column mapping (ClickHouse → validation):
    source_tick_xxhash64_checksum  - Must match compute_source_tick_fingerprint()
    source_tick_row_count          - Must match len(tick_df)
    source_tick_first_timestamp_ms - Must match tick_df["timestamp"].min()
    source_tick_last_timestamp_ms  - Must match tick_df["timestamp"].max()
"""

from typing import Any

import polars as pl

from ith_python.provenance.source_tick_fingerprint import (
    compute_source_tick_fingerprint,
)


class SourceTickAlignmentError(Exception):
    """Raised when bar cannot be traced to its source tick data.

    This error indicates that the bar's anchor columns (source_tick_*)
    do not match the actual tick data, meaning incremental feature
    addition would produce inconsistent results.

    Attributes:
        bar_id: Identifier for the misaligned bar
        field: Which field failed validation
        expected: Expected value from tick data
        actual: Actual value from bar anchor columns
    """

    def __init__(
        self,
        bar_id: str,
        field: str,
        expected: Any,
        actual: Any,
        message: str | None = None,
    ) -> None:
        self.bar_id = bar_id
        self.field = field
        self.expected = expected
        self.actual = actual

        if message is None:
            message = (
                f"Bar '{bar_id}' alignment failed: "
                f"{field} mismatch (expected={expected}, actual={actual})"
            )
        super().__init__(message)


def validate_source_tick_alignment(
    bar_anchor_columns: dict[str, Any],
    tick_df: pl.DataFrame,
    strict: bool = True,
) -> tuple[bool, str]:
    """Validate bar's source_tick_* columns match actual tick data.

    Args:
        bar_anchor_columns: Dict with keys matching source_tick_* column suffixes:
            - xxhash64_checksum: UInt64 hash from bar
            - row_count: Number of ticks from bar
            - first_timestamp_ms: First tick timestamp from bar
            - last_timestamp_ms: Last tick timestamp from bar
        tick_df: Polars DataFrame of tick data to validate against.
            Must have 'timestamp' column.
        strict: If True, raise SourceTickAlignmentError on mismatch.
            If False, return (False, error_message) instead.

    Returns:
        Tuple of (is_valid, message).
        message is empty string on success, error description on failure.

    Raises:
        SourceTickAlignmentError: If strict=True and validation fails.

    Example:
        >>> bar_anchors = {
        ...     "xxhash64_checksum": 12345678,
        ...     "row_count": 1000,
        ...     "first_timestamp_ms": 1704067200000,
        ...     "last_timestamp_ms": 1704153600000,
        ... }
        >>> validate_source_tick_alignment(bar_anchors, tick_df, strict=True)
        (True, "")
    """
    # Compute fingerprint from actual tick data
    actual_fingerprint = compute_source_tick_fingerprint(tick_df)

    # Get bar identifier for error messages
    bar_id = bar_anchor_columns.get("bar_id", "unknown")

    # Validate each field
    validation_fields = [
        "xxhash64_checksum",
        "row_count",
        "first_timestamp_ms",
        "last_timestamp_ms",
    ]

    for field_key in validation_fields:
        expected = actual_fingerprint.get(field_key)
        actual = bar_anchor_columns.get(field_key)

        # Skip if bar doesn't have this anchor (legacy data)
        if actual is None or actual == 0:
            continue

        if expected != actual:
            error_msg = (
                f"source_tick_{field_key} mismatch: "
                f"expected {expected} from tick data, "
                f"got {actual} from bar anchor"
            )
            if strict:
                raise SourceTickAlignmentError(
                    bar_id=str(bar_id),
                    field=field_key,
                    expected=expected,
                    actual=actual,
                )
            return False, error_msg

    return True, ""


def validate_segment_continuity(
    bars: pl.DataFrame,
    segment_id: str,
) -> tuple[bool, list[str]]:
    """Validate that bars form a continuous segment without gaps.

    Checks:
    1. All bars have matching ouroboros_segment_id
    2. bar_position_index_in_segment is continuous (0, 1, 2, ...)
    3. Exactly one bar has bar_position_is_segment_first = 1
    4. Exactly one bar has bar_position_is_segment_last = 1

    Args:
        bars: Polars DataFrame with bar_position_* columns
        segment_id: Expected ouroboros_segment_id value

    Returns:
        Tuple of (is_valid, list of error messages).
        Empty list on success.

    Example:
        >>> is_valid, errors = validate_segment_continuity(bars_df, "2024_01")
        >>> if not is_valid:
        ...     print(f"Segment errors: {errors}")
    """
    errors: list[str] = []

    # Check required columns exist
    required_cols = {
        "ouroboros_segment_id",
        "bar_position_index_in_segment",
        "bar_position_is_segment_first",
        "bar_position_is_segment_last",
    }
    missing = required_cols - set(bars.columns)
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return False, errors

    # Check all bars have matching segment_id
    unique_segments = bars["ouroboros_segment_id"].unique().to_list()
    if len(unique_segments) != 1 or unique_segments[0] != segment_id:
        errors.append(
            f"Segment ID mismatch: expected '{segment_id}', "
            f"found {unique_segments}"
        )

    # Check index continuity
    indices = bars["bar_position_index_in_segment"].sort().to_list()
    expected_indices = list(range(len(indices)))
    if indices != expected_indices:
        # Find gaps
        gaps = [i for i in expected_indices if i not in indices]
        if gaps:
            errors.append(f"Missing bar indices: {gaps[:10]}...")  # Show first 10

    # Check exactly one first bar
    first_count = bars["bar_position_is_segment_first"].sum()
    if first_count != 1:
        errors.append(f"Expected 1 first bar, found {first_count}")

    # Check exactly one last bar
    last_count = bars["bar_position_is_segment_last"].sum()
    if last_count != 1:
        errors.append(f"Expected 1 last bar, found {last_count}")

    return len(errors) == 0, errors
