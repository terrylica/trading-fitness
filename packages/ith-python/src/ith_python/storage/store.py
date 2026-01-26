"""FeatureStore - Central feature storage with view generators.

This module provides the FeatureStore class which is the main interface
for storing and retrieving ITH features in Long Format.

Architecture: Multi-View Feature Architecture with Separation of Concerns
- Layer 2: Feature Storage
- See: docs/plans/2026-01-25-multi-view-feature-architecture-plan.md
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ith_python.features.config import FeatureConfig
from ith_python.storage.schemas import (
    validate_long_format,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class FeatureStore:
    """Central feature storage with view generators.

    The FeatureStore maintains features in Long Format (SSoT) and provides
    methods to generate various views on-demand.

    Long Format Schema:
        bar_index, symbol, threshold_dbps, lookback, feature, value, valid, computed_at, nav_hash

    Example:
        >>> store = FeatureStore.from_parquet("artifacts/ssot/features_long.parquet")
        >>> wide_df = store.to_wide(threshold=25)
        >>> nested = store.to_nested()
    """

    def __init__(self, df: pl.DataFrame) -> None:
        """Initialize FeatureStore from a Long Format DataFrame.

        Args:
            df: DataFrame in Long Format schema
        """
        is_valid, errors = validate_long_format(df)
        if not is_valid:
            msg = f"Invalid Long Format DataFrame: {errors}"
            raise ValueError(msg)
        self._df = df

    @property
    def df(self) -> pl.DataFrame:
        """Access the underlying Long Format DataFrame."""
        return self._df

    @classmethod
    def from_parquet(cls, path: Path | str) -> FeatureStore:
        """Load FeatureStore from a Parquet file.

        Args:
            path: Path to Parquet file in Long Format

        Returns:
            FeatureStore instance
        """
        path = Path(path)
        if not path.exists():
            msg = f"Parquet file not found: {path}"
            raise FileNotFoundError(msg)
        df = pl.read_parquet(path)
        return cls(df)

    @classmethod
    def from_wide(
        cls,
        wide_df: pl.DataFrame,
        symbol: str,
        threshold_dbps: int,
        nav_hash: str | None = None,
    ) -> FeatureStore:
        """Create FeatureStore from Wide Format DataFrame.

        Converts wide format (columns like ith_rb25_lb100_bull_ed) to Long Format.

        Args:
            wide_df: DataFrame with feature columns in wide format
            symbol: Symbol identifier
            threshold_dbps: Threshold used for computation
            nav_hash: Optional NAV hash for provenance

        Returns:
            FeatureStore instance
        """
        # Find feature columns
        feature_cols = [c for c in wide_df.columns if c.startswith("ith_")]

        # Parse column names and melt to long format
        records = []
        for col in feature_cols:
            # Parse column name: ith_rb{threshold}_lb{lookback}_{feature}
            match = re.match(r"ith_rb(\d+)_lb(\d+)_(.+)", col)
            if not match:
                continue

            col_threshold = int(match.group(1))
            lookback = int(match.group(2))
            feature = match.group(3)

            # Only include columns matching the specified threshold
            if col_threshold != threshold_dbps:
                continue

            for bar_index, value in enumerate(wide_df[col].to_list()):
                records.append({
                    "bar_index": bar_index,
                    "symbol": symbol,
                    "threshold_dbps": threshold_dbps,
                    "lookback": lookback,
                    "feature": feature,
                    "value": value,
                    "valid": value is not None and not (isinstance(value, float) and np.isnan(value)),
                    "computed_at": datetime.now(timezone.utc),
                    "nav_hash": nav_hash or "",
                })

        df = pl.DataFrame(records)

        # Cast to proper types
        df = df.with_columns([
            pl.col("bar_index").cast(pl.UInt32),
            pl.col("symbol").cast(pl.Categorical),
            pl.col("threshold_dbps").cast(pl.UInt16),
            pl.col("lookback").cast(pl.UInt16),
            pl.col("feature").cast(pl.Categorical),
            pl.col("value").cast(pl.Float64),
            pl.col("valid").cast(pl.Boolean),
        ])

        return cls(df)

    @classmethod
    def from_computation(
        cls,
        nav_data: dict[str, NDArray[np.float64]],
        config: FeatureConfig,
        threshold_dbps: int,
    ) -> FeatureStore:
        """Create FeatureStore by computing features from NAV data.

        Args:
            nav_data: Dict mapping symbol -> NAV array
            config: Feature computation configuration
            threshold_dbps: Threshold for column naming

        Returns:
            FeatureStore instance with computed features
        """
        from ith_python.features.compute import compute_features
        from ith_python.telemetry.provenance import fingerprint_array

        all_records = []
        computed_at = datetime.now(timezone.utc)

        for symbol, nav in nav_data.items():
            # Compute features
            nav_fp = fingerprint_array(nav, f"{symbol}_nav")
            config_with_threshold = FeatureConfig(
                lookbacks=config.lookbacks,
                threshold_dbps=threshold_dbps,
            )
            wide_df = compute_features(nav, config_with_threshold)

            # Convert to long format
            feature_cols = [c for c in wide_df.columns if c.startswith("ith_")]

            for col in feature_cols:
                match = re.match(r"ith_rb(\d+)_lb(\d+)_(.+)", col)
                if not match:
                    continue

                lookback = int(match.group(2))
                feature = match.group(3)

                for row in wide_df.iter_rows(named=True):
                    bar_index = row["bar_index"]
                    value = row[col]
                    all_records.append({
                        "bar_index": bar_index,
                        "symbol": symbol,
                        "threshold_dbps": threshold_dbps,
                        "lookback": lookback,
                        "feature": feature,
                        "value": value,
                        "valid": value is not None and not (isinstance(value, float) and np.isnan(value)),
                        "computed_at": computed_at,
                        "nav_hash": nav_fp["sha256"][:16],
                    })

        df = pl.DataFrame(all_records)

        # Cast to proper types
        df = df.with_columns([
            pl.col("bar_index").cast(pl.UInt32),
            pl.col("symbol").cast(pl.Categorical),
            pl.col("threshold_dbps").cast(pl.UInt16),
            pl.col("lookback").cast(pl.UInt16),
            pl.col("feature").cast(pl.Categorical),
            pl.col("value").cast(pl.Float64),
            pl.col("valid").cast(pl.Boolean),
        ])

        return cls(df)

    def save(self, path: Path | str) -> None:
        """Save FeatureStore to Parquet file.

        Args:
            path: Output path for Parquet file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._df.write_parquet(path)

    # =========================================================================
    # View Generators
    # =========================================================================

    def to_wide(
        self,
        threshold: int | None = None,
        symbol: str | None = None,
        drop_warmup: bool = True,
    ) -> pl.DataFrame:
        """Generate wide format suitable for ML training.

        Args:
            threshold: Filter by threshold_dbps (optional)
            symbol: Filter by symbol (optional)
            drop_warmup: If True, drop rows with any null values (warmup period)

        Returns:
            Wide format DataFrame with columns: bar_index, symbol, ith_rb{t}_lb{lb}_{f}
        """
        from ith_python.storage.views import to_wide
        return to_wide(self._df, threshold=threshold, symbol=symbol, drop_warmup=drop_warmup)

    def to_nested(self) -> list[dict]:
        """Generate nested JSON for semantic queries and API responses.

        Returns:
            List of nested dictionaries with structure:
            {bar_index, symbol, features: {rb{t}: {lb{lb}: {feature: value}}}}
        """
        from ith_python.storage.views import to_nested
        return to_nested(self._df)

    def to_dense(self, threshold: int) -> pl.DataFrame:
        """Generate dense format for a single threshold (no sparsity).

        Args:
            threshold: The threshold_dbps to filter by

        Returns:
            Dense DataFrame with only the specified threshold's features
        """
        from ith_python.storage.views import to_dense
        return to_dense(self._df, threshold=threshold)

    def to_clickhouse(self):
        """Generate Arrow RecordBatch for ClickHouse insertion.

        Returns:
            PyArrow RecordBatch suitable for ClickHouse bulk insert
        """
        from ith_python.storage.views import to_clickhouse
        return to_clickhouse(self._df)

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_symbols(self) -> list[str]:
        """Get list of unique symbols in the store."""
        return self._df["symbol"].unique().to_list()

    def get_thresholds(self) -> list[int]:
        """Get list of unique thresholds in the store."""
        return sorted(self._df["threshold_dbps"].unique().to_list())

    def get_lookbacks(self) -> list[int]:
        """Get list of unique lookbacks in the store."""
        return sorted(self._df["lookback"].unique().to_list())

    def get_features(self) -> list[str]:
        """Get list of unique feature names in the store."""
        return self._df["feature"].unique().to_list()

    def filter(
        self,
        symbol: str | None = None,
        threshold: int | None = None,
        lookback: int | None = None,
        feature: str | None = None,
    ) -> pl.DataFrame:
        """Filter the Long Format DataFrame.

        Args:
            symbol: Filter by symbol
            threshold: Filter by threshold_dbps
            lookback: Filter by lookback
            feature: Filter by feature name

        Returns:
            Filtered DataFrame in Long Format
        """
        df = self._df

        if symbol is not None:
            df = df.filter(pl.col("symbol") == symbol)
        if threshold is not None:
            df = df.filter(pl.col("threshold_dbps") == threshold)
        if lookback is not None:
            df = df.filter(pl.col("lookback") == lookback)
        if feature is not None:
            df = df.filter(pl.col("feature") == feature)

        return df

    def __len__(self) -> int:
        """Return number of rows in the store."""
        return len(self._df)

    def __repr__(self) -> str:
        """String representation of the store."""
        return (
            f"FeatureStore("
            f"rows={len(self):,}, "
            f"symbols={self.get_symbols()}, "
            f"thresholds={self.get_thresholds()}, "
            f"lookbacks={self.get_lookbacks()}"
            f")"
        )
