"""Scattering Transform for automatic multi-scale feature extraction.

Alternative approach to manual lookback selection:
- Extracts features at ALL temporal scales automatically
- Wavelet-based invariants stable across time shifts
- Sidesteps lb20/lb100/lb500 choice problem

GitHub Issue: https://github.com/terrylica/cc-skills/issues/21

Uses forked kymatio (scipy.special.sph_harm_y fix for scipy 1.14+).
"""

from __future__ import annotations

import numpy as np
import polars as pl


def _nearest_power_of_two(n: int) -> int:
    """Find nearest power of 2 >= n."""
    return 2 ** int(np.ceil(np.log2(n)))


def extract_scattering_features(
    df: pl.DataFrame,
    price_col: str = "Close",
    J: int = 6,
    Q: int = 8,
    *,
    normalize: bool = True,
    emit_telemetry: bool = False,
) -> pl.DataFrame:
    """Extract scattering transform features from price series.

    The scattering transform extracts multi-scale invariant features
    that are stable to time-shifts and deformations. This sidesteps
    the manual lookback selection (lb20 vs lb100 vs lb500).

    Args:
        df: DataFrame with price column
        price_col: Name of price column (default "Close")
        J: Number of octaves (scales), determines temporal resolution
           J=6 covers ~2^6=64 bar windows, J=9 covers ~512 bars
        Q: Number of wavelets per octave (frequency resolution)
           Q=8 is typical for audio/financial, Q=1 for sparse
        normalize: If True, normalize input to zero mean, unit variance
        emit_telemetry: If True, log events to NDJSON

    Returns:
        DataFrame with scattering coefficients as columns:
        - scat_order0: Zeroth-order (low-pass, mean)
        - scat_order1_j{j}: First-order at scale j
        - scat_order2_j{j1}_j{j2}: Second-order cross-scale
    """
    from kymatio.numpy import Scattering1D

    if price_col not in df.columns:
        msg = f"Price column '{price_col}' not found in DataFrame"
        raise ValueError(msg)

    # Get price series as numpy array
    prices = df[price_col].drop_nulls().to_numpy().astype(np.float32)
    n_samples = len(prices)

    if n_samples < 64:
        msg = f"Insufficient data for scattering: {n_samples} samples (need >= 64)"
        raise ValueError(msg)

    # Scattering requires power-of-2 length
    T = _nearest_power_of_two(n_samples)

    # Pad signal to power of 2
    if n_samples < T:
        pad_len = T - n_samples
        prices = np.pad(prices, (0, pad_len), mode="edge")

    # Normalize for stability
    if normalize:
        mean_val = np.mean(prices)
        std_val = np.std(prices)
        if std_val > 0:
            prices = (prices - mean_val) / std_val

    # Initialize scattering transform
    # J determines max scale: 2^J samples
    # Limit J to avoid exceeding signal length
    J_max = int(np.log2(T)) - 2
    J_use = min(J, J_max)

    scattering = Scattering1D(J=J_use, shape=(T,), Q=Q)

    # Compute scattering coefficients
    # Input shape: (batch, time)
    x = prices.reshape(1, -1)
    Sx = scattering(x)  # Shape: (1, n_paths, n_time_coeffs)

    # Sx has shape (batch=1, n_scattering_paths, n_time_samples)
    # We aggregate over time to get per-path statistics
    Sx = Sx[0]  # Remove batch dimension: (n_paths, n_time)

    # Create feature columns from scattering coefficients
    records = []
    n_paths = Sx.shape[0]

    for path_idx in range(n_paths):
        path_coeffs = Sx[path_idx]

        # Compute statistics over time dimension
        records.append({
            f"scat_path{path_idx}_mean": float(np.mean(path_coeffs)),
            f"scat_path{path_idx}_std": float(np.std(path_coeffs)),
            f"scat_path{path_idx}_max": float(np.max(path_coeffs)),
            f"scat_path{path_idx}_min": float(np.min(path_coeffs)),
        })

    # Flatten records into single row
    flat_record = {}
    for r in records:
        flat_record.update(r)

    # Add summary statistics
    flat_record["scat_n_paths"] = n_paths
    flat_record["scat_J"] = J_use
    flat_record["scat_Q"] = Q
    flat_record["scat_T"] = T
    flat_record["scat_n_samples"] = n_samples

    result = pl.DataFrame([flat_record])

    if emit_telemetry:
        _log_scattering_extraction(
            n_samples=n_samples,
            J=J_use,
            Q=Q,
            n_paths=n_paths,
        )

    return result


def extract_scattering_time_series(
    df: pl.DataFrame,
    price_col: str = "Close",
    J: int = 6,
    Q: int = 8,
    window_size: int = 256,
    stride: int = 64,
    *,
    normalize: bool = True,
    emit_telemetry: bool = False,
) -> pl.DataFrame:
    """Extract scattering features as rolling time series.

    Applies scattering transform to sliding windows over the price series,
    producing a time series of scattering features aligned with bar indices.

    Args:
        df: DataFrame with price and bar_index columns
        price_col: Name of price column
        J: Number of octaves
        Q: Wavelets per octave
        window_size: Size of sliding window (should be power of 2)
        stride: Step between windows
        normalize: Normalize each window
        emit_telemetry: Log to NDJSON

    Returns:
        DataFrame with bar_index and scattering features per window
    """
    from kymatio.numpy import Scattering1D

    if price_col not in df.columns:
        msg = f"Price column '{price_col}' not found"
        raise ValueError(msg)

    prices = df[price_col].to_numpy().astype(np.float32)
    n_samples = len(prices)

    if n_samples < window_size:
        msg = f"Insufficient data: {n_samples} < window_size={window_size}"
        raise ValueError(msg)

    # Ensure window_size is power of 2
    T = _nearest_power_of_two(window_size)
    if T != window_size:
        T = window_size  # Use as-is, kymatio will handle

    J_max = int(np.log2(T)) - 2
    J_use = min(J, J_max)

    scattering = Scattering1D(J=J_use, shape=(T,), Q=Q)

    # Get bar indices if available
    if "bar_index" in df.columns:
        bar_indices = df["bar_index"].to_numpy()
    else:
        bar_indices = np.arange(n_samples)

    all_records = []

    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        window = prices[start:end]

        # Handle NaNs
        if np.any(np.isnan(window)):
            continue

        if normalize:
            mean_val = np.mean(window)
            std_val = np.std(window)
            if std_val > 0:
                window = (window - mean_val) / std_val

        x = window.reshape(1, -1).astype(np.float32)
        Sx = scattering(x)[0]  # (n_paths, n_time)

        # Aggregate each path
        record = {"bar_index": int(bar_indices[end - 1])}  # Align with window end

        for path_idx in range(Sx.shape[0]):
            record[f"scat_p{path_idx}"] = float(np.mean(Sx[path_idx]))

        all_records.append(record)

    if not all_records:
        return pl.DataFrame(schema={"bar_index": pl.Int64})

    result = pl.DataFrame(all_records)

    if emit_telemetry:
        _log_scattering_extraction(
            n_samples=n_samples,
            J=J_use,
            Q=Q,
            n_paths=Sx.shape[0],
            window_size=window_size,
            n_windows=len(all_records),
        )

    return result


def get_scattering_summary(
    df: pl.DataFrame,
    price_col: str = "Close",
    J: int = 6,
    Q: int = 8,
) -> dict:
    """Get summary of scattering transform extraction.

    Args:
        df: DataFrame with price column
        price_col: Name of price column
        J: Number of octaves
        Q: Wavelets per octave

    Returns:
        Dict with scattering analysis metadata
    """
    try:
        result = extract_scattering_features(
            df, price_col=price_col, J=J, Q=Q, emit_telemetry=False
        )

        n_features = len([c for c in result.columns if c.startswith("scat_path")])
        n_paths = result["scat_n_paths"].item() if "scat_n_paths" in result.columns else 0

        return {
            "method": "ScatteringTransform",
            "J": J,
            "Q": Q,
            "n_paths": n_paths,
            "n_features": n_features,
            "T_used": result["scat_T"].item() if "scat_T" in result.columns else None,
            "n_samples": result["scat_n_samples"].item() if "scat_n_samples" in result.columns else None,
        }

    except ValueError as e:
        return {
            "method": "ScatteringTransform",
            "error": str(e),
            "J": J,
            "Q": Q,
        }


def _log_scattering_extraction(
    n_samples: int,
    J: int,
    Q: int,
    n_paths: int,
    window_size: int | None = None,
    n_windows: int | None = None,
) -> None:
    """Log scattering extraction to NDJSON telemetry."""
    try:
        from ith_python.ndjson_logger import log_ndjson_event

        log_ndjson_event(
            event_type="feature_extraction",
            method="scattering_transform",
            n_samples=n_samples,
            J=J,
            Q=Q,
            n_paths=n_paths,
            window_size=window_size,
            n_windows=n_windows,
        )
    except ImportError:
        pass


if __name__ == "__main__":
    import sys

    print("Scattering Transform Feature Extraction Module")
    print("=" * 50)
    print()
    print("Alternative to manual lookback selection (lb20/lb100/lb500).")
    print("Extracts multi-scale invariant features automatically.")
    print()
    print("Key properties:")
    print("  - Wavelet-based: captures patterns at ALL scales")
    print("  - Translation invariant: stable to time shifts")
    print("  - Non-expansive: bounded feature space")
    print()
    print("Usage:")
    print("  from ith_python.statistical_examination.scattering import (")
    print("      extract_scattering_features,")
    print("      extract_scattering_time_series,")
    print("  )")
    print("  features = extract_scattering_features(df, price_col='Close', J=6, Q=8)")
    print()
    print("Parameters:")
    print("  J: Number of octaves (scales). J=6 covers ~64 bars, J=9 ~512 bars")
    print("  Q: Wavelets per octave. Q=8 typical, Q=1 for sparse")
    print()
    print("GitHub Issue: https://github.com/terrylica/cc-skills/issues/21")

    sys.exit(0)
