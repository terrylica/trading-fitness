"""Statistical examination runner - CLI entry point and orchestration.

Orchestrates the full statistical examination pipeline:
1. Load/compute ITH features from NAV data
2. Run statistical rigor analyses (correlation, stability, distribution, regime)
3. Run ML feature engineering analyses (importance, dimensionality, selection, temporal)
4. Output machine-readable artifacts (Parquet, NDJSON, JSON)

Data Flow:
    Rust compute_multiscale_ith() -> Arrow RecordBatch (zero-copy)
    -> Polars DataFrame (zero-copy) -> Statistical Analysis -> Parquet/NDJSON
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ith_python.ndjson_logger import setup_ndjson_logger, get_trace_id
from ith_python.statistical_examination._utils import drop_warmup, get_feature_columns
from ith_python.statistical_examination.cross_scale import compute_all_cross_scale_correlations
from ith_python.statistical_examination.dimensionality import compute_vif, perform_pca
from ith_python.statistical_examination.distribution import analyze_all_distributions
from ith_python.statistical_examination.regime import analyze_regime_dependence, detect_regime
from ith_python.statistical_examination.schemas import validate_ith_features
from ith_python.statistical_examination.selection import select_optimal_subset
from ith_python.statistical_examination.temporal import compute_autocorrelation, compute_stationarity
from ith_python.statistical_examination.threshold_stability import compute_all_threshold_stability

if TYPE_CHECKING:
    from collections.abc import Sequence

# Setup NDJSON logger for structured logging
logger = setup_ndjson_logger("statistical_examination")

# Default configuration
DEFAULT_THRESHOLDS = [25, 50, 100, 250, 500, 1000]
DEFAULT_LOOKBACKS = [20, 50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000]


def run_examination(
    nav: np.ndarray,
    thresholds: Sequence[int] = DEFAULT_THRESHOLDS,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    output_dir: Path | str | None = None,
    target_col: str | None = None,
    include_shap: bool = False,
    verbose: bool = True,
) -> dict:
    """Run complete statistical examination pipeline.

    Data flow:
    1. Rust compute_multiscale_ith() -> Arrow RecordBatch (zero-copy)
    2. pl.from_arrow() -> Polars DataFrame (zero-copy)
    3. Statistical analysis on Polars DataFrame
    4. df.write_parquet() for persistence

    Args:
        nav: NAV series (numpy array, starting at 1.0)
        thresholds: Range bar thresholds in dbps
        lookbacks: Lookback windows in bars
        output_dir: Directory for output artifacts (None = no file output)
        target_col: Optional target column name for supervised analysis
        include_shap: Whether to include SHAP analysis (slower)
        verbose: Whether to log progress

    Returns:
        Dict with all examination results and summary
    """
    from trading_fitness_metrics import MultiscaleIthConfig, compute_multiscale_ith

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    examination_id = f"exam_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    start_time = datetime.now(timezone.utc)
    trace_id = get_trace_id()

    if verbose:
        logger.bind(context={
            "examination_id": examination_id,
            "trace_id": trace_id,
            "nav_length": len(nav),
            "thresholds": list(thresholds),
            "lookbacks": list(lookbacks),
        }).info("Starting examination")

    # Step 1: Compute features for each threshold
    # Each threshold produces columns like ith_rbXXX_lbYYY_feature
    # We horizontally combine all thresholds since each row = same bar index
    phase_start = time.perf_counter()
    combined = None
    for threshold in thresholds:
        if verbose:
            logger.bind(context={
                "threshold_dbps": threshold,
                "phase": "feature_computation",
            }).debug("Computing features for threshold")

        config = MultiscaleIthConfig(threshold_dbps=threshold, lookbacks=list(lookbacks))
        features = compute_multiscale_ith(nav, config)

        # Arrow -> Polars (zero-copy)
        arrow_batch = features.to_arrow()
        df = pl.from_arrow(arrow_batch)

        if combined is None:
            combined = df
        else:
            # Horizontally join - all DataFrames have same row count
            combined = combined.hstack(df)

    feature_duration_ms = (time.perf_counter() - phase_start) * 1000

    if verbose:
        logger.bind(context={
            "phase": "feature_computation",
            "combined_shape": list(combined.shape),
            "duration_ms": round(feature_duration_ms, 2),
        }).info("Feature computation complete")

    # Validate features
    is_valid, errors = validate_ith_features(combined)
    if not is_valid:
        logger.bind(context={"errors": errors, "phase": "validation"}).warning(
            "Feature validation errors"
        )

    # Drop warmup rows
    combined_clean = drop_warmup(combined, lookbacks)

    # Get feature columns
    feature_cols = get_feature_columns(combined_clean)

    if verbose:
        logger.bind(context={
            "rows_after_warmup": len(combined_clean),
            "n_features": len(feature_cols),
            "phase": "data_preparation",
        }).info("Data preparation complete")

    # Initialize results
    results = {
        "examination_id": examination_id,
        "trace_id": trace_id,
        "timestamp": start_time.isoformat(),
        "config": {
            "nav_length": len(nav),
            "thresholds": list(thresholds),
            "lookbacks": list(lookbacks),
            "n_features": len(feature_cols),
        },
        "validation": {"is_valid": is_valid, "errors": errors},
    }

    # Step 2: Statistical Rigor Analyses
    phase_start = time.perf_counter()
    if verbose:
        logger.bind(context={"phase": "cross_scale_correlation"}).info(
            "Running cross-scale correlation analysis"
        )
    cross_scale_result = compute_all_cross_scale_correlations(
        combined_clean, thresholds=thresholds, method="spearman"
    )
    results["cross_scale_correlation"] = cross_scale_result.get("summary", {})
    cross_scale_duration_ms = (time.perf_counter() - phase_start) * 1000

    if verbose:
        logger.bind(context={
            "phase": "cross_scale_correlation",
            "mean_correlation": results["cross_scale_correlation"].get("overall_mean_correlation"),
            "duration_ms": round(cross_scale_duration_ms, 2),
        }).info("Cross-scale correlation complete")

    phase_start = time.perf_counter()
    if verbose:
        logger.bind(context={"phase": "threshold_stability"}).info(
            "Running threshold stability analysis"
        )
    threshold_stability_result = compute_all_threshold_stability(combined_clean)
    results["threshold_stability"] = threshold_stability_result.get("summary", {})
    threshold_duration_ms = (time.perf_counter() - phase_start) * 1000

    if verbose:
        logger.bind(context={
            "phase": "threshold_stability",
            "stable_features": results["threshold_stability"].get("stable_features"),
            "duration_ms": round(threshold_duration_ms, 2),
        }).info("Threshold stability complete")

    phase_start = time.perf_counter()
    if verbose:
        logger.bind(context={
            "phase": "distribution",
            "n_features_sampled": min(50, len(feature_cols)),
        }).info("Running distribution analysis")
    distribution_result = analyze_all_distributions(combined_clean, feature_cols[:50])  # Sample for speed
    results["distribution"] = distribution_result.get("summary", {})
    distribution_duration_ms = (time.perf_counter() - phase_start) * 1000

    if verbose:
        logger.bind(context={
            "phase": "distribution",
            "n_analyzed": results["distribution"].get("n_analyzed"),
            "duration_ms": round(distribution_duration_ms, 2),
        }).info("Distribution analysis complete")

    phase_start = time.perf_counter()
    if verbose:
        logger.bind(context={"phase": "regime_dependence"}).info(
            "Running regime dependence analysis"
        )
    # Need NAV for regime detection
    regimes = detect_regime(nav, lookback=min(lookbacks))
    # Truncate to match cleaned df
    regimes_clean = regimes[len(nav) - len(combined_clean) :]
    regime_results = analyze_regime_dependence(combined_clean, regimes_clean, feature_cols[:50])
    from ith_python.statistical_examination.regime import summarize_regime_dependence
    results["regime_dependence"] = summarize_regime_dependence(regime_results)
    regime_duration_ms = (time.perf_counter() - phase_start) * 1000

    if verbose:
        logger.bind(context={
            "phase": "regime_dependence",
            "regime_dependent_features": results["regime_dependence"].get("n_regime_dependent"),
            "duration_ms": round(regime_duration_ms, 2),
        }).info("Regime dependence analysis complete")

    # Step 3: ML Feature Engineering Analyses
    phase_start = time.perf_counter()
    if verbose:
        logger.bind(context={
            "phase": "pca",
            "n_features": len(feature_cols),
        }).info("Running PCA analysis")
    pca_result = perform_pca(combined_clean, feature_cols)
    results["pca"] = {
        "n_components_95_variance": pca_result.get("n_components_95_variance"),
        "effective_dimensionality": pca_result.get("effective_dimensionality"),
        "dimensionality_ratio": pca_result.get("dimensionality_ratio_95"),
    }
    pca_duration_ms = (time.perf_counter() - phase_start) * 1000

    if verbose:
        logger.bind(context={
            "phase": "pca",
            "n_components_95": results["pca"].get("n_components_95_variance"),
            "dimensionality_ratio": results["pca"].get("dimensionality_ratio"),
            "duration_ms": round(pca_duration_ms, 2),
        }).info("PCA analysis complete")

    phase_start = time.perf_counter()
    if verbose:
        logger.bind(context={
            "phase": "vif",
            "n_features_sampled": min(50, len(feature_cols)),
        }).info("Running VIF analysis")
    vif_df = compute_vif(combined_clean, feature_cols[:50])  # Sample for speed
    high_vif_count = vif_df.filter(pl.col("high_multicollinearity")).height
    results["vif"] = {
        "n_high_vif": high_vif_count,
        "high_vif_rate": high_vif_count / len(vif_df) if len(vif_df) > 0 else 0,
    }
    vif_duration_ms = (time.perf_counter() - phase_start) * 1000

    if verbose:
        logger.bind(context={
            "phase": "vif",
            "n_high_vif": high_vif_count,
            "high_vif_rate": results["vif"]["high_vif_rate"],
            "duration_ms": round(vif_duration_ms, 2),
        }).info("VIF analysis complete")

    phase_start = time.perf_counter()
    if verbose:
        logger.bind(context={
            "phase": "temporal",
            "n_features_sampled": min(50, len(feature_cols)),
        }).info("Running temporal analysis")
    acf_df = compute_autocorrelation(combined_clean, feature_cols[:50])
    stationarity_df = compute_stationarity(combined_clean, feature_cols[:50])
    n_stationary = stationarity_df.filter(pl.col("stationary").eq(True)).height

    # Compute median half-life from ACF results
    half_lives = [r["half_life"] for r in acf_df.iter_rows(named=True) if r["half_life"] is not None]
    median_half_life = float(np.median(half_lives)) if half_lives else None
    results["temporal"] = {
        "n_stationary": n_stationary,
        "stationarity_rate": n_stationary / len(stationarity_df) if len(stationarity_df) > 0 else 0,
        "median_half_life": median_half_life,
    }
    temporal_duration_ms = (time.perf_counter() - phase_start) * 1000

    if verbose:
        logger.bind(context={
            "phase": "temporal",
            "n_stationary": n_stationary,
            "stationarity_rate": results["temporal"]["stationarity_rate"],
            "duration_ms": round(temporal_duration_ms, 2),
        }).info("Temporal analysis complete")

    # Feature selection
    phase_start = time.perf_counter()
    if verbose:
        logger.bind(context={
            "phase": "feature_selection",
            "max_features": 24,
        }).info("Running feature selection")
    selection_result = select_optimal_subset(
        combined_clean,
        feature_cols,
        target_col=target_col,
        max_features=24,
    )
    results["feature_selection"] = {
        "initial_features": selection_result["initial_features"],
        "final_selected": selection_result["final_selected"],
        "reduction_ratio": selection_result["reduction_ratio"],
        "selected_features": selection_result["selected_features"],
    }
    selection_duration_ms = (time.perf_counter() - phase_start) * 1000

    if verbose:
        logger.bind(context={
            "phase": "feature_selection",
            "initial_features": results["feature_selection"]["initial_features"],
            "final_selected": results["feature_selection"]["final_selected"],
            "reduction_ratio": results["feature_selection"]["reduction_ratio"],
            "duration_ms": round(selection_duration_ms, 2),
        }).info("Feature selection complete")

    # Compute summary
    end_time = datetime.now(timezone.utc)
    total_duration = (end_time - start_time).total_seconds()
    results["execution"] = {
        "duration_seconds": total_duration,
        "end_time": end_time.isoformat(),
        "phase_durations_ms": {
            "feature_computation": round(feature_duration_ms, 2),
            "cross_scale_correlation": round(cross_scale_duration_ms, 2),
            "threshold_stability": round(threshold_duration_ms, 2),
            "distribution": round(distribution_duration_ms, 2),
            "regime_dependence": round(regime_duration_ms, 2),
            "pca": round(pca_duration_ms, 2),
            "vif": round(vif_duration_ms, 2),
            "temporal": round(temporal_duration_ms, 2),
            "feature_selection": round(selection_duration_ms, 2),
        },
    }

    # Step 4: Write outputs
    if output_dir is not None:
        if verbose:
            logger.bind(context={
                "phase": "output",
                "output_dir": str(output_dir),
            }).info("Writing outputs")

        # Features parquet
        combined_clean.write_parquet(output_dir / "features.parquet")

        # Summary JSON
        with open(output_dir / "summary.json", "w") as f:
            json.dump(_make_json_serializable(results), f, indent=2)

        # NDJSON examination log with proper schema
        with open(output_dir / "examination.ndjson", "w") as f:
            f.write(json.dumps({
                "ts": start_time.isoformat(),
                "event": "start",
                "examination_id": examination_id,
                "trace_id": trace_id,
                "config": results["config"],
            }) + "\n")
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "cross_scale",
                "examination_id": examination_id,
                "trace_id": trace_id,
                "summary": _make_json_serializable(results.get("cross_scale_correlation", {})),
                "duration_ms": results["execution"]["phase_durations_ms"]["cross_scale_correlation"],
            }) + "\n")
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "threshold_stability",
                "examination_id": examination_id,
                "trace_id": trace_id,
                "summary": _make_json_serializable(results.get("threshold_stability", {})),
                "duration_ms": results["execution"]["phase_durations_ms"]["threshold_stability"],
            }) + "\n")
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "distribution",
                "examination_id": examination_id,
                "trace_id": trace_id,
                "summary": _make_json_serializable(results.get("distribution", {})),
                "duration_ms": results["execution"]["phase_durations_ms"]["distribution"],
            }) + "\n")
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "regime",
                "examination_id": examination_id,
                "trace_id": trace_id,
                "summary": _make_json_serializable(results.get("regime_dependence", {})),
                "duration_ms": results["execution"]["phase_durations_ms"]["regime_dependence"],
            }) + "\n")
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "pca",
                "examination_id": examination_id,
                "trace_id": trace_id,
                "summary": _make_json_serializable(results.get("pca", {})),
                "duration_ms": results["execution"]["phase_durations_ms"]["pca"],
            }) + "\n")
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "feature_selection",
                "examination_id": examination_id,
                "trace_id": trace_id,
                "summary": _make_json_serializable(results.get("feature_selection", {})),
                "duration_ms": results["execution"]["phase_durations_ms"]["feature_selection"],
            }) + "\n")
            f.write(json.dumps({
                "ts": end_time.isoformat(),
                "event": "end",
                "examination_id": examination_id,
                "trace_id": trace_id,
                "duration_seconds": total_duration,
            }) + "\n")

        if verbose:
            logger.bind(context={
                "phase": "complete",
                "examination_id": examination_id,
                "output_dir": str(output_dir),
                "duration_seconds": round(total_duration, 2),
                "n_features": len(feature_cols),
                "files_written": ["features.parquet", "summary.json", "examination.ndjson"],
            }).info("Examination complete")

    return results


def _make_json_serializable(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def run_from_parquet(
    parquet_path: Path | str,
    output_dir: Path | str | None = None,
    target_col: str | None = None,
    verbose: bool = True,
) -> dict:
    """Run examination on pre-computed features from Parquet.

    Args:
        parquet_path: Path to features Parquet file
        output_dir: Directory for output artifacts
        target_col: Optional target column for supervised analysis
        verbose: Whether to log progress

    Returns:
        Dict with examination results
    """
    trace_id = get_trace_id()

    if verbose:
        logger.bind(context={
            "parquet_path": str(parquet_path),
            "trace_id": trace_id,
        }).info("Loading features from Parquet")

    df = pl.read_parquet(parquet_path)

    if verbose:
        logger.bind(context={
            "shape": list(df.shape),
            "n_rows": len(df),
            "n_cols": len(df.columns),
        }).info("DataFrame loaded")

    # Validate
    is_valid, errors = validate_ith_features(df)
    if not is_valid:
        logger.bind(context={
            "errors": errors,
            "phase": "validation",
        }).warning("Validation errors")

    feature_cols = get_feature_columns(df)

    if verbose:
        logger.bind(context={
            "n_features": len(feature_cols),
        }).info("Feature columns identified")

    # Run analyses (subset of full pipeline)
    results = {
        "source": str(parquet_path),
        "n_rows": len(df),
        "n_features": len(feature_cols),
    }

    # Cross-scale correlation
    cross_scale = compute_all_cross_scale_correlations(df, method="spearman")
    results["cross_scale_correlation"] = cross_scale.get("summary", {})

    # Distribution
    dist = analyze_all_distributions(df, feature_cols[:50])
    results["distribution"] = dist.get("summary", {})

    # PCA
    pca = perform_pca(df, feature_cols)
    results["pca"] = {
        "n_components_95": pca.get("n_components_95_variance"),
        "effective_dimensionality": pca.get("effective_dimensionality"),
    }

    # Temporal
    stationarity = compute_stationarity(df, feature_cols[:50])
    n_stationary = stationarity.filter(pl.col("stationary") == True).height  # noqa: E712
    results["temporal"] = {"n_stationary": n_stationary}

    # Feature selection
    selection = select_optimal_subset(df, feature_cols, target_col=target_col)
    results["feature_selection"] = selection

    # Write outputs
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "summary.json", "w") as f:
            json.dump(_make_json_serializable(results), f, indent=2)

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Statistical Examination Framework for ITH Multi-Scale Features"
    )
    parser.add_argument(
        "--nav-file",
        type=Path,
        help="Path to NAV CSV file (column: nav)",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        help="Path to pre-computed features Parquet file",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="25,50,100,250,500,1000",
        help="Comma-separated threshold values in dbps",
    )
    parser.add_argument(
        "--lookbacks",
        type=str,
        default="20,50,100,200,500,1000,1500,2000,3000,4000,5000,6000",
        help="Comma-separated lookback values in bars",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/statistical_examination"),
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        help="Target column name for supervised analysis",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # NDJSON logger is already configured at module import
    # No additional logging setup needed - loguru handles it

    if args.parquet:
        # Run from pre-computed Parquet
        results = run_from_parquet(
            args.parquet,
            output_dir=args.output_dir,
            target_col=args.target_col,
            verbose=args.verbose,
        )
    elif args.nav_file:
        # Load NAV using Polars and convert to numpy
        nav_df = pl.read_csv(args.nav_file)
        # Case-insensitive column lookup for NAV
        nav_col = None
        for col in nav_df.columns:
            if col.lower() == "nav":
                nav_col = col
                break
        if nav_col is None:
            logger.error("NAV file must have 'nav' or 'NAV' column")
            sys.exit(1)

        nav = nav_df.get_column(nav_col).to_numpy()

        thresholds = [int(t) for t in args.thresholds.split(",")]
        lookbacks = [int(lb) for lb in args.lookbacks.split(",")]

        results = run_examination(
            nav,
            thresholds=thresholds,
            lookbacks=lookbacks,
            output_dir=args.output_dir,
            target_col=args.target_col,
            verbose=args.verbose,
        )
    else:
        # Demo with synthetic NAV
        logger.bind(context={
            "mode": "demo",
            "nav_length": 10000,
        }).info("Running demo with synthetic NAV")

        np.random.seed(42)
        returns = np.random.randn(10000) * 0.01
        nav = np.cumprod(1 + returns)
        nav = nav / nav[0]  # Normalize to start at 1.0

        results = run_examination(
            nav,
            thresholds=[100, 250],  # Smaller set for demo
            lookbacks=[20, 50, 100, 200],  # Smaller set for demo
            output_dir=args.output_dir,
            verbose=args.verbose,
        )

    # Print summary
    print("\n" + "=" * 60)
    print("EXAMINATION SUMMARY")
    print("=" * 60)
    print(f"Features analyzed: {results.get('config', {}).get('n_features', 'N/A')}")
    print(f"Cross-scale mean correlation: {results.get('cross_scale_correlation', {}).get('overall_mean_correlation', 'N/A'):.3f}")
    print(f"Stable features: {results.get('threshold_stability', {}).get('stable_features', 'N/A')}")
    print(f"PCA 95% components: {results.get('pca', {}).get('n_components_95_variance', 'N/A')}")
    print(f"Stationary features: {results.get('temporal', {}).get('n_stationary', 'N/A')}")
    print(f"Selected features: {results.get('feature_selection', {}).get('final_selected', 'N/A')}")
    print("=" * 60)

    if args.output_dir:
        print(f"\nOutputs written to: {args.output_dir}")

    # Ensure loguru queue is flushed before exit (enqueue=True requires this)
    logger.complete()


if __name__ == "__main__":
    main()
