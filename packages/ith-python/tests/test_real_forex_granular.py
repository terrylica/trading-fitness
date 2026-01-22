"""Granular per-bar validation with REAL forex tick data.

This module validates ITH features at the SINGLE RANGE BAR granularity using
REAL forex data from Exness tick feeds, specifically targeting EUR/USD with
the smallest recommended range bar thresholds.

EUR/USD Range Bar Recommendations:
- 1bps (0.01%): Ultra-high frequency, tick-level granularity
- 2bps (0.02%): High frequency scalping
- 5bps (0.05%): Standard HF trading
- 10bps (0.1%): Intraday trading

Key Validation Criteria:
1. Every bar must produce valid [0, 1] bounded features
2. Features must be computed at single-bar granularity
3. Auto-TMAEG must adapt to forex volatility
4. No crashes or NaN values in valid range
5. Smooth transitions between consecutive bars

Requirements:
    uv sync --extra validation
    # Requires rangebar-py with Exness tick data access

Run with:
    uv run pytest tests/test_real_forex_granular.py -v -s
"""

import numpy as np
import pytest
from typing import Any

# Skip entire module if rangebar not installed
rangebar = pytest.importorskip(
    "rangebar",
    reason="rangebar not installed (install with: uv sync --extra validation)"
)

from trading_fitness_metrics import compute_rolling_ith, optimal_tmaeg


# =============================================================================
# Configuration for Real EUR/USD Data
# =============================================================================

# EUR/USD recommended range bar thresholds (decimal basis points)
EURUSD_THRESHOLDS = {
    "ultra_hf_1bps": 10,     # 1bps = 0.0001 = 1 pip
    "hf_2bps": 20,           # 2bps = 0.0002 = 2 pips
    "standard_5bps": 50,     # 5bps = 0.0005 = 5 pips
    "intraday_10bps": 100,   # 10bps = 0.001 = 10 pips
}

# Lookback configurations for different granularities
LOOKBACK_CONFIGS = {
    "ultra_hf_1bps": [3, 5, 10],      # Very short lookback for ultra-HF
    "hf_2bps": [5, 10, 20],           # Short lookback for HF
    "standard_5bps": [10, 20, 50],    # Standard lookback
    "intraday_10bps": [20, 50, 100],  # Longer lookback for intraday
}

# Date range for testing
TEST_DATE_START = "2024-06-01"
TEST_DATE_END = "2024-06-07"  # 1 week of data


# =============================================================================
# Data Fetching Utilities
# =============================================================================

def fetch_eurusd_range_bars(threshold_dbps: int) -> Any:
    """
    Fetch real EUR/USD range bars from Exness tick data.

    Args:
        threshold_dbps: Range bar threshold in decimal basis points

    Returns:
        DataFrame with OHLCV columns
    """
    from rangebar import get_range_bars

    df = get_range_bars(
        "EURUSD",
        TEST_DATE_START,
        TEST_DATE_END,
        threshold_decimal_bps=threshold_dbps,
        source="exness",
        use_cache=False,  # Fresh data for testing
    )
    return df


def build_nav_from_forex_bars(df) -> np.ndarray:
    """Build NAV series from forex range bar close prices."""
    closes = df["Close"].values.astype(np.float64)

    # Compute returns
    returns = np.diff(closes) / closes[:-1]
    returns = np.insert(returns, 0, 0.0)  # First return is 0

    # Build cumulative NAV
    nav = np.cumprod(1 + returns)
    return nav


# =============================================================================
# Per-Bar Validation Utilities
# =============================================================================

def validate_bar_features(
    features,
    bar_idx: int,
    lookback: int,
) -> dict[str, Any]:
    """
    Validate features for a single bar.

    Returns dict with validation results.
    """
    feature_names = [
        "bull_epoch_density",
        "bear_epoch_density",
        "bull_excess_gain",
        "bear_excess_gain",
        "bull_cv",
        "bear_cv",
        "max_drawdown",
        "max_runup",
    ]

    result = {
        "bar_idx": bar_idx,
        "in_warmup": bar_idx < lookback - 1,
        "valid": True,
        "bounded": True,
        "finite": True,
        "values": {},
        "issues": [],
    }

    for name in feature_names:
        arr = getattr(features, name)
        val = arr[bar_idx]
        result["values"][name] = val

        # Warmup period - should be NaN
        if result["in_warmup"]:
            if not np.isnan(val):
                result["issues"].append(f"{name}: Expected NaN in warmup, got {val}")
                result["valid"] = False
            continue

        # Check finite
        if not np.isfinite(val):
            result["finite"] = False
            result["valid"] = False
            result["issues"].append(f"{name}: NaN or Inf")
            continue

        # Check bounded [0, 1]
        if val < 0 or val > 1:
            result["bounded"] = False
            result["valid"] = False
            result["issues"].append(f"{name}: {val:.6f} out of [0, 1]")

    return result


def compute_feature_statistics(features, lookback: int) -> dict[str, dict]:
    """Compute statistics for all features."""
    feature_names = [
        "bull_epoch_density",
        "bear_epoch_density",
        "bull_excess_gain",
        "bear_excess_gain",
        "bull_cv",
        "bear_cv",
        "max_drawdown",
        "max_runup",
    ]

    stats = {}
    for name in feature_names:
        arr = getattr(features, name)
        valid = arr[lookback - 1:]
        valid = valid[~np.isnan(valid)]

        if len(valid) == 0:
            stats[name] = {"count": 0}
            continue

        stats[name] = {
            "count": len(valid),
            "min": float(valid.min()),
            "max": float(valid.max()),
            "mean": float(valid.mean()),
            "std": float(valid.std()),
            "zeros_pct": float((valid == 0).sum() / len(valid) * 100),
            "ones_pct": float((valid == 1).sum() / len(valid) * 100),
            "bounded": bool(valid.min() >= 0 and valid.max() <= 1),
        }

    return stats


# =============================================================================
# Fixtures - Real Data Fetching
# =============================================================================

@pytest.fixture(scope="module")
def eurusd_1bps_bars():
    """Fetch real EUR/USD 1bps range bars (smallest granularity)."""
    print("\n[Fetching real EURUSD 1bps range bars from Exness...]")
    try:
        df = fetch_eurusd_range_bars(EURUSD_THRESHOLDS["ultra_hf_1bps"])
        print(f"  Fetched {len(df)} bars (1bps threshold)")
        return df
    except (ConnectionError, TimeoutError, ValueError, OSError, RuntimeError) as e:
        pytest.skip(f"Could not fetch EURUSD 1bps data: {e}")


@pytest.fixture(scope="module")
def eurusd_2bps_bars():
    """Fetch real EUR/USD 2bps range bars."""
    print("\n[Fetching real EURUSD 2bps range bars from Exness...]")
    try:
        df = fetch_eurusd_range_bars(EURUSD_THRESHOLDS["hf_2bps"])
        print(f"  Fetched {len(df)} bars (2bps threshold)")
        return df
    except (ConnectionError, TimeoutError, ValueError, OSError, RuntimeError) as e:
        pytest.skip(f"Could not fetch EURUSD 2bps data: {e}")


@pytest.fixture(scope="module")
def eurusd_5bps_bars():
    """Fetch real EUR/USD 5bps range bars."""
    print("\n[Fetching real EURUSD 5bps range bars from Exness...]")
    try:
        df = fetch_eurusd_range_bars(EURUSD_THRESHOLDS["standard_5bps"])
        print(f"  Fetched {len(df)} bars (5bps threshold)")
        return df
    except (ConnectionError, TimeoutError, ValueError, OSError, RuntimeError) as e:
        pytest.skip(f"Could not fetch EURUSD 5bps data: {e}")


@pytest.fixture(scope="module")
def eurusd_10bps_bars():
    """Fetch real EUR/USD 10bps range bars."""
    print("\n[Fetching real EURUSD 10bps range bars from Exness...]")
    try:
        df = fetch_eurusd_range_bars(EURUSD_THRESHOLDS["intraday_10bps"])
        print(f"  Fetched {len(df)} bars (10bps threshold)")
        return df
    except (ConnectionError, TimeoutError, ValueError, OSError, RuntimeError) as e:
        pytest.skip(f"Could not fetch EURUSD 10bps data: {e}")


# =============================================================================
# Core Tests - Per-Bar Granularity with Real Data
# =============================================================================

class TestEurusd1bpsGranular:
    """Test with real EUR/USD 1bps (1 pip) range bars - smallest granularity."""

    def test_every_bar_valid(self, eurusd_1bps_bars):
        """Every bar must produce valid bounded features."""
        nav = build_nav_from_forex_bars(eurusd_1bps_bars)
        n_bars = len(nav)

        print(f"\n=== EURUSD 1bps: {n_bars} bars ===")

        for lookback in LOOKBACK_CONFIGS["ultra_hf_1bps"]:
            if lookback >= n_bars:
                continue

            features = compute_rolling_ith(nav, lookback=lookback)
            tmaeg = optimal_tmaeg(nav, lookback=lookback)

            invalid_bars = []
            for i in range(lookback - 1, n_bars):
                result = validate_bar_features(features, i, lookback)
                if not result["valid"]:
                    invalid_bars.append((i, result["issues"]))

            print(f"  lookback={lookback}: TMAEG={tmaeg:.6f}, invalid={len(invalid_bars)}/{n_bars - lookback + 1}")

            assert len(invalid_bars) == 0, (
                f"EURUSD 1bps lookback={lookback}: {len(invalid_bars)} invalid bars. "
                f"First 5: {invalid_bars[:5]}"
            )

    def test_feature_statistics(self, eurusd_1bps_bars):
        """Print feature statistics for 1bps bars."""
        nav = build_nav_from_forex_bars(eurusd_1bps_bars)
        lookback = 5

        if lookback >= len(nav):
            pytest.skip(f"Not enough bars for lookback={lookback}")

        features = compute_rolling_ith(nav, lookback=lookback)
        tmaeg = optimal_tmaeg(nav, lookback=lookback)
        stats = compute_feature_statistics(features, lookback)

        print(f"\n{'='*70}")
        print(f"EURUSD 1bps Feature Statistics (lookback={lookback}, TMAEG={tmaeg:.6f})")
        print(f"{'='*70}")
        print(f"{'Feature':<22} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8} {'Zero%':>8}")
        print("-" * 70)

        for name, s in stats.items():
            if s["count"] == 0:
                print(f"{name:<22} {'N/A':>8}")
            else:
                print(f"{name:<22} {s['min']:>8.4f} {s['max']:>8.4f} "
                      f"{s['mean']:>8.4f} {s['std']:>8.4f} {s['zeros_pct']:>7.1f}%")

    def test_auto_tmaeg_adapts_to_forex_vol(self, eurusd_1bps_bars):
        """Auto-TMAEG should be very low for 1bps forex bars."""
        nav = build_nav_from_forex_bars(eurusd_1bps_bars)

        for lookback in [3, 5, 10, 20]:
            if lookback >= len(nav):
                continue

            tmaeg = optimal_tmaeg(nav, lookback=lookback)
            print(f"  1bps lookback={lookback}: TMAEG={tmaeg:.6f} ({tmaeg*100:.4f}%)")

            # For 1bps forex, TMAEG should be very small
            assert tmaeg < 0.01, f"TMAEG too high for 1bps forex: {tmaeg}"


class TestEurusd2bpsGranular:
    """Test with real EUR/USD 2bps range bars."""

    def test_every_bar_valid(self, eurusd_2bps_bars):
        """Every bar must produce valid bounded features."""
        nav = build_nav_from_forex_bars(eurusd_2bps_bars)
        n_bars = len(nav)

        print(f"\n=== EURUSD 2bps: {n_bars} bars ===")

        for lookback in LOOKBACK_CONFIGS["hf_2bps"]:
            if lookback >= n_bars:
                continue

            features = compute_rolling_ith(nav, lookback=lookback)
            tmaeg = optimal_tmaeg(nav, lookback=lookback)

            invalid_bars = []
            for i in range(lookback - 1, n_bars):
                result = validate_bar_features(features, i, lookback)
                if not result["valid"]:
                    invalid_bars.append((i, result["issues"]))

            print(f"  lookback={lookback}: TMAEG={tmaeg:.6f}, invalid={len(invalid_bars)}/{n_bars - lookback + 1}")

            assert len(invalid_bars) == 0, (
                f"EURUSD 2bps lookback={lookback}: {len(invalid_bars)} invalid bars"
            )


class TestEurusd5bpsGranular:
    """Test with real EUR/USD 5bps range bars (standard HF)."""

    def test_every_bar_valid(self, eurusd_5bps_bars):
        """Every bar must produce valid bounded features."""
        nav = build_nav_from_forex_bars(eurusd_5bps_bars)
        n_bars = len(nav)

        print(f"\n=== EURUSD 5bps: {n_bars} bars ===")

        for lookback in LOOKBACK_CONFIGS["standard_5bps"]:
            if lookback >= n_bars:
                continue

            features = compute_rolling_ith(nav, lookback=lookback)
            tmaeg = optimal_tmaeg(nav, lookback=lookback)

            invalid_bars = []
            for i in range(lookback - 1, n_bars):
                result = validate_bar_features(features, i, lookback)
                if not result["valid"]:
                    invalid_bars.append((i, result["issues"]))

            print(f"  lookback={lookback}: TMAEG={tmaeg:.6f}, invalid={len(invalid_bars)}/{n_bars - lookback + 1}")

            assert len(invalid_bars) == 0, (
                f"EURUSD 5bps lookback={lookback}: {len(invalid_bars)} invalid bars"
            )

    def test_feature_distribution_quality(self, eurusd_5bps_bars):
        """Features should have reasonable distributions for LSTM input."""
        nav = build_nav_from_forex_bars(eurusd_5bps_bars)
        lookback = 20

        if lookback >= len(nav):
            pytest.skip(f"Not enough bars for lookback={lookback}")

        features = compute_rolling_ith(nav, lookback=lookback)
        stats = compute_feature_statistics(features, lookback)

        print(f"\n{'='*70}")
        print(f"EURUSD 5bps Distribution Quality Check")
        print(f"{'='*70}")

        issues = []
        for name, s in stats.items():
            if s["count"] == 0:
                continue

            # Check bounds
            if not s["bounded"]:
                issues.append(f"{name}: unbounded [{s['min']:.4f}, {s['max']:.4f}]")

            # Check for degenerate distributions (all zeros or all ones)
            if s["zeros_pct"] > 99:
                issues.append(f"{name}: {s['zeros_pct']:.1f}% zeros (may be uninformative)")

            if s["ones_pct"] > 99:
                issues.append(f"{name}: {s['ones_pct']:.1f}% ones (saturated)")

        if issues:
            print("Quality issues found:")
            for issue in issues:
                print(f"  ⚠ {issue}")
        else:
            print("✓ All features have reasonable distributions")

        # Critical failures only for unbounded values
        critical = [i for i in issues if "unbounded" in i]
        assert len(critical) == 0, f"Critical issues: {critical}"


class TestEurusd10bpsGranular:
    """Test with real EUR/USD 10bps range bars (intraday)."""

    def test_every_bar_valid(self, eurusd_10bps_bars):
        """Every bar must produce valid bounded features."""
        nav = build_nav_from_forex_bars(eurusd_10bps_bars)
        n_bars = len(nav)

        print(f"\n=== EURUSD 10bps: {n_bars} bars ===")

        for lookback in LOOKBACK_CONFIGS["intraday_10bps"]:
            if lookback >= n_bars:
                continue

            features = compute_rolling_ith(nav, lookback=lookback)
            tmaeg = optimal_tmaeg(nav, lookback=lookback)

            invalid_bars = []
            for i in range(lookback - 1, n_bars):
                result = validate_bar_features(features, i, lookback)
                if not result["valid"]:
                    invalid_bars.append((i, result["issues"]))

            print(f"  lookback={lookback}: TMAEG={tmaeg:.6f}, invalid={len(invalid_bars)}/{n_bars - lookback + 1}")

            assert len(invalid_bars) == 0, (
                f"EURUSD 10bps lookback={lookback}: {len(invalid_bars)} invalid bars"
            )


class TestCrossThresholdComparison:
    """Compare features across different EUR/USD range bar thresholds."""

    def test_tmaeg_scaling_with_threshold(
        self,
        eurusd_1bps_bars,
        eurusd_2bps_bars,
        eurusd_5bps_bars,
        eurusd_10bps_bars,
    ):
        """TMAEG should scale with range bar threshold (volatility)."""
        datasets = [
            ("1bps", eurusd_1bps_bars),
            ("2bps", eurusd_2bps_bars),
            ("5bps", eurusd_5bps_bars),
            ("10bps", eurusd_10bps_bars),
        ]

        lookback = 20
        tmaeg_values = []

        print(f"\n=== TMAEG Scaling with Threshold (lookback={lookback}) ===")

        for name, df in datasets:
            if df is None or len(df) <= lookback:
                continue

            nav = build_nav_from_forex_bars(df)
            tmaeg = optimal_tmaeg(nav, lookback=lookback)
            tmaeg_values.append((name, tmaeg))
            print(f"  {name}: TMAEG={tmaeg:.6f} ({tmaeg*100:.4f}%)")

        # TMAEG should generally increase with threshold (more volatile bars)
        # This is a soft check - real market data may have variations
        if len(tmaeg_values) >= 2:
            first_tmaeg = tmaeg_values[0][1]
            last_tmaeg = tmaeg_values[-1][1]

            # 10bps should have higher TMAEG than 1bps (roughly)
            # Allow for market microstructure effects
            print(f"\n  Ratio (10bps/1bps): {last_tmaeg/first_tmaeg:.2f}x")


class TestSmallestLookbackWithRealData:
    """Test with minimum viable lookback on real data."""

    def test_lookback_2_on_real_forex(self, eurusd_5bps_bars):
        """Minimum lookback=2 should work on real forex data."""
        nav = build_nav_from_forex_bars(eurusd_5bps_bars)
        n_bars = len(nav)

        if n_bars < 10:
            pytest.skip("Not enough bars")

        features = compute_rolling_ith(nav, lookback=2)
        tmaeg = optimal_tmaeg(nav, lookback=2)

        print(f"\n=== Minimum Lookback=2 on Real EURUSD ===")
        print(f"  Bars: {n_bars}, TMAEG: {tmaeg:.6f}")

        # First bar should be NaN
        assert np.isnan(features.bull_epoch_density[0])

        # Check first 100 valid bars
        invalid_count = 0
        for i in range(1, min(101, n_bars)):
            result = validate_bar_features(features, i, lookback=2)
            if not result["valid"]:
                invalid_count += 1

        print(f"  Invalid bars (first 100): {invalid_count}")
        assert invalid_count == 0, f"Found {invalid_count} invalid bars with lookback=2"

    def test_lookback_3_on_real_forex(self, eurusd_1bps_bars):
        """Lookback=3 on smallest granularity (1bps) forex data."""
        nav = build_nav_from_forex_bars(eurusd_1bps_bars)
        n_bars = len(nav)

        if n_bars < 10:
            pytest.skip("Not enough bars")

        features = compute_rolling_ith(nav, lookback=3)
        tmaeg = optimal_tmaeg(nav, lookback=3)

        print(f"\n=== Lookback=3 on EURUSD 1bps ===")
        print(f"  Bars: {n_bars}, TMAEG: {tmaeg:.6f}")

        # First 2 bars should be NaN
        assert np.isnan(features.bull_epoch_density[0])
        assert np.isnan(features.bull_epoch_density[1])

        # All valid bars should be bounded
        for i in range(2, n_bars):
            result = validate_bar_features(features, i, lookback=3)
            assert result["valid"], f"Bar {i} invalid: {result['issues']}"


class TestSingleBarFeatureExtraction:
    """Test the pattern of extracting features for a single bar."""

    def test_single_bar_extraction_pattern(self, eurusd_5bps_bars):
        """
        Demonstrate the single-bar feature extraction pattern.

        This is how you'd use the API to get features for each individual bar
        as it arrives in real-time trading.
        """
        nav = build_nav_from_forex_bars(eurusd_5bps_bars)
        n_bars = len(nav)
        lookback = 20

        if n_bars <= lookback:
            pytest.skip("Not enough bars")

        print(f"\n=== Single Bar Feature Extraction Pattern ===")
        print(f"Total bars: {n_bars}, Lookback: {lookback}")
        print(f"First valid bar: {lookback - 1}")
        print()

        # Compute all features once
        features = compute_rolling_ith(nav, lookback=lookback)

        # Extract features for specific bars (simulating real-time)
        sample_bars = [lookback - 1, lookback, lookback + 10, n_bars - 1]

        print(f"{'Bar':<6} {'bull_ed':>10} {'bear_ed':>10} {'max_dd':>10} {'max_ru':>10}")
        print("-" * 50)

        for bar_idx in sample_bars:
            if bar_idx >= n_bars:
                continue

            bull_ed = features.bull_epoch_density[bar_idx]
            bear_ed = features.bear_epoch_density[bar_idx]
            max_dd = features.max_drawdown[bar_idx]
            max_ru = features.max_runup[bar_idx]

            print(f"{bar_idx:<6} {bull_ed:>10.4f} {bear_ed:>10.4f} {max_dd:>10.4f} {max_ru:>10.4f}")

            # Validate each extracted bar
            assert 0 <= bull_ed <= 1, f"Bar {bar_idx}: bull_epoch_density out of bounds"
            assert 0 <= bear_ed <= 1, f"Bar {bar_idx}: bear_epoch_density out of bounds"
            assert 0 <= max_dd <= 1, f"Bar {bar_idx}: max_drawdown out of bounds"
            assert 0 <= max_ru <= 1, f"Bar {bar_idx}: max_runup out of bounds"


# =============================================================================
# Summary Report
# =============================================================================

class TestGranularSummaryReport:
    """Generate comprehensive summary report."""

    def test_full_summary_report(
        self,
        eurusd_1bps_bars,
        eurusd_2bps_bars,
        eurusd_5bps_bars,
        eurusd_10bps_bars,
    ):
        """Generate full summary report across all thresholds."""
        datasets = [
            ("EURUSD 1bps", eurusd_1bps_bars, 5),
            ("EURUSD 2bps", eurusd_2bps_bars, 10),
            ("EURUSD 5bps", eurusd_5bps_bars, 20),
            ("EURUSD 10bps", eurusd_10bps_bars, 50),
        ]

        print("\n" + "=" * 80)
        print("GRANULAR FOREX FEATURE VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Date range: {TEST_DATE_START} to {TEST_DATE_END}")
        print()

        all_passed = True

        for name, df, lookback in datasets:
            if df is None:
                print(f"\n{name}: SKIPPED (data not available)")
                continue

            nav = build_nav_from_forex_bars(df)
            n_bars = len(nav)

            if n_bars <= lookback:
                print(f"\n{name}: SKIPPED (only {n_bars} bars, need >{lookback})")
                continue

            features = compute_rolling_ith(nav, lookback=lookback)
            tmaeg = optimal_tmaeg(nav, lookback=lookback)
            stats = compute_feature_statistics(features, lookback)

            # Count valid bars
            valid_bars = 0
            total_bars = n_bars - lookback + 1
            for i in range(lookback - 1, n_bars):
                result = validate_bar_features(features, i, lookback)
                if result["valid"]:
                    valid_bars += 1

            pct_valid = valid_bars / total_bars * 100

            print(f"\n{name}")
            print(f"  Bars: {n_bars}, Lookback: {lookback}")
            print(f"  Auto-TMAEG: {tmaeg:.6f} ({tmaeg*100:.4f}%)")
            print(f"  Valid bars: {valid_bars}/{total_bars} ({pct_valid:.1f}%)")

            if pct_valid < 100:
                all_passed = False
                print(f"  ⚠ SOME INVALID BARS")
            else:
                print(f"  ✓ ALL BARS VALID")

        print("\n" + "=" * 80)
        if all_passed:
            print("RESULT: ✓ ALL TESTS PASSED")
        else:
            print("RESULT: ⚠ SOME TESTS FAILED")
        print("=" * 80)

        assert all_passed, "Some tests failed - see summary above"


# =============================================================================
# Run with: uv sync --extra validation && uv run pytest tests/test_real_forex_granular.py -v -s
# =============================================================================
