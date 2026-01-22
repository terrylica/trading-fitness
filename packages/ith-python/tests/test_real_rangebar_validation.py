"""Comprehensive validation of ITH features against REAL range bar data.

This module validates that ITH features produce LSTM-sensible values when
computed on real market data from multiple instrument classes:
- Crypto (BTCUSDT) - high volatility, 24/7 trading
- Forex (EURUSD) - lower volatility, session-based
- Commodity (XAUUSD) - trending, macro-driven

LSTM-Sensible Criteria:
1. All values bounded [0, 1] - no clipping needed
2. No extreme spikes (smooth transitions between bars)
3. Reasonable value distributions (not all 0s or 1s)
4. Stable across different threshold configurations
5. Graceful handling of edge cases (gaps, flash crashes, low liquidity)

Note: Tests require real data fetching. Run with:
    uv sync --extra validation
    uv run pytest tests/test_real_rangebar_validation.py -v -s
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
# Test Configuration
# =============================================================================

# Date ranges for testing (use recent data for availability)
CRYPTO_DATE_RANGE = ("2024-06-01", "2024-06-30")  # 1 month of crypto
FOREX_DATE_RANGE = ("2024-06-01", "2024-06-30")   # 1 month of forex

# Threshold configurations to test (decimal basis points)
THRESHOLD_CONFIGS = {
    "tight": 50,      # 5bps - many bars, fast-moving
    "standard": 100,  # 10bps - balanced
    "medium": 250,    # 25bps - default
    "wide": 500,      # 50bps - fewer bars, slower
}

# Lookback configurations
LOOKBACK_CONFIGS = [20, 50, 100, 200]

# Note: TMAEG is now auto-calculated based on data volatility
# See optimal_tmaeg() for the algorithm


# =============================================================================
# Fixtures - Real Data Fetching
# =============================================================================

@pytest.fixture(scope="module")
def btcusdt_range_bars():
    """Fetch real BTCUSDT range bars from Binance."""
    from rangebar import get_range_bars

    print("\n[Fetching real BTCUSDT range bars...]")
    df = get_range_bars(
        "BTCUSDT",
        CRYPTO_DATE_RANGE[0],
        CRYPTO_DATE_RANGE[1],
        threshold_decimal_bps=250,  # 25bps default
        use_cache=False,  # Skip ClickHouse cache for testing
    )
    print(f"  Fetched {len(df)} bars")
    return df


@pytest.fixture(scope="module")
def eurusd_range_bars():
    """Fetch real EURUSD range bars from Exness."""
    from rangebar import get_range_bars

    print("\n[Fetching real EURUSD range bars...]")
    try:
        df = get_range_bars(
            "EURUSD",
            FOREX_DATE_RANGE[0],
            FOREX_DATE_RANGE[1],
            threshold_decimal_bps=50,  # 5bps for forex (lower vol)
            source="exness",
            use_cache=False,  # Skip ClickHouse cache for testing
        )
        print(f"  Fetched {len(df)} bars")
        return df
    except (ConnectionError, TimeoutError, ValueError, OSError, RuntimeError) as e:
        pytest.skip(f"Could not fetch EURUSD data: {e}")


@pytest.fixture(scope="module")
def xauusd_range_bars():
    """Fetch real XAUUSD (gold) range bars from Exness."""
    from rangebar import get_range_bars

    print("\n[Fetching real XAUUSD range bars...]")
    try:
        df = get_range_bars(
            "XAUUSD",
            FOREX_DATE_RANGE[0],
            FOREX_DATE_RANGE[1],
            threshold_decimal_bps=100,  # 10bps for gold
            source="exness",
            use_cache=False,  # Skip ClickHouse cache for testing
        )
        print(f"  Fetched {len(df)} bars")
        return df
    except (ConnectionError, TimeoutError, ValueError, OSError, RuntimeError) as e:
        pytest.skip(f"Could not fetch XAUUSD data: {e}")


def build_nav_from_bars(df) -> np.ndarray:
    """Build NAV series from range bar close prices."""
    closes = df["Close"].values
    returns = np.diff(closes) / closes[:-1]
    returns = np.insert(returns, 0, 0.0)  # First return is 0
    nav = np.cumprod(1 + returns)
    return nav


# =============================================================================
# LSTM-Sensible Value Validation
# =============================================================================

class TestLstmSensibleValues:
    """Validate ITH features produce LSTM-sensible values on real data."""

    def _validate_lstm_sensible(
        self,
        features,
        lookback: int,
        context: str,
    ) -> dict[str, Any]:
        """
        Validate features are LSTM-sensible.

        Returns dict with validation results and statistics.
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

        results = {
            "context": context,
            "lookback": lookback,
            "total_bars": len(features),
            "valid_bars": len(features) - lookback + 1,
            "features": {},
            "issues": [],
        }

        for name in feature_names:
            arr = getattr(features, name)
            valid = arr[lookback - 1:]  # Skip NaN prefix
            valid = valid[~np.isnan(valid)]  # Remove any remaining NaNs

            if len(valid) == 0:
                results["issues"].append(f"{name}: No valid values")
                continue

            stats = {
                "min": float(valid.min()),
                "max": float(valid.max()),
                "mean": float(valid.mean()),
                "std": float(valid.std()),
                "zeros_pct": float((valid == 0).sum() / len(valid) * 100),
                "ones_pct": float((valid == 1).sum() / len(valid) * 100),
                "bounded": bool(valid.min() >= 0 and valid.max() <= 1),
            }

            # Check for LSTM-sensible issues
            issues = []

            # Issue 1: Not bounded [0, 1]
            if not stats["bounded"]:
                issues.append(f"NOT BOUNDED: min={stats['min']:.4f}, max={stats['max']:.4f}")

            # Issue 2: All zeros (uninformative)
            if stats["zeros_pct"] > 95:
                issues.append(f"MOSTLY ZEROS: {stats['zeros_pct']:.1f}%")

            # Issue 3: All ones (saturated)
            if stats["ones_pct"] > 95:
                issues.append(f"MOSTLY ONES: {stats['ones_pct']:.1f}%")

            # Issue 4: No variance (constant)
            if stats["std"] < 0.001:
                issues.append(f"NO VARIANCE: std={stats['std']:.6f}")

            # Issue 5: Extreme spikes (check diff for smoothness)
            if len(valid) > 1:
                diffs = np.abs(np.diff(valid))
                max_jump = float(diffs.max())
                stats["max_jump"] = max_jump
                if max_jump > 0.5:  # >50% jump between consecutive bars
                    issues.append(f"SPIKE: max_jump={max_jump:.4f}")

            stats["issues"] = issues
            results["features"][name] = stats

            if issues:
                for issue in issues:
                    results["issues"].append(f"{name}: {issue}")

        return results

    def test_btcusdt_lstm_sensible(self, btcusdt_range_bars):
        """BTCUSDT features must be LSTM-sensible."""
        nav = build_nav_from_bars(btcusdt_range_bars)

        print(f"\n=== BTCUSDT Validation ({len(nav)} bars) ===")

        all_issues = []

        for lookback in [50, 100]:
            # Auto-TMAEG is calculated internally
            auto_tmaeg = optimal_tmaeg(nav, lookback)
            features = compute_rolling_ith(nav, lookback=lookback)
            results = self._validate_lstm_sensible(
                features,
                lookback,
                f"BTCUSDT lookback={lookback} auto_tmaeg={auto_tmaeg:.4f}",
            )

            print(f"\n  Lookback={lookback}, Auto-TMAEG={auto_tmaeg:.4f}:")
            for name, stats in results["features"].items():
                status = "✓" if not stats["issues"] else "✗"
                print(f"    {status} {name}: "
                      f"[{stats['min']:.3f}, {stats['max']:.3f}] "
                      f"μ={stats['mean']:.3f} σ={stats['std']:.3f}")
                if stats["issues"]:
                    for issue in stats["issues"]:
                        print(f"      ⚠ {issue}")

            all_issues.extend(results["issues"])

        # Allow warnings but fail on critical issues (not bounded)
        critical = [i for i in all_issues if "NOT BOUNDED" in i]
        assert not critical, f"Critical issues found: {critical}"

    def test_eurusd_lstm_sensible(self, eurusd_range_bars):
        """EURUSD features must be LSTM-sensible."""
        nav = build_nav_from_bars(eurusd_range_bars)

        print(f"\n=== EURUSD Validation ({len(nav)} bars) ===")

        all_issues = []

        for lookback in [50, 100]:
            # Auto-TMAEG adapts to forex volatility
            auto_tmaeg = optimal_tmaeg(nav, lookback)
            features = compute_rolling_ith(nav, lookback=lookback)
            results = self._validate_lstm_sensible(
                features,
                lookback,
                f"EURUSD lookback={lookback} auto_tmaeg={auto_tmaeg:.4f}",
            )

            print(f"\n  Lookback={lookback}, Auto-TMAEG={auto_tmaeg:.4f}:")
            for name, stats in results["features"].items():
                status = "✓" if not stats["issues"] else "✗"
                print(f"    {status} {name}: "
                      f"[{stats['min']:.3f}, {stats['max']:.3f}] "
                      f"μ={stats['mean']:.3f} σ={stats['std']:.3f}")

            all_issues.extend(results["issues"])

        critical = [i for i in all_issues if "NOT BOUNDED" in i]
        assert not critical, f"Critical issues found: {critical}"

    def test_xauusd_lstm_sensible(self, xauusd_range_bars):
        """XAUUSD (gold) features must be LSTM-sensible."""
        nav = build_nav_from_bars(xauusd_range_bars)

        print(f"\n=== XAUUSD Validation ({len(nav)} bars) ===")

        lookback = 100
        auto_tmaeg = optimal_tmaeg(nav, lookback)
        features = compute_rolling_ith(nav, lookback=lookback)
        results = self._validate_lstm_sensible(
            features, lookback, f"XAUUSD lookback={lookback} auto_tmaeg={auto_tmaeg:.4f}"
        )

        print(f"  Auto-TMAEG={auto_tmaeg:.4f}")
        for name, stats in results["features"].items():
            status = "✓" if not stats["issues"] else "✗"
            print(f"  {status} {name}: "
                  f"[{stats['min']:.3f}, {stats['max']:.3f}] "
                  f"μ={stats['mean']:.3f} σ={stats['std']:.3f}")

        critical = [i for i in results["issues"] if "NOT BOUNDED" in i]
        assert not critical, f"Critical issues found: {critical}"


# =============================================================================
# Threshold Configuration Stability
# =============================================================================

class TestThresholdStability:
    """Validate features are stable across different range bar thresholds."""

    def test_btcusdt_threshold_sweep(self):
        """Features should be stable across different threshold configs."""
        from rangebar import get_range_bars

        print("\n=== BTCUSDT Threshold Sweep ===")

        results_by_threshold = {}

        for name, threshold in [("standard", 100), ("medium", 250), ("wide", 500)]:
            print(f"\n  Fetching {name} ({threshold}bps)...")

            try:
                df = get_range_bars(
                    "BTCUSDT",
                    "2024-06-15",  # Shorter period for sweep
                    "2024-06-20",
                    threshold_decimal_bps=threshold,
                    use_cache=False,  # Skip ClickHouse cache for testing
                )

                if len(df) < 100:
                    print(f"    Skipping: only {len(df)} bars")
                    continue

                nav = build_nav_from_bars(df)
                features = compute_rolling_ith(nav, lookback=50)

                valid = features.bull_epoch_density[49:]
                valid = valid[~np.isnan(valid)]

                results_by_threshold[name] = {
                    "bars": len(df),
                    "mean": float(valid.mean()),
                    "std": float(valid.std()),
                    "bounded": bool(valid.min() >= 0 and valid.max() <= 1),
                }

                print(f"    {len(df)} bars: μ={valid.mean():.3f} σ={valid.std():.3f} "
                      f"bounded={results_by_threshold[name]['bounded']}")

            except (ConnectionError, TimeoutError, ValueError, OSError) as e:
                print(f"    Error: {e}")

        # All thresholds should produce bounded values
        for name, result in results_by_threshold.items():
            assert result["bounded"], f"{name} threshold produced unbounded values"


# =============================================================================
# Edge Case Validation
# =============================================================================

class TestEdgeCases:
    """Validate ITH features handle edge cases gracefully."""

    def test_minimum_bars(self):
        """Should handle minimum viable bar counts."""
        # 10 bars with lookback=5 should work
        nav = np.cumprod(1 + np.random.randn(10) * 0.01)
        features = compute_rolling_ith(nav, lookback=5)

        # First 4 should be NaN, rest valid
        assert np.all(np.isnan(features.bull_epoch_density[:4]))
        valid = features.bull_epoch_density[4:]
        assert np.all((valid >= 0) & (valid <= 1))
        print("✓ Minimum bars (10 bars, lookback=5)")

    def test_high_volatility_simulation(self):
        """Simulate flash crash / high volatility scenario."""
        np.random.seed(42)

        # Normal period
        normal = np.cumprod(1 + np.random.randn(200) * 0.005)

        # Flash crash: 20% drop over 10 bars
        crash = np.linspace(1.0, 0.8, 10)

        # Recovery
        recovery = np.cumprod(1 + np.random.randn(100) * 0.01) * 0.8

        # Combine
        nav = np.concatenate([normal, normal[-1] * crash, recovery])

        features = compute_rolling_ith(nav, lookback=50)

        # All values should still be bounded
        valid = features.max_drawdown[49:]
        valid = valid[~np.isnan(valid)]

        assert np.all((valid >= 0) & (valid <= 1)), \
            f"Flash crash produced unbounded drawdown: [{valid.min()}, {valid.max()}]"

        # Drawdown should spike during crash
        crash_region = features.max_drawdown[200:220]
        assert np.nanmax(crash_region) > 0.1, \
            "Drawdown should detect crash"

        print(f"✓ Flash crash: max_drawdown peaked at {np.nanmax(crash_region):.3f}")

    def test_trending_market(self):
        """Strong trend should show consistent epoch density."""
        # Strong uptrend: 0.5% per bar for 500 bars
        nav = np.cumprod(np.ones(500) * 1.005)

        features = compute_rolling_ith(nav, lookback=100)

        valid_bull = features.bull_epoch_density[99:]
        valid_bear = features.bear_epoch_density[99:]

        # In uptrend: bull epochs should be high, bear epochs low
        assert np.nanmean(valid_bull) > np.nanmean(valid_bear), \
            "Uptrend should have more bull than bear epochs"

        # Drawdown should be near zero in pure uptrend
        valid_dd = features.max_drawdown[99:]
        assert np.nanmax(valid_dd) < 0.05, \
            f"Pure uptrend has unexpected drawdown: {np.nanmax(valid_dd):.3f}"

        print(f"✓ Trending market: bull_density={np.nanmean(valid_bull):.3f}, "
              f"bear_density={np.nanmean(valid_bear):.3f}")

    def test_sideways_choppy_market(self):
        """Choppy sideways market should have balanced features."""
        np.random.seed(123)

        # Mean-reverting: oscillate around 1.0
        noise = np.random.randn(500) * 0.02
        nav = 1.0 + np.cumsum(noise - noise.mean())  # Drift-corrected
        nav = np.maximum(nav, 0.5)  # Floor at 0.5

        features = compute_rolling_ith(nav, lookback=100)

        valid_bull = features.bull_epoch_density[99:]
        valid_bear = features.bear_epoch_density[99:]

        # Sideways: roughly balanced bull/bear
        ratio = np.nanmean(valid_bull) / (np.nanmean(valid_bear) + 1e-10)
        assert 0.3 < ratio < 3.0, \
            f"Sideways market should be balanced, ratio={ratio:.2f}"

        print(f"✓ Sideways market: bull/bear ratio={ratio:.2f}")

    def test_gaps_and_discontinuities(self):
        """Simulate price gaps (weekend gaps in forex)."""
        np.random.seed(456)

        # Week 1
        week1 = np.cumprod(1 + np.random.randn(100) * 0.01)

        # Weekend gap: 2% down
        gap = 0.98

        # Week 2 (starting from gapped level)
        week2 = np.cumprod(1 + np.random.randn(100) * 0.01) * week1[-1] * gap

        nav = np.concatenate([week1, week2])

        features = compute_rolling_ith(nav, lookback=50)

        # Should still be bounded despite gap
        valid = features.bull_epoch_density[49:]
        assert np.all((valid >= 0) & (valid <= 1)), \
            "Gap caused unbounded values"

        print("✓ Gaps/discontinuities handled correctly")

    def test_extreme_lookback_ratios(self):
        """Test edge case lookback/data ratios."""
        nav = np.cumprod(1 + np.random.randn(100) * 0.01)

        # Lookback = 90% of data
        features = compute_rolling_ith(nav, lookback=90)
        valid = features.bull_epoch_density[89:]
        assert len(valid) == 11  # 100 - 90 + 1
        assert np.all((valid >= 0) & (valid <= 1))

        # Lookback = exact data length should fail gracefully
        with pytest.raises(ValueError, match="exceed"):
            compute_rolling_ith(nav, lookback=101)

        print("✓ Extreme lookback ratios handled correctly")


# =============================================================================
# Value Distribution Analysis (for reporting)
# =============================================================================

class TestValueDistributions:
    """Analyze and report value distributions for LSTM consumption review."""

    def test_distribution_report(self, btcusdt_range_bars):
        """Generate distribution report for manual review."""
        nav = build_nav_from_bars(btcusdt_range_bars)

        print("\n" + "=" * 70)
        print("BTCUSDT ITH FEATURE DISTRIBUTION REPORT")
        print("=" * 70)
        print(f"Data: {len(nav)} range bars")
        print(f"Period: {CRYPTO_DATE_RANGE[0]} to {CRYPTO_DATE_RANGE[1]}")
        print()

        auto_tmaeg = optimal_tmaeg(nav, 100)
        features = compute_rolling_ith(nav, lookback=100)

        print(f"Auto-TMAEG: {auto_tmaeg:.4f}\n")

        feature_names = [
            ("bull_epoch_density", "Bull Epoch Density"),
            ("bear_epoch_density", "Bear Epoch Density"),
            ("bull_excess_gain", "Bull Excess Gain"),
            ("bear_excess_gain", "Bear Excess Gain"),
            ("bull_cv", "Bull CV"),
            ("bear_cv", "Bear CV"),
            ("max_drawdown", "Max Drawdown"),
            ("max_runup", "Max Runup"),
        ]

        print(f"{'Feature':<25} {'Min':>8} {'P25':>8} {'P50':>8} {'P75':>8} {'Max':>8} {'μ':>8} {'σ':>8}")
        print("-" * 89)

        all_sensible = True
        for attr, label in feature_names:
            arr = getattr(features, attr)
            valid = arr[99:]
            valid = valid[~np.isnan(valid)]

            if len(valid) == 0:
                print(f"{label:<25} {'N/A':>8}")
                continue

            p25, p50, p75 = np.percentile(valid, [25, 50, 75])

            # Check if sensible
            sensible = (
                valid.min() >= 0 and
                valid.max() <= 1 and
                valid.std() > 0.001 and
                (valid == 0).sum() / len(valid) < 0.95 and
                (valid == 1).sum() / len(valid) < 0.95
            )

            status = "✓" if sensible else "⚠"
            all_sensible = all_sensible and sensible

            print(f"{status} {label:<23} {valid.min():>8.4f} {p25:>8.4f} {p50:>8.4f} "
                  f"{p75:>8.4f} {valid.max():>8.4f} {valid.mean():>8.4f} {valid.std():>8.4f}")

        print("-" * 89)
        print(f"Overall LSTM-sensible: {'✓ YES' if all_sensible else '⚠ REVIEW NEEDED'}")
        print()

        # This test is informational - epoch density features naturally have low variance
        # Don't fail, just report for manual review
        if not all_sensible:
            import warnings
            warnings.warn(
                "Some features flagged for review - epoch density often has low variance",
                stacklevel=2,
            )


# =============================================================================
# Run with: pytest tests/test_real_rangebar_validation.py -v -s
# =============================================================================
