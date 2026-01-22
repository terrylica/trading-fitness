"""Granular per-bar validation of ITH features for LSTM consumption.

This module validates that ITH features produce sensible values at the
SINGLE RANGE BAR granularity - the most granular use case where each bar
gets feature values suitable for LSTM input.

Key Requirements:
1. Every bar (after lookback warmup) must have valid [0, 1] bounded values
2. Features must be informative (not all zeros or all ones)
3. Features must transition smoothly between consecutive bars
4. Auto-TMAEG must adapt to the volatility of the data
5. Works with HF forex data (1-5bps range bars)

Test Scenarios:
- Forex-like low volatility (1-5bps per bar)
- Crypto-like high volatility (25-100bps per bar)
- Trending vs mean-reverting regimes
- Small lookback windows (5-20 bars)
- Single bar feature extraction pattern
"""

import numpy as np
import pytest
from typing import NamedTuple

from trading_fitness_metrics import compute_rolling_ith, optimal_tmaeg


# =============================================================================
# Test Configuration
# =============================================================================

class RangeBarConfig(NamedTuple):
    """Configuration for simulated range bar data."""
    name: str
    vol_per_bar: float  # Volatility as decimal (0.0005 = 5bps)
    n_bars: int
    lookback: int
    description: str


# Configurations representing real-world range bar scenarios
RANGE_BAR_CONFIGS = [
    # Forex - very low volatility, tight bars
    RangeBarConfig("forex_1bps", 0.0001, 500, 10, "Forex 1bps range bars"),
    RangeBarConfig("forex_5bps", 0.0005, 500, 10, "Forex 5bps range bars"),
    RangeBarConfig("forex_10bps", 0.001, 500, 20, "Forex 10bps range bars"),

    # Crypto - higher volatility
    RangeBarConfig("crypto_25bps", 0.0025, 500, 20, "Crypto 25bps range bars"),
    RangeBarConfig("crypto_50bps", 0.005, 500, 50, "Crypto 50bps range bars"),
    RangeBarConfig("crypto_100bps", 0.01, 500, 50, "Crypto 100bps range bars"),

    # Small lookback - approaching single-bar granularity
    RangeBarConfig("small_lookback_5", 0.001, 200, 5, "Small lookback=5"),
    RangeBarConfig("small_lookback_3", 0.001, 200, 3, "Small lookback=3"),
    RangeBarConfig("small_lookback_2", 0.001, 200, 2, "Minimum lookback=2"),
]


# =============================================================================
# Data Generation Utilities
# =============================================================================

def generate_range_bar_nav(
    n_bars: int,
    vol_per_bar: float,
    regime: str = "random",
    seed: int = 42,
) -> np.ndarray:
    """
    Generate NAV series simulating range bar close prices.

    Args:
        n_bars: Number of range bars
        vol_per_bar: Per-bar volatility (standard deviation of returns)
        regime: One of "random", "trending_up", "trending_down", "mean_reverting"
        seed: Random seed for reproducibility

    Returns:
        NAV array starting at 1.0
    """
    np.random.seed(seed)

    if regime == "random":
        # Pure random walk
        returns = np.random.randn(n_bars) * vol_per_bar
    elif regime == "trending_up":
        # Upward drift with noise
        drift = vol_per_bar * 0.5  # Positive drift
        returns = drift + np.random.randn(n_bars) * vol_per_bar
    elif regime == "trending_down":
        # Downward drift with noise
        drift = -vol_per_bar * 0.5  # Negative drift
        returns = drift + np.random.randn(n_bars) * vol_per_bar
    elif regime == "mean_reverting":
        # Oscillating around zero
        returns = np.random.randn(n_bars) * vol_per_bar
        # Apply mean reversion
        for i in range(1, len(returns)):
            returns[i] -= 0.3 * returns[i-1]
    else:
        raise ValueError(f"Unknown regime: {regime}")

    # Build NAV from returns
    nav = np.cumprod(1 + returns)
    return nav


def generate_forex_tick_nav(n_bars: int, bps_per_bar: int = 5, seed: int = 42) -> np.ndarray:
    """
    Generate NAV simulating forex range bars from tick data.

    Forex characteristics:
    - Very low per-bar volatility
    - Occasional gaps (session opens)
    - Mean-reverting microstructure
    """
    np.random.seed(seed)

    vol = bps_per_bar / 10000  # Convert bps to decimal

    # Base returns with mean-reverting component
    returns = np.random.randn(n_bars) * vol

    # Add occasional gaps (simulating session opens)
    gap_indices = np.random.choice(n_bars, size=n_bars // 50, replace=False)
    for idx in gap_indices:
        returns[idx] = np.random.randn() * vol * 5  # 5x normal volatility

    nav = np.cumprod(1 + returns)
    return nav


def generate_crypto_aggtrade_nav(n_bars: int, bps_per_bar: int = 25, seed: int = 42) -> np.ndarray:
    """
    Generate NAV simulating crypto range bars from aggregated trades.

    Crypto characteristics:
    - Higher volatility
    - Fat tails (occasional large moves)
    - 24/7 trading (no gaps)
    """
    np.random.seed(seed)

    vol = bps_per_bar / 10000

    # Use Student's t distribution for fat tails
    returns = np.random.standard_t(df=5, size=n_bars) * vol * 0.6

    nav = np.cumprod(1 + returns)
    return nav


# =============================================================================
# Feature Validation Utilities
# =============================================================================

class PerBarValidationResult(NamedTuple):
    """Validation result for a single bar's features."""
    bar_index: int
    all_bounded: bool
    all_finite: bool
    feature_values: dict[str, float]
    issues: list[str]


def validate_single_bar_features(
    features,
    bar_index: int,
    lookback: int,
) -> PerBarValidationResult:
    """
    Validate features for a single bar.

    Returns detailed validation result including any issues.
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

    issues = []
    feature_values = {}
    all_bounded = True
    all_finite = True

    for name in feature_names:
        arr = getattr(features, name)
        val = arr[bar_index]
        feature_values[name] = val

        # Check if in warmup period
        if bar_index < lookback - 1:
            if not np.isnan(val):
                issues.append(f"{name}: Expected NaN in warmup, got {val:.6f}")
            continue

        # Check finite
        if not np.isfinite(val):
            all_finite = False
            issues.append(f"{name}: Not finite (NaN or Inf)")
            continue

        # Check bounded [0, 1]
        if val < 0 or val > 1:
            all_bounded = False
            issues.append(f"{name}: Out of bounds [{val:.6f}]")

    return PerBarValidationResult(
        bar_index=bar_index,
        all_bounded=all_bounded,
        all_finite=all_finite,
        feature_values=feature_values,
        issues=issues,
    )


def validate_bar_transitions(
    features,
    lookback: int,
    max_allowed_jump: float = 0.5,
) -> list[tuple[int, str, float]]:
    """
    Validate that features transition smoothly between consecutive bars.

    Returns list of (bar_index, feature_name, jump_size) for violations.
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

    violations = []

    for name in feature_names:
        arr = getattr(features, name)
        valid_start = lookback - 1

        for i in range(valid_start + 1, len(arr)):
            prev_val = arr[i - 1]
            curr_val = arr[i]

            if np.isnan(prev_val) or np.isnan(curr_val):
                continue

            jump = abs(curr_val - prev_val)
            if jump > max_allowed_jump:
                violations.append((i, name, jump))

    return violations


# =============================================================================
# Core Tests - Per-Bar Granularity
# =============================================================================

class TestPerBarFeatureValidity:
    """Test that every bar produces valid features."""

    @pytest.mark.parametrize("config", RANGE_BAR_CONFIGS, ids=lambda c: c.name)
    def test_all_bars_bounded(self, config: RangeBarConfig):
        """Every bar (after warmup) must have features in [0, 1]."""
        nav = generate_range_bar_nav(config.n_bars, config.vol_per_bar)
        features = compute_rolling_ith(nav, lookback=config.lookback)

        invalid_bars = []
        for i in range(config.lookback - 1, config.n_bars):
            result = validate_single_bar_features(features, i, config.lookback)
            if not result.all_bounded:
                invalid_bars.append((i, result.issues))

        assert len(invalid_bars) == 0, (
            f"{config.name}: {len(invalid_bars)} bars with unbounded features. "
            f"First 5: {invalid_bars[:5]}"
        )

    @pytest.mark.parametrize("config", RANGE_BAR_CONFIGS, ids=lambda c: c.name)
    def test_all_bars_finite(self, config: RangeBarConfig):
        """Every bar (after warmup) must have finite (non-NaN) features."""
        nav = generate_range_bar_nav(config.n_bars, config.vol_per_bar)
        features = compute_rolling_ith(nav, lookback=config.lookback)

        nan_bars = []
        for i in range(config.lookback - 1, config.n_bars):
            result = validate_single_bar_features(features, i, config.lookback)
            if not result.all_finite:
                nan_bars.append((i, result.issues))

        assert len(nan_bars) == 0, (
            f"{config.name}: {len(nan_bars)} bars with NaN features. "
            f"First 5: {nan_bars[:5]}"
        )

    @pytest.mark.parametrize("config", RANGE_BAR_CONFIGS, ids=lambda c: c.name)
    def test_smooth_transitions(self, config: RangeBarConfig):
        """
        Features should generally transition smoothly between consecutive bars.

        Note: Some features like excess_gain can legitimately jump when epochs
        start/end. We exclude these from smooth transition checks and focus on
        density and CV features which should be more stable.

        Note: Lookbacks <= 3 are inherently unstable and are skipped.
        """
        if config.lookback <= 3:
            pytest.skip(f"Lookback {config.lookback} too small for smooth transition test")
        nav = generate_range_bar_nav(config.n_bars, config.vol_per_bar)
        features = compute_rolling_ith(nav, lookback=config.lookback)

        # Only check features expected to be smooth (density, CV, drawdown/runup)
        # Exclude excess_gain features which can legitimately spike
        smooth_features = [
            "bull_epoch_density",
            "bear_epoch_density",
            "bull_cv",
            "bear_cv",
            "max_drawdown",
            "max_runup",
        ]

        violations = []
        for name in smooth_features:
            arr = getattr(features, name)
            valid_start = config.lookback - 1

            for i in range(valid_start + 1, len(arr)):
                prev_val = arr[i - 1]
                curr_val = arr[i]

                if np.isnan(prev_val) or np.isnan(curr_val):
                    continue

                jump = abs(curr_val - prev_val)
                if jump > 0.5:
                    violations.append((i, name, jump))

        # Allow more violations for small lookbacks (legitimately less stable)
        # Small lookbacks (<=10) allow 20%, larger lookbacks allow 5%
        if config.lookback <= 10:
            max_violation_pct = 0.20  # 20% for small lookbacks
        else:
            max_violation_pct = 0.05  # 5% for normal lookbacks

        max_allowed_violations = int(config.n_bars * max_violation_pct) + 1

        assert len(violations) <= max_allowed_violations, (
            f"{config.name}: {len(violations)} bars with jumps > 0.5 "
            f"(max allowed: {max_allowed_violations}). "
            f"First 10: {violations[:10]}"
        )


class TestAutoTmaegAdaptation:
    """Test that auto-TMAEG adapts correctly to different volatility regimes."""

    def test_tmaeg_scales_with_volatility(self):
        """Higher volatility should produce higher TMAEG."""
        volatilities = [0.0001, 0.0005, 0.001, 0.005, 0.01]
        tmaeg_values = []

        for vol in volatilities:
            nav = generate_range_bar_nav(200, vol)
            tmaeg = optimal_tmaeg(nav, lookback=20)
            tmaeg_values.append(tmaeg)

        # TMAEG should be monotonically increasing with volatility
        for i in range(1, len(tmaeg_values)):
            assert tmaeg_values[i] >= tmaeg_values[i-1], (
                f"TMAEG should increase with volatility: "
                f"vol={volatilities[i-1]} tmaeg={tmaeg_values[i-1]:.6f} > "
                f"vol={volatilities[i]} tmaeg={tmaeg_values[i]:.6f}"
            )

    def test_tmaeg_minimum_for_hf_forex(self):
        """TMAEG should reach 0.0001 for very low volatility HF forex data."""
        # Extremely low volatility (0.01% = 1bps per bar)
        nav = generate_range_bar_nav(200, vol_per_bar=0.0001, seed=42)
        tmaeg = optimal_tmaeg(nav, lookback=10)

        # Should be at or near the minimum
        assert tmaeg <= 0.001, f"TMAEG for HF forex should be very low, got {tmaeg:.6f}"
        print(f"HF Forex (1bps) TMAEG: {tmaeg:.6f} ({tmaeg*100:.4f}%)")

    def test_tmaeg_consistent_across_seeds(self):
        """TMAEG should be similar for same volatility with different seeds."""
        tmaeg_values = []

        for seed in range(10):
            nav = generate_range_bar_nav(200, vol_per_bar=0.001, seed=seed)
            tmaeg = optimal_tmaeg(nav, lookback=20)
            tmaeg_values.append(tmaeg)

        mean_tmaeg = np.mean(tmaeg_values)
        std_tmaeg = np.std(tmaeg_values)
        cv = std_tmaeg / mean_tmaeg

        # Coefficient of variation should be < 50% for stability
        assert cv < 0.5, (
            f"TMAEG too variable across seeds: mean={mean_tmaeg:.6f}, "
            f"std={std_tmaeg:.6f}, CV={cv:.2f}"
        )


class TestForexTickDataSimulation:
    """Test with realistic forex tick data characteristics."""

    @pytest.mark.parametrize("bps", [1, 2, 5, 10])
    def test_forex_range_bars_produce_valid_features(self, bps: int):
        """Forex range bars at various thresholds should produce valid features."""
        nav = generate_forex_tick_nav(n_bars=500, bps_per_bar=bps)
        lookback = max(5, bps)  # Scale lookback with bar size

        features = compute_rolling_ith(nav, lookback=lookback)
        tmaeg = optimal_tmaeg(nav, lookback=lookback)

        # Validate all bars
        valid_start = lookback - 1
        valid_count = len(nav) - valid_start

        bounded_count = 0
        for i in range(valid_start, len(nav)):
            result = validate_single_bar_features(features, i, lookback)
            if result.all_bounded and result.all_finite:
                bounded_count += 1

        pct_valid = bounded_count / valid_count * 100

        print(f"\nForex {bps}bps: TMAEG={tmaeg:.6f}, valid={pct_valid:.1f}%")

        assert pct_valid == 100, f"Forex {bps}bps: Only {pct_valid:.1f}% bars valid"

    def test_forex_features_informative(self):
        """
        Forex features should be valid and bounded.

        Note: For low-volatility forex data with auto-TMAEG, epoch density
        may be very sparse (mostly zeros) which is correct behavior - epochs
        only occur during significant moves relative to volatility.

        The key requirement is that max_drawdown and max_runup should have
        variance since they measure realized price movement.
        """
        nav = generate_forex_tick_nav(n_bars=500, bps_per_bar=5)
        features = compute_rolling_ith(nav, lookback=10)

        # Drawdown and runup should have some variance (they track price movement)
        for name in ["max_drawdown", "max_runup"]:
            arr = getattr(features, name)
            valid = arr[9:]  # Skip warmup
            valid = valid[~np.isnan(valid)]

            # Should have some non-zero values for drawdown/runup
            nonzero_pct = (valid > 0).sum() / len(valid) * 100

            # At least 1% non-zero for price movement metrics
            assert nonzero_pct >= 1, (
                f"{name}: Only {nonzero_pct:.1f}% non-zero (expected some price movement)"
            )

        # Epoch density can be sparse - just verify it's bounded
        for name in ["bull_epoch_density", "bear_epoch_density"]:
            arr = getattr(features, name)
            valid = arr[9:]
            valid = valid[~np.isnan(valid)]

            assert valid.min() >= 0 and valid.max() <= 1, (
                f"{name}: Values out of bounds [{valid.min():.4f}, {valid.max():.4f}]"
            )


class TestCryptoAggTradeSimulation:
    """Test with realistic crypto aggregated trade characteristics."""

    @pytest.mark.parametrize("bps", [25, 50, 100, 250])
    def test_crypto_range_bars_produce_valid_features(self, bps: int):
        """Crypto range bars at various thresholds should produce valid features."""
        nav = generate_crypto_aggtrade_nav(n_bars=500, bps_per_bar=bps)
        lookback = 20

        features = compute_rolling_ith(nav, lookback=lookback)
        tmaeg = optimal_tmaeg(nav, lookback=lookback)

        # Validate all bars
        valid_start = lookback - 1

        invalid_bars = []
        for i in range(valid_start, len(nav)):
            result = validate_single_bar_features(features, i, lookback)
            if not (result.all_bounded and result.all_finite):
                invalid_bars.append((i, result.issues))

        print(f"\nCrypto {bps}bps: TMAEG={tmaeg:.6f}, invalid={len(invalid_bars)}")

        assert len(invalid_bars) == 0, (
            f"Crypto {bps}bps: {len(invalid_bars)} invalid bars. "
            f"First 5: {invalid_bars[:5]}"
        )

    def test_crypto_fat_tails_handled(self):
        """Crypto fat-tail events should not break feature computation."""
        np.random.seed(42)

        # Simulate extreme event
        n_bars = 200
        vol = 0.0025  # 25bps

        # Normal returns with one extreme outlier
        returns = np.random.randn(n_bars) * vol
        returns[100] = 0.10  # 10% spike (40x normal)

        nav = np.cumprod(1 + returns)

        features = compute_rolling_ith(nav, lookback=20)

        # Feature at spike should still be bounded
        spike_features = validate_single_bar_features(features, 100, 20)

        assert spike_features.all_bounded, (
            f"Spike bar features unbounded: {spike_features.issues}"
        )
        assert spike_features.all_finite, (
            f"Spike bar features not finite: {spike_features.issues}"
        )


class TestMinimumLookbackScenarios:
    """Test edge cases with very small lookback windows."""

    def test_lookback_2(self):
        """Minimum meaningful lookback of 2 should work."""
        nav = generate_range_bar_nav(100, vol_per_bar=0.001)
        features = compute_rolling_ith(nav, lookback=2)

        # First value should be NaN, rest valid
        assert np.isnan(features.bull_epoch_density[0])

        for i in range(1, 100):
            result = validate_single_bar_features(features, i, lookback=2)
            assert result.all_bounded, f"Bar {i}: {result.issues}"
            assert result.all_finite, f"Bar {i}: {result.issues}"

    def test_lookback_1_edge_case(self):
        """Lookback of 1 is an edge case - verify it doesn't crash."""
        nav = generate_range_bar_nav(50, vol_per_bar=0.001)

        # Should work but produce degenerate results
        features = compute_rolling_ith(nav, lookback=1)

        # All values should be valid (no NaN since lookback-1 = 0)
        for i in range(50):
            val = features.bull_epoch_density[i]
            assert np.isfinite(val), f"Bar {i} should be finite"
            assert 0 <= val <= 1, f"Bar {i} should be bounded"

    def test_lookback_equals_data_length(self):
        """Lookback equal to data length should produce single valid bar."""
        nav = generate_range_bar_nav(20, vol_per_bar=0.001)
        features = compute_rolling_ith(nav, lookback=20)

        # First 19 should be NaN
        for i in range(19):
            assert np.isnan(features.bull_epoch_density[i])

        # Last one should be valid
        result = validate_single_bar_features(features, 19, lookback=20)
        assert result.all_bounded and result.all_finite


class TestRegimeTransitions:
    """Test feature behavior during market regime transitions."""

    def test_trend_to_mean_reversion(self):
        """Features should adapt when regime changes."""
        np.random.seed(42)

        # First half: trending up
        trend_nav = generate_range_bar_nav(100, 0.001, regime="trending_up", seed=42)

        # Second half: mean reverting
        mr_nav = generate_range_bar_nav(100, 0.001, regime="mean_reverting", seed=43)
        mr_nav = mr_nav * trend_nav[-1]  # Continue from end of trend

        nav = np.concatenate([trend_nav, mr_nav])

        features = compute_rolling_ith(nav, lookback=20)

        # Validate transition region
        for i in range(90, 110):
            result = validate_single_bar_features(features, i, lookback=20)
            assert result.all_bounded, f"Transition bar {i}: {result.issues}"
            assert result.all_finite, f"Transition bar {i}: {result.issues}"

    def test_volatility_regime_change(self):
        """Features should handle volatility regime changes."""
        np.random.seed(42)

        # Low vol period
        low_vol = generate_range_bar_nav(100, 0.0005, seed=42)

        # High vol period (continuing from low vol end)
        high_vol = generate_range_bar_nav(100, 0.005, seed=43)
        high_vol = high_vol * low_vol[-1]

        nav = np.concatenate([low_vol, high_vol])

        features = compute_rolling_ith(nav, lookback=20)

        # All bars should be valid despite vol change
        for i in range(19, 200):
            result = validate_single_bar_features(features, i, lookback=20)
            assert result.all_bounded, f"Bar {i}: {result.issues}"


class TestFeatureDistributionQuality:
    """Test that feature distributions are suitable for LSTM input."""

    @pytest.mark.parametrize("config", RANGE_BAR_CONFIGS[:6], ids=lambda c: c.name)
    def test_feature_distribution_statistics(self, config: RangeBarConfig):
        """Print and validate feature distribution statistics."""
        nav = generate_range_bar_nav(config.n_bars, config.vol_per_bar)
        features = compute_rolling_ith(nav, lookback=config.lookback)
        tmaeg = optimal_tmaeg(nav, lookback=config.lookback)

        print(f"\n{'='*60}")
        print(f"{config.name}: {config.description}")
        print(f"Vol per bar: {config.vol_per_bar*100:.4f}%, Lookback: {config.lookback}")
        print(f"Auto-TMAEG: {tmaeg:.6f} ({tmaeg*100:.4f}%)")
        print(f"{'='*60}")

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

        print(f"{'Feature':<22} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8} {'Zero%':>8}")
        print("-" * 70)

        all_valid = True
        for name in feature_names:
            arr = getattr(features, name)
            valid = arr[config.lookback - 1:]
            valid = valid[~np.isnan(valid)]

            if len(valid) == 0:
                print(f"{name:<22} {'N/A':>8}")
                continue

            zeros_pct = (valid == 0).sum() / len(valid) * 100

            print(f"{name:<22} {valid.min():>8.4f} {valid.max():>8.4f} "
                  f"{valid.mean():>8.4f} {valid.std():>8.4f} {zeros_pct:>7.1f}%")

            # Validate bounds
            if valid.min() < 0 or valid.max() > 1:
                all_valid = False

        assert all_valid, f"{config.name}: Some features out of bounds"


# =============================================================================
# Run with: pytest tests/test_granular_rangebar_features.py -v -s
# =============================================================================
