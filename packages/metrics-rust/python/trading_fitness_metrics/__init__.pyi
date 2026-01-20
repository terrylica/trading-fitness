"""Type stubs for trading-fitness-metrics."""

from typing import Sequence, overload
import numpy as np
from numpy.typing import NDArray

# ============================================================================
# Entropy Metrics
# ============================================================================

def permutation_entropy(
    prices: NDArray[np.float64] | Sequence[float],
    m: int = 3,
) -> float:
    """
    Compute permutation entropy of a price series.

    Args:
        prices: Array of prices
        m: Embedding dimension (default: 3)

    Returns:
        Permutation entropy value in [0, 1]

    Raises:
        ValueError: If prices array is empty or m < 2
    """
    ...

def sample_entropy(
    data: NDArray[np.float64] | Sequence[float],
    m: int = 2,
    r: float | None = None,
) -> float:
    """
    Compute sample entropy of a data series.

    Args:
        data: Array of data points
        m: Embedding dimension (default: 2)
        r: Tolerance (default: computed from data using MAD)

    Returns:
        Normalized sample entropy value in [0, 1)

    Raises:
        ValueError: If data array is empty
    """
    ...

def shannon_entropy(
    data: NDArray[np.float64] | Sequence[float],
    n_bins: int | None = None,
) -> float:
    """
    Compute Shannon entropy of a data series.

    Args:
        data: Array of data points
        n_bins: Number of histogram bins (default: Freedman-Diaconis)

    Returns:
        Normalized Shannon entropy value in [0, 1]

    Raises:
        ValueError: If data array is empty
    """
    ...

# ============================================================================
# Risk Metrics
# ============================================================================

def omega_ratio(
    returns: NDArray[np.float64] | Sequence[float],
    threshold: float = 0.0,
) -> float:
    """
    Compute Omega ratio of returns.

    Args:
        returns: Array of returns
        threshold: Threshold for gains/losses (default: 0.0)

    Returns:
        Normalized Omega ratio value in [0, 1)

    Raises:
        ValueError: If returns array is empty
    """
    ...

def ulcer_index(
    prices: NDArray[np.float64] | Sequence[float],
) -> float:
    """
    Compute Ulcer Index of a price series.

    Args:
        prices: Array of prices

    Returns:
        Ulcer Index value in [0, 1]

    Raises:
        ValueError: If prices array is empty
    """
    ...

def garman_klass_volatility(
    open: float,
    high: float,
    low: float,
    close: float,
) -> float:
    """
    Compute Garman-Klass volatility for a single OHLC bar.

    Args:
        open: Open price
        high: High price
        low: Low price
        close: Close price

    Returns:
        Normalized Garman-Klass volatility in [0, 1)

    Raises:
        ValueError: If OHLC values are invalid
    """
    ...

def kaufman_efficiency_ratio(
    prices: NDArray[np.float64] | Sequence[float],
) -> float:
    """
    Compute Kaufman Efficiency Ratio of a price series.

    Args:
        prices: Array of prices

    Returns:
        Efficiency ratio value in [0, 1]

    Raises:
        ValueError: If prices array has fewer than 2 elements
    """
    ...

# ============================================================================
# Fractal Metrics
# ============================================================================

def hurst_exponent(
    prices: NDArray[np.float64] | Sequence[float],
) -> float:
    """
    Compute Hurst exponent of a price series using DFA.

    Args:
        prices: Array of prices

    Returns:
        Hurst exponent value in [0, 1]

    Raises:
        ValueError: If prices array has fewer than 256 elements
    """
    ...

def fractal_dimension(
    prices: NDArray[np.float64] | Sequence[float],
    k_max: int = 10,
) -> float:
    """
    Compute fractal dimension of a price series using Higuchi's method.

    Args:
        prices: Array of prices
        k_max: Maximum scale parameter (default: 10)

    Returns:
        Normalized fractal dimension value in [0, 1]

    Raises:
        ValueError: If prices array is too short for k_max
    """
    ...

# ============================================================================
# NAV & Utility Functions
# ============================================================================

def build_nav_from_closes(
    closes: NDArray[np.float64] | Sequence[float],
) -> NDArray[np.float64]:
    """
    Build NAV series from close prices.

    Args:
        closes: Array of close prices

    Returns:
        NumPy array of NAV values

    Raises:
        ValueError: If closes array is empty
    """
    ...

def adaptive_windows(
    data_len: int,
    num_scales: int,
) -> list[int]:
    """
    Generate log-spaced window sizes for multi-scale analysis.

    Args:
        data_len: Length of data series
        num_scales: Number of window sizes to generate

    Returns:
        List of window sizes
    """
    ...

def optimal_bins_freedman_diaconis(
    data: NDArray[np.float64] | Sequence[float],
) -> int:
    """
    Compute optimal number of histogram bins using Freedman-Diaconis rule.

    Args:
        data: Array of data points

    Returns:
        Optimal number of bins

    Raises:
        ValueError: If data array has fewer than 4 elements
    """
    ...

def optimal_embedding_dimension(
    data: NDArray[np.float64] | Sequence[float],
    max_m: int,
) -> int:
    """
    Compute optimal embedding dimension for permutation entropy.

    Args:
        data: Array of data points
        max_m: Maximum embedding dimension to consider

    Returns:
        Optimal embedding dimension

    Raises:
        ValueError: If data array is empty
    """
    ...

def optimal_sample_entropy_tolerance(
    data: NDArray[np.float64] | Sequence[float],
) -> float:
    """
    Compute optimal tolerance for sample entropy using MAD.

    Args:
        data: Array of data points

    Returns:
        Optimal tolerance value

    Raises:
        ValueError: If data array is empty
    """
    ...

def relative_epsilon(operand: float) -> float:
    """
    Compute relative epsilon for adaptive division guards.

    Args:
        operand: The operand magnitude

    Returns:
        Adaptive epsilon value
    """
    ...

# ============================================================================
# ITH Analysis
# ============================================================================

class BullIthResult:
    """Result of Bull ITH (long position) analysis."""

    @property
    def num_of_epochs(self) -> int:
        """Number of bull epochs detected."""
        ...

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown observed."""
        ...

    @property
    def intervals_cv(self) -> float:
        """Coefficient of variation of epoch intervals."""
        ...

    def excess_gains(self) -> NDArray[np.float64]:
        """Get the excess gains array."""
        ...

    def excess_losses(self) -> NDArray[np.float64]:
        """Get the excess losses array."""
        ...

    def epochs(self) -> NDArray[np.bool_]:
        """Get the epochs boolean array."""
        ...

class BearIthResult:
    """Result of Bear ITH (short position) analysis."""

    @property
    def num_of_epochs(self) -> int:
        """Number of bear epochs detected."""
        ...

    @property
    def max_runup(self) -> float:
        """Maximum runup observed."""
        ...

    @property
    def intervals_cv(self) -> float:
        """Coefficient of variation of epoch intervals."""
        ...

    def excess_gains(self) -> NDArray[np.float64]:
        """Get the excess gains array."""
        ...

    def excess_losses(self) -> NDArray[np.float64]:
        """Get the excess losses array."""
        ...

    def epochs(self) -> NDArray[np.bool_]:
        """Get the epochs boolean array."""
        ...

def bull_ith(
    nav: NDArray[np.float64] | Sequence[float],
    tmaeg: float,
) -> BullIthResult:
    """
    Perform Bull ITH (long position) analysis.

    Args:
        nav: Array of NAV values
        tmaeg: Target Maximum Acceptable Excess Gain threshold

    Returns:
        BullIthResult with epochs, excess gains/losses, and statistics

    Raises:
        ValueError: If nav has fewer than 2 elements or tmaeg <= 0
    """
    ...

def bear_ith(
    nav: NDArray[np.float64] | Sequence[float],
    tmaeg: float,
) -> BearIthResult:
    """
    Perform Bear ITH (short position) analysis.

    Args:
        nav: Array of NAV values
        tmaeg: Target Maximum Acceptable Excess Gain threshold

    Returns:
        BearIthResult with epochs, excess gains/losses, and statistics

    Raises:
        ValueError: If nav has fewer than 2 elements or tmaeg <= 0
    """
    ...

# ============================================================================
# Stateful Normalizers
# ============================================================================

class GarmanKlassNormalizer:
    """EMA-based Garman-Klass volatility normalizer."""

    def __init__(self, expected_len: int) -> None:
        """
        Create a new normalizer.

        Args:
            expected_len: Expected sequence length for decay factor calculation
        """
        ...

    def normalize(self, raw: float) -> float:
        """
        Normalize a raw volatility value.

        Args:
            raw: Raw Garman-Klass volatility value

        Returns:
            Normalized value in (0, 1)
        """
        ...

    def reset(self) -> None:
        """Reset the normalizer state."""
        ...

class OnlineNormalizer:
    """Welford-based online normalizer."""

    def __init__(self, expected_len: int) -> None:
        """
        Create a new normalizer.

        Args:
            expected_len: Expected sequence length for decay factor calculation
        """
        ...

    def normalize(self, raw: float) -> float:
        """
        Normalize a raw value.

        Args:
            raw: Raw input value

        Returns:
            Normalized value in (0, 1)
        """
        ...

    def reset(self) -> None:
        """Reset the normalizer state."""
        ...

# ============================================================================
# Batch API
# ============================================================================

class MetricsResult:
    """Result containing all 9 metrics."""

    @property
    def permutation_entropy(self) -> float: ...
    @property
    def sample_entropy(self) -> float: ...
    @property
    def shannon_entropy(self) -> float: ...
    @property
    def omega_ratio(self) -> float: ...
    @property
    def ulcer_index(self) -> float: ...
    @property
    def garman_klass_vol(self) -> float: ...
    @property
    def kaufman_er(self) -> float: ...
    @property
    def hurst_exponent(self) -> float: ...
    @property
    def fractal_dimension(self) -> float: ...

    def all_bounded(self) -> bool:
        """Check if all metrics are bounded [0, 1]."""
        ...

    def has_nan(self) -> bool:
        """Check if any metric is NaN."""
        ...

def compute_all_metrics(
    prices: NDArray[np.float64] | Sequence[float],
    returns: NDArray[np.float64] | Sequence[float],
    ohlc: tuple[float, float, float, float] | None = None,
) -> MetricsResult:
    """
    Compute all 9 metrics in a single call.

    Args:
        prices: Array of prices
        returns: Array of returns
        ohlc: Optional tuple of (open, high, low, close) for Garman-Klass

    Returns:
        MetricsResult with all 9 metrics

    Raises:
        ValueError: If arrays are too short or OHLC is invalid
    """
    ...
