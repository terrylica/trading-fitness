"""Type stubs for trading-fitness-metrics."""

from typing import Sequence, overload, Any
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

def optimal_tmaeg(
    nav: NDArray[np.float64] | Sequence[float],
    lookback: int,
) -> float:
    """
    Compute optimal TMAEG threshold based on data volatility.

    Uses MAD (Median Absolute Deviation) based volatility estimation
    with sqrt(lookback) scaling. This is the same algorithm used internally
    by compute_rolling_ith().

    Formula: tmaeg = 3.0 × MAD_std × sqrt(lookback), clamped to [0.001, 0.50]
    where MAD_std = 1.4826 × MAD(returns)

    Args:
        nav: Array of NAV values
        lookback: Number of bars in the lookback window

    Returns:
        Optimal TMAEG threshold, clamped to [0.0001, 0.50]

    Raises:
        ValueError: If nav has fewer than 2 elements or lookback is 0

    Example:
        >>> import numpy as np
        >>> from trading_fitness_metrics import optimal_tmaeg
        >>> nav = np.cumprod(1 + np.random.randn(200) * 0.01)
        >>> tmaeg = optimal_tmaeg(nav, lookback=50)
        >>> print(f"Auto-TMAEG: {tmaeg:.4f}")
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
# Rolling ITH Features (Time-Agnostic, Bounded [0, 1])
# ============================================================================

class RollingIthFeatures:
    """
    Rolling ITH features - all bounded [0, 1] for LSTM consumption.

    All 8 feature arrays are time-agnostic (bar-based, not time-based) and
    suitable for use with range bars, tick bars, or any bar type.

    First `lookback - 1` values are NaN (insufficient data).
    """

    @property
    def bull_epoch_density(self) -> NDArray[np.float64]:
        """Bull epoch density: epochs / expected_epochs, saturated to [0, 1]."""
        ...

    @property
    def bear_epoch_density(self) -> NDArray[np.float64]:
        """Bear epoch density: epochs / expected_epochs, saturated to [0, 1]."""
        ...

    @property
    def bull_excess_gain(self) -> NDArray[np.float64]:
        """Bull excess gain (sum in window): tanh-normalized to [0, 1]."""
        ...

    @property
    def bear_excess_gain(self) -> NDArray[np.float64]:
        """Bear excess gain (sum in window): tanh-normalized to [0, 1]."""
        ...

    @property
    def bull_cv(self) -> NDArray[np.float64]:
        """Bull intervals CV: sigmoid-normalized to [0, 1]."""
        ...

    @property
    def bear_cv(self) -> NDArray[np.float64]:
        """Bear intervals CV: sigmoid-normalized to [0, 1]."""
        ...

    @property
    def max_drawdown(self) -> NDArray[np.float64]:
        """Max drawdown in window: already [0, 1]."""
        ...

    @property
    def max_runup(self) -> NDArray[np.float64]:
        """Max runup in window: already [0, 1]."""
        ...

    def __len__(self) -> int:
        """Get the length of the feature arrays."""
        ...

def compute_rolling_ith(
    nav: NDArray[np.float64] | Sequence[float],
    lookback: int,
) -> RollingIthFeatures:
    """
    Compute rolling ITH features over lookback windows.

    This function computes Bull and Bear ITH metrics over sliding windows
    of the NAV series, normalizing all outputs to [0, 1] for LSTM consumption.

    The TMAEG threshold is automatically calculated based on the data's volatility
    using MAD-based estimation with sqrt(lookback) scaling. This ensures sensible
    epoch density regardless of the bar type (range bars, tick bars, time bars)
    or instrument volatility. See `optimal_tmaeg()` for details.

    All 8 feature arrays are time-agnostic (bar-based, not time-based) and
    suitable for use with range bars, tick bars, or any bar type.

    Args:
        nav: Array of NAV values
        lookback: Number of bars to look back for each computation

    Returns:
        RollingIthFeatures with 8 bounded [0, 1] feature arrays:
        - bull_epoch_density: Normalized bull epoch count
        - bear_epoch_density: Normalized bear epoch count
        - bull_excess_gain: Normalized sum of bull excess gains
        - bear_excess_gain: Normalized sum of bear excess gains
        - bull_cv: Normalized bull intervals coefficient of variation
        - bear_cv: Normalized bear intervals coefficient of variation
        - max_drawdown: Maximum drawdown in window
        - max_runup: Maximum runup in window

    Note:
        First `lookback - 1` values are NaN (insufficient data for window).

    Raises:
        ValueError: If nav is empty or lookback is 0 or exceeds nav length

    Example:
        >>> import numpy as np
        >>> from trading_fitness_metrics import compute_rolling_ith, optimal_tmaeg
        >>> nav = np.cumprod(1 + np.random.randn(500) * 0.01)
        >>> features = compute_rolling_ith(nav, lookback=100)
        >>> features.bull_epoch_density[:99]  # First 99 are NaN
        >>> valid = features.bull_epoch_density[99:]  # All in [0, 1]
        >>>
        >>> # To inspect the auto-calculated TMAEG:
        >>> tmaeg = optimal_tmaeg(nav, lookback=100)
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

# ============================================================================
# Multi-Scale ITH Features (Arrow-Native Integration)
# ============================================================================

class MultiscaleIthConfig:
    """Configuration for multi-scale ITH computation.

    Column Naming Convention:
        Output columns follow the pattern: `ith_rb{threshold}_lb{lookback}_{feature}`
        - threshold: Range bar threshold in dbps (for column naming only)
        - lookback: Lookback window in bars
        - feature: Short feature name (e.g., `bull_ed`, `max_dd`)

    Feature Short Names:
        | Full Name | Short Name |
        |-----------|------------|
        | bull_epoch_density | bull_ed |
        | bear_epoch_density | bear_ed |
        | bull_excess_gain | bull_eg |
        | bear_excess_gain | bear_eg |
        | bull_cv | bull_cv |
        | bear_cv | bear_cv |
        | max_drawdown | max_dd |
        | max_runup | max_ru |
    """

    def __init__(
        self,
        threshold_dbps: int = 250,
        lookbacks: list[int] | None = None,
    ) -> None:
        """
        Create a new multi-scale ITH configuration.

        Args:
            threshold_dbps: Range bar threshold in dbps (for column naming only)
            lookbacks: List of lookback window sizes
                       (default: [20, 50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000])
        """
        ...

    @property
    def threshold_dbps(self) -> int:
        """Get the threshold in dbps."""
        ...

    @property
    def lookbacks(self) -> list[int]:
        """Get the list of lookback windows."""
        ...


class MultiscaleIthFeatures:
    """Multi-scale ITH features across multiple lookback windows.

    Contains features for all lookback × feature combinations.
    All values are bounded [0, 1] for LSTM consumption.

    Column naming follows: `ith_rb{threshold}_lb{lookback}_{feature}`

    Example:
        >>> import polars as pl
        >>> from trading_fitness_metrics import compute_multiscale_ith
        >>>
        >>> features = compute_multiscale_ith(nav, config)
        >>> df = pl.from_arrow(features.to_arrow())  # Zero-copy!
    """

    @property
    def n_points(self) -> int:
        """Get the number of data points."""
        ...

    @property
    def n_features(self) -> int:
        """Get the number of feature columns."""
        ...

    @property
    def config(self) -> MultiscaleIthConfig:
        """Get the configuration used for computation."""
        ...

    def to_arrow(self) -> Any:
        """
        Convert to Arrow RecordBatch (zero-copy via PyCapsule Interface).

        This method provides efficient zero-copy integration with Polars:

        Example:
            >>> import polars as pl
            >>> df = pl.from_arrow(features.to_arrow())  # Zero-copy!

        Returns:
            Arrow RecordBatch compatible with Polars, PyArrow, etc.
        """
        ...

    def get(self, column_name: str) -> NDArray[np.float64]:
        """
        Get a feature array by column name.

        Args:
            column_name: Name of the column (e.g., 'ith_rb250_lb100_bull_ed')

        Returns:
            NumPy array of feature values

        Raises:
            ValueError: If column name not found
        """
        ...

    def column_names(self) -> list[str]:
        """Get all column names (sorted)."""
        ...

    def all_bounded(self) -> bool:
        """Check if all features are bounded [0, 1]."""
        ...

    def __len__(self) -> int:
        """Get the number of data points."""
        ...


def compute_multiscale_ith(
    nav: NDArray[np.float64] | Sequence[float],
    config: MultiscaleIthConfig | None = None,
) -> MultiscaleIthFeatures:
    """
    Compute multi-scale ITH features across multiple lookback windows.

    This function computes rolling ITH features at multiple lookback scales,
    producing columnar outputs suitable for LSTM feature engineering.

    All features are bounded [0, 1] and follow the naming convention:
    `ith_rb{threshold}_lb{lookback}_{feature}`

    Args:
        nav: Array of NAV values
        config: MultiscaleIthConfig with threshold and lookbacks
                (default: 250 dbps, lookbacks [20, 50, 100, ..., 6000])

    Returns:
        MultiscaleIthFeatures with zero-copy Arrow conversion via `to_arrow()`.

    Raises:
        ValueError: If nav array is empty

    Example:
        >>> import polars as pl
        >>> import numpy as np
        >>> from trading_fitness_metrics import compute_multiscale_ith, MultiscaleIthConfig
        >>>
        >>> nav = np.cumprod(1 + np.random.randn(5000) * 0.01)
        >>> config = MultiscaleIthConfig(threshold_dbps=250, lookbacks=[100, 500, 1000])
        >>> features = compute_multiscale_ith(nav, config)
        >>>
        >>> # Zero-copy conversion to Polars
        >>> df = pl.from_arrow(features.to_arrow())
        >>> print(df.columns)  # ['ith_rb250_lb100_bull_ed', ...]
        >>>
        >>> # Or get individual feature arrays
        >>> bull_ed_100 = features.get('ith_rb250_lb100_bull_ed')
    """
    ...
