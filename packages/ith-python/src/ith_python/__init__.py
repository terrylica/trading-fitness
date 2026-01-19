"""ITH Python - Investment Time Horizon Analysis.

Trading strategy fitness analysis using TMAEG/TMAER thresholds for
Bull (long) and Bear (short) position profitability.

Example:
    >>> from ith_python import bull_ith_numba, bear_ith_numba
    >>> from ith_python.paths import get_artifacts_dir
"""

__version__ = "1.0.1"

# Core algorithm modules
from ith_python import bull_ith_numba
from ith_python import bear_ith_numba

# Path utilities
from ith_python.paths import (
    get_artifacts_dir,
    get_custom_nav_dir,
    get_data_dir,
    get_log_dir,
    get_synth_bear_ithes_dir,
    get_synth_bull_ithes_dir,
    ensure_dirs,
)

# Shared metrics
from ith_python.metrics import calculate_cv, calculate_cv_sample

# Bull (Long) algorithm exports
from ith_python.bull_ith_numba import (
    BullExcessGainLossResult,
    bull_excess_gain_excess_loss,
    max_drawdown,
    generate_synthetic_nav as generate_bull_nav,
)

# Bear (Short) algorithm exports
from ith_python.bear_ith_numba import (
    BearExcessGainLossResult,
    bear_excess_gain_excess_loss,
    max_runup,
)

__all__ = [
    # Version
    "__version__",
    # Modules
    "bull_ith_numba",
    "bear_ith_numba",
    # Path utilities
    "get_artifacts_dir",
    "get_custom_nav_dir",
    "get_data_dir",
    "get_log_dir",
    "get_synth_bear_ithes_dir",
    "get_synth_bull_ithes_dir",
    "ensure_dirs",
    # Shared metrics
    "calculate_cv",
    "calculate_cv_sample",
    # Bull (Long) exports
    "BullExcessGainLossResult",
    "bull_excess_gain_excess_loss",
    "max_drawdown",
    "generate_bull_nav",
    # Bear (Short) exports
    "BearExcessGainLossResult",
    "bear_excess_gain_excess_loss",
    "max_runup",
]
