"""Shared test fixtures for ith-python tests."""

import sys
from pathlib import Path
from typing import NamedTuple

# Add tests directory to sys.path for fixtures module import
# This is required for test_bear_ith_edge_cases.py to import fixtures.edge_cases
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

import numpy as np
import pandas as pd
import pytest
import tempfile
import shutil
from scipy import stats


# === Bear Synthetic NAV Fixtures ===


class BearSyntheticNavParams(NamedTuple):
    """Parameters for generating synthetic bear market NAV data.

    Used by test fixtures to generate consistent test data without
    importing from bear_ith.py (avoiding circular imports).

    Supports two generation modes:
    1. Point-based (recommended): Set n_points, dates auto-generated
    2. Date-based (legacy): Set start_date/end_date
    """

    n_points: int | None = None  # If set, generates exactly this many points
    start_date: str = "2020-01-01"
    end_date: str = "2020-06-30"  # Shorter period for faster tests
    avg_period_return: float = -0.001  # Negative drift for bear market
    period_return_volatility: float = 0.008
    df: int = 5  # Degrees of freedom for t-distribution
    rally_prob: float = 0.05
    rally_magnitude_low: float = 0.001
    rally_magnitude_high: float = 0.003
    rally_recovery_prob: float = 0.05


@pytest.fixture
def generate_synthetic_bear_nav_func():
    """Factory fixture for generating synthetic bear market NAV data.

    Returns a function that can be called with optional params to generate NAV.
    This avoids importing from bear_ith.py in tests, reducing coupling.
    """

    def _generate(params: BearSyntheticNavParams | None = None) -> pd.DataFrame:
        if params is None:
            params = BearSyntheticNavParams()

        if params.n_points is not None:
            # Point-based mode
            n = params.n_points
            dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
        else:
            # Date-based mode (legacy)
            dates = pd.date_range(params.start_date, params.end_date)
            n = len(dates)

        # Generate period returns using t-distribution
        period_returns = stats.t.rvs(
            params.df,
            loc=params.avg_period_return,
            scale=params.period_return_volatility,
            size=n,
        )

        # Add dead cat bounces (rallies in bear market)
        rally = False
        for i in range(n):
            if rally:
                period_returns[i] += np.random.uniform(
                    params.rally_magnitude_low, params.rally_magnitude_high
                )
                if np.random.rand() < params.rally_recovery_prob:
                    rally = False
            elif np.random.rand() < params.rally_prob:
                rally = True

        # Use MULTIPLICATIVE returns to guarantee NAV stays positive
        period_returns = np.clip(period_returns, -0.99, None)
        walk = np.cumprod(1 + period_returns)

        nav = pd.DataFrame(data=walk, index=dates, columns=["NAV"])
        nav.index.name = "Date"
        nav["PnL"] = nav["NAV"].diff()
        nav["PnL"] = nav["PnL"].fillna(nav["NAV"].iloc[0] - 1)
        return nav

    return _generate


@pytest.fixture
def sample_nav_data() -> pd.DataFrame:
    """Create sample NAV data for testing."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    # Simple upward trend with some noise
    nav_values = 1.0 + np.cumsum(np.random.randn(100) * 0.01 + 0.001)
    nav = pd.DataFrame({"NAV": nav_values}, index=dates)
    nav.index.name = "Date"
    return nav


@pytest.fixture
def sample_nav_with_pnl(sample_nav_data: pd.DataFrame) -> pd.DataFrame:
    """Create sample NAV data with PnL column."""
    nav = sample_nav_data.copy()
    nav["PnL"] = nav["NAV"].diff()
    nav["PnL"] = nav["PnL"].fillna(nav["NAV"].iloc[0] - 1)
    return nav


@pytest.fixture
def sample_nav_array() -> np.ndarray:
    """Create sample NAV array for numba functions."""
    np.random.seed(42)
    returns = np.random.randn(100) * 0.01 + 0.001
    nav = np.cumprod(1 + returns)
    return nav


# === Point-Based Fixtures (Time-Agnostic) ===


@pytest.fixture
def sample_nav_array_100_points() -> np.ndarray:
    """Generate 100-point NAV array (no dates, pure point-based)."""
    np.random.seed(42)
    returns = 1 + np.random.randn(100) * 0.02 + 0.001
    return np.cumprod(returns) * 100


@pytest.fixture
def sample_nav_array_500_points() -> np.ndarray:
    """Generate 500-point NAV array (no dates, pure point-based)."""
    np.random.seed(42)
    returns = 1 + np.random.randn(500) * 0.02 + 0.001
    return np.cumprod(returns) * 100


@pytest.fixture
def sample_nav_array_1000_points() -> np.ndarray:
    """Generate 1000-point NAV array (no dates, pure point-based)."""
    np.random.seed(42)
    returns = 1 + np.random.randn(1000) * 0.02 + 0.001
    return np.cumprod(returns) * 100


@pytest.fixture(params=[100, 500, 1000])
def sample_nav_array_parametrized(request) -> np.ndarray:
    """Parametrized NAV array fixture for various point counts."""
    n_points = request.param
    np.random.seed(42)
    returns = 1 + np.random.randn(n_points) * 0.02 + 0.001
    return np.cumprod(returns) * 100


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def sample_csv_file(temp_dir: Path, sample_nav_with_pnl: pd.DataFrame) -> Path:
    """Create a sample CSV file for testing."""
    csv_path = temp_dir / "test_nav.csv"
    sample_nav_with_pnl.to_csv(csv_path)
    return csv_path


@pytest.fixture
def invalid_csv_file(temp_dir: Path) -> Path:
    """Create an invalid CSV file (missing NAV column)."""
    csv_path = temp_dir / "invalid.csv"
    df = pd.DataFrame({"Date": ["2020-01-01"], "Value": [1.0]})
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def empty_csv_file(temp_dir: Path) -> Path:
    """Create an empty CSV file."""
    csv_path = temp_dir / "empty.csv"
    csv_path.touch()
    return csv_path
