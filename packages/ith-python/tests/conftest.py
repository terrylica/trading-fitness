"""Shared test fixtures for ith-python tests."""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
from scipy import stats

# Add tests directory to sys.path for fixtures module import
# This is required for test_bear_ith_edge_cases.py to import fixtures.edge_cases
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))


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


# === ClickHouse Fixture for Rangebar Cache Tests ===


def _find_clickhouse_binary() -> Path | None:
    """Find the ClickHouse binary, PATH first and toolchain-agnostic.

    This used to look ONLY under ~/.local/share/mise/installs/clickhouse. mise was retired
    machine-wide (proto is the only toolchain manager), so that tree stopped existing and
    this returned None forever -- meaning every ClickHouse-dependent test here SILENTLY
    SKIPPED while ClickHouse was installed at /opt/homebrew/bin/clickhouse the whole time.

    A silent skip is worse than a crash: the suite stays green over a domain that has
    quietly become empty. The sibling repo gapless-crypto-clickhouse had the identical
    lookup but fail-hard semantics, so there it surfaced as 12 loud errors and was fixed
    the same day; here it produced no signal at all.

    PATH first is the tool-agnostic answer -- proto shims, Homebrew and system installs all
    feed PATH, so this needs no edit the next time the toolchain manager changes. The mise
    tree is searched LAST, only so a machine that still has one keeps working.
    """
    found = shutil.which("clickhouse")
    if found:
        return Path(found)

    for candidate in (
        Path.home() / ".proto/shims/clickhouse",
        Path("/opt/homebrew/bin/clickhouse"),
        Path("/usr/local/bin/clickhouse"),
    ):
        if candidate.is_file():
            return candidate

    mise_installs = Path.home() / ".local/share/mise/installs/clickhouse"
    if not mise_installs.exists():
        return None

    # Prefer 'latest' symlink, then find newest version
    latest = mise_installs / "latest/bin/clickhouse"
    if latest.exists():
        return latest

    # Find any version
    for version_dir in sorted(mise_installs.iterdir(), reverse=True):
        binary = version_dir / "bin/clickhouse"
        if binary.exists():
            return binary

    return None


def _is_clickhouse_running() -> bool:
    """Check if ClickHouse server is already running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "clickhouse.*server"],
            capture_output=True,
            timeout=5,
            check=False,  # pgrep returns non-zero when no match - that's expected
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _wait_for_clickhouse(timeout: int = 30) -> bool:
    """Wait for ClickHouse to accept connections."""
    binary = _find_clickhouse_binary()
    if not binary:
        return False

    start = time.time()
    while time.time() - start < timeout:
        try:
            result = subprocess.run(
                [str(binary), "client", "--query", "SELECT 1"],
                capture_output=True,
                timeout=5,
                check=False,  # We check returncode manually
            )
            if result.returncode == 0:
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            # Server not ready yet, keep waiting
            pass
        time.sleep(0.5)
    return False


@pytest.fixture(scope="session")
def clickhouse_server():
    """Start ClickHouse server for the test session, stop after.

    Usage:
        @pytest.mark.usefixtures("clickhouse_server")
        def test_with_clickhouse_cache():
            # ClickHouse is running
            bars = get_range_bars(..., use_cache=True)

    Or request directly:
        def test_something(clickhouse_server):
            if clickhouse_server is None:
                pytest.skip("ClickHouse not available")
    """
    binary = _find_clickhouse_binary()
    if binary is None:
        yield None
        return

    # Check if already running (external process)
    if _is_clickhouse_running():
        yield {"binary": binary, "started_by_fixture": False, "pid": None}
        return

    # Start ClickHouse server
    data_dir = Path.home() / ".clickhouse-local"
    data_dir.mkdir(exist_ok=True)

    proc = subprocess.Popen(
        [
            str(binary),
            "server",
            "--",
            f"--path={data_dir}",
            "--listen_host=127.0.0.1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # Don't kill with parent
    )

    # Wait for server to be ready
    if not _wait_for_clickhouse(timeout=30):
        proc.terminate()
        proc.wait(timeout=5)
        yield None
        return

    yield {"binary": binary, "started_by_fixture": True, "pid": proc.pid}

    # Cleanup: stop server if we started it
    try:
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.kill(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.fixture
def require_clickhouse(clickhouse_server):
    """Skip test if ClickHouse is not available.

    Usage:
        def test_cache_functionality(require_clickhouse):
            # Test only runs if ClickHouse is available
            ...
    """
    if clickhouse_server is None:
        pytest.skip("ClickHouse not available (no clickhouse binary found)")
    return clickhouse_server
