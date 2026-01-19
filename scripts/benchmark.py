#!/usr/bin/env python3
"""Benchmark trading fitness calculations across implementations."""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

# Add the ith-python package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages/ith-python/src"))

from ith_python.bull_ith_numba import _bull_excess_gain_excess_loss_numba


@njit
def _sharpe_ratio_numba(returns: np.ndarray, periods_per_year: float, risk_free_rate: float) -> float:
    """Numba-accelerated Sharpe ratio calculation."""
    n = len(returns)
    if n < 2:
        return np.nan

    total = 0.0
    for r in returns:
        total += r
    mean = total / n

    var_sum = 0.0
    for r in returns:
        var_sum += (r - mean) ** 2
    std_dev = np.sqrt(var_sum / (n - 1))

    if std_dev == 0:
        return np.nan

    excess_return = mean - risk_free_rate
    return np.sqrt(periods_per_year) * (excess_return / std_dev)


@njit
def _max_drawdown_numba(nav_values: np.ndarray) -> float:
    """Numba-accelerated max drawdown calculation."""
    if len(nav_values) == 0:
        return 0.0

    running_max = nav_values[0]
    max_dd = 0.0

    for nav in nav_values:
        if nav > running_max:
            running_max = nav
        drawdown = 1.0 - nav / running_max
        if drawdown > max_dd:
            max_dd = drawdown

    return max_dd


def generate_nav_series(n: int, seed: int = 42) -> np.ndarray:
    """Generate synthetic NAV series with realistic characteristics."""
    rng = np.random.default_rng(seed)
    # Start at 100, random walk with slight upward drift
    returns = rng.normal(0.0005, 0.02, n)  # 0.05% daily drift, 2% volatility
    nav = 100 * np.cumprod(1 + returns)
    return nav


def benchmark_python_numba(nav: np.ndarray, iterations: int = 100) -> dict:
    """Benchmark Numba-accelerated Python implementation."""
    hurdle = 0.05

    # Calculate returns for sharpe ratio
    returns = np.diff(nav) / nav[:-1]

    # Warm up JIT compilation
    _sharpe_ratio_numba(returns[:100], 252.0, 0.0)
    _max_drawdown_numba(nav[:100])
    _bull_excess_gain_excess_loss_numba(nav[:100], hurdle)

    # Benchmark sharpe_ratio
    start = time.perf_counter()
    for _ in range(iterations):
        _sharpe_ratio_numba(returns, 252.0, 0.0)
    sharpe_time = (time.perf_counter() - start) / iterations * 1000

    # Benchmark max_drawdown
    start = time.perf_counter()
    for _ in range(iterations):
        _max_drawdown_numba(nav)
    mdd_time = (time.perf_counter() - start) / iterations * 1000

    # Benchmark excess_gain_excess_loss (ITH)
    start = time.perf_counter()
    for _ in range(iterations):
        _bull_excess_gain_excess_loss_numba(nav, hurdle)
    ith_time = (time.perf_counter() - start) / iterations * 1000

    return {
        "sharpe_ratio_ms": sharpe_time,
        "max_drawdown_ms": mdd_time,
        "ith_analysis_ms": ith_time,
        "total_ms": sharpe_time + mdd_time + ith_time,
    }


def benchmark_rust(nav: np.ndarray, iterations: int = 100) -> dict | None:
    """Benchmark Rust implementation via CLI."""
    rust_dir = Path(__file__).parent.parent / "packages/core-rust"
    benchmark_bin = rust_dir / "target/release/benchmark"

    # Check if Rust benchmark binary exists
    if not benchmark_bin.exists():
        print("[Rust benchmark binary not found, skipping]")
        return None

    # Write NAV data to temp file
    nav_file = Path("/tmp/benchmark_nav.csv")
    np.savetxt(nav_file, nav, delimiter=",")

    try:
        result = subprocess.run(
            [str(benchmark_bin), str(nav_file), str(iterations)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,  # We handle failure explicitly below
        )

        if result.returncode != 0:
            print(f"Rust benchmark error: {result.stderr}")
            return None

        # Parse output
        metrics = {}
        for line in result.stdout.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":")
                metrics[key] = float(value)

        return metrics
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        print(f"Rust benchmark failed: {e}")
        return None


def benchmark_bun(nav: np.ndarray, iterations: int = 100) -> dict | None:
    """Benchmark Bun/TypeScript implementation."""
    bun_dir = Path(__file__).parent.parent / "packages/core-bun"

    # Write NAV data to temp file as JSON
    nav_file = Path("/tmp/benchmark_nav.json")
    import json
    with open(nav_file, "w") as f:
        json.dump(nav.tolist(), f)

    # Create benchmark script
    bench_script = bun_dir / "benchmark.ts"
    bench_code = f'''
import {{ sharpeRatio, maxDrawdown, pnlFromNav }} from "./src/metrics";
import {{ excessGainExcessLoss }} from "./src/ith";

const nav: number[] = await Bun.file("/tmp/benchmark_nav.json").json();
const iterations = {iterations};
const hurdle = 0.05;
const pnl = pnlFromNav(nav);

// Benchmark sharpe_ratio
let start = performance.now();
for (let i = 0; i < iterations; i++) {{
    sharpeRatio(pnl, 252, 0);
}}
const sharpeTime = (performance.now() - start) / iterations;

// Benchmark max_drawdown
start = performance.now();
for (let i = 0; i < iterations; i++) {{
    maxDrawdown(nav);
}}
const mddTime = (performance.now() - start) / iterations;

// Benchmark ITH analysis
start = performance.now();
for (let i = 0; i < iterations; i++) {{
    excessGainExcessLoss(nav, hurdle);
}}
const ithTime = (performance.now() - start) / iterations;

console.log(`sharpe_ratio_ms:${{sharpeTime}}`);
console.log(`max_drawdown_ms:${{mddTime}}`);
console.log(`ith_analysis_ms:${{ithTime}}`);
console.log(`total_ms:${{sharpeTime + mddTime + ithTime}}`);
'''

    with open(bench_script, "w") as f:
        f.write(bench_code)

    try:
        result = subprocess.run(
            ["bun", "run", str(bench_script)],
            cwd=bun_dir,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,  # We handle failure explicitly below
        )

        if result.returncode != 0:
            print(f"Bun benchmark error: {result.stderr}")
            return None

        # Parse output
        metrics = {}
        for line in result.stdout.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":")
                metrics[key] = float(value)

        return metrics
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        print(f"Bun benchmark failed: {e}")
        return None
    finally:
        bench_script.unlink(missing_ok=True)


def format_results(results: dict, name: str, baseline: dict | None = None) -> None:
    """Format and print benchmark results."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")

    for key, value in results.items():
        speedup = ""
        if baseline and key in baseline and value > 0.0001:
            ratio = baseline[key] / value
            if ratio > 1:
                speedup = f"  ({ratio:.1f}x faster)"
            elif ratio < 1:
                speedup = f"  ({1/ratio:.1f}x slower)"
            else:
                speedup = "  (same)"
        print(f"  {key:20s}: {value:8.4f} ms{speedup}")


def main():
    print("=" * 60)
    print("Trading Fitness Benchmark")
    print("=" * 60)

    # Test different data sizes
    sizes = [1_000, 10_000, 100_000, 1_000_000]
    iterations_map = {1_000: 1000, 10_000: 100, 100_000: 10, 1_000_000: 5}

    for size in sizes:
        iterations = iterations_map[size]
        print(f"\n\n{'#'*60}")
        print(f"# Dataset size: {size:,} data points")
        print(f"# Iterations: {iterations}")
        print(f"{'#'*60}")

        nav = generate_nav_series(size)

        # Python/Numba benchmark
        python_results = benchmark_python_numba(nav, iterations)
        format_results(python_results, "Python + Numba JIT")

        # Rust benchmark
        rust_results = benchmark_rust(nav, iterations)
        if rust_results:
            format_results(rust_results, "Rust (native)", baseline=python_results)
        else:
            print("\n[Rust benchmark skipped]")

        # Bun/TypeScript benchmark
        bun_results = benchmark_bun(nav, iterations)
        if bun_results:
            format_results(bun_results, "Bun/TypeScript", baseline=python_results)
        else:
            print("\n[Bun benchmark skipped]")


if __name__ == "__main__":
    main()
