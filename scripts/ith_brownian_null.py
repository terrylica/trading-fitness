#!/usr/bin/env python3
"""Brownian null for the ITH bull epoch count.

Calls the REAL shipped implementation:
  ~/eon/trading-fitness/packages/ith-python/src/ith_python/bull_ith_numba.py

The repo's own `uv run` is broken (missing local dep claspy at
~/fork-tools/claspy), so deps come from an ephemeral env and PYTHONPATH points
at the repo src. The repo's own source file is what executes.

Run:
  cd /tmp && PYTHONPATH=$HOME/eon/trading-fitness/packages/ith-python/src \
    uv run --no-project --python 3.13 --with numpy --with numba --with scipy \
    --with pandas python /tmp/ith_brownian_null.py
"""

from __future__ import annotations

import numpy as np

# ---- the repo's own implementation (NOT a reimplementation) ----------------
from ith_python.bull_ith_numba import (
    bull_excess_gain_excess_loss,
    generate_synthetic_nav,  # the repo's OWN NAV generator (t-dist arithmetic walk)
    max_drawdown,
)

SEED = 20260823
N_PATHS = 2000
TRADING_DAYS = 252.0
QUALIFIED_FLOOR = 8  # docs/ITH.md: "typically 8-11 for qualified strategies"


def gbm_nav(rng, n, sigma_step, mu_step=0.0, martingale=False):
    """Driftless geometric Brownian NAV, normalised to start at 1.0.

    mu_step is the per-step LOG drift. martingale=True subtracts sigma^2/2 so
    E[NAV_t] = 1 exactly (arithmetic-zero-drift reading of "no edge").
    """
    drift = mu_step - (0.5 * sigma_step**2 if martingale else 0.0)
    logret = rng.normal(drift, sigma_step, size=n - 1)
    nav = np.empty(n, dtype=np.float64)
    nav[0] = 1.0
    nav[1:] = np.exp(np.cumsum(logret))
    return nav


def run_cell(rng, n, ann_vol, ann_mu=0.0, hurdle_mode="mdd", fixed_hurdle=0.05,
             martingale=False, n_paths=N_PATHS):
    sigma_step = ann_vol / np.sqrt(TRADING_DAYS)
    mu_step = ann_mu / TRADING_DAYS
    counts = np.empty(n_paths, dtype=np.int64)
    events = np.empty(n_paths, dtype=np.int64)   # False->True transitions
    hurdles = np.empty(n_paths, dtype=np.float64)
    finals = np.empty(n_paths, dtype=np.float64)
    for k in range(n_paths):
        nav = gbm_nav(rng, n, sigma_step, mu_step, martingale)
        h = max_drawdown(nav) if hurdle_mode == "mdd" else fixed_hurdle
        res = bull_excess_gain_excess_loss(nav, h)
        counts[k] = res.num_of_bull_epochs
        e = res.bull_epochs
        events[k] = int(np.count_nonzero(e[1:] & ~e[:-1]) + (1 if e[0] else 0))
        hurdles[k] = h
        finals[k] = nav[-1]
    return {
        "n": n, "ann_vol": ann_vol, "ann_mu": ann_mu, "hurdle_mode": hurdle_mode,
        "martingale": martingale, "n_paths": n_paths,
        "mean": counts.mean(), "sd": counts.std(ddof=1),
        "median": float(np.median(counts)),
        "p5": float(np.percentile(counts, 5)),
        "p50": float(np.percentile(counts, 50)),
        "p95": float(np.percentile(counts, 95)),
        "pmin": int(counts.min()), "pmax": int(counts.max()),
        "frac_ge8": float(np.mean(counts >= QUALIFIED_FLOOR)),
        "frac_ge5": float(np.mean(counts >= 5)),
        "frac_8_11": float(np.mean((counts >= 8) & (counts <= 11))),
        "frac_zero": float(np.mean(counts == 0)),
        "mean_events": events.mean(),
        "mean_hurdle": hurdles.mean(),
        "median_final": float(np.median(finals)),
    }


HDR = (f"{'N':>6} {'annvol':>7} {'annmu':>7} {'hurdle':>7} "
       f"{'mean':>7} {'sd':>7} {'med':>5} {'p5':>5} {'p95':>6} "
       f"{'P(>=8)':>7} {'P(8-11)':>8} {'P(=0)':>6} {'evts':>6} {'TMAEG':>7}")


def fmt(r):
    return (f"{r['n']:>6d} {r['ann_vol']:>7.0%} {r['ann_mu']:>7.0%} "
            f"{r['hurdle_mode']:>7} {r['mean']:>7.2f} {r['sd']:>7.2f} "
            f"{r['median']:>5.0f} {r['p5']:>5.0f} {r['p95']:>6.0f} "
            f"{r['frac_ge8']:>7.3f} {r['frac_8_11']:>8.3f} {r['frac_zero']:>6.3f} "
            f"{r['mean_events']:>6.2f} {r['mean_hurdle']:>7.4f}")


def negative_controls():
    """A check that verifies nothing also reports success. These must differ."""
    print("=== NEGATIVE CONTROLS (n=1000) ===")
    flat = np.ones(1000)
    up = np.cumprod(np.r_[1.0, np.full(999, 1.001)])
    down = np.cumprod(np.r_[1.0, np.full(999, 0.999)])
    for name, nav in [("flat", flat), ("monotone_up", up), ("monotone_down", down)]:
        h = max_drawdown(nav)
        print(f"  {name:14s} mdd={h:.4f} "
              f"epochs(mdd-hurdle)={bull_excess_gain_excess_loss(nav, h).num_of_bull_epochs:5d} "
              f"epochs(h=0.05)={bull_excess_gain_excess_loss(nav, 0.05).num_of_bull_epochs:5d}")
    print()


def main():
    import ith_python.bull_ith_numba as _m
    print("PROVENANCE — module actually executed:")
    print(f"  {_m.__file__}\n")

    negative_controls()

    rng = np.random.default_rng(SEED)
    print("ITH BULL EPOCH COUNT — BROWNIAN NULL")
    print(f"seed={SEED}  paths/cell={N_PATHS}  hurdle 'mdd' = the path's OWN max drawdown "
          f"(repo default TMAEG_dynamically_determined_by='mdd')")
    print("'evts' = mean number of False->True runs (distinct rallies) vs 'mean' = mean flagged BARS\n")

    print("=== STEP 3/4a: ZERO DRIFT (log drift = 0), hurdle = own MDD ===")
    print(HDR)
    zero = []
    for n in (250, 1000, 4000):
        for v in (0.10, 0.20, 0.40):
            r = run_cell(rng, n, v)
            zero.append(r)
            print(fmt(r))

    print("\n=== STEP 4b: POSITIVE DRIFT contrast, hurdle = own MDD, N=1000, vol=20% ===")
    print(HDR)
    pos = []
    for mu in (0.0, 0.10, 0.20, 0.40, 0.80):
        r = run_cell(rng, 1000, 0.20, ann_mu=mu)
        pos.append(r)
        print(fmt(r))

    print("\n=== SENSITIVITY: martingale NAV (E[NAV]=1, log drift = -sigma^2/2) ===")
    print(HDR)
    for n in (250, 1000, 4000):
        print(fmt(run_cell(rng, n, 0.20, martingale=True)))

    print("\n=== SENSITIVITY: FIXED hurdle 0.05 instead of own-MDD (N=1000) ===")
    print(HDR)
    for mu in (0.0, 0.20):
        for v in (0.10, 0.20, 0.40):
            print(fmt(run_cell(rng, 1000, v, ann_mu=mu, hurdle_mode="fixed")))

    # ---- STEP 4c: the repo's OWN generator (t-dist ARITHMETIC walk) --------
    # generate_synthetic_nav() takes no rng handle, so seed the legacy global
    # RandomState that scipy.stats.t.rvs draws from. Reproducible, not default_rng.
    print("\n=== STEP 4c: the repo's OWN generate_synthetic_nav (t(5) arithmetic walk, n=1000) ===")
    print(f"{'avg_period_return':>18} {'mean':>7} {'sd':>7} {'med':>5} {'p5':>5} {'p95':>6} "
          f"{'P(>=8)':>7} {'P(8-11)':>8} {'P(>=5)':>7} {'TMAEG':>7}")
    np.random.seed(SEED)
    for apr in (0.0, 0.0001):  # 0.0001 = the repo's own default drift
        counts = np.empty(N_PATHS, dtype=np.int64)
        hs = np.empty(N_PATHS, dtype=np.float64)
        for k in range(N_PATHS):
            nav = generate_synthetic_nav(
                avg_period_return=apr, period_return_volatility=0.01, df=5, n_points=1000
            )["NAV"].to_numpy()
            h = max_drawdown(nav)
            hs[k] = h
            counts[k] = bull_excess_gain_excess_loss(nav, h).num_of_bull_epochs
        print(f"{apr:>18.4f} {counts.mean():>7.2f} {counts.std(ddof=1):>7.2f} "
              f"{np.median(counts):>5.0f} {np.percentile(counts, 5):>5.0f} "
              f"{np.percentile(counts, 95):>6.0f} {np.mean(counts >= 8):>7.3f} "
              f"{np.mean((counts >= 8) & (counts <= 11)):>8.3f} "
              f"{np.mean(counts >= 5):>7.3f} {hs.mean():>7.4f}")

    print("\n=== HEADLINE ===")
    a = [r["frac_ge8"] for r in zero]
    print(f"zero-drift P(epochs >= 8), 9 cells: min={min(a):.3f} max={max(a):.3f}")
    print(f"zero-drift mean epoch count, 9 cells: "
          f"min={min(r['mean'] for r in zero):.2f} max={max(r['mean'] for r in zero):.2f}")
    print(f"N=1000 vol=20%: mu=0 -> mean {pos[0]['mean']:.2f}, P(>=8)={pos[0]['frac_ge8']:.3f}  |  "
          f"mu=+80%/yr -> mean {pos[-1]['mean']:.2f}, P(>=8)={pos[-1]['frac_ge8']:.3f}")


if __name__ == "__main__":
    main()
