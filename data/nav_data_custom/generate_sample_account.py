"""Generate `sample_account.csv` — SYNTHETIC NAV/PnL data for tests and demos.

This file is the generator of record for `data/nav_data_custom/sample_account.csv`.
The output is 100% machine-generated from a seeded RNG (SEED below). It is NOT
derived from, sampled from, shuffled from, or scaled from any real trading
account. It exists only to preserve the *shape* (schema + row count + rough
volatility regime) that tests and demos depend on.

Usage:
    uv run python data/nav_data_custom/generate_sample_account.py
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

SEED = 20260828
START_DATE = date(2016, 1, 4)
N_ROWS = 2726
START_NAV = 1_000_000.0
OUT = Path(__file__).resolve().parent / "sample_account.csv"


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Slow-varying volatility regime (synthetic, arbitrary constants).
    t = np.arange(N_ROWS)
    vol = 0.006 * (1.0 + 0.45 * np.sin(2 * np.pi * t / 430.0)) + 0.0015
    drift = 0.00022
    # Student-t innovations give plausible fat tails without any real data.
    shocks = rng.standard_t(df=4.0, size=N_ROWS) / np.sqrt(4.0 / 2.0)
    returns = drift + vol * shocks

    nav = START_NAV
    rows = []
    d = START_DATE
    for r in returns:
        while d.weekday() >= 5:  # business days only
            d += timedelta(days=1)
        pnl = nav * float(r)
        nav += pnl
        rows.append((d.isoformat(), round(pnl, 2), round(nav, 2)))
        d += timedelta(days=1)

    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Date", "PnL", "NAV"])
        w.writerows(rows)

    print(f"wrote {OUT} ({len(rows)} rows, seed={SEED})")


if __name__ == "__main__":
    main()
