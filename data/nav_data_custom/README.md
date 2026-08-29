# `data/nav_data_custom/`

Drop CSV files with `Date`, `PnL`, `NAV` columns here for analysis.

## `sample_account.csv` — SYNTHETIC, NOT REAL TRADING HISTORY

`sample_account.csv` is **generated sample data**, produced entirely by a seeded
random number generator. It is **not** a real account, not a real person's P&L,
and is **not** derived (shuffled, scaled, perturbed, or otherwise transformed)
from any real trading history.

- Generator: [`generate_sample_account.py`](./generate_sample_account.py)
- RNG seed: `20260828` (`numpy.random.default_rng`)
- Shape: 2,726 business-day rows, schema `Date,PnL,NAV`, starting NAV 1,000,000

Regenerate with:

```bash
uv run --with numpy python data/nav_data_custom/generate_sample_account.py
```

Any test, demo, or doc that needs a NAV series should use this file.
