# The null ITH does not have

**Date**: 2026-08-24 · **Status**: research finding, no code changed · **Scope**: `packages/ith-python`, `packages/metrics-rust`, `packages/core-rust`

**Provenance.** This started in the sibling repo `~/eon/opendeviationbar-patterns` as a literature
question about event-driven bars, and ended here because the answer is about **equity-curve
performance evaluation**, which is this repo's subject. Two archived transcripts sit in
[`docs/research/external/`](./external/); the repo-side write-up on the bar side is
`findings/methodology/22-event-driven-bars-are-a-ruler-before-they-are-a-finding.md` in that repo.

**What was done here.** Every claim below about this repo was verified against **this repo's
source**, and the epoch-count null was produced by **executing the shipped implementation**
(`m.__file__` printed and checked), not a reimplementation. The reproducer is
[`scripts/ith_brownian_null.py`](../../scripts/ith_brownian_null.py).

---

## 0. The short version

**ITH's epoch count is sound against noise and fragile against smoothness, and nothing in this
repository has ever measured either.**

1. 🔴 **No metric in this repo is compared against any null.** Not ITH epochs, not TMAEG, not MRU,
   not max drawdown, not CV. A repo-wide sweep for eleven keywords returned 308 matches across 52
   files and **every one is a false positive**.
2. ✅ **The good news, now measured**: on a zero-drift equity curve the shipped bull-ITH epoch count
   averages **0.58–1.79** and clears the documented 8-epoch bar **at most 1.2% of the time**. Noise
   does not fake fitness under the default `mdd` hurdle.
3. 🔴 **But that is entirely a property of the hurdle mode.** With the selectable **fixed 0.05**
   hurdle — which is what _every edge-case fixture uses_ — pure noise at 40% vol clears 8 epochs
   **24.6%** of the time.
4. 🔴 **The `mdd` hurdle degenerates catastrophically as MDD → 0.** A monotone +0.1%/bar curve scores
   **999 epochs out of 999 bars**, because the hurdle is zero so every bar is flagged. _The 8-epoch
   bar is safe against noise; it is not safe against smoothness._
5. 🔴 **The documented "8–11 epochs" band is not a floor — it is a narrow window around Sharpe ≈ 2.**
   `P(count ∈ [8,11])` peaks at **0.389** at +40%/yr drift and 20% vol; at Sharpe 4.0 it is **0.003**,
   because a better strategy _overshoots the band_.
6. 🔴 **Sharpe is the first gate**, is `required` in the cross-language schema, and pre-filters which
   candidates ever reach ITH — while `README.md:13` markets ITH as _"a more nuanced view of strategy
   quality than simple Sharpe ratios"_.

---

## 1. There is no null anywhere in this repository

A repo-wide sweep for `null model | surrogate | monte carlo | permutation | shuffl | placebo |
brownian | random.walk | bootstrap | baseline | benchmark | synthetic` returned **308 matches across
52 files**. Every hit is a false positive:

| Hit                      | What it actually is                | Where                                          |
| ------------------------ | ---------------------------------- | ---------------------------------------------- |
| "permutation"            | permutation **entropy**, a feature | `packages/metrics-rust/src/entropy.rs:22`      |
| "random walk"            | a Hurst `H ≈ 0.5` **label**        | `…/statistical_examination/regime.py:137`      |
| "null"                   | dataframe nulls                    | —                                              |
| "baseline" / "benchmark" | **wall-clock speed**               | `README.md:57`                                 |
| "Total nulls \| 0"       | a schema completeness row          | `docs/forensic/ANALYSIS_REPORT_20260125.md:86` |

The only resampler in the tree, the **circular block bootstrap**
(`…/statistical_examination/block_bootstrap.py:103`), puts confidence intervals on **feature-importance
stability** over the 160-feature matrix. It never touches a fitness metric.

The only control arm anywhere is an **anti-aligned mirror** in one research notebook
(`packages/ith-python/research/sma_regime_filter.py:9-11`).

🔴 **Worse than absent — the synthetic generator is not a null, it is the sample population.**
`generate_synthetic_nav` (`packages/ith-python/src/ith_python/ith.py:657-721`) emits a drifted
t-distributed random walk; the batch is then **pre-filtered by the fitness criteria** (`ith.py:705`
`valid_mask = (sharpes >= sr_lower) & (sharpes <= sr_upper)`) and the survivors published as
`required_qualified_results: int = 100  # Generate 100 qualified samples` (`ith.py:172`). **The null
and the subject are drawn from the same distribution, and nothing compares them.**

`docs/VERIFICATION.md` is a 2026-01-15 **file-migration checklist**, not a statistical document.
`docs/forensic/*` audits schema, redundancy, autocorrelation and cross-language parity — the
pipeline, never the edge. The 819-test "hypothesis telemetry" emits exactly seven test names
(`anderson_darling_beta`, `augmented_dickey_fuller`, `mann_whitney_u`, `pca_dimensionality`,
`ridge_vif`, `shapiro_wilk`, `temporal_structure_analysis`) — all about feature distributions, none
about whether a strategy beats chance.

---

## 2. The Brownian null for the ITH epoch count

**Method.** Driftless geometric Brownian NAV, `nav[0] = 1.0`, log-returns `N(μ/252, (σ/√252)²)`;
`N ∈ {250, 1000, 4000}` bars × annual vol `∈ {10%, 20%, 40%}`; **2000 paths per cell**;
`default_rng(20260823)`. Hurdle = each path's own `max_drawdown()`, the shipped default
(`ith.py:152` `TMAEG_dynamically_determined_by: str = "mdd"`).

**Rust/Python parity** was established by reading `packages/metrics-rust/src/ith.rs::bull_ith`
against `packages/ith-python/src/ith_python/bull_ith_numba.py` — identical statement for statement,
same reset condition, same epoch flag. ⚠️ Only the **Python** was executed.

### 2.1 Zero drift — what a no-edge curve scores

|    N | ann. vol | mean |   sd | median | p95 | **P(≥8)** | P(=0) | mean TMAEG |
| ---: | -------: | ---: | ---: | -----: | --: | --------: | ----: | ---------: |
|  250 |      10% | 0.58 | 0.97 |      0 |   2 | **0.000** | 0.635 |     0.1092 |
|  250 |      20% | 0.64 | 1.06 |      0 |   3 | **0.002** | 0.621 |     0.2049 |
|  250 |      40% | 0.79 | 1.15 |      0 |   3 | **0.003** | 0.545 |     0.3613 |
| 1000 |      10% | 0.63 | 1.01 |      0 |   3 | **0.000** | 0.620 |     0.2132 |
| 1000 |      20% | 0.76 | 1.18 |      0 |   3 | **0.001** | 0.576 |     0.3763 |
| 1000 |      40% | 1.07 | 1.36 |      1 |   4 | **0.001** | 0.451 |     0.5909 |
| 4000 |      10% | 0.77 | 1.13 |      0 |   3 | **0.001** | 0.549 |     0.3736 |
| 4000 |      20% | 1.06 | 1.38 |      1 |   4 | **0.001** | 0.460 |     0.6019 |
| 4000 |      40% | 1.79 | 1.88 |      1 |   6 | **0.012** | 0.302 |     0.8176 |

**30–64% of coin-flip paths score exactly zero epochs.** The 8-epoch bar is cleared by noise at most
**1.2%** of the time, usually ~0.1%.

A **martingale** variant (`E[NAV] = 1`, log drift `−σ²/2`, the arithmetic-zero-drift reading) gives
0.55 / 0.61 / 0.70 mean and `P(≥8)` ≤ 0.001 — the conclusion does not depend on which "no drift"
you mean.

The repo's **own** `generate_synthetic_nav` (t(5) arithmetic walk, n=1000) gives mean **0.64** at
`avg_period_return=0.0` and **0.87** at the repo default `0.0001`. **Fat tails do not change the
null.** Note `P(≥5)` — the code's actual `bull_epochs_lower_bound` — is **0.010** and **0.024**.

### 2.2 Positive drift — the contrast

`N = 1000`, vol 20%, hurdle = own MDD:

| annual μ | ≈ Sharpe | mean epochs | P(≥8) | **P(∈[8,11])** |
| -------: | -------: | ----------: | ----: | -------------: |
|       0% |      0.0 |        0.79 | 0.001 |          0.001 |
|     +10% |      0.5 |        2.09 | 0.015 |          0.013 |
|     +20% |      1.0 |        4.11 | 0.114 |          0.102 |
|     +40% |      2.0 |   **10.07** | 0.717 |      **0.389** |
|     +80% |      4.0 |       28.24 | 1.000 |      **0.003** |

🔴 **The "8–11" band is a Sharpe-2 window, not a quality floor.** It peaks at 0.389 around Sharpe 2
and collapses to **0.003** at Sharpe 4 — _a strategy twice as good fails the band by overshooting
it._ If the band is used as a filter it is selecting **for a specific drift level**, not for quality.

### 2.3 🔴 The null depends entirely on the hurdle mode

Same paths, hurdle = **fixed 0.05** (`ith.py:1573`, `bear_ith.py:729`) instead of own-MDD:

|    N | ann. vol | ann. μ | mean | **P(≥8)** |
| ---: | -------: | -----: | ---: | --------: |
| 1000 |      10% |     0% | 2.03 | **0.021** |
| 1000 |      20% |     0% | 3.31 | **0.108** |
| 1000 |      40% |     0% | 5.21 | **0.246** |

**At 40% vol, pure noise clears the 8-epoch bar a quarter of the time.** The `mdd` hurdle is
self-normalising — it is effectively a return/drawdown ratio, near scale-free in volatility — and
that is _why_ its null is low. A fixed hurdle is not, so the same epoch count means something
entirely different. **Every edge-case fixture in the repo uses `fixed`**; the analysis path uses
`mdd`. An epoch count reported without its hurdle mode is uninterpretable.

### 2.4 🔴 Negative controls on the harness — and the degeneracy they expose

| NAV                    |                epochs |
| ---------------------- | --------------------: |
| flat                   |                     0 |
| monotone −0.1%/bar     |                     0 |
| **monotone +0.1%/bar** | **999** (of 999 bars) |

The harness discriminates. But the third row is a real defect: **MDD → 0 makes the hurdle → 0, so
every bar is flagged.** A perfectly smooth rising curve with no drawdown scores an unbounded epoch
count. _ITH is safe against noise and unsafe against smoothness_, and a strategy that is smooth for
mechanical reasons — heavy netting, infrequent marks, a stale price source — would score
arbitrarily high.

**Structural note.** `num_of_bull_epochs` sums **flagged bars**, not distinct rallies. Under the
`mdd` hurdle the two coincided in every cell; under `fixed 0.05` at 40% vol they diverged (5.21 bars
vs 5.10 runs). The name says "epochs"; the code counts bars.

---

## 3. What ITH actually anchors on

Verified against source. **TMAEG is the calibration, not the origin.**

| Claim                                                                                    | Verdict       | Evidence                                                                                                                                                                                                                                   |
| ---------------------------------------------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Bull hurdle TMAEG derives from the window's max drawdown; bear from max runup            | **CONFIRMED** | `ith_rolling.rs:205` `let bull_tmaeg = compute_max_drawdown(&normalized_window); let bear_tmaeg = compute_max_runup(&normalized_window);` — unconditional. Python defaults `"mdd"` / `"mru"`                                               |
| Bull ITH measures excess gain from `endorsing_crest`, a confirmed **peak**, not a trough | **CONFIRMED** | `ith.rs:96` `excess_gain = next_equity / endorsing_crest - 1.0;` — `endorsing_crest` is only ever assigned `candidate_crest`, a monotone running high never reset downward (unlike `candidate_nadir`)                                      |
| A fixture scores a +21.2% rise off a trough as **zero** epochs                           | **CONFIRMED** | `tests/fixtures/edge_cases.py:102` `CASE_3_V_RECOVERY` nav `[100, 95, 90, 85, 88, 92, 96, 100, 103]`, `"expected_bull_epochs": []`. `103/85 − 1 = 21.18%`. Ran `bull_ith` on it directly: **0 epochs, peak excess_gain 0.03**              |
| `endorsing_nadir` is dead code                                                           | **PARTLY**    | `ith.rs:124` — it **is** read, but only by its own `.min()`, so no compiler warning fires. Correct wording: a **self-referential no-op state variable whose value never escapes**. Appears at exactly 3 lines; absent from `BullIthResult` |

**Corrections worth carrying:**

- The Rust bear hurdle parameter is literally named `tmaeg` (`ith.rs:187`), not `tmaer`. "TMAER"
  exists only in Python and the README.
- The derivation lives at the **call site**, not the kernel — `bull_ith`/`bear_ith` accept any hurdle.
- `adaptive.rs:361`'s description of the rolling TMAEG ("empirical max deviation") is **stale**.
- The same self-referential state exists in the bear path (`endorsing_peak`, `ith.rs:267`) and is
  **duplicated** in `packages/core-rust/src/ith.rs:26,54,57` and
  `packages/ith-python/src/ith_python/bull_ith_numba.py:63,97,100` — and is documented as real state
  in `README.md:176,201`.

### 3.1 `max_runup` anchors on `cummin`, not on a drawdown

`max_ru = max(1 − cummin(nav)/nav)` with `running_min` seeded at `window[0]`
(`ith_rolling.rs:123-137`; `bear_ith_numba.py:246-249`; `README.md:143`).

Executed on the real crate: `nav = [1.0, 1.1, 1.25, 1.5]` → `max_runup = 0.3333`, `max_drawdown =
2.2e-16` (zero). **A statistic that is well-defined and non-zero in the total absence of a drawdown
does not use the drawdown as its anchor.**

🔴 **A correction to an earlier draft of this finding** (carried in the sibling repo's methodology 22
§11.5): it does **not** return the window's total return. Total return there is 0.50; `max_runup` is
0.3333. It returns `1 − first/last = R/(1+R)` — the rise measured over the **final** value, not the
initial one. The denominator convention is documented in four places
(`drawdown.rs:136`, `ith_normalize.rs:140`, `bear_ith.py:631-633`) and commit `d46d64f` deliberately
chose it for boundedness — but **no magnitude for the difference is stated anywhere**.

---

## 4. 🔴 Sharpe is the first gate

`README.md:13` markets ITH as _"a more nuanced view of strategy quality than simple Sharpe ratios."_
In the code, Sharpe decides what reaches ITH at all.

| Where                                                   | What                                                                                                                      |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `README.md:262`                                         | `\| Sharpe Ratio \| 0.5 < SR < 9.9 \| -9.9 < SR < -0.5 \|` under **"Fitness Qualification Criteria"**                     |
| `ith.py:167-170`                                        | `sr_lower_bound: float = 0.5` / `sr_upper_bound: float = 9.9` / `aggcv_up_bound: float = 0.70`                            |
| `ith.py:1231`                                           | the gate is a plain conjunction, evaluated **once, in-sample**, no p-value                                                |
| `ith.py:664`                                            | _"Generate a batch of synthetic NAVs and **pre-filter by Sharpe ratio**"_                                                 |
| `packages/shared-types/schemas/fitness-metrics.json:50` | `"required": ["sharpe_ratio", "max_drawdown", "total_return"]` — structurally load-bearing across all four language ports |

**This is a direct conflict with the operator's standing anti-Sharpe doctrine** in the sibling repo
(`feedback_never_sharpe.md`: no Sharpe, no DSR/PSR/ΔSharpe). It is recorded here as a fact about the
code, not resolved — resolving it is a design decision.

**Two documentation defects found alongside it:**

- `README.md:263` publishes an epoch bound of `> ceil(points/168)`. **It is not implemented
  anywhere** — `rg '168'` over `packages/` returns nothing. The code uses a fixed
  `bull_epochs_lower_bound = 5`.
- `docs/ITH.md:21`'s _"typically 8-11 for qualified strategies"_ corresponds to **no constant in the
  code**, and §2.2 shows it is a Sharpe-2 window rather than a floor.

---

## 5. The literature that supplies a richer null

§2's null is Brownian. Real equity curves have volatility clustering and serial dependence, so the
stronger null is a **stationary/block bootstrap of real strategy returns**. Beyond that, the
drawdown/drawup first-passage literature gives closed forms for exactly the objects ITH manipulates.
**All twelve were resolved against publisher pages, Crossref, arXiv and DOI resolvers; 9 verified as
given, 3 corrected.**

**Anchored on the extremum — the objects ITH uses:**

- **Taylor, H. M. (1975).** "A stopped Brownian motion formula." _Annals of Probability_ 3(2),
  234–246. doi:[10.1214/aop/1176996395](https://doi.org/10.1214/aop/1176996395) — for driftless
  Brownian motion the running maximum before a δ-drawdown is **Exponential with mean δ**.
- **Lehoczky, J. P. (1977).** "Formulas for stopped diffusion processes with stopping times based on
  the maximum." _Annals of Probability_ 5(4), 601–607.
  doi:[10.1214/aop/1176995770](https://doi.org/10.1214/aop/1176995770) — closed-form
  `P(reach +b before a drawdown of h)` for a general diffusion. **This is the null for any
  "did it recover?" statistic.**
- **Magdon-Ismail, M., Atiya, A. F., Pratap, A. & Abu-Mostafa, Y. S. (2004).** "On the maximum
  drawdown of a Brownian motion." _J. Applied Probability_ 41(1), 147–161.
  doi:[10.1239/jap/1077134674](https://doi.org/10.1239/jap/1077134674) — max drawdown scales
  **√T at zero drift**, log T at positive drift, linearly at negative. 🔴 **Directly relevant here:
  TMAEG is a max drawdown, so the hurdle itself has a known null growth rate, and any epoch count
  compared across window lengths must account for it.**
- **Pospisil, L., Vecer, J. & Hadjiliadis, O. (2009).** "Formulas for stopped diffusion processes
  with stopping times based on drawdowns and drawups." _Stochastic Processes and their Applications_
  119(8).
- **Mijatović, A. & Pistorius, M. R. (2011).** "On the drawdown of completely asymmetric Lévy
  processes." [arXiv:1103.1460](https://arxiv.org/abs/1103.1460).

**Drawdown _and drawup_ — the modern successors:**

- **Landriault, D., Li, B. & Zhang, H. (2017).** "A unified approach for drawdown (drawup) of
  time-homogeneous Markov processes." _J. Applied Probability_ 54(2), 603–626.
  doi:[10.1017/jpr.2017.20](https://doi.org/10.1017/jpr.2017.20)
- **González Cázares, J. I. & Mijatović, A. (2022).** "Simulation of the drawdown and its duration
  in Lévy models via stick-breaking Gaussian approximation." _Finance and Stochastics_ 26(4),
  671–732. doi:[10.1007/s00780-022-00486-7](https://doi.org/10.1007/s00780-022-00486-7)
- ⚠️ **Zhang, G. & Li, L. (2023).** "A general method for analysis and valuation of drawdown risk."
  _Journal of Economic Dynamics and Control_ 152, 104669.
  doi:[10.1016/j.jedc.2023.104669](https://doi.org/10.1016/j.jedc.2023.104669) — **CTMC method
  covering time-inhomogeneous Markov, stochastic volatility, regime switching and jumps.** The most
  computationally usable successor. _Corrected: **Gongqiu** Zhang and **Lingfei** Li, not Hongzhong
  Zhang and Bin Li._
- ⚠️ **Li, L., Zeng, P. & Zhang, G. (2024).** "Speed and duration of drawdown under general Markov
  models." _Quantitative Finance_ 24(3–4), 367–386.
  doi:[10.1080/14697688.2024.2336154](https://doi.org/10.1080/14697688.2024.2336154) — 🔴 **speed and
  duration are exactly the ITH "time horizon" idea, with a null attached.** _Corrected initials._
- ⚠️ 🔴 **Zeng, P., Zhang, G. & Zhang, W. (2026).** "Drawdowns, drawups, and occupation times under
  general Markov models." _European Journal of Operational Research_ 333(3), 956–968.
  doi:[10.1016/j.ejor.2026.04.033](https://doi.org/10.1016/j.ejor.2026.04.033) — **drawdown-before-
  drawup probabilities and occupation times.** The closest published object to what a
  drawdown-anchored fitness metric would need. _Corrected: Gongqiu Zhang and Weinan Zhang._
- **Proietti, T. (2026).** "Ups and (draw) downs." _International Journal of Forecasting_ 42(2),
  457–473. doi:[10.1016/j.ijforecast.2025.07.008](https://doi.org/10.1016/j.ijforecast.2025.07.008)
  — models time since the latest peak and trough as Markov chains; persistence, duration,
  peak/trough dating, **out-of-sample prediction**.
- **Geboers, H., Depaire, B. & Annaert, J. (2023).** "A review on drawdown risk measures and their
  implications for risk management." _Journal of Economic Surveys_ 37(3), 865–889.
  doi:[10.1111/joes.12520](https://doi.org/10.1111/joes.12520) — the survey doorway.

🔴 **Three of these had wrong author initials as originally received** from an AI transcript, and
all three are in the group this repo would most likely cite. **Resolve every DOI you print.**

---

## 6. What this does not establish

Stated so it is not over-read:

1. **Rust parity was read, not executed.** The two implementations match statement for statement;
   only the Python ran.
2. **GBM may not be the right null.** Real equity curves have volatility clustering, serial
   dependence and overlapping-position autocorrelation. A stationary/block bootstrap of real
   strategy returns is stronger. Fat tails alone are ruled out — the repo's own t(5) generator gave
   0.64.
3. **The `mdd` hurdle makes the statistic self-normalising**, which is why the null is low — and
   also why the monotone-up case explodes. The 8-epoch bar is safe against noise, **not** against
   smoothness.
4. **Only the epoch count was tested.** The shipped gate is epochs ≥ 5 **and** Sharpe **and** CV.
   End-to-end pipeline behaviour is untested.
5. **The repo's own `uv run` is broken** — a missing local dependency `claspy` expected at
   `~/fork-tools/claspy`. The reproducer documents its own ephemeral-env invocation as a workaround.
   That is a real reproducibility defect, reported and not fixed here.

---

## 7. Suggested next steps, in order of value

1. **Report the hurdle mode with every epoch count.** `mdd` and `fixed` have nulls that differ by
   two orders of magnitude in `P(≥8)`. Nothing downstream currently records which was used.
2. **Guard the MDD → 0 degeneracy.** A floor on TMAEG, or an explicit refusal when
   `max_drawdown < ε`, turns a silent 999 into an error.
3. **Replace the "8–11" band with a stated drift target, or drop it.** As written it rejects
   Sharpe-4 strategies.
4. **Bootstrap the null from real returns**, not GBM, and publish it beside the fitness thresholds.
5. **Decide the Sharpe question explicitly** — it is currently both the first gate and the thing the
   README says ITH improves on.
6. **Fix the two documentation defects**: `ceil(points/168)` is unimplemented; "8–11 epochs"
   corresponds to no constant.

---

## Reproducing §2

```bash
cd /tmp && PYTHONPATH=$HOME/eon/trading-fitness/packages/ith-python/src \
  uv run --no-project --python 3.13 --with numpy --with numba --with scipy --with pandas \
  python scripts/ith_brownian_null.py
```

Seed `20260823`, 2000 paths per cell. Verified to reproduce every number in §2 exactly.

---

## See also

- [`docs/research/external/2026-08-23-…-chatgpt.md`](./external/2026-08-23-multiscale-tradability-of-event-driven-bars-xauusd-chatgpt.md) — the event-driven-bar literature question this began as
- [`docs/research/external/2026-08-24-…-chatgpt.md`](./external/2026-08-24-drawdown-drawup-first-passage-successors-to-mijatovic-pistorius-chatgpt.md) — the drawdown/drawup turn
- [`docs/ITH.md`](../ITH.md) — the algorithm this measures
- [`scripts/ith_brownian_null.py`](../../scripts/ith_brownian_null.py) — the reproducer
