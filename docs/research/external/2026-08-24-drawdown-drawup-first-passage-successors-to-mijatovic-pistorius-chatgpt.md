---
source_url: https://chatgpt.com/share/6a8b86f2-3e30-83e8-8d13-879d7cbd9236
source_type: chatgpt-share
scraped_at: "2026-08-24T00:20:00Z"
model_name: unknown
model_name_note: >-
  A ChatGPT share link does not expose the model identifier. Two extended-reasoning
  turns ("Worked for 16m 13s", "Worked for 7m 38s") with live web access. The exact
  name is NOT recoverable from the artifact and is therefore not asserted here.
custom_gpt_name: ""
claude_code_uuid: 691d4b6f-79c2-4c9f-859f-528e32c06333
supersedes_note: >-
  This share link is the SAME conversation as
  docs/research/2026-08-23-multiscale-tradability-of-event-driven-bars-xauusd-chatgpt.md,
  extended by one turn. MEASURED, not assumed - turn 1 of this scrape (15,795 chars,
  normalised) is a strict SUBSTRING of that file's body (15,818 chars); the only
  delta is the trailing AI disclaimer, which moved to the end of turn 2. Only turn 2
  (9,268 chars) is reproduced below. The earlier file remains the archive of record
  for turn 1 and is NOT duplicated here - a file cannot have two SSoTs.
github_issue_url: "https://github.com/terrylica/opendeviationbar-patterns/issues/162"
github_issue_number: "162"
scraper: firecrawl-v2
scraped_chars: 25613
turn_2_chars: 9268
---

> **Archival note.** Second turn only. The operator handed ChatGPT
> [arXiv:1103.1460](https://arxiv.org/pdf/1103.1460) — Mijatović & Pistorius, _"On the drawdown of
> completely asymmetric Lévy processes"_ — and asked whether it bears on the ODB question and what
> supersedes it.
>
> **Why this turn is worth archiving separately.** It opens a literature family that neither the
> repo-side sweeps nor turn 1 reached: **drawdown/drawup first-passage theory**. And it arrives at
> [methodology 22](../../findings/methodology/22-event-driven-bars-are-a-ruler-before-they-are-a-finding.md)'s
> central result from a **third independent direction**, having been given only a probability paper:
>
> - it independently derives `P(+δ before −δ) = 1/(1 + exp(−2μδ/σ²))` and
>   `E[τ_δ] = (δ/μ)·tanh(μδ/σ²)` with driftless limit `δ²/σ²`;
> - it states the theorem verbatim — _"by the strong Markov property, consecutive ODB signs are
>   **independent** in this model. There is no predictive mean reversion or momentum"_;
> - and it draws the conclusion methodology 22 §3.3 draws: _"'small bars chop, large bars trend' can
>   arise **mechanically** from drift plus diffusion, without an inefficiency."_
>
> 🔴 **It also supplies the sign-imbalance-corrected statistic** methodology 22 §1 says is the right
> object, in closed form: `A_δ = P(S_{n+1}=S_n) − [p_δ² + (1−p_δ)²]`. That is strictly better than
> `p_c − ½`, because at large δ drift makes `p_δ ≠ ½` and a fair-coin runs null would then report
> **sign imbalance as dependence** — the exact trap methodology 22 warns about but did not close in
> closed form.
>
> **The terminological correction worth keeping.** A drawdown-barrier **overshoot** (the amount a
> jump crosses the drawdown barrier by) and a directional-change **overshoot** (continuation after a
> DC confirmation, before the next reversal) are **different random variables**. Methodology 22 §3.1
> uses the DC sense throughout; this is the disambiguation.
>
> **The successor stack it names** — Landriault/Li/Zhang (2017) on drawdown **and drawup**;
> González Cázares & Mijatović (2022); Zhang & Li (2023); Li/Zeng/Zhang (2024) on **speed and
> duration** of drawdown; Zeng/Zhang/Zhang (2026) on **drawdowns, drawups and occupation times**,
> including **drawdown-before-drawup** probabilities; Proietti (2026) _"Ups and (Draw) Downs"_; and
> Geboers/Depaire/Annaert (2023) as the survey doorway.
>
> ⚠️ **Its own stated caveat**: cite 1103.1460 as _"extremum-anchored first-passage theory and
> drawdown/rally null models"_ — **not** as evidence of lower-threshold mean reversion, safe
> regression exits, or XAU/USD tradability. And as in turn 1, the `~/eon` directories were not
> mounted, so its ODB construction is inferred from public `opendeviationbar-core` docs.
>
> Citations in this turn were **not** put through the 39-citation verification pass that
> methodology 22 §9 records; they are archived as received.

---

Today3:52 PM

#### You said

<https://arxiv.org/pdf/1103.1460>

Does this paper have any relevance or merits to our discovery of the above? If yes, what are the even more state-of-the-art and recent papers that encompass its content and directions and provide even more insights?

#### ChatGPT said

Worked for 7m 38s

Yes—but it should be cited as a mathematical foundation and null model, not as evidence that XAU/USD is profitably mean-reverting.

Mijatović and Pistorius, “On the drawdown of completely asymmetric Lévy processes” is a strong peer-reviewed probability paper. Its relevance is closest to directional-change sampling, not directly to Open Deviation Bars.

### What the paper actually contributes

For log price Xt​, it studies

Dt​=Xt​−Xt​,τa​=inf{t:Dt​>a},

where Xt​ is the running maximum. It derives the joint law of:

- time until a drawdown of size a;

- time of the preceding maximum;

- running minimum and maximum;

- position immediately before the threshold crossing;

- threshold undershoot and leapover;

- probabilities such as “drawdown occurs before rally.”

That is mathematically valuable because it describes an entire threshold event—not merely its direction. It provides objects directly related to adverse excursion, recovery risk, event duration and barrier ordering.

| Connection to your hypothesis                              | Assessment                                                  |
| ---------------------------------------------------------- | ----------------------------------------------------------- |
| Event-driven rather than clock-time analysis               | Strong                                                      |
| One downward directional-change leg from a running maximum | Strong                                                      |
| Standard ODB resetting from the previous bar close         | Adjacent, not identical                                     |
| Comparing many thresholds                                  | Possible, but not developed as a multiscale empirical model |
| Explaining genuine lower-scale mean reversion              | No                                                          |
| Explaining high-threshold trend persistence                | Only indirectly through drift and jump asymmetry            |
| XAU/USD evidence, spread or execution costs                | None                                                        |
| Demonstration of a profitable strategy                     | None                                                        |

Two terminological distinctions are important:

1. Its “overshoot” is the amount by which a jump crosses the drawdown barrier at τa​.

2. A directional-change “overshoot” normally means price continuation after a DC confirmation and before the next reversal.

Those are different random variables.

### A particularly useful implication for ODB

The same first-passage framework produces a revealing ODB null model. Suppose a symmetric log-price ODB resets after every event and

Xt​=μt+σWt​.

For an ODB threshold δ,

P(+δ is reached before −δ)=pδ​=1+exp(−2μδ/σ2)1​.

Therefore:

- At small δ, pδ​≈21​: the ODB sequence looks densely choppy and balanced.

- As δ increases, drift becomes more visible and the event sequence becomes directionally biased.

- Yet, by the strong Markov property, consecutive ODB signs are independent in this model. There is no predictive mean reversion or momentum.

The expected bar duration is

E\[τδ​\]=μδ​tanh(σ2μδ​),

with the driftless limit E\[τδ​\]=δ2/σ2. This alone creates many small-threshold bars and relatively few large-threshold bars.

So “small bars chop, large bars trend” can arise mechanically from drift plus diffusion, without an inefficiency. The correct test is whether observed transitions exceed this null:

Aδ​=P(Sn+1​=Sn​)−\[pδ2​+(1−pδ​)2\].

- Aδ​<0: excess reversal—evidence consistent with mean reversion.

- Aδ​>0: excess continuation—evidence consistent with persistence.

- Aδ​=0: apparent chop/trend is explained by an independently incremented process.

This is probably the paper’s greatest merit for your project: it helps specify the null you must beat.

## Strongest direct successors

1. [Landriault, Li and Zhang (2017), “A Unified Approach for Drawdown (Drawup) of Time-Homogeneous Markov Processes”](https://doi.org/10.1017/jpr.2017.20)

The clearest theoretical generalization. It covers general time-homogeneous Markov processes and possibly two-sided-jump Lévy processes, deriving the joint law of first-passage time, overshoot and running maximum through integral equations.

1. [González Cázares and Mijatović (2022), “Simulation of the drawdown and its duration in Lévy models via stick-breaking Gaussian approximation”](https://doi.org/10.1007/s00780-022-00486-7)

A direct continuation by one of the original authors. It moves from one-sided-jump analytic formulas to computational treatment of general Lévy processes and adds drawdown duration and time of the maximum.

1. Zhang and Li (2023), “A general method for analysis and valuation of drawdown risk”

Probably the best modern computational successor. Its continuous-time Markov-chain method handles time-inhomogeneous Markov processes, stochastic volatility, regime switching and jump models. It computes first-passage and maximum-drawdown distributions, with convergence analysis.

1. Li, Zeng and Zhang (2024), “Speed and duration of drawdown under general Markov models”

Especially relevant to your discovery. Speed and duration help distinguish a rapid, transitory microstructural excursion from a sustained macro move. These should become ODB state variables alongside direction.

1. Zeng, Zhang and Zhang (2026), “Drawdowns, Drawups, and Occupation Times under General Markov Models”

The broadest recent extension. It treats drawdown-before-drawup probabilities, occupation times, and repeated drawdowns with and without recovery. This is much closer to modeling your “regression exit before adverse continuation” question systematically.

1. [Proietti (2026), “Ups and (Draw) Downs”](https://doi.org/10.1016/j.ijforecast.2025.07.008)

The strongest recent statistical complement. It models time since the latest peak and trough as Markov chains and studies drawdown size, serial correlation, persistence, duration, peak/trough dating and out-of-sample prediction. It is not ODB-based, but it moves beyond theoretical risk pricing toward testable persistence.

A useful survey doorway is Geboers, Depaire and Annaert (2023), covering drawdown definitions, statistical properties, portfolio optimization and risk measures.

## Papers closer to ODB than the 2012 paper

These are more important for validating your actual bar construction:

- Hong, Nolte, Taylor and Zhao (2023), “Volatility Estimation and Forecasts Based on Price Durations”

Price-duration events are nearly the same stopping-time object as symmetric ODB closes. The paper explicitly studies threshold selection, bid/ask spread, discrete prices, irregular observations, microstructure noise and jumps.

- Li, Nolte, Nolte and Yu (2025), “Decoupling Interday and Intraday Volatility Dynamics With Price Durations”

Introduces adaptive price-change thresholds that separate daily volatility from intraday dynamics. This is directly relevant to selecting ODB thresholds relative to the current volatility regime.

- Dimitriadis et al. (2026), “Efficient Sampling for Realized Variance Estimation in Time-Changed Diffusion Models”

Finds that hitting-time sampling can be highly efficient when noise is low, but loses its advantage as microstructure noise increases. This is an important correction to “the spread is low, therefore very small thresholds are safe.”

- Li et al. (2026), “Testing for Jumps … with Endogenous Sampling Times”

Uses recursive first exits from a symmetric double barrier—essentially an ODB observation scheme—to distinguish diffusion-generated exits from jumps with a noise-robust test.

## Directional-change papers worth adding

[Glattfelder’s 2024 “Theory of Intrinsic Time”](https://arxiv.org/abs/2406.07354) explicitly connects drawdowns/drawups, DC events, overshoots and multiple thresholds. It is a useful conceptual synthesis, but remains a preprint rather than trading-edge evidence.

Two recent empirical papers expose the central implementation problem:

- [Tsang, Ma and Chinthalapati (2024)](https://doi.org/10.1002/isaf.1552) attempts to nowcast a DC before its later confirmation.

- Ao and Li (2024) shows that the gap between theoretical and actual DC confirmations can drastically distort strategy performance.

That confirmation gap is analogous to entering at an ODB extreme that was identifiable only after a later threshold crossing.

### Bottom line

Keep the 2012 paper in the bibliography under:

> Extremum-anchored first-passage theory and drawdown/rally null models.

Do not cite it as proof of lower-threshold mean reversion, safe regression exits or XAU/USD tradability. The strongest research stack for your thesis is:

1. Hong et al. for ODB/price-duration econometrics;

2. Dimitriadis et al. for intrinsic sampling versus noise;

3. Zhang and Li for general drawdown computation;

4. Li, Zeng and Zhang for excursion speed and duration;

5. Proietti for persistence and out-of-sample forecasting;

6. Li et al. for separating jump exits from diffusion exits.

None of these papers yet establishes your specific XAU/USD result. That remains a defensible empirical contribution if it survives Brownian/time-changed-diffusion, stochastic-volatility, jump and executable bid/ask nulls.

I used the standard “first ±δ deviation from bar open” interpretation of ODB; the two `~/eon/...` directories were not present in this runtime, so implementation-specific reset or threshold details remain to be checked.

ChatGPT is AI and can make mistakes.
