# Statistical Examination Framework

> Analyze ITH multi-scale features for redundancy, stability, and ML readiness.

**← [Back to ith-python](../../../CLAUDE.md)**

> **Note**: This module is the SSoT for statistical methods. It is wrapped by `ith_python.analysis` (Layer 3).
>
> ```python
> from ith_python.analysis import AnalysisConfig, analyze_features
> ```

## Quick Start

```bash
cd packages/ith-python
uv run python -m ith_python.statistical_examination.runner \
    --thresholds 25,50,100,250 \
    --lookbacks 20,50,100,200,500 \
    --output-dir ../../artifacts/statistical_examination
```

---

## Module Structure

### Core Analysis Modules

| Module                   | Purpose                               | Method                              |
| ------------------------ | ------------------------------------- | ----------------------------------- |
| `runner.py`              | CLI + orchestration                   | -                                   |
| `cross_scale.py`         | Feature correlation across lookbacks  | Spearman rank correlation           |
| `threshold_stability.py` | Feature consistency across thresholds | CV-based stability                  |
| `distribution.py`        | Feature distribution shapes           | Shapiro-Wilk W, Beta fit + AD test  |
| `regime.py`              | Trending vs mean-reverting behavior   | Mann-Whitney U, Cliff's Delta       |
| `dimensionality.py`      | Redundancy detection                  | PCA, Participation Ratio, Ridge VIF |
| `selection.py`           | Optimal feature subset                | Variance + correlation filters      |
| `temporal.py`            | Time-series properties                | ACF, ADF stationarity               |
| `schemas.py`             | Data validation                       | Pandera                             |

### OOD-Robust Feature Selection Pipeline (2026-02-04)

Principled pipeline for high-ACF financial time series: `mRMR -> dCor -> PCMCI -> Stability`

| Module               | Phase | Purpose                         | Method                                        |
| -------------------- | ----- | ------------------------------- | --------------------------------------------- |
| `suppression.py`     | 0     | Filter known-unstable features  | Pattern matching (fnmatch)                    |
| `mrmr.py`            | 1     | Fast redundancy-aware filtering | Minimum Redundancy Max Relevance              |
| `dcor_filter.py`     | 2     | Nonlinear redundancy detection  | Distance Correlation (dCor=0 iff independent) |
| `pcmci_filter.py`    | 3     | Causal discovery under ACF      | PCMCI via tigramite                           |
| `block_bootstrap.py` | 4a    | ACF-robust importance stability | Circular Block Bootstrap + Politis-White      |
| `walk_forward.py`    | 4b    | Temporal importance stability   | TimeSeriesSplit CV + CV filter                |
| `_utils.py`          | -     | Shared utilities                | Feature column detection                      |
| `scattering.py`      | Alt   | Automatic multi-scale features  | Wavelet scattering transform (kymatio fork)   |

**Pipeline Flow**: 160 features -> 50 (mRMR) -> 30 (dCor) -> 15 (PCMCI) -> 10 (Stability)

**Alternative**: Scattering transform sidesteps manual lookback selection by extracting features at ALL temporal scales automatically.

**GitHub Issue**: [cc-skills#21](https://github.com/terrylica/cc-skills/issues/21)

---

## Methods Rectification (2026-01-23)

| Method            | Issue                                  | Fix Applied                        |
| ----------------- | -------------------------------------- | ---------------------------------- |
| **Friedman Test** | Independence violation (time series)   | **REMOVED** - CV stability instead |
| **Beta Fit KS**   | Invalid p-values with estimated params | AD test with parametric bootstrap  |
| **Cohen's d**     | Wrong pooled SD for unequal samples    | Weighted formula                   |
| **VIF**           | Matrix instability at 576 features     | Ridge VIF (cond=100)               |
| **PCA**           | Missing effective dimensionality       | Added Participation Ratio          |

**Research**: `docs/research/2026-01-23-statistical-methods-verification-gemini.md`

---

## Key Thresholds

| Analysis             | Threshold | Interpretation                   |
| -------------------- | --------- | -------------------------------- |
| Cross-scale corr     | > 0.9     | Highly redundant                 |
| Threshold CV         | < 0.2     | Stable across thresholds         |
| Shapiro-Wilk W       | > 0.95    | Approximately normal             |
| Cliff's Delta        | > 0.30    | Large effect (check for bias!)   |
| VIF                  | > 10      | Severe multicollinearity         |
| PCA components (95%) | < 10%     | High redundancy in feature space |

**Note**: Cliff's Delta thresholds (0.05/0.15/0.30) are finance-specific, NOT Cohen's behavioral thresholds.

---

## Output Artifacts

| File                 | Format  | Contents               |
| -------------------- | ------- | ---------------------- |
| `features.parquet`   | Parquet | All features, all rows |
| `summary.json`       | JSON    | Analysis results       |
| `examination.ndjson` | NDJSON  | Event log with timing  |

---

## NDJSON Telemetry

```json
{
  "ts": "2026-01-26T01:08:28Z",
  "event_type": "hypothesis_result",
  "test_name": "shapiro_wilk",
  "statistic": 0.847,
  "p_value": 0.001,
  "decision": "non_normal",
  "feature": "ith_rb25_lb100_bull_ed"
}
```

```bash
# Query telemetry
jq 'select(.decision == "non_normal") | .feature' logs/ndjson/statistical_examination.jsonl
```

---

## Dependencies

```toml
examination = ["polars", "scipy", "scikit-learn", "pandera"]
feature-selection = ["mrmr-selection", "dcor", "tigramite", "recombinator", "tscv"]
```

---

## Feature Selection Tasks (mise)

```bash
# Run full pipeline
mise run feature-selection:pipeline

# Run individual phases
mise run feature-selection:mrmr        # Phase 1: 160 -> 50
mise run feature-selection:dcor        # Phase 2: 50 -> 30
mise run feature-selection:pcmci       # Phase 3: 30 -> 15
mise run feature-selection:stability   # Phase 4: 15 -> 10

# Run tests
mise run feature-selection:test        # 64 pipeline tests
mise run feature-selection:validate    # Full test suite (172 tests)
```

---

## Related Documentation

| Document                                                                                         | Purpose                      |
| ------------------------------------------------------------------------------------------------ | ---------------------------- |
| [ith-python CLAUDE.md](../../../CLAUDE.md)                                                       | Parent package               |
| [docs/ITH.md](../../../../../../docs/ITH.md)                                                     | ITH methodology              |
| [docs/LOGGING.md](../../../../../../docs/LOGGING.md)                                             | Telemetry contract           |
| [docs/forensic/E2E.md](../../../../../../docs/forensic/E2E.md)                                   | E2E pipeline                 |
| [docs/features/SUPPRESSION_REGISTRY.md](../../../../../../docs/features/SUPPRESSION_REGISTRY.md) | Feature suppression patterns |
| [cc-skills#21](https://github.com/terrylica/cc-skills/issues/21)                                 | Feature selection research   |
