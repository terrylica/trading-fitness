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
```

---

## Related Documentation

| Document                                                       | Purpose            |
| -------------------------------------------------------------- | ------------------ |
| [ith-python CLAUDE.md](../../../CLAUDE.md)                     | Parent package     |
| [docs/ITH.md](../../../../../../docs/ITH.md)                   | ITH methodology    |
| [docs/LOGGING.md](../../../../../../docs/LOGGING.md)           | Telemetry contract |
| [docs/forensic/E2E.md](../../../../../../docs/forensic/E2E.md) | E2E pipeline       |
