# Feature Suppression Registry

> Features excluded from selection but retained in data for analysis/debugging.

**← [Back to Feature Registry](./REGISTRY.md)**

## Philosophy

Features are **suppressed** (not deleted) when:

- **Known redundant**: Highly correlated with selected features
- **Domain-specific exclusion**: Not applicable to current analysis context
- **Debugging artifacts**: Intermediate features kept for diagnostics
- **Regime-specific**: Valid only in certain market conditions

**Key principle**: Suppression is reversible. Features remain in data and can be
un-suppressed by removing from this registry.

---

## Suppression Categories

| Category   | Description                                | Typical Reason            |
| ---------- | ------------------------------------------ | ------------------------- |
| redundant  | Highly correlated with retained feature    | dCor > 0.7 or r > 0.95    |
| superseded | Older version replaced by improved feature | Algorithm upgrade         |
| debug      | Intermediate computation for diagnostics   | Not predictive            |
| regime     | Only valid in specific market conditions   | OOD robustness concern    |
| unstable   | High variance across walk-forward folds    | CV > 0.5 in WFO stability |

---

## Suppressed Features

| Feature Pattern    | Category  | Suppressed Date | Reason                                    | Superseded By            |
| ------------------ | --------- | --------------- | ----------------------------------------- | ------------------------ |
| `ith_rb25_lb*_*`   | redundant | 2026-01-23      | Highly correlated with rb1000 equivalents | `ith_rb1000_lb*_*`       |
| `ith_rb50_lb*_*`   | redundant | 2026-01-23      | Highly correlated with rb1000 equivalents | `ith_rb1000_lb*_*`       |
| `ith_rb100_lb*_*`  | redundant | 2026-01-23      | Highly correlated with rb1000 equivalents | `ith_rb1000_lb*_*`       |
| `ith_rb250_lb*_*`  | redundant | 2026-01-23      | Highly correlated with rb1000 equivalents | `ith_rb1000_lb*_*`       |
| `ith_rb500_lb*_*`  | redundant | 2026-01-23      | Highly correlated with rb1000 equivalents | `ith_rb1000_lb*_*`       |
| `ith_*_lb20_*`     | unstable  | 2026-02-02      | High variance in walk-forward validation  | `ith_*_lb100_*` or lb500 |
| `*_intermediate_*` | debug     | -               | Diagnostic features, not for ML           | -                        |

**Note**: Patterns use glob syntax. `*` matches any characters.

---

## Integration with Selection Pipeline

The suppression registry is applied **before** feature selection:

```python
from ith_python.statistical_examination.suppression import load_suppressed_patterns
from ith_python.statistical_examination.selection import select_optimal_subset

# Load suppressed patterns
suppressed = load_suppressed_patterns()

# Filter out suppressed features before selection
available_features = [f for f in all_features if not is_suppressed(f, suppressed)]

# Run selection on non-suppressed features only
result = select_optimal_subset(df, feature_cols=available_features, ...)
```

---

## Validation

Suppression decisions are logged to NDJSON for audit:

```json
{
  "ts": "2026-02-02T10:00:00Z",
  "event_type": "feature_suppression",
  "feature": "ith_rb25_lb100_bull_ed",
  "pattern_matched": "ith_rb25_lb*_*",
  "category": "redundant",
  "reason": "Highly correlated with rb1000 equivalents"
}
```

---

## Un-suppressing Features

To un-suppress a feature:

1. Remove or comment out the pattern from this registry
2. Re-run feature selection pipeline
3. Log decision in commit message with rationale

---

## Related Documentation

- [Feature Registry](./REGISTRY.md) - Active/Legacy feature definitions
- [Statistical Examination](../../packages/ith-python/src/ith_python/statistical_examination/CLAUDE.md)
- [OOD Robustness Research](../research/external/2026-02-02-feature-selection-ood-robustness-gemini.md)
