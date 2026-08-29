# Feature Registry

> Single source of truth for all extractable features in trading-fitness.

**← [Back to CLAUDE.md](../../CLAUDE.md)**

## Philosophy

We try **many features**, but never delete them. Features are either:

- **Active**: Currently used in analysis
- **Legacy**: Disabled due to redundancy (kept for historical reference)
- **Experimental**: Under evaluation

This creates an archive of what works, what doesn't, and why.

---

## Feature Categories

| Category                                                    | Source                  | Status       | Count | Docs             |
| ----------------------------------------------------------- | ----------------------- | ------------ | ----- | ---------------- |
| [ITH Core](#ith-core-features)                              | ith-python              | Active       | 8     | Below            |
| [ITH Multi-scale](#ith-multi-scale-features)                | metrics-rust            | Active       | 384   | Below            |
| [ClaSPy Segmentation](#claspy-segmentation-features)        | claspy                  | Experimental | 12+   | [→](./CLASPY.md) |
| [Statistical Examination](#statistical-examination-derived) | statistical_examination | Active       | 6     | Below            |

---

## ITH Core Features

| Feature        | Type  | Range   | Description              | Status |
| -------------- | ----- | ------- | ------------------------ | ------ |
| `bull_epochs`  | int   | [0, ∞)  | Count of bull ITH epochs | Active |
| `bear_epochs`  | int   | [0, ∞)  | Count of bear ITH epochs | Active |
| `bull_cv`      | float | [0, ∞)  | Bull epoch interval CV   | Active |
| `bear_cv`      | float | [0, ∞)  | Bear epoch interval CV   | Active |
| `max_drawdown` | float | [0, 1]  | Maximum drawdown         | Active |
| `max_runup`    | float | [0, 1]  | Maximum runup            | Active |
| `sharpe_ratio` | float | (-∞, ∞) | Risk-adjusted return     | Active |
| `tmaeg`        | float | [0, 1]  | TMAEG threshold used     | Active |

---

## ITH Multi-scale Features

Generated via `metrics-rust` with configurable thresholds and lookbacks.

**Naming Convention**: `ith_rb{threshold}_lb{lookback}_{type}`

| Parameter   | Values                                  | Description                |
| ----------- | --------------------------------------- | -------------------------- |
| `threshold` | 25, 50, 100, 250, 500, 1000             | Range bar threshold (dbps) |
| `lookback`  | 20, 50, 100, 200, 500, 1000, 1500, 2000 | Lookback window            |
| `type`      | bull_eg, bull_cv, bear_eg, bear_cv, ... | Feature type               |

**Total**: 6 thresholds × 8 lookbacks × 8 types = **384 features**

### Redundancy Analysis (2026-01-23)

From statistical examination on sample_account.csv (2,726 NAV observations):

| Metric                   | Value | Interpretation                 |
| ------------------------ | ----- | ------------------------------ |
| Cross-scale correlation  | 0.106 | Low - features are independent |
| PCA components (95% var) | 28    | High redundancy                |
| Participation ratio      | 13.3  | ~13 effective dimensions       |
| Selected features        | 18    | All from rb1000 threshold      |

**Recommended**: Use rb1000 threshold features only. Others are redundant.

---

## ClaSPy Segmentation Features

**Status**: Experimental (2026-01-25)

| Feature                  | Type  | Range  | Description                  | Computational Cost |
| ------------------------ | ----- | ------ | ---------------------------- | ------------------ |
| `clasp_n_segments`       | int   | [1, ∞) | Number of detected segments  | Medium             |
| `clasp_n_changepoints`   | int   | [0, ∞) | Number of change points      | Medium             |
| `clasp_profile_mean`     | float | [0, 1] | Mean of score profile        | Low                |
| `clasp_profile_max`      | float | [0, 1] | Max score (strongest change) | Low                |
| `clasp_profile_std`      | float | [0, 1] | Profile variability          | Low                |
| `clasp_segment_mean_len` | float | [0, ∞) | Mean segment duration        | Low                |
| `clasp_segment_cv`       | float | [0, ∞) | Segment length CV            | Low                |
| `clasp_first_cp_idx`     | int   | [0, n] | Index of first change point  | Low                |
| `clasp_last_cp_idx`      | int   | [0, n] | Index of last change point   | Low                |
| `clasp_cp_density`       | float | [0, 1] | Change points per unit time  | Low                |
| `clasp_window_size`      | int   | [3, ∞) | Auto-detected window size    | Low                |
| `clasp_max_score`        | float | [0, 1] | Highest change point score   | Low                |

**Deep Dive**: [ClaSPy Integration](./CLASPY.md)

---

## Statistical Examination Derived

Features derived from statistical examination of multi-scale ITH features.

| Feature                    | Type  | Description                   | Status |
| -------------------------- | ----- | ----------------------------- | ------ |
| `effective_dimensionality` | float | PCA participation ratio       | Active |
| `feature_stability_rate`   | float | % features with CV < 0.2      | Active |
| `gaussianity_ratio`        | float | % practically normal features | Active |
| `regime_dependence_rate`   | float | % features with regime effect | Active |
| `cross_scale_correlation`  | float | Mean Spearman correlation     | Active |
| `selected_feature_count`   | int   | Features after filtering      | Active |

---

## Legacy Features (Disabled)

Features disabled due to redundancy or poor performance.

| Feature    | Disabled Date | Reason | Superseded By |
| ---------- | ------------- | ------ | ------------- |
| (none yet) | -             | -      | -             |

---

## Feature Discovery Workflow

```
1. Add experimental feature
2. Run statistical examination
3. Check redundancy via:
   - Cross-scale correlation
   - PCA/VIF analysis
   - Regime dependence
4. If redundant: Mark as Legacy
5. If orthogonal: Mark as Active
6. Log decision to NDJSON
```

---

## NDJSON Artifact Format

Feature discovery results are logged to `artifacts/feature_discovery/`.

```json
{
  "ts": "2026-01-25T00:00:00Z",
  "event": "feature_evaluation",
  "feature_name": "clasp_n_segments",
  "status": "experimental",
  "metrics": {
    "correlation_with_existing": 0.15,
    "information_gain": 0.23,
    "computational_cost_ms": 150
  },
  "decision": "promote_to_active",
  "reason": "Low correlation, high information gain"
}
```

---

## Adding New Features

1. Create feature extractor in appropriate module
2. Add to this registry with status "Experimental"
3. Run statistical examination
4. Update status based on results
5. Document in child CLAUDE.md

---

## Related Documentation

- [ClaSPy Integration](./CLASPY.md)
- [Statistical Examination](../../packages/ith-python/src/ith_python/statistical_examination/CLAUDE.md)
- [Telemetry Events](../../packages/ith-python/src/ith_python/telemetry/)
- [Root CLAUDE.md](../../CLAUDE.md)
