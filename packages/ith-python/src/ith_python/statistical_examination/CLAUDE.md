# Statistical Examination Framework

> Analyze ITH multi-scale features for redundancy, stability, and ML readiness.

**[Back to ith-python](../../../CLAUDE.md)**

## Quick Start

```bash
cd packages/ith-python
UV_PYTHON=python3.13 uv run python -m ith_python.statistical_examination.runner \
    --thresholds 25,50,100,250,500,1000 \
    --lookbacks 20,50,100,200,500,1000,1500,2000,3000,4000,5000,6000 \
    --output-dir ../../artifacts/statistical_examination
```

## Module Structure

| Module                   | Purpose                                | Mathematical Method                        |
| ------------------------ | -------------------------------------- | ------------------------------------------ |
| `runner.py`              | CLI + orchestration                    | -                                          |
| `cross_scale.py`         | Feature correlation across lookbacks   | Spearman rank correlation                  |
| `threshold_stability.py` | Feature consistency across thresholds  | CV-based stability (Friedman test removed) |
| `distribution.py`        | Feature distribution shapes            | Shapiro-Wilk W, Beta fit with AD test      |
| `regime.py`              | Behavior in trending vs mean-reverting | Mann-Whitney U, Cohen's d, Cliff's Delta   |
| `dimensionality.py`      | Redundancy detection                   | PCA, Participation Ratio, Ridge VIF        |
| `selection.py`           | Optimal feature subset                 | Variance filter, correlation filter        |
| `temporal.py`            | Time-series properties                 | ACF, ADF stationarity test                 |
| `schemas.py`             | Data validation                        | Pandera                                    |
| `_utils.py`              | Column parsing, warmup handling        | -                                          |

---

## Methods Rectification (2026-01-23)

The following statistical methods were audited and corrected based on AI deep research verification:

| Method            | Issue                                                  | Fix Applied                                                         |
| ----------------- | ------------------------------------------------------ | ------------------------------------------------------------------- |
| **Friedman Test** | Independence violation (time series blocks correlated) | **REMOVED** - replaced with CV stability                            |
| **Beta Fit KS**   | Standard KS p-values invalid when params estimated     | Two-stage AD test with `scipy.stats.goodness_of_fit`                |
| **Cohen's d**     | Wrong pooled SD formula for unequal samples            | Weighted formula: `sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))`   |
| **Shapiro-Wilk**  | Overpowered at N>5000; bounds preclude normality       | W reported as continuous metric, not binary filter                  |
| **Mann-Whitney**  | Reports p-value only, not effect size                  | Added Cliff's Delta with finance-specific thresholds                |
| **VIF**           | Matrix inversion instability at 576 features           | Ridge VIF with condition number targeting ~100                      |
| **PCA**           | Missing effective dimensionality metric                | Added Participation Ratio: `D_PR = (sum(lambda))^2 / sum(lambda^2)` |
| **ADF**           | Misinterpretation for bounded [0,1] data               | Documented as test of LOCAL PERSISTENCE, not global stationarity    |

**Reference**: `docs/research/2026-01-23-statistical-methods-verification-gemini.md`

---

## Analysis Methods Explained

### 1. Cross-Scale Correlation

**Question:** Do features measured at different lookback windows contain redundant information?

**Method:** Spearman rank correlation between each pair of lookbacks for the same feature type.

**Math:**

```
rho = 1 - (6 * sum(d^2)) / (n(n^2-1))

where d = difference in ranks between paired observations
```

**Interpretation:**

| Correlation | Meaning                                  |
| ----------- | ---------------------------------------- |
| > 0.9       | Highly redundant - consider dropping one |
| 0.5 - 0.9   | Moderate overlap                         |
| < 0.5       | Mostly independent information           |

**Why Spearman (not Pearson)?** ITH features are bounded [0,1] and often non-normal. Spearman handles non-linear monotonic relationships and is robust to outliers.

**Code:** `cross_scale.py:compute_cross_scale_correlation()`

---

### 2. Threshold Stability

**Question:** Do features remain consistent when the range bar threshold changes?

**Method:** CV across threshold means.

> **NOTE (2026-01-23):** Friedman test was **REMOVED** due to independence violation. The Friedman test assumes mutually independent blocks, but time series rows are serially correlated. This inflates Type I error rates and produces invalid p-values.

**Math:**

```
CV = std(means) / mean(means)

where means = [mean(feature_at_threshold_1), mean(feature_at_threshold_2), ...]
```

**Interpretation:**

| CV     | Meaning                                       |
| ------ | --------------------------------------------- |
| < 0.2  | Stable - threshold choice doesn't matter much |
| >= 0.2 | Unstable - significant threshold effect       |

**Code:** `threshold_stability.py:compute_threshold_stability()`

---

### 3. Distribution Analysis

**Question:** What statistical distribution do features follow?

**Methods:**

1. **Shapiro-Wilk W** - continuous Gaussianity metric (NOT binary filter)
2. **Beta distribution fit** - natural for [0,1] bounded data
3. **Anderson-Darling test** - goodness of fit with parametric bootstrap

> **NOTE (2026-01-23):** KS test was **REPLACED** with Anderson-Darling. Standard KS p-values are invalid when distribution parameters are estimated from the same data. AD test uses `scipy.stats.goodness_of_fit` with parametric bootstrap for valid p-values.

**Math (Shapiro-Wilk):**

```
W = (sum(a_i * x_(i)))^2 / sum((x_i - x_bar)^2)

where x_(i) = ordered sample values, a_i = tabulated constants
```

**Shapiro-Wilk W Interpretation:**

| W Value   | Gaussianity Classification |
| --------- | -------------------------- |
| > 0.99    | Practically normal         |
| 0.95-0.99 | Minor departures           |
| 0.90-0.95 | Moderate non-normality     |
| < 0.90    | Substantial non-normality  |

> **Note:** W is a continuous metric. At N>5000, p-values are unreliable (overpowered). Bounded [0,1] data cannot be truly normal.

**Two-Stage Anderson-Darling Test:**

- **Stage 1**: B=99 iterations at alpha=0.10 (fast screening)
- **Stage 2**: B=999 only for borderline cases (0.001 < p < 0.10)

AD test emphasizes tails (critical for financial data) and uses parametric bootstrap via `scipy.stats.goodness_of_fit` for valid p-values.

**Code:** `distribution.py:analyze_distribution()`, `distribution.py:analyze_beta_fit()`

---

### 4. Regime Dependence

**Question:** Do features behave differently in trending vs mean-reverting markets?

**Methods:**

1. **Regime detection** - Hurst exponent proxy on rolling windows
2. **Mann-Whitney U test** - compare distributions between regimes
3. **Cliff's Delta** - primary non-parametric effect size (NEW)
4. **Cohen's d** - corrected parametric effect size

**Math (Cliff's Delta):**

```
delta = 1 - 2*U / (n1 * n2)

where U = Mann-Whitney U statistic
```

Cliff's Delta is mathematically identical to rank-biserial correlation. Measures the probability that a randomly selected value from group 1 is greater than a randomly selected value from group 2, minus the reverse probability.

**Math (Cohen's d - CORRECTED):**

```
d = (mu_1 - mu_2) / s_pooled

where s_pooled = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))
```

> **NOTE (2026-01-23):** The pooled SD formula was corrected. The simple average `sqrt((s1^2 + s2^2)/2)` is **WRONG** for unequal sample sizes.

**Finance-Specific Effect Size Thresholds (Cliff's Delta):**

|           | delta                                         |     | Interpretation |
| --------- | --------------------------------------------- | --- | -------------- |
| < 0.05    | Negligible (likely noise)                     |
| 0.05-0.15 | Small but potentially tradable                |
| 0.15-0.30 | Medium/strong signal                          |
| > 0.30    | Large/suspicious (check for look-ahead bias!) |

> **WARNING:** These are NOT Cohen's behavioral science thresholds (0.2/0.5/0.8). In efficient markets, large effect sizes are suspicious.

**Code:** `regime.py:analyze_regime_dependence()`, `regime.py:cliffs_delta()`, `regime.py:cohens_d_corrected()`

---

### 5. PCA Dimensionality

**Question:** How many truly independent dimensions exist in the feature space?

**Method:** Principal Component Analysis (PCA) with Participation Ratio

**Math:**

```
Covariance matrix: C = (1/n) X^T X
Eigendecomposition: C = V * Lambda * V^T

Explained variance ratio: lambda_i / sum(lambda_j)
```

**Participation Ratio (NEW):**

```
D_PR = (sum(lambda))^2 / sum(lambda^2)
```

The Participation Ratio measures effective dimensionality from statistical physics. It gives the number of "participating" dimensions, accounting for unequal eigenvalue contributions.

**Key Metrics:**

| Metric               | Meaning                            |
| -------------------- | ---------------------------------- |
| n_components_95      | Components needed for 95% variance |
| Participation Ratio  | Effective dimensionality (D_PR)    |
| Dimensionality ratio | n_components_95 / total_features   |

**Interpretation:** If 576 features compress to 54 components (9%), then ~90% of features are redundant.

**Code:** `dimensionality.py:perform_pca()`, `dimensionality.py:participation_ratio()`

---

### 6. Variance Inflation Factor (VIF)

**Question:** Are features too correlated to use together in regression?

**Methods:**

1. **Standard VIF** - for smaller feature sets
2. **Ridge VIF** - regularized VIF for large/ill-conditioned matrices (NEW)

**Math (Standard VIF):**

```
VIF_i = 1 / (1 - R^2_i)

where R^2_i = R-squared from regressing feature i on all other features
```

**Math (Ridge VIF - NEW):**

```
R_reg = (R + lambda * I) / (1 + lambda)
VIF_ridge = diag(R_reg^(-1))

Lambda selection: lambda = max(0, (lambda_max - c * lambda_min) / (c - 1))
where c = target condition number (default 100)
```

> **NOTE (2026-01-23):** Ridge VIF was added to handle matrix inversion instability with 576 features. The regularization targets a condition number of ~100 using the Garcia et al. (2015) formula.

**Interpretation:**

| VIF  | Meaning                       |
| ---- | ----------------------------- |
| 1    | No multicollinearity          |
| 1-5  | Moderate - usually acceptable |
| 5-10 | High - consider removing      |
| > 10 | Severe - definitely remove    |

**Code:** `dimensionality.py:compute_vif()`, `dimensionality.py:compute_vif_regularized()`

---

### 7. Temporal Analysis

**Question:** Are features stationary (stable over time)?

**Methods:**

1. **Autocorrelation Function (ACF)** - correlation with lagged self
2. **Augmented Dickey-Fuller (ADF) test** - unit root test for stationarity

**Math (ACF at lag k):**

```
rho_k = Cov(X_t, X_{t-k}) / Var(X_t)
```

**Math (ADF):**

```
Delta(X_t) = alpha + beta * X_{t-1} + epsilon

H0: beta = 0 (unit root, non-stationary)
H1: beta < 0 (stationary)
```

**Interpretation:**

| ADF statistic | p-value | Meaning                             |
| ------------- | ------- | ----------------------------------- |
| < -2.86       | < 0.05  | Stationary - safe for ML            |
| >= -2.86      | >= 0.05 | Non-stationary - needs differencing |

> **NOTE (2026-01-23) - ADF Interpretation for Bounded [0,1] Data:**
>
> A true unit root process has variance growing linearly with time: `Var(y_t) = t * sigma^2`. As t approaches infinity, variance approaches infinity. But a variable bounded in [0,1] has maximum variance of 0.25. Therefore, a bounded variable **CANNOT** be a unit root process.
>
> The ADF test on bounded data will almost certainly reject the null (boundaries force mean reversion). **Interpret as test of LOCAL PERSISTENCE rather than global stationarity.**

**Why Stationarity Matters:** Non-stationary features cause spurious correlations and models that don't generalize.

**Code:** `temporal.py:compute_stationarity()`, `temporal.py:compute_autocorrelation()`

---

### 8. Feature Selection

**Question:** Which features should we actually use?

**Method:** Sequential filtering pipeline:

1. **Variance filter** - remove near-constant features
2. **Correlation filter** - remove redundant features (r > 0.95)
3. **Information filter** - keep high mutual information with target (if provided)

**Math (Mutual Information):**

```
MI(X;Y) = sum_x sum_y p(x,y) log(p(x,y) / (p(x)p(y)))
```

**Code:** `selection.py:select_optimal_subset()`

---

## Output Artifacts

| File                 | Format          | Contents                            |
| -------------------- | --------------- | ----------------------------------- |
| `features.parquet`   | Parquet (Arrow) | All 576 features, all rows          |
| `summary.json`       | JSON            | Analysis results, selected features |
| `examination.ndjson` | NDJSON          | Event log with timing               |

---

## Typical Results Interpretation

Example from 15,000-bar synthetic NAV:

| Analysis                       | Result            | Interpretation                          |
| ------------------------------ | ----------------- | --------------------------------------- |
| Cross-scale correlation: 0.079 | Low               | Lookbacks provide independent info      |
| Threshold stability (CV-based) | Varies by feature | Use CV < 0.2 as stability threshold     |
| Distribution: 28 right-skewed  | Non-normal        | Use robust methods, consider transforms |
| PCA: 54 components (9.4%)      | High redundancy   | 90% of features are redundant           |
| Participation Ratio: ~50       | Effective dims    | Confirms PCA finding                    |
| VIF: 1/50 high                 | Low collinearity  | Features can be used together           |
| Stationarity: 86%              | Mostly stationary | Good for ML modeling                    |
| Selection: 576 -> 24           | 96% reduction     | Use these 24 features                   |

---

## Mathematical Verification Checklist

For auditing the statistical methods:

- [x] **Spearman correlation**: Verify rank transformation before Pearson
- [x] **Threshold stability**: CV-based (Friedman removed due to independence violation)
- [x] **Shapiro-Wilk**: W as continuous metric; thresholds 0.99/0.95/0.90
- [x] **Beta fit**: Anderson-Darling with parametric bootstrap (KS removed)
- [x] **Mann-Whitney**: Two-sided alternative with Cliff's Delta effect size
- [x] **Cohen's d**: Weighted pooled SD formula for unequal samples
- [x] **PCA**: Participation Ratio for effective dimensionality
- [x] **VIF**: Ridge regularization for ill-conditioned matrices
- [x] **ADF**: Documented as LOCAL PERSISTENCE test for bounded data
- [x] **ACF**: Verify lag-k covariance normalization

---

## Implementation Constants

| Parameter            | Value                    | Rationale                                |
| -------------------- | ------------------------ | ---------------------------------------- |
| **Friedman Test**    | **REMOVED**              | Independence violation for time series   |
| **Beta GoF Test**    | **Anderson-Darling**     | Better tail sensitivity for finance      |
| **GoF Iterations**   | B=99 screen, B=999 final | Two-stage saves ~80% compute             |
| **Effect Size**      | **Cliff's Delta**        | delta = rank-biserial (identical)        |
| **Delta Thresholds** | 0.05/0.15/0.30           | Finance-specific (NOT Cohen's)           |
| **VIF Method**       | Ridge, cond=100          | Garcia formula: diag((R+lambda\*I)^(-1)) |
| **Shapiro-Wilk**     | W as metric              | Thresholds: 0.99/0.95/0.90               |
| **CV Stability**     | < 0.20                   | Feature considered threshold-stable      |

---

## Dependencies

```toml
# In pyproject.toml [project.optional-dependencies]
examination = [
    "polars",        # DataFrame operations
    "scipy",         # Statistical tests
    "scikit-learn",  # PCA, MI
    "pandera",       # Validation
]
```

---

## Related Documentation

- **ITH Methodology**: [docs/ITH.md](../../../../../docs/ITH.md)
- **Logging Contract**: [docs/LOGGING.md](../../../../../docs/LOGGING.md)
- **Parent Package**: [ith-python CLAUDE.md](../../../CLAUDE.md)
- **Research Reference**: `docs/research/2026-01-23-statistical-methods-verification-gemini.md`
