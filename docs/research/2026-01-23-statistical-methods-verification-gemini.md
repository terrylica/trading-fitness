# Statistical Verification of High-Dimensional Feature Analysis Framework

> **Source**: Gemini 2.5 Pro Deep Research
> **Date**: 2026-01-23
> **Request**: Verification of 10 statistical methods used in ITH feature examination framework

## 1. Introduction and Theoretical Architecture

The rigorous evaluation of feature engineering frameworks within quantitative finance demands a scrutiny that transcends standard textbook applications of statistical methods. The specific constraints of the dataset in question—comprising 576 features derived from price data, with a sample size of approximately N≈9000, and a strict support boundary of [0,1]—impose a unique topological and probabilistic structure that challenges the assumptions of classical parametric and non-parametric statistics.

This report provides an exhaustive verification of the ten statistical methods implemented in the user's framework: Spearman Rank Correlation, Friedman Test, Shapiro-Wilk Test, Beta Distribution Fitting, Mann-Whitney U Test, Cohen's d, Principal Component Analysis (PCA), Variance Inflation Factor (VIF), Augmented Dickey-Fuller (ADF) Test, and Autocorrelation Function (ACF).

The analysis proceeds from the premise that statistical "correctness" is not merely a matter of computational accuracy but of theoretical congruence between the data generating process (DGP) and the inferential machinery employed. In the context of financial time series, three dominant characteristics—boundedness, serial dependence, and high dimensionality—act as potential confounders for standard hypothesis testing:

1. **Boundedness** of data to the unit interval [0,1] immediately invalidates the fundamental assumption of unbounded support inherent in Gaussian-based tests
2. **Temporal dependence** inherent in price-derived features violates the Independence of Identically Distributed (I.I.D.) observations required by rank-based tests like Friedman
3. **Large sample regime** (N=9000) introduces the "p-value fallacy," where statistically significant deviations from null hypotheses become trivial in magnitude

---

## 2. Distributional Analysis and Normality Testing

### 2.1 The Shapiro-Wilk Test in Large Sample Regimes

The Shapiro-Wilk test is widely regarded as the most powerful omnibus test for univariate normality. The test statistic W is:

```
W = (Σaᵢx₍ᵢ₎)² / Σ(xᵢ - x̄)²

where x₍ᵢ₎ = i-th order statistic, aᵢ = tabulated constants
```

**Mathematical Verification and Sample Size Constraints:**

- Historical implementations limited to N≤50
- Modern implementations (Royston) valid up to N=2000
- Rahman-Govidarajulu extension valid up to N=5000
- For N≈9000, standard implementations return approximations with unverified accuracy

**Theoretical Appropriateness:**

The application of Shapiro-Wilk to data bounded in [0,1] presents a **fundamental theoretical contradiction**. The normal distribution has support (-∞, ∞). Any variable with compact support [0,1] has a probability of 0 of being normally distributed. Consequently, the null hypothesis H₀: F(x) ∈ N(μ,σ²) is **false a priori**.

Furthermore, at N=9000, the test is **statistically overpowered**. The standard error diminishes as 1/√N, rendering the test sensitive to microscopic deviations from normality that are substantively irrelevant. Rejection of the null hypothesis is virtually guaranteed for all 576 features.

**Recommendation:**

- Do NOT use as binary filter (reject/accept)
- Use W statistic as continuous metric of "Gaussianity"
- Consider Jarque-Bera test (relies on skewness/kurtosis, valid for large N)

### 2.2 Beta Distribution Fitting and Goodness-of-Fit

Given bounded data [0,1], the Beta distribution is theoretically appropriate:

```
f(x; α, β) = [Γ(α+β) / (Γ(α)Γ(β))] × x^(α-1) × (1-x)^(β-1)
```

**Parameter Estimation and Boundary Constraints:**

A critical implementation nuance: `scipy.stats.beta.fit()` has `loc` and `scale` parameters. For features strictly bounded in [0,1], the solver **must** be constrained with `floc=0` and `fscale=1`. If left free, the optimizer may "super-fit" the data by estimating a narrower support interval.

**Goodness-of-Fit Verification:**

The Kolmogorov-Smirnov test measures:

```
Dₙ = sup|Fₙ(x) - F(x; α̂, β̂)|
```

**CRITICAL VIOLATION:** Standard KS critical values assume parameters are **known a priori**, not estimated from the data. When parameters are estimated via MLE from the same sample, the empirical distance Dₙ is systematically smaller, rendering standard p-values **conservative** (fails to reject poor fits).

**Correction via Parametric Bootstrap:**

1. Estimate α̂, β̂ from original sample X
2. Compute observed statistic D_obs
3. Generate B (e.g., 1000) synthetic datasets from Beta(α̂, β̂)
4. For each synthetic dataset, re-estimate parameters and compute D\*
5. Valid p-value = proportion of D\* values exceeding D_obs

---

## 3. Rank-Based Dependence and Comparative Testing

### 3.1 Spearman Rank Correlation

**Mathematical Correctness and Ties:**

The simplified formula:

```
ρ = 1 - (6Σd²) / (n(n²-1))
```

operates on the assumption of **no tied ranks**. Financial feature data frequently exhibits ties (e.g., multiple periods where indicator = 0 or 1).

**Correct Implementation:** Calculate Pearson correlation on ranked data, or use tie-corrected formula:

```
ρ = Σ(R(Xᵢ) - R̄ₓ)(R(Yᵢ) - R̄ᵧ) / √[Σ(R(Xᵢ) - R̄ₓ)² × Σ(R(Yᵢ) - R̄ᵧ)²]
```

**Appropriateness:** With N=9000, critical value for significance at α=0.05 is ≈0.02. Features with negligible correlation will be deemed "significant." **Ignore p-values; filter based on effect size (e.g., |ρ| > 0.1).**

### 3.2 The Friedman Test: A Critical Evaluation

The Friedman test statistic:

```
Q = (12 / nk(k+1)) × Σ(Rⱼ)² - 3n(k+1)
```

**CRITICAL ASSUMPTION VIOLATION: Independence of Blocks**

The derivation relies on blocks (rows) being **mutually independent**. In the dataset, rows correspond to time steps t, t+1, ... Financial time series have serial dependence—the state at time t is correlated with t-1.

**Consequences:**

- Effective sample size is reduced
- Variance of rank sums is mis-estimated
- With positive serial correlation: **inflated Type I error rate**
- "Significant differences" may be artifacts of persistent market regimes

**Robust Alternatives:**

1. **Block Bootstrap** - Resample blocks preserving autocorrelation structure
2. **Quade Test** - Weights ranks by dispersion (but still assumes independence)
3. **Skillings-Mack Test** - Handles missing data (shares independence assumption)

**Recommendation:** Given failure of independence assumption, **discard Friedman test** or implement Block Bootstrap for valid p-values.

### 3.3 Mann-Whitney U Test

```
U = n₁n₂ + n₁(n₁+1)/2 - R₁
```

**Verification:** It is a common misconception that Mann-Whitney tests for median difference. It tests for **stochastic dominance** (whether values from one sample are systematically larger).

**Effect Size:** Report **Cliff's Delta**:

```
δ = (2U / n₁n₂) - 1
```

Cliff's Delta provides a metric on [-1, 1] representing degree of non-overlap between distributions.

---

## 4. Effect Size Estimation and Multicollinearity

### 4.1 Cohen's d

**Mathematical Verification:** For independent samples of unequal size, use **weighted pooled standard deviation**:

```
s_pooled = √[((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2)]
```

The simple average √((s₁² + s₂²)/2) is **incorrect** for unequal sample sizes.

**Assumption Suitability:** Cohen's d assumes normality and homogeneity of variance. For Beta distributions, variance is coupled to mean (σ² ≈ μ(1-μ)). Comparing features with different means using pooled SD is methodologically flawed.

**Recommendation:** Use **Cliff's Delta** (non-parametric) instead of Cohen's d.

### 4.2 Variance Inflation Factor (VIF)

```
VIF_i = 1 / (1 - R²_i)
```

**High-Dimensional Stability:** Calculating VIF for 576 features requires inverting a 576×576 correlation matrix. Financial features are often highly collinear, leading to ill-conditioned matrices with determinant near zero. Numerical inversion is unstable, potentially producing VIF values of 10¹⁴ as floating-point artifacts.

**Recommendation:** Use **Regularized VIF** or iterative feature elimination based on correlation thresholds before computing full VIF.

---

## 5. Dimensionality Reduction and PCA

### 5.1 Covariance vs. Correlation Matrix

For variables bounded in [0,1]:

- **Covariance Matrix** preserves variance differences (prioritizes high-amplitude features)
- **Correlation Matrix** standardizes to unit variance (treats all features equally)

Given that noise in financial data often manifests as high variance, **Correlation Matrix is generally safer**.

### 5.2 The Participation Ratio (D_PR)

A rigorous measure of effective dimensionality from statistical physics:

```
D_PR = (Σλᵢ)² / Σλᵢ² = (Tr(C))² / Tr(C²)
```

**Interpretation:**

- Variance evenly distributed across all K dimensions → D_PR ≈ K
- Variance concentrated in single dimension → D_PR ≈ 1

This provides theoretically grounded target for dimensionality reduction.

---

## 6. Time Series Properties

### 6.1 Augmented Dickey-Fuller (ADF) Test

```
Δyₜ = α + βt + γyₜ₋₁ + Σδⱼ Δyₜ₋ⱼ + εₜ
```

**The Bounded Random Walk Paradox:**

A true unit root process has variance growing linearly with time: Var(yₜ) = t×σ². As t → ∞, variance approaches infinity. But a variable bounded in [0,1] has maximum variance of 0.25. Therefore, a bounded variable **cannot** be a unit root process.

**Interpretation:** The ADF test on bounded data will almost certainly reject the null (boundaries force mean reversion). Use as test of **local persistence** rather than global stationarity.

**Lag Selection:** Use AIC/BIC to select lag length p.

### 6.2 Autocorrelation Function (ACF)

**Formula Verification:**

```
ρ̂(k) = Σ(yₜ - ȳ)(yₜ₊ₖ - ȳ) / Σ(yₜ - ȳ)²
```

Uses full sample variance in denominator (ensures positive definite autocorrelation matrix).

**LSTM Sequence Length Heuristics:**

1. **Decay to Noise:** Lag where ACF falls within ±1.96/√N (≈0.02 for N=9000)
2. **Decay to e⁻¹:** Lag where ACF < 0.367 (correlation time)
3. **First Zero Crossing:** First lag where ACF < 0

**Practical Constraint:** While ACF might suggest 500+ steps, LSTMs struggle with vanishing gradients for sequences >200. Slow ACF decay indicates long-memory processes requiring Transformers or Dilated CNNs.

---

## 7. Summary: Methodological Audit and Corrective Actions

| Method           | Status            | Critical Issue                                        | Corrective Action                                  |
| ---------------- | ----------------- | ----------------------------------------------------- | -------------------------------------------------- |
| **Spearman**     | Correction Needed | Ties invalidate simple formula                        | Use Pearson-on-ranks or correction factor T        |
| **Friedman**     | **INVALID**       | Independence violation: time series blocks correlated | **Discard.** Use Block Bootstrap if essential      |
| **Shapiro-Wilk** | Weak              | Overpowered at N=9000; bounds preclude normality      | Use W as continuous metric; not binary filter      |
| **Beta Fit**     | Correction Needed | Estimation bias; goodness-of-fit biased               | Fix loc=0, scale=1. Use Parametric Bootstrap KS    |
| **Mann-Whitney** | Valid             | Tests stochastic dominance, not median                | Report Cliff's Delta for effect size               |
| **Cohen's d**    | Approximation     | Assumes normality/homogeneity                         | Use weighted pooled variance. Prefer Cliff's Delta |
| **PCA**          | Valid             | Variance disparity in [0,1] data                      | Use Participation Ratio (D_PR) for dimensionality  |
| **VIF**          | Valid             | Matrix inversion instability                          | Use Regularized VIF or iterative elimination       |
| **ADF**          | Nuanced           | Bounds enforce global stationarity                    | Interpret as test for local persistence            |
| **ACF**          | Valid             | Denominator choice                                    | Use to define LSTM window (decay to noise/zero)    |

---

## 8. Priority Fixes Required

**CRITICAL (Invalid Results):**

1. **Friedman Test** - Results invalid due to autocorrelation. Must discard or implement Block Bootstrap.
2. **Beta Distribution Fit** - Must constrain loc/scale and use Parametric Bootstrap KS for p-values.

**HIGH (Incorrect Implementation):** 3. **Spearman Correlation** - Must handle ties properly (Pearson-on-ranks). 4. **Cohen's d** - Must use weighted pooled SD formula for unequal samples.

**MEDIUM (Interpretation Issues):** 5. **Shapiro-Wilk** - Don't use as binary filter; use W as continuous metric. 6. **Mann-Whitney** - Report Cliff's Delta, not just p-value. 7. **VIF** - Use regularization for numerical stability.

**LOW (Theoretical Awareness):** 8. **ADF** - Interpret as local persistence test for bounded data. 9. **PCA** - Consider Participation Ratio for effective dimensionality. 10. **ACF** - Standard implementation is correct; use for LSTM window selection.
