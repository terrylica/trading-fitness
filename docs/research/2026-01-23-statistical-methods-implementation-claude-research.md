# Statistical Methods for Financial Time Series Feature Selection

> **Source**: Claude Research
> **Date**: 2026-01-23
> **URL**: <https://claude.ai/public/artifacts/9504f584-2a5c-450c-b832-3341c55b1d17>

---

## Executive Summary

**Cliff's Delta and rank-biserial correlation are mathematically identical for two independent samples—report either.** For your pipeline with **9,000 observations** and **576 features** bounded in [0,1], the key recommendations are:

- Use **Stationary Bootstrap with block size ~21** and **B≥2,000** iterations
- Prefer **Anderson-Darling over KS** for Beta goodness-of-fit testing with **B=99 for screening, B=999 for final decisions**
- Apply **Ridge-based VIF** with λ targeting condition number ~100 after pre-filtering at |r|>0.95
- For practical pipelines, **stability metrics often outperform formal hypothesis testing**

---

## 1. Block Bootstrap Configuration for Friedman Test

The **Politis-White (2004) automatic block selection** method estimates optimal block length by identifying where the correlogram becomes negligible, then computing b̂_opt using spectral density estimation. For variance-based statistics with n=9,000, the optimal block length formula yields **b ≈ n^(1/3) ≈ 21 observations**. The `arch` package implements this directly via `optimal_block_length()`.

### Stationary vs Circular Block Bootstrap

**Stationary Bootstrap is preferred over Circular Block Bootstrap** for your application. While Circular Block achieves lower asymptotic variance (the asymptotic relative efficiency of Circular/Stationary is bounded between 0.33–0.48), the Stationary Bootstrap with geometric block lengths is **less sensitive to block size misspecification**—critical when the true dependence structure is uncertain in financial data. The Moving Block Bootstrap should be avoided entirely as it systematically undersamples observations near series endpoints.

### Bootstrap Iterations

For bootstrap iterations, **B=500 is insufficient for reliable p-values**. Monte Carlo standard error at p=0.05 with B=500 is approximately ±2%, which can meaningfully affect decisions near significance thresholds. The minimum recommendations are:

| Purpose                                            | Minimum B |
| -------------------------------------------------- | --------- |
| Standard error estimates and confidence intervals  | B=1,000   |
| Publication-quality p-values at α=0.05             | B=2,000   |
| High-stakes hypothesis testing or α=0.01 decisions | B=5,000+  |

For bounded [0,1] data, neither bootstrap method requires modification—both handle bounded distributions naturally.

```python
from arch.bootstrap import StationaryBootstrap, optimal_block_length

opt = optimal_block_length(features[:, 0])
b = max(15, min(50, opt['stationary'].iloc[0]))  # Bounded 15-50
bs = StationaryBootstrap(b, features, target, seed=42)
results = bs.apply(your_statistic, 2000)
```

---

## 2. Parametric Bootstrap KS Test: Computational Efficiency

**No closed-form Lilliefors-type corrections exist for Beta distributions**—the null distribution depends on estimated parameters in ways that preclude pre-computed tables. However, `scipy.stats.goodness_of_fit` implements efficient parametric bootstrap by fixing location and scale parameters for Beta fitting on [0,1] data, reducing MLE to only two parameters (α, β).

### Minimum Bootstrap Iterations (Davidson & MacKinnon 2000)

The critical rule is choosing B such that **α(B+1) is an integer** for proper p-value calculation:

| Purpose                  | Minimum B | Notes                         |
| ------------------------ | --------- | ----------------------------- |
| Screening at α=0.05      | **99**    | ~3% power loss acceptable     |
| Final decision at α=0.05 | **499**   | Exceeds recommended B=399     |
| Final decision at α=0.01 | **1,499** | Required for α=0.01 precision |

### Two-Stage Testing

**Two-stage testing is valid when properly designed.** Run Stage 1 with B=99 at α=0.10, classifying features as "definite pass" (p>0.10), "definite fail" (p<0.001), or "borderline." Stage 2 applies B=999 only to borderline cases. This typically achieves **~80% computational savings** while avoiding selection bias by using lenient initial thresholds.

### Anderson-Darling vs KS

**Anderson-Darling dominates KS for Beta goodness-of-fit**—AD's weighting function 1/[F(x)(1-F(x))] provides high sensitivity at distribution tails, exactly where Beta distributions with [0,1] bounded data show mass accumulation. Empirical power comparisons consistently show AD > Cramér-von Mises > KS for detecting shape, scale, and symmetry differences.

```python
from scipy.stats import goodness_of_fit, beta
from joblib import Parallel, delayed

def test_beta_gof(data, n_mc=999):
    return goodness_of_fit(beta, data, statistic='ad',
                           known_params={'loc': 0, 'scale': 1},
                           n_mc_samples=n_mc, rng=42)

# Parallel execution for 576 features
results = Parallel(n_jobs=-1)(
    delayed(test_beta_gof)(X[:, j], n_mc=999) for j in range(576))
```

---

## 3. Cliff's Delta and Rank-Biserial: Mathematical Equivalence

**Cliff's Delta (δ) and rank-biserial correlation (r_rb) are identical statistics for two independent samples.** The formula δ = 2r_rb - 1 in the original question is **incorrect**—they are not different statistics requiring conversion. Both equal the dominance statistic:

```
δ = r_rb = [#(Xᵢ > Yⱼ) - #(Xᵢ < Yⱼ)] / (n₁ × n₂)
```

The relationship to Vargha-Delaney A is: **δ = 2×VDA - 1**, where VDA = U/(n₁×n₂) from the Mann-Whitney U statistic. This explains the confusion—the "2x - 1" transformation converts between VDA (range [0,1]) and Cliff's Delta (range [-1,1]).

### Effect Size Thresholds

**Report Cliff's Delta as primary effect size** for regime dependence analysis. The probabilistic interpretation is direct: δ=0.35 means a net 35% excess in pairwise comparisons where Regime A exceeds Regime B.

For thresholds, **Romano et al. (2006) values are appropriate** as defaults:

|            | δ          | threshold | Interpretation |
| ---------- | ---------- | --------- | -------------- |
| <0.147     | Negligible |
| 0.147–0.33 | Small      |
| 0.33–0.474 | Medium     |
| ≥0.474     | Large      |

**However, with n=9,000, statistical significance is trivially achieved.** Consider domain-specific practical significance thresholds: δ<0.10 as negligible regardless of p-value, and δ≥0.20 as the minimum for "meaningfully different regimes" in financial applications.

---

## 4. VIF Regularization for 576 Features

Standard VIF becomes unstable when the correlation matrix is ill-conditioned. **Ridge-based VIF** computes:

```
VIF_ridge = diag(R_δ⁻¹)

where R_δ = (R + λI_p)/(1+λ) is the regularized correlation matrix
```

This is computationally efficient—**O(p³) matrix inversion versus O(p⁴) for p separate regressions**.

### Lambda Selection

For λ selection, **target a condition number of 100–200** using:

```
λ = max(0, (λ_max - c × λ_min) / (c - 1))
```

where c is the target condition number and λ_max, λ_min are the extreme eigenvalues. For 576 features with n/p ≈ 15.6, you likely have adequate samples for stable estimation, but regularization still improves numerical stability.

### Pre-filtering

**Pre-filtering at |r|>0.95 is recommended** before VIF computation. Features with near-perfect correlation will have extreme VIF regardless and can be removed deterministically. This reduces the correlation matrix from 576×576 to perhaps ~500×500, saving ~25% computation while introducing minimal bias.

### VIF Thresholds

| Threshold  | Use case                                          |
| ---------- | ------------------------------------------------- |
| VIF<5      | Regression diagnostics, interpretability required |
| **VIF<10** | **ML feature selection (recommended)**            |
| VIF<20     | Tree-based models with inherent robustness        |

```python
def compute_vif_regularized(X, target_cond=100):
    R = np.corrcoef(X, rowvar=False)
    eig = np.linalg.eigvalsh(R)
    lam = max(0, (eig[-1] - target_cond*eig[0])/(target_cond - 1))
    R_reg = (R + lam*np.eye(R.shape[0]))/(1 + lam)
    return np.diag(np.linalg.inv(R_reg)), lam
```

---

## 5. Shapiro-Wilk W Interpretation for Large Samples

At n=9,000, **the Shapiro-Wilk test is fundamentally problematic**—it becomes overpowered, detecting trivial departures from normality as highly significant. Most implementations (R, SciPy) limit n to 5,000 with accuracy warnings.

### Use W as Continuous Index

**Use W as a continuous normality index rather than conducting hypothesis tests.** Interpretation thresholds:

| W value   | Interpretation                                      |
| --------- | --------------------------------------------------- |
| >0.99     | Practically indistinguishable from normal           |
| 0.95–0.99 | Minor departures, acceptable for parametric methods |
| 0.90–0.95 | Moderate non-normality, visual inspection warranted |
| <0.90     | Substantial non-normality                           |

Report W directly without transformation—1-W or log(1-W) add complexity without interpretive benefit.

### Beta-likeness Alternative

For Beta-likeness on bounded [0,1] data, **Jensen-Shannon divergence** provides a symmetric, bounded [0,1] measure: JS<0.05 indicates "sufficiently Beta-like." Report the fitted Beta concentration parameter (α+β) as an interpretable summary—higher values indicate more peaked distributions.

---

## 6. Practical Guidance: When Block Bootstrap is Worth It

**Block bootstrap is worth the ~3× computational overhead when lag-1 autocorrelation |ρ₁| > 0.2.** At lower autocorrelation, i.i.d. bootstrap typically produces adequate inference. The impact of ignoring autocorrelation manifests as underestimated standard errors and inflated Type I error rates (nominal 5% → actual 10–15%).

### CV-Based Stability Metrics

For applied feature selection pipelines, **CV-based stability metrics often outperform formal hypothesis testing**:

- **Nogueira stability measure** provides confidence intervals and satisfies all desirable theoretical properties (correction for chance, monotonicity, proper bounds)
- Benchmark: Φ>0.75 indicates excellent stability, 0.40–0.75 is acceptable
- Run M=50–100 bootstrap/CV iterations for stable estimates

**Stability selection** (Meinshausen & Bühlmann 2010) offers automatic FDR control by selecting features appearing in >60–90% of subsamples, avoiding the multiple testing correction problem entirely.

### Context-Based Recommendations

| Context                    | Recommendation                                    |
| -------------------------- | ------------------------------------------------- |
| Academic publication       | Formal hypothesis tests with BH-FDR correction    |
| **Production ML pipeline** | **Stability metrics + CV performance validation** |
| Regulatory/compliance      | Formal tests, conservative thresholds             |

When the ultimate validation is predictive performance on a holdout set, the burden of proof shifts from "is this feature statistically significant?" to "does this feature set improve prediction?"—a question CV answers directly.

---

## 7. Implementation Summary

For your 9,000×576 bounded [0,1] financial feature selection pipeline:

```python
# Block Bootstrap for Friedman-type tests
from arch.bootstrap import StationaryBootstrap, optimal_block_length
b = optimal_block_length(data)['stationary'].iloc[0]
bs = StationaryBootstrap(max(15, min(50, b)), data, seed=42)

# Beta GoF testing (two-stage)
from scipy.stats import goodness_of_fit, beta
p_screen = [goodness_of_fit(beta, X[:,j], statistic='ad',
            known_params={'loc':0,'scale':1}, n_mc_samples=99).pvalue
            for j in range(576)]
borderline = (0.001 < p_screen) & (p_screen < 0.10)
# Run B=999 only on borderline features

# Regularized VIF
R = np.corrcoef(X_filtered, rowvar=False)  # After |r|>0.95 filter
eig = np.linalg.eigvalsh(R)
lam = max(0, (eig[-1] - 100*eig[0])/99)  # Target cond=100
VIF = np.diag(np.linalg.inv((R + lam*np.eye(len(R)))/(1+lam)))

# Effect size (Cliff's Delta = rank-biserial)
from scipy.stats import mannwhitneyu
U, _ = mannwhitneyu(regime_a, regime_b)
delta = 1 - 2*U/(len(regime_a)*len(regime_b))
```

The overarching principle: **computational efficiency trades favorably against formal statistical rigor in production ML** when cross-validation provides empirical validation, stability metrics quantify reproducibility, and iteration speed enables rapid experimentation.
