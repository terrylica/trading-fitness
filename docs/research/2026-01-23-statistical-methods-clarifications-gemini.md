# Implementation of Robust Statistical Methods for Financial Feature Selection

> **Source**: Gemini 3 Pro Deep Research
> **Date**: 2026-01-23
> **URL**: <https://gemini.google.com/share/b093ed16545e>
> **Request**: Clarifications on block bootstrap parameters, computational efficiency, effect sizes, and VIF regularization

---

## 1. Executive Summary and Architectural Overview

This report provides an exhaustive technical specification for implementing advanced statistical corrections within a high-dimensional financial feature selection pipeline. The analysis is predicated on a dataset characterized by N≈9000 temporal observations and P≈576 candidate features—a dimensionality profile that presents unique challenges regarding serial dependence, multicollinearity, and computational tractability.

The central finding of this investigation is that standard statistical protocols—specifically the Friedman test for rank comparisons and uncorrected Goodness-of-Fit (GoF) tests—are theoretically invalid for financial time series due to the violation of the Independence and Identically Distributed (I.I.D.) assumption. The serial correlation inherent in asset returns (volatility clustering, momentum effects) inflates Type I error rates in hypothesis testing, necessitating the adoption of **Block Bootstrap** methodologies. Furthermore, the standard Variance Inflation Factor (VIF) metric for detecting multicollinearity becomes numerically unstable in high-dimensional regimes (N/P<20), requiring **Ridge Regularization** with corrected variance formulas.

### Core Implementation Recommendations

1. **Dependence Correction via Circular Block Bootstrap (CBB):**
   - **Mechanism:** Replace standard resampling with a Circular Block Bootstrap to preserve the temporal dependence structure while eliminating boundary bias inherent in standard block methods.
   - **Parameterization:** Abandon fixed block lengths. Implement the **Politis-White (2004)** spectral density method (corrected by Patton et al., 2009) to adaptively select block sizes based on the specific autocorrelation decay of each feature.

2. **Computational Optimization via Analytical Approximations:**
   - **Constraint:** A full parametric bootstrap for 576 features involving 1,000 iterations implies ~5.76×10⁵ distribution fits, creating a prohibitive bottleneck.
   - **Solution:** Adopt the **Zhang & Wu (2002)** Beta approximation for the Kolmogorov-Smirnov (KS) test. This analytical correction adjusts the null distribution of the KS statistic for estimated parameters without requiring Monte Carlo simulation, reducing computational complexity from O(B·N) to O(1) per feature.

3. **Effect Size Standardization:**
   - **Metric:** Standardize on **Cliff's Delta (δ)** rather than Rank-Biserial Correlation. While mathematically equivalent, Cliff's Delta offers a superior interpretive framework ("dominance probability") for regime-based trading analysis.
   - **Thresholds:** Reject Cohen's behavioral science thresholds (0.2/0.5/0.8). Adopt finance-specific thresholds where |δ|>0.05 indicates a tradable edge and |δ|>0.15 represents a strong signal.

4. **Regularized Collinearity Diagnostics:**
   - **Methodology:** Implement **Ridge VIF** using the matrix-diagonal formulation derived by Garcia et al. (2015). Standard VIF formulas (1/(1−R²)) fail under regularization.
   - **Tuning:** Select the regularization parameter λ (or k) using **Generalized Cross-Validation (GCV)** to balance bias induction with matrix invertibility.

5. **Pipeline Architecture:**
   - **Decision:** Remove the Friedman test from the feature filtering phase entirely. Its computational cost (via bootstrap) outweighs its utility for ranking.
   - **Replacement:** Utilize a **Coefficient of Variation (CV)** stability metric on Cliff's Delta across rolling windows for initial selection, reserving the rigorous Block Bootstrap only for the final validation of the ensemble model.

---

## 2. Block Bootstrap for Time Series: Implementation Mechanics

The primary statistical violation in financial feature selection is the assumption of independence. Financial time series exhibit "memory"—the value of a feature at time t is predictive of its volatility or magnitude at time t+1. Standard rank tests (like Friedman) treat observations as exchangeable, which destroys this structure and leads to vastly underestimated p-values (false positives). The Block Bootstrap restores validity by resampling contiguous blocks of data, thereby preserving the short-range dependence structure.

### 2.1 Block Size Selection: The Politis-White Adaptive Method

The choice of block size b is the single most consequential parameter in the bootstrap design. It governs the bias-variance trade-off:

- **Bias (Block size too small):** If b→1, the procedure reverts to an I.I.D. bootstrap, destroying the dependence structure we aim to preserve.
- **Variance (Block size too large):** If b→N, we have only one block (the original series). The number of independent samples available for the bootstrap is roughly N/b.

#### Why Fixed Block Sizes Fail

A fixed block size (e.g., b=50) is insufficient because the "memory" of financial features is heterogeneous:

- **Return-based features** (e.g., RSI, Momentum) often have short memory (autocorrelation decays within 5-10 days).
- **Volatility-based features** (e.g., ATR, Bollinger Band Width) exhibit long memory (autocorrelation persists for 100+ days).

#### Algorithm: Politis-White (2004) with Patton Correction

**Detailed Implementation Steps:**

1. **Input:** A time series vector X={x₁,…,xₙ} (e.g., a candidate feature).

2. **Step 1: Estimate Autocorrelations (ρ̂ₖ):** Compute the sample autocorrelation function for a large number of lags (e.g., up to K=⌈log₁₀(N)²⌉).

3. **Step 2: Determine Bandwidth (m):** The algorithm identifies a "flat-top" lag window. It searches for the smallest integer m such that the autocorrelations for lags k>m are negligible:

   ```
   |ρ̂ₖ| < c√(log₁₀N / N)
   ```

   where c≈2 corresponds to a 95% confidence bound. The bandwidth M is then set as 2m.

4. **Step 3: Estimate Spectral Quantities:** The optimal block size depends on the spectral density at frequency zero (g(0)) and the generalized second derivative of the spectral density.

5. **Step 4: Calculate b_opt:** The analytical solution for the optimal block length for the Circular Block Bootstrap is:

   ```
   b_opt^CB = (D_CB / 2G²)^(1/3) × N^(1/3)
   ```

**Implementation Recommendation:** Use the `optimal_block_length` function from the Python `arch` package (`arch.bootstrap`), which implements the Patton (2009) correction.

### 2.2 Resampling Scheme: Circular vs. Stationary vs. Moving Blocks

#### Stationary Bootstrap (SB)

- **Mechanism:** Uses random block lengths distributed geometrically with mean 1/p.
- **Drawback:** The randomization introduces a second source of stochasticity into the bootstrap estimator.

#### Moving Block Bootstrap (MBB)

- **Mechanism:** Uses fixed-length blocks b. Blocks are overlapping.
- **Drawback:** Suffers from the "End Effect." Observations at the beginning and end of the series are sampled fewer times than observations in the middle.

#### Circular Block Bootstrap (CBB) - **Recommended**

- **Mechanism:** The CBB wraps the data around a circle, such that xₙ₊₁=x₁, xₙ₊₂=x₂, etc.
- **Advantage:** This eliminates the "End Effect." Every observation xᵢ has an equal probability (b/N) of appearing in a resampled block. This property is crucial for **bounded [0,1] data** where values near the boundaries might be clustered at specific time periods.

**Conclusion:** Use the **Circular Block Bootstrap (CBB)** with overlapping blocks.

### 2.3 Iteration Count (B)

The number of bootstrap replications B dictates the precision of the p-value estimate:

- **For Feature Screening (p≈0.05):** **B=500** is sufficient for the initial filtering pass.
- **For Final Validation (p≈0.01):** Use **B=2000** for reduced standard error (~0.2%).

---

## 3. Computational Efficiency in Goodness-of-Fit (KS Test)

### 3.1 The Analytical Solution: Zhang & Wu (2002)

**The Statistical Problem:** When parameters (α,β) are estimated from the data (e.g., via MLE), the empirical distribution function fits the data "too well." Using standard critical values results in a test that is **conservative** (very low power).

**The Analytical Correction:** Zhang and Wu (2002) derived an analytical approximation for the null distribution of Dₙ when parameters are estimated:

```
Dₙ ~ a·Beta(p,q) + b
```

The parameters a,b,p,q are functions of the sample size N and the estimated shape parameters of the data.

**Implementation Strategy:**

1. Fit the Beta distribution to the feature data X to get θ̂=(α̂,β̂).
2. Compute the observed KS statistic D_obs between X and Beta(θ̂).
3. **Do not bootstrap.** Instead, calculate the p-value using the Zhang-Wu approximation formula.

### 3.2 Anderson-Darling vs. KS for Financial Data

The **Anderson-Darling (AD)** test is statistically superior for financial data:

- **Sensitivity:** KS focuses on the maximum vertical distance between CDFs, which typically occurs near the median. It is insensitive to tail deviations.
- **Relevance:** In finance, risk and alpha are concentrated in the tails. The AD test applies a weight function that emphasizes the tails.
- **Recommendation:** **Replace KS with Anderson-Darling**.

### 3.3 Two-Stage Filtering (The "Fast Fail" Heuristic)

1. **Stage 1 (Standard KS):** Run the standard `scipy.stats.kstest`.
   - If p-value <0.01 under the standard test, reject immediately.
   - If p-value >0.05, it _might_ pass or fail the corrected test.

2. **Stage 2 (Bootstrap):** Run the parametric bootstrap only on the survivors of Stage 1. This typically reduces the workload by >80%.

---

## 4. Effect Size Measures: Cliff's Delta

### 4.1 Mathematical Equivalence

Cliff's Delta (δ) and the Rank-Biserial Correlation (r_rb) are mathematically equivalent in the two-sample case:

```
δ = [#(x₁ > x₂) - #(x₁ < x₂)] / (n₁ × n₂)
```

Since U effectively counts the number of times observations from one group precede the other:

```
δ = -(r_rb)
```

(The sign depends on which group is defined as "group 1", but the absolute magnitude is identical).

**Recommendation:** Report **Cliff's Delta**. Its interpretation—"Net Dominance"—is more intuitive for financial stakeholders.

### 4.2 Thresholds for Financial Returns

Standard thresholds from Cohen (1988) are: Small (0.2), Medium (0.5), Large (0.8). **These are wholly inappropriate for finance.** In efficient markets, a predictive feature with a correlation of 0.8 is effectively impossible.

**Proposed Financial Thresholds:**

|           | δ                                                   |     | Interpretation |
| --------- | --------------------------------------------------- | --- | -------------- |
| <0.05     | **Negligible** (Likely noise)                       |
| 0.05–0.15 | **Small / Tradable** (~IC of 0.05, a "good" factor) |
| 0.15–0.30 | **Medium / Strong** (High-quality signal)           |
| ≥0.30     | **Large / Suspicious** (Check for look-ahead bias)  |

---

## 5. VIF Regularization (Ridge VIF)

With N=9000 and P=576, the ratio N/P≈15 is low enough that the correlation matrix X'X will be ill-conditioned.

### 5.1 The "Garcia" Correction Formula

Using **Ridge Regression** stabilizes the inversion by adding a penalty parameter k (or λ) to the diagonal: (X'X+kI)⁻¹. However, simply calculating VIFs from Ridge coefficients using the standard formula VIFⱼ=1/(1−Rⱼ²) is **incorrect**.

**Correct Matrix Formula:**

```
VIF_ridge = diag((X'X + kI)⁻¹ X'X (X'X + kI)⁻¹)
```

### 5.2 Selecting the Regularization Parameter (k)

**Recommended Strategy:**

1. **Ridge Trace (Heuristic):** Compute the maximum VIF across all features for a grid of k values (e.g., 10⁻⁴, 10⁻³, …, 1.0).

2. **Selection Rule:** Plot max(VIF) vs. k. Select the smallest k where the maximum VIF drops below a safety threshold (e.g., 10 or 30).

3. **Typical Value:** For financial data with this dimensionality, a value of **k≈0.01 to 0.1** is often sufficient.

### 5.3 Pre-Filtering to Reduce Dimensionality

1. **Pairwise Correlation Filter (O(P²)):** Before VIF, calculate the Spearman correlation matrix. If any pair has ρ>0.95, drop the feature with the lower univariate stability.

2. **Ridge VIF (O(P³)):** Run the regularized VIF on the remaining subset to detect **multi-variable** collinearity.

---

## 6. Continuous Normality Metrics (Shapiro-Wilk)

### 6.1 Thresholds and Transformation

**Interpretation:**

| W         | Interpretation      |
| --------- | ------------------- |
| >0.99     | Effectively Normal  |
| 0.95–0.99 | Near-Normal         |
| <0.90     | Heavy-tailed/Skewed |

**Transformation:** The W scale is compressed near 1.0. To create a usable ranking feature, apply a log-distance transformation:

```
Score = -log₁₀(1 - W)
```

### 6.2 Beta-Specific Goodness-of-Fit Metric

For data strictly bounded in [0,1], testing for Normality (which assumes infinite support) is theoretically inconsistent.

**Recommendation:** Instead of W (Normality), compute the **KS distance D** between the feature and its fitted **Beta distribution**:

- **Metric:** 1 - D_KS(Beta)
- A score near 1.0 indicates the feature is "perfectly Beta-like."

---

## 7. Practical Decision: The Friedman Test

**Question:** Option A (Remove) vs. Option B (Implement Block Bootstrap).

**Recommendation: Option A - Remove Friedman Test.**

**Reasoning:**

1. **Misalignment of Goals:** The Friedman test answers the question: _"Is there a significant difference in ranking between classifiers/features?"_ In a feature selection pipeline, you generally care about _identifying_ the best features, not proving that the rankings are statistically different from random noise.

2. **Computational ROI:** Implementing a rigorous Block Bootstrap for the Friedman test is computationally expensive (B × Ranking calculation). The marginal value of this p-value for feature _filtering_ is low.

3. **Superior Alternative:** The **Coefficient of Variation (CV) of the Information Coefficient (IC)** across Rolling Windows.
   - Calculate the feature's IC (correlation with target) in 10 sequential windows.
   - Compute Stability = μ_IC / σ_IC.
   - This metric directly captures what you care about (performance consistency).

**Revised Pipeline:**

1. **Correlation Filter:** Remove ρ>0.95.
2. **Ridge VIF:** Remove VIF>10 (using k≈0.1).
3. **Ranking:** Sort by **Cliff's Delta** (Regime Separation).
4. **Stability:** Filter by Rolling IC Stability (Mean/Std>0.5).
5. **Validation:** Only use **Block Bootstrap** at the very end of the pipeline to validate the final model's performance, not for intermediate feature selection.

---

## 8. Summary of Implementation Constants

| Parameter           | Value / Method             | Rationale                                                   |
| ------------------- | -------------------------- | ----------------------------------------------------------- |
| **Block Size (b)**  | **Politis-White (2004)**   | Adaptive to feature-specific memory depth                   |
| **Bootstrap Type**  | **Circular (CBB)**         | Handles [0,1] bounds correctly; minimizes variance          |
| **Iterations (B)**  | **500 (Screening)**        | Sufficient for p<0.05; use 2000 for final validation        |
| **KS Test**         | **Zhang-Wu Approx**        | O(1) analytical correction; eliminates bootstrap bottleneck |
| **Effect Size**     | **Cliff's Delta**          | Thresholds: **>0.05 (Tradeable), >0.15 (Strong)**           |
| **VIF Formula**     | **Garcia et al. (Matrix)** | diag((X'X+kI)⁻¹X'X(X'X+kI)⁻¹); prevents VIF < 1             |
| **VIF Lambda**      | **Ridge Trace (k≈0.1)**    | Minimal bias required to stabilize inversion                |
| **Normality Score** | **−log(1−W)**              | Linearizes resolution near 1.0 for ranking                  |

This roadmap replaces theoretically invalid tests with robust, computationally optimized alternatives tailored specifically for the statistical properties of financial time series.
