---
source_url: https://claude.ai/public/artifacts/a49965f8-bca5-46cb-b791-50abd0492102
source_type: claude-artifact
scraped_at: 2026-02-02T09:03:46-08:00
purpose: Research parameter-free feature selection methods for high-autocorrelation financial time series with OOD robustness
tags:
  [
    feature-selection,
    time-series,
    ood-robustness,
    causal-discovery,
    autocorrelation,
  ]

# REQUIRED provenance
model_name: Claude (Artifact)
model_version: "3.5-sonnet"
tools: []

# REQUIRED for Claude Code backtracking + context
claude_code_uuid: 14ac6476-077b-4c82-a3fe-217dae94cff6
claude_code_project_path: "~/.claude/projects/-Users-terryli-eon-trading-fitness/14ac6476-077b-4c82-a3fe-217dae94cff6"

# REQUIRED backlink metadata (filled after ensuring issue exists)
github_issue_url: https://github.com/terrylica/cc-skills/issues/21
---

# Parameter-free feature selection for high-autocorrelation financial time series

Standard feature selection methods fail catastrophically when applied to your 160 bounded [0,1] cryptocurrency features with lag-1 ACF of 0.94. The extreme temporal dependence violates i.i.d. assumptions underlying LASSO, Random Forest importance, and mutual information—causing inflated significance, importance spreading across correlated features, and unreliable cross-validation. **The solution lies in three methodological shifts**: using block bootstrap variants that preserve temporal structure, applying causal discovery methods (PCMCI) that explicitly model autocorrelation, and validating feature stability across market regimes rather than single-environment performance.

The recommended implementation pipeline begins with **mRMR** for fast initial filtering (24→15 features), followed by **PCMCI** from tigramite for causal filtering with proper autocorrelation handling, then **walk-forward importance stability** to verify OOD robustness. For your lb20/lb100/lb500 lookback question, the **scattering transform** (kymatio) sidesteps selection entirely by extracting multi-scale invariant features automatically.

## High autocorrelation fundamentally breaks standard methods

With lag-1 ACF = 0.94, your cryptocurrency features exhibit near-unit-root behavior that invalidates most feature selection approaches. **LASSO** suffers from underestimated standard errors and inconsistent variable selection for persistent regressors—theoretical guarantees in Lee, Shi & Gao (arXiv:1810.03140) show selection inconsistency for cointegrated/near-I(1) data. **Random Forest importance** spreads importance across correlated features rather than concentrating it, with unconditional permutation importance overestimating correlated predictors (Strobl et al., 2008). **Mutual information** estimates become positively biased because the effective sample size is much smaller than actual N.

The critical insight from econometrics is that the Two-stage Adaptive LASSO (TAlasso) achieves variable selection consistency for mixed-roots problems by first breaking cointegration ties, then applying adaptive penalties. However, no direct Python implementation exists—custom implementation is required using sklearn's LassoCV with TimeSeriesSplit followed by weighted second-stage estimation.

**Block bootstrap** provides the most direct solution for preserving temporal structure during feature importance estimation. The `tsbootstrap` package (pip install tsbootstrap) implements Moving Block Bootstrap, Stationary Bootstrap, and Circular Block Bootstrap with sklearn compatibility. Critically, the Politis & White (2004) algorithm for **automatic optimal block length selection** is implemented in the `recombinator` package—making block size a data-driven parameter rather than arbitrary choice. For ACF = 0.94, expect optimal block sizes of **20-50 observations**.

MethodACF-RobustParametersPackageImplementationBlock Bootstrap + LASSO✅ Highblock_length (data-driven)tsbootstrapMedium-HighStability Selection❌ i.i.d. onlythreshold, lambda_gridstability-selectionMediumGranger Causality⚠️ Requires stationaritymaxlag (AIC/BIC)statsmodelsLowTransfer Entropy✅ Highhistory k (auto-embed)JIDT/PyInformHighmRMR⚠️ ModerateK (your choice)mrmr-selectionLow

## OOD robustness requires environment-based selection methods

Three families of methods provide theoretical or empirical guarantees for out-of-distribution robustness: **Invariant Causal Prediction (ICP)**, **Anchor Regression**, and **walk-forward importance stability**.

**ICP** (Peters, Bühlmann, Meinshausen 2016) identifies causal parents of the target by testing which feature subsets yield invariant conditional distributions across environments. The Python implementation `causalicp` (pip install causalicp) provides a clean interface. For financial data, environments should be defined as volatility regimes (quartiles of realized volatility), HMM-detected market states, or time periods around structural breaks. **Critical caveat**: ICP assumes same error variance across environments—violated by volatility clustering—and often returns empty sets for financial data due to its conservative nature.

**seqICP** (Pfister, Bühlmann, Peters 2019) extends ICP to sequential data where environments are not known a priori, automatically inferring environment structure from temporal heterogeneity. The R package `seqICP` includes an autoregressive model option making it more suitable than standard ICP, though research indicates it often returns empty sets for financial time series.

**Anchor Regression** (Rothenhäusler et al., 2021) provides the most practical OOD approach. It uses exogenous anchor variables to provide distributional robustness guarantees, interpolating between OLS (γ=0) and instrumental variable estimation (γ→∞) via a single hyperparameter:

`b_γ = argmin_b ||Y - Xb||² + γ·||P_A(Y - Xb)||²`
For cryptocurrency data, anchors should be regime indicators (bull/bear categorical), volatility regime labels from HMM, or time period dummies. The implementation is a straightforward closed-form modification of OLS—**the lowest implementation complexity of any OOD method**.

**Walk-forward importance stability** offers the most practical empirical validation: compute feature importance in each walk-forward optimization fold, track variance across folds, and select features with consistently high importance and low coefficient of variation. The `tscv` package (pip install tscv) provides GapRollForward for proper time series CV with gaps preventing leakage.

## Nonlinear redundancy detection goes beyond correlation

Your current correlation filtering (|r| > 0.95) misses nonlinear dependencies. Three parameter-free or near-parameter-free methods detect all types of redundancy:

**Distance Correlation** (Székely et al., 2007) is completely parameter-free and equals zero if and only if variables are independent—detecting all dependency types. The `dcor` package (pip install dcor) provides O(n²) computation with bias-corrected estimators. For 160 features, the ~12,720 pairwise comparisons are tractable. Create a 160×160 distance correlation matrix and threshold at dCor > 0.6-0.7 to identify redundant pairs.

**HSIC-Lasso** (Yamada et al., 2014) performs nonlinear feature selection while minimizing redundancy, equivalent to convex mRMR with kernel embeddings. The `pyHSICLasso` package uses Block HSIC Lasso for memory efficiency—O(dnBM) complexity with default B=20, M=3. For your 160 bounded features, this is highly tractable:

python`from pyHSICLasso import HSICLasso
hsic = HSICLasso()
hsic.input(X, y)
hsic.regression(10)  # Select 10 features`
**Knockoff Filters** (Barber & Candès, 2015) provide FDR-controlled feature selection through creating "knockoff" variables as negative controls. The `knockpy` package (pip install knockpy) implements Model-X knockoffs. **Critical limitation**: FDR guarantees do NOT hold under temporal dependence. The recent **DeepLINK-T** (arXiv:2404.15227, 2024) extends knockoffs for time series using LSTM autoencoders—the only theoretically valid knockoff method for your autocorrelated data.

MethodDetectsParameter-FreePackageComplexityDistance CorrelationAll nonlinear✅ YesdcorO(n²) per pairHSIC-LassoAll nonlinear⚠️ Block paramspyHSICLassoO(dnBM)KnockoffsLinear + FDR❌ ManyknockpyO(p²) + modelMICAll functional⚠️ α, cminepyO(n^2.4)mRMRMI-based⚠️ K selectionmrmr-selectionO(mkn)

## Automatic scale selection sidesteps the lookback problem

Your lb20/lb100/lb500 variants lack empirical justification. Four approaches provide data-driven solutions:

The **Scattering Transform** (Mallat, via kymatio) offers the most elegant solution—instead of choosing one lookback, it automatically extracts features at ALL relevant scales through cascaded wavelet transforms with modulus nonlinearity. Set J=9 to capture patterns up to 512 bars, let the downstream BiLSTM learn which scales matter:

python`from kymatio.numpy import Scattering1D
scattering = Scattering1D(J=9, shape=1024, Q=8)
features = scattering(price_series)  # Multi-scale invariant features`
**Wavelet variance analysis** via MODWT identifies which temporal scales carry the most signal variance. Using PyWavelets, compute wavelet variance by level—scales where variance is highest relative to noise indicate optimal lookback periods. The scale-to-period mapping: level 4-5 corresponds to ~lb20, level 7 to ~lb100, level 9 to ~lb500.

**Mutual information lag selection** provides direct comparison: compute MI(feature_lb20, target), MI(feature_lb100, target), MI(feature_lb500, target), and select the lookback with highest MI. For nonlinear dependencies, use sklearn's mutual_info_regression with k-NN estimators.

**VMD (Variational Mode Decomposition)** offers better separation than EMD for financial data—18% improvement in directional forecast accuracy versus EMD for S&P 500. The `vmdpy` package (also in sktime) decomposes signals into adaptive modes; correlation between each mode and target identifies relevant timescales.

## PCMCI provides the gold standard for causal feature importance

**PCMCI** (Runge et al., Science Advances 2019) is the only method explicitly designed for causal discovery in autocorrelated time series. The MCI (Momentary Conditional Independence) test explicitly controls for autocorrelation by conditioning on lagged variables, while the PC1 condition selection removes spurious links driven by temporal persistence.

The `tigramite` package (pip install tigramite, 1.6k GitHub stars, actively maintained v5.2) provides complete implementation:

python`from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr

dataframe = pp.DataFrame(data)
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())
results = pcmci.run_pcmciplus(tau_max=5, pc_alpha=0.05)

# Extract significant parents as important features`

For 160 features, **computational cost is the main limitation**—O(n²·p·τ_max·k) complexity means several hours for 10,000+ samples. Practical approach: run mRMR first to reduce to top-50 features, then apply PCMCI. Use `tau_max=5-10` based on domain knowledge (for ACF=0.94, longer lags carry little additional information), and enable parallelization via `n_jobs=-1`.

**Block permutation importance** provides a faster alternative that preserves temporal structure. Replace sklearn's standard permutation with block-wise shuffling:

python`def block_shuffle(x, block_size=20):
    n_blocks = len(x) // block_size
    blocks = np.array_split(x, n_blocks)
    np.random.shuffle(blocks)
    return np.concatenate(blocks)`
For ACF=0.94, set block_size=20-50 based on the optimal block length from Politis & White algorithm in `recombinator`.

## Validating OOD robustness requires multi-regime testing

The key insight for validating OOD-robust feature selection is that **importance must remain stable across different market regimes**. Implement a four-stage validation protocol:

**Stage 1: Regime detection**. Use Hidden Markov Models (hmmlearn) with 2-3 states to identify distinct market regimes. Alternative: segment by realized volatility quartiles or by rolling correlation structure changes (detected via change point algorithms).

**Stage 2: Per-regime importance**. Compute feature importance separately within each regime using block permutation importance. Features with high importance in ALL regimes are OOD-robust candidates.

**Stage 3: Walk-forward stability**. Using `tscv.GapRollForward`, compute importance in each fold and track coefficient of variation:

python`stability = np.std(importance_per_fold, axis=0) / np.mean(importance_per_fold, axis=0)
stable_features = np.where(stability < threshold)[0]`
**Stage 4: DoWhy refutation**. For the final feature set, use DoWhy's refutation API to test sensitivity to unmeasured confounding (`add_random_common_cause`), placebo treatments (`placebo_treatment_refuter`), and data subsets (`data_subset_refuter`).

## Recommended implementation pipeline

Given your constraints—160 bounded [0,1] features, ACF=0.94, BiLSTM SelectiveNet, need for parameter-free methods—follow this prioritized pipeline:

**Phase 1: Fast initial filtering (24→15 features)**

python`from mrmr import mrmr_regression
selected = mrmr_regression(X=df_features, y=df_target, K=15)`
mRMR is fast, has only one parameter (K), and captures redundancy. Limitation: assumes linear relationships for redundancy computation.

**Phase 2: Nonlinear redundancy check**

python`import dcor
redundancy_matrix = np.array([[dcor.distance_correlation(X[:,i], X[:,j])
                               for j in range(X.shape[1])]
                              for i in range(X.shape[1])])`
Remove features with dCor > 0.7 pairwise, keeping the one with higher target relevance.

**Phase 3: Causal filtering via PCMCI**

python`from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr

# Run on Phase 2 output (~15 features)

pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())
results = pcmci.run_pcmciplus(tau_max=5, pc_alpha=0.05)
causal_features = extract_significant_parents(results)`
**Phase 4: Walk-forward stability validation**

python`from tscv import GapRollForward
cv = GapRollForward(min_train_size=500, gap_size=20, max_test_size=50)

# Compute importance in each fold, select features with CV < 0.5`

**Phase 5: Scale selection for lookbacks**
Use scattering transform to automatically extract multi-scale features, or compute mutual information for each lb variant to select empirically.

## Specific guidance for ACF = 0.94

Your lag-1 ACF of 0.94 indicates near-unit-root behavior requiring specific handling:

**Never use standard k-fold CV**—always use TimeSeriesSplit or walk-forward validation
**Block bootstrap block sizes**: Use `recombinator.optimal_block_length()` for data-driven selection; expect 20-50 observations
**PCMCI tau_max**: Set to 5-10; with ACF=0.94, information decays slowly but most causal relationships manifest within few lags
**Effective sample size**: Your actual independent information is approximately N × (1-ACF)/(1+ACF) ≈ N × 0.03—a 10,000 sample dataset contains roughly 300 effective samples
**Granger causality**: Requires stationarity; apply ADF tests and difference if needed, though your bounded [0,1] features cannot have true unit roots—likely fractionally integrated

The combination of block bootstrap preserving temporal structure, PCMCI's MCI test controlling for autocorrelation, and walk-forward stability validation addresses the unique challenges of your high-ACF cryptocurrency data while maintaining the parameter-free philosophy you require.

## Critical implementation packages

PackageInstallPurposetigramite`pip install tigramite`PCMCI causal discoverymrmr-selection`pip install mrmr-selection`Fast redundancy-aware filteringdcor`pip install dcor`Distance correlation (parameter-free)pyHSICLasso`pip install pyHSICLasso`Nonlinear feature selectiontsbootstrap`pip install tsbootstrap`Block bootstrap for time seriesrecombinator`pip install recombinator`Optimal block length selectiontscv`pip install tscv`Walk-forward CV with gapskymatio`pip install kymatio`Scattering transform multi-scalecausalicp`pip install causalicp`Invariant causal predictionknockpy`pip install knockpy`FDR-controlled selection

## Conclusion

The path to principled, OOD-robust feature selection for your cryptocurrency data requires abandoning standard methods that assume i.i.d. observations. **PCMCI from tigramite** provides the theoretical gold standard—the only method explicitly designed for causal discovery under high autocorrelation—but computational cost limits it to pre-filtered feature sets. **mRMR + distance correlation** offers the most practical initial filtering with minimal hyperparameters. **Walk-forward importance stability** provides the empirical validation that theoretical methods cannot guarantee in non-stationary financial markets.

The scattering transform elegantly sidesteps your lb20/lb100/lb500 question by extracting invariant features at all scales simultaneously—letting your BiLSTM learn which scales matter rather than requiring a priori specification. Combined with anchor regression using volatility regime labels as anchors (single hyperparameter γ), this pipeline achieves principled OOD robustness without magic numbers.
