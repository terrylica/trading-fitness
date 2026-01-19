# SR&ED Claim Evidence

> Scientific Research & Experimental Development tax credit documentation.

## Claim Period: 2026-Q1

### Project: ITH (Investment Time Horizon) Analysis

**Technological Uncertainty**:

- Standard fitness metrics (Sharpe, Sortino) don't capture epoch-based performance
- No existing solution for TMAEG (Target Maximum Acceptable Excess Gain) calculation
- Uncertainty in optimal JIT compilation strategy for numerical Python
- Cross-language type consistency for polyglot trading systems

**Technological Advancement**:

- Novel ITH epoch detection algorithm based on drawdown-adjusted thresholds
- Proof that Numba JIT matches native Rust for trading calculations (5.5ms vs 4.0ms)
- Cross-language type system via JSON Schema code generation
- Unified polyglot monorepo architecture for trading analysis

**Systematic Investigation**:

- Benchmark-driven development with controlled experiments
- Iterative refinement of TMAEG calculation methodology
- Comparative analysis across Python, Rust, TypeScript implementations
- Performance profiling with 1M data point datasets

---

## Commit Log (SR&ED Tagged)

Extract with:

```bash
git log --grep="SR&ED-CLAIM" --format="| %ad | %h | %s |" --date=short
```

| Date | Commit Hash | Type | Description |
| ---- | ----------- | ---- | ----------- |
| TBD  | -           | -    | -           |

---

## Time Allocation

| Activity                 | Hours | % of Total |
| ------------------------ | ----- | ---------- |
| Experimental Development | TBD   | -          |
| Applied Research         | TBD   | -          |
| Documentation & Analysis | TBD   | -          |

---

## Evidence Artifacts

| Artifact               | Location                 | Purpose                           |
| ---------------------- | ------------------------ | --------------------------------- |
| Performance benchmarks | `scripts/benchmark.py`   | Quantitative advancement proof    |
| ITH algorithm          | `packages/ith-python/`   | Core experimental development     |
| Cross-language types   | `packages/shared-types/` | Systematic investigation evidence |
| Test coverage          | `packages/*/tests/`      | Experimental validation           |

---

## Related Resources

- [CRA SR&ED Program](https://www.canada.ca/en/revenue-agency/services/scientific-research-experimental-development-tax-incentive-program.html)
- [SR&ED Eligibility](https://www.canada.ca/en/revenue-agency/services/scientific-research-experimental-development-tax-incentive-program/eligibility-work-sred-tax-incentives.html)
- [T4088 Claim Guide](https://www.canada.ca/en/revenue-agency/services/forms-publications/publications/t4088.html)
