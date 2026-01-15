# ITH (Investment Time Horizon) Analysis

## Concept

ITH measures trading strategy fitness by analyzing how frequently a strategy exceeds drawdown-based performance thresholds.

## Key Metrics

### TMAEG (Target Maximum Acceptable Excess Gain)

The drawdown threshold that defines an ITH epoch boundary. Dynamically determined by maximum drawdown (MDD).

### ITH Epochs

Time periods where the strategy's cumulative returns exceed the TMAEG threshold. More epochs indicate better fitness.

### Fitness Criteria

| Metric                   | Description                                                           |
| ------------------------ | --------------------------------------------------------------------- |
| ITH Epoch Count          | Number of qualifying epochs (typically 8-11 for qualified strategies) |
| Sharpe Ratio             | Risk-adjusted return (typically > 0.5)                                |
| Coefficient of Variation | Return consistency measure                                            |

## Usage

Place CSV files with Date and NAV columns in `data/nav_data_custom/`, then run:

```bash
mise run analyze
```

Results appear in `artifacts/results.html`.
