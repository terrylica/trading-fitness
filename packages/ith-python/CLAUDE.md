# ith-python

Primary Python package for ITH (Investment Time Horizon) fitness analysis.

## Quick Start

```bash
uv sync                           # Install dependencies
uv run python -m ith_python.ith   # Run analysis
```

## Module Structure

| Module                   | Purpose                             |
| ------------------------ | ----------------------------------- |
| `ith.py`                 | Main ITH analysis script            |
| `ith_numba.py`           | Numba-accelerated calculations      |
| `paths.py`               | Repository-local path configuration |
| `templates/results.html` | HTML template for results           |

## Key Concepts

- **TMAEG**: Target Maximum Acceptable Excess Gain (drawdown threshold)
- **ITH Epochs**: Periods where strategy exceeds TMAEG threshold
- **Fitness Criteria**: Epoch count, Sharpe ratio, coefficient of variation

## Dependencies

Core: pandas, numpy, plotly, scipy, numba, loguru, rich
Dev: pytest, ruff
