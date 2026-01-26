# Logging Contract

> Structured NDJSON logging for forensic analysis, scientific reproducibility, and P&L attribution.

**Updated**: 2026-01-25 (Observability Telemetry Enhancement Plan - Phase 3)

## Overview

All packages in trading-fitness emit structured NDJSON logs to `logs/` directory. The logging infrastructure supports:

1. **Scientific Reproducibility** - Random seed capture, data fingerprinting, provenance chain
2. **Trading Domain Events** - Decision snapshots, feature values at decision time, ITH epoch events
3. **Batch Analysis Forensics** - Post-hoc debugging of ITH analysis and statistical examination

## Unified Schema

Each log line is a valid JSON object:

```json
{
  "ts": "2026-01-25T14:30:52.123456Z",
  "level": "INFO",
  "msg": "Human readable message",
  "component": "bull_ith",
  "package": "ith-python",
  "env": "development",
  "pid": 12345,
  "tid": 67890,
  "trace_id": "abc123def456",

  "provenance": {
    "session_id": "sess_20260125_120000",
    "git_sha": "0dc100c",
    "input_hash": "sha256:...",
    "random_seed": 42
  },

  "context": {}
}
```

## Required Fields

| Field       | Type     | Description                                                    |
| ----------- | -------- | -------------------------------------------------------------- |
| `ts`        | ISO 8601 | UTC timestamp with microseconds                                |
| `level`     | string   | DEBUG, INFO, WARNING, ERROR                                    |
| `msg`       | string   | Human-readable message                                         |
| `component` | string   | Source component (e.g., `bull_ith`, `statistical_examination`) |
| `package`   | string   | Source package name                                            |

## Optional Fields

| Field         | Type   | Description                        |
| ------------- | ------ | ---------------------------------- |
| `trace_id`    | string | Correlation ID for request tracing |
| `context`     | object | Additional structured data         |
| `provenance`  | object | Reproducibility context            |
| `error`       | object | Error details with stack trace     |
| `duration_ms` | number | Operation duration                 |

## Provenance Context

For scientific reproducibility, events can include provenance data:

```json
{
  "provenance": {
    "session_id": "sess_20260125_120000",
    "experiment_id": "exp_001",
    "git_sha": "0dc100c",
    "input_hash": "sha256:abc123...",
    "random_seed": 42,
    "config_hash": "sha256:def456..."
  }
}
```

## Event Types

### Data Lineage Events

| Event Type         | Purpose                | Key Fields                                           |
| ------------------ | ---------------------- | ---------------------------------------------------- |
| `data.load`        | Input fingerprinting   | `source_path`, `sha256_hash`, `row_count`, `schema`  |
| `algorithm.init`   | Reproducibility anchor | `algorithm_name`, `version`, `random_seed`, `config` |
| `algorithm.result` | Outcome capture        | `final_metrics`, `output_hash`                       |

### ITH Analysis Events

| Event Type       | Purpose            | Key Fields                                               |
| ---------------- | ------------------ | -------------------------------------------------------- |
| `epoch_detected` | P&L attribution    | `epoch_index`, `bar_index`, `excess_gain`, `excess_loss` |
| `ith_step`       | Step-by-step trace | `bar`, `nav`, `crest`, `nadir`, `hurdle`                 |

### Hypothesis Testing Events

| Event Type          | Purpose           | Key Fields                                                     |
| ------------------- | ----------------- | -------------------------------------------------------------- |
| `hypothesis_result` | Statistical audit | `test_name`, `statistic`, `p_value`, `decision`, `effect_size` |

Example hypothesis result:

```json
{
  "ts": "2026-01-25T14:30:52.123456Z",
  "level": "INFO",
  "msg": "Hypothesis test: shapiro_wilk",
  "component": "statistical_examination",
  "context": {
    "event_type": "hypothesis_result",
    "hypothesis_id": "shapiro_wilk_ith_rb100_lb50_bull_epochs",
    "test_name": "shapiro_wilk",
    "statistic": 0.9523,
    "p_value": 0.0001,
    "decision": "non_normal",
    "effect_size": 0.9523,
    "feature": "ith_rb100_lb50_bull_epochs",
    "n_samples": 5000,
    "gaussianity_class": "moderate_non_normality"
  }
}
```

### Statistical Examination Events

| Event Type                | Purpose                  | Key Fields                                       |
| ------------------------- | ------------------------ | ------------------------------------------------ |
| `shapiro_wilk`            | Normality testing        | `w_stat`, `gaussianity_class`                    |
| `anderson_darling_beta`   | Beta fit testing         | `ad_statistic`, `alpha`, `beta`, `stage_used`    |
| `mann_whitney_u`          | Regime dependence        | `n_trending`, `n_mean_reverting`, `cliffs_delta` |
| `pca_dimensionality`      | Effective dimensionality | `participation_ratio`, `n_components_95`         |
| `ridge_vif`               | Multicollinearity        | `n_high_vif`, `ridge_lambda`                     |
| `augmented_dickey_fuller` | Stationarity             | `adf_stat`, `critical_value`                     |

## Log Levels

- **DEBUG**: Detailed diagnostic information, step-by-step traces
- **INFO**: General operational events, phase completions
- **WARNING**: Potential issues, validation errors, non-fatal problems
- **ERROR**: Failures requiring attention, pipeline aborts

## File Organization

```
logs/
├── ndjson/                          # Component-specific logs
│   ├── bull_ith.jsonl
│   ├── bear_ith.jsonl
│   └── statistical_examination.jsonl
├── forensic/                        # Forensic analysis sessions
│   └── 20260125-143052/
│       ├── examination.ndjson
│       └── hypothesis_results.ndjson
└── {package}-{date}.jsonl           # Legacy format
```

## Python Implementation (loguru)

The `ndjson_logger.py` module provides the standard logger:

```python
from ith_python.ndjson_logger import setup_ndjson_logger, get_trace_id

logger = setup_ndjson_logger("bull_ith")
trace_id = get_trace_id()

# Basic logging
logger.info("Processing started")

# With context
logger.bind(context={
    "phase": "feature_computation",
    "threshold": 100,
}).info("Computing features")

# Hypothesis result event
from ith_python.telemetry.events import log_hypothesis_result

log_hypothesis_result(
    hypothesis_id="shapiro_wilk_feature_x",
    test_name="shapiro_wilk",
    statistic=0.95,
    p_value=0.001,
    decision="non_normal",
    effect_size=0.95,
    context={"feature": "feature_x", "n_samples": 1000},
)
```

## Rust Implementation (tracing)

```rust
use tracing_subscriber::fmt::format::json;
tracing_subscriber::fmt().json().init();

// With structured fields
tracing::info!(
    component = "core_rust",
    trace_id = %trace_id,
    "Processing complete"
);
```

## Bun Implementation (pino)

```typescript
import pino from "pino";

const logger = pino({
  transport: {
    target: "pino/file",
    options: { destination: "./logs/core-bun.jsonl" },
  },
});

logger.info({ component: "core_bun", trace_id }, "Processing started");
```

## mise Tasks for Forensic Analysis

```bash
# Full forensic analysis with NDJSON telemetry
mise run forensic:full

# Audit hypothesis test results
mise run forensic:hypothesis-audit

# View recent logs
mise run forensic:view-logs

# Telemetry event summary
mise run forensic:telemetry-summary
```

## Querying Logs

### jq Examples

```bash
# Find all hypothesis results
cat logs/ndjson/*.jsonl | jq -c 'select(.context.event_type == "hypothesis_result")'

# Filter by test type
cat logs/ndjson/*.jsonl | jq -c 'select(.context.test_name == "shapiro_wilk")'

# Get decisions summary
cat logs/ndjson/*.jsonl | jq -c 'select(.context.event_type == "hypothesis_result") | .context.decision' | sort | uniq -c
```

### Python Analysis

```python
import json
from pathlib import Path
from collections import Counter

decisions = Counter()
for line in Path("logs/ndjson/statistical_examination.jsonl").read_text().splitlines():
    event = json.loads(line)
    if event.get("context", {}).get("event_type") == "hypothesis_result":
        decisions[event["context"]["decision"]] += 1

print(decisions.most_common())
```

## Related Documentation

- **ITH Methodology**: [docs/ITH.md](ITH.md)
- **Statistical Examination**: [packages/ith-python/src/ith_python/statistical_examination/CLAUDE.md](../packages/ith-python/src/ith_python/statistical_examination/CLAUDE.md)
- **Telemetry Module**: `packages/ith-python/src/ith_python/telemetry/`
