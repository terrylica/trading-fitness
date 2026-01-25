# Observability Telemetry Enhancement Plan

## Goal

Maximize replayable telemetry for forensic analysis, scientific reproducibility, and P&L attribution in this polyglot monorepo, with focus on:

1. **Scientific Reproducibility** - Random seed capture, data fingerprinting, provenance chain
2. **Trading Domain Events** - Decision snapshots, feature values at decision time, ITH epoch events
3. **Batch Analysis Forensics** - Post-hoc debugging of ITH analysis and statistical examination

**Scope**: NDJSON-only (no OpenTelemetry SDK). Cross-language logging activation is secondary to domain event capture.

---

## Current State

| Component | Library | Status |
|-----------|---------|--------|
| **Python (ith-python)** | loguru + `ndjson_logger.py` | ACTIVE - trace_id, rotation, compression |
| **Rust (core-rust, metrics-rust)** | tracing | DECLARED, NOT USED |
| **Bun (core-bun)** | pino | DECLARED, NOT USED |
| **Profiling** | None | No py-spy infrastructure |

**Key Existing Files**:
- `packages/ith-python/src/ith_python/ndjson_logger.py` - Production NDJSON logger
- `packages/ith-python/src/ith_python/statistical_examination/runner.py` - Already uses NDJSON logging
- `docs/LOGGING.md` - Logging contract (needs update)

---

## Implementation Plan

### Phase 1: Scientific Reproducibility Infrastructure

**Goal**: Enable any ITH analysis to be reproduced from logs alone.

#### 1.1 Provenance Context Module

Create `packages/ith-python/src/ith_python/telemetry/provenance.py`:

```python
@dataclass
class ProvenanceContext:
    """Track data lineage through analysis pipeline."""
    session_id: str
    experiment_id: str | None
    input_hashes: dict[str, str]  # name -> sha256
    random_seeds: dict[str, int]  # component -> seed
    git_sha: str
    config_hash: str

def fingerprint_array(arr: np.ndarray, name: str) -> dict:
    """Generate reproducibility fingerprint for array."""
    return {
        "name": name,
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "range": [float(arr.min()), float(arr.max())],
        "checksum_first_10": hashlib.sha256(arr[:10].tobytes()).hexdigest()[:16],
    }

def capture_random_state() -> dict:
    """Capture numpy random state for reproducibility."""
    state = np.random.get_state()
    return {
        "numpy_seed": int(state[1][0]),
        "numpy_state_hash": hashlib.sha256(state[1].tobytes()).hexdigest()[:16],
    }
```

#### 1.2 Extend ndjson_logger.py

Add to existing `ndjson_logger.py`:

```python
# New fields in NDJSONFormatter
"provenance": {
    "session_id": "sess_20260125_120000",
    "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()[:8],
    "input_hash": None,  # Set per-event
    "random_seed": None, # Set per-event
}
```

#### 1.3 Reproducibility Events

| Event Type | Fields | Purpose |
|------------|--------|---------|
| `data.load` | `source_path`, `sha256_hash`, `row_count`, `schema` | Input fingerprinting |
| `algorithm.init` | `algorithm_name`, `version`, `random_seed`, `config` | Reproducibility anchor |
| `algorithm.result` | `final_metrics`, `output_hash` | Outcome capture |

**Files to Modify**:
- `packages/ith-python/src/ith_python/ith.py` - Add data.load event at CSV read
- `packages/ith-python/src/ith_python/bear_ith.py` - Add algorithm.init with seed

---

### Phase 2: Trading Domain Events

**Goal**: Log sufficient context for P&L attribution and decision debugging.

#### 2.1 Decision Snapshot Schema

Create `packages/shared-types/schemas/telemetry/decision-snapshot.json`:

```json
{
  "$id": "decision-snapshot.json",
  "type": "object",
  "required": ["decision_id", "timestamp_ns", "features_at_decision"],
  "properties": {
    "decision_id": { "type": "string" },
    "timestamp_ns": { "type": "integer" },
    "trace_id": { "type": "string" },

    "features_at_decision": {
      "type": "object",
      "properties": {
        "bull_ith_epochs": { "type": "integer" },
        "bear_ith_epochs": { "type": "integer" },
        "bull_cv": { "type": "number" },
        "bear_cv": { "type": "number" },
        "max_drawdown": { "type": "number" },
        "max_runup": { "type": "number" },
        "tmaeg": { "type": "number" },
        "nav_at_decision": { "type": "number" }
      }
    },

    "thresholds": {
      "type": "object",
      "properties": {
        "tmaeg_method": { "type": "string" },
        "tmaeg_value": { "type": "number" },
        "lookback": { "type": "integer" },
        "threshold_dbps": { "type": "integer" }
      }
    }
  }
}
```

#### 2.2 ITH Epoch Event Schema

Create `packages/shared-types/schemas/telemetry/ith-epoch-event.json`:

```json
{
  "$id": "ith-epoch-event.json",
  "type": "object",
  "required": ["epoch_index", "bar_index", "excess_gain", "excess_loss"],
  "properties": {
    "epoch_index": { "type": "integer" },
    "bar_index": { "type": "integer" },
    "timestamp": { "type": "string", "format": "date-time" },
    "trace_id": { "type": "string" },

    "excess_gain": { "type": "number" },
    "excess_loss": { "type": "number" },
    "endorsing_crest": { "type": "number" },
    "candidate_nadir": { "type": "number" },
    "tmaeg_threshold": { "type": "number" },

    "position_type": { "enum": ["bull", "bear"] }
  }
}
```

#### 2.3 ITHStepLogger Enhancement

The existing `ITHStepLogger` in `ndjson_logger.py` is already well-designed. Extend it:

```python
# Add to ITHStepLogger
def log_epoch_event(
    self,
    epoch_index: int,
    bar_index: int,
    excess_gain: float,
    excess_loss: float,
    endorsing_crest: float,
    candidate_nadir: float,
    tmaeg: float,
) -> None:
    """Log epoch detection event for P&L attribution."""
    self._log_step({
        "event_type": "epoch_detected",
        "epoch_index": epoch_index,
        "bar_index": bar_index,
        "excess_gain": excess_gain,
        "excess_loss": excess_loss,
        "endorsing_crest": endorsing_crest,
        "candidate_nadir": candidate_nadir,
        "tmaeg_threshold": tmaeg,
        "position_type": self.component.split("_")[0],  # bull or bear
    })
```

**Files to Modify**:
- `packages/ith-python/src/ith_python/ndjson_logger.py` - Add epoch event logging
- `packages/ith-python/src/ith_python/ith.py` - Emit epoch events in ITH loop

---

### Phase 3: Statistical Examination Telemetry

**Goal**: Track hypothesis testing and statistical decisions for audit.

#### 3.1 Hypothesis Tracking

Add to `runner.py`:

```python
def log_hypothesis_result(
    hypothesis_id: str,
    test_name: str,
    statistic: float,
    p_value: float,
    effect_size: float | None,
    decision: str,
    context: dict,
) -> None:
    """Log statistical test result with full context."""
    logger.bind(context={
        "event_type": "hypothesis_result",
        "hypothesis_id": hypothesis_id,
        "test_name": test_name,
        "statistic": statistic,
        "p_value": p_value,
        "effect_size": effect_size,
        "decision": decision,
        **context,
    }).info(f"Hypothesis test: {test_name}")
```

#### 3.2 Statistical Test Events

Instrument each statistical module:

| Module | Events to Add |
|--------|---------------|
| `distribution.py` | `shapiro_wilk_result`, `beta_fit_result`, `ad_test_result` |
| `regime.py` | `mann_whitney_result`, `cohens_d_result`, `cliffs_delta_result` |
| `dimensionality.py` | `pca_result`, `vif_result`, `participation_ratio` |
| `temporal.py` | `adf_result`, `acf_result` |

**Files to Modify**:
- `packages/ith-python/src/ith_python/statistical_examination/distribution.py`
- `packages/ith-python/src/ith_python/statistical_examination/regime.py`
- `packages/ith-python/src/ith_python/statistical_examination/dimensionality.py`
- `packages/ith-python/src/ith_python/statistical_examination/temporal.py`

---

### Phase 4: py-spy Profiling Infrastructure

**Goal**: Flight-recorder style profiling with consistent artifact naming.

#### 4.1 Add py-spy to mise.toml

```toml
# SSoT-OK: These are runtime configuration values, not package versions
[tools]
"pipx:py-spy" = "latest"

[env]
TF_PROFILE_ENABLED = "false"
TF_PROFILE_RATE = "100"      # Hz sampling rate
TF_PROFILE_DURATION = "60"   # seconds
```

#### 4.2 Profiling Mise Tasks

Add to `mise.toml`:

```toml
[tasks."profile:ith"]
description = "Profile ITH analysis with flamegraph"
run = """
mkdir -p artifacts/profiles
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT="artifacts/profiles/ith-python-$TIMESTAMP.svg"
cd packages/ith-python
py-spy record \
  --output "../../$OUTPUT" \
  --format flamegraph \
  --rate ${TF_PROFILE_RATE} \
  --duration ${TF_PROFILE_DURATION} \
  -- uv run python -m ith_python.ith
echo "Profile saved: $OUTPUT"
"""

[tasks."profile:examination"]
description = "Profile statistical examination"
run = """
mkdir -p artifacts/profiles
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT="artifacts/profiles/examination-$TIMESTAMP.svg"
cd packages/ith-python
py-spy record \
  --output "../../$OUTPUT" \
  --format flamegraph \
  --rate ${TF_PROFILE_RATE} \
  --duration ${TF_PROFILE_DURATION} \
  -- uv run python -m ith_python.statistical_examination.runner --verbose
echo "Profile saved: $OUTPUT"
"""
```

#### 4.3 Profile Artifact Naming

Pattern: `{package}-{YYYYMMDD-HHMMSS}.svg`

Directory structure:
```
artifacts/profiles/
├── ith-python-20260125-143052.svg
├── examination-20260125-150000.svg
└── INDEX.json  # Generated by mise task
```

---

### Phase 5: Documentation Update

**Goal**: Update LOGGING.md with unified schema.

#### 5.1 Updated Schema

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

#### 5.2 CLAUDE.md Update

Add to `packages/ith-python/src/ith_python/statistical_examination/CLAUDE.md`:

```markdown
## Telemetry Events

| Event Type | Purpose | Key Fields |
|------------|---------|------------|
| `data.load` | Input fingerprinting | sha256_hash, row_count |
| `algorithm.init` | Reproducibility anchor | random_seed, config_hash |
| `epoch_detected` | P&L attribution | epoch_index, excess_gain |
| `hypothesis_result` | Statistical audit | test_name, p_value, decision |
```

---

## Critical Files to Modify

| File | Priority | Changes |
|------|----------|---------|
| `packages/ith-python/src/ith_python/ndjson_logger.py` | P0 | Add provenance context, epoch event logging |
| `packages/ith-python/src/ith_python/ith.py` | P0 | Add data.load, algorithm.init, epoch events |
| `packages/ith-python/src/ith_python/bear_ith.py` | P1 | Mirror ith.py changes |
| `packages/ith-python/src/ith_python/statistical_examination/runner.py` | P1 | Add hypothesis tracking |
| `packages/ith-python/src/ith_python/statistical_examination/distribution.py` | P2 | Add test result events |
| `packages/ith-python/src/ith_python/statistical_examination/regime.py` | P2 | Add effect size events |
| `mise.toml` | P1 | Add profiling tasks |
| `docs/LOGGING.md` | P2 | Update schema documentation |
| `packages/shared-types/schemas/telemetry/` | P2 | New event schemas |

---

## New Files to Create

| File | Purpose |
|------|---------|
| `packages/ith-python/src/ith_python/telemetry/__init__.py` | Telemetry module |
| `packages/ith-python/src/ith_python/telemetry/provenance.py` | Provenance tracking |
| `packages/ith-python/src/ith_python/telemetry/events.py` | Event type definitions |
| `packages/shared-types/schemas/telemetry/decision-snapshot.json` | Decision event schema |
| `packages/shared-types/schemas/telemetry/ith-epoch-event.json` | Epoch event schema |
| `packages/shared-types/schemas/telemetry/hypothesis-result.json` | Hypothesis schema |

---

## Verification Plan

### Unit Tests

```bash
# Test provenance module
UV_PYTHON=python3.13 uv run pytest tests/test_telemetry/ -v

# Verify NDJSON output format
UV_PYTHON=python3.13 uv run python -c "
from ith_python.ndjson_logger import setup_ndjson_logger
logger = setup_ndjson_logger('test')
logger.info('Test event')
"
cat logs/ndjson/test.jsonl | jq .
```

### Integration Test

```bash
# Run ITH analysis and verify telemetry
UV_PYTHON=python3.13 uv run python -m ith_python.ith --verbose

# Check for required events
grep 'data.load' logs/ndjson/bull_ith.jsonl
grep 'algorithm.init' logs/ndjson/bull_ith.jsonl
grep 'epoch_detected' logs/ndjson/bull_ith.jsonl
```

### Profiling Test

```bash
# Verify py-spy works
mise run profile:ith
ls -la artifacts/profiles/*.svg
```

### Reproducibility Test

```bash
# Run twice with same seed, compare output hashes
UV_PYTHON=python3.13 uv run python -m ith_python.ith --seed 42 > /tmp/run1.txt
UV_PYTHON=python3.13 uv run python -m ith_python.ith --seed 42 > /tmp/run2.txt
diff /tmp/run1.txt /tmp/run2.txt  # Should be identical
```

---

## Implementation Sequence

### Day 1: Foundation
- [ ] Create `telemetry/` module with provenance.py
- [ ] Extend ndjson_logger.py with provenance fields
- [ ] Add data.load event to ith.py

### Day 2: ITH Events
- [ ] Add algorithm.init event to ith.py
- [ ] Add epoch_detected events in ITH loop
- [ ] Mirror changes to bear_ith.py

### Day 3: Statistical Examination
- [ ] Add hypothesis tracking to runner.py
- [ ] Instrument distribution.py with test result events
- [ ] Instrument regime.py with effect size events

### Day 4: Profiling
- [ ] Add py-spy to mise.toml
- [ ] Create profiling tasks
- [ ] Test flamegraph generation

### Day 5: Documentation
- [ ] Update LOGGING.md
- [ ] Update CLAUDE.md files
- [ ] Create telemetry event schemas

---

## Success Criteria

1. **Reproducibility**: Running ITH with `--seed 42` twice produces identical results
2. **Forensics**: Can reconstruct which epochs triggered from logs alone
3. **Auditability**: All statistical test decisions logged with full context
4. **Profiling**: Can generate flamegraph for any analysis run
5. **No Regressions**: All 81 tests still passing

---

## Out of Scope (Deferred)

- OpenTelemetry SDK integration
- Rust tracing activation (secondary priority per user)
- Bun pino activation (secondary priority per user)
- Live trading order lifecycle events
- Real-time latency tracking
