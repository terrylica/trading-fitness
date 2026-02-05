# telemetry/

> Provenance tracking and NDJSON event logging for reproducibility.

**← [Back to ith-python](../../../CLAUDE.md)**

## Purpose

Enable scientific reproducibility by:

1. Tracking data provenance (fingerprints, random seeds, git SHA)
2. Logging structured events (NDJSON format)
3. Capturing algorithm initialization state

---

## Module Map

| Module          | Purpose                                                        |
| --------------- | -------------------------------------------------------------- |
| `provenance.py` | ProvenanceContext, fingerprint_array(), capture_random_state() |
| `events.py`     | log_data_load(), log_algorithm_init(), log_epoch_detected()    |

---

## Quick Start

```python
from ith_python.telemetry import ProvenanceContext, log_data_load

# Create provenance context
ctx = ProvenanceContext(
    session_id="abc123",
    experiment_id="exp001",
)

# Log data load event
log_data_load(
    source="bigblack:rangebar_cache",
    rows=1000000,
    columns=["timestamp_ms", "close", "volume"],
    fingerprint=ctx.fingerprint_array(df.to_numpy()),
)
```

---

## ProvenanceContext Fields

| Field           | Type           | Description                  |
| --------------- | -------------- | ---------------------------- |
| `session_id`    | str            | Unique session identifier    |
| `experiment_id` | str \| None    | Optional experiment grouping |
| `input_hashes`  | dict[str, str] | SHA256 of input data         |
| `random_seeds`  | dict[str, int] | NumPy/Python random seeds    |
| `git_sha`       | str            | Current git commit           |
| `config_hash`   | str \| None    | Hash of configuration        |
| `start_time`    | str            | ISO8601 timestamp            |

---

## NDJSON Event Format

```json
{
  "ts": "2026-01-26T01:08:28Z",
  "event_type": "data_load",
  "source": "bigblack:rangebar_cache",
  "rows": 1000000,
  "session_id": "abc123"
}
```

---

## Related Documentation

| Document                                          | Purpose                     |
| ------------------------------------------------- | --------------------------- |
| [ith-python CLAUDE.md](../../../CLAUDE.md)        | Parent package              |
| [docs/LOGGING.md](../../../../../docs/LOGGING.md) | NDJSON telemetry contract   |
| [provenance/](../provenance/)                     | Incremental feature anchors |
