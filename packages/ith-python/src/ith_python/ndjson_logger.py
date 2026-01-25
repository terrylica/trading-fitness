"""NDJSON structured logging for ITH cross-validation.

This module provides a standardized NDJSON logging infrastructure with:
- UTC timestamps (ISO 8601)
- Stable core schema for machine parsing
- Correlation IDs for tracing across components
- Provenance tracking for scientific reproducibility
- Safe defaults for rotation/retention
- Graceful degradation (logging never crashes the app)

Schema:
{
    "ts": "2026-01-20T00:00:00.000000Z",  # UTC ISO 8601
    "level": "INFO",                       # DEBUG/INFO/WARNING/ERROR/CRITICAL
    "msg": "Human readable message",
    "component": "bull_ith",               # Logger name
    "env": "development",                  # Environment
    "pid": 12345,                          # Process ID
    "tid": 67890,                          # Thread ID
    "trace_id": "abc123",                  # Correlation ID
    "provenance": {                        # Reproducibility context
        "session_id": "sess_20260125_120000",
        "git_sha": "0dc100c",
        "input_hash": null,                # Set per-event
        "random_seed": null                # Set per-event
    },
    "context": {...}                       # Structured context data
}
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


def get_trace_id() -> str:
    """Get or create a trace ID for the current session."""
    if not hasattr(get_trace_id, "_trace_id"):
        get_trace_id._trace_id = uuid.uuid4().hex[:16]
    return get_trace_id._trace_id


def set_trace_id(trace_id: str) -> None:
    """Set a specific trace ID (useful for cross-component correlation)."""
    get_trace_id._trace_id = trace_id


def get_session_id() -> str:
    """Get or create a session ID for provenance tracking."""
    if not hasattr(get_session_id, "_session_id"):
        get_session_id._session_id = f"sess_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    return get_session_id._session_id


def get_git_sha() -> str:
    """Get git SHA for provenance tracking (cached)."""
    if not hasattr(get_git_sha, "_git_sha"):
        import subprocess
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            get_git_sha._git_sha = result.stdout.strip()[:8]
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            get_git_sha._git_sha = "unknown"
    return get_git_sha._git_sha


# Thread-local storage for per-event provenance context
_provenance_context = threading.local()


def set_provenance_context(
    input_hash: str | None = None,
    random_seed: int | None = None,
) -> None:
    """Set per-event provenance context (thread-local).

    Args:
        input_hash: SHA256 hash of input data for this event
        random_seed: Random seed used for this computation
    """
    _provenance_context.input_hash = input_hash
    _provenance_context.random_seed = random_seed


def get_provenance() -> dict:
    """Get current provenance context for logging."""
    return {
        "session_id": get_session_id(),
        "git_sha": get_git_sha(),
        "input_hash": getattr(_provenance_context, "input_hash", None),
        "random_seed": getattr(_provenance_context, "random_seed", None),
    }


class NDJSONFormatter:
    """Format log records as NDJSON with stable schema."""

    def __init__(self, component: str, env: str = "development"):
        self.component = component
        self.env = env

    def format(self, record: dict) -> str:
        """Format a loguru record as NDJSON."""
        try:
            # Extract context from extra if present
            extra = record.get("extra", {})
            context = extra.pop("context", {}) if isinstance(extra.get("context"), dict) else {}

            # Merge remaining extra into context
            for k, v in extra.items():
                if k not in ("context",) and not k.startswith("_"):
                    try:
                        # Ensure JSON serializable
                        json.dumps(v)
                        context[k] = v
                    except (TypeError, ValueError):
                        context[k] = str(v)

            log_entry = {
                "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "level": record["level"].name,
                "msg": record["message"],
                "component": self.component,
                "env": self.env,
                "pid": os.getpid(),
                "tid": threading.get_ident(),
                "trace_id": get_trace_id(),
                "provenance": get_provenance(),
            }

            if context:
                log_entry["context"] = context

            # Add exception info if present
            if record.get("exception"):
                exc = record["exception"]
                log_entry["exception"] = {
                    "type": exc.type.__name__ if exc.type else None,
                    "value": str(exc.value) if exc.value else None,
                }

            # Escape braces for loguru format_map() - {{ and }} become literal { and }
            json_str = json.dumps(log_entry, default=str)
            return json_str.replace("{", "{{").replace("}", "}}") + "\n"

        except (TypeError, ValueError, KeyError, AttributeError) as e:
            # Graceful degradation for serialization/access errors
            # These are the specific errors that can occur during log formatting
            fallback = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": "ERROR",
                "msg": f"Logging format error ({type(e).__name__}): {e}",
                "component": self.component,
                "trace_id": get_trace_id(),
            }
            json_str = json.dumps(fallback)
            return json_str.replace("{", "{{").replace("}", "}}") + "\n"


def setup_ndjson_logger(
    component: str,
    log_dir: Path | str | None = None,
    env: str = "development",
    level: str = "DEBUG",
    rotation: str = "10 MB",
    retention: str = "7 days",
    console_level: str = "WARNING",
) -> logger:
    """Setup NDJSON logging for a component.

    Args:
        component: Component name (e.g., "bull_ith", "cross_validation")
        log_dir: Directory for log files (default: repo logs/ndjson/)
        env: Environment name
        level: Minimum log level for file logging
        rotation: Log rotation policy (e.g., "10 MB", "1 day")
        retention: Log retention policy (e.g., "7 days", "3 files")
        console_level: Minimum log level for console output

    Returns:
        Configured loguru logger instance
    """
    # Determine log directory
    if log_dir is None:
        # Use repo-local logs/ndjson/ directory
        repo_root = Path(__file__).parent.parent.parent.parent.parent
        log_dir = repo_root / "logs" / "ndjson"
    else:
        log_dir = Path(log_dir)

    log_dir.mkdir(parents=True, exist_ok=True)

    # Create formatter
    formatter = NDJSONFormatter(component=component, env=env)

    # Remove default handlers
    logger.remove()

    # Add NDJSON file handler
    log_file = log_dir / f"{component}.jsonl"
    logger.add(
        str(log_file),
        format=lambda r: formatter.format(r),
        level=level,
        rotation=rotation,
        retention=retention,
        compression="gz",  # Compress rotated files
        enqueue=True,  # Thread-safe async logging
        catch=True,  # Never crash on logging errors
    )

    # Add minimal console handler for critical issues
    if console_level:
        logger.add(
            sys.stderr,
            format="<level>{level}</level>: <level>{message}</level>",
            level=console_level,
            catch=True,
        )

    return logger


class ITHStepLogger:
    """Structured logger for ITH algorithm step-by-step debugging.

    Logs every iteration of the ITH algorithm with full state for
    cross-validation between Numba and Rust implementations.
    """

    def __init__(self, component: str, implementation: str, nav_hash: str):
        """Initialize step logger.

        Args:
            component: Algorithm name (e.g., "bull_ith", "bear_ith")
            implementation: "numba" or "rust"
            nav_hash: Hash of input NAV array for correlation
        """
        self.component = component
        self.implementation = implementation
        self.nav_hash = nav_hash
        self.steps: list[dict] = []

    def log_init(
        self,
        nav_len: int,
        tmaeg: float,
        nav_first: float,
        nav_last: float,
    ) -> None:
        """Log initialization state."""
        self.steps.append({
            "step": "init",
            "i": 0,
            "nav_len": nav_len,
            "tmaeg": tmaeg,
            "nav_first": nav_first,
            "nav_last": nav_last,
            "impl": self.implementation,
            "nav_hash": self.nav_hash,
        })

    def log_iteration(
        self,
        i: int,
        equity: float,
        next_equity: float,
        excess_gain: float,
        excess_loss: float,
        endorsing_crest: float,
        candidate_crest: float,
        candidate_nadir: float,
        reset_condition: bool,
        epoch: bool,
        **extra,
    ) -> None:
        """Log a single iteration's state."""
        self.steps.append({
            "step": "iter",
            "i": i,
            "equity": round(equity, 10),
            "next_equity": round(next_equity, 10),
            "excess_gain": round(excess_gain, 10),
            "excess_loss": round(excess_loss, 10),
            "endorsing_crest": round(endorsing_crest, 10),
            "candidate_crest": round(candidate_crest, 10),
            "candidate_nadir": round(candidate_nadir, 10),
            "reset_condition": reset_condition,
            "epoch": epoch,
            **{k: round(v, 10) if isinstance(v, float) else v for k, v in extra.items()},
        })

    def log_epoch_event(
        self,
        epoch_index: int,
        bar_index: int,
        excess_gain: float,
        excess_loss: float,
        endorsing_crest: float,
        candidate_nadir: float,
        tmaeg: float,
        timestamp: str | None = None,
        nav_at_epoch: float | None = None,
    ) -> None:
        """Log epoch detection event for P&L attribution.

        This method logs when an ITH epoch is detected, capturing all
        necessary context for forensic analysis and P&L attribution.

        Args:
            epoch_index: Sequential epoch number
            bar_index: Index in data array where epoch occurred
            excess_gain: Excess gain value at epoch
            excess_loss: Excess loss value at epoch
            endorsing_crest: Endorsing crest value
            candidate_nadir: Candidate nadir value
            tmaeg: TMAEG threshold used
            timestamp: Optional ISO timestamp
            nav_at_epoch: Optional NAV value at epoch
        """
        self.steps.append({
            "step": "epoch_detected",
            "epoch_index": epoch_index,
            "bar_index": bar_index,
            "timestamp": timestamp,
            "excess_gain": round(excess_gain, 10),
            "excess_loss": round(excess_loss, 10),
            "endorsing_crest": round(endorsing_crest, 10),
            "candidate_nadir": round(candidate_nadir, 10),
            "tmaeg_threshold": round(tmaeg, 10),
            "position_type": self.component.split("_")[0],  # "bull" or "bear"
            "nav_at_epoch": round(nav_at_epoch, 10) if nav_at_epoch is not None else None,
        })

    def log_result(
        self,
        num_epochs: int,
        intervals_cv: float,
        max_drawdown: float | None = None,
        max_runup: float | None = None,
    ) -> None:
        """Log final result."""
        result = {
            "step": "result",
            "num_epochs": num_epochs,
            "intervals_cv": round(intervals_cv, 10) if intervals_cv == intervals_cv else None,  # Handle NaN
            "impl": self.implementation,
            "nav_hash": self.nav_hash,
        }
        if max_drawdown is not None:
            result["max_drawdown"] = round(max_drawdown, 10)
        if max_runup is not None:
            result["max_runup"] = round(max_runup, 10)
        self.steps.append(result)

    def to_ndjson(self) -> str:
        """Export all steps as NDJSON."""
        lines = []
        for step in self.steps:
            step_with_meta = {
                "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "component": self.component,
                "trace_id": get_trace_id(),
                **step,
            }
            lines.append(json.dumps(step_with_meta, default=str))
        return "\n".join(lines)

    def write_to_file(self, path: Path | str) -> None:
        """Write all steps to an NDJSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_ndjson() + "\n")


def compare_step_logs(
    numba_log: ITHStepLogger,
    rust_log: ITHStepLogger,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> dict:
    """Compare two step logs and identify discrepancies.

    Args:
        numba_log: Step log from Numba implementation
        rust_log: Step log from Rust implementation
        rtol: Relative tolerance for float comparison
        atol: Absolute tolerance for float comparison

    Returns:
        Dictionary with comparison results and any discrepancies
    """
    discrepancies = []

    # Get iteration steps only
    numba_iters = [s for s in numba_log.steps if s.get("step") == "iter"]
    rust_iters = [s for s in rust_log.steps if s.get("step") == "iter"]

    if len(numba_iters) != len(rust_iters):
        discrepancies.append({
            "type": "length_mismatch",
            "numba_len": len(numba_iters),
            "rust_len": len(rust_iters),
        })

    # Compare each iteration
    float_fields = [
        "equity", "next_equity", "excess_gain", "excess_loss",
        "endorsing_crest", "candidate_crest", "candidate_nadir",
    ]
    bool_fields = ["reset_condition", "epoch"]

    for idx, (ns, rs) in enumerate(zip(numba_iters, rust_iters)):
        iter_discrepancies = []

        for field in float_fields:
            nv, rv = ns.get(field, 0.0), rs.get(field, 0.0)
            if abs(nv - rv) > atol + rtol * abs(nv):
                iter_discrepancies.append({
                    "field": field,
                    "numba": nv,
                    "rust": rv,
                    "diff": abs(nv - rv),
                })

        for field in bool_fields:
            nv, rv = ns.get(field), rs.get(field)
            if nv != rv:
                iter_discrepancies.append({
                    "field": field,
                    "numba": nv,
                    "rust": rv,
                })

        if iter_discrepancies:
            discrepancies.append({
                "type": "iteration_mismatch",
                "index": idx,
                "i": ns.get("i"),
                "fields": iter_discrepancies,
            })

    # Compare results
    numba_result = next((s for s in numba_log.steps if s.get("step") == "result"), None)
    rust_result = next((s for s in rust_log.steps if s.get("step") == "result"), None)

    if numba_result and rust_result:
        if numba_result.get("num_epochs") != rust_result.get("num_epochs"):
            discrepancies.append({
                "type": "epoch_count_mismatch",
                "numba": numba_result.get("num_epochs"),
                "rust": rust_result.get("num_epochs"),
            })

        ncv = numba_result.get("intervals_cv")
        rcv = rust_result.get("intervals_cv")
        if ncv is not None and rcv is not None:
            if abs(ncv - rcv) > atol + rtol * abs(ncv):
                discrepancies.append({
                    "type": "cv_mismatch",
                    "numba": ncv,
                    "rust": rcv,
                    "diff": abs(ncv - rcv),
                })

    return {
        "aligned": len(discrepancies) == 0,
        "total_iterations": len(numba_iters),
        "discrepancy_count": len(discrepancies),
        "discrepancies": discrepancies,
    }
