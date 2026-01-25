"""Telemetry event definitions for ITH analysis.

This module defines structured event types for:
- data.load: Input data fingerprinting
- algorithm.init: Reproducibility anchor with seeds and config
- epoch_detected: ITH epoch events for P&L attribution
- hypothesis_result: Statistical test results for audit

Events are designed to be logged as NDJSON for forensic analysis
and scientific reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from ith_python.ndjson_logger import get_trace_id


@dataclass
class DataLoadEvent:
    """Event logged when input data is loaded.

    Captures fingerprint of input data for reproducibility verification.
    """

    source_path: str
    sha256_hash: str
    row_count: int
    column_count: int
    columns: list[str]
    value_range: tuple[float, float] | None = None
    event_type: str = "data.load"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "event_type": self.event_type,
            "source_path": self.source_path,
            "sha256_hash": self.sha256_hash,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": self.columns,
            "value_range": list(self.value_range) if self.value_range else None,
        }


@dataclass
class AlgorithmInitEvent:
    """Event logged when algorithm is initialized.

    Serves as reproducibility anchor with all necessary state.
    """

    algorithm_name: str
    version: str
    random_seed: int | None
    config: dict[str, Any]
    input_hash: str
    event_type: str = "algorithm.init"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "event_type": self.event_type,
            "algorithm_name": self.algorithm_name,
            "version": self.version,
            "random_seed": self.random_seed,
            "config": self.config,
            "input_hash": self.input_hash,
        }


@dataclass
class EpochDetectedEvent:
    """Event logged when ITH epoch is detected.

    Captures full state at epoch detection for P&L attribution.
    """

    epoch_index: int
    bar_index: int
    timestamp: str | None
    excess_gain: float
    excess_loss: float
    endorsing_crest: float
    candidate_nadir: float
    tmaeg_threshold: float
    position_type: str  # "bull" or "bear"
    nav_at_epoch: float | None = None
    event_type: str = "epoch_detected"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "event_type": self.event_type,
            "epoch_index": self.epoch_index,
            "bar_index": self.bar_index,
            "timestamp": self.timestamp,
            "excess_gain": round(self.excess_gain, 10),
            "excess_loss": round(self.excess_loss, 10),
            "endorsing_crest": round(self.endorsing_crest, 10),
            "candidate_nadir": round(self.candidate_nadir, 10),
            "tmaeg_threshold": round(self.tmaeg_threshold, 10),
            "position_type": self.position_type,
            "nav_at_epoch": round(self.nav_at_epoch, 10) if self.nav_at_epoch else None,
        }


@dataclass
class HypothesisResultEvent:
    """Event logged for statistical hypothesis test results.

    Provides full context for audit trail of statistical decisions.
    """

    hypothesis_id: str
    test_name: str
    statistic: float
    p_value: float
    effect_size: float | None
    decision: str  # "reject", "fail_to_reject", "inconclusive"
    alpha: float = 0.05
    context: dict[str, Any] = field(default_factory=dict)
    event_type: str = "hypothesis_result"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "event_type": self.event_type,
            "hypothesis_id": self.hypothesis_id,
            "test_name": self.test_name,
            "statistic": round(self.statistic, 10) if self.statistic is not None else None,
            "p_value": round(self.p_value, 10) if self.p_value is not None else None,
            "effect_size": round(self.effect_size, 10) if self.effect_size is not None else None,
            "decision": self.decision,
            "alpha": self.alpha,
            "context": self.context,
        }


def _emit_event(event_dict: dict[str, Any], level: str = "INFO") -> None:
    """Emit event to loguru with proper formatting.

    Args:
        event_dict: Event data to log
        level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    trace_id = get_trace_id()

    log_entry = {
        "ts": ts,
        "trace_id": trace_id,
        **event_dict,
    }

    # Use loguru's bind to add context, then log with message
    bound_logger = logger.bind(context=log_entry)
    msg = f"[{event_dict.get('event_type', 'event')}]"

    if level == "DEBUG":
        bound_logger.debug(msg)
    elif level == "WARNING":
        bound_logger.warning(msg)
    elif level == "ERROR":
        bound_logger.error(msg)
    else:
        bound_logger.info(msg)


def log_data_load(
    source_path: str,
    sha256_hash: str,
    row_count: int,
    column_count: int,
    columns: list[str],
    value_range: tuple[float, float] | None = None,
) -> DataLoadEvent:
    """Log data load event and return the event object.

    Args:
        source_path: Path to loaded file
        sha256_hash: SHA256 hash of file contents
        row_count: Number of rows loaded
        column_count: Number of columns
        columns: List of column names
        value_range: Optional (min, max) of primary value column

    Returns:
        DataLoadEvent instance
    """
    event = DataLoadEvent(
        source_path=source_path,
        sha256_hash=sha256_hash,
        row_count=row_count,
        column_count=column_count,
        columns=columns,
        value_range=value_range,
    )
    _emit_event(event.to_dict())
    return event


def log_algorithm_init(
    algorithm_name: str,
    version: str,
    config: dict[str, Any],
    input_hash: str,
    random_seed: int | None = None,
) -> AlgorithmInitEvent:
    """Log algorithm initialization event.

    Args:
        algorithm_name: Name of algorithm (e.g., "bull_ith", "bear_ith")
        version: Version string
        config: Configuration dictionary
        input_hash: Hash of input data
        random_seed: Random seed if applicable

    Returns:
        AlgorithmInitEvent instance
    """
    event = AlgorithmInitEvent(
        algorithm_name=algorithm_name,
        version=version,
        random_seed=random_seed,
        config=config,
        input_hash=input_hash,
    )
    _emit_event(event.to_dict())
    return event


def log_epoch_detected(
    epoch_index: int,
    bar_index: int,
    excess_gain: float,
    excess_loss: float,
    endorsing_crest: float,
    candidate_nadir: float,
    tmaeg_threshold: float,
    position_type: str,
    timestamp: str | None = None,
    nav_at_epoch: float | None = None,
) -> EpochDetectedEvent:
    """Log ITH epoch detection event.

    Args:
        epoch_index: Sequential epoch number
        bar_index: Index in data array where epoch occurred
        excess_gain: Excess gain value at epoch
        excess_loss: Excess loss value at epoch
        endorsing_crest: Endorsing crest value
        candidate_nadir: Candidate nadir value
        tmaeg_threshold: TMAEG threshold used
        position_type: "bull" or "bear"
        timestamp: Optional ISO timestamp
        nav_at_epoch: Optional NAV value at epoch

    Returns:
        EpochDetectedEvent instance
    """
    event = EpochDetectedEvent(
        epoch_index=epoch_index,
        bar_index=bar_index,
        timestamp=timestamp,
        excess_gain=excess_gain,
        excess_loss=excess_loss,
        endorsing_crest=endorsing_crest,
        candidate_nadir=candidate_nadir,
        tmaeg_threshold=tmaeg_threshold,
        position_type=position_type,
        nav_at_epoch=nav_at_epoch,
    )
    _emit_event(event.to_dict(), level="DEBUG")
    return event


def log_hypothesis_result(
    hypothesis_id: str,
    test_name: str,
    statistic: float,
    p_value: float,
    decision: str,
    effect_size: float | None = None,
    alpha: float = 0.05,
    context: dict[str, Any] | None = None,
) -> HypothesisResultEvent:
    """Log statistical hypothesis test result.

    Args:
        hypothesis_id: Unique identifier for hypothesis
        test_name: Name of statistical test
        statistic: Test statistic value
        p_value: P-value from test
        decision: "reject", "fail_to_reject", or "inconclusive"
        effect_size: Optional effect size measure
        alpha: Significance level used
        context: Additional context dictionary

    Returns:
        HypothesisResultEvent instance
    """
    event = HypothesisResultEvent(
        hypothesis_id=hypothesis_id,
        test_name=test_name,
        statistic=statistic,
        p_value=p_value,
        effect_size=effect_size,
        decision=decision,
        alpha=alpha,
        context=context or {},
    )
    _emit_event(event.to_dict())
    return event
