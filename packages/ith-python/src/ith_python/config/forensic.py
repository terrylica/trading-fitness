"""Forensic Configuration Loader and Validator.

Loads config/forensic.toml and provides typed access to configuration.

Architecture: Multi-View Feature Architecture with DAG-based orchestration
Reference: docs/plans/2026-01-25-multi-view-feature-architecture-plan.md
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass
class DateRangeConfig:
    """Date range configuration for data fetching."""

    strategy: Literal["all_available", "rolling_days", "fixed", "n_bars"] = "all_available"
    rolling_days: int = 90
    start_date: str | None = None
    end_date: str | None = None
    n_bars: int | None = None


@dataclass
class ValidationConfig:
    """Continuity validation configuration."""

    preset: Literal["permissive", "research", "standard", "strict", "paranoid"] = "research"
    tolerance_pct: float | None = None
    on_failure: Literal["error", "warn", "skip"] = "warn"

    @property
    def effective_tolerance(self) -> float:
        """Get effective tolerance percentage."""
        if self.tolerance_pct is not None:
            return self.tolerance_pct
        presets = {
            "permissive": 0.05,
            "research": 0.02,
            "standard": 0.01,
            "strict": 0.005,
            "paranoid": 0.001,
        }
        return presets[self.preset]


@dataclass
class ClickHouseConfig:
    """ClickHouse connection configuration."""

    auto_start: bool = True
    timeout: int = 30
    max_retries: int = 3


@dataclass
class PreflightConfig:
    """Preflight check configuration."""

    min_valid_bars: int = 500
    auto_fetch: bool = True
    max_lookback_days: int = 365


@dataclass
class OutputConfig:
    """Output directory configuration."""

    ssot_dir: str = "artifacts/ssot"
    views_dir: str = "artifacts/views"
    analysis_dir: str = "artifacts/analysis"
    logs_dir: str = "logs/ndjson"
    include_microstructure: bool = False


@dataclass
class TelemetryConfig:
    """Telemetry configuration."""

    enabled: bool = True
    emit_progress: bool = True
    emit_provenance: bool = True


@dataclass
class ForensicConfig:
    """Complete forensic analysis configuration."""

    # Data configuration
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    thresholds: list[int] = field(default_factory=lambda: [25, 50, 100, 250])
    lookbacks: list[int] = field(default_factory=lambda: [20, 50, 100, 200, 500])
    source: str = "binance"
    market: str = "spot"

    # Nested configs
    date_range: DateRangeConfig = field(default_factory=DateRangeConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    clickhouse: ClickHouseConfig = field(default_factory=ClickHouseConfig)
    preflight: PreflightConfig = field(default_factory=PreflightConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    # Computed properties
    @property
    def max_lookback(self) -> int:
        """Maximum lookback window."""
        return max(self.lookbacks)

    @property
    def warmup_bars(self) -> int:
        """Number of warmup bars required."""
        return self.max_lookback - 1

    @property
    def symbol_threshold_matrix(self) -> list[tuple[str, int]]:
        """All symbol x threshold combinations."""
        return [(s, t) for s in self.symbols for t in self.thresholds]

    @property
    def total_combinations(self) -> int:
        """Total number of symbol x threshold combinations."""
        return len(self.symbols) * len(self.thresholds)


def load_forensic_config(config_path: Path | str | None = None) -> ForensicConfig:
    """Load forensic configuration from TOML file.

    Args:
        config_path: Path to config file. Defaults to config/forensic.toml
                     relative to project root.

    Returns:
        ForensicConfig with all settings loaded.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config validation fails.
    """
    if config_path is None:
        # Find project root (look for mise.toml)
        current = Path.cwd()
        while current != current.parent:
            if (current / "mise.toml").exists():
                config_path = current / "config" / "forensic.toml"
                break
            current = current.parent
        else:
            config_path = Path("config/forensic.toml")

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    # Parse data section
    data = raw.get("data", {})
    date_range_raw = data.get("date_range", {})

    date_range = DateRangeConfig(
        strategy=date_range_raw.get("strategy", "all_available"),
        rolling_days=date_range_raw.get("rolling_days", 90),
        start_date=date_range_raw.get("start_date"),
        end_date=date_range_raw.get("end_date"),
        n_bars=date_range_raw.get("n_bars"),
    )

    # Parse validation section
    validation_raw = raw.get("validation", {})
    validation = ValidationConfig(
        preset=validation_raw.get("preset", "research"),
        tolerance_pct=validation_raw.get("tolerance_pct"),
        on_failure=validation_raw.get("on_failure", "warn"),
    )

    # Parse clickhouse section
    ch_raw = raw.get("clickhouse", {})
    clickhouse = ClickHouseConfig(
        auto_start=ch_raw.get("auto_start", True),
        timeout=ch_raw.get("timeout", 30),
        max_retries=ch_raw.get("max_retries", 3),
    )

    # Parse preflight section
    pf_raw = raw.get("preflight", {})
    preflight = PreflightConfig(
        min_valid_bars=pf_raw.get("min_valid_bars", 500),
        auto_fetch=pf_raw.get("auto_fetch", True),
        max_lookback_days=pf_raw.get("max_lookback_days", 365),
    )

    # Parse output section
    out_raw = raw.get("output", {})
    output = OutputConfig(
        ssot_dir=out_raw.get("ssot_dir", "artifacts/ssot"),
        views_dir=out_raw.get("views_dir", "artifacts/views"),
        analysis_dir=out_raw.get("analysis_dir", "artifacts/analysis"),
        logs_dir=out_raw.get("logs_dir", "logs/ndjson"),
        include_microstructure=out_raw.get("include_microstructure", False),
    )

    # Parse telemetry section
    tel_raw = raw.get("telemetry", {})
    telemetry = TelemetryConfig(
        enabled=tel_raw.get("enabled", True),
        emit_progress=tel_raw.get("emit_progress", True),
        emit_provenance=tel_raw.get("emit_provenance", True),
    )

    config = ForensicConfig(
        symbols=data.get("symbols", ["BTCUSDT", "ETHUSDT"]),
        thresholds=data.get("thresholds", [25, 50, 100, 250]),
        lookbacks=data.get("lookbacks", [20, 50, 100, 200, 500]),
        source=data.get("source", "binance"),
        market=data.get("market", "spot"),
        date_range=date_range,
        validation=validation,
        clickhouse=clickhouse,
        preflight=preflight,
        output=output,
        telemetry=telemetry,
    )

    return config


def validate_forensic_config(config: ForensicConfig) -> tuple[bool, list[str]]:
    """Validate forensic configuration.

    Args:
        config: ForensicConfig to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors = []

    # Validate symbols
    if not config.symbols:
        errors.append("No symbols specified")
    for symbol in config.symbols:
        if not symbol.endswith("USDT") and not symbol.endswith("USD"):
            errors.append(f"Invalid symbol format: {symbol} (expected *USDT or *USD)")

    # Validate thresholds
    if not config.thresholds:
        errors.append("No thresholds specified")
    for t in config.thresholds:
        if t < 1 or t > 100000:
            errors.append(f"Threshold {t} out of range [1, 100000]")

    # Validate lookbacks
    if not config.lookbacks:
        errors.append("No lookbacks specified")
    for lb in config.lookbacks:
        if lb < 2:
            errors.append(f"Lookback {lb} too small (min 2)")

    # Validate date range
    dr = config.date_range
    if dr.strategy == "fixed" and (not dr.start_date or not dr.end_date):
        errors.append("Fixed date range requires start_date and end_date")
    elif dr.strategy == "rolling_days" and dr.rolling_days < 1:
        errors.append("rolling_days must be positive")
    elif dr.strategy == "n_bars" and (not dr.n_bars or dr.n_bars < 1):
        errors.append("n_bars strategy requires positive n_bars")

    # Validate preflight
    if config.preflight.min_valid_bars < config.warmup_bars:
        errors.append(
            f"min_valid_bars ({config.preflight.min_valid_bars}) must be >= "
            f"warmup_bars ({config.warmup_bars})"
        )

    # Validate source
    if config.source not in ("binance", "exness"):
        errors.append(f"Unknown source: {config.source}")

    # Validate market
    if config.market not in ("spot", "futures-um", "futures-cm"):
        errors.append(f"Unknown market: {config.market}")

    return len(errors) == 0, errors


def print_config_summary(config: ForensicConfig) -> None:
    """Print configuration summary to stdout."""
    print("=" * 60)
    print("FORENSIC ANALYSIS CONFIGURATION")
    print("=" * 60)
    print()
    print("Data:")
    print(f"  Symbols: {', '.join(config.symbols)}")
    print(f"  Thresholds (dbps): {config.thresholds}")
    print(f"  Lookbacks: {config.lookbacks}")
    print(f"  Source: {config.source} / {config.market}")
    print(f"  Total combinations: {config.total_combinations}")
    print()
    print("Date Range:")
    print(f"  Strategy: {config.date_range.strategy}")
    if config.date_range.strategy == "rolling_days":
        print(f"  Rolling days: {config.date_range.rolling_days}")
    elif config.date_range.strategy == "fixed":
        print(f"  Range: {config.date_range.start_date} to {config.date_range.end_date}")
    print()
    print("Validation:")
    print(f"  Preset: {config.validation.preset}")
    print(f"  Tolerance: {config.validation.effective_tolerance:.2%}")
    print(f"  On failure: {config.validation.on_failure}")
    print()
    print("Preflight:")
    print(f"  Min valid bars: {config.preflight.min_valid_bars}")
    print(f"  Warmup bars: {config.warmup_bars}")
    print(f"  Auto-fetch: {config.preflight.auto_fetch}")
    print()
    print("ClickHouse:")
    print(f"  Auto-start: {config.clickhouse.auto_start}")
    print()
    print("=" * 60)


if __name__ == "__main__":
    # CLI for testing config loading
    import argparse

    parser = argparse.ArgumentParser(description="Load and validate forensic config")
    parser.add_argument("--config", "-c", type=Path, help="Config file path")
    parser.add_argument("--validate", "-v", action="store_true", help="Validate config")
    args = parser.parse_args()

    try:
        config = load_forensic_config(args.config)
        print_config_summary(config)

        if args.validate:
            is_valid, errors = validate_forensic_config(config)
            if not is_valid:
                print("\n❌ Configuration errors:")
                for err in errors:
                    print(f"  - {err}")
                sys.exit(1)
            print("\n✅ Configuration is valid")

    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
