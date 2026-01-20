# Rust Development Skill

Claude Code skill for Rust development in trading-fitness.

**← [Back to trading-fitness](../../../CLAUDE.md)**

## Package Documentation

- **Core**: [core-rust/CLAUDE.md](../../../packages/core-rust/CLAUDE.md)
- **Metrics + PyO3**: [metrics-rust/CLAUDE.md](../../../packages/metrics-rust/CLAUDE.md)

## Triggers

- Rust file changes in `packages/core-rust/` or `packages/metrics-rust/`
- Performance optimization requests
- cargo, maturin commands

## Guidelines

### Building

```bash
cargo build            # Debug build
cargo build --release  # Release build
cargo check            # Type check only
```

### Testing

```bash
cargo test             # Run all tests
cargo test <name>      # Run specific test
cargo nextest run      # Parallel testing (if installed)
```

### Linting

```bash
cargo clippy           # Lint
cargo fmt              # Format
```

### Logging

Use tracing with JSON output:

```rust
use tracing::{info, warn, error};
use tracing_subscriber::fmt::format::json;

tracing_subscriber::fmt().json().init();
info!(package = "core-rust", "Operation started");
```
