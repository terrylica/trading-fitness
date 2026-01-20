# Python Development Skill

Claude Code skill for Python development in trading-fitness.

**← [Back to trading-fitness](../../../CLAUDE.md)**

## Package Documentation

- **Primary**: [ith-python/CLAUDE.md](../../../packages/ith-python/CLAUDE.md)
- **Metrics (PyO3)**: [metrics-rust/CLAUDE.md](../../../packages/metrics-rust/CLAUDE.md)

## Triggers

- Python file changes in `packages/ith-python/`
- Questions about ITH analysis, NAV data processing
- pytest, ruff, uv commands

## Guidelines

### Package Management

```bash
uv add <package>      # Add dependency
uv add --dev <pkg>    # Add dev dependency
uv sync               # Install all deps
uv run <cmd>          # Run in venv
```

### Testing

```bash
uv run pytest                    # Run all tests
uv run pytest -k "test_name"     # Run specific test
```

### Linting

```bash
uv run ruff check --fix          # Lint and auto-fix
uv run ruff format               # Format code
```

### Logging

Use loguru with NDJSON output:

```python
from loguru import logger
from ith_python.paths import get_log_dir

logger.add(get_log_dir() / "ith-python.jsonl", serialize=True)
```
