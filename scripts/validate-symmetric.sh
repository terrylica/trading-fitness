#!/usr/bin/env bash
# Symmetric dogfooding: trading-fitness ↔ rangebar-py (docs/adr/2026-01-27-symmetric-dogfooding-rangebar.md).
# Run from packages/ith-python (moon does this).
set -euo pipefail

echo "→ rangebar-py version pin"
pin=$(grep 'rangebar = ' pyproject.toml | grep -o 'tag = "[^"]*"' || true)
if [[ -z "$pin" ]]; then
    echo "  ✗ rangebar-py is not pinned to a tag (pyproject.toml uses a branch)"
    exit 1
fi
echo "  ✓ $pin"

echo "→ Sync dependencies (dev group)"
uv sync --group dev

echo "→ Integration tests"
if compgen -G "tests/integration/test_*.py" >/dev/null; then
    uv run pytest tests/integration/ -v
else
    echo "  ⚠ none found (tests/integration/test_*.py)"
fi

echo "→ rangebar import smoke test"
uv run python -c 'import rangebar; print("  ✓ rangebar", getattr(rangebar, "__version__", "unknown"))'
