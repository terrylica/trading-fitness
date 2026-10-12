#!/usr/bin/env bash
# Fail when pytest cannot collect every ith-python test (catches missing dev dependencies).
# Run from packages/ith-python (moon does this).
set -euo pipefail
rc=0
output=$(uv run pytest --collect-only -q 2>&1) || rc=$?
if (( rc != 0 )) || grep -qE "(ImportError|ModuleNotFoundError|ERROR collecting)" <<<"$output"; then
    echo "✗ Test collection failed (exit $rc) - check dev dependencies"
    grep -E "(ImportError|ModuleNotFoundError|ERROR collecting|error:|cause:)" <<<"$output" | head -5
    exit 1
fi
echo "✓ $(grep -oE '[0-9]+ tests? collected' <<<"$output")"
