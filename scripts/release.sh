#!/usr/bin/env bash
# semantic-release with a GH_TOKEN minted per run from the repo account's gh profile
# (gh-token-for-repo), never stored. .releaserc.yml's verifyConditionsCmd checks the tree is clean.
#   bash scripts/release.sh            release
#   bash scripts/release.sh --dry-run  preview
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
[[ -d node_modules ]] || npm install --silent
GH_TOKEN="$("$HOME/.claude/tools/bin/gh-token-for-repo")"
export GH_TOKEN
exec npx semantic-release --no-ci "$@"
