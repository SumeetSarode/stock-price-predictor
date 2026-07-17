#!/usr/bin/env bash
# check_no_walmart_traces.sh
#
# Pre-ship gate: scrub for any Walmart-internal references that would
# leak into a public OSS release. This MUST exit 0 before tagging v1.0.0.
#
# Pattern catches:
#   - 'walmart' / 'wal-mart' (any case)
#   - 'wmlink' / 'gecgithub' / 'sysproxy'
#   - 'artifacts.walmart' (artifactory URLs)
#   - 'code-puppy' / 'code_puppy' (AI coding-tool names -- must not leak into this project)
#
# Exclusions:
#   - .git/, .venv/, node_modules/ (vendored or VCS internals)
#   - This script itself
#   - The reminder block in pyproject.toml (intentional opt-in instructions)
#
# Exit 0  -> clean, safe to ship
# Exit 1  -> traces found, must scrub first
#
# Run from the repo root:  ./scripts/check_no_walmart_traces.sh

set -euo pipefail

cd "$(dirname "$0")/.."

PATTERN='walmart|wal-mart|wmlink|gecgithub|sysproxy|code-puppy|code_puppy|artifacts\.walmart'

EXCLUDES=(
    --glob='!.git'
    --glob='!.venv'
    --glob='!node_modules'
    --glob='!uv.lock'        # regenerate against public PyPI as part of purge
    --glob='!scripts/check_no_walmart_traces.sh'  # this file
)

if rg -i -n "${EXCLUDES[@]}" "$PATTERN" . 2>/dev/null; then
    echo ""
    echo "🚨 Walmart traces found. Cannot ship until clean."
    echo "   Walmart-purge is the final pre-ship gate: no internal traces may ship."
    exit 1
fi

echo "✅ Clean. No Walmart traces — safe to ship."
