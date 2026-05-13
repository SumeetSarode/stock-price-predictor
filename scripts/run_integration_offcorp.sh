#!/usr/bin/env bash
# Run the v1 backtest integration test off-corp and capture output.
#
# Why this script exists
# ======================
# Walmart wifi blocks GDELT (DNS) and rate-limits some other endpoints.
# Integration tests need real internet, so they only work when you're
# OFF corp wifi (home wifi / personal hotspot / Eagle).
#
# This wrapper:
#   1. Sanity-checks you're actually off-corp.
#   2. Confirms required API keys are loaded.
#   3. Timestamps the log so multiple runs don't clobber each other.
#   4. Tees output to both terminal AND log file (so you see progress
#      live AND have a file to share back).
set -euo pipefail

cd "$(dirname "$0")/.."

# ── 1. Sanity check: are we actually off Walmart wifi?
#     gecgithub01.walmart.com is internal-only; resolves on corp,
#     fails off-corp. We INVERT that: failure here = good.
echo "── Checking you're off corp wifi..."
if curl -s --max-time 3 -o /dev/null -w "%{http_code}" \
        https://gecgithub01.walmart.com 2>/dev/null | grep -q "^[23]"; then
    echo "❌ STILL ON CORP WIFI. Disconnect and switch to home wifi / hotspot."
    echo "   GDELT (DNS-blocked on corp) won't resolve, half the test will fail."
    exit 1
fi
echo "✅ Off corp wifi (gecgithub01 not reachable, as expected)."

# ── 2. Sanity check: required API keys present.
#     `uv run` natively loads .env, so we ask it to introspect the
#     environment instead of source-ing .env ourselves (which would
#     break on the 'KEY = VALUE' spaces-around-equals format some
#     tools emit).
echo "── Checking GEMINI_API_KEY is loadable via uv run..."
if ! uv run python -c "import os, sys; sys.exit(0 if os.environ.get('GEMINI_API_KEY') else 1)" 2>/dev/null; then
    echo "❌ GEMINI_API_KEY not visible to uv run. Check .env (note:"
    echo "   spaces around '=' will break some loaders; use KEY=VALUE"
    echo "   with NO spaces)."
    exit 1
fi
echo "✅ GEMINI_API_KEY visible to uv run."

# ── 3. Pick a timestamped log path.
mkdir -p reports
LOG="reports/integration_run_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "── Log will be written to: $LOG"
echo

# ── 4. Run the backtest integration test ONLY (not all 8 integration
#     tests -- those each cost LLM quota and we want a focused signal
#     on the v1 acceptance criterion).
#     `tee` keeps you watching live; the log captures everything.
#     `--tb=long` gives full tracebacks if anything fails (vs the default
#     short tb in pyproject.toml).
echo "── Running: pytest -m integration tests/test_backtest_integration.py"
echo "   (this should take 1-5 minutes; rate-limit-aware skip kicks in if quota dies)"
echo
{
    echo "=== Run started: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    echo "=== git HEAD: $(git rev-parse --short HEAD) ==="
    echo "=== Python: $(uv run python --version) ==="
    echo "=== Working directory: $(pwd) ==="
    echo
    uv run pytest \
        -m integration \
        tests/test_backtest_integration.py \
        -v --no-cov --tb=long \
        2>&1
    echo
    echo "=== Run ended: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    echo "=== Exit code: $? ==="
} | tee "$LOG"

echo
echo "── Done. Full log: $LOG"
echo "   Reconnect to corp wifi when ready; Thor will read this on your return."
