#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# OFFLINE REGRESSION RUNNER  —  run this anywhere, VPN or not.
# ─────────────────────────────────────────────────────────────────
# Everything here is deterministic: NO network, NO LLM, NO API keys.
# All data inputs are synthetic and every model call is stubbed, so it
# runs identically on-VPN, off-VPN, or on a plane.
#
# What it does:
#   1. Runs the FULL deterministic test suite (excludes integration/slow,
#      which are the only network/LLM tests).
#   2. Runs the golden pipeline tests with -s so the snapshot values are
#      printed for human review.
#   3. Runs both ship gates (no-HTML-in-Python, no-Walmart-traces).
#   4. Tees everything to a timestamped log you can paste back.
#
# Usage:   bash scripts/run_regression.sh
set -uo pipefail

cd "$(dirname "$0")/.."

# ── Pick an interpreter that needs NO dependency resolution (works
#    offline). Prefer the project venv; fall back to uv run.
if [[ -x ".venv/bin/python" ]]; then
    PY=(".venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
    PY=(uv run python)
else
    PY=(python3)
fi
echo "── Interpreter: ${PY[*]}"

mkdir -p reports
LOG="reports/regression_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "── Log: $LOG"
echo

FAIL=0
{
    echo "=== Regression run: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    echo "=== git HEAD: $(git rev-parse --short HEAD 2>/dev/null || echo n/a) ==="
    echo "=== Python:  $("${PY[@]}" --version 2>&1) ==="
    echo "=== CWD:     $(pwd) ==="
    echo

    # ── 1. Full deterministic suite ──────────────────────────────
    echo "########################################################"
    echo "# 1/3  FULL DETERMINISTIC TEST SUITE (no network / no LLM)"
    echo "########################################################"
    "${PY[@]}" -m pytest \
        -m "not integration and not slow" \
        -q --no-cov -p no:cacheprovider 2>&1
    SUITE_RC=${PIPESTATUS[0]}
    [[ $SUITE_RC -eq 0 ]] || FAIL=1
    echo "-- suite exit code: $SUITE_RC"
    echo

    # ── 2. Golden snapshot (printed for review) ──────────────────
    echo "########################################################"
    echo "# 2/3  GOLDEN PIPELINE SNAPSHOT (values for human review)"
    echo "########################################################"
    "${PY[@]}" -m pytest \
        tests/test_golden_pipeline.py \
        -v -s --no-cov -p no:cacheprovider 2>&1 \
        | grep -vE "GEMINI|LiteLlm|Authlib|EXPERIMENTAL|feature_decorator|authlib|check_feature|NewsSnapshot|installed default"
    GOLDEN_RC=${PIPESTATUS[0]}
    [[ $GOLDEN_RC -eq 0 ]] || FAIL=1
    echo "-- golden exit code: $GOLDEN_RC"
    echo

    # ── 3. Ship gates ────────────────────────────────────────────
    echo "########################################################"
    echo "# 3/3  SHIP GATES"
    echo "########################################################"
    if bash scripts/check_no_html_in_python.sh >/dev/null 2>&1; then
        echo "no-HTML-in-Python : PASS"
    else
        echo "no-HTML-in-Python : FAIL"; FAIL=1
    fi
    if bash scripts/check_no_walmart_traces.sh >/dev/null 2>&1; then
        echo "no-Walmart-traces : PASS"
    else
        echo "no-Walmart-traces : FAIL"; FAIL=1
    fi
    echo

    echo "========================================================"
    if [[ $FAIL -eq 0 ]]; then
        echo "RESULT: ALL GREEN "
    else
        echo "RESULT: FAILURES DETECTED  — scroll up for details."
    fi
    echo "=== Ended: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
} | tee "$LOG"

echo
echo "── Done. Paste this file back for review: $LOG"
exit $FAIL
