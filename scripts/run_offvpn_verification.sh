#!/usr/bin/env bash
# Run the OFF-VPN verification suite and capture everything to a report.
#
# WHY THIS EXISTS
# ===============
# A corporate/VPN network blocks openrouter.ai, Yahoo Finance and nseindia.com
# outright, so three things can NEVER be verified from there:
#
#   1. Nemotron actually returns schema-valid JSON (not prose) on a real call.
#      This was previously asserted from a litellm metadata flag that turned
#      out to mean "unknown model", not "unsupported" -- never proven live.
#   2. json_extract.py's reasoning-model handling survives REAL Nemotron
#      output, not just the synthetic strings in the unit tests.
#   3. The price chain's behaviour against Yahoo's "Invalid Crumb" error.
#
# Run this from a HOME / non-VPN network. It hits the real internet on purpose.
#
# USAGE
#   ./scripts/run_offvpn_verification.sh              # everything
#   ./scripts/run_offvpn_verification.sh --quick      # skip the slow pytest suite
#   ./scripts/run_offvpn_verification.sh --no-llm     # skip LLM (saves quota)
#
# OUTPUT
#   reports/offvpn_verification_<UTC timestamp>.md    <- send me this one
#   reports/offvpn_verification_<UTC timestamp>.json  <- machine-readable
#   reports/offvpn_console_<UTC timestamp>.log        <- raw console transcript
#
# Deliberately NOT `set -e`: a FAILING CHECK IS DATA, not a reason to abort.
# We want the report written no matter what, so the exit code is captured and
# re-raised at the very end instead.
set -uo pipefail

cd "$(dirname "$0")/.."

ARGS=()
for a in "$@"; do
    case "$a" in
        --quick)  ARGS+=("--skip-suite") ;;
        --no-llm) ARGS+=("--skip-llm") ;;
        *)        ARGS+=("$a") ;;
    esac
done

mkdir -p reports
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CONSOLE="reports/offvpn_console_${STAMP}.log"

echo "─────────────────────────────────────────────────────────────"
echo " Off-VPN verification"
echo "─────────────────────────────────────────────────────────────"
echo " Started : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " Git HEAD: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo " Console : $CONSOLE"
echo

# ── Advisory only. A missing OPENROUTER_API_KEY doesn't stop the run: the
#    script SKIPs the Nemotron section and still verifies everything else.
#
#    NOTE: we ask *Settings* (which reads .env directly), NOT os.environ.
#    Nothing in this repo calls load_dotenv(), so `uv run` alone does not
#    export .env into the environment -- checking os.environ here would
#    print a false "key missing" warning even when the key IS set.
if ! uv run python -c "
import sys
sys.path.insert(0, 'src')
from price_predictor.config.settings import settings
sys.exit(0 if settings.openrouter_api_key.get_secret_value().strip() else 1)
" 2>/dev/null; then
    echo "NOTE: OPENROUTER_API_KEY is empty in .env."
    echo "      The Nemotron checks (the main event) will be SKIPPED."
    echo "      Paste your key after 'OPENROUTER_API_KEY=' in .env, then re-run."
    echo
fi

# tee so you watch it live AND get a file to send back.
#
# ${ARGS[@]+"${ARGS[@]}"} -- NOT "${ARGS[@]:-}". Under `set -u` an empty
# array needs guarding, but the `:-` form substitutes an EMPTY STRING, so
# running with no flags passed argv=[''] and argparse died with
# "unrecognized arguments:" before a single check ran. The `+` form expands
# to nothing at all when the array is empty, which is what we actually want.
uv run python scripts/verify_offvpn.py ${ARGS[@]+"${ARGS[@]}"} 2>&1 | tee "$CONSOLE"
STATUS="${PIPESTATUS[0]}"

echo
echo "────────────────────────────────────────────────────────────"

# A .md report is written by the Python script for every real run. If it
# isn't there, the script died before doing ANY work (bad flag, import
# error, crash) -- say so loudly instead of printing a cheerful summary
# over a 107-byte log. This exact failure already cost one wasted off-VPN
# trip when an empty-array expansion sent argv=[''] to argparse.
if ! ls reports/offvpn_verification_*.md >/dev/null 2>&1; then
    echo " THE SCRIPT DID NOT RUN. No report was produced."
    echo " Console output was:"
    echo
    sed 's/^/     /' "$CONSOLE"
    echo
    echo " Nothing was verified. Send the above and don't bother re-running"
    echo " off-VPN until it's fixed."
    echo "────────────────────────────────────────────────────────────"
    exit 1
fi

if [[ "$STATUS" -eq 0 ]]; then
    echo " All checks passed (or were intentionally skipped)."
else
    echo " Some checks FAILED -- that's useful data, not a crash."
fi
echo " Reports written to: reports/"
ls -1t reports/offvpn_* 2>/dev/null | head -3 | sed 's/^/   /'
echo
echo " Send me the .md file and I'll work from it."
echo "─────────────────────────────────────────────────────────────"

exit "$STATUS"
