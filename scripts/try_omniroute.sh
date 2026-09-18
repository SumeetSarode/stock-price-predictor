#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# try_omniroute.sh -- install, boot and TEST OmniRoute end-to-end.
#
# WHY THIS EXISTS
#   OmniRoute (https://github.com/diegosouzapw/OmniRoute) is a local
#   OpenAI-compatible gateway that fronts many providers behind one
#   endpoint. Before it goes anywhere near CHAIN_AGENTIC it has to prove
#   the ONE thing this project actually needs: can it return
#   SCHEMA-VALID JSON on demand, or does it emit prose? That is exactly
#   the check that caught Nemotron's real failure mode, so we reuse the
#   same protocol rather than inventing a second, weaker one.
#
#   It could not be verified on the machine this script was written on:
#   registry.npmjs.org was DNS-blocked (NXDOMAIN) and no Docker daemon
#   was installed. Hence: a script you run where the network works.
#
# WHAT IT DOES (nothing destructive, nothing permanent unless you say so)
#   1. Starts OmniRoute -- Docker by default (no global npm install,
#      container is removed on exit), or npm with --npm.
#   2. Waits for the gateway to actually answer.
#   3. Asks it which models it exposes.
#   4. Runs a plain smoke-test chat call.
#   5. Runs the REAL test: scripts/verify_offvpn.py --test-model ...
#      pointed at the gateway, which checks schema-valid JSON, bare
#      parseability, json_extract rescue, and reasoning-tag leakage.
#   6. Writes EVERYTHING to reports/omniroute_probe_<timestamp>.md
#      plus the verifier's own reports/offvpn_verification_*.md|json.
#
# USAGE
#   bash scripts/try_omniroute.sh              # Docker (recommended)
#   bash scripts/try_omniroute.sh --npm        # npm -g instead of Docker
#   bash scripts/try_omniroute.sh --model auto # override the model tested
#
# WHEN IT FINISHES, send back the reports/omniroute_probe_*.md file.
# ─────────────────────────────────────────────────────────────
set -uo pipefail   # deliberately NOT -e: a failing probe must still
                   # reach the report instead of killing the script.

MODE="docker"
MODEL="auto"
PORT="20128"
CONTAINER_NAME="omniroute-probe"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --npm)   MODE="npm"; shift ;;
    --model) MODEL="${2:-auto}"; shift 2 ;;
    --port)  PORT="${2:-20128}"; shift 2 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1 (try --help)"; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
mkdir -p reports
TS="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="reports/omniroute_probe_${TS}.md"
BASE="http://localhost:${PORT}/v1"

# Everything written by `say` lands on BOTH your screen and the report,
# so the file you send back is the whole story, not a summary of it.
say() { echo -e "$*" | tee -a "$REPORT"; }
run_logged() {  # run_logged <label> <cmd...>
  local label="$1"; shift
  say "\n### ${label}\n"
  say '```'
  { "$@" 2>&1 || echo "[command exited non-zero: $?]"; } | tee -a "$REPORT"
  say '```'
}

{
  echo "# OmniRoute probe report"
  echo
  echo "- **When (UTC):** $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- **Host:** $(hostname) ($(uname -s) $(uname -r))"
  echo "- **Mode:** ${MODE}"
  echo "- **Gateway:** ${BASE}"
  echo "- **Model under test:** ${MODEL}"
  echo "- **Git HEAD:** $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
} > "$REPORT"

say "\n## 0. Environment"
run_logged "tooling versions" bash -c 'echo "node: $(node --version 2>&1 || echo absent)"; echo "npm:  $(npm --version 2>&1 || echo absent)"; echo "docker: $(docker --version 2>&1 || echo absent)"'

# ── Teardown always runs, so a failed probe never leaves a stray
#    container or a port held hostage.
cleanup() {
  say "\n## 99. Cleanup"
  if [[ "$MODE" == "docker" ]]; then
    run_logged "stop container" docker rm -f "$CONTAINER_NAME"
  else
    if [[ -n "${OMNI_PID:-}" ]]; then
      run_logged "kill omniroute pid ${OMNI_PID}" kill "$OMNI_PID"
    fi
  fi
  say "\n---\n**Report written to:** ${REPORT}"
  echo
  echo "=========================================================="
  echo "DONE. Send this file back:"
  echo "  ${REPO_ROOT}/${REPORT}"
  echo "Also attach any reports/offvpn_verification_*.md from just now."
  echo "=========================================================="
}
trap cleanup EXIT

# ── 1. Start the gateway ─────────────────────────────────────
say "\n## 1. Start OmniRoute (${MODE})"
if [[ "$MODE" == "docker" ]]; then
  if ! docker info >/dev/null 2>&1; then
    say "\n**FATAL: Docker daemon not reachable.** Start Docker Desktop and re-run,"
    say "or use: bash scripts/try_omniroute.sh --npm"
    exit 1
  fi
  run_logged "docker pull" docker pull diegosouzapw/omniroute:latest
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1
  run_logged "docker run" docker run -d --name "$CONTAINER_NAME" \
    -p "${PORT}:20128" diegosouzapw/omniroute:latest
else
  run_logged "npm install -g omniroute" npm install -g omniroute
  say "\nStarting omniroute in the background (log: /tmp/omniroute_${TS}.log)"
  nohup omniroute > "/tmp/omniroute_${TS}.log" 2>&1 &
  OMNI_PID=$!
  say "pid: ${OMNI_PID}"
fi

# ── 2. Wait for it to actually answer ────────────────────────
say "\n## 2. Wait for the gateway to respond"
READY="no"
for i in $(seq 1 60); do
  if curl -fsS --max-time 3 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    READY="yes"
    say "\nGateway answered after ~${i}s."
    break
  fi
  sleep 2
done

if [[ "$READY" != "yes" ]]; then
  say "\n**FAIL: gateway never answered on ${BASE} after ~120s.**"
  [[ "$MODE" == "docker" ]] && run_logged "container logs" docker logs --tail 80 "$CONTAINER_NAME"
  [[ "$MODE" == "npm" ]] && run_logged "npm logs" tail -80 "/tmp/omniroute_${TS}.log"
  exit 1
fi

# ── 3-4. What does it expose, and does a basic call work? ────
say "\n## 3. Models exposed"
run_logged "GET /v1/models (first 60 ids)" bash -c \
  "curl -fsS --max-time 15 'http://localhost:${PORT}/v1/models' | python3 -c \"
import json,sys
try:
    d = json.load(sys.stdin)
except Exception as e:
    print('could not parse /v1/models as JSON:', e); raise SystemExit
items = d.get('data', d if isinstance(d, list) else [])
print('total models:', len(items))
for m in items[:60]:
    print(' -', m.get('id') if isinstance(m, dict) else m)
\""

say "\n## 4. Plain smoke-test chat call (model=${MODEL})"
run_logged "POST /v1/chat/completions" bash -c \
  "curl -fsS --max-time 120 'http://localhost:${PORT}/v1/chat/completions' \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: OK\"}]}' \
    | python3 -m json.tool"

# ── 5. The real test: our own schema-validity protocol ───────
# This is the check that matters. A gateway that chats fine but cannot
# honour response_format=json_schema is useless to this project, because
# every synthesizer call depends on schema-constrained JSON.
say "\n## 5. THE REAL TEST -- schema-valid JSON via verify_offvpn.py"
say "\nRunning the same 4-step protocol that caught Nemotron's prose failure."
run_logged "verify_offvpn.py --test-model openai/${MODEL} --test-api-base ${BASE}" \
  uv run python scripts/verify_offvpn.py \
    --test-model "openai/${MODEL}" \
    --test-api-base "${BASE}" \
    --skip-suite

say "\n## 6. Verdict guide (how to read section 3 above)"
say "- **3a PASS** -> gateway honoured response_format=json_schema at all."
say "- **3b PASS** -> output was bare-parseable JSON, no cleanup needed."
say "- **3c PASS** -> json_extract.py could rescue it (acceptable fallback)."
say "- **3d**      -> whether it leaked <think> reasoning tags."
say "\nAll four PASS = genuinely viable as a CHAIN_AGENTIC hop."
say "3a FAIL      = cannot do schema-constrained JSON -> unusable here,"
say "               regardless of how many models it advertises."
