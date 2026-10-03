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
DASHBOARD_PORT="20128"
# THE ACTUAL OpenAI-compatible API lives on a SEPARATE port from the
# dashboard in the Docker deployment -- confirmed by reading their own
# docker-compose.yml (API_PORT defaults to 20129, DASHBOARD_PORT to
# 20128). The npm quickstart puts both on 20128 in a single process; the
# Docker image does not. A bare `docker run -p 20128:20128` (what we did
# on the first two attempts) never exposed the real API port at all --
# that's WHY /v1/models never answered, not a timing issue.
API_PORT="20129"
CONTAINER_NAME="omniroute-probe"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --npm)      MODE="npm"; shift ;;
    --model)    MODEL="${2:-auto}"; shift 2 ;;
    --port)     DASHBOARD_PORT="${2:-20128}"; shift 2 ;;
    --api-port) API_PORT="${2:-20129}"; shift 2 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1 (try --help)"; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
mkdir -p reports
TS="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="reports/omniroute_probe_${TS}.md"
# The npm path really does put everything on one port; the Docker path
# needs the split API_PORT.
if [[ "$MODE" == "npm" ]]; then
  BASE="http://localhost:${DASHBOARD_PORT}/v1"
else
  BASE="http://localhost:${API_PORT}/v1"
fi

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
    run_logged "stop redis sidecar" docker rm -f "${CONTAINER_NAME}-redis"
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
  # The pre-built image is wired for their docker-compose "web" profile,
  # which expects a Redis sidecar (REDIS_URL) and enforces Bearer-token
  # auth on /v1/* by default (confirmed via their AUTHZ_GUIDE.md: the
  # OMNIROUTE_API_KEY / ROUTER_API_KEY env vars are accepted as a
  # persistent passthrough key). Without both, the container reports
  # "healthy" and listens fine, but every /v1 call 401s -- which looks
  # identical to "never started" from a plain curl -f readiness probe.
  docker rm -f "${CONTAINER_NAME}-redis" >/dev/null 2>&1
  run_logged "start redis sidecar" docker run -d --name "${CONTAINER_NAME}-redis" redis:7-alpine
  TEST_API_KEY="omniroute-probe-$(date +%s)"
  say "\nUsing generated test API key (not a real credential): ${TEST_API_KEY}"

  run_logged "docker pull" docker pull diegosouzapw/omniroute:latest
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1
  run_logged "docker run" docker run -d --name "$CONTAINER_NAME" \
    -p "${DASHBOARD_PORT}:${DASHBOARD_PORT}" \
    -p "${API_PORT}:${API_PORT}" \
    --link "${CONTAINER_NAME}-redis:redis" \
    -e "DASHBOARD_PORT=${DASHBOARD_PORT}" \
    -e "API_PORT=${API_PORT}" \
    -e "API_HOST=0.0.0.0" \
    -e "PORT=${DASHBOARD_PORT}" \
    -e "REDIS_URL=redis://redis:6379" \
    -e "OMNIROUTE_API_KEY=${TEST_API_KEY}" \
    -e "ROUTER_API_KEY=${TEST_API_KEY}" \
    diegosouzapw/omniroute:latest
else
  run_logged "npm install -g omniroute" npm install -g omniroute
  say "\nStarting omniroute in the background (log: /tmp/omniroute_${TS}.log)"
  nohup omniroute > "/tmp/omniroute_${TS}.log" 2>&1 &
  OMNI_PID=$!
  say "pid: ${OMNI_PID}"
fi

# npm's zero-config quickstart genuinely has no auth gate; the Docker image
# enforces Bearer auth by default (see step 1), so only send the header
# when we actually generated a key for it.
if [[ "$MODE" == "docker" ]]; then
  AUTH_HEADER="Authorization: Bearer ${TEST_API_KEY}"
else
  AUTH_HEADER="X-Omniroute-Probe: 1"   # harmless no-op header, npm needs no auth
fi

# ── 2. Wait for it to actually answer ─────────────
say "\n## 2. Wait for the gateway to respond (checking ${BASE}/models)"
READY="no"
for i in $(seq 1 90); do
  if curl -fsS --max-time 3 -H "${AUTH_HEADER}" "${BASE}/models" >/dev/null 2>&1; then
    READY="yes"
    say "\nGateway answered after ~$((i * 3))s."
    break
  fi
  sleep 3
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
  "curl -fsS --max-time 15 -H '${AUTH_HEADER}' '${BASE}/models' | python3 -c \"
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
  "curl -fsS --max-time 120 '${BASE}/chat/completions' \
    -H 'Content-Type: application/json' \
    -H '${AUTH_HEADER}' \
    -d '{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: OK\"}]}' \
    | python3 -m json.tool"

say "\n(diagnostics on failure: run_logged shows a non-2xx body verbatim above --\n"\
"if you see 'noauth'/'exhausted_connection' for opencode/felo, or 'ENOTFOUND'\n"\
"for openrouter.ai/aihorde.net/api.wulong.dev, that's a DNS/VPN-level block\n"\
"on OmniRoute's own free-tier backend hosts, not a bug in OmniRoute or in\n"\
"this script. Retrying off-VPN is the next real step.)"

# ── 5. The real test: our own schema-validity protocol ───────
# This is the check that matters. A gateway that chats fine but cannot
# honour response_format=json_schema is useless to this project, because
# every synthesizer call depends on schema-constrained JSON.
say "\n## 5. THE REAL TEST -- schema-valid JSON via verify_offvpn.py"
say "\nRunning the same 4-step protocol that caught Nemotron's prose failure."
export OMNIROUTE_PROBE_KEY="${TEST_API_KEY:-unused}"
run_logged "verify_offvpn.py --test-model openai/${MODEL} --test-api-base ${BASE} --test-key-env OMNIROUTE_PROBE_KEY" \
  uv run python scripts/verify_offvpn.py \
    --test-model "openai/${MODEL}" \
    --test-api-base "${BASE}" \
    --test-key-env OMNIROUTE_PROBE_KEY \
    --skip-suite

say "\n## 6. Verdict guide (how to read section 3 above)"
say "- **3a PASS** -> gateway honoured response_format=json_schema at all."
say "- **3b PASS** -> output was bare-parseable JSON, no cleanup needed."
say "- **3c PASS** -> json_extract.py could rescue it (acceptable fallback)."
say "- **3d**      -> whether it leaked <think> reasoning tags."
say "\nAll four PASS = genuinely viable as a CHAIN_AGENTIC hop."
say "3a FAIL      = cannot do schema-constrained JSON -> unusable here,"
say "               regardless of how many models it advertises."
