#!/usr/bin/env bash
#
# Verify the BUILT image, not the configuration that describes it.
#
# This exists because the distinction turned out to matter more than anything
# else in the repair. Five test assertions asserted that .dockerignore
# *contained* the right patterns; all five passed; the image they described
# shipped backend/.env with five live API keys. Nothing short of building and
# running the thing could see it. See REPAIR_REPORT.md §F.
#
# Every check prints PASS/FAIL and the evidence it judged on. Exits non-zero if
# any check fails, so CI fails on a regression rather than reporting one.
#
# Usage:
#   scripts/verify_image.sh [--build] [--live] [--keep]
#
#   --build  build the image first, for linux/amd64 (the deployment target).
#            The Dockerfile's own leak guard runs as part of this.
#   --live   additionally drive a real POST /portfolio/analyze. Needs outbound
#            network and a market-data provider that will answer this host —
#            Yahoo blocks many datacentre IP ranges (see d0acc54), so this is
#            opt-in rather than part of the default run. Off by default so that
#            a provider refusing a CI runner cannot be mistaken for a defect in
#            OptionQ, and cannot be papered over either.
#   --keep   leave the verification container running for inspection.
#
# Environment:
#   ENGINE   container engine; auto-detected (docker, then podman).
#   IMAGE    image tag to verify. Default optionq-api:verify.
#   PORT     host port for the serving check. Default 8921.

set -uo pipefail

ENGINE="${ENGINE:-}"
IMAGE="${IMAGE:-optionq-api:verify}"
PORT="${PORT:-8921}"
DO_BUILD=0
DO_LIVE=0
DO_KEEP=0
CONTAINER=optionq-verify

for arg in "$@"; do
  case "$arg" in
    --build) DO_BUILD=1 ;;
    --live)  DO_LIVE=1 ;;
    --keep)  DO_KEEP=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [ -z "$ENGINE" ]; then
  for candidate in docker podman; do
    command -v "$candidate" >/dev/null 2>&1 && { ENGINE="$candidate"; break; }
  done
fi
if [ -z "$ENGINE" ]; then
  cat >&2 <<'EOF'
No container engine found (looked for docker, then podman).

This script cannot be substituted with reasoning about what the image would
contain. That substitution is exactly what hid a P0 for two repair cycles. Run
it where an engine exists — CI does, via .github/workflows/ci.yml.
EOF
  exit 127
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

pass=0
fail=0
ok() { echo "PASS  $1"; pass=$((pass + 1)); }
no() { echo "FAIL  $1"; fail=$((fail + 1)); }

echo "engine: $ENGINE    image: $IMAGE    live: $DO_LIVE"
echo

if [ "$DO_BUILD" = 1 ]; then
  echo "################ 0. build for the deployment target ################"
  # linux/amd64 explicitly: Cloud Run is x86-64 and developer laptops are often
  # arm64, so an unqualified build can produce an image that cannot run there.
  # The Dockerfile's leak guard is a RUN step, so a leak fails this build.
  "$ENGINE" build --platform linux/amd64 -t "$IMAGE" -f "$REPO_ROOT/Dockerfile" "$REPO_ROOT" \
    2>&1 | tail -25
  # shellcheck disable=SC2181
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    no "image builds for linux/amd64"
    echo
    echo "RESULT: $pass passed, $((fail)) failed — build failed, nothing else could run"
    exit 1
  fi
  ok "image builds for linux/amd64 (the Dockerfile leak guard passed)"
  echo
fi

echo "################ 1. what the image contains ################"
"$ENGINE" run --rm --entrypoint sh "$IMAGE" -c '
  echo "--- uid/gid:"; id
  echo "--- ownership:"; ls -ld /app /app/backend /app/backend/ml_models /var/cache/optionq
  echo "--- secrets present?"; find /app -name ".env" -o -name ".env.*" ! -name ".env.example" -o -name "*.pem" -o -name "*.key" | head
  echo "--- tests present?"; ls /app/backend/tests 2>&1 | head -2
  echo "--- db files present?"; find /app -name "*.db" -o -name "*.sqlite*" | head
  echo "--- venv present?"; find /app -name ".venv" -maxdepth 3 | head
  echo "--- python/platform:"; python -c "import sys,platform;print(sys.version);print(platform.machine(), platform.libc_ver())"
' > "$WORK/contents.log" 2>&1
cat "$WORK/contents.log"

section() { sed -n "/--- $1/,/^--- /p" "$WORK/contents.log" | sed '1d;$d'; }

grep -q 'uid=10001' "$WORK/contents.log" \
  && ok "runs as uid 10001, not root" \
  || no "not running as uid 10001"

[ -z "$(section 'secrets present?' | grep -E '\.env|\.pem|\.key')" ] \
  && ok "no .env / .pem / .key in the image" \
  || no "secret-shaped file found in the image"

section 'tests present?' | grep -qi 'No such file' \
  && ok "backend/tests excluded from the image" \
  || no "backend/tests present in the image"

[ -z "$(section 'db files present?' | grep -E '\.db|\.sqlite')" ] \
  && ok "no SQLite database baked into the image" \
  || no "SQLite database shipped in the image"

[ -z "$(section 'venv present?' | grep -E '\.venv')" ] \
  && ok "host virtualenv excluded from the image" \
  || no "host virtualenv shipped in the image"

grep -q 'x86_64' "$WORK/contents.log" \
  && ok "image is x86-64, matching Cloud Run" \
  || no "image is not x86-64"

echo
echo "################ 2. imports resolve inside the image ################"
# The failure mode this pins: the container once installed a different
# numerical stack from the one any test had exercised, and a transitive
# prerelease (wrapt 2.6.0.dev1) resolved into the pricing dependency graph.
"$ENGINE" run --rm --entrypoint python "$IMAGE" -c '
import importlib
mods = ["fastapi","uvicorn","pydantic","pydantic_settings","yfinance","curl_cffi",
        "requests","pandas","numpy","scipy","sklearn","joblib","arch","anthropic",
        "huggingface_hub","apscheduler","dotenv","wrapt"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append((m, repr(e)))
print("FAILED_IMPORTS:", bad)
import wrapt, pandas, numpy
print("wrapt", wrapt.__version__, "pandas", pandas.__version__, "numpy", numpy.__version__)
import backend.main
print("backend.main imported OK")
' > "$WORK/imports.log" 2>&1
cat "$WORK/imports.log"

grep -q 'FAILED_IMPORTS: \[\]' "$WORK/imports.log" \
  && ok "every runtime dependency imports inside the image" \
  || no "an import failed inside the image"
grep -q 'backend.main imported OK' "$WORK/imports.log" \
  && ok "backend.main imports inside the image" \
  || no "backend.main does not import inside the image"
grep -qE 'wrapt 2\.5\.2' "$WORK/imports.log" \
  && ok "installed wrapt is stable 2.5.2, not a dev prerelease" \
  || no "wrapt in the image is not the pinned 2.5.2"

echo
echo "################ 3. fail-closed startup ################"
# Each of these must exit non-zero. A container that boots with no secrets, or
# with a localhost CORS origin in production, is a container that will serve
# somebody.
GOOD_ADMIN=Zk8mQv2pLx7wRt4yHb9nJc3sDf6gTa1e
GOOD_CRON=Np5qXr8tVb2mKd7wYj4hGc9sLf3zAe6u

check_refuses() {
  local label="$1"; shift
  "$ENGINE" run --rm "$@" "$IMAGE" > "$WORK/refuse.log" 2>&1
  local rc=$?
  tail -6 "$WORK/refuse.log"
  echo "exit=$rc"
  if [ $rc -ne 0 ]; then ok "$label (exit $rc)"; else no "$label — IT STARTED"; fi
}

echo "--- 3a. production, no secrets at all"
check_refuses "refuses to start with no secrets" \
  -e ENVIRONMENT=production -e FRONTEND_URL=https://optionq.example.com

echo "--- 3b. production, secrets too short to be secrets"
check_refuses "refuses to start on under-length secrets" \
  -e ENVIRONMENT=production -e ADMIN_API_KEY=short -e CRON_SECRET=alsoshort \
  -e FRONTEND_URL=https://optionq.example.com

echo "--- 3c. production, http://localhost CORS origin"
check_refuses "refuses an http/localhost origin in production" \
  -e ENVIRONMENT=production -e ADMIN_API_KEY="$GOOD_ADMIN" -e CRON_SECRET="$GOOD_CRON" \
  -e FRONTEND_URL=http://localhost:3000

echo
echo "################ 4. a correctly configured container serves ################"
"$ENGINE" rm -f "$CONTAINER" >/dev/null 2>&1
"$ENGINE" run -d --name "$CONTAINER" -p "${PORT}:8080" \
  -e ENVIRONMENT=production \
  -e ADMIN_API_KEY="$GOOD_ADMIN" \
  -e CRON_SECRET="$GOOD_CRON" \
  -e FRONTEND_URL=https://optionq.example.com \
  "$IMAGE" >/dev/null 2>&1

cleanup_container() {
  if [ "$DO_KEEP" = 1 ]; then
    echo "(container $CONTAINER left running on port $PORT)"
  else
    "$ENGINE" logs "$CONTAINER" > "$WORK/server.log" 2>&1
    "$ENGINE" rm -f "$CONTAINER" >/dev/null 2>&1
  fi
}

code=""
for _ in $(seq 1 40); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/health" 2>/dev/null)
  [ "$code" = "200" ] && break
  sleep 3
done
echo "--- /health: $code"
if [ "$code" = "200" ]; then
  ok "/health returns 200 from the running container"
  curl -s "http://localhost:${PORT}/health"; echo
else
  no "/health never returned 200 (last=$code)"
  "$ENGINE" logs "$CONTAINER" 2>&1 | tail -30
  cleanup_container
  echo
  echo "RESULT: $pass passed, $fail failed"
  exit 1
fi

echo
echo "################ 5. auth matrix ################"
# Two credentials, two routers. The cells that matter most are the ones where a
# VALID credential is refused: CRON_SECRET must not unlock /admin, and
# ADMIN_API_KEY must not unlock /admin/cron.
status_noauth() { curl -s -o /dev/null -w '%{http_code}' -X "$1" "http://localhost:${PORT}$2"; }
status_auth()   { curl -s -o /dev/null -w '%{http_code}' -X "$1" -H "Authorization: Bearer $3" "http://localhost:${PORT}$2"; }

# A 405 means the path exists but not with that method, so the cell never
# reached the auth dependency and proves nothing about authorization. It is
# called out separately from a wrong status code because the defect is in this
# script, not in the app: an expectation written against a method the app does
# not serve is a check that cannot fail for the reason it claims to test.
expect() {
  local label="$1" want="$2" got="$3"
  if [ "$got" = "405" ] && [ "$want" != "405" ]; then
    no "$label → 405: no such method on this route; the check never reached auth"
  elif [ "$got" = "$want" ]; then
    ok "$label → $got"
  else
    no "$label → $got (expected $want)"
  fi
}

expect "GET /admin/training/status  no header"   401 "$(status_noauth GET /admin/training/status)"
expect "GET /admin/training/status  wrong token" 401 "$(status_auth GET /admin/training/status wrong-token-entirely)"
expect "GET /admin/training/status  ADMIN key"   200 "$(status_auth GET /admin/training/status "$GOOD_ADMIN")"
expect "GET /admin/training/status  CRON secret" 401 "$(status_auth GET /admin/training/status "$GOOD_CRON")"
expect "GET /admin/cron/retrain-fast  no header"   401 "$(status_noauth GET /admin/cron/retrain-fast)"
expect "GET /admin/cron/retrain-fast  wrong token" 401 "$(status_auth GET /admin/cron/retrain-fast wrong-token-entirely)"
expect "GET /admin/cron/retrain-fast  ADMIN key"   401 "$(status_auth GET /admin/cron/retrain-fast "$GOOD_ADMIN")"
# The positive cell is deliberately behind --live. A 200 here is not a cheap
# assertion: the handler runs a real fast retrain, which fetches market data and
# fits a model. Asserting 200 without outbound data access would either fail for
# a reason that is not an authorization defect, or have to accept a 5xx — and a
# cell that accepts an error is not evidence that the credential works.
if [ "$DO_LIVE" = 1 ]; then
  expect "GET /admin/cron/retrain-fast  CRON secret" 200 "$(status_auth GET /admin/cron/retrain-fast "$GOOD_CRON")"
else
  echo "SKIP  GET /admin/cron/retrain-fast  CRON secret → 200 (needs --live: the handler retrains)"
fi

body=$(curl -s "http://localhost:${PORT}/admin/training/status")
echo "--- unauthorized body: $body"
case "$body" in
  *Unauthorized*) ok "unauthorized body is a bare message, no internals" ;;
  *) no "unexpected unauthorized body" ;;
esac
echo "$body" | grep -qiE 'traceback|File "|line [0-9]+|\.py' \
  && no "unauthorized body leaks internal detail" \
  || ok "unauthorized body leaks no paths or tracebacks"

if [ "$DO_LIVE" = 1 ]; then
  echo
  echo "################ 6. a real analysis through the container ################"
  cat > "$WORK/req.json" <<'JSON'
{
  "holdings": [
    {"ticker": "AAPL", "quantity": 500, "position_type": "stock"},
    {"ticker": "MSFT", "quantity": 300, "position_type": "stock"},
    {"ticker": "SPY",  "quantity": 400, "position_type": "etf"}
  ],
  "protection_level": 0.1,
  "max_hedge_cost_pct": 0.02,
  "time_horizon_days": 60
}
JSON
  start=$(date +%s)
  http=$(curl -s -o "$WORK/analyze.json" -w '%{http_code}' \
    -X POST "http://localhost:${PORT}/portfolio/analyze" \
    -H 'Content-Type: application/json' --data @"$WORK/req.json" --max-time 180)
  elapsed=$(( $(date +%s) - start ))
  echo "--- HTTP $http in ${elapsed}s, $(wc -c < "$WORK/analyze.json" | tr -d ' ') bytes"

  if [ "$http" = "200" ]; then
    ok "POST /portfolio/analyze returns 200 through the container"
    "$ENGINE" run --rm -i --entrypoint python "$IMAGE" -c '
import json, sys
d = json.load(sys.stdin)
def dig(o, k, acc=None):
    acc = [] if acc is None else acc
    if isinstance(o, dict):
        for kk, vv in o.items():
            if kk == k: acc.append(vv)
            dig(vv, k, acc)
    elif isinstance(o, list):
        for vv in o: dig(vv, k, acc)
    return acc
tickers = sorted({t for t in dig(d, "contract_symbol") if isinstance(t, str)})
print("OCC contracts:", tickers[:6], f"({len(tickers)} total)")
print("premium_source:", sorted(set(dig(d, "premium_source"))))
print("iv_source:", sorted(set(dig(d, "iv_source"))))
print("top-level keys:", sorted(d.keys())[:14])
' < "$WORK/analyze.json" || echo "(response shape probe failed; raw body saved)"
  else
    no "POST /portfolio/analyze returned $http, not 200"
    head -c 800 "$WORK/analyze.json"; echo
  fi
fi

echo
echo "################ container logs: tracebacks? ################"
"$ENGINE" logs "$CONTAINER" > "$WORK/server.log" 2>&1
if grep -qE 'Traceback \(most recent call last\)' "$WORK/server.log"; then
  no "traceback in the container logs"
  grep -A8 'Traceback' "$WORK/server.log" | head -20
else
  ok "no tracebacks in the container logs"
fi

cleanup_container

echo
echo "RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
