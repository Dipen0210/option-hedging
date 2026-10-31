#!/usr/bin/env bash
#
# Deploy the OptionQ backend to Cloud Run.
#
# PRINTS THE PLAN AND EXITS unless given --execute. That is deliberate: this
# script is the record of how the service is configured, and reading it must
# never be the same action as deploying it.
#
# Usage:
#   scripts/deploy_backend.sh                  # print the plan, change nothing
#   scripts/deploy_backend.sh --execute        # actually deploy
#
# Required (no defaults — a guessed project or origin is worse than a refusal):
#   PROJECT_ID     GCP project that holds the service and the Vertex quota
#   FRONTEND_URL   exact HTTPS origin(s) of the deployed frontend, comma-
#                  separated. The API serves credentialed CORS and therefore
#                  cannot answer a wildcard; an origin missing from this list
#                  is a browser preflight failure, not a warning.
#
# Optional:
#   REGION         default us-central1
#   SERVICE        default optionq-api
#   SERVICE_ACCOUNT  default optionq-api@$PROJECT_ID.iam.gserviceaccount.com
#
# Prerequisites, once per project (see docs/DEPLOYMENT.md for the exact commands):
#   - run.googleapis.com, cloudbuild.googleapis.com, aiplatform.googleapis.com,
#     secretmanager.googleapis.com enabled
#   - secrets ADMIN_API_KEY and CRON_SECRET created in Secret Manager
#   - a service account holding roles/aiplatform.user and
#     roles/secretmanager.secretAccessor on those two secrets

set -euo pipefail

EXECUTE=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-optionq-api}"
PROJECT_ID="${PROJECT_ID:-}"
FRONTEND_URL="${FRONTEND_URL:-}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-optionq-api@${PROJECT_ID}.iam.gserviceaccount.com}"

# ── Resource shape ───────────────────────────────────────────────────────────
#
# CPU 2 / 2Gi: the pipeline is CPU-bound — Monte Carlo paths, GARCH fitting and
# a GMM fit on first use — and it runs its own ThreadPoolExecutor, which needs
# more than one core to be anything other than overhead. 2Gi because the
# response cache is a SQLite file on a tmpfs, and tmpfs counts against the
# instance's memory.
CPU=2
MEMORY=2Gi

# Concurrency 8, against Cloud Run's default of 80. Eighty concurrent full
# pipelines on two cores would not be served slowly, they would time out, and
# the platform would report a healthy instance throughout.
CONCURRENCY=8

# 300s, and it must stay above the frontend's own analyze budget of 240s
# (TIMEOUT_MS.analyze in frontend/lib/api.ts). If the platform cuts the request
# first, the user sees a connection drop instead of the explanation the client
# is written to show, and the run is lost after the market data was already
# fetched. backend/tests/test_deployment_contract.py asserts the ordering.
TIMEOUT=300

# ── Instance count: pinned to exactly one, on purpose ────────────────────────
#
# This is the interim posture for two unresolved problems, not a performance
# choice:
#
#   1. Fitted regime models live in the container's writable layer
#      (backend/ml_models). They do not survive a cold start, and two
#      instances can hold models fitted at different times — so the same
#      portfolio could be told it is in two different volatility regimes
#      depending on which instance answered. That is a financial-consistency
#      defect, not a cache miss.
#   2. The response cache is per-instance. N instances means N times the
#      upstream market-data request volume, against a provider that already
#      rate-limits.
#
# One instance makes both impossible rather than unlikely. It also caps
# throughput at CONCURRENCY requests, which is the trade being made knowingly.
# Lifting it requires moving models to GCS and the cache to a shared store;
# until then, raising max-instances reintroduces both problems silently.
MIN_INSTANCES=1
MAX_INSTANCES=1

missing=()
[ -n "$PROJECT_ID" ]   || missing+=("PROJECT_ID")
[ -n "$FRONTEND_URL" ] || missing+=("FRONTEND_URL")
if [ ${#missing[@]} -gt 0 ]; then
  echo "Refusing to deploy: ${missing[*]} not set." >&2
  echo "See the header of this script, or docs/DEPLOYMENT.md." >&2
  exit 2
fi

# The backend makes the same checks at startup and refuses to boot in
# production if they fail. Making them here too turns a failed revision into a
# failed command, which is a much shorter feedback loop.
case "$FRONTEND_URL" in
  *localhost*|*127.0.0.1*)
    echo "Refusing to deploy: FRONTEND_URL names localhost (${FRONTEND_URL})." >&2
    echo "The service would refuse to start; see backend/api/security.py." >&2
    exit 2 ;;
esac
for origin in ${FRONTEND_URL//,/ }; do
  case "$origin" in
    https://*) ;;
    *) echo "Refusing to deploy: origin '${origin}' is not HTTPS." >&2; exit 2 ;;
  esac
done

ARGS=(
  run deploy "$SERVICE"
  --source .
  --project "$PROJECT_ID"
  --region "$REGION"
  --platform managed

  # Public, because the client is a browser on another origin and cannot
  # present a Google identity token. The /admin/* surface is guarded
  # separately by ADMIN_API_KEY and CRON_SECRET, which answer 503 rather than
  # serving themselves unauthenticated when unset.
  --allow-unauthenticated

  --service-account "$SERVICE_ACCOUNT"
  --cpu "$CPU"
  --memory "$MEMORY"
  --concurrency "$CONCURRENCY"
  --timeout "$TIMEOUT"
  --min-instances "$MIN_INSTANCES"
  --max-instances "$MAX_INSTANCES"

  # Vertex needs no key: the attached service account's token is minted by
  # google-auth from the metadata server. VERTEX_LOCATION stays `global` —
  # every regional endpoint answers the 3.x models with 404.
  --set-env-vars "ENVIRONMENT=production,LOG_LEVEL=INFO,LLM_PROVIDER=vertex,VERTEX_PROJECT=${PROJECT_ID},VERTEX_LOCATION=global,FRONTEND_URL=${FRONTEND_URL}"

  # Mounted from Secret Manager, never passed on this command line and never
  # baked into the image.
  --set-secrets "ADMIN_API_KEY=ADMIN_API_KEY:latest,CRON_SECRET=CRON_SECRET:latest"
)

echo "Service:         $SERVICE"
echo "Project/region:  $PROJECT_ID / $REGION"
echo "Service account: $SERVICE_ACCOUNT"
echo "Shape:           cpu=$CPU memory=$MEMORY concurrency=$CONCURRENCY timeout=${TIMEOUT}s"
echo "Instances:       min=$MIN_INSTANCES max=$MAX_INSTANCES  (pinned — see the comment in this script)"
echo "CORS origins:    $FRONTEND_URL"
echo
echo "gcloud ${ARGS[*]}"
echo

if [ "$EXECUTE" -ne 1 ]; then
  cat <<'EOF'
Nothing was deployed. This was a plan.

Re-run with --execute to apply it. Before the first deploy, confirm the
prerequisites in docs/DEPLOYMENT.md are in place — the service refuses to start if
ADMIN_API_KEY or CRON_SECRET is unset or FRONTEND_URL is not an HTTPS,
non-localhost origin, so a missing prerequisite surfaces as a revision that
never becomes healthy.
EOF
  exit 0
fi

command -v gcloud >/dev/null 2>&1 || { echo "gcloud is not installed" >&2; exit 1; }
exec gcloud "${ARGS[@]}"
