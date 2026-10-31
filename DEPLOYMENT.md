# Deploying OptionQ

Two deployments, one repository:

| Half | Host | Built from | Entry point |
|---|---|---|---|
| Backend API | Cloud Run (container) | `Dockerfile` + `backend/requirements.txt` | `backend.main:app` on `$PORT` |
| Frontend | Vercel (static files) | `vercel.json` → `frontend/out` | `frontend/app/page.tsx` |

The frontend is a **static export** (`output: "export"` in `frontend/next.config.ts`).
There is no Next.js server and no serverless function anywhere in this
repository — a serverless backend was tried and removed, because the runtime
closure is 370MB against Vercel's 250MB function limit and a function cannot
host the retrain scheduler. See commit `17258a4`.

Because the export is static, **`NEXT_PUBLIC_API_URL` is baked into the
JavaScript at build time.** Changing the backend URL means a rebuild and
redeploy of the frontend, not a restart.

---

## Before the first deploy

### 1. Backend prerequisites (once per GCP project)

```bash
PROJECT_ID=<your-project>

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  secretmanager.googleapis.com \
  --project "$PROJECT_ID"
```

Two admin credentials. The service **refuses to start** in production without
them — it does not fall back to serving `/admin/*` unauthenticated:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))" \
  | gcloud secrets create ADMIN_API_KEY --data-file=- --project "$PROJECT_ID"
python -c "import secrets; print(secrets.token_urlsafe(32))" \
  | gcloud secrets create CRON_SECRET  --data-file=- --project "$PROJECT_ID"
```

A service account. **Vertex AI needs no API key** — the attached account *is*
the credential, minted from the metadata server by `google-auth`. This is the
main reason the backend belongs on Cloud Run rather than anywhere else:

```bash
gcloud iam service-accounts create optionq-api --project "$PROJECT_ID"
SA="optionq-api@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member "serviceAccount:$SA" --role roles/aiplatform.user

for s in ADMIN_API_KEY CRON_SECRET; do
  gcloud secrets add-iam-policy-binding "$s" \
    --member "serviceAccount:$SA" \
    --role roles/secretmanager.secretAccessor --project "$PROJECT_ID"
done
```

Billing must be enabled on the project: **Vertex AI has no free tier.**

### 2. Frontend prerequisites

In the Vercel project settings, add an environment variable for each
environment you intend to build:

```
NEXT_PUBLIC_API_URL = https://<service>-<hash>.<region>.run.app
```

Leave the build settings alone — `vercel.json` sets the install command, build
command and output directory, and installs with `npm ci` so the deployed
bundle is built from the same lockfile CI verified.

---

## Deploying

The two halves depend on each other's URLs, so the first deploy is a loop:
deploy the backend to learn its URL, build the frontend against it, then tell
the backend the frontend's origin.

### Backend

```bash
PROJECT_ID=<your-project> \
FRONTEND_URL=https://<your-app>.vercel.app \
  scripts/deploy_backend.sh            # prints the plan, deploys nothing
```

Read the plan, then:

```bash
PROJECT_ID=... FRONTEND_URL=... scripts/deploy_backend.sh --execute
```

The script refuses, before calling `gcloud`, if `PROJECT_ID` or
`FRONTEND_URL` is unset, if the origin is `localhost`, or if it is not HTTPS —
all three are conditions under which the service would deploy and then fail to
start. It has no default project and no default origin on purpose.

On the very first run you do not yet know the Vercel domain. Deploy with a
placeholder you control, or deploy the frontend first and take its production
domain from Vercel.

### Frontend

Push the branch, or `vercel --prod`. Then confirm the backend knows the real
origin — a mismatch here is a browser preflight failure, not a server error,
so it shows up as an apparently dead "Analyze" button:

```bash
gcloud run services update optionq-api \
  --region us-central1 --project "$PROJECT_ID" \
  --update-env-vars "FRONTEND_URL=https://<your-app>.vercel.app"
```

### Scheduled retrains (optional)

A long-lived instance runs these in-process via APScheduler
(`backend/ml/regime/scheduler.py`), so this is only needed if the service is
ever allowed to scale to zero:

```bash
CRON=$(gcloud secrets versions access latest --secret CRON_SECRET --project "$PROJECT_ID")
URL=$(gcloud run services describe optionq-api --region us-central1 \
        --project "$PROJECT_ID" --format 'value(status.url)')

gcloud scheduler jobs create http optionq-retrain-fast \
  --schedule "0 2 * * *" --uri "$URL/admin/cron/retrain-fast" \
  --http-method GET --headers "Authorization=Bearer $CRON" \
  --location us-central1 --project "$PROJECT_ID"
```

---

## What the deployed configuration is, and why

Every value below is set in `scripts/deploy_backend.sh` and the ones that are
contracts with code elsewhere are pinned by
`backend/tests/test_deployment_contract.py`, so they cannot drift silently.

| Setting | Value | Why |
|---|---|---|
| `--cpu` | 2 | The pipeline runs its own `ThreadPoolExecutor` over pricing and simulation. On one vCPU that pool is pure overhead. |
| `--memory` | 2Gi | The response cache is SQLite on a tmpfs, and tmpfs counts against instance memory. |
| `--concurrency` | 8 | Cloud Run's default is 80. Eighty concurrent pipelines on two cores do not run slowly, they time out, while the platform reports a healthy instance throughout. |
| `--timeout` | 300s | Must exceed the frontend's own 240s analyze budget (`TIMEOUT_MS.analyze` in `frontend/lib/api.ts`). If the platform cuts first, the user sees a dropped connection instead of the written explanation, and the run is lost after the data was already fetched. |
| `--min-instances` | 1 | Avoids a cold start that imports scipy/sklearn/pandas and may fit a GMM on first use. One instance is then billed continuously whether or not anyone uses it — price it with the Cloud Run calculator before committing. |
| `--max-instances` | **1** | **Not a performance setting — see below.** |
| `--allow-unauthenticated` | yes | The client is a browser on another origin and cannot present a Google identity token. `/admin/*` is guarded separately by `ADMIN_API_KEY`, which answers 503 rather than serving itself unauthenticated when unset. |
| `ENVIRONMENT` | `production` | Flips the config guard from *warn* to *refuse to start*. |

`.gcloudignore` is a security control, not a size optimisation:
`gcloud run deploy --source .` writes the build context to a Cloud Storage
staging bucket **before** the image is built, so the Dockerfile's secret-leak
guard is one step too late to keep `backend/.env` out of that bucket. Note that
it uses **gitignore** semantics, where a bare pattern matches at every
directory level — the opposite of `.dockerignore` in this repository, which is
`**/`-anchored throughout. Confusing the two is what once shipped five live API
keys in an image layer (`REPAIR_REPORT.md` §F). Do not copy patterns between
the files.

---

## Known open problems

These are real and unresolved. The configuration above contains them rather
than fixing them.

**1. Fitted models and the response cache are per-instance.**
`backend/ml_models` is in the container's writable layer, so fitted regime
models do not survive a cold start, and two instances can hold models fitted at
different times — meaning the same portfolio could be told it is in two
different volatility regimes depending on which instance answered. That is a
financial-consistency defect, not a cache miss. The response cache has the same
shape, and N instances also means N× the market-data request volume against a
provider that already rate-limits.

`--max-instances 1` makes both impossible rather than unlikely, at the cost of
capping throughput at 8 concurrent requests. **Raising it reintroduces both
problems silently**, which is why a test asserts the value. The fix is a GCS
FUSE volume mount for `ml_models` (Cloud Run gen2) and a shared cache.

**2. Yahoo Finance versus datacentre IPs — untested from Cloud Run.**
All market data comes from Yahoo via `yfinance`. Yahoo rate-limits and
sometimes blocks cloud egress ranges. This project's own CI gates its live
container test behind a manual flag for exactly this reason
(`.github/workflows/ci.yml`), and it has already been worked around once with a
browser user-agent session and direct chart-API calls. **It works from a laptop;
that says nothing about GCP's ranges.** If Yahoo refuses Cloud Run, no setting
here fixes it — it needs a paid data provider. This is the single thing most
worth testing immediately after the first deploy.

**3. Vercel preview deployments will fail CORS.**
Every preview gets its own origin, and `FRONTEND_URL` names exact origins with
no wildcard (a test forbids `*`, and credentialed CORS cannot be served to a
wildcard anyway). Point previews at a separate backend, or add the specific
preview origin while it is in use.

---

## Verifying a deploy

```bash
URL=$(gcloud run services describe optionq-api --region us-central1 \
        --project "$PROJECT_ID" --format 'value(status.url)')

curl -fsS "$URL/health"                      # expect {"status":"ok","version":...}
curl -fsS -o /dev/null -w '%{http_code}\n' \
  "$URL/admin/training/status"               # expect 401/403, never 200

curl -fsS -X POST "$URL/portfolio/analyze" \
  -H 'Content-Type: application/json' \
  -d '{"holdings":[{"ticker":"AAPL","shares":100,"purchase_price":150,
        "purchase_date":"2024-01-15"}],"max_hedge_cost_pct":0.02}' \
  | head -c 2000
```

The third call is the one that matters: it is the only check that proves the
market-data provider answers from inside GCP. In the response, confirm
`data_quality` reports live rather than stale sources, and that
`llm_provider` is `vertex` with an empty `llm_notice` — a notice means the
model was not reached and the narrative is deterministic text.

To check the image locally before deploying, `scripts/verify_image.sh --build`
builds for linux/amd64 and runs the container through its import, startup-refusal
and auth checks. CI runs it on every push.

**OptionQ has no trade execution path.** It produces hedge suggestions. No
broker is configured, and `test_security.py::TestNoTradeExecutionPath` asserts
that stays true.
