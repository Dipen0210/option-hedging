# Python 3.13 matches the interpreter the test suite runs under
# (backend/.venv). The image was on 3.12 while pandas 3.x / numpy 2.x were
# resolved and tested on 3.13, so the container's numerical stack was never the
# one any test exercised.
FROM python:3.13-slim

WORKDIR /app

# Runtime dependencies only — no torch/transformers/test deps.
# Copied on its own so the layer is cached independently of source changes.
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Backend package and trained ML models.
# .dockerignore keeps backend/.env, backend/.venv and backend/tests out of the
# image; without it this line published every API key in an image layer.
COPY backend/ ./backend/

# Fail the build rather than publish a leak.
#
# The exclusions above are the only thing standing between `COPY backend/` and
# an image layer containing backend/.env. They were wrong once already — bare
# .dockerignore patterns match only at the context root, so every rule aimed at
# something inside backend/ silently matched nothing, and the image carried
# five live API keys, the 417MB host virtualenv and a 6.4MB SQLite cache. A
# test asserting the patterns were present in the file could not see it.
#
# This check can: it runs against the copied tree, inside the image, on every
# build. A leak becomes a failed build instead of a published secret.
RUN set -e; \
    bad=""; \
    for p in $(find ./backend \
                 -name '.env' -o -name '.env.*' ! -name '.env.example' \
                 -o -name '*.pem' -o -name '*.key' \
                 -o -name '*.db' -o -name '*.sqlite*' \
                 -o -name '.venv' -o -name 'tests' -type d); do \
      bad="$bad $p"; \
    done; \
    if [ -n "$bad" ]; then \
      echo "REFUSING TO BUILD: .dockerignore let these into the image:" >&2; \
      for p in $bad; do echo "  $p" >&2; done; \
      echo "Patterns inside a subdirectory must be '**/'-anchored." >&2; \
      exit 1; \
    fi

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run unprivileged. Two paths need to be writable by the runtime user: the
# retrain job writes fitted models to backend/ml_models, and the response cache
# needs a database plus a SQLite journal alongside it.
#
# The cache was previously assumed to live next to the code at
# /app/backend/hedgeos_cache.db. That directory is copied as root and is not
# writable by uid 10001, so SQLite could read the database but not commit to
# it — and because `_get_conn` writes on every call, a cache miss raised
# `attempt to write a readonly database` and failed a request whose data had
# already been fetched. A dedicated, owned directory removes the ambiguity;
# backend/data/data_cache.py falls back to a temp path if even this is
# unwritable, so a stricter runtime degrades instead of failing.
ENV OPTIONQ_CACHE_DB=/var/cache/optionq/hedgeos_cache.db
# backend/ml_models is created here rather than assumed. Every file in it is
# gitignored (*.joblib, *.pkl, *_trained_at.txt), so in a clean clone the
# directory does not exist and `COPY backend/` cannot bring it in. The build
# succeeded on the machine where it was first run only because that machine had
# locally trained models sitting there untracked — so the image was never
# buildable from the repository, and `chown` failed the moment CI tried it from
# a fresh checkout. gmm_detector calls os.makedirs(MODEL_DIR, exist_ok=True) at
# import time, which uid 10001 cannot do inside root-owned /app/backend, so the
# directory has to exist and be owned before USER switches.
RUN useradd --create-home --uid 10001 optionq \
 && mkdir -p /var/cache/optionq /app/backend/ml_models \
 && chown -R optionq:optionq /app/backend/ml_models /var/cache/optionq
USER optionq

# ADMIN_API_KEY, CRON_SECRET, FRONTEND_URL and ENVIRONMENT are supplied by the
# platform at deploy time, never baked in. With ENVIRONMENT=production the app
# refuses to start until the two secrets are set and FRONTEND_URL names an
# HTTPS, non-localhost origin.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
