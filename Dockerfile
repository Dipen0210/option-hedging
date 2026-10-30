# Python 3.13 matches the interpreter the test suite runs under
# (backend/.venv). The image was on 3.12 while pandas 3.x / numpy 2.x were
# resolved and tested on 3.13, so the container's numerical stack was never the
# one any test exercised.
FROM python:3.13-slim

WORKDIR /app

# Runtime dependencies only — no torch/transformers/test deps.
# Copied on its own so the layer is cached independently of source changes.
COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Backend package and trained ML models.
# .dockerignore keeps backend/.env, backend/.venv and backend/tests out of the
# image; without it this line published every API key in an image layer.
COPY backend/ ./backend/

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
RUN useradd --create-home --uid 10001 optionq \
 && mkdir -p /var/cache/optionq \
 && chown -R optionq:optionq /app/backend/ml_models /var/cache/optionq
USER optionq

# ADMIN_API_KEY, CRON_SECRET, FRONTEND_URL and ENVIRONMENT are supplied by the
# platform at deploy time, never baked in. With ENVIRONMENT=production the app
# refuses to start until the two secrets are set and FRONTEND_URL names an
# HTTPS, non-localhost origin.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
