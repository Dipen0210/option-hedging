# ml_models

Fitted regime-detection artefacts live here at runtime:

| File | Written by |
|---|---|
| `fast_gmm.joblib`, `fast_scaler.joblib`, `fast_vix_thresholds.joblib` | the 24h fast retrain |
| `main_gmm.joblib`, `main_scaler.joblib` | the monthly main retrain |
| `*_trained_at.txt` | the retrain, to decide staleness |
| `logs/` | `backend/ml/regime/training_log.py` |

**All of them are gitignored, deliberately.** They are fitted to whatever macro
history was available when they were written, so a committed model would be a
stale measurement shipped as a current one, and a binary one at that.

This README is tracked so the *directory* exists in a clean clone, which the
files cannot do. That is not cosmetic. `gmm_detector` calls
`os.makedirs(MODEL_DIR, exist_ok=True)` at import time, and inside the
container `/app/backend` is root-owned while the process runs as uid 10001 — so
the directory must already exist and be owned before `USER` switches. The
Dockerfile `chown`ed this path for exactly that reason while nothing guaranteed
it was there, and the image turned out not to be buildable from the repository
at all: it had only ever been built on a machine that happened to have
untracked models here. CI caught it on its first run. See docs/REPAIR_REPORT.md §F.

An empty directory is the correct state for a fresh deployment.
`GMMRegimeDetector.load()` returns `False` when `model_exists()` is `False`,
and the caller trains on first use (`trigger="startup"`). If the macro series
cannot be aligned into enough observations, it raises `InsufficientRegimeData`
and the pipeline degrades to a **disclosed** default regime rather than a
silent one.
