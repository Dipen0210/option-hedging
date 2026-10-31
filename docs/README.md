# Documentation

| File | What it is |
|---|---|
| [DEPLOYMENT.md](DEPLOYMENT.md) | The deploy runbook: one-time GCP setup, the two deploys, what the Cloud Run configuration is and why, and the known open problems. Start here to deploy. |
| [PRD.md](PRD.md) | Product requirements, the architecture, and the decision history — including the deviations from the original spec and why they were made. |
| [REPAIR_REPORT.md](REPAIR_REPORT.md) | The audit and repair record. Every defect found, its root cause, the fix, and how the fix was verified. Referenced by section (§F, §E…) from comments throughout the code. |
| [reference/](reference/) | Background material: the derivatives paper the pricing models follow, and the project presentation. |

Nothing in here is read at build or run time — `.dockerignore` and
`.gcloudignore` both exclude this directory, so the ~2MB of PDFs never reach a
build context. Code comments cite these files by path, so moving or renaming
one means updating those citations; `backend/tests/test_deployment_contract.py`
fails if the runbook in particular goes missing.
