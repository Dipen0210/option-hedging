"""
Evaluate `.dockerignore` the way Docker and Buildah actually evaluate it.

This exists because two separate test classes — in test_security.py and
test_deployment_contract.py — each asserted that the strings ".env", ".venv/"
and "*.pem" appeared somewhere in .dockerignore. All five of those assertions
passed, and the image built from that file contained backend/.env with five
live API keys, the 417MB host virtualenv and a 6.4MB SQLite cache.

Checking that a rule is *written* is not checking that it *matches*.

.dockerignore is not .gitignore. Git re-applies a bare pattern at every
directory level; Docker and Buildah match each pattern once against the whole
path relative to the context root. So `.env` excludes `./.env` and nothing
else, and every rule aimed at something inside `backend/` matched nothing at
all. `backend/tests/` was the only rule in the file that worked, because it
names its path — which is why the leak looked like a solved problem.

The semantics below were calibrated against real builds rather than read off a
doc page: a context holding `.env`, `sub/.env`, `sub/cache.db` and
`sub/.venv/lib.py` was built once with bare patterns (everything nested
survived into the image) and once with `**/`-anchored ones (nothing did). That
second build is also what establishes that `**/` matches *zero* or more leading
components, so `**/.env` covers a root-level .env as well as a nested one.
"""
import pathlib
import re

DOCKERIGNORE = pathlib.Path(__file__).resolve().parents[3] / ".dockerignore"


def rules() -> list:
    """[(negated, compiled regex)] in file order. Last match wins, as in Docker."""
    out = []
    for line in DOCKERIGNORE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        pattern = (line[1:] if negated else line).strip("/")
        if not pattern:
            continue
        out.append((negated, re.compile(_translate(pattern) + r"(/.*)?$")))
    return out


def _translate(pattern: str) -> str:
    """Glob → regex. `**` spans path separators; `*` and `?` do not."""
    parts, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            # Zero or more leading components, so an anchored pattern still
            # covers the context root.
            parts.append("(.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            parts.append(".*")
            i += 2
        elif pattern[i] == "*":
            parts.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(pattern[i]))
            i += 1
    return "".join(parts)


def excluded(path: str) -> bool:
    """Would Docker leave `path` (relative to the context root) out of the image?"""
    verdict = False
    for negated, rx in rules():
        if rx.fullmatch(path):
            verdict = not negated
    return verdict
