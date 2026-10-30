"""
The deployment contract: what the container installs, copies and runs.

These are regression tests for a failure mode that has now bitten this project
four times in a row. Every one of the last four commits before this suite
existed was a production-only crash of the same shape:

    e94f2e6  browser user-agent session to bypass cloud IP blocking
    18bfaff  add Dockerfile for Cloud Run
    b59d1e2  add apscheduler, pandas-datareader, feedparser to Cloud Run reqs
    0454bb7  bypass the yfinance wrapper with direct chart API calls

The third is the clearest case: three modules the backend imported were absent
from the image's requirements, so the container started, served `/health`, and
then raised ImportError on the first real request. Nothing in the test suite
could see it, because the tests install `backend/requirements.txt` while the
image installs `api/requirements.txt`, and nothing compared the two to the
code that has to run against them.

So these tests deliberately assert against the *files that configure the
deployment* rather than against behaviour. That is unusual, and it is the
point: the behaviour they protect cannot be reproduced on this machine, where
Docker is not installed. A missing pin is still fully detectable from here.
"""
import ast
import pathlib
import re
import sys

import pytest
from importlib.metadata import packages_distributions, version, PackageNotFoundError

REPO = pathlib.Path(__file__).resolve().parents[2]
BACKEND = REPO / "backend"
RUNTIME_REQS = REPO / "api" / "requirements.txt"
DOCKERFILE = REPO / "Dockerfile"
DOCKERIGNORE = REPO / ".dockerignore"

# The virtualenv lives under backend/, so a naive rglob walks 300 site-packages
# modules and reports half of PyPI as unpinned.
EXCLUDE_PARTS = {"tests", ".venv", "venv", "__pycache__", "ml_models"}


def _pinned() -> dict:
    """{distribution-name: pinned version} from the runtime requirements."""
    pins = {}
    for line in RUNTIME_REQS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]+\])?==(.+)$", line)
        assert m, f"unpinned or unparseable requirement: {line!r}"
        pins[m.group(1).lower().replace("_", "-")] = m.group(3).strip()
    return pins


def _backend_imports() -> dict:
    """
    {top-level module: {files importing it}} for everything backend/ imports.

    Walks the AST rather than grepping, so a function-local or conditional
    import — the kind added to keep startup fast, and the kind most likely to
    be missed by a requirements file — is caught too.
    """
    found: dict = {}
    for f in BACKEND.rglob("*.py"):
        if EXCLUDE_PARTS & set(f.parts):
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    found.setdefault(a.name.split(".")[0], set()).add(f)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.setdefault(node.module.split(".")[0], set()).add(f)

    stdlib = set(sys.stdlib_module_names)
    return {m: v for m, v in found.items() if m not in stdlib and m != "backend"}


class TestTheImageInstallsWhatTheCodeImports:

    def test_every_third_party_import_is_pinned_for_the_container(self):
        pins = _pinned()
        provided = packages_distributions()

        unsatisfied = []
        for mod, files in sorted(_backend_imports().items()):
            dists = {d.lower().replace("_", "-") for d in provided.get(mod, [])}
            if not dists:
                unsatisfied.append(f"{mod} (no installed distribution provides it) "
                                   f"— imported by {sorted(str(f.relative_to(REPO)) for f in files)[0]}")
            elif not (dists & pins.keys()):
                unsatisfied.append(f"{mod} (provided by {sorted(dists)}, none pinned) "
                                   f"— imported by {sorted(str(f.relative_to(REPO)) for f in files)[0]}")

        assert not unsatisfied, (
            "api/requirements.txt does not install these, so the container "
            "would raise ImportError on the request that first touches them:\n  "
            + "\n  ".join(unsatisfied)
        )

    def test_the_scan_actually_sees_the_backend(self):
        # A guard on the guard: if EXCLUDE_PARTS or the walk root ever stops
        # matching the tree, the test above passes by examining nothing. The
        # core numerical and web dependencies must show up.
        mods = _backend_imports()
        for expected in ("fastapi", "pandas", "numpy", "yfinance", "pydantic"):
            assert expected in mods, (
                f"{expected} not seen by the import scan — the scan is broken, "
                f"not the requirements file"
            )

    def test_pins_match_the_versions_the_suite_ran_against(self):
        # A pin the tests never exercised is a guess. This is how the image
        # came to run pandas 2.2.3 while every test ran on 3.0.2.
        drift = []
        for name, pin in sorted(_pinned().items()):
            try:
                have = version(name)
            except PackageNotFoundError:
                drift.append(f"{name}: pinned {pin}, not installed here")
                continue
            if have != pin:
                drift.append(f"{name}: pinned {pin}, tested against {have}")
        assert not drift, (
            "the container would run libraries no test exercised:\n  "
            + "\n  ".join(drift)
        )

    def test_nothing_unused_is_pinned(self):
        # Not a crash risk, but every extra pin is attack surface and image
        # weight, and a stale pin is how `feedparser` outlived its last import.
        provided = packages_distributions()
        imported_dists = set()
        for mod in _backend_imports():
            imported_dists |= {d.lower().replace("_", "-") for d in provided.get(mod, [])}

        # Pinned deliberately despite no direct import, each for a reason
        # that would be invisible to an import scan:
        #   uvicorn       — the entrypoint, invoked by the Dockerfile's CMD
        #   python-dotenv — backs pydantic-settings' env_file loading
        #   curl-cffi     — yfinance's transport; a version mismatch there
        #                   costs every quote, so it is not left floating
        not_imported_on_purpose = {"uvicorn", "python-dotenv", "curl-cffi"}
        unused = sorted(set(_pinned()) - imported_dists - not_imported_on_purpose)
        assert not unused, f"pinned but never imported by backend/: {unused}"


class TestTheImageDoesNotShipSecretsOrTests:

    def test_dockerignore_excludes_the_env_file(self):
        # `COPY backend/ ./backend/` copies backend/.env unless excluded, and
        # an image layer is readable by anyone who can pull the image.
        patterns = {
            l.strip() for l in DOCKERIGNORE.read_text().splitlines()
            if l.strip() and not l.strip().startswith("#")
        }
        assert ".env" in patterns
        assert ".env.*" in patterns
        assert "!.env.example" in patterns, (
            "the example file is documentation and should survive the exclusion"
        )

    def test_dockerignore_excludes_the_local_virtualenv(self):
        patterns = {l.strip() for l in DOCKERIGNORE.read_text().splitlines()}
        assert ".venv/" in patterns, (
            "copying the host virtualenv shadows the versions pip just "
            "resolved inside the image"
        )

    def test_dockerignore_excludes_private_keys(self):
        patterns = {l.strip() for l in DOCKERIGNORE.read_text().splitlines()}
        assert "*.pem" in patterns and "*.key" in patterns

    def test_the_env_example_holds_no_real_values(self):
        example = REPO / "backend" / ".env.example"
        if not example.exists():
            pytest.skip(".env.example not present")
        for line in example.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip().strip('"').strip("'")
            assert not re.match(r"^sk-ant-|^hf_[A-Za-z0-9]{20,}|^AKIA[0-9A-Z]{16}", value), (
                f"{key} in .env.example looks like a real credential"
            )


class TestTheContainerRunsSafely:

    def test_the_image_does_not_run_as_root(self):
        body = DOCKERFILE.read_text()
        assert re.search(r"^USER\s+(?!root\b)\S+", body, re.M), (
            "no non-root USER directive — a compromised process would run as "
            "root inside the container"
        )

    def test_the_entrypoint_honours_the_platform_port(self):
        # Cloud Run assigns $PORT and health-checks it; a hardcoded port makes
        # every revision fail its readiness probe.
        body = DOCKERFILE.read_text()
        cmd = [l for l in body.splitlines() if l.startswith("CMD")]
        assert cmd, "no CMD in the Dockerfile"
        assert "PORT" in cmd[-1], f"CMD ignores $PORT: {cmd[-1]}"
        assert "0.0.0.0" in cmd[-1], (
            "binding to localhost inside a container makes the service "
            "unreachable from outside it"
        )

    def test_the_image_installs_the_runtime_requirements_not_the_dev_ones(self):
        body = DOCKERFILE.read_text()
        assert "api/requirements.txt" in body
        assert "backend/requirements.txt" not in body, (
            "backend/requirements.txt pulls in test and training dependencies "
            "that have no place in a production image"
        )

    def test_the_interpreter_matches_the_one_the_suite_runs_under(self):
        # The image was on 3.12 while the numerical stack was resolved and
        # tested on 3.13, so the container's numpy/pandas build was never the
        # one any test exercised.
        body = DOCKERFILE.read_text()
        m = re.search(r"^FROM\s+python:(\d+)\.(\d+)", body, re.M)
        assert m, "cannot determine the image's Python version"
        assert (int(m.group(1)), int(m.group(2))) == sys.version_info[:2], (
            f"image runs Python {m.group(1)}.{m.group(2)}, tests run "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )
