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
point: a `pytest` run must not require a container runtime, a network, and the
several minutes an image build costs, so the suite checks the only thing it can
check cheaply and on every run — that the configuration is consistent with the
code that has to run against it. A missing pin is fully detectable from here.

What these tests therefore do *not* do is prove the image builds or runs.
Treating a green run of this file as evidence that the container works would be
the same mistake in a new place — and that is not hypothetical: every test here
passed while the image was unbuildable from a clean checkout, because the only
build anyone had run was from a working tree that happened to contain untracked
model files.

That proof now lives in `.github/workflows/ci.yml`, which builds the image for
linux/amd64 and runs `scripts/verify_image.sh` on every push. It is deliberately
not out-of-band any more: when it was, it ran once, on a laptop, with a
container engine installed for the occasion and removed the next day.
"""
import ast
import pathlib
import re
import sys

import pytest
from importlib.metadata import packages_distributions, version, PackageNotFoundError

# Shared with test_security.py, which asserted the same thing about the same
# file and was wrong in the same way. One implementation, calibrated against
# real builds — see the module docstring.
from backend.tests.fixtures.dockerignore import excluded as dockerignore_excluded

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
        #   wrapt         — transitive via arch → statsmodels → formulaic, and
        #                   pinned only to keep a *prerelease* out of the
        #                   runtime closure. See
        #                   test_no_pin_is_a_prerelease for the defect.
        not_imported_on_purpose = {
            "uvicorn", "python-dotenv", "curl-cffi", "wrapt",
        }
        unused = sorted(set(_pinned()) - imported_dists - not_imported_on_purpose)
        assert not unused, f"pinned but never imported by backend/: {unused}"


class TestNoPinIsAPrerelease:
    """
    The runtime closure resolved `wrapt-2.6.0.dev1` — a development build — in
    preference to the 2.5.2 stable release, and installed it on both macOS and
    linux/amd64.

    The cause is not obvious and is worth stating, because it will recur:
    `formulaic` (transitive via arch → statsmodels) declares
    `wrapt>=1.17.0rc1` for Python >= 3.13. A version specifier that *names* a
    prerelease makes pip willing to consider prereleases for that requirement,
    so the resolver was free to take the newest thing in the index — a dev
    build — rather than the newest stable one. Nothing warned.

    A dev prerelease carries no release guarantees and can change behaviour
    between two builds whose requirements files are byte-identical. That is
    unacceptable in the dependency graph of something that prices hedges, and
    it is exactly the kind of drift the rest of this suite exists to catch.
    """

    # PEP 440 pre/dev/post-release suffixes on the final version segment.
    _PRERELEASE = re.compile(r"(a|b|rc|alpha|beta|dev)\d*$|\.dev\d*$", re.I)

    def test_no_pin_is_a_prerelease(self):
        suspect = sorted(
            f"{name}=={ver}" for name, ver in _pinned().items()
            if self._PRERELEASE.search(ver)
        )
        assert not suspect, (
            "prerelease versions pinned for the runtime image: "
            f"{suspect}. A dev or rc build has no release guarantees; pin the "
            "stable release that satisfies the same constraint."
        )

    def test_the_installed_wrapt_is_the_stable_release(self):
        # The pin is only worth having if it is what actually gets installed,
        # and this is the one package where a transitive constraint actively
        # pulls the other way.
        try:
            installed = version("wrapt")
        except PackageNotFoundError:
            pytest.skip("wrapt is not installed in this environment")
        assert not self._PRERELEASE.search(installed), (
            f"wrapt {installed} is a prerelease. formulaic's "
            "`wrapt>=1.17.0rc1` lets pip resolve prereleases, so the pin in "
            "api/requirements.txt is what holds this to a stable build — "
            "reinstall from that file."
        )
        assert installed == _pinned()["wrapt"], (
            f"installed wrapt {installed} does not match the pin "
            f"{_pinned()['wrapt']}"
        )


class TestTheImageDoesNotShipSecretsOrTests:

    # The three tests this class used to open with asserted that the strings
    # ".env", ".venv/" and "*.pem" appeared somewhere in .dockerignore. All
    # three passed. The image built from that file contained backend/.env with
    # five live API keys, the 417MB host virtualenv and a 6.4MB SQLite cache.
    #
    # Checking that a rule is *written* is not checking that it *matches*.
    # .dockerignore is not .gitignore: Git re-applies a bare pattern at every
    # directory level, Docker and Buildah match each pattern once against the
    # whole path relative to the context root. So `.env` excludes `./.env` and
    # nothing else, and every rule aimed at something inside backend/ matched
    # nothing at all. `backend/tests/` was the only one that worked, because it
    # names its path.
    #
    # These tests evaluate the file instead, against the paths that actually
    # leaked, using a matcher calibrated against two real builds: with bare
    # patterns a nested sub/.env survives into the image, with `**/`-anchored
    # patterns it does not.

    def test_the_paths_that_leaked_are_excluded(self):
        for path in (
            "backend/.env",
            "backend/.env.local",
            "backend/.venv/lib/python3.13/site-packages/certifi/cacert.pem",
            "backend/hedgeos_cache.db",
            "backend/data/hedgeos_cache.db-wal",
            "backend/key.pem",
            "backend/secrets/server.key",
            "backend/tests/test_deployment_contract.py",
        ):
            assert dockerignore_excluded(path), (
                f"{path} would be copied into the image. A pattern meant to "
                f"match inside a subdirectory must be '**/'-anchored or name "
                f"its path; a bare pattern matches only at the context root."
            )

    def test_the_root_level_paths_are_still_excluded(self):
        # Anchoring the patterns must not stop them matching at the root.
        for path in (".env", "local.db", "server.key"):
            assert dockerignore_excluded(path), f"{path} at the context root is not excluded"

    def test_the_env_example_survives_the_exclusion(self):
        # It is documentation, and the image is where someone looks to find out
        # which variables the service expects.
        assert not dockerignore_excluded("backend/.env.example"), (
            "the example file is documentation and should survive the exclusion"
        )

    def test_the_backend_package_itself_is_not_excluded(self):
        # An over-broad rule that removed the application would be caught by
        # the import test, but far later and far less legibly.
        for path in (
            "backend/main.py",
            "backend/api/security.py",
            "backend/ml_models/gmm_regime.joblib",
        ):
            assert not dockerignore_excluded(path), f"{path} is excluded from the image"

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


class TestTheImageIsBuildableFromTheRepository:
    """
    The image must build from a clean checkout, not from one machine's leftovers.

    This class exists because CI's very first run failed on:

        chown: cannot access '/app/backend/ml_models': No such file or directory

    Every file in `backend/ml_models/` is gitignored — `*.joblib`, `*.pkl`,
    `*_trained_at.txt` — and nothing else was tracked there, so the directory
    does not exist in a clean clone and `COPY backend/ ./backend/` cannot bring
    it into the image. The Dockerfile then `chown`ed a path that was not there.

    The build had nonetheless "succeeded" when it was run by hand, because that
    machine had locally trained models sitting in the directory untracked. So
    the image that was verified 11/11 and reported as deployment-ready was
    never buildable from the repository at all. Nothing in this suite could see
    it, and neither could a build run from a working tree — only a build from a
    fresh checkout.

    That is defect class ⑥, verified-here-only, in a new disguise: the earlier
    verification was real, and was contingent on untracked local state nobody
    had thought to look at.
    """

    def _dockerfile_mkdir_targets(self) -> set:
        """Absolute paths the Dockerfile creates for itself."""
        made = set()
        for m in re.finditer(r"mkdir\s+(-\w+\s+)*([^\n&|;]+)", DOCKERFILE.read_text()):
            for tok in m.group(2).split():
                if tok.startswith("/"):
                    made.add(tok.rstrip("/"))
        return made

    def _dockerfile_chown_targets(self) -> set:
        """Absolute paths the Dockerfile chowns, which must therefore exist."""
        targets = set()
        for m in re.finditer(r"chown\s+(?:-\w+\s+)*[\w:]+\s+([^\n&|;]+)", DOCKERFILE.read_text()):
            for tok in m.group(1).split():
                if tok.startswith("/"):
                    targets.add(tok.rstrip("/"))
        return targets

    def test_the_dockerfile_chowns_nothing_it_has_not_been_given(self):
        # The exact failure CI caught. A chowned path has to come from
        # somewhere: either the Dockerfile creates it, or a COPY brings it in
        # from a tracked file in the repository. A path that depends on
        # untracked local state is a build that works only on one machine.
        import subprocess

        tracked = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout.split("\0")
        tracked_dirs = set()
        for f in tracked:
            if not f:
                continue
            parts = pathlib.PurePosixPath(f).parts
            for i in range(1, len(parts)):
                tracked_dirs.add("/".join(parts[:i]))

        made = self._dockerfile_mkdir_targets()
        unmet = []
        for target in sorted(self._dockerfile_chown_targets()):
            if target in made or any(target.startswith(m + "/") for m in made):
                continue  # the Dockerfile creates it
            if target.startswith("/app/"):
                repo_rel = target[len("/app/"):]
                if repo_rel in tracked_dirs:
                    continue  # a COPY brings it in from tracked content
                unmet.append(
                    f"{target} — nothing tracked under {repo_rel}/, so it is "
                    f"absent from a clean clone"
                )
            else:
                unmet.append(f"{target} — outside /app and never created")

        assert not unmet, (
            "the Dockerfile chowns paths it is not guaranteed to have:\n  "
            + "\n  ".join(unmet)
        )

    def test_the_ml_models_directory_survives_a_clean_clone(self):
        # The specific regression. Every artefact in here is gitignored on
        # purpose — a committed model is a stale measurement shipped as a
        # current one — so the directory needs at least one tracked file to
        # exist at all.
        import subprocess

        out = subprocess.run(
            ["git", "ls-files", "backend/ml_models"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout.split()
        assert out, (
            "nothing is tracked under backend/ml_models/, so the directory does "
            "not exist in a clean clone. gmm_detector calls os.makedirs at "
            "import time and cannot create it as uid 10001 inside root-owned "
            "/app/backend, and the Dockerfile's chown fails outright."
        )

    def test_no_fitted_model_is_committed(self):
        # The other half of the same contract: the directory must exist, and it
        # must not carry a model. A committed .joblib would be a measurement
        # fitted to whatever macro history existed when it was written, shipped
        # as though it described today.
        import subprocess

        out = subprocess.run(
            ["git", "ls-files", "backend/ml_models"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout.split()
        fitted = [f for f in out if f.endswith((".joblib", ".pkl"))]
        assert not fitted, f"fitted models are committed: {fitted}"
