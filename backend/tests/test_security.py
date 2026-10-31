"""
Security regression tests for the API surface.

Every test here pins a defect that was live in the deployed app:

  * `/admin/retrain/main` — a blocking, CPU-bound model retrain — had no
    authentication at all.
  * `_verify_cron_secret` returned early when `CRON_SECRET` was unset
    ("no secret configured — open access"), so a deployment that forgot the
    secret exposed model retraining to anyone who could reach the host.
  * The secret was compared with `!=`, which short-circuits on the first
    differing byte and leaks its prefix through response timing.
  * Nine handlers raised `HTTPException(status_code=500, detail=str(exc))`,
    forwarding file paths, driver errors and provider URLs to the client.

The app is exercised through `TestClient`, so these assert the behaviour of the
real router stack and dependency graph, not of the helper functions in
isolation.
"""

import pytest
from fastapi.testclient import TestClient

from backend.tests.fixtures.dockerignore import excluded as dockerignore_excluded
from backend.api import security
from backend.api.security import (
    ConfigurationError, MIN_SECRET_LENGTH, _bearer, _matches,
    internal_error, require_admin, require_cron, validate_environment,
)

GOOD_ADMIN = "admin-key-of-sufficient-length"
GOOD_CRON  = "cron-secret-of-sufficient-length"


@pytest.fixture
def client():
    """
    The real application, with the lifespan left unstarted.

    `TestClient(app)` without a context manager does not run the lifespan, so
    neither APScheduler nor the retrain jobs start — the routing and dependency
    wiring is what is under test.
    """
    from backend.main import app
    return TestClient(app)


@pytest.fixture
def no_secrets(monkeypatch):
    """A deployment that configured neither secret."""
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    monkeypatch.delenv("CRON_SECRET", raising=False)
    monkeypatch.setattr(security.settings, "admin_api_key", "", raising=False)
    monkeypatch.setattr(security.settings, "cron_secret", "", raising=False)


@pytest.fixture
def secrets_set(monkeypatch, no_secrets):
    monkeypatch.setenv("ADMIN_API_KEY", GOOD_ADMIN)
    monkeypatch.setenv("CRON_SECRET", GOOD_CRON)


# Every mounted admin route, with a body where the handler needs one. None of
# these may ever be reachable without a credential.
ADMIN_ROUTES = [
    ("get",  "/admin/training/log", None),
    ("get",  "/admin/training/log/main", None),
    ("get",  "/admin/training/status", None),
    ("post", "/admin/retrain/main", None),
    ("post", "/admin/retrain/fast", None),
    ("get",  "/admin/earnings/AAPL", None),
    ("post", "/admin/hedge/check-triggers", {
        "hedge_ticker": "SPY", "strike": 500.0, "option_type": "put",
        "expiry_date": "2027-01-15", "entry_delta": -0.3,
    }),
]
CRON_ROUTES = [
    ("get", "/admin/cron/retrain-fast", None),
    ("get", "/admin/cron/retrain-main", None),
]


class TestFailsClosed:
    """An unconfigured secret must refuse the request, never allow it."""

    @pytest.mark.parametrize("method,path,body", ADMIN_ROUTES + CRON_ROUTES)
    def test_unconfigured_secret_refuses_instead_of_allowing(
        self, client, no_secrets, method, path, body
    ):
        resp = client.request(method.upper(), path, json=body)
        # 503, not 200: the retrain must not run. Explicitly asserting the
        # status is not 2xx is the point — the old code returned the retrain
        # result here.
        assert resp.status_code == 503, (
            f"{method.upper()} {path} answered {resp.status_code} with no "
            f"secret configured — this endpoint fails open"
        )
        assert resp.status_code != 200

    @pytest.mark.parametrize("method,path,body", ADMIN_ROUTES + CRON_ROUTES)
    def test_unconfigured_secret_cannot_be_satisfied_by_any_credential(
        self, client, no_secrets, method, path, body
    ):
        # Presenting a token must not accidentally match an absent secret
        # (e.g. by both sides being empty).
        for token in ("", "Bearer ", "Bearer anything", GOOD_ADMIN):
            resp = client.request(method.upper(), path, json=body, headers={"Authorization": token}
            )
            assert resp.status_code == 503

    def test_secret_shorter_than_the_minimum_is_treated_as_absent(
        self, client, no_secrets, monkeypatch
    ):
        # A two-character key is not a credential. Accepting one would make the
        # fail-closed check cosmetic.
        short = "x" * (MIN_SECRET_LENGTH - 1)
        monkeypatch.setenv("ADMIN_API_KEY", short)
        resp = client.get(
            "/admin/training/status", headers={"Authorization": f"Bearer {short}"}
        )
        assert resp.status_code == 503

    def test_secret_from_dotenv_is_honoured(self, client, no_secrets, monkeypatch):
        # Every other credential in the app loads from backend/.env. If the
        # auth check read only os.environ, a key set in .env would silently
        # authorise nothing and the admin surface would answer 503 forever.
        monkeypatch.setattr(security.settings, "admin_api_key", GOOD_ADMIN,
                            raising=False)
        resp = client.get(
            "/admin/training/status",
            headers={"Authorization": f"Bearer {GOOD_ADMIN}"},
        )
        assert resp.status_code != 503
        assert resp.status_code != 401


class TestAuthentication:
    @pytest.mark.parametrize("method,path,body", ADMIN_ROUTES)
    def test_admin_route_rejects_missing_credential(
        self, client, secrets_set, method, path, body
    ):
        resp = client.request(method.upper(), path, json=body)
        assert resp.status_code == 401

    @pytest.mark.parametrize("method,path,body", ADMIN_ROUTES)
    def test_admin_route_rejects_wrong_credential(
        self, client, secrets_set, method, path, body
    ):
        resp = client.request(method.upper(), path, json=body, headers={"Authorization": "Bearer wrong-but-long-enough"}
        )
        assert resp.status_code == 401

    @pytest.mark.parametrize("header", [
        GOOD_ADMIN,                       # no scheme
        f"Basic {GOOD_ADMIN}",            # wrong scheme
        f"bearer{GOOD_ADMIN}",            # no separator
        "Bearer",                         # scheme only
        "Bearer ",                        # empty token
    ])
    def test_malformed_authorization_header_is_rejected(
        self, client, secrets_set, header
    ):
        resp = client.get("/admin/training/status",
                          headers={"Authorization": header})
        assert resp.status_code == 401

    def test_bearer_scheme_is_case_insensitive(self, client, secrets_set):
        # RFC 7235 makes the scheme token case-insensitive; rejecting "bearer"
        # would be a correctness bug, not a security control.
        resp = client.get("/admin/training/status",
                          headers={"Authorization": f"bEaReR {GOOD_ADMIN}"})
        assert resp.status_code != 401

    def test_valid_admin_credential_is_accepted(self, client, secrets_set):
        resp = client.get("/admin/training/status",
                          headers={"Authorization": f"Bearer {GOOD_ADMIN}"})
        assert resp.status_code not in (401, 403, 503)


class TestLeastPrivilege:
    """The cron credential triggers retrains; it is not an admin key."""

    @pytest.mark.parametrize("method,path,body", ADMIN_ROUTES)
    def test_cron_secret_does_not_unlock_the_admin_surface(
        self, client, secrets_set, method, path, body
    ):
        resp = client.request(method.upper(), path, json=body, headers={"Authorization": f"Bearer {GOOD_CRON}"}
        )
        assert resp.status_code == 401

    @pytest.mark.parametrize("method,path,body", CRON_ROUTES)
    def test_admin_key_does_not_satisfy_the_cron_routes(
        self, client, secrets_set, method, path, body
    ):
        # Not a security hole in itself, but it pins the separation: if the two
        # guards were collapsed into one, this test says so.
        resp = client.request(method.upper(), path, json=body, headers={"Authorization": f"Bearer {GOOD_ADMIN}"}
        )
        assert resp.status_code == 401

    def test_every_admin_route_carries_a_guard(self):
        # A route added later must not be able to ship unauthenticated. The
        # guard lives on the mount, so this asserts the dependency is actually
        # present on each resolved route rather than trusting the mount.
        from backend.main import app
        guard_names = {"require_admin", "require_cron"}
        unguarded = []
        for route in app.routes:
            path = getattr(route, "path", "")
            if not path.startswith("/admin"):
                continue
            deps = {d.call.__name__ for d in route.dependant.dependencies}
            if not (deps & guard_names):
                unguarded.append(path)
        assert unguarded == []

    def test_public_routes_are_not_accidentally_guarded(self, client, no_secrets):
        # The fail-closed change must not take the product offline: analysis
        # and health are public by design.
        assert client.get("/health").status_code == 200


class TestConstantTimeComparison:
    def test_comparison_does_not_short_circuit_on_the_first_byte(self):
        # `_matches` must delegate to secrets.compare_digest. A plain `!=`
        # returns as soon as two bytes differ, so an attacker can recover the
        # secret one character at a time from response timing. Timing itself is
        # too noisy to assert, so this pins the implementation.
        import ast, inspect
        tree = ast.parse(inspect.getsource(_matches).strip())
        fn = tree.body[0]
        if ast.get_docstring(fn):
            fn.body = fn.body[1:]          # the docstring names the old bug
        body = ast.unparse(ast.Module(body=fn.body, type_ignores=[]))
        assert "compare_digest" in body
        for op in ("!=", "=="):
            assert op not in body, f"{op} comparison in a credential check"

    @pytest.mark.parametrize("presented,expected,want", [
        (GOOD_ADMIN, GOOD_ADMIN, True),
        (GOOD_ADMIN + "x", GOOD_ADMIN, False),
        (GOOD_ADMIN[:-1], GOOD_ADMIN, False),
        ("", GOOD_ADMIN, False),
        (None, GOOD_ADMIN, False),
    ])
    def test_match_semantics(self, presented, expected, want):
        assert _matches(presented, expected) is want

    def test_empty_presented_token_never_matches(self):
        # compare_digest("", "") is True, so a blank secret must be filtered
        # out before it reaches the comparison.
        assert _matches(None, "") is False
        assert _matches("", "") is False

    @pytest.mark.parametrize("header,want", [
        ("Bearer abc", "abc"),
        ("bearer abc", "abc"),
        ("Bearer  abc ", "abc"),
        ("Bearer", None),
        ("Bearer ", None),
        ("abc", None),
        ("Basic abc", None),
        (None, None),
        ("", None),
    ])
    def test_bearer_parsing(self, header, want):
        assert _bearer(header) == want


class TestErrorSanitisation:
    def test_internal_exception_text_never_reaches_the_client(self):
        secret_text = (
            "psycopg2.OperationalError: could not connect to "
            "/Users/dipen/Projects/optionQ/backend/data/cache.db password=hunter2"
        )
        exc = RuntimeError(secret_text)
        http_exc = internal_error(exc, "options analysis")

        assert http_exc.status_code == 500
        detail = str(http_exc.detail)
        for leaked in ("psycopg2", "/Users/dipen", "cache.db", "hunter2",
                       "OperationalError"):
            assert leaked not in detail
        assert "options analysis" in detail
        assert "Reference:" in detail

    def test_each_error_carries_a_distinct_reference_id(self):
        a = str(internal_error(RuntimeError("x"), "ctx").detail)
        b = str(internal_error(RuntimeError("x"), "ctx").detail)
        assert a != b, "reference id must identify the individual failure"

    def test_reference_id_is_logged_so_support_can_find_the_trace(self, caplog):
        import logging
        with caplog.at_level(logging.ERROR, logger="backend.api.security"):
            detail = str(internal_error(RuntimeError("boom"), "ctx").detail)
        error_id = detail.rsplit("Reference: ", 1)[1].strip()
        assert error_id in caplog.text
        # The real exception text belongs in the log, not the response.
        assert "boom" in caplog.text

    def test_handlers_do_not_return_raw_exception_text(self):
        # Pins the whole route layer: a new handler added with
        # `detail=str(exc)` fails here rather than in production.
        import pathlib
        api_dir = pathlib.Path(__file__).resolve().parents[1] / "api"
        offenders = [
            f"{p.name}:{i}"
            for p in sorted(api_dir.glob("routes_*.py"))
            for i, line in enumerate(p.read_text().splitlines(), 1)
            if "detail=str(exc)" in line and "dq.record" not in line
        ]
        assert offenders == [], f"raw exception text returned at {offenders}"

    def test_a_verbatim_message_requires_a_declared_safe_exception(self):
        # The only sanctioned path to a verbatim detail refuses an exception
        # that has not been declared user-facing, so a library message cannot
        # be forwarded even by a handler that tries to.
        from backend.api.security import user_facing_error, UserFacingError

        class Safe(UserFacingError, ValueError):
            pass

        assert user_facing_error(Safe("priced nothing"), 422).detail == "priced nothing"
        with pytest.raises(TypeError):
            user_facing_error(ValueError("internal pandas detail"), 422)


class TestStartupValidation:
    def test_development_reports_gaps_without_blocking_startup(
        self, no_secrets, monkeypatch
    ):
        monkeypatch.setattr(security.settings, "environment", "development")
        problems = validate_environment()
        assert any("ADMIN_API_KEY" in p for p in problems)
        assert any("CRON_SECRET" in p for p in problems)

    def test_production_refuses_to_start_without_admin_key(
        self, no_secrets, monkeypatch
    ):
        monkeypatch.setattr(security.settings, "environment", "production")
        monkeypatch.setattr(security.settings, "frontend_url",
                            "https://optionq.example.com")
        with pytest.raises(ConfigurationError) as err:
            validate_environment()
        assert "ADMIN_API_KEY" in str(err.value)

    def test_production_rejects_a_localhost_cors_origin(
        self, secrets_set, monkeypatch
    ):
        monkeypatch.setattr(security.settings, "environment", "production")
        monkeypatch.setattr(security.settings, "frontend_url",
                            "http://localhost:3000")
        with pytest.raises(ConfigurationError) as err:
            validate_environment()
        assert "localhost" in str(err.value)

    def test_production_rejects_plain_http_origin_with_credentials(
        self, secrets_set, monkeypatch
    ):
        monkeypatch.setattr(security.settings, "environment", "production")
        monkeypatch.setattr(security.settings, "frontend_url",
                            "http://optionq.example.com")
        with pytest.raises(ConfigurationError) as err:
            validate_environment()
        assert "HTTPS" in str(err.value)

    def test_fully_configured_production_passes(self, secrets_set, monkeypatch):
        monkeypatch.setattr(security.settings, "environment", "production")
        monkeypatch.setattr(security.settings, "frontend_url",
                            "https://optionq.example.com")
        assert validate_environment() == []

    def test_empty_frontend_url_is_a_problem(self, secrets_set, monkeypatch):
        monkeypatch.setattr(security.settings, "environment", "development")
        monkeypatch.setattr(security.settings, "frontend_url", "")
        assert any("FRONTEND_URL" in p for p in validate_environment())


class TestCorsPolicy:
    def test_credentialed_cors_is_not_wildcarded(self):
        # allow_credentials=True with a wildcard origin is both rejected by
        # browsers and a policy error; and "*" methods/headers let any page on
        # an allowed origin replay a credential against any verb.
        from backend.main import app
        cors = [m for m in app.user_middleware
                if m.cls.__name__ == "CORSMiddleware"]
        assert len(cors) == 1
        opts = cors[0].kwargs
        assert "*" not in opts["allow_origins"]
        assert "*" not in opts["allow_methods"]
        assert "*" not in opts["allow_headers"]
        assert set(opts["allow_methods"]) <= {"GET", "POST", "OPTIONS"}

    def test_disallowed_origin_gets_no_access_control_header(self, client):
        resp = client.get(
            "/health", headers={"Origin": "https://evil.example.com"}
        )
        assert "access-control-allow-origin" not in {
            k.lower() for k in resp.headers
        }


class TestSecretsNotExposed:
    def test_openapi_schema_does_not_contain_secret_values(
        self, client, secrets_set
    ):
        body = client.get("/openapi.json").text
        assert GOOD_ADMIN not in body
        assert GOOD_CRON not in body

    def test_frontend_source_contains_no_secret_names(self):
        # A secret referenced in frontend code is a published secret: Next.js
        # inlines anything it can reach into the client bundle.
        import pathlib, re
        root = pathlib.Path(__file__).resolve().parents[2] / "frontend"
        if not root.exists():
            pytest.skip("frontend not present")
        banned = re.compile(
            r"ADMIN_API_KEY|CRON_SECRET|ANTHROPIC_API_KEY|ALPACA_SECRET_KEY"
            r"|HF_API_TOKEN|FRED_API_KEY"
        )
        hits = []
        for path in root.rglob("*"):
            if path.is_dir() or "node_modules" in path.parts or ".next" in path.parts:
                continue
            if path.suffix not in (".ts", ".tsx", ".js", ".jsx", ".json", ".mjs"):
                continue
            for i, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
                if banned.search(line):
                    hits.append(f"{path.relative_to(root)}:{i}")
        assert hits == [], f"secret names referenced in frontend code: {hits}"


class TestDeploymentConfiguration:
    """
    The published container is part of the attack surface.

    `COPY backend/ ./backend/` with no .dockerignore copied `backend/.env` —
    every API key — into an image layer, where it stays readable to anyone who
    can pull the image, and dragged the 412MB host virtualenv along with it.
    """

    @staticmethod
    def _repo_root():
        import pathlib
        return pathlib.Path(__file__).resolve().parents[2]

    def test_dockerignore_exists(self):
        path = self._repo_root() / ".dockerignore"
        assert path.exists(), (
            "no .dockerignore: `COPY backend/` would publish backend/.env"
        )

    # These two tests used to assert that the strings ".env", "*.pem", "*.key"
    # and ".venv/" appeared in .dockerignore. They passed, and the image built
    # from that file contained backend/.env with five live API keys. A bare
    # .dockerignore pattern matches only at the context root, so every rule
    # aimed inside backend/ matched nothing. See
    # backend/tests/fixtures/dockerignore.py — the matcher there is calibrated
    # against real builds, and these now check what the rules *do*.

    def test_dockerignore_excludes_env_files_inside_the_backend_package(self):
        assert dockerignore_excluded("backend/.env"), (
            "backend/.env would be copied into an image layer, where it stays "
            "readable to anyone who can pull the image"
        )
        assert dockerignore_excluded("backend/.env.local")
        assert dockerignore_excluded(".env")
        # The template carries no values and documents the required vars.
        assert not dockerignore_excluded("backend/.env.example")

    def test_dockerignore_excludes_private_keys_and_the_virtualenv(self):
        assert dockerignore_excluded("backend/server.pem")
        assert dockerignore_excluded("backend/secrets/server.key")
        assert dockerignore_excluded(
            "backend/.venv/lib/python3.13/site-packages/certifi/cacert.pem"
        ), "the host virtualenv shadows the versions pip resolved in the image"

    def test_dockerfile_installs_the_tested_runtime_requirements(self):
        text = (self._repo_root() / "Dockerfile").read_text()
        assert "api/requirements.txt" in text
        # The heavy unused stack must not come back into the image. Comments
        # are exempt: the Dockerfile names these to record why they are gone.
        instructions = "\n".join(
            l.lower() for l in text.splitlines() if not l.strip().startswith("#")
        )
        for pkg in ("torch", "transformers", "alpaca", "plotly"):
            assert pkg not in instructions

    def test_dockerfile_does_not_bake_secrets_into_the_image(self):
        text = (self._repo_root() / "Dockerfile").read_text()
        import re
        # An ENV/ARG line assigning anything secret-shaped a literal value.
        offenders = [
            line for line in text.splitlines()
            if re.match(r"\s*(ENV|ARG)\s", line)
            and re.search(r"(KEY|SECRET|TOKEN|PASSWORD)\s*=\s*\S", line, re.I)
        ]
        assert offenders == [], f"secret baked into the image: {offenders}"

    def test_container_does_not_run_as_root(self):
        text = (self._repo_root() / "Dockerfile").read_text()
        users = [l.split()[1] for l in text.splitlines()
                 if l.strip().startswith("USER ")]
        assert users, "no USER directive — the container runs as root"
        assert users[-1] != "root"

    def test_runtime_requirements_contain_only_imported_packages(self):
        # torch/transformers/hdbscan/alpaca-trade-api/plotly were pinned but
        # imported nowhere, and were not installed in the environment the suite
        # passes in — listed, never verified.
        text = (self._repo_root() / "api" / "requirements.txt").read_text().lower()
        for pkg in ("torch", "transformers", "hdbscan", "alpaca", "plotly",
                    "sqlalchemy", "aiosqlite", "feedparser", "datareader"):
            # Mentions in comments are fine; a requirement line is not.
            lines = [
                l for l in text.splitlines()
                if pkg in l and not l.strip().startswith("#")
            ]
            assert lines == [], f"{pkg} is required but imported nowhere: {lines}"

    def test_env_example_documents_the_required_secrets(self):
        # An operator must be able to discover what production needs without
        # reading the source, or they discover it as a 503.
        text = (self._repo_root() / "backend" / ".env.example").read_text()
        assert "ADMIN_API_KEY" in text
        assert "CRON_SECRET" in text
        assert "FRONTEND_URL" in text

    def test_env_example_carries_no_real_values(self):
        import re
        path = self._repo_root() / "backend" / ".env.example"
        assigned = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if re.search(r"(KEY|SECRET|TOKEN|PASSWORD)$", key, re.I) and value.strip():
                assigned[key] = value
        assert assigned == {}, f"template ships credential values: {list(assigned)}"


class TestNoTradeExecutionPath:
    """
    OptionQ recommends hedges and never places one.

    The sibling tests above pin that no trading SDK is installed or pinned.
    These pin the stronger property the user actually relies on: that no code
    path submits an order, and that nothing in the configuration claims a
    brokerage link exists. Three `alpaca_*` fields sat in `Settings` with no
    reader anywhere — not configuration, but a claim, in the one file someone
    checks to learn what the service connects to.
    """

    def _repo_root(self):
        from pathlib import Path
        return Path(__file__).resolve().parents[2]

    def _source_files(self):
        root = self._repo_root()
        for sub in ("backend", "frontend/app", "frontend/components", "frontend/lib"):
            base = root / sub
            if not base.exists():
                continue
            for pattern in ("*.py", "*.ts", "*.tsx"):
                for path in base.rglob(pattern):
                    parts = set(path.parts)
                    if ".venv" in parts or "node_modules" in parts:
                        continue
                    if path.name == "test_security.py":
                        continue          # this file names the forbidden calls
                    yield path

    def test_no_settings_field_claims_a_broker_connection(self):
        from backend.config.settings import Settings
        offenders = [
            name for name in Settings.model_fields
            if any(t in name.lower() for t in
                   ("alpaca", "broker", "tradier", "oanda", "ibkr"))
        ]
        assert offenders == [], f"broker configuration present: {offenders}"

    def test_no_source_file_submits_an_order(self):
        """
        An execution call is the one thing that turns a suggestion into a
        trade, so it is worth failing the build over. Comment lines are exempt:
        `requirements.txt` and `settings.py` both record in prose why the
        broker integration is absent, and that prose must stay allowed.
        """
        import re
        forbidden = re.compile(
            r"\b(submit_order|place_order|create_order|execute_trade|"
            r"TradingClient|REST\s*\(\s*key_id)",
            re.I,
        )
        offenders = []
        for path in self._source_files():
            for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith(("#", "//", "*", "/*")):
                    continue
                if forbidden.search(line):
                    offenders.append(f"{path.relative_to(self._repo_root())}:{n}")
        assert offenders == [], f"order-submission call present: {offenders}"

    def test_no_route_is_named_for_execution(self, client):
        """
        A POST that sounds like it trades is a liability even if it is a stub —
        the stub endpoints that named a financial cause for an empty result
        were exactly this failure in a milder form.
        """
        from backend.main import app
        paths = [getattr(r, "path", "") for r in app.routes]
        offenders = [
            p for p in paths
            if any(t in p.lower() for t in
                   ("/order", "/execute", "/trade", "/buy", "/sell", "/fill"))
        ]
        assert offenders == [], f"execution-shaped route exposed: {offenders}"

    def test_the_hedge_response_is_labelled_as_a_suggestion(self):
        """
        The output model carries recommendations, not instructions: no field
        may look like an order ticket a caller could forward to a broker.
        """
        from backend.models.hedge_models import HedgeOutput, InstrumentCandidate
        for model in (HedgeOutput, InstrumentCandidate):
            offenders = [
                name for name in model.model_fields
                if any(t in name.lower() for t in
                       ("order_id", "client_order", "time_in_force",
                        "order_type", "filled", "execution"))
            ]
            assert offenders == [], f"{model.__name__} has order fields: {offenders}"
