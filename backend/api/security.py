"""
API authentication and error sanitisation.

Two rules drive this module:

1. **Authorisation fails closed.** A missing secret is a misconfiguration, not
   a grant of public access. `_verify_cron_secret` previously returned early
   when `CRON_SECRET` was unset, with the comment "no secret configured — open
   access", which left `POST /admin/retrain/main` — a blocking, CPU-bound model
   retrain — callable by anyone who could reach the host.

2. **Clients never see internal exception text.** Nine handlers raised
   `HTTPException(status_code=500, detail=str(exc))`, which forwards file
   paths, driver errors, SQL and provider URLs to the caller. The detail is
   logged server-side and the client gets a stable, generic message plus an
   error id it can quote in a support request.
"""

import logging
import os
import secrets
import uuid
from typing import Optional

from fastapi import Header, HTTPException

from backend.config.settings import settings

logger = logging.getLogger(__name__)

# Below this length a configured secret is treated as absent: a two-character
# key is not a credential, and accepting one would make the fail-closed check
# cosmetic.
MIN_SECRET_LENGTH = 16


def _configured(name: str) -> Optional[str]:
    """
    Read a secret, treating blank or too-short values as unconfigured so they
    cannot be used to satisfy an auth check.

    The process environment wins over `backend/.env` so a deployment can
    override the file without editing it, but the file is consulted too —
    otherwise a key set in `.env` would silently fail to authorise anything
    while every other credential in the app loads from there.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        raw = (getattr(settings, name.lower(), "") or "").strip()
    if not raw:
        return None
    if len(raw) < MIN_SECRET_LENGTH:
        logger.error(
            "%s is set but only %d characters; refusing to accept it as a "
            "credential (minimum %d)", name, len(raw), MIN_SECRET_LENGTH,
        )
        return None
    return raw


def _bearer(authorization: Optional[str]) -> Optional[str]:
    """Extract the token from an `Authorization: Bearer <token>` header."""
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def _matches(presented: Optional[str], expected: str) -> bool:
    """
    Constant-time comparison.

    The previous check was `authorization != expected`, a plain string compare
    that short-circuits on the first differing byte and so leaks the secret's
    prefix through response timing.
    """
    if not presented:
        return False
    return secrets.compare_digest(presented, expected)


def require_admin(authorization: Optional[str] = Header(default=None)) -> None:
    """
    FastAPI dependency guarding every `/admin/*` route except the cron
    endpoints, which carry their own narrower credential.

    Returns None on success and raises 401 otherwise. When `ADMIN_API_KEY` is
    not configured the route is refused with 503: an unconfigured deployment
    must not expose retraining and training logs to the public internet.
    """
    expected = _configured("ADMIN_API_KEY")
    if expected is None:
        logger.error(
            "ADMIN_API_KEY is not configured — refusing admin request rather "
            "than serving it unauthenticated"
        )
        raise HTTPException(
            status_code=503,
            detail="Administrative endpoints are not configured on this server.",
        )

    if not _matches(_bearer(authorization), expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


def require_cron(authorization: Optional[str] = Header(default=None)) -> None:
    """
    FastAPI dependency for the scheduled-retrain endpoints.

    Scoped to those two routes only: a scheduler needs to trigger a retrain,
    not to read training logs or run hedge checks, so the cron credential does
    not satisfy `require_admin`.
    """
    expected = _configured("CRON_SECRET")
    if expected is None:
        logger.error(
            "CRON_SECRET is not configured — refusing cron request rather than "
            "allowing unauthenticated model retraining"
        )
        raise HTTPException(
            status_code=503,
            detail="Scheduled tasks are not configured on this server.",
        )

    if not _matches(_bearer(authorization), expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


class UserFacingError(Exception):
    """
    Marks an exception whose message was written for the end user.

    Without this distinction there is no way to tell, at the point a handler
    turns an exception into a response, whether `str(exc)` is a sentence the
    product authored ("Could not retrieve a live price for any holding") or
    text from a library ("Length of values (3) does not match length of index
    (5)", a driver error, a file path). The first belongs in the response; the
    second must never leave the server. Subclass this — alongside the natural
    base, e.g. `class Foo(UserFacingError, ValueError)` — for the first kind,
    and let everything else go down the `internal_error` path.
    """


def user_facing_error(exc: "UserFacingError", status_code: int) -> HTTPException:
    """
    Surface `exc`'s message verbatim, which is only valid for a
    `UserFacingError`.

    The isinstance check is deliberate and not defensive dressing: it is what
    makes `detail=str(exc)` impossible to reintroduce by accident, since the
    only sanctioned route to a verbatim message now refuses an exception that
    has not been declared safe.
    """
    if not isinstance(exc, UserFacingError):
        raise TypeError(
            f"{type(exc).__name__} is not a UserFacingError; its message may "
            f"contain internal detail. Use internal_error() instead."
        )
    logger.warning("%s: %s", type(exc).__name__, exc)
    return HTTPException(status_code=status_code, detail=str(exc))


def internal_error(exc: BaseException, context: str) -> HTTPException:
    """
    Log `exc` with a correlation id and return a sanitised 500 for the client.

    Use as `raise internal_error(exc, "options hedge")`. The caller sees only
    the context and the id; the traceback stays in the server log.

    `exc_info=exc` rather than `logger.exception`: the latter reads the
    traceback from the ambient `sys.exc_info()`, so a call made outside an
    `except` block logged "NoneType: None" and the correlation id pointed at
    nothing. Passing the exception that was handed to us records the trace
    whatever the call site.
    """
    error_id = uuid.uuid4().hex[:12]
    logger.error("[%s] %s failed", error_id, context, exc_info=exc)
    return HTTPException(
        status_code=500,
        detail=(
            f"{context} could not be completed due to an internal error. "
            f"Reference: {error_id}"
        ),
    )


# ── Startup configuration validation ─────────────────────────────────────────

#: Secrets whose absence disables functionality rather than breaking it. Named
#: here so startup can report the whole picture at once instead of the operator
#: discovering each gap through a 503 in production.
_SECRET_REQUIREMENTS = (
    ("ADMIN_API_KEY", "administrative endpoints (/admin/*)"),
    ("CRON_SECRET",   "scheduled retrain endpoints (/admin/cron/*)"),
)


class ConfigurationError(RuntimeError):
    """The deployment cannot start safely with the configuration it was given."""


def validate_environment() -> list[str]:
    """
    Check security-relevant configuration at startup and return the problems.

    In development the problems are logged as warnings and the server starts,
    because running locally without an admin key is normal — the endpoints
    refuse themselves, so nothing is exposed.

    In production (`ENVIRONMENT=production`) the same problems raise
    `ConfigurationError` and the server does not start. A deployment whose
    admin routes all answer 503, or whose CORS policy still points at
    `http://localhost:3000` while sending credentials, is misconfigured in a
    way that is better discovered at boot than by a user.
    """
    problems: list[str] = []

    for name, what in _SECRET_REQUIREMENTS:
        if _configured(name) is None:
            problems.append(
                f"{name} is not configured (or is shorter than "
                f"{MIN_SECRET_LENGTH} characters): {what} will refuse all "
                f"requests with 503"
            )

    origins = settings.allowed_origins
    if not origins:
        problems.append(
            "FRONTEND_URL is empty, so no browser origin is allowed to call "
            "this API"
        )
    elif settings.is_production:
        local = [o for o in origins if "localhost" in o or "127.0.0.1" in o]
        if local:
            problems.append(
                f"FRONTEND_URL still contains development origin(s) {local} in "
                f"a production deployment"
            )
        insecure = [o for o in origins if o.startswith("http://")]
        if insecure:
            problems.append(
                f"FRONTEND_URL contains non-HTTPS origin(s) {insecure}; "
                f"credentialed cross-origin requests must not be sent over "
                f"plain HTTP"
            )

    if problems:
        if settings.is_production:
            raise ConfigurationError(
                "Refusing to start: " + "; ".join(problems)
            )
        for p in problems:
            logger.warning("Configuration: %s", p)

    return problems
