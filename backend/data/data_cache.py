"""
SQLite response cache.

The cache is an optimisation, so losing it must cost latency and nothing else.
It did not: every market-data fetch ended in an unguarded `cache_set`, and
`_get_conn` commits a `CREATE TABLE IF NOT EXISTS` on *every* call including
reads, so a database the process cannot write turned a successful fetch into
`sqlite3.OperationalError: attempt to write a readonly database` and took down
the request that had already got its data.

That is precisely the container's configuration. The image copies this package
to /app/backend as root and then runs as uid 10001, chowning only
backend/ml_models — so /app/backend is not writable by the runtime user, and
SQLite needs to write both the database file and a rollback journal beside it.
Reads happened to survive (CREATE TABLE IF NOT EXISTS on an existing table
needs no write), which made the failure worse rather than better: a cache hit
returned data and a cache miss raised, so the product failed intermittently
depending on what the shipped database happened to contain.

Two changes follow from that. The database is located at the first writable
candidate path rather than assumed to be writable where the code lives, and
every cache operation degrades to a miss on a database error instead of
propagating it. Degrading is safe here in a way it would not be for market
data: the value still comes live from the provider on the next line. It is
recorded against the data-quality ledger all the same, because a product
silently running with no cache is a product whose latency and rate-limit
behaviour nobody can explain.
"""
import sqlite3
import json
import logging
import time
import hashlib
import os
import tempfile
from typing import Optional, Any

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "hedgeos_cache.db")


def _candidate_db_paths():
    """
    Where to look for a writable cache, best first.

    An explicit setting wins, so a deployment can put the cache on a mounted
    volume. The package directory stays the default because that is where a
    developer's existing cache already is. The temp directory is the last
    resort: it is writable in every container this runs in, and a cache that
    does not survive a restart is still worth having within one.
    """
    override = os.environ.get("OPTIONQ_CACHE_DB")
    if override:
        yield override
    yield _DEFAULT_DB_PATH
    yield os.path.join(tempfile.gettempdir(), "hedgeos_cache.db")


def _probe_writable(path: str) -> bool:
    """
    Open the database and write to it for real.

    `os.access` and a bare `sqlite3.connect` both report success on a path
    SQLite cannot actually commit to — connect is lazy, and the directory
    permission that blocks the journal file is not the file permission
    os.access checks. The only reliable probe is a transaction, so this does
    one and rolls it back.
    """
    try:
        parent = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(path, timeout=5)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    stored_at REAL
                )
            """)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(cache)").fetchall()}
            if "stored_at" not in cols:
                conn.execute("ALTER TABLE cache ADD COLUMN stored_at REAL")
            # The probe that matters: a real write, then undo it.
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, expires_at, stored_at) "
                "VALUES (?, ?, ?, ?)",
                ("__writability_probe__", "null", 0.0, 0.0),
            )
            conn.execute("DELETE FROM cache WHERE key = ?", ("__writability_probe__",))
            conn.commit()
            return True
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as exc:
        logger.info("cache: %s is not usable (%s)", path, exc)
        return False


_RESOLVED_PATH: Optional[str] = None
_RESOLUTION_DONE = False


def _resolve_db_path() -> Optional[str]:
    """
    First writable candidate, resolved once. None means caching is disabled.
    """
    global _RESOLVED_PATH, _RESOLUTION_DONE
    if _RESOLUTION_DONE:
        return _RESOLVED_PATH

    tried = []
    for path in _candidate_db_paths():
        tried.append(path)
        if _probe_writable(path):
            _RESOLVED_PATH = path
            _RESOLUTION_DONE = True
            if len(tried) > 1:
                logger.warning(
                    "cache: falling back to %s — the preferred location(s) %s "
                    "are not writable by this process",
                    path, tried[:-1],
                )
            return _RESOLVED_PATH

    logger.error(
        "cache: no writable database location among %s — running without a "
        "cache. Every request will refetch from the provider, which is "
        "slower and consumes provider rate limit, but no value is stale.",
        tried,
    )
    _RESOLVED_PATH = None
    _RESOLUTION_DONE = True
    return None


def reset_resolution() -> None:
    """Forget the resolved path. For tests that change the environment."""
    global _RESOLVED_PATH, _RESOLUTION_DONE
    _RESOLVED_PATH = None
    _RESOLUTION_DONE = False


#: Why the cache is off, when no location could be opened at all. Separate from
#: the per-error details so the three call sites stay consistent.
_DISABLED_DETAIL = (
    "no writable cache location; every value is fetched live, which is slower "
    "and uses more provider rate limit but is never stale"
)


def _record_cache_unavailable(detail: str) -> None:
    """
    Disclose a disabled cache on the request ledger.

    Imported lazily: data_quality is a leaf module today, and importing it at
    module scope would make the dependency between the two cache-adjacent
    modules circular the moment that changes.
    """
    try:
        from backend.data import data_quality as dq
        dq.record("response_cache", dq.SOURCE_UNAVAILABLE, detail=detail)
    except Exception:       # pragma: no cover - never let disclosure break a request
        pass


def _get_conn() -> Optional[sqlite3.Connection]:
    path = _resolve_db_path()
    if path is None:
        return None
    conn = sqlite3.connect(path, timeout=5)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            expires_at REAL NOT NULL,
            stored_at REAL
        )
    """)
    # Migrate caches created before stored_at existed.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(cache)").fetchall()}
    if "stored_at" not in cols:
        conn.execute("ALTER TABLE cache ADD COLUMN stored_at REAL")
    conn.commit()
    return conn


def _make_key(namespace: str, *args) -> str:
    raw = namespace + "|" + "|".join(str(a) for a in args)
    return hashlib.md5(raw.encode()).hexdigest()


def cache_get_entry(namespace: str, *args) -> Optional[tuple]:
    """
    Returns (value, age_seconds) for a live cache entry, or None if absent/expired.

    Age lets callers disclose how stale a served value is instead of presenting
    cached data as if it were live.
    """
    key = _make_key(namespace, *args)
    try:
        conn = _get_conn()
        if conn is None:
            _record_cache_unavailable(_DISABLED_DETAIL)
            return None
        try:
            row = conn.execute(
                "SELECT value, expires_at, stored_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
        finally:
            conn.close()
    except (sqlite3.Error, OSError, ValueError) as exc:
        # A cache that cannot be read is a cache miss, not a failed request.
        logger.warning("cache: read failed for %s (%s)", namespace, exc)
        _record_cache_unavailable(f"cache read failed: {exc}")
        return None

    if row is None:
        return None
    value, expires_at, stored_at = row
    now = time.time()
    if now > expires_at:
        return None
    age = max(0.0, now - (stored_at if stored_at is not None else expires_at))
    try:
        return json.loads(value), age
    except (json.JSONDecodeError, TypeError) as exc:
        # A corrupt row must not be served as data. Treat it as absent.
        logger.warning("cache: discarding unreadable entry for %s (%s)", namespace, exc)
        return None


def cache_get(namespace: str, *args) -> Optional[Any]:
    entry = cache_get_entry(namespace, *args)
    return None if entry is None else entry[0]


def cache_set(namespace: str, *args, value: Any, ttl: int = 3600) -> None:
    key = _make_key(namespace, *args)
    now = time.time()
    try:
        conn = _get_conn()
        if conn is None:
            # Not an exception path, so this needs its own disclosure: a cache
            # that was never opened degrades the product just as much as one
            # that failed mid-write, and saying nothing here is the silent
            # substitution this codebase is meant to be rid of.
            _record_cache_unavailable(_DISABLED_DETAIL)
            return
        try:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, expires_at, stored_at) "
                "VALUES (?, ?, ?, ?)",
                (key, json.dumps(value, default=str), now + ttl, now),
            )
            conn.commit()
        finally:
            conn.close()
    except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
        # Failing to *store* a value the caller already has in hand must never
        # fail the caller. This is the line that took the container down: the
        # fetch had succeeded and the result was discarded along with it.
        logger.warning("cache: write failed for %s (%s)", namespace, exc)
        _record_cache_unavailable(f"cache write failed: {exc}")


def cache_clear_expired() -> int:
    try:
        conn = _get_conn()
        if conn is None:
            _record_cache_unavailable(_DISABLED_DETAIL)
            return 0
        try:
            cursor = conn.execute(
                "DELETE FROM cache WHERE expires_at < ?", (time.time(),)
            )
            deleted = cursor.rowcount
            conn.commit()
        finally:
            conn.close()
        return deleted
    except (sqlite3.Error, OSError) as exc:
        logger.warning("cache: expiry sweep failed (%s)", exc)
        _record_cache_unavailable(f"cache expiry sweep failed: {exc}")
        return 0
