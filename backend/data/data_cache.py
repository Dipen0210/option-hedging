import sqlite3
import json
import time
import hashlib
import os
from typing import Optional, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "hedgeos_cache.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            expires_at REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def _make_key(namespace: str, *args) -> str:
    raw = namespace + "|" + "|".join(str(a) for a in args)
    return hashlib.md5(raw.encode()).hexdigest()


def cache_get(namespace: str, *args) -> Optional[Any]:
    key = _make_key(namespace, *args)
    conn = _get_conn()
    row = conn.execute(
        "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    value, expires_at = row
    if time.time() > expires_at:
        return None
    return json.loads(value)


def cache_set(namespace: str, *args, value: Any, ttl: int = 3600) -> None:
    key = _make_key(namespace, *args)
    expires_at = time.time() + ttl
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
        (key, json.dumps(value, default=str), expires_at),
    )
    conn.commit()
    conn.close()


def cache_clear_expired() -> int:
    conn = _get_conn()
    cursor = conn.execute(
        "DELETE FROM cache WHERE expires_at < ?", (time.time(),)
    )
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted
