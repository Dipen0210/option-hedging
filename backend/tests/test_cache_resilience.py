"""
A cache failure must cost latency, not availability.

The defect: `_get_conn` commits a `CREATE TABLE IF NOT EXISTS` on every call,
including reads, and `get_price_history` ended in an unguarded `cache_set`. So
a database the process could not write turned an already-successful fetch into
`sqlite3.OperationalError: attempt to write a readonly database`.

That was the production container's exact configuration and nothing else's. The
image copies the package to /app/backend as root, chowns only
backend/ml_models, and runs as uid 10001 — so /app/backend is not writable, and
SQLite needs to write both the database and a journal file beside it. Reads
survived (CREATE TABLE IF NOT EXISTS on an existing table needs no write) while
writes did not, so with the developer's 6.4MB cache baked into the image the
product failed *intermittently*, on whichever tickers happened not to be cached.

No test on a developer machine could see it: the package directory is writable
there. These tests reproduce the permission condition instead of assuming it,
by revoking write permission on a real directory.
"""
import os
import sqlite3
import stat
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.data import data_cache as dc
from backend.data import data_quality as dq


@pytest.fixture
def readonly_dir(tmp_path):
    """A directory the test process genuinely cannot write to."""
    d = tmp_path / "approot"
    d.mkdir()
    # Seed a database, as the image does by copying the developer's cache in.
    conn = sqlite3.connect(str(d / "hedgeos_cache.db"))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        " expires_at REAL NOT NULL, stored_at REAL)"
    )
    conn.commit()
    conn.close()
    d.chmod(stat.S_IRUSR | stat.S_IXUSR)        # r-x, no write
    yield d
    d.chmod(stat.S_IRWXU)                        # restore so tmp_path cleanup works


@pytest.fixture
def cache_at(monkeypatch, tmp_path):
    """
    Point the cache's preferred location somewhere, and reset its one-time
    path resolution around the test.
    """
    def _set(preferred, fallback_dir=None):
        monkeypatch.setattr(dc, "_DEFAULT_DB_PATH", str(preferred))
        monkeypatch.delenv("OPTIONQ_CACHE_DB", raising=False)
        monkeypatch.setenv("TMPDIR", str(fallback_dir or (tmp_path / "fallback")))
        os.makedirs(str(fallback_dir or (tmp_path / "fallback")), exist_ok=True)
        dc.reset_resolution()
    yield _set
    dc.reset_resolution()


# ── Path resolution ──────────────────────────────────────────────────────────

class TestItFindsAWritableLocation:

    def test_a_writable_preferred_path_is_used_unchanged(self, cache_at, tmp_path):
        preferred = tmp_path / "ok" / "hedgeos_cache.db"
        preferred.parent.mkdir()
        cache_at(preferred)
        assert dc._resolve_db_path() == str(preferred)

    def test_an_unwritable_preferred_path_falls_back(self, cache_at, readonly_dir):
        cache_at(readonly_dir / "hedgeos_cache.db")
        resolved = dc._resolve_db_path()
        assert resolved is not None, "caching should fall back, not switch off"
        assert str(readonly_dir) not in resolved

    def test_an_explicit_setting_wins(self, cache_at, monkeypatch, tmp_path):
        chosen = tmp_path / "chosen" / "c.db"
        chosen.parent.mkdir()
        cache_at(tmp_path / "ignored.db")
        monkeypatch.setenv("OPTIONQ_CACHE_DB", str(chosen))
        dc.reset_resolution()
        assert dc._resolve_db_path() == str(chosen)

    def test_resolution_happens_once(self, cache_at, tmp_path):
        preferred = tmp_path / "ok2" / "hedgeos_cache.db"
        preferred.parent.mkdir()
        cache_at(preferred)
        calls = []
        real = dc._probe_writable
        dc._probe_writable = lambda p: (calls.append(p), real(p))[1]
        try:
            dc._resolve_db_path()
            dc._resolve_db_path()
            dc._resolve_db_path()
        finally:
            dc._probe_writable = real
        assert len(calls) == 1, f"probed {len(calls)} times, expected 1"

    def test_the_probe_detects_a_directory_it_cannot_journal_in(self, readonly_dir):
        # The precise failure: the file is readable, the directory is not
        # writable, so SQLite cannot create its rollback journal. os.access on
        # the file alone does not catch this, which is why the probe writes.
        db = readonly_dir / "hedgeos_cache.db"
        assert db.exists()
        assert dc._probe_writable(str(db)) is False


# ── Degradation ──────────────────────────────────────────────────────────────

class TestAnUnwritableCacheDoesNotFailTheRequest:

    @pytest.fixture
    def no_cache(self, cache_at, tmp_path, monkeypatch):
        """Every candidate location unwritable — caching fully off."""
        cache_at(tmp_path / "whatever.db")
        monkeypatch.setattr(dc, "_probe_writable", lambda p: False)
        dc.reset_resolution()
        assert dc._resolve_db_path() is None

    def test_cache_set_is_a_no_op_rather_than_an_error(self, no_cache):
        dc.cache_set("price_history", "SPY", "2y", value=[{"Close": 1.0}], ttl=60)

    def test_cache_get_entry_reports_a_miss(self, no_cache):
        assert dc.cache_get_entry("price_history", "SPY", "2y") is None

    def test_cache_get_reports_a_miss(self, no_cache):
        assert dc.cache_get("price_history", "SPY", "2y") is None

    def test_the_expiry_sweep_reports_nothing_deleted(self, no_cache):
        assert dc.cache_clear_expired() == 0

    def test_a_disabled_cache_is_disclosed_on_the_ledger(self, no_cache):
        with dq.capture() as log:
            dc.cache_set("price_history", "SPY", "2y", value=[{"Close": 1.0}], ttl=60)
        names = [p.name for p in log.points]
        assert "response_cache" in names, (
            "a product running with no cache must say so — its latency and "
            "provider rate-limit behaviour are otherwise unexplainable"
        )
        point = next(p for p in log.points if p.name == "response_cache")
        assert point.source == dq.SOURCE_UNAVAILABLE

    def test_a_write_error_is_disclosed_rather_than_raised(self, cache_at, tmp_path):
        cache_at(tmp_path / "ok3" / "hedgeos_cache.db")
        (tmp_path / "ok3").mkdir()
        dc._resolve_db_path()

        def exploding_conn():
            raise sqlite3.OperationalError("attempt to write a readonly database")

        orig = dc._get_conn
        dc._get_conn = exploding_conn
        try:
            with dq.capture() as log:
                dc.cache_set("price_history", "SPY", "2y", value=[{"a": 1}], ttl=60)
                assert dc.cache_get_entry("price_history", "SPY", "2y") is None
        finally:
            dc._get_conn = orig
        assert any(p.name == "response_cache" for p in log.points)


# ── The call path that actually broke ────────────────────────────────────────

class TestTheFetchPathSurvivesAnUnwritableCache:

    def test_get_price_history_returns_data_when_the_cache_cannot_be_written(
        self, cache_at, readonly_dir, monkeypatch
    ):
        """
        The regression in one test: before the fix this raised
        OperationalError *after* the fetch had already succeeded.
        """
        from backend.data import market_data as md

        cache_at(readonly_dir / "hedgeos_cache.db")
        monkeypatch.setattr(dc, "_probe_writable", lambda p: False)
        dc.reset_resolution()

        bars = pd.DataFrame(
            {"Open": [99.0], "High": [101.0], "Low": [98.0],
             "Close": [100.0], "Volume": [1_000_000]},
            index=pd.to_datetime(["2026-10-28"]),
        )
        monkeypatch.setattr(md, "_yahoo_history_direct", lambda t, p="2y": bars)
        # The module imported these by name, so patch where they are used.
        monkeypatch.setattr(md, "cache_get_entry", dc.cache_get_entry)
        monkeypatch.setattr(md, "cache_set", dc.cache_set)

        df = md.get_price_history("SPY", "2y")
        assert len(df) == 1
        assert df["Close"].iloc[0] == pytest.approx(100.0)

    def test_the_value_still_comes_from_the_provider_not_a_default(
        self, cache_at, tmp_path, monkeypatch
    ):
        # Degrading the cache must not degrade the data. Whatever the provider
        # returned is what the caller gets.
        from backend.data import market_data as md

        cache_at(tmp_path / "x.db")
        monkeypatch.setattr(dc, "_probe_writable", lambda p: False)
        dc.reset_resolution()

        bars = pd.DataFrame(
            {"Open": [1.0], "High": [1.0], "Low": [1.0],
             "Close": [123.456], "Volume": [7]},
            index=pd.to_datetime(["2026-10-28"]),
        )
        monkeypatch.setattr(md, "_yahoo_history_direct", lambda t, p="2y": bars)
        monkeypatch.setattr(md, "cache_get_entry", dc.cache_get_entry)
        monkeypatch.setattr(md, "cache_set", dc.cache_set)

        assert md.get_price_history("SPY", "2y")["Close"].iloc[0] == pytest.approx(123.456)


# ── Corrupt entries are not served as data ───────────────────────────────────

class TestACorruptEntryIsTreatedAsAbsent:

    def test_unparseable_json_is_a_miss_not_an_exception(self, cache_at, tmp_path):
        import time
        path = tmp_path / "ok4" / "hedgeos_cache.db"
        path.parent.mkdir()
        cache_at(path)
        dc._resolve_db_path()
        dc.cache_set("price_history", "SPY", "2y", value=[{"a": 1}], ttl=600)

        key = dc._make_key("price_history", "SPY", "2y")
        conn = sqlite3.connect(str(path))
        conn.execute(
            "UPDATE cache SET value = ? WHERE key = ?", ("{not json", key)
        )
        conn.commit()
        conn.close()

        assert dc.cache_get_entry("price_history", "SPY", "2y") is None
