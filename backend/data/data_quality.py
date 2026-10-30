"""
Request-scoped data provenance ledger.

Every market-data read records where its value came from and how old it is, so
the API can tell a user the difference between a recommendation built from live
exchange data and one built from an hour-old cache or a fallback assumption.

Usage (one ledger per analysis request):

    with data_quality.capture() as dq:
        ...run the pipeline...
    output.data_quality = dq.summary()

Modules deeper in the stack just call ``data_quality.record(...)``; if no ledger
is active (unit tests, scripts, scheduler jobs) the call is a no-op.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ── Provenance classes ───────────────────────────────────────────────────────
SOURCE_LIVE = "live"                # fetched from the provider during this request
SOURCE_CACHE = "cache"              # served from the local cache
SOURCE_FALLBACK = "fallback"        # a documented substitute (e.g. realized vol for IV)
SOURCE_UNAVAILABLE = "unavailable"  # could not be obtained at all

# A cached value older than this is reported as stale to the user.
STALE_AFTER_SECONDS = 900  # 15 minutes


@dataclass
class DataPoint:
    """Provenance for a single named piece of market data."""
    name: str
    source: str
    as_of: Optional[str] = None
    age_seconds: Optional[float] = None
    detail: str = ""
    degraded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name, "source": self.source, "degraded": self.degraded}
        if self.as_of:
            out["as_of"] = self.as_of
        if self.age_seconds is not None:
            out["age_seconds"] = round(self.age_seconds, 1)
        if self.detail:
            out["detail"] = self.detail
        return out


# A grouped warning names at most this many fields inline; the rest are
# counted. Nothing is hidden — `degraded_fields` always carries every point.
MAX_SUBJECTS_PER_WARNING = 10


def _grouped_warnings(degraded: List[DataPoint]) -> List[str]:
    """
    One line per distinct condition, naming the fields it affects.

    Data points are named ``field:subject`` (``atm_iv:AAPL``, ``garch:SPY``),
    and a degradation almost never affects one subject alone: out of market
    hours the same "no quoted strike within 10% of spot" applies to every
    ticker at once. Emitting a line per point produced 77 warnings in a real
    run, in which the two a user actually needed to act on — a holding left
    unhedged, a regime computed from stale data — were indistinguishable from
    the other 75. Grouping is presentational only: the condition, the detail
    and every affected subject are all still stated, and the per-point records
    remain in ``degraded_fields`` for anyone who needs them.

    Cached groups report the age of their *oldest* member, so the figure is
    never more reassuring than the worst value behind it.
    """
    groups: Dict[tuple, List[DataPoint]] = {}
    for p in degraded:
        field_name, _, _subject = p.name.partition(":")
        key = (field_name, p.source, p.detail or "")
        groups.setdefault(key, []).append(p)

    lines: List[str] = []
    for (field_name, source, detail), pts in groups.items():
        subjects = [s for s in (p.name.partition(":")[2] for p in pts) if s]
        if len(subjects) > MAX_SUBJECTS_PER_WARNING:
            shown = subjects[:MAX_SUBJECTS_PER_WARNING]
            subject_txt = (
                f" ({', '.join(shown)} "
                f"and {len(subjects) - len(shown)} more)"
            )
        elif subjects:
            subject_txt = f" ({', '.join(subjects)})"
        else:
            subject_txt = ""

        if source == SOURCE_UNAVAILABLE:
            condition = "unavailable"
        elif source == SOURCE_FALLBACK:
            condition = "using fallback"
        else:
            mins = max(int((p.age_seconds or 0) // 60) for p in pts)
            condition = (
                f"cached data up to ~{mins} min old" if len(pts) > 1
                else f"cached data ~{mins} min old"
            )

        suffix = f" — {detail}" if detail else ""
        lines.append(f"{field_name}: {condition}{subject_txt}{suffix}")

    return lines


@dataclass
class DataQualityLog:
    """Collects provenance for one analysis request."""
    _points: Dict[str, DataPoint] = field(default_factory=dict)

    def record(
        self,
        name: str,
        source: str,
        *,
        as_of: Optional[str] = None,
        age_seconds: Optional[float] = None,
        detail: str = "",
        degraded: Optional[bool] = None,
    ) -> None:
        if degraded is None:
            if source in (SOURCE_FALLBACK, SOURCE_UNAVAILABLE):
                degraded = True
            elif source == SOURCE_CACHE and (age_seconds or 0) > STALE_AFTER_SECONDS:
                degraded = True
            else:
                degraded = False

        if as_of is None and source == SOURCE_LIVE:
            as_of = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # Worst observation wins: one stale fetch of a field shouldn't be erased
        # by a later live fetch of the same field.
        existing = self._points.get(name)
        if existing is not None and existing.degraded and not degraded:
            return

        self._points[name] = DataPoint(
            name=name, source=source, as_of=as_of,
            age_seconds=age_seconds, detail=detail, degraded=degraded,
        )

    # ── Reporting ────────────────────────────────────────────────────────────
    @property
    def points(self) -> List[DataPoint]:
        return list(self._points.values())

    @property
    def degradations(self) -> List[DataPoint]:
        return [p for p in self._points.values() if p.degraded]

    def has_unavailable(self, prefix: str = "") -> bool:
        return any(
            p.source == SOURCE_UNAVAILABLE and p.name.startswith(prefix)
            for p in self._points.values()
        )

    def summary(self) -> Dict[str, Any]:
        """Compact, user-facing provenance block for the API response."""
        counts: Dict[str, int] = {}
        for p in self._points.values():
            counts[p.source] = counts.get(p.source, 0) + 1

        degraded = sorted(self.degradations, key=lambda p: p.name)

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_counts": counts,
            "degraded_fields": [p.to_dict() for p in degraded],
            "warnings": _grouped_warnings(degraded),
            "is_degraded": bool(degraded),
        }


# ── Ambient (request-scoped) ledger ──────────────────────────────────────────
_CURRENT: ContextVar[Optional[DataQualityLog]] = ContextVar("optionq_data_quality", default=None)


def current_log() -> Optional[DataQualityLog]:
    return _CURRENT.get()


def record(name: str, source: str, **kwargs) -> None:
    """Record provenance against the active ledger, if any."""
    log = _CURRENT.get()
    if log is not None:
        log.record(name, source, **kwargs)


@contextmanager
def capture(reuse: bool = False):
    """
    Open a ledger for the duration of one request.

    reuse=True joins an already-active ledger instead of shadowing it, so an
    inner entry point (e.g. the orchestrator called directly) can guarantee a
    ledger exists without discarding the caller's provenance. Note that
    `asyncio.to_thread` copies the context, so worker threads share the same
    ledger object and their records land in the caller's ledger.
    """
    existing = _CURRENT.get()
    if reuse and existing is not None:
        yield existing
        return

    log = DataQualityLog()
    token = _CURRENT.set(log)
    try:
        yield log
    finally:
        _CURRENT.reset(token)
