"""
A deterministic stand-in for Yahoo Finance, for integration tests.

THIS IS SYNTHETIC TEST DATA. None of the prices, implied volatilities, open
interest figures or contract symbols produced here are market observations, and
nothing in this module may be used to populate a user-facing recommendation.
It exists so the pipeline can be exercised end to end with a fixed input, on a
weekend, with no network, and get the same answer every time.

Why it is built at the network boundary rather than by stubbing the data layer:
the thing most worth testing in OptionQ is the data layer. Replacing
`get_contract_quote` with a stub that returns a tidy dict would test the
selector against an idealisation of the chain and prove nothing about the code
that reads a real one. So this module fakes exactly two things —
`requests.Session.get` against Yahoo's chart API, and `yfinance.Ticker` — and
leaves every line of parsing, timestamp normalisation, caching, IV
interpolation, liquidity assessment and data-quality recording to run for real.

The option book is internally consistent: each contract is priced from
Black-Scholes at an implied volatility drawn from a downward-sloping skew, the
bid/ask straddles that value with a spread that widens as the strike moves out
of the money, and open interest thins in the wings. A test can therefore assert
financial relationships (a 5%-OTM put costs less than a 2%-OTM one; a spread
costs less than the long leg alone) and not merely that fields are populated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# A fixed "today" so expiries, day counts and VaR horizons never shift under
# the tests. Chosen as a Wednesday so the weekly expiry ladder is unambiguous.
AS_OF = date(2026, 6, 10)

#: Spot price each ticker is anchored to. Round, obviously-synthetic numbers:
#: nobody should be able to mistake a fixture for a quote.
BASE_PRICES: Dict[str, float] = {
    "AAPL": 200.00, "MSFT": 400.00, "NVDA": 120.00, "AMZN": 180.00,
    "GOOGL": 160.00, "META": 500.00, "TSLA": 250.00, "AMD": 150.00,
    "NFLX": 600.00, "CRM": 260.00,
    "JPM": 200.00, "BAC": 40.00, "GS": 450.00, "MS": 100.00, "V": 280.00,
    "MA": 450.00, "BRKB": 420.00,
    "JNJ": 160.00, "PFE": 30.00, "UNH": 500.00, "ABBV": 180.00,
    "XOM": 110.00, "CVX": 160.00, "KO": 60.00, "WMT": 70.00,
    "SPY": 500.00, "QQQ": 440.00, "DIA": 400.00, "IWM": 200.00, "VTI": 260.00,
    "TLT": 90.00, "HYG": 78.00, "LQD": 108.00, "GLD": 210.00, "UUP": 28.00,
    "XLB": 90.00, "XLE": 90.00, "XLF": 42.00, "XLI": 125.00, "XLK": 220.00,
    "XLP": 78.00, "XLRE": 40.00, "XLU": 70.00, "XLV": 145.00, "XLY": 190.00,
    # Index products the data layer reaches for by name. Every one is listed
    # explicitly: a symbol that falls through to the hash-derived default gets
    # a nonsense level (^VIX6M came out at 419, which the term-structure
    # classifier would happily report as extreme backwardation).
    "^VIX": 17.50, "^VIX3M": 19.00, "^VIX6M": 20.50,
    "^IRX": 4.30, "^FVX": 4.05, "^TNX": 4.25, "^TYX": 4.55,
}

#: Annualised volatility of the generated price path, per ticker. Deliberately
#: spread out so a test can tell a high-vol name from a low-vol one.
BASE_VOLS: Dict[str, float] = {
    "AAPL": 0.24, "MSFT": 0.22, "NVDA": 0.48, "TSLA": 0.52, "AMD": 0.45,
    "SPY": 0.14, "QQQ": 0.19, "DIA": 0.13, "IWM": 0.21, "VTI": 0.14,
    "TLT": 0.13, "HYG": 0.07, "LQD": 0.07, "GLD": 0.13, "UUP": 0.06,
    "^VIX": 0.80, "^VIX3M": 0.60, "^VIX6M": 0.45,
    "^IRX": 0.15, "^FVX": 0.15, "^TNX": 0.15, "^TYX": 0.15,
}
DEFAULT_VOL = 0.25

#: Per-ticker beta to the market factor. The generated paths share a common
#: factor scaled by this, so correlations and betas the pipeline measures are
#: real properties of the fixture rather than noise.
BASE_BETAS: Dict[str, float] = {
    "SPY": 1.00, "VTI": 1.00, "QQQ": 1.15, "DIA": 0.92, "IWM": 1.10,
    "AAPL": 1.10, "MSFT": 1.05, "NVDA": 1.70, "TSLA": 1.60, "AMD": 1.65,
    "META": 1.25, "GOOGL": 1.10, "AMZN": 1.20, "NFLX": 1.15, "CRM": 1.20,
    "JPM": 1.05, "BAC": 1.15, "GS": 1.20, "MS": 1.15, "V": 0.95, "MA": 0.98,
    "BRKB": 0.85, "JNJ": 0.55, "PFE": 0.60, "UNH": 0.65, "ABBV": 0.60,
    "XOM": 0.80, "CVX": 0.85, "KO": 0.55, "WMT": 0.50,
    "TLT": -0.25, "HYG": 0.35, "LQD": 0.15, "GLD": 0.05, "UUP": -0.20,
    "XLB": 0.95, "XLE": 0.85, "XLF": 1.05, "XLI": 1.00, "XLK": 1.25,
    "XLP": 0.55, "XLRE": 0.85, "XLU": 0.50, "XLV": 0.70, "XLY": 1.10,
    "^VIX": -4.0, "^VIX3M": -2.5, "^VIX6M": -1.8,
    "^IRX": 0.0, "^FVX": 0.0, "^TNX": 0.0, "^TYX": 0.0,
}

RISK_FREE = 0.043          # what ^IRX / 100 resolves to in this fixture
DIVIDEND_YIELD = 0.012     # flat across the fixture universe


def _stable_seed(ticker: str) -> int:
    """A seed that depends only on the ticker, so paths never shuffle."""
    # Not `hash()`: PYTHONHASHSEED randomises it per process, so the paths
    # would differ between runs and the fixture would not be deterministic.
    return sum((i + 1) * ord(c) for i, c in enumerate(ticker))


def base_price(ticker: str) -> float:
    if ticker in BASE_PRICES:
        return BASE_PRICES[ticker]
    return 50.0 + (_stable_seed(ticker) % 400)


def base_vol(ticker: str) -> float:
    return BASE_VOLS.get(ticker, DEFAULT_VOL)


def base_beta(ticker: str) -> float:
    return BASE_BETAS.get(ticker, 1.0)


# ── Price paths ──────────────────────────────────────────────────────────────

_PERIOD_DAYS = {
    "5d": 7, "1mo": 31, "3mo": 93, "6mo": 186,
    "1y": 366, "2y": 732, "5y": 1830, "10y": 3660, "max": 3660,
}


def trading_days(n: int) -> pd.DatetimeIndex:
    """`n` business days ending at AS_OF, midnight-normalised like the real code."""
    end = pd.Timestamp(AS_OF)
    return pd.bdate_range(end=end, periods=n).normalize()


def _market_factor(n: int) -> np.ndarray:
    """
    The one common return series every path loads on.

    Shared across tickers so that the betas and correlations the pipeline
    estimates come out near BASE_BETAS — a cross-hedge test is meaningless if
    every series is independent noise.
    """
    rng = np.random.default_rng(20260610)
    # Generated once at full length and sliced, so a 1y request is the tail of
    # the same history a 2y request sees. Otherwise realized vol would depend
    # on which window happened to be asked for.
    full = rng.standard_normal(_PERIOD_DAYS["10y"]) * (0.14 / math.sqrt(252))
    return full[-n:]


def price_path(ticker: str, n_days: int) -> pd.Series:
    """
    Deterministic daily closes: a shared market factor plus idiosyncratic noise,
    scaled so realized vol lands near `base_vol(ticker)`.
    """
    beta = base_beta(ticker)
    vol = base_vol(ticker)
    mkt = _market_factor(n_days)

    # Residual vol chosen so total vol ≈ target: σ² = β²σ_m² + σ_e².
    mkt_vol = 0.14
    resid_var = max(vol**2 - (beta * mkt_vol) ** 2, (0.25 * vol) ** 2)
    resid_sd = math.sqrt(resid_var) / math.sqrt(252)

    rng = np.random.default_rng(_stable_seed(ticker))
    resid = rng.standard_normal(n_days) * resid_sd

    # Zero drift: a drifting fixture makes the "unrealized P&L" assertions
    # depend on how long the window is, which is not what they are testing.
    log_ret = beta * mkt + resid
    log_ret = log_ret - log_ret.mean()

    # Anchor the LAST close to base_price so spot is exactly the round number.
    closes = base_price(ticker) * np.exp(np.cumsum(log_ret) - np.sum(log_ret))
    return pd.Series(closes, index=trading_days(n_days), name="Close")


def ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    n = max(len(pd.bdate_range(
        start=pd.Timestamp(AS_OF) - timedelta(days=_PERIOD_DAYS.get(period, 732)),
        end=pd.Timestamp(AS_OF))), 30)
    close = price_path(ticker, n)
    return pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]),
        "High": close * 1.004,
        "Low": close * 0.996,
        "Close": close,
        "Volume": np.full(len(close), 1_000_000, dtype="int64"),
    }, index=close.index)


# ── Option chain ─────────────────────────────────────────────────────────────

def _strike_step(spot: float) -> float:
    """The ladder spacing a listing exchange would actually use."""
    if spot < 25:
        return 0.5
    if spot < 50:
        return 1.0
    if spot < 200:
        return 2.5
    return 5.0


def expiries(ticker: str = "") -> List[str]:
    """
    A realistic ladder: four weeklies, then monthly third Fridays out two years.
    """
    out: List[str] = []
    d = pd.Timestamp(AS_OF)
    # Weeklies — next four Fridays.
    friday = d + timedelta(days=(4 - d.weekday()) % 7 or 7)
    for i in range(4):
        out.append((friday + timedelta(weeks=i)).date().isoformat())
    # Monthlies — third Friday of each of the next 24 months.
    for m in range(1, 25):
        year = AS_OF.year + (AS_OF.month - 1 + m) // 12
        month = (AS_OF.month - 1 + m) % 12 + 1
        first = date(year, month, 1)
        third_friday = first + timedelta(days=(4 - first.weekday()) % 7 + 14)
        iso = third_friday.isoformat()
        if iso not in out:
            out.append(iso)
    return sorted(out)


def _bsm_put(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    from math import erf, exp, log, sqrt
    nd = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * nd(-d2) - S * exp(-q * T) * nd(-d1)


def _bsm_call(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    # Put-call parity, so the two sides of the fixture can never disagree.
    put = _bsm_put(S, K, T, r, q, sigma)
    return put + S * math.exp(-q * T) - K * math.exp(-r * T)


def _skewed_iv(ticker: str, moneyness: float, T: float) -> float:
    """
    A downward-sloping smile: OTM puts carry more vol than ATM.

    `moneyness` is K/S. Without a skew the fixture would make
    contract-specific IV indistinguishable from a single ATM number, and the
    tests that check skew is respected would pass against a flat surface.
    """
    atm = base_vol(ticker)
    slope = -0.55                      # vol per unit of log-moneyness
    smile = 0.18 * (math.log(moneyness) ** 2)
    term = 0.02 * math.sqrt(max(T, 1 / 365))
    iv = atm + slope * math.log(moneyness) + smile + term
    return float(min(max(iv, 0.06), 1.80))


def _occ_symbol(ticker: str, expiry: str, option_type: str, strike: float) -> str:
    y, m, d = expiry.split("-")
    side = "P" if option_type == "put" else "C"
    return f"{ticker}{y[2:]}{m}{d}{side}{int(round(strike * 1000)):08d}"


@dataclass
class ChainConfig:
    """
    Knobs for the degraded-data scenarios.

    `quotes_published=False` reproduces the overnight Yahoo snapshot that
    caused the worst class of bug in this codebase: strikes come back, bid and
    ask are cleared to zero and open interest is not republished, so an
    unusable snapshot is indistinguishable from a tight, liquid market unless
    the code looks for it.
    """
    quotes_published: bool = True
    chain_available: bool = True
    wide_spreads: bool = False
    publish_iv: bool = True


def chain_side(
    ticker: str,
    expiry: str,
    option_type: str,
    cfg: Optional[ChainConfig] = None,
) -> pd.DataFrame:
    cfg = cfg or ChainConfig()
    S = base_price(ticker)
    step = _strike_step(S)
    T = max((date.fromisoformat(expiry) - AS_OF).days, 1) / 365.25

    lo = math.floor(S * 0.70 / step) * step
    hi = math.ceil(S * 1.30 / step) * step
    strikes = [round(lo + i * step, 2)
               for i in range(int(round((hi - lo) / step)) + 1)]

    rows = []
    for K in strikes:
        m = K / S
        iv = _skewed_iv(ticker, m, T)
        theo = (_bsm_put(S, K, T, RISK_FREE, DIVIDEND_YIELD, iv)
                if option_type == "put"
                else _bsm_call(S, K, T, RISK_FREE, DIVIDEND_YIELD, iv))
        theo = max(theo, 0.01)

        # Spread widens away from the money and for longer tenors, as it does
        # in a real book.
        # Calibrated so a near-the-money contract sits inside the 5%
        # liquidity gate (a real SPY put trades ~1-3% of mid) while the wings
        # blow out past it. A fixture whose every strike failed the gate would
        # test only the rejection path.
        rel = 0.012 + 0.22 * abs(math.log(m)) + 0.02 * math.sqrt(T)
        if cfg.wide_spreads:
            rel *= 12
        half = max(theo * rel / 2, 0.01)
        bid, ask = round(max(theo - half, 0.01), 2), round(theo + half, 2)

        # Depth concentrates at the money and on third Fridays.
        depth = math.exp(-((math.log(m) / 0.10) ** 2))
        oi = int(round(8000 * depth)) + 25
        vol = int(round(900 * depth))

        if not cfg.quotes_published:
            bid = ask = 0.0
            oi = 0
            # Volume survives the session even after quotes are cleared: that
            # asymmetry is the tell the production code looks for.
        rows.append({
            "strike": K,
            "bid": bid,
            "ask": ask,
            "lastPrice": round(theo, 2),
            "impliedVolatility": round(iv, 4) if cfg.publish_iv else 0.0,
            "openInterest": oi,
            "volume": vol,
            "contractSymbol": _occ_symbol(ticker, expiry, option_type, K),
        })
    return pd.DataFrame(rows)


# ── Fake yfinance / fake Yahoo HTTP ──────────────────────────────────────────

@dataclass
class FakeYahoo:
    """
    The whole synthetic market, installable over the network boundary.

    `calls` records what was asked for, so a test can assert the pipeline did
    not quietly skip the chain or request a nonsense period.
    """
    cfg: ChainConfig = field(default_factory=ChainConfig)
    calls: Dict[str, int] = field(default_factory=dict)

    def _note(self, what: str) -> None:
        self.calls[what] = self.calls.get(what, 0) + 1

    # -- yfinance ---------------------------------------------------------
    def ticker(self, symbol: str, *args, **kwargs):
        market = self

        class _FakeTicker:
            def __init__(self, sym: str):
                self.sym = sym

            @property
            def options(self):
                market._note(f"expiries:{self.sym}")
                if not market.cfg.chain_available:
                    return ()
                return tuple(expiries(self.sym))

            def option_chain(self, expiry: str):
                market._note(f"chain:{self.sym}:{expiry}")
                if not market.cfg.chain_available:
                    raise RuntimeError(
                        f"[synthetic] chain unavailable for {self.sym}"
                    )
                return SimpleNamespace(
                    puts=chain_side(self.sym, expiry, "put", market.cfg),
                    calls=chain_side(self.sym, expiry, "call", market.cfg),
                )

            def history(self, period: str = "2y", **kw):
                market._note(f"history:{self.sym}:{period}")
                return ohlcv(self.sym, period)

            @property
            def info(self):
                market._note(f"info:{self.sym}")
                return {
                    "dividendYield": DIVIDEND_YIELD,
                    "regularMarketPrice": base_price(self.sym),
                    "shortName": f"{self.sym} (synthetic fixture)",
                }

            @property
            def calendar(self):
                return {}

        return _FakeTicker(symbol)

    # -- Yahoo chart API over requests.Session -----------------------------
    def session_get(self, url: str, params=None, timeout=None, **kwargs):
        params = params or {}
        symbol = url.rstrip("/").rsplit("/", 1)[-1]
        period = params.get("range", "2y")
        self._note(f"chart:{symbol}:{period}")

        df = ohlcv(symbol, period if period in _PERIOD_DAYS else "2y")
        # Yahoo stamps each bar at the session open in UTC; the production
        # parser normalises that back to a trade date, and that normalisation
        # is part of what these tests cover, so the fixture reproduces it.
        ts = [int(pd.Timestamp(i).tz_localize("UTC").timestamp()) + 13 * 3600
              for i in df.index]
        payload = {"chart": {"result": [{
            "meta": {"symbol": symbol, "regularMarketPrice": float(df["Close"].iloc[-1])},
            "timestamp": ts,
            "indicators": {
                "quote": [{
                    "open": df["Open"].round(4).tolist(),
                    "high": df["High"].round(4).tolist(),
                    "low": df["Low"].round(4).tolist(),
                    "close": df["Close"].round(4).tolist(),
                    "volume": df["Volume"].tolist(),
                }],
                # Yahoo returns `adjclose` for every symbol it serves —
                # checked against the live endpoint for equities, ETFs and the
                # ^VIX family, which have no dividends and still carry the
                # field. Omitting it here made the harness unfaithful in a way
                # that mattered: the production parser treats a missing
                # adjusted close as a disclosed degradation, so every
                # synthetic run was marked degraded for a reason that cannot
                # occur against the real provider.
                #
                # The fixture pays no dividends, so adjclose equals close and
                # the adjustment factor is exactly 1.0. That is the point: it
                # exercises the adjustment path without changing any price a
                # test asserts on.
                "adjclose": [{
                    "adjclose": df["Close"].round(4).tolist(),
                }],
            },
        }], "error": None}}

        return SimpleNamespace(
            status_code=200,
            json=lambda: payload,
            raise_for_status=lambda: None,
            text="",
        )


#: Modules that imported `cache_get_entry`/`cache_set` by name. The SQLite
#: response cache is shared with the developer's real runs, so a test that left
#: it enabled would read whatever happened to be cached and stop being
#: deterministic — or worse, pass against live data and fail on a fresh clone.
_CACHE_CONSUMERS = (
    "backend.data.market_data",
    "backend.data.options_chain",
    "backend.data.vix_data",
    "backend.data.covariance",
    "backend.data.earnings_calendar",
)


def install(monkeypatch, cfg: Optional[ChainConfig] = None) -> FakeYahoo:
    """
    Replace the network boundary with the synthetic market and disable caching.

    Everything above the boundary — parsing, normalisation, IV interpolation,
    liquidity assessment, data-quality recording — runs unmodified.
    """
    import importlib
    import yfinance

    market = FakeYahoo(cfg=cfg or ChainConfig())

    monkeypatch.setattr(yfinance, "Ticker", market.ticker, raising=True)

    md = importlib.import_module("backend.data.market_data")
    monkeypatch.setattr(md._SESSION, "get", market.session_get, raising=True)

    for name in _CACHE_CONSUMERS:
        try:
            mod = importlib.import_module(name)
        except ImportError:                     # optional module
            continue
        if hasattr(mod, "cache_get_entry"):
            monkeypatch.setattr(mod, "cache_get_entry",
                                lambda *a, **k: None, raising=True)
        if hasattr(mod, "cache_get"):
            monkeypatch.setattr(mod, "cache_get",
                                lambda *a, **k: None, raising=True)
        if hasattr(mod, "cache_set"):
            monkeypatch.setattr(mod, "cache_set",
                                lambda *a, **k: None, raising=True)

    # Earnings dates are a separate provider; a fixed "no earnings in the
    # window" keeps the earnings warning out of the scoring comparisons. The
    # earnings path has its own tests.
    try:
        ec = importlib.import_module("backend.data.earnings_calendar")
        monkeypatch.setattr(ec, "get_next_earnings_date",
                            lambda ticker: None, raising=False)
        monkeypatch.setattr(ec, "crosses_earnings",
                            lambda *a, **k: None, raising=False)
    except ImportError:
        pass

    return market


def market_data_for(tickers) -> Dict[str, dict]:
    """
    The `market_data` mapping the orchestrator takes, built from the fixture.

    Mirrors what `routes_portfolio.fetch_market_data` assembles, so a test can
    drive the orchestrator directly without going through the route.
    """
    out: Dict[str, dict] = {}
    for t in tickers:
        hist = ohlcv(t, "1y")
        log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        out[t] = {
            "spot": base_price(t),
            "volatility": float(log_ret.rolling(20).std().iloc[-1] * math.sqrt(252)),
            "dividend_yield": DIVIDEND_YIELD,
        }
    out["_global"] = {"risk_free_rate": RISK_FREE}
    return out
