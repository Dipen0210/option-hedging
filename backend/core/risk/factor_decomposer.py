"""
Factor decomposer: computes beta, OLS hedge ratio, min-variance hedge ratio,
and factor exposures for a single holding.
Pure quant math — no ML here.

Three things were wrong with how every estimate in this module was built.

**The window was taken before the alignment.** Each leg was cut to its own
last `window_days` rows and only then intersected, so whenever the two series
ended on different dates — a missing final bar, a different listing history, a
tz-aware index on one path and a naive one on the other after commits
`0454bb7`/`d0acc54` added a second Yahoo fetch route — the two windows covered
different calendar periods. The intersection then silently shrank, and in the
tz-mismatch case collapsed to nothing. Alignment now happens first, through
`align_series`, which is the project's one cross-asset join and normalises
both indexes; the window is taken from the aligned frame.

**An unmeasurable beta was reported as 1.0.** `return 1.0` on fewer than 30
observations is not a neutral default: it is the specific claim that the
holding moves one-for-one with the benchmark, and it flows into
`RiskProfile.beta_vs_spy` and from there into the hedge ratio that sizes every
cross hedge. The same line served `compute_beta(ticker, "TLT")`, so a failed
measurement became `rate_sensitivity = -1.0` — a strong rate exposure
asserted on no data at all. These now raise `InsufficientFactorData`, and the
caller discloses the gap instead of trading on a placeholder.

**The variance decomposition was invented.** `factor_contributions` reported
`market = |beta| x 0.6`, `sector = 0.2` as a literal constant, and
`idiosyncratic = 1 - |beta| x 0.6`, labelled "% of total variance explained".
None of those is a variance share; `|beta| x 0.6` is not a quantity at all,
and a hardcoded 0.2 is a fabricated financial result. The shares are now the
regression R² against the benchmark and its complement, which is what a
one-factor variance decomposition actually is. Sector is not reported, because
nothing in this codebase maps a ticker to a sector return series, and omitting
a number is the only honest alternative to making one up.
"""
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Minimum aligned observations before a regression estimate is reported.
# Thirty daily returns is six weeks: enough to fit a line, nowhere near enough
# to trust its slope. A beta over 30 points carries a standard error roughly
# 2.9x that of the same estimate over 252.
MIN_REGRESSION_OBS = 120

DEFAULT_WINDOW_DAYS = 252


class InsufficientFactorData(Exception):
    """
    Not enough overlapping history to estimate a factor exposure.

    Raised rather than returning a placeholder, because every caller of these
    methods feeds the result into position sizing or risk disclosure, where
    "1.0" and "unknown" lead to materially different trades.
    """


class FactorDecomposer:

    # ── Shared data path ─────────────────────────────────────────────────────

    @staticmethod
    def _aligned(
        asset_ticker: str,
        hedge_ticker: str,
        window_days: int = DEFAULT_WINDOW_DAYS,
    ) -> pd.DataFrame:
        """
        Overlapping daily returns for two tickers, newest `window_days` rows.

        Aligned before being windowed, and through `align_series` rather than
        a bare index intersection, so a tz-aware and a tz-naive index — or
        midnight-stamped and intraday-stamped dates for the same session —
        still match instead of yielding an almost-empty frame.
        """
        from backend.data.market_data import align_series, get_returns

        aligned = align_series(
            asset=get_returns(asset_ticker),
            hedge=get_returns(hedge_ticker),
        ).iloc[-window_days:]

        if len(aligned) < MIN_REGRESSION_OBS:
            raise InsufficientFactorData(
                f"{asset_ticker} vs {hedge_ticker}: {len(aligned)} aligned "
                f"observations, fewer than the {MIN_REGRESSION_OBS} required "
                f"to estimate a factor exposure"
            )
        return aligned

    @classmethod
    def _regress(
        cls,
        asset_ticker: str,
        hedge_ticker: str,
        window_days: int = DEFAULT_WINDOW_DAYS,
    ) -> Dict[str, float]:
        """
        OLS of asset returns on hedge returns, with the estimate's own error.

        `stderr` is reported because a hedge ratio is an estimate, not a
        measurement, and the caller is sizing a real position with it.
        """
        aligned = cls._aligned(asset_ticker, hedge_ticker, window_days)
        x = aligned["hedge"].values
        y = aligned["asset"].values

        if float(np.std(x)) == 0.0:
            raise InsufficientFactorData(
                f"{hedge_ticker} returns have zero variance over the window; "
                f"no slope is defined"
            )

        slope, intercept, r_value, _, stderr = stats.linregress(x, y)
        return {
            "slope":       round(float(slope), 4),
            "intercept":   round(float(intercept), 6),
            "r2":          round(float(r_value ** 2), 4),
            "stderr":      round(float(stderr), 4),
            "n_obs":       int(len(aligned)),
            "window_days": int(window_days),
        }

    # ── Beta ─────────────────────────────────────────────────────────────────

    def compute_beta(
        self,
        ticker: str,
        benchmark: str = "SPY",
        window_days: int = DEFAULT_WINDOW_DAYS,
    ) -> float:
        """
        Beta via OLS: beta = Cov(r_i, r_b) / Var(r_b).

        Raises `InsufficientFactorData` when it cannot be estimated. Callers
        must decide what to disclose; this function will not answer "1.0".
        """
        return self._regress(ticker, benchmark, window_days)["slope"]

    def compute_beta_with_error(
        self,
        ticker: str,
        benchmark: str = "SPY",
        window_days: int = DEFAULT_WINDOW_DAYS,
    ) -> Dict[str, float]:
        """Beta plus R², standard error and the sample it rests on."""
        out = self._regress(ticker, benchmark, window_days)
        out["beta"] = out.pop("slope")
        return out

    # ── Hedge ratios ─────────────────────────────────────────────────────────

    def compute_ols_hedge_ratio(
        self,
        asset_ticker: str,
        hedge_ticker: str,
        window_days: int = DEFAULT_WINDOW_DAYS,
    ) -> float:
        """
        Empirical hedge ratio via OLS: dS = alpha + h x dF, h = the slope.
        More accurate than theoretical beta for proxy hedges.
        """
        return self._regress(asset_ticker, hedge_ticker, window_days)["slope"]

    def compute_ols_hedge_ratio_with_r2(
        self,
        asset_ticker: str,
        hedge_ticker: str,
        window_days: int = DEFAULT_WINDOW_DAYS,
    ) -> Dict[str, float]:
        """
        Hedge ratio and R², with the provenance of the estimate.

        R² is the basis-risk measure the scorer consumes, and it is also the
        share of the asset's variance the hedge explains — so `n_obs` and
        `stderr` travel with it rather than letting a 120-point R² read like
        a 1,000-point one.
        """
        out = self._regress(asset_ticker, hedge_ticker, window_days)
        return {
            "hedge_ratio": out["slope"],
            "r2":          out["r2"],
            "stderr":      out["stderr"],
            "n_obs":       out["n_obs"],
            "window_days": out["window_days"],
        }

    def compute_min_variance_hedge_ratio(
        self,
        asset_ticker: str,
        futures_ticker: str,
        window_days: int = DEFAULT_WINDOW_DAYS,
    ) -> float:
        """
        h* = rho(S,F) x (sigma_S / sigma_F).
        Use for commodity proxy hedges (e.g. PDBC → CL futures).
        """
        aligned = self._aligned(asset_ticker, futures_ticker, window_days)

        sigma_s = float(aligned["asset"].std())
        sigma_f = float(aligned["hedge"].std())
        if sigma_f == 0:
            raise InsufficientFactorData(
                f"{futures_ticker} returns have zero variance over the "
                f"window; h* is undefined"
            )

        rho = float(aligned["asset"].corr(aligned["hedge"]))
        if pd.isna(rho):
            raise InsufficientFactorData(
                f"{asset_ticker} vs {futures_ticker}: correlation is undefined"
            )

        return round(rho * (sigma_s / sigma_f), 4)

    # ── Factor decomposition ─────────────────────────────────────────────────

    def decompose_factors(self, ticker: str, notional: float) -> Dict:
        """
        Factor exposures for a single holding.

        Each exposure is attempted independently: a ticker with no overlapping
        UUP history should still get its SPY beta. An exposure that cannot be
        measured is reported as None and listed in `unavailable`, never as a
        number — `rate_sensitivity: None` and `rate_sensitivity: 0.0` are
        different claims, and the second one is a measurement that was never
        taken.
        """
        from backend.data.company_data import get_company

        exposures: Dict[str, Optional[float]] = {}
        unavailable: Dict[str, str] = {}

        def beta_against(name: str, benchmark: str) -> Optional[float]:
            try:
                return self.compute_beta(ticker, benchmark)
            except InsufficientFactorData as exc:
                unavailable[name] = str(exc)
                logger.info("factors: %s unavailable — %s", name, exc)
                return None
            except Exception as exc:
                unavailable[name] = f"{benchmark} data unavailable: {exc}"
                logger.warning(
                    "factors: %s vs %s failed: %s", ticker, benchmark, exc
                )
                return None

        # Measured first so the market R² below can reuse the same regression.
        try:
            market = self._regress(ticker, "SPY")
            beta_spy: Optional[float] = market["slope"]
            market_r2: Optional[float] = market["r2"]
            market_n_obs: Optional[int] = int(market["n_obs"])
            market_stderr: Optional[float] = market["stderr"]
        except Exception as exc:
            unavailable["systematic_beta"] = str(exc)
            logger.warning("factors: %s vs SPY failed: %s", ticker, exc)
            beta_spy = market_r2 = market_n_obs = market_stderr = None

        beta_tlt = beta_against("rate_sensitivity", "TLT")
        beta_hyg = beta_against("credit_spread_sensitivity", "HYG")
        beta_iwm = beta_against("small_cap_beta", "IWM")
        beta_usd = beta_against("usd_sensitivity", "UUP")

        company = get_company(ticker)

        # A one-factor variance decomposition is R² and its complement. That
        # is the whole of what a single benchmark regression supports; the
        # previous `sector: 0.2` constant was not derived from anything.
        contributions: Dict[str, float] = {}
        if market_r2 is not None:
            contributions = {
                "market":        round(market_r2, 3),
                "idiosyncratic": round(1.0 - market_r2, 3),
            }

        return {
            "ticker": ticker,
            "sector": company.sector if company else "Unknown",
            "market_cap_category": company.market_cap_category if company else "unknown",
            "notional": notional,
            "systematic_beta": beta_spy,
            # SMB proxy. Only meaningful when both legs were measured.
            "small_cap_beta": (
                round(beta_iwm - beta_spy, 4)
                if (beta_iwm is not None and beta_spy is not None) else None
            ),
            # Negative: a rate-sensitive holding has negative TLT beta.
            "rate_sensitivity": (
                round(-beta_tlt, 4) if beta_tlt is not None else None
            ),
            "credit_spread_sensitivity": beta_hyg,
            "usd_sensitivity": beta_usd,
            "factor_contributions": contributions,
            # Provenance for the market regression, so the caller can disclose
            # what the beta rests on instead of presenting it as exact.
            "beta_r2": market_r2,
            "beta_stderr": market_stderr,
            "beta_n_obs": market_n_obs,
            # Named exposures that could not be measured, and why.
            "unavailable": unavailable,
        }
