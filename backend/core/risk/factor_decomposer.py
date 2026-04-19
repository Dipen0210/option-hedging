"""
Factor decomposer: computes beta, OLS hedge ratio, min-variance hedge ratio,
and full factor exposures for a single holding.
Pure quant math — no ML here.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional


class FactorDecomposer:

    def compute_beta(
        self,
        ticker: str,
        benchmark: str = "SPY",
        window_days: int = 252,
    ) -> float:
        """
        Beta via OLS: β = Cov(r_i, r_b) / Var(r_b)
        Uses up to `window_days` of daily log returns.
        """
        from backend.data.market_data import get_returns
        r_asset = get_returns(ticker).iloc[-window_days:]
        r_bench = get_returns(benchmark).iloc[-window_days:]

        # Align on common dates
        common = r_asset.index.intersection(r_bench.index)
        r_asset = r_asset.loc[common]
        r_bench = r_bench.loc[common]

        if len(r_asset) < 30:
            return 1.0   # default if insufficient data

        slope, _, _, _, _ = stats.linregress(r_bench.values, r_asset.values)
        return round(float(slope), 4)

    def compute_ols_hedge_ratio(
        self,
        asset_ticker: str,
        hedge_ticker: str,
        window_days: int = 252,
    ) -> float:
        """
        Empirical hedge ratio via OLS regression.
        ΔS = α + h·ΔF  →  h = slope coefficient
        More accurate than theoretical beta for proxy hedges.
        Returns R² as second value via compute_ols_hedge_ratio_with_r2().
        """
        from backend.data.market_data import get_returns
        r_asset = get_returns(asset_ticker).iloc[-window_days:]
        r_hedge = get_returns(hedge_ticker).iloc[-window_days:]

        common = r_asset.index.intersection(r_hedge.index)
        r_asset = r_asset.loc[common]
        r_hedge = r_hedge.loc[common]

        if len(r_asset) < 30:
            return 1.0

        slope, _, _, _, _ = stats.linregress(r_hedge.values, r_asset.values)
        return round(float(slope), 4)

    def compute_ols_hedge_ratio_with_r2(
        self,
        asset_ticker: str,
        hedge_ticker: str,
        window_days: int = 252,
    ) -> Dict[str, float]:
        """Returns both hedge ratio and R² (basis risk measure)."""
        from backend.data.market_data import get_returns
        r_asset = get_returns(asset_ticker).iloc[-window_days:]
        r_hedge = get_returns(hedge_ticker).iloc[-window_days:]

        common = r_asset.index.intersection(r_hedge.index)
        r_asset = r_asset.loc[common]
        r_hedge = r_hedge.loc[common]

        if len(r_asset) < 30:
            return {"hedge_ratio": 1.0, "r2": 0.0}

        slope, intercept, r_value, _, _ = stats.linregress(r_hedge.values, r_asset.values)
        return {
            "hedge_ratio": round(float(slope), 4),
            "r2": round(float(r_value ** 2), 4),
        }

    def compute_min_variance_hedge_ratio(
        self,
        asset_ticker: str,
        futures_ticker: str,
        window_days: int = 252,
    ) -> float:
        """
        h* = ρ(S,F) × (σ_S / σ_F)
        Use for commodity proxy hedges (e.g. PDBC → CL futures).
        """
        from backend.data.market_data import get_returns
        r_asset = get_returns(asset_ticker).iloc[-window_days:]
        r_futures = get_returns(futures_ticker).iloc[-window_days:]

        common = r_asset.index.intersection(r_futures.index)
        r_asset = r_asset.loc[common]
        r_futures = r_futures.loc[common]

        if len(r_asset) < 30:
            return 1.0

        sigma_s = float(r_asset.std())
        sigma_f = float(r_futures.std())
        rho = float(np.corrcoef(r_asset.values, r_futures.values)[0, 1])

        if sigma_f == 0:
            return 1.0

        return round(rho * (sigma_s / sigma_f), 4)

    def decompose_factors(self, ticker: str, notional: float) -> Dict:
        """
        Full factor decomposition for a single holding.
        Uses sector data from company_data.py to enrich context.
        """
        from backend.data.company_data import get_company
        from backend.data.market_data import get_returns

        beta_spy = self.compute_beta(ticker, "SPY")

        # Rate sensitivity: beta vs TLT (inverted — TLT moves opposite to rates)
        beta_tlt = self.compute_beta(ticker, "TLT")

        # Credit sensitivity: beta vs HYG
        beta_hyg = self.compute_beta(ticker, "HYG")

        # Small-cap proxy: beta vs IWM
        beta_iwm = self.compute_beta(ticker, "IWM")

        # USD sensitivity: beta vs UUP (USD ETF)
        try:
            beta_usd = self.compute_beta(ticker, "UUP")
        except Exception:
            beta_usd = 0.0

        # Factor contributions (% of total variance explained)
        r_asset = get_returns(ticker).iloc[-252:]
        total_var = float(r_asset.var()) if len(r_asset) > 5 else 1.0

        company = get_company(ticker)
        sector = company.sector if company else "Unknown"
        mkt_cap_cat = company.market_cap_category if company else "unknown"

        return {
            "ticker": ticker,
            "sector": sector,
            "market_cap_category": mkt_cap_cat,
            "notional": notional,
            "systematic_beta": beta_spy,
            "small_cap_beta": round(beta_iwm - beta_spy, 4),  # SMB proxy
            "rate_sensitivity": round(-beta_tlt, 4),          # negative: rate-sensitive = TLT negative beta
            "credit_spread_sensitivity": round(beta_hyg, 4),
            "usd_sensitivity": round(beta_usd, 4),
            "factor_contributions": {
                "market": round(min(abs(beta_spy) * 0.6, 1.0), 3),
                "sector": round(0.2, 3),
                "idiosyncratic": round(max(0.2, 1.0 - abs(beta_spy) * 0.6), 3),
            },
        }
