"""
Layer 3 — Per-Stock Risk Analysis (Step 3 in pipeline)

For EACH holding in the portfolio:
  - Beta vs SPY
  - Historical VaR + CVaR (95%, scaled to hedge horizon)
  - GARCH vol forecast (per stock)
  - Min-variance hedge ratio h*
  - OLS regression hedge ratio + R² (basis risk)
  - Tail correlation vs SPY (bottom 10%)
  - Full factor decomposition (sector, rate, credit, USD sensitivity)

Uses company metadata from cleaned_companies.csv to enrich
sector-specific hedge logic downstream (Layer 5).
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict
from backend.models.portfolio_models import PortfolioInput, PortfolioPosition
from backend.models.risk_models import RegimeState, RiskProfile, PortfolioRiskSummary, PortfolioGreeks

logger = logging.getLogger(__name__)

# Default hedge instruments by asset class for OLS R² calculation
BENCHMARK_HEDGE_MAP = {
    "equity":    "SPY",
    "commodity": "DJP",     # Bloomberg Commodity ETF proxy
    "bond":      "TLT",
    "fx":        "UUP",
    "crypto":    "BTC-USD",
}


class RiskAnalysisEngine:

    def __init__(self):
        from backend.core.risk.factor_decomposer import FactorDecomposer
        from backend.core.risk.var_calculator import VaRCalculator
        from backend.core.risk.hedge_ratio import HedgeRatioCalculator
        self.decomposer = FactorDecomposer()
        self.var_calc   = VaRCalculator()
        self.hr_calc    = HedgeRatioCalculator()

    def analyze(
        self,
        portfolio: PortfolioInput,
        regime: RegimeState,
    ) -> PortfolioRiskSummary:
        """
        Run full per-stock risk analysis for the portfolio.
        Returns PortfolioRiskSummary with one RiskProfile per holding.
        """
        from backend.data.market_data import get_returns, get_current_price

        risk_profiles: List[RiskProfile] = []
        returns_dict: Dict[str, pd.Series] = {}

        for holding in portfolio.stock_positions:
            ticker = holding.ticker
            logger.info(f"Analyzing risk for {ticker}...")

            try:
                profile = self._analyze_holding(holding, portfolio, regime)
                risk_profiles.append(profile)

                # Cache returns for portfolio-level VaR
                returns_dict[ticker] = get_returns(ticker)

            except Exception as e:
                logger.error(f"Risk analysis failed for {ticker}: {e}")
                # Graceful degradation: create minimal profile
                risk_profiles.append(self._fallback_profile(holding, portfolio))

        # Options positions — compute delta-adjusted exposure as a risk profile
        for opt in portfolio.options_positions:
            ticker_key = f"{opt.ticker}_{opt.option_type[0].upper()}{int(opt.strike)}"
            logger.info(f"Analyzing risk for options position {ticker_key}...")
            try:
                profile = self._analyze_option_holding(opt, portfolio, regime)
                risk_profiles.append(profile)
                # Use underlying's returns as delta-approximation for portfolio VaR
                returns_dict[ticker_key] = get_returns(opt.ticker)
            except Exception as e:
                logger.error(f"Risk analysis failed for {ticker_key}: {e}")

        # Portfolio-level metrics
        portfolio_beta = self._compute_portfolio_beta(risk_profiles, portfolio)
        port_var = self._compute_portfolio_var(returns_dict, portfolio)
        corr_matrix, tickers = self._compute_correlation_matrix(returns_dict)
        port_greeks = self._compute_portfolio_greeks(portfolio, regime)

        return PortfolioRiskSummary(
            total_notional=portfolio.total_notional,
            portfolio_beta=portfolio_beta,
            portfolio_var_5pct=port_var["var_dollar"],
            portfolio_cvar_5pct=port_var["cvar_dollar"],
            concentration_top1=self._concentration(risk_profiles, portfolio),
            risk_profiles=risk_profiles,
            correlation_matrix=corr_matrix,
            tickers=tickers,
            portfolio_greeks=port_greeks,
        )

    def _analyze_holding(
        self,
        holding,
        portfolio: PortfolioInput,
        regime: RegimeState,
    ) -> RiskProfile:
        from backend.data.market_data import get_returns, get_current_price
        from backend.ml.volatility.garch_model import GARCHVolModel

        ticker = holding.ticker
        asset_class = holding.asset_class
        current_price = get_current_price(ticker)
        notional = current_price * holding.shares

        # Returns
        returns = get_returns(ticker, period="2y")
        spy_returns = get_returns("SPY", period="2y")

        # Beta
        beta = self.decomposer.compute_beta(ticker, "SPY")

        # GARCH vol (per stock) — computed first so it feeds into VaR scaling
        try:
            garch_model = GARCHVolModel()
            garch_model.fit(returns)
            garch_vol = garch_model.forecast_vol()
        except Exception:
            from backend.data.market_data import get_historical_volatility
            garch_vol = get_historical_volatility(ticker)

        # VaR / CVaR — pass GARCH vol for forward-looking horizon scaling
        var_result = self.var_calc.historical_var(
            returns,
            notional=notional,
            confidence=0.95,
            horizon_days=portfolio.hedge_horizon_days,
            garch_vol=garch_vol,
        )

        # Hedge instrument for this asset class
        hedge_instrument = BENCHMARK_HEDGE_MAP.get(asset_class, "SPY")

        # OLS hedge ratio + R² (basis risk)
        try:
            ols_result = self.decomposer.compute_ols_hedge_ratio_with_r2(
                ticker, hedge_instrument
            )
            ols_ratio = ols_result["hedge_ratio"]
            basis_r2  = ols_result["r2"]
        except Exception:
            ols_ratio = beta
            basis_r2  = 0.5

        # Min-variance hedge ratio h*
        try:
            opt_ratio = self.decomposer.compute_min_variance_hedge_ratio(
                ticker, hedge_instrument
            )
        except Exception:
            opt_ratio = ols_ratio

        # Tail correlation (bottom 10%)
        common_idx = returns.index.intersection(spy_returns.index)
        tail_corr = self.var_calc.tail_correlation(
            returns.loc[common_idx],
            spy_returns.loc[common_idx],
            percentile=0.10,
        )

        # Factor decomposition
        factors = self.decomposer.decompose_factors(ticker, notional)

        return RiskProfile(
            ticker=ticker,
            asset_class=asset_class,
            notional_value=round(notional, 2),
            beta_vs_spy=beta,
            small_cap_beta=factors.get("small_cap_beta", 0.0),
            value_beta=0.0,
            rate_sensitivity=factors.get("rate_sensitivity", 0.0),
            credit_spread_sensitivity=factors.get("credit_spread_sensitivity", 0.0),
            usd_sensitivity=factors.get("usd_sensitivity", 0.0),
            factor_contributions=factors.get("factor_contributions", {}),
            var_5pct=var_result["var_dollar"],
            cvar_5pct=var_result["cvar_dollar"],
            var_pct=var_result["var_pct"],
            cvar_pct=var_result["cvar_pct"],
            notional_at_risk=var_result["var_dollar"],
            optimal_hedge_ratio=opt_ratio,
            regression_hedge_ratio=ols_ratio,
            tail_correlation=tail_corr,
        )

    def _analyze_option_holding(
        self,
        opt,            # OptionsPosition
        portfolio: PortfolioInput,
        regime: RegimeState,
    ) -> RiskProfile:
        """
        Build a RiskProfile for an existing options position.

        The effective equity exposure = |delta| × 100 × spot × contracts.
        This is what needs to be hedged, just like a stock notional.

        ticker key = "{underlying}_{C|P}{strike}"  e.g. "SPY_C410"
        """
        from backend.data.market_data import (
            get_current_price, get_risk_free_rate,
            get_historical_volatility, get_dividend_yield,
        )
        from backend.instruments.options.bsm_pricer import bsm_greeks
        from datetime import date as _date

        ticker = opt.ticker
        spot   = get_current_price(ticker)
        r      = get_risk_free_rate()

        try:
            q = get_dividend_yield(ticker)
        except Exception:
            q = 0.0

        try:
            sigma = get_historical_volatility(ticker)
        except Exception:
            sigma = 0.20

        today = _date.today()
        T = max((opt.expiry - today).days / 365.0, 1.0 / 365.0)

        greeks = bsm_greeks(spot, opt.strike, r, q, sigma, T, opt.option_type)
        delta  = greeks["delta"]   # calls > 0, puts < 0

        # direction sign: long → hold as-is, short → flip sign
        direction_sign    = 1 if opt.direction == "long" else -1
        effective_delta   = delta * direction_sign

        # Dollar exposure equivalent: |Δ| × 100 shares × spot × contracts
        effective_notional = abs(effective_delta) * 100 * spot * opt.contracts

        # Beta of the underlying vs SPY
        try:
            beta = self.decomposer.compute_beta(ticker, "SPY")
        except Exception:
            beta = 1.0

        # Delta-VaR and CVaR: use VaRCalculator on underlying returns.
        # effective_notional already encodes |Δ|×100×spot×contracts, so
        # historical_var on the underlying returns × that notional gives correct
        # dollar VaR/CVaR with Cornish-Fisher tail correction.
        try:
            from backend.data.market_data import get_returns
            ret_series = get_returns(ticker, period="2y")
            var_result = self.var_calc.historical_var(
                ret_series,
                notional=max(effective_notional, 1.0),
                confidence=0.95,
                horizon_days=portfolio.hedge_horizon_days,
            )
            var_dollar  = var_result["var_dollar"]
            cvar_dollar = var_result["cvar_dollar"]
        except Exception:
            # Parametric fallback
            horizon_T     = portfolio.hedge_horizon_days / 252.0
            price_move_95 = spot * sigma * (horizon_T ** 0.5) * 1.645
            var_dollar    = abs(effective_delta) * price_move_95 * 100 * opt.contracts
            cvar_dollar   = var_dollar * 1.25

        safe_notional = max(effective_notional, 1.0)
        ticker_key    = f"{ticker}_{opt.option_type[0].upper()}{int(opt.strike)}"
        delta_sign    = 1 if effective_delta >= 0 else -1

        return RiskProfile(
            ticker             = ticker_key,
            asset_class        = "equity",
            notional_value     = round(effective_notional, 2),
            beta_vs_spy        = round(beta, 4),
            var_5pct           = round(var_dollar, 2),
            cvar_5pct          = round(cvar_dollar, 2),
            var_pct            = round(var_dollar / safe_notional, 4),
            cvar_pct           = round(cvar_dollar / safe_notional, 4),
            notional_at_risk   = round(var_dollar, 2),
            optimal_hedge_ratio    = 1.0,
            regression_hedge_ratio = round(beta, 4),
            tail_correlation       = 0.8,
            effective_delta_sign   = delta_sign,
        )

    def _fallback_profile(self, holding, portfolio: PortfolioInput) -> RiskProfile:
        """Minimal safe profile used when data fetch fails for a ticker."""
        notional = holding.shares * holding.purchase_price
        return RiskProfile(
            ticker=holding.ticker,
            asset_class=holding.asset_class,
            notional_value=notional,
            beta_vs_spy=1.0,
            var_5pct=notional * 0.15,
            cvar_5pct=notional * 0.20,
            var_pct=0.15,
            cvar_pct=0.20,
            notional_at_risk=notional * 0.15,
            optimal_hedge_ratio=1.0,
            regression_hedge_ratio=1.0,
            tail_correlation=0.7,
        )

    def _compute_portfolio_beta(
        self,
        profiles: List[RiskProfile],
        portfolio: PortfolioInput,
    ) -> float:
        total = portfolio.total_notional
        if total == 0:
            return 1.0
        weighted = sum(p.beta_vs_spy * p.notional_value for p in profiles)
        return round(weighted / total, 4)

    def _compute_portfolio_var(
        self,
        returns_dict: Dict[str, pd.Series],
        portfolio: PortfolioInput,
    ) -> Dict:
        from backend.data.market_data import get_current_price

        if not returns_dict:
            return {"var_dollar": 0.0, "cvar_dollar": 0.0}

        tickers = list(returns_dict.keys())
        # Align all return series on common dates
        df = pd.DataFrame({t: returns_dict[t] for t in tickers}).dropna()
        if df.empty:
            return {"var_dollar": 0.0, "cvar_dollar": 0.0}

        # Use current market price, not purchase price, for accurate notional weights
        def _current_notional(ticker: str) -> float:
            match = next((h for h in portfolio.stock_positions if h.ticker == ticker), None)
            if match is None:
                return 0.0
            try:
                return match.shares * get_current_price(ticker)
            except Exception:
                return match.shares * match.purchase_price   # graceful fallback

        weights = np.array([_current_notional(t) for t in tickers])
        total_w = weights.sum()
        if total_w == 0:
            return {"var_dollar": 0.0, "cvar_dollar": 0.0}
        weights = weights / total_w

        return self.var_calc.portfolio_var(
            df, weights,
            notional=portfolio.total_notional,
            horizon_days=portfolio.hedge_horizon_days,
        )

    def _compute_correlation_matrix(
        self,
        returns_dict: Dict[str, pd.Series],
    ):
        if len(returns_dict) < 2:
            tickers = list(returns_dict.keys())
            return [[1.0]], tickers

        df = pd.DataFrame(returns_dict).dropna()
        corr = df.corr().values.tolist()
        return corr, list(df.columns)

    def _compute_portfolio_greeks(
        self,
        portfolio: PortfolioInput,
        regime: RegimeState,
    ) -> PortfolioGreeks:
        """
        Aggregate dollar-denominated Greeks for the entire portfolio.

        Stock positions:
          - Contribute delta only: dollar_delta += shares × spot × 1.0
          - No gamma/vega/theta (stocks have no optionality)

        Options positions:
          - Recompute BSM Greeks on the effective notional
          - dollar_delta += effective_delta × spot × 100 × contracts  (signed)
          - dollar_gamma += gamma × spot² × 0.01 × 100 × contracts   (per 1% move)
          - dollar_vega  += vega  × 0.01 × 100 × contracts           (per 1 vol-pt)
          - dollar_theta += theta × 100 × contracts                   (daily decay)

        dominant_risk: which Greek has the largest absolute $ exposure.
        """
        from backend.data.market_data import get_current_price, get_risk_free_rate, get_dividend_yield, get_historical_volatility
        from backend.instruments.options.bsm_pricer import bsm_greeks
        from datetime import date as _date

        dollar_delta = 0.0
        dollar_gamma = 0.0
        dollar_vega  = 0.0
        dollar_theta = 0.0

        # ── Stock positions ───────────────────────────────────────────────────
        for h in portfolio.stock_positions:
            try:
                spot = get_current_price(h.ticker)
                # Long stock = delta of 1.0 per share; short = -1.0
                direction = 1.0 if getattr(h, "direction", "long") == "long" else -1.0
                dollar_delta += direction * h.shares * spot
            except Exception:
                pass

        # ── Options positions ─────────────────────────────────────────────────
        r = get_risk_free_rate()
        today = _date.today()

        for opt in portfolio.options_positions:
            try:
                spot  = get_current_price(opt.ticker)
                q     = get_dividend_yield(opt.ticker)
                sigma = get_historical_volatility(opt.ticker)
                T     = max((opt.expiry - today).days / 365.0, 1.0 / 365.0)

                g = bsm_greeks(spot, opt.strike, r, q, sigma, T, opt.option_type)

                direction = 1.0 if opt.direction == "long" else -1.0
                multiplier = 100 * opt.contracts * direction

                # dollar_delta: $ change per $1 move in underlying
                dollar_delta += g["delta"]  * multiplier * spot
                # dollar_gamma: $ change in delta per 1% spot move
                #   gamma (BSM) = Δ²S / 1-pt; per 1% move: gamma × spot × 0.01 × spot
                dollar_gamma += g["gamma"]  * spot * spot * 0.01 * abs(multiplier)
                # dollar_vega: $ change per 1 vol-point (1%) move in IV
                dollar_vega  += g["vega"]   * multiplier * 0.01
                # dollar_theta: daily $ decay
                dollar_theta += g["theta"]  * multiplier

            except Exception:
                continue

        # ── Dominant risk ─────────────────────────────────────────────────────
        abs_exposures = {
            "delta": abs(dollar_delta),
            "gamma": abs(dollar_gamma),
            "vega":  abs(dollar_vega),
        }
        dominant_risk = max(abs_exposures, key=abs_exposures.get)

        net_delta_pct = (
            dollar_delta / portfolio.total_notional
            if portfolio.total_notional > 0 else 0.0
        )

        return PortfolioGreeks(
            dollar_delta  = round(dollar_delta,  2),
            dollar_gamma  = round(dollar_gamma,  2),
            dollar_vega   = round(dollar_vega,   2),
            dollar_theta  = round(dollar_theta,  2),
            net_delta_pct = round(net_delta_pct, 4),
            dominant_risk = dominant_risk,
        )

    def _concentration(
        self,
        profiles: List[RiskProfile],
        portfolio: PortfolioInput,
    ) -> float:
        if not profiles or portfolio.total_notional == 0:
            return 0.0
        max_notional = max(p.notional_value for p in profiles)
        return round(max_notional / portfolio.total_notional, 4)
