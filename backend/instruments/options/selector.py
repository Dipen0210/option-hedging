"""
Rule-based Options Instrument Selector (Layer 4 — Options branch).

Decision flow for each holding:
  1. Determine expiry window
       Stock position  → today … today + hedge_horizon_days
       Option position → today … position.expiry
  2. Fetch real option-chain expiry dates within that window (yfinance)
  3. Sample up to N representative expiry dates (short / mid / long)
  4. Direct hedge  — options on the underlying itself across sampled expiries
  5. Cross-hedge   — options on covariance-ranked ETFs (from covariance.py)
       Positive corr  → buy puts  on that ETF (opposite direction)
       Negative corr  → buy calls on that ETF (same direction — it rises when asset falls)
  6. Collar        — direct-hedge only when cost constraint is very tight
  7. Direction validation guard (safety net, Layer 4 also re-checks)
  8. Score, deduplicate, return top MAX_CANDIDATES

Scoring formula (0–100):
    score = 40 × (delta_offset / target_delta)     ← hedge effectiveness
          + 30 × (1 - cost_pct / max_cost_pct)     ← cost efficiency
          + 20 × basis_r2                           ← low basis risk
          + 10 × abs(lambda_)                       ← capital efficiency (capped)
"""
import logging
from datetime import date, timedelta
from typing import List, Dict, Optional

from backend.instruments.base import InstrumentSelector
from backend.instruments.options.bsm_pricer import bsm_price, bsm_greeks
from backend.instruments.options.binomial_pricer import crr_greeks, early_exercise_premium
from backend.instruments.options.black76_pricer import black76_price, black76_greeks
from backend.instruments.options.position_sizer import compute_position
from backend.instruments.options.greeks import net_greeks, collar_greeks
from backend.models.risk_models import RiskProfile, RegimeState
from backend.models.portfolio_models import PortfolioInput, StockPosition, OptionsPosition
from backend.models.hedge_models import InstrumentCandidate

logger = logging.getLogger(__name__)

MAX_CANDIDATES = 5

# Max expiries sampled per ticker (keeps candidate count manageable)
MAX_EXPIRIES_DIRECT = 3
MAX_EXPIRIES_CROSS  = 2

# Strike moneyness by upside_preservation (1.0 = ATM, >1.0 = OTM)
UPSIDE_STRIKE_MAP = [
    (0.00, 1.00),
    (0.30, 1.03),
    (0.50, 1.05),
    (0.70, 1.08),
    (1.00, 1.10),
]


class OptionsSelector(InstrumentSelector):

    def find_candidates(
        self,
        profile: RiskProfile,
        portfolio: PortfolioInput,
        regime: RegimeState,
    ) -> List[InstrumentCandidate]:
        """
        Generate and rank option hedge candidates for this holding.

        Hedge categories:
          "direct_hedge" — same underlying, opposite direction
          "cross_hedge"  — positively correlated ETF, opposite direction (puts)
          "macro_hedge"  — negatively correlated / macro asset, same direction (calls)
        """
        try:
            from backend.data.market_data import get_risk_free_rate
            from backend.data.options_chain import get_expiry_dates_until
            from backend.data.covariance import get_ranked_cross_hedges

            base_ticker = profile.ticker.split("_")[0]
            r = get_risk_free_rate()

            # ── Expiry window ─────────────────────────────────────────────────
            max_expiry = self._get_max_expiry(profile, portfolio)

            # For options positions the hedge must not expire before the position:
            # a hedge that expires first leaves the user unprotected at expiry.
            original_pos = self._find_original_position(portfolio, profile.ticker)
            min_expiry: Optional[date] = (
                original_pos.expiry if isinstance(original_pos, OptionsPosition) else None
            )

            candidates: List[InstrumentCandidate] = []

            # ── 1. Direct hedge (options on the underlying itself) ─────────────
            direct_expiries = get_expiry_dates_until(base_ticker, max_expiry)
            if min_expiry:
                direct_expiries = [e for e in direct_expiries if date.fromisoformat(e) >= min_expiry]
            if not direct_expiries:
                # Fallback: synthesise a date at hedge horizon
                direct_expiries = [(date.today() + timedelta(days=portfolio.hedge_horizon_days)).isoformat()]

            for expiry_str in self._sample_expiries(direct_expiries, MAX_EXPIRIES_DIRECT):
                exp_date = date.fromisoformat(expiry_str)
                T = max((exp_date - date.today()).days / 365.0, 1 / 365)
                cands = self._build_direct_hedge_candidates(
                    profile, portfolio, regime, base_ticker, r, T, exp_date,
                )
                candidates += cands

            # ── 2. Cross-hedge (covariance-ranked ETFs) ───────────────────────
            cross_ranked = get_ranked_cross_hedges(base_ticker)

            # Asset vol needed for h* sizing in cross-hedges
            try:
                from backend.data.market_data import get_historical_volatility
                sigma_asset = get_historical_volatility(base_ticker)
            except Exception:
                sigma_asset = None

            for ct in cross_ranked:
                cross_ticker = ct["ticker"]
                if cross_ticker.upper() == base_ticker.upper():
                    continue

                option_type  = "put" if ct["hedge_direction"] == "opposite" else "call"
                basis_r2     = ct["r2"]
                hedge_cat    = ct["hedge_category"]
                correlation  = ct["correlation"]

                try:
                    from backend.data.market_data import get_current_price, get_dividend_yield
                    spot       = get_current_price(cross_ticker)
                    q          = get_dividend_yield(cross_ticker)
                    sigma_hedge = self._get_vol(cross_ticker, profile, regime)

                    # h* = ρ × (σ_asset / σ_hedge) — minimum-variance hedge ratio
                    # specific to this cross-hedge instrument (not SPY beta)
                    if sigma_asset and sigma_hedge > 0:
                        h_star = abs(correlation) * (sigma_asset / sigma_hedge)
                    else:
                        h_star = profile.beta_vs_spy

                    cross_expiries = get_expiry_dates_until(cross_ticker, max_expiry)
                    if min_expiry:
                        cross_expiries = [e for e in cross_expiries if date.fromisoformat(e) >= min_expiry]
                    if not cross_expiries:
                        continue

                    for expiry_str in self._sample_expiries(cross_expiries, MAX_EXPIRIES_CROSS):
                        exp_date = date.fromisoformat(expiry_str)
                        T = max((exp_date - date.today()).days / 365.0, 1 / 365)

                        # Single-leg put or call
                        cands = self._build_single_leg_candidates(
                            profile, portfolio, regime,
                            cross_ticker, option_type,
                            spot, r, q, sigma_hedge, T, exp_date,
                            basis_r2=basis_r2,
                            hedge_category=hedge_cat,
                            correlation=correlation,
                            h_star=h_star,
                            expiry_str=expiry_str,
                        )
                        candidates += cands

                        # Bear put spread — only for put-direction cross-hedges in high-vol
                        if (option_type == "put"
                                and (regime.regime_label in ("high_vol", "anomaly")
                                     or profile.var_pct > 0.20)):
                            spread_cands = self._build_spread_candidates(
                                profile, portfolio, regime,
                                cross_ticker, spot, r, q, sigma_hedge, T, exp_date,
                                basis_r2=basis_r2, hedge_category=hedge_cat,
                                h_star=h_star, expiry_str=expiry_str,
                            )
                            candidates += spread_cands

                except Exception as e:
                    logger.error("Cross-hedge failed for %s: %s", cross_ticker, e)
                    continue

            # ── 3. Collar (direct hedge, tight-budget regime) ─────────────────
            if (not portfolio.upside_preservation
                    and portfolio.max_hedge_cost_pct < 0.015
                    and regime.vix_level > 25
                    and direct_expiries):
                exp_date = date.fromisoformat(direct_expiries[0])
                T = max((exp_date - date.today()).days / 365.0, 1 / 365)
                try:
                    from backend.data.market_data import get_current_price, get_dividend_yield
                    spot  = get_current_price(base_ticker)
                    q     = get_dividend_yield(base_ticker)
                    sigma = self._get_vol(base_ticker, profile, regime)
                    collar_cands = self._build_collar_candidates(
                        profile, portfolio, regime,
                        base_ticker, spot, r, q, sigma, T, exp_date,
                    )
                    for c in collar_cands:
                        c.hedge_category = "direct_hedge"
                    candidates += collar_cands
                except Exception as e:
                    logger.debug("Collar build failed for %s: %s", base_ticker, e)

            # ── Direction validation guard ─────────────────────────────────────
            user_sign = profile.effective_delta_sign
            valid: List[InstrumentCandidate] = []
            for c in candidates:
                if c.hedge_category == "direct_hedge":
                    if c.delta == 0 or (c.delta * user_sign < 0):
                        valid.append(c)
                    else:
                        logger.debug(
                            "OptionsSelector: dropped %s %s — delta=%.3f same direction as user (sign=%d)",
                            c.ticker, c.strategy, c.delta, user_sign,
                        )
                else:
                    valid.append(c)

            # ── Gap 3 & 9: enrich each candidate with liquidity + earnings data ─
            enriched: List[InstrumentCandidate] = []
            for c in valid:
                try:
                    enriched.append(self._enrich_candidate(c))
                except Exception:
                    enriched.append(c)   # never drop a candidate due to enrichment failure

            enriched.sort(key=lambda c: c.score, reverse=True)
            return enriched[:MAX_CANDIDATES]

        except Exception as e:
            logger.error("OptionsSelector failed for %s: %s", profile.ticker, e)
            return []

    # ── Expiry window helpers ─────────────────────────────────────────────────

    def _get_max_expiry(self, profile: RiskProfile, portfolio: PortfolioInput) -> date:
        """
        Determine the upper bound for hedge expiry dates.
          Stock position  → today + hedge_horizon_days
          Option position → the option's own expiry date
        """
        pos = self._find_original_position(portfolio, profile.ticker)
        if isinstance(pos, OptionsPosition):
            return pos.expiry
        return date.today() + timedelta(days=portfolio.hedge_horizon_days)

    @staticmethod
    def _find_original_position(portfolio: PortfolioInput, profile_ticker: str):
        """
        Find the original holding in portfolio that matches this risk profile ticker.
        profile_ticker is either "AAPL" (stock) or "SPY_C410" (option).
        """
        if "_" not in profile_ticker:
            # Stock
            for h in portfolio.holdings:
                if isinstance(h, StockPosition) and h.ticker == profile_ticker:
                    return h
        else:
            # Option: parse "SPY_C410" → underlying=SPY, type=call, strike=410
            parts      = profile_ticker.split("_")
            underlying = parts[0]
            opt_char   = parts[1][0]           # "C" or "P"
            strike     = float(parts[1][1:])
            opt_type   = "call" if opt_char == "C" else "put"
            for h in portfolio.holdings:
                if (isinstance(h, OptionsPosition)
                        and h.ticker    == underlying
                        and h.option_type == opt_type
                        and abs(h.strike  - strike) < 0.01):
                    return h
        return None

    @staticmethod
    def _sample_expiries(expiries: List[str], n: int) -> List[str]:
        """
        Pick up to n representative expiry dates evenly distributed across
        the available list: shortest, ~middle, longest within window.
        """
        if not expiries:
            return []
        if len(expiries) <= n:
            return expiries
        step = (len(expiries) - 1) / (n - 1)
        indices = {round(i * step) for i in range(n)}
        return [expiries[i] for i in sorted(indices)]

    # ── Direct hedge builder ──────────────────────────────────────────────────

    def _build_direct_hedge_candidates(
        self,
        profile: RiskProfile,
        portfolio: PortfolioInput,
        regime: RegimeState,
        underlying_ticker: str,
        r: float,
        T: float,
        expiry: date,
    ) -> List[InstrumentCandidate]:
        """
        Options on the underlying itself.

        Direction is always opposite to user's effective_delta_sign:
          Long position  (sign=+1) → puts  (negative delta)
          Short position (sign=-1) → calls (positive delta)

        basis_risk_r2 = 1.0 (same asset, zero basis risk).
        """
        try:
            from backend.data.market_data import get_current_price, get_dividend_yield

            spot  = get_current_price(underlying_ticker)
            q     = get_dividend_yield(underlying_ticker)
            sigma = self._get_vol(underlying_ticker, profile, regime)

            if spot <= 0:
                return []

            user_sign   = profile.effective_delta_sign
            option_type = "put" if user_sign > 0 else "call"
            base_label  = "Protective Put" if option_type == "put" else "Protective Call"

            candidates: List[InstrumentCandidate] = []

            expiry_str = expiry.isoformat()

            # For puts  OTM = strike < spot (moneyness < 1.0)
            # For calls OTM = strike > spot (moneyness > 1.0)
            moneyness_grid = [1.00, 0.97] if option_type == "put" else [1.00, 1.03]

            for moneyness in moneyness_grid:
                K = round(spot * moneyness / 5) * 5

                greeks = self._price_option(option_type, spot, K, r, q, sigma, T, underlying_ticker, expiry_str=expiry_str)
                if not greeks:
                    continue

                prc   = greeks["price"]
                delta = greeks["delta"]

                sizing = compute_position(
                    portfolio_notional=profile.notional_value,
                    asset_beta=profile.beta_vs_spy,
                    option_delta=delta,
                    option_price=prc,
                    spot_price=spot,
                    max_cost_pct=portfolio.max_hedge_cost_pct * (portfolio.total_notional / profile.notional_value),
                    hedge_target_pct=portfolio.protection_level,
                )
                if not sizing["viable"]:
                    continue

                n        = sizing["gated"]["n_contracts_gated"]
                cost     = sizing["gated"]["total_cost"]
                cost_pct = cost / portfolio.total_notional if portfolio.total_notional > 0 else 0.0

                score = self._score(
                    delta_offset=abs(delta) * n * 100,
                    target_delta=profile.notional_value * profile.beta_vs_spy / spot,
                    cost_pct=cost_pct,
                    max_cost_pct=portfolio.max_hedge_cost_pct,
                    basis_r2=1.0,
                    lambda_=greeks["lambda_"],
                )

                otm_pct = round((1 - moneyness) * 100) if option_type == "put" else round((moneyness - 1) * 100)
                strategy = base_label if otm_pct == 0 else f"Direct {option_type.capitalize()} ({otm_pct}% OTM)"

                candidates.append(InstrumentCandidate(
                    instrument_type="option",
                    asset_class=profile.asset_class,
                    ticker=underlying_ticker,
                    strategy=strategy,
                    expiry_date=expiry.isoformat(),
                    strike=K,
                    option_type=option_type,
                    n_contracts=n,
                    total_cost=cost,
                    delta=round(delta, 4),
                    gamma=round(greeks["gamma"], 6),
                    theta=round(greeks["theta"], 4),
                    vega=round(greeks["vega"],   4),
                    lambda_leverage=round(greeks["lambda_"], 4),
                    basis_risk_r2=1.0,
                    efficiency_ratio=round(abs(greeks["lambda_"]), 4),
                    score=score,
                    hedge_category="direct_hedge",
                    rationale=(
                        f"Direct hedge on {underlying_ticker} itself. "
                        f"R²=1.0 — zero basis risk. "
                        f"{'Puts protect against downside.' if option_type == 'put' else 'Calls offset short delta exposure.'} "
                        f"Strike {K} (exp {expiry.isoformat()})."
                    ),
                    pros=[
                        "Zero basis risk — tracks your position exactly",
                        "No cross-asset correlation assumptions needed",
                        f"Expiry {expiry.isoformat()} aligned to your position window",
                    ],
                    cons=[
                        "Higher premium than ETF proxy (no diversification discount)",
                        "Theta decay erodes value daily",
                        "May have wider bid/ask spreads than index options",
                    ],
                    partial_hedge_options=sizing["partial_series"],
                ))

            return candidates

        except Exception as e:
            logger.warning("Direct-hedge candidates failed for %s: %s", underlying_ticker, e)
            return []

    # ── Single-leg cross-hedge builder ────────────────────────────────────────

    def _build_single_leg_candidates(
        self,
        profile: RiskProfile,
        portfolio: PortfolioInput,
        regime: RegimeState,
        hedge_ticker: str,
        option_type: str,           # "put" or "call"
        spot: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        expiry: date,
        basis_r2: float = 0.5,
        hedge_category: str = "cross_hedge",
        correlation: float = 0.0,
        h_star: Optional[float] = None,
        expiry_str: Optional[str] = None,
    ) -> List[InstrumentCandidate]:
        """
        Single-leg put or call on a cross-hedge ticker.

        option_type = "put"  → positive-corr asset, opposite-direction puts hedge the user's long
        option_type = "call" → negative-corr asset, calls profit as user's asset falls

        h_star: minimum-variance hedge ratio ρ×(σ_asset/σ_hedge).
                Used instead of beta_vs_spy for cross-hedge sizing accuracy.
        expiry_str: ISO string passed to _price_option for per-strike IV lookup.
        """
        candidates = []
        sizing_ratio = h_star if h_star is not None else profile.beta_vs_spy
        _expiry_str  = expiry_str or expiry.isoformat()

        for moneyness in self._strike_moneyness_grid(portfolio.upside_preservation, option_type):
            K = round(spot * moneyness / 5) * 5

            greeks = self._price_option(option_type, spot, K, r, q, sigma, T, hedge_ticker, expiry_str=_expiry_str)
            if not greeks:
                continue

            prc   = greeks["price"]
            delta = greeks["delta"]

            sizing = compute_position(
                portfolio_notional=profile.notional_value,
                asset_beta=sizing_ratio,   # h* not SPY-beta
                option_delta=delta,
                option_price=prc,
                spot_price=spot,
                max_cost_pct=portfolio.max_hedge_cost_pct * (portfolio.total_notional / profile.notional_value),
                hedge_target_pct=portfolio.protection_level,
            )
            if not sizing["viable"]:
                continue

            n        = sizing["gated"]["n_contracts_gated"]
            cost     = sizing["gated"]["total_cost"]
            cost_pct = cost / portfolio.total_notional if portfolio.total_notional > 0 else 0.0

            score = self._score(
                delta_offset=abs(delta) * n * 100,
                target_delta=profile.notional_value * profile.beta_vs_spy / spot,
                cost_pct=cost_pct,
                max_cost_pct=portfolio.max_hedge_cost_pct,
                basis_r2=basis_r2,
                lambda_=greeks["lambda_"],
            )

            otm_pct  = round((1 - moneyness) * 100) if option_type == "put" else round((moneyness - 1) * 100)
            if option_type == "put":
                base_name = "Proxy Put" if hedge_category == "cross_hedge" else "Macro Put"
            else:
                base_name = "Inverse Hedge" if hedge_category == "cross_hedge" else "Macro Call"

            strategy = base_name if otm_pct == 0 else f"{base_name} ({otm_pct}% OTM)"

            corr_sign = "+" if correlation >= 0 else ""
            if option_type == "put":
                rationale = (
                    f"Cross-hedge: {hedge_ticker} {option_type} profits when {hedge_ticker} falls "
                    f"(ρ={corr_sign}{correlation:.2f} with {profile.ticker.split('_')[0]}). "
                    f"R²={basis_r2:.2f}. Exp {expiry.isoformat()}."
                )
                pros = [
                    f"Correlated to {profile.ticker.split('_')[0]} (ρ={corr_sign}{correlation:.2f})",
                    f"Basis R²={basis_r2:.2f} — reasonable hedge tracking",
                    "More liquid than direct single-stock options",
                ]
            else:
                rationale = (
                    f"Inverse hedge: {hedge_ticker} rises when {profile.ticker.split('_')[0]} falls "
                    f"(ρ={corr_sign}{correlation:.2f}). Long call profits on that inverse move. "
                    f"R²={basis_r2:.2f}. Exp {expiry.isoformat()}."
                )
                pros = [
                    f"Negative correlation (ρ={corr_sign}{correlation:.2f}) — natural diversifier",
                    "Profits when risk assets fall and safe-havens rally",
                    "Lower cost than direct puts in some regimes",
                ]

            candidates.append(InstrumentCandidate(
                instrument_type="option",
                asset_class=profile.asset_class,
                ticker=hedge_ticker,
                strategy=strategy,
                expiry_date=expiry.isoformat(),
                strike=K,
                option_type=option_type,
                n_contracts=n,
                total_cost=cost,
                delta=round(delta, 4),
                gamma=round(greeks["gamma"], 6),
                theta=round(greeks["theta"], 4),
                vega=round(greeks["vega"],   4),
                lambda_leverage=round(greeks["lambda_"], 4),
                basis_risk_r2=basis_r2,
                efficiency_ratio=round(abs(greeks["lambda_"]), 4),
                score=score,
                hedge_category=hedge_category,
                rationale=rationale,
                pros=pros,
                cons=self._put_cons(moneyness, cost_pct, portfolio.max_hedge_cost_pct),
                partial_hedge_options=sizing["partial_series"],
            ))

        return candidates

    # ── Spread and collar builders (unchanged interface, new basis_r2 param) ──

    def _build_spread_candidates(
        self,
        profile: RiskProfile,
        portfolio: PortfolioInput,
        regime: RegimeState,
        hedge_etf: str,
        spot: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        expiry: date,
        basis_r2: Optional[float] = None,
        hedge_category: str = "cross_hedge",
        h_star: Optional[float] = None,
        expiry_str: Optional[str] = None,
    ) -> List[InstrumentCandidate]:
        """Bear put spread: long ATM put + short 10% OTM put."""
        if basis_r2 is None:
            basis_r2 = profile.tail_correlation

        sizing_ratio = h_star if h_star is not None else profile.beta_vs_spy
        _expiry_str  = expiry_str or expiry.isoformat()

        K_long  = round(spot / 5) * 5
        K_short = round(spot * 0.90 / 5) * 5

        g_long  = self._price_option("put", spot, K_long,  r, q, sigma, T, hedge_etf, expiry_str=_expiry_str)
        g_short = self._price_option("put", spot, K_short, r, q, sigma, T, hedge_etf, expiry_str=_expiry_str)
        if not g_long or not g_short:
            return []

        net      = net_greeks([(g_long, +1), (g_short, -1)])
        net_prc  = net["price"]
        net_delta = net["delta"]
        if net_prc <= 0:
            return []

        sizing = compute_position(
            portfolio_notional=profile.notional_value,
            asset_beta=sizing_ratio,
            option_delta=net_delta,
            option_price=net_prc,
            spot_price=spot,
            max_cost_pct=portfolio.max_hedge_cost_pct * (portfolio.total_notional / profile.notional_value),
            hedge_target_pct=portfolio.protection_level,
        )
        if not sizing["viable"]:
            return []

        n        = sizing["gated"]["n_contracts_gated"]
        cost     = sizing["gated"]["total_cost"]
        cost_pct = cost / portfolio.total_notional if portfolio.total_notional > 0 else 0.0

        score = self._score(
            delta_offset=abs(net_delta) * n * 100,
            target_delta=profile.notional_value * profile.beta_vs_spy / spot,
            cost_pct=cost_pct,
            max_cost_pct=portfolio.max_hedge_cost_pct,
            basis_r2=basis_r2,
            lambda_=net["lambda_"],
        )

        return [InstrumentCandidate(
            instrument_type="option",
            asset_class=profile.asset_class,
            ticker=hedge_etf,
            strategy="Bear Put Spread",
            expiry_date=expiry.isoformat(),
            strike=K_long,
            option_type="put",
            short_strike=K_short,
            short_option_type="put",
            n_contracts=n,
            total_cost=cost,
            delta=round(net_delta,     4),
            gamma=round(net["gamma"],  6),
            theta=round(net["theta"],  4),
            vega=round(net["vega"],    4),
            lambda_leverage=round(net["lambda_"], 4),
            basis_risk_r2=basis_r2,
            efficiency_ratio=round(abs(net["lambda_"]), 4),
            score=score,
            hedge_category=hedge_category,
            rationale=(
                f"Bear put spread on {hedge_etf}: long {K_long} put / short {K_short} put. "
                f"Cheaper than naked put; protects ~{round((1 - 0.90)*100)}% downside from ATM. "
                f"Exp {expiry.isoformat()}."
            ),
            pros=["Lower net premium than naked put", "Defined max profit and loss",
                  "Effective in high-IV environments"],
            cons=["Protection capped at short strike", "Needs larger move to profit",
                  "Gamma risk near expiry"],
            partial_hedge_options=sizing["partial_series"],
        )]

    def _build_collar_candidates(
        self,
        profile: RiskProfile,
        portfolio: PortfolioInput,
        regime: RegimeState,
        hedge_etf: str,
        spot: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        expiry: date,
    ) -> List[InstrumentCandidate]:
        """Collar: long ATM put + short 5–10% OTM call. Near zero net cost."""
        K_put  = round(spot / 5) * 5
        K_call = round(spot * 1.07 / 5) * 5

        g_put  = self._price_option("put",  spot, K_put,  r, q, sigma, T, hedge_etf)
        g_call = self._price_option("call", spot, K_call, r, q, sigma, T, hedge_etf)
        if not g_put or not g_call:
            return []

        net     = collar_greeks(g_put, g_call)
        net_prc = net["price"]
        if net_prc < 0:
            net_prc = 0.01

        sizing = compute_position(
            portfolio_notional=profile.notional_value,
            asset_beta=profile.beta_vs_spy,
            option_delta=net["delta"],
            option_price=net_prc,
            spot_price=spot,
            max_cost_pct=portfolio.max_hedge_cost_pct * (portfolio.total_notional / profile.notional_value),
            hedge_target_pct=portfolio.protection_level,
        )
        if not sizing["viable"]:
            return []

        n        = sizing["gated"]["n_contracts_gated"]
        cost     = sizing["gated"]["total_cost"]
        cost_pct = cost / portfolio.total_notional if portfolio.total_notional > 0 else 0.0

        score = self._score(
            delta_offset=abs(net["delta"]) * n * 100,
            target_delta=profile.notional_value * profile.beta_vs_spy / spot,
            cost_pct=cost_pct,
            max_cost_pct=portfolio.max_hedge_cost_pct,
            basis_r2=profile.tail_correlation,
            lambda_=net["lambda_"],
        )

        return [InstrumentCandidate(
            instrument_type="option",
            asset_class=profile.asset_class,
            ticker=hedge_etf,
            strategy="Collar",
            expiry_date=expiry.isoformat(),
            strike=K_put,
            option_type="put",
            short_strike=K_call,
            short_option_type="call",
            n_contracts=n,
            total_cost=max(cost, 0.0),
            delta=round(net["delta"],   4),
            gamma=round(net["gamma"],   6),
            theta=round(net["theta"],   4),
            vega=round(net["vega"],     4),
            lambda_leverage=round(net["lambda_"], 4),
            basis_risk_r2=profile.tail_correlation,
            efficiency_ratio=round(abs(net["lambda_"]), 4),
            score=score,
            hedge_category="direct_hedge",
            rationale=(
                f"Zero-cost collar on {hedge_etf}: long {K_put} put (protection) + "
                f"short {K_call} call (finances the put). "
                f"Near-zero net premium; upside capped at {round((K_call/spot - 1)*100)}% above current. "
                f"Exp {expiry.isoformat()}."
            ),
            pros=["Near-zero net cost", "Strong downside protection",
                  "Works well in high-IV, budget-constrained environments"],
            cons=[f"Upside capped at {K_call}", "Requires call position management",
                  "Not suitable if strong upside conviction"],
            partial_hedge_options=sizing["partial_series"],
        )]

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _get_vol(
        self,
        hedge_etf: str,
        profile: RiskProfile,
        regime: RegimeState,
    ) -> float:
        """Best available vol: ATM IV → GARCH forecast → historical vol."""
        try:
            from backend.data.options_chain import get_atm_iv
            iv = get_atm_iv(hedge_etf, "put")
            if iv and 0.05 < iv < 2.0:
                return iv
        except Exception:
            pass

        if regime.vol_forecast_garch and 0.05 < regime.vol_forecast_garch < 2.0:
            return regime.vol_forecast_garch

        try:
            from backend.data.market_data import get_historical_volatility
            return get_historical_volatility(hedge_etf)
        except Exception:
            return 0.20

    def _price_option(
        self,
        option_type: str,
        S: float,
        K: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        ticker: str,
        expiry_str: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        BSM by default; Binomial for American puts with dividends.

        If expiry_str is provided, looks up the actual implied volatility for
        strike K from the real option chain (volatility skew).  Falls back to
        the supplied sigma (ATM IV) if the chain lookup fails.
        """
        # ── Per-strike IV from real chain (fixes flat-vol underpricing of OTM) ─
        if expiry_str:
            try:
                from backend.data.options_chain import get_strike_iv
                strike_iv = get_strike_iv(ticker, expiry_str, K, option_type)
                if strike_iv:
                    sigma = strike_iv
            except Exception:
                pass   # keep ATM sigma as fallback

        try:
            if q > 0.01 and option_type == "put":
                eep = early_exercise_premium(S, K, r, q, sigma, T, option_type)
                if eep > 0.05:
                    return crr_greeks(S, K, r, q, sigma, T, option_type, american=True)
            return bsm_greeks(S, K, r, q, sigma, T, option_type)
        except Exception as e:
            logger.warning("Pricing failed for %s %s K=%s: %s", ticker, option_type, K, e)
            return None

    @staticmethod
    def _strike_moneyness_grid(upside_preservation, option_type: str = "put") -> List[float]:
        """
        1–3 strike moneyness values based on upside preservation preference.
        For puts:  OTM = below spot → moneyness < 1.0 (inverted)
        For calls: OTM = above spot → moneyness > 1.0
        """
        pct = float(upside_preservation)
        for threshold, moneyness in reversed(UPSIDE_STRIKE_MAP):
            if pct >= threshold:
                if option_type == "put":
                    otm = 2.0 - moneyness          # e.g. 1.10 → 0.90 (10% OTM put)
                    grid = sorted({1.00, otm, max(otm - 0.05, 0.80)})
                else:
                    grid = sorted({1.00, moneyness, min(moneyness + 0.05, 1.15)})
                return grid
        return [1.00]

    @staticmethod
    def _score(
        delta_offset: float,
        target_delta: float,
        cost_pct: float,
        max_cost_pct: float,
        basis_r2: float,
        lambda_: float,
    ) -> float:
        """0–100 composite score."""
        eff_score    = min(delta_offset / target_delta, 1.0) * 40 if target_delta > 0 else 0.0
        cost_score   = max(0.0, 1 - cost_pct / max_cost_pct) * 30 if max_cost_pct > 0 else 0.0
        basis_score  = min(max(basis_r2, 0.0), 1.0) * 20
        lambda_score = min(abs(lambda_) / 5.0, 1.0) * 10
        return round(eff_score + cost_score + basis_score + lambda_score, 2)

    # ── Liquidity + earnings enrichment ──────────────────────────────────────

    @staticmethod
    def _enrich_candidate(candidate: "InstrumentCandidate") -> "InstrumentCandidate":
        """
        Attach liquidity metrics, market premium, and earnings-warning flag
        to a candidate in-place.

        Gap 3: Liquidity — adds open_interest, daily_volume, bid_ask_spread_pct,
               liquidity_ok.  Low-liquidity candidates are kept but flagged in cons.

        Gap 8: Market premium — sets market_premium = (bid+ask)/2 and
               market_total_cost = market_premium × 100 × n_contracts.
               For multi-leg (spread/collar), both legs are fetched and the
               net mid is used.  Falls back to 0.0 if chain is unavailable.

        Gap 9: Earnings — if the candidate's expiry window contains a known
               earnings date, sets earnings_warning=True and appends an IV-crush
               warning to cons.

        Returns the mutated candidate (same object).
        """
        from backend.data.options_chain import check_option_liquidity
        from backend.data.earnings_calendar import crosses_earnings

        # ── Liquidity + market premium (long leg) ────────────────────────────
        if candidate.expiry_date and candidate.strike and candidate.option_type:
            liq = check_option_liquidity(
                ticker=candidate.ticker,
                expiry_str=candidate.expiry_date,
                strike=candidate.strike,
                option_type=candidate.option_type,
            )
            candidate.liquidity_ok       = liq["passes"]
            candidate.open_interest      = liq["open_interest"]
            candidate.daily_volume       = liq["volume"]
            candidate.bid_ask_spread_pct = liq["spread_pct"]

            if not liq["passes"] and liq["reason"]:
                candidate.cons = list(candidate.cons) + [
                    f"Low liquidity: {liq['reason']} — verify before trading"
                ]
                candidate.score = round(max(candidate.score * 0.80, 0.0), 2)

            # Gap 8: market mid for long leg
            long_bid, long_ask = liq["bid"], liq["ask"]
            long_mid = round((long_bid + long_ask) / 2, 4) if (long_bid + long_ask) > 0 else 0.0

            # For multi-leg (spread / collar): fetch short leg and net the mids
            net_mid = long_mid
            if candidate.short_strike and candidate.short_option_type:
                try:
                    short_liq = check_option_liquidity(
                        ticker=candidate.ticker,
                        expiry_str=candidate.expiry_date,
                        strike=candidate.short_strike,
                        option_type=candidate.short_option_type,
                    )
                    short_bid, short_ask = short_liq["bid"], short_liq["ask"]
                    short_mid = round((short_bid + short_ask) / 2, 4) if (short_bid + short_ask) > 0 else 0.0
                    # Long leg cost minus short leg credit
                    net_mid = max(round(long_mid - short_mid, 4), 0.0)
                except Exception:
                    pass

            if net_mid > 0:
                candidate.market_premium    = net_mid
                candidate.market_total_cost = round(net_mid * 100 * max(candidate.n_contracts, 1), 2)

        # ── Earnings ─────────────────────────────────────────────────────────
        if candidate.expiry_date:
            try:
                ed = crosses_earnings(candidate.ticker, candidate.expiry_date)
                if ed:
                    candidate.earnings_warning = True
                    candidate.earnings_date    = ed
                    candidate.cons = list(candidate.cons) + [
                        f"Expiry crosses earnings ({ed}) — IV crush risk: hedge may lose "
                        f"30–50% of value the day after the announcement"
                    ]
                    candidate.score = round(max(candidate.score * 0.85, 0.0), 2)
            except Exception:
                pass

        return candidate

    @staticmethod
    def _put_pros(moneyness: float, regime: RegimeState) -> List[str]:
        pros = ["Simple single-leg hedge", "Unlimited downside protection"]
        if moneyness <= 1.01:
            pros.append("ATM delta — highest sensitivity to decline")
        else:
            pros.append(f"OTM: lower premium cost vs ATM")
        if regime.regime_label in ("low_vol", "mid_vol"):
            pros.append("Low-vol regime: cheaper premium than high-stress periods")
        return pros

    @staticmethod
    def _put_cons(moneyness: float, cost_pct: float, max_cost_pct: float) -> List[str]:
        cons = []
        if cost_pct > max_cost_pct * 0.8:
            cons.append("Near budget ceiling — consider spread or collar instead")
        if moneyness > 1.05:
            cons.append("OTM: no protection until underlying falls below strike")
        cons.append("Theta decay erodes value daily — monitor position")
        return cons
