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
    score = 35 × (delta_offset / target_delta)     ← hedge effectiveness
          + 25 × (1 - cost_pct / max_cost_pct)     ← cost efficiency vs budget
          + 15 × basis_r2                           ← low basis risk
          + 15 × liquidity                          ← tradeability + transaction cost
          + 10 × abs(lambda_)                       ← capital efficiency (capped)

The liquidity term exists because a theoretically excellent hedge is worthless
if it cannot be executed. Scoring on the model premium alone once ranked a
MSFT put with 20 contracts of open interest and a 34.7% bid-ask spread as the
single best candidate in the book. Candidates are therefore re-scored *after*
the live quote is attached, on the executable cost rather than the theoretical
one, with the quoted spread and depth priced in.
"""
import logging
from datetime import date, timedelta
from typing import Any, List, Dict, Optional, Tuple

from backend.instruments.base import InstrumentSelector
from backend.instruments.options.bsm_pricer import bsm_greeks
from backend.instruments.options.binomial_pricer import crr_greeks, early_exercise_premium
from backend.instruments.options.position_sizer import compute_position
from backend.instruments.options.greeks import net_greeks, collar_greeks
from backend.models.risk_models import RiskProfile, RegimeState
from backend.models.portfolio_models import PortfolioInput, StockPosition, OptionsPosition
from backend.models.hedge_models import InstrumentCandidate
from backend.data.options_chain import OptionsDataUnavailable
from backend.data import data_quality as dq
from backend.core.disclosures import disclose

logger = logging.getLogger(__name__)


def _basis_phrase(measured: bool, basis_r2: float) -> str:
    """
    How to state a basis R² in prose, including when there isn't one.

    An unmeasured basis used to print as "R²=0.50" — the builder's old default
    — which reads as a measurement of tracking that nobody performed.
    """
    return f"R²={basis_r2:.2f}." if measured else "Basis R² not measured."


def _basis_provenance(
    hedge_category: str,
    basis_stats: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    The sample behind a candidate's `basis_risk_r2`, as constructor kwargs.

    `get_ranked_cross_hedges` has always returned `n_obs`, `window_days` and
    `corr_stderr` alongside each correlation — the comment in covariance.py
    says they exist "so the caller can disclose what the estimate rests on" —
    but no caller read them, so an R2 fitted on the 30-observation minimum and
    one fitted on a full year reached the user as the same number.

    A direct hedge gets `identity`: the underlying is the same asset, so R2=1
    is definitional and there is no sample to report. Returning the key with
    an empty source rather than omitting it keeps the absence explicit.
    """
    if hedge_category == "direct_hedge" and not basis_stats:
        return {"basis_risk_source": "identity"}
    if not basis_stats:
        return {}
    return {
        "basis_risk_source":      "correlation",
        "basis_risk_n_obs":       basis_stats.get("n_obs"),
        "basis_risk_window_days": basis_stats.get("window_days"),
        "basis_risk_stderr":      basis_stats.get("corr_stderr"),
    }

# Hedge ratio for an option written on the holding itself. One share of the
# underlying offsets one share of the position, so the ratio is 1.0 — beta only
# enters when a *different* instrument stands in for the exposure.
DIRECT_HEDGE_RATIO = 1.0

MAX_CANDIDATES = 5

# How far a model-derived target strike may be snapped onto a listed strike
# before the candidate is abandoned. Beyond this the contract no longer
# represents the protection level the user asked for.
STRIKE_SNAP_TOLERANCE = 0.06

# Max expiries sampled per ticker (keeps candidate count manageable)
MAX_EXPIRIES_DIRECT = 3
MAX_EXPIRIES_CROSS  = 2

# Score component weights (sum to 100).
W_EFFECTIVENESS = 35.0
W_COST          = 25.0
W_BASIS         = 15.0
W_LIQUIDITY     = 15.0
W_EFFICIENCY    = 10.0

# Liquidity reference points. Open interest at or above GOOD_OI_CONTRACTS earns
# full depth marks; a bid-ask spread at or above MAX_GOOD_SPREAD_PCT of mid
# earns none, because crossing it costs more than the edge most hedges provide.
GOOD_OI_CONTRACTS   = 500
MAX_GOOD_SPREAD_PCT = 0.10

# Above this bid-ask spread the mid is not a price anyone can transact near,
# so the candidate is not executable at the figure we would be showing. A live
# LQD 2027-02-19 99 put quoted 0.05 / 3.00 has a spread of 198% of its mid:
# "cost $773.68" is meaningless when the real fill is somewhere in a 60x range.
# These were still scoring ~50 and ranking mid-book, because a wide spread only
# zeroed the 15-point liquidity term instead of disqualifying the quote.
MAX_EXECUTABLE_SPREAD_PCT = 0.50


def is_executably_quoted(candidate) -> bool:
    """
    Is this candidate's premium a price the user could actually transact near?

    Two conditions, and both are needed: the premium came from a real
    two-sided quote rather than the model, and that quote is tight enough
    that the mid is an attainable fill. A contract quoted 0.05 / 3.00 is
    "quoted" in the sense that `premium_source == "market_mid"`, but showing
    its mid as the cost of the hedge is not honest, so it is not executable
    here either.

    Defined once, at module level, because two different layers now have to
    agree on it: `_filter_executable` uses it to rank within a holding, and
    L4 uses it to decide whether a holding can be hedged at all. Two copies
    of this predicate drifting apart would mean L4 refusing hedges the
    selector considered fine, or the reverse.
    """
    if candidate.premium_source != "market_mid":
        return False
    spread = candidate.bid_ask_spread_pct
    return spread is None or spread <= MAX_EXECUTABLE_SPREAD_PCT

# A hedge must actually span the horizon the user asked to be protected over.
# An expiry covering less than this fraction of the horizon leaves the position
# naked for the remainder, so it is not offered as a horizon hedge.
MIN_HORIZON_COVERAGE = 0.5

# Strike moneyness by upside_preservation (1.0 = ATM, >1.0 = OTM)
UPSIDE_STRIKE_MAP = [
    (0.00, 1.00),
    (0.30, 1.03),
    (0.50, 1.05),
    (0.70, 1.08),
    (1.00, 1.10),
]


class OptionsSelector(InstrumentSelector):

    label = "options"

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
                # No fabricated expiry here. `today + hedge_horizon_days` is
                # almost never a real expiration date, so every candidate built
                # on it priced a contract that does not exist. With no listed
                # expiry in the window there is no direct hedge to offer.
                logger.info(
                    "No listed %s expiries on or before %s — no direct hedge available",
                    base_ticker, max_expiry.isoformat(),
                )
                dq.record(f"expiries:{base_ticker}", dq.SOURCE_UNAVAILABLE,
                          detail=f"no listed expiry on or before {max_expiry.isoformat()}")

            for expiry_str in self._sample_expiries(
                direct_expiries, MAX_EXPIRIES_DIRECT,
                horizon_days=portfolio.hedge_horizon_days, ticker=base_ticker,
            ):
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
                    sigma_hedge, sigma_hedge_source = self._get_vol(cross_ticker, profile, regime)

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

                    for expiry_str in self._sample_expiries(
                        cross_expiries, MAX_EXPIRIES_CROSS,
                        horizon_days=portfolio.hedge_horizon_days,
                        ticker=cross_ticker,
                    ):
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
                            sigma_source=sigma_hedge_source,
                            basis_stats=ct,
                        )
                        candidates += cands

                        # Bear put spread on the proxy. Previously gated on a
                        # high-vol regime or VaR > 20%, which meant the strategy
                        # the product advertises was unreachable in normal
                        # markets. Cost-reduction structures are worth comparing
                        # in any regime, so the scorer decides instead.
                        if option_type == "put":
                            spread_cands = self._build_spread_candidates(
                                profile, portfolio, regime,
                                cross_ticker, spot, r, q, sigma_hedge, T, exp_date,
                                basis_r2=basis_r2, hedge_category=hedge_cat,
                                h_star=h_star, expiry_str=expiry_str,
                                sigma_source=sigma_hedge_source,
                                basis_stats=ct,
                            )
                            candidates += spread_cands

                except Exception as e:
                    logger.error("Cross-hedge failed for %s: %s", cross_ticker, e)
                    continue

            # ── 3. Collars ────────────────────────────────────────────────────
            # Built inside the direct-hedge builder, per sampled expiry, gated
            # only on the user's upside_preservation preference. The former gate
            # here also required max_hedge_cost_pct < 1.5% AND VIX > 25
            # simultaneously, so collars were effectively never offered despite
            # being an advertised strategy.

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

            enriched = self._filter_executable(enriched, base_ticker)
            enriched.sort(key=lambda c: c.score, reverse=True)
            return self._diversify(enriched, MAX_CANDIDATES)

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
    def _sample_expiries(
        expiries: List[str],
        n: int,
        horizon_days: Optional[int] = None,
        ticker: str = "",
    ) -> List[str]:
        """
        Pick up to n representative expiry dates from the window.

        Sampling evenly across the whole listed range put the nearest weekly in
        the book: for a 180-day horizon MSFT was offered a put expiring in 7
        days, which leaves the position unhedged for the other 173 and prices
        off a near-dated, earnings-inflated IV. An expiry must cover at least
        MIN_HORIZON_COVERAGE of the requested horizon to be a horizon hedge.

        When nothing listed reaches that threshold the longest available
        expiries are still returned — a short hedge beats none — but the
        shortfall is disclosed rather than presented as full-horizon cover.
        """
        if not expiries:
            return []

        usable = expiries
        if horizon_days and horizon_days > 0:
            min_days = MIN_HORIZON_COVERAGE * horizon_days
            today    = date.today()
            covering = [
                e for e in expiries
                if (date.fromisoformat(e) - today).days >= min_days
            ]
            if covering:
                usable = covering
            else:
                longest      = max(expiries, key=lambda e: date.fromisoformat(e))
                covered_days = (date.fromisoformat(longest) - today).days
                logger.info(
                    "No %s expiry covers %.0f of %d horizon days; longest is %s "
                    "(%d days)", ticker or "listed", min_days, horizon_days,
                    longest, covered_days,
                )
                dq.record(
                    f"expiry_coverage:{ticker}" if ticker else "expiry_coverage",
                    dq.SOURCE_FALLBACK,
                    detail=(
                        f"longest listed expiry {longest} covers {covered_days} "
                        f"of {horizon_days} requested horizon days"
                    ),
                )

        if len(usable) <= n:
            return usable
        step = (len(usable) - 1) / (n - 1)
        indices = {round(i * step) for i in range(n)}
        return [usable[i] for i in sorted(indices)]

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
            sigma, sigma_source = self._get_vol(underlying_ticker, profile, regime)

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
                # Snap the target onto a strike the exchange actually lists;
                # skip the candidate rather than quote an unlisted contract.
                K = self._resolve_strike(
                    underlying_ticker, expiry_str, spot * moneyness, option_type
                )
                if K is None:
                    continue

                greeks = self._price_option(option_type, spot, K, r, q, sigma, T, underlying_ticker,
                                            expiry_str=expiry_str, sigma_source=sigma_source)
                if not greeks:
                    continue

                prc   = greeks["price"]
                delta = greeks["delta"]

                sizing = compute_position(
                    portfolio_notional=profile.notional_value,
                    # A direct hedge is an option on the holding itself, so the
                    # exposure needs no translation: one unit of the underlying
                    # is one unit of the thing being protected. Beta belongs in
                    # a cross hedge, where it converts the holding's exposure
                    # into the proxy's units. Passing beta_vs_spy here inflated
                    # every high-beta name's contract count by its own beta.
                    asset_beta=DIRECT_HEDGE_RATIO,
                    option_delta=delta,
                    option_price=prc,
                    spot_price=spot,
                    max_cost_pct=self._holding_budget_pct(profile, portfolio),
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

                cand = InstrumentCandidate(
                    instrument_type="option",
                    asset_class=profile.asset_class,
                    ticker=underlying_ticker,
                    strategy=strategy,
                    expiry_date=expiry.isoformat(),
                    strike=K,
                    option_type=option_type,
                    n_contracts=n,
                    extended_metrics={"sizing_ratio": DIRECT_HEDGE_RATIO, "hedge_spot": spot},
                    total_cost=cost,
                    delta=round(delta, 4),
                    gamma=round(greeks["gamma"], 6),
                    theta=round(greeks["theta"], 4),
                    vega=round(greeks["vega"],   4),
                    lambda_leverage=round(greeks["lambda_"], 4),
                    basis_risk_r2=1.0,
                    # Same underlying: R2=1 is definitional. Naming the source
                    # keeps a blank stderr here from reading as a failed
                    # measurement.
                    basis_risk_source="identity",
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
                    ] + self._option_pros(moneyness, option_type, regime)[1:],
                    cons=self._option_cons(
                        moneyness, option_type, spot, cost_pct,
                        portfolio.max_hedge_cost_pct,
                        extra=[
                            "Higher premium than an ETF proxy (no diversification "
                            "discount)",
                            "May have wider bid/ask spreads than index options",
                        ],
                    ),
                    partial_hedge_options=sizing["partial_series"],
                    iv_used=round(greeks["sigma_used"], 4),
                    iv_source=greeks["iv_source"],
                    model_premium=round(prc, 4),
                )
                # Reconcile the theoretical premium against the live quote and
                # record the contract symbol, so the user is quoted a real,
                # placeable contract at a price they could actually pay.
                self._finalize_pricing(
                    cand,
                    self._contract_quote(underlying_ticker, expiry_str, K, option_type),
                    model_premium=prc,
                    delta_offset=abs(delta) * n * 100,
                    target_delta=profile.notional_value * profile.beta_vs_spy / spot,
                    portfolio=portfolio,
                )
                candidates.append(cand)

            # ── Multi-leg structures on the same underlying ───────────────────
            # These are advertised strategies, so they are constructed and left
            # to the scorer to rank against the naked put rather than hidden
            # behind a regime gate. Each returns [] when the listed chain cannot
            # support it, so nothing is fabricated to fill the slot.
            if option_type == "put":
                # Bear put spread: cheaper than the naked put, protection capped
                # at the short strike. A genuine cost/coverage trade-off the user
                # should be able to see and compare.
                candidates += self._build_spread_candidates(
                    profile, portfolio, regime, underlying_ticker,
                    spot, r, q, sigma, T, expiry,
                    basis_r2=1.0, hedge_category="direct_hedge",
                    h_star=profile.beta_vs_spy, expiry_str=expiry_str,
                    sigma_source=sigma_source,
                )

                # Collar: only when the user has said they do not need to keep
                # upside, because it caps their gains. That is a stated
                # preference, not a market condition.
                if not portfolio.upside_preservation:
                    candidates += self._build_collar_candidates(
                        profile, portfolio, regime, underlying_ticker,
                        spot, r, q, sigma, T, expiry,
                        sigma_source=sigma_source,
                    )

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
        # No default: 0.5 here was a basis R² nobody measured, and it was
        # formatted straight into the user-facing rationale as though it had
        # been. None means unmeasured and is scored as such.
        basis_r2: Optional[float] = None,
        hedge_category: str = "cross_hedge",
        correlation: float = 0.0,
        h_star: Optional[float] = None,
        expiry_str: Optional[str] = None,
        sigma_source: str = "atm_iv",
        # The covariance row `basis_r2` came from. It already carries the
        # sample the correlation was measured over; nothing read it, so the
        # R2 reached the user as a bare number.
        basis_stats: Optional[Dict[str, Any]] = None,
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
        # An unmeasured basis is 0.0 and carries no source, so the scorer
        # treats it as unmeasured rather than as a middling measurement.
        basis_measured = basis_r2 is not None
        if not basis_measured:
            basis_r2 = 0.0
        sizing_ratio = h_star if h_star is not None else profile.beta_vs_spy
        _expiry_str  = expiry_str or expiry.isoformat()

        for moneyness in self._strike_moneyness_grid(portfolio.upside_preservation, option_type):
            K = self._resolve_strike(hedge_ticker, _expiry_str, spot * moneyness, option_type)
            if K is None:
                continue

            greeks = self._price_option(option_type, spot, K, r, q, sigma, T, hedge_ticker,
                                        expiry_str=_expiry_str, sigma_source=sigma_source)
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
                max_cost_pct=self._holding_budget_pct(profile, portfolio),
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
                    f"{_basis_phrase(basis_measured, basis_r2)} "
                    f"Exp {expiry.isoformat()}."
                )
                # "reasonable hedge tracking" used to be asserted at any R²,
                # so a measured 0.08 was presented to the user as a point in
                # the hedge's favour. The cons list states the tracking
                # honestly either way, so a weak or unmeasured basis earns no
                # pro here rather than a contradicted one.
                pros = [
                    f"Correlated to {profile.ticker.split('_')[0]} (ρ={corr_sign}{correlation:.2f})",
                    "More liquid than direct single-stock options",
                ] + self._option_pros(moneyness, option_type, regime)[1:]
                if basis_measured and basis_r2 >= 0.5:
                    pros.insert(1, f"Basis R²={basis_r2:.2f} — hedge tracks the asset closely")
                elif basis_measured and basis_r2 >= 0.25:
                    pros.insert(1, f"Basis R²={basis_r2:.2f} — partial tracking")
            else:
                rationale = (
                    f"Inverse hedge: {hedge_ticker} rises when {profile.ticker.split('_')[0]} falls "
                    f"(ρ={corr_sign}{correlation:.2f}). Long call profits on that inverse move. "
                    f"{_basis_phrase(basis_measured, basis_r2)} "
                    f"Exp {expiry.isoformat()}."
                )
                pros = [
                    f"Negative correlation (ρ={corr_sign}{correlation:.2f}) — natural diversifier",
                    "Profits when risk assets fall and safe-havens rally",
                    "Lower cost than direct puts in some regimes",
                ] + self._option_pros(moneyness, option_type, regime)[1:]

            cand = InstrumentCandidate(
                instrument_type="option",
                asset_class=profile.asset_class,
                ticker=hedge_ticker,
                strategy=strategy,
                expiry_date=expiry.isoformat(),
                strike=K,
                option_type=option_type,
                n_contracts=n,
                extended_metrics={"sizing_ratio": sizing_ratio, "hedge_spot": spot},
                total_cost=cost,
                delta=round(delta, 4),
                gamma=round(greeks["gamma"], 6),
                theta=round(greeks["theta"], 4),
                vega=round(greeks["vega"],   4),
                lambda_leverage=round(greeks["lambda_"], 4),
                basis_risk_r2=basis_r2,
                **_basis_provenance(hedge_category, basis_stats),
                efficiency_ratio=round(abs(greeks["lambda_"]), 4),
                score=score,
                hedge_category=hedge_category,
                rationale=rationale,
                pros=pros,
                cons=self._option_cons(
                    moneyness, option_type, spot, cost_pct,
                    portfolio.max_hedge_cost_pct,
                    extra=[
                        (
                            f"Basis risk: {hedge_ticker} is a proxy, not "
                            f"{profile.ticker.split('_')[0]} — R²={basis_r2:.2f} of "
                            f"the move tracks, the remainder does not"
                        ) if basis_measured else (
                            f"Basis risk: {hedge_ticker} is a proxy, not "
                            f"{profile.ticker.split('_')[0]}, and how closely it "
                            f"tracks was not measured"
                        ),
                    ],
                ),
                partial_hedge_options=sizing["partial_series"],
                iv_used=round(greeks["sigma_used"], 4),
                iv_source=greeks["iv_source"],
                model_premium=round(prc, 4),
            )
            self._finalize_pricing(
                cand,
                self._contract_quote(hedge_ticker, _expiry_str, K, option_type),
                model_premium=prc,
                delta_offset=abs(delta) * n * 100,
                target_delta=profile.notional_value * profile.beta_vs_spy / spot,
                portfolio=portfolio,
            )
            candidates.append(cand)

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
        sigma_source: str = "atm_iv",
        basis_stats: Optional[Dict[str, Any]] = None,
    ) -> List[InstrumentCandidate]:
        """Bear put spread: long ATM put + short 10% OTM put."""
        # Strike resolution needs the expiry as the chain keys it.
        expiry_str = expiry_str or expiry.isoformat()
        if basis_r2 is None:
            # The profile's tail correlation is a real measurement, but a
            # different quantity from a full-sample R², so it is labelled as
            # such. Leaving the source empty would now read as "never
            # measured" and score zero in L10.
            basis_r2 = profile.tail_correlation
            basis_provenance = {"basis_risk_source": "tail_correlation"}
        else:
            basis_provenance = _basis_provenance(hedge_category, basis_stats)

        sizing_ratio = h_star if h_star is not None else profile.beta_vs_spy
        _expiry_str  = expiry_str or expiry.isoformat()

        K_long  = self._resolve_strike(hedge_etf, _expiry_str, spot,        "put")
        K_short = self._resolve_strike(hedge_etf, _expiry_str, spot * 0.90, "put")
        # A spread needs both legs listed, and they must be distinct strikes or
        # the "spread" collapses to zero width and zero protection.
        if K_long is None or K_short is None or K_short >= K_long:
            logger.info(
                "Bear put spread on %s %s not constructible from listed strikes "
                "(long=%s short=%s)", hedge_etf, _expiry_str, K_long, K_short,
            )
            return []

        g_long  = self._price_option("put", spot, K_long,  r, q, sigma, T, hedge_etf,
                                    expiry_str=_expiry_str, sigma_source=sigma_source)
        g_short = self._price_option("put", spot, K_short, r, q, sigma, T, hedge_etf,
                                    expiry_str=_expiry_str, sigma_source=sigma_source)
        if not g_long or not g_short:
            return []

        net      = net_greeks([(g_long, +1), (g_short, -1)])
        net_prc  = net["price"]
        net_delta = net["delta"]
        if net_prc <= 0:
            # A bear put spread (long higher strike, short lower) is a debit
            # structure by construction. A non-positive net price means the two
            # legs were mispriced relative to each other, so the candidate is
            # dropped rather than quoted as free protection.
            logger.info(
                "Bear put spread on %s %s priced at net %.4f — rejected as non-debit",
                hedge_etf, _expiry_str, net_prc,
            )
            return []

        sizing = compute_position(
            portfolio_notional=profile.notional_value,
            asset_beta=sizing_ratio,
            option_delta=net_delta,
            option_price=net_prc,
            spot_price=spot,
            max_cost_pct=self._holding_budget_pct(profile, portfolio),
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

        cand = InstrumentCandidate(
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
            extended_metrics={"sizing_ratio": sizing_ratio, "hedge_spot": spot},
            total_cost=cost,
            delta=round(net_delta,     4),
            gamma=round(net["gamma"],  6),
            theta=round(net["theta"],  4),
            vega=round(net["vega"],    4),
            lambda_leverage=round(net["lambda_"], 4),
            basis_risk_r2=basis_r2,
            **basis_provenance,
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
            iv_used=round(g_long["sigma_used"], 4),
            iv_source=g_long["iv_source"],
        )
        # Both legs must be real, quotable contracts for the spread to be
        # executable; the net quote also carries the two-leg transaction cost
        # into the ranking.
        self._apply_net_market_quote(
            cand,
            self._contract_quote(hedge_etf, _expiry_str, K_long,  "put"),
            self._contract_quote(hedge_etf, _expiry_str, K_short, "put"),
            model_premium=net_prc,
        )
        self._rescore(
            cand,
            delta_offset=abs(net_delta) * n * 100,
            target_delta=profile.notional_value * profile.beta_vs_spy / spot,
            portfolio=portfolio,
        )
        return [cand]

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
        sigma_source: str = "atm_iv",
    ) -> List[InstrumentCandidate]:
        """Collar: long ATM put + short 5–10% OTM call. Near zero net cost."""
        expiry_str = expiry.isoformat()
        K_put  = self._resolve_strike(hedge_etf, expiry_str, spot,        "put")
        K_call = self._resolve_strike(hedge_etf, expiry_str, spot * 1.07, "call")
        # Both legs must exist, and the call must sit above the put or the
        # collar caps upside below the protection floor.
        if K_put is None or K_call is None or K_call <= K_put:
            logger.info(
                "Collar on %s %s not constructible from listed strikes "
                "(put=%s call=%s)", hedge_etf, expiry_str, K_put, K_call,
            )
            return []

        # Pass the expiry so each leg is priced on its own strike's implied vol
        # (skew), not one flat ATM number for both legs.
        g_put  = self._price_option("put",  spot, K_put,  r, q, sigma, T, hedge_etf,
                                    expiry_str=expiry_str, sigma_source=sigma_source)
        g_call = self._price_option("call", spot, K_call, r, q, sigma, T, hedge_etf,
                                    expiry_str=expiry_str, sigma_source=sigma_source)
        if not g_put or not g_call:
            return []

        net     = collar_greeks(g_put, g_call)
        net_prc = net["price"]          # > 0 = net debit, < 0 = net credit

        if net_prc < 0:
            # Net credit: the short call finances more than the long put, so the
            # user is PAID to establish the collar. The cost budget is not a
            # binding constraint, so size on the delta target alone and report
            # the credit as a negative cost.
            #
            # Previously this was floored to net_prc = 0.01, which turned a
            # credit into a fake one-cent debit: it hid a real cash inflow from
            # the user and fed a wrong price into the budget gate and the cost
            # score.
            from backend.instruments.options.position_sizer import size_by_delta
            n = size_by_delta(
                profile.notional_value, DIRECT_HEDGE_RATIO, net["delta"],
                spot, 100, portfolio.protection_level,
            )
            if n <= 0:
                return []
            cost           = round(net_prc * 100 * n, 2)   # negative = credit
            partial_series = []
        else:
            sizing = compute_position(
                portfolio_notional=profile.notional_value,
                asset_beta=DIRECT_HEDGE_RATIO,   # collar is on the holding itself
                option_delta=net["delta"],
                option_price=net_prc,
                spot_price=spot,
                max_cost_pct=self._holding_budget_pct(profile, portfolio),
                hedge_target_pct=portfolio.protection_level,
            )
            if not sizing["viable"]:
                return []
            n              = sizing["gated"]["n_contracts_gated"]
            cost           = sizing["gated"]["total_cost"]
            partial_series = sizing["partial_series"]

        cost_pct = cost / portfolio.total_notional if portfolio.total_notional > 0 else 0.0

        score = self._score(
            delta_offset=abs(net["delta"]) * n * 100,
            target_delta=profile.notional_value * profile.beta_vs_spy / spot,
            cost_pct=cost_pct,
            max_cost_pct=portfolio.max_hedge_cost_pct,
            basis_r2=profile.tail_correlation,
            lambda_=net["lambda_"],
        )

        cand = InstrumentCandidate(
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
            extended_metrics={"sizing_ratio": DIRECT_HEDGE_RATIO, "hedge_spot": spot},
            # Negative for a credit collar — the user receives net premium.
            # Clamping this to 0 hid the inflow from the budget gate and the UI.
            total_cost=cost,
            delta=round(net["delta"],   4),
            gamma=round(net["gamma"],   6),
            theta=round(net["theta"],   4),
            vega=round(net["vega"],     4),
            lambda_leverage=round(net["lambda_"], 4),
            basis_risk_r2=profile.tail_correlation,
            basis_risk_source="tail_correlation",
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
            partial_hedge_options=partial_series,
            model_premium=round(net_prc, 4),
            iv_used=round(g_put["sigma_used"], 4),
            iv_source=g_put["iv_source"],
        )
        # Collar economics hinge on the two quotes: the call's bid is what
        # actually finances the put. Model prices alone cannot establish whether
        # the structure is a credit or a debit in the live market.
        self._apply_net_market_quote(
            cand,
            self._contract_quote(hedge_etf, expiry_str, K_put,  "put"),
            self._contract_quote(hedge_etf, expiry_str, K_call, "call"),
            model_premium=net_prc,
        )
        self._rescore(
            cand,
            delta_offset=abs(net["delta"]) * n * 100,
            target_delta=profile.notional_value * profile.beta_vs_spy / spot,
            portfolio=portfolio,
        )
        return [cand]

    # ── Budget allocation ─────────────────────────────────────────────────────

    @staticmethod
    def _holding_budget_pct(profile: RiskProfile, portfolio: PortfolioInput) -> float:
        """
        This holding's hedge budget, as a fraction of *its own* notional.

        `compute_position` is called with portfolio_notional=holding notional,
        so max_cost_pct must be expressed against the holding, not the
        portfolio.

        The previous expression was
            max_hedge_cost_pct * (portfolio_notional / holding_notional)
        which granted **every** holding the entire portfolio budget. A
        three-holding portfolio with a 5% cap could therefore be sized to spend
        15% — the user's cost constraint was not enforced at portfolio level at
        all. Budget is instead allocated pro-rata by notional:

            holding_budget_$   = max_hedge_cost_pct x portfolio_mv x weight
            weight             = holding_mv / portfolio_mv
            holding_budget_pct = holding_budget_$ / holding_mv
                               = max_hedge_cost_pct

        so a pro-rata allocation is simply the portfolio-level percentage,
        unscaled. Summed over holdings the total cannot exceed the cap,
        provided the holding values add up to the portfolio value.
        `EvaluatorEngine.enforce_portfolio_budget` verifies that on the
        selected set and records the portfolio hedge cost, rather than leaving
        the invariant asserted but unchecked.
        """
        return portfolio.max_hedge_cost_pct

    # ── Contract resolution ───────────────────────────────────────────────────

    def _resolve_strike(
        self,
        ticker: str,
        expiry_str: str,
        target_strike: float,
        option_type: str,
    ) -> Optional[float]:
        """
        Snap a model-derived target strike onto a strike the exchange actually
        lists for this expiry.

        Returns None when nothing is listed close enough. Callers must skip the
        candidate in that case: recommending a strike that is not listed gives
        the user an order they cannot place, and its premium, Greeks and risk
        reduction would all describe a contract that does not exist.

        Replaces `round(spot * moneyness / 5) * 5`, which assumed a universal
        $5 strike ladder. Real ladders vary by underlying and price level
        ($1 on SPY near the money, $2.50 and $5 elsewhere), so that rounding
        produced unlisted strikes.
        """
        try:
            from backend.data.options_chain import select_listed_strike
            listed = select_listed_strike(
                ticker, expiry_str, target_strike, option_type,
                max_distance_pct=STRIKE_SNAP_TOLERANCE,
            )
        except Exception as exc:
            logger.warning(
                "Strike resolution unavailable for %s %s %s: %s",
                ticker, expiry_str, option_type, exc,
            )
            dq.record(f"strikes:{ticker}:{expiry_str}", dq.SOURCE_UNAVAILABLE,
                      detail=str(exc)[:80])
            return None

        if listed is None:
            logger.info(
                "No listed %s strike within %.0f%% of %.2f for %s %s — candidate skipped",
                option_type, STRIKE_SNAP_TOLERANCE * 100, target_strike, ticker, expiry_str,
            )
            return None
        return listed

    def _contract_quote(
        self, ticker: str, expiry_str: str, strike: float, option_type: str
    ) -> Optional[Dict]:
        """Live quote for one listed contract, or None if unavailable."""
        try:
            from backend.data.options_chain import get_contract_quote
            return get_contract_quote(ticker, expiry_str, strike, option_type)
        except Exception as exc:
            logger.warning("Quote unavailable for %s %s %s %s: %s",
                           ticker, expiry_str, strike, option_type, exc)
            return None

    @staticmethod
    def _apply_market_quote(
        cand: InstrumentCandidate,
        quote: Optional[Dict],
        model_premium: float,
    ) -> None:
        """
        Attach the real quote to a candidate and reconcile it against the model.

        The premium the user sees is the market mid when a two-sided quote
        exists, because that is what they would actually pay. The theoretical
        price is kept alongside it so a large divergence is visible instead of
        being averaged away.
        """
        cand.model_premium = round(model_premium, 4)

        if not quote:
            cand.premium_source = "model"
            return

        cand.contract_symbol  = quote.get("contract_symbol")
        cand.strike_is_listed = True
        cand.bid = quote.get("bid")
        cand.ask = quote.get("ask")
        # Depth and spread are only measurements when there is a market to
        # measure. Yahoo returns the strike row with bid=ask=OI=0 outside
        # market hours; writing those zeros in would advertise a 0.00%
        # bid-ask spread on a contract that has no bid at all.
        if quote.get("has_two_sided_quote"):
            cand.open_interest = int(quote.get("open_interest") or 0)
            cand.daily_volume  = int(quote.get("volume") or 0)
            spread = quote.get("spread_pct")
            cand.bid_ask_spread_pct = round(spread, 4) if spread is not None else None

        iv = quote.get("implied_volatility")
        if iv:
            cand.iv_used = round(float(iv), 4)
            cand.iv_source = "contract_iv"

        mid = quote.get("mid")
        if quote.get("has_two_sided_quote") and mid and mid > 0:
            cand.market_premium    = round(mid, 4)
            cand.market_total_cost = round(mid * 100 * cand.n_contracts, 2)
            cand.premium_source    = "market_mid"
            if model_premium > 0:
                cand.premium_vs_model_pct = round(mid / model_premium - 1.0, 4)
            # Cost the user is quoted must be the executable one.
            cand.total_cost = cand.market_total_cost
        else:
            # Listed but not quotable right now (closed market, no bids).
            cand.premium_source = "model"

    def _apply_net_market_quote(
        self,
        cand: InstrumentCandidate,
        long_quote: Optional[Dict],
        short_quote: Optional[Dict],
        model_premium: float,
    ) -> None:
        """
        Reconcile a two-leg structure (spread / collar) against live quotes.

        Such a structure is only executable if BOTH legs are quotable, so a
        missing quote on either leg leaves the candidate on its model price
        rather than implying half of it is tradeable.

        Net premium is long_mid − short_mid, which is NEGATIVE for a credit
        structure and must stay negative. The transaction cost is the sum of
        both legs' dollar spreads, because the user crosses both; depth is the
        thinner leg, which is what actually gates the fill.
        """
        cand.model_premium = round(model_premium, 4)

        if not (long_quote and short_quote):
            cand.premium_source = "model"
            return

        cand.contract_symbol       = long_quote.get("contract_symbol")
        cand.short_contract_symbol = short_quote.get("contract_symbol")
        cand.strike_is_listed      = True
        cand.bid = long_quote.get("bid")
        cand.ask = long_quote.get("ask")
        if (long_quote.get("has_two_sided_quote")
                and short_quote.get("has_two_sided_quote")):
            # Depth is the thinner leg — that is what gates the fill.
            cand.open_interest = min(int(long_quote.get("open_interest") or 0),
                                     int(short_quote.get("open_interest") or 0))
            cand.daily_volume  = min(int(long_quote.get("volume") or 0),
                                     int(short_quote.get("volume") or 0))

        iv = long_quote.get("implied_volatility")
        if iv:
            cand.iv_used   = round(float(iv), 4)
            cand.iv_source = "contract_iv"

        l_mid = long_quote.get("mid")
        s_mid = short_quote.get("mid")
        if not (long_quote.get("has_two_sided_quote")
                and short_quote.get("has_two_sided_quote")
                and l_mid and s_mid):
            cand.premium_source = "model"
            return

        net_mid  = l_mid - s_mid
        l_spread = (long_quote.get("ask")  or 0.0) - (long_quote.get("bid")  or 0.0)
        s_spread = (short_quote.get("ask") or 0.0) - (short_quote.get("bid") or 0.0)
        cand.bid_ask_spread_pct = (
            round((l_spread + s_spread) / abs(net_mid), 4) if net_mid else None
        )

        cand.market_premium    = round(net_mid, 4)
        cand.market_total_cost = round(net_mid * 100 * cand.n_contracts, 2)
        cand.premium_source    = "market_mid"
        if model_premium != 0:
            cand.premium_vs_model_pct = round(net_mid / model_premium - 1.0, 4)
        cand.total_cost = cand.market_total_cost

    def _rescore(
        self,
        cand: InstrumentCandidate,
        *,
        delta_offset: float,
        target_delta: float,
        portfolio: PortfolioInput,
    ) -> None:
        """
        Re-score a candidate once its live quote is attached.

        The first score is computed on the model premium, because sizing needs a
        price before the chain is read. That score cannot see what the contract
        actually costs to trade or whether anyone is trading it, so the ranking
        is recomputed here against the executable cost and the quoted market.
        """
        cost_pct = (cand.total_cost / portfolio.total_notional
                    if portfolio.total_notional > 0 else 0.0)
        cand.score = self._score(
            delta_offset=delta_offset,
            target_delta=target_delta,
            cost_pct=cost_pct,
            max_cost_pct=portfolio.max_hedge_cost_pct,
            basis_r2=cand.basis_risk_r2,
            lambda_=cand.lambda_leverage,
            liquidity=self._liquidity_score(
                cand.open_interest,
                cand.daily_volume,
                cand.bid_ask_spread_pct,
                has_quote=(cand.premium_source == "market_mid"),
            ),
        )

    def _finalize_pricing(
        self,
        cand: InstrumentCandidate,
        quote: Optional[Dict],
        *,
        model_premium: float,
        delta_offset: float,
        target_delta: float,
        portfolio: PortfolioInput,
    ) -> None:
        """Attach a single-leg quote, then re-rank on what it would cost to buy."""
        self._apply_market_quote(cand, quote, model_premium)
        self._rescore(cand, delta_offset=delta_offset,
                      target_delta=target_delta, portfolio=portfolio)

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _get_vol(
        self,
        hedge_etf: str,
        profile: RiskProfile,
        regime: RegimeState,
    ) -> Tuple[float, str]:
        """
        Best available vol for the ticker, with the provenance of the number.

        Returns (sigma, source) where source is one of "atm_iv",
        "realized_vol" or "garch_forecast". The caller must carry the source
        onto the candidate: a premium computed from a market-wide GARCH
        forecast cannot be reported as having come from the option market.

        Order matters, and it used to be wrong. GARCH was preferred over
        realized vol, but `regime.vol_forecast_garch` is a single
        *market-level* forecast from Layer 2 — not a figure for this ticker.
        With option quotes unavailable overnight, that one number priced every
        contract in the book: AAPL, SPY and MSFT puts were all marked at
        sigma=0.1212 while their own 20-day realized vols were 0.1362, 0.0999
        and 0.1956. Pricing MSFT at 12% when it is realising 20% understates
        the premium badly, and a ticker-specific measurement beats a
        market-wide forecast for a single-name contract every time. The
        market-wide forecast is now the last resort and says so.

        Per-contract pricing still prefers the strike's own implied vol —
        `_price_option` looks that up so skew is respected, because realized
        vol stood 2.5x below the true IV of an 85%-moneyness SPY put during
        verification. This is the fallback when the chain has no usable IV.
        """
        try:
            from backend.data.options_chain import get_atm_iv
            iv = get_atm_iv(hedge_etf, "put")
            if iv and 0.05 < iv < 2.0:
                dq.record(f"iv:{hedge_etf}", dq.SOURCE_LIVE, detail="ATM IV")
                return iv, "atm_iv"
        except Exception as exc:
            logger.debug("ATM IV unavailable for %s: %s", hedge_etf, exc)

        try:
            from backend.data.market_data import get_historical_volatility
            vol = get_historical_volatility(hedge_etf)
            if vol and 0.01 < vol < 3.0:
                dq.record(
                    f"iv:{hedge_etf}", dq.SOURCE_FALLBACK,
                    detail=(f"no ATM implied vol quoted; priced on {hedge_etf} "
                            f"20-day realized vol {vol:.1%} — a backward-looking "
                            f"measurement, not the option market's view"),
                )
                return vol, "realized_vol"
        except Exception as exc:
            logger.debug("Realized vol unavailable for %s: %s", hedge_etf, exc)

        if regime.vol_forecast_garch and 0.05 < regime.vol_forecast_garch < 2.0:
            dq.record(
                f"iv:{hedge_etf}", dq.SOURCE_FALLBACK,
                detail=(f"no implied or realized vol for {hedge_etf}; priced on "
                        f"the market-wide GARCH forecast "
                        f"{regime.vol_forecast_garch:.1%}, which is not specific "
                        f"to this underlying"),
            )
            return regime.vol_forecast_garch, "garch_forecast"

        # No vol at all: raise rather than pricing off an invented 0.20,
        # which silently mis-states every premium and Greek.
        dq.record(f"iv:{hedge_etf}", dq.SOURCE_UNAVAILABLE,
                  detail="no implied, realized or forecast volatility available")
        raise OptionsDataUnavailable(
            f"No volatility estimate available for {hedge_etf}"
        )

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
        sigma_source: str = "atm_iv",
    ) -> Optional[Dict]:
        """
        BSM by default; Binomial for American puts with dividends.

        If expiry_str is provided, looks up the actual implied volatility for
        strike K from the real option chain (volatility skew). Otherwise the
        supplied sigma is used, and `sigma_source` must describe where the
        caller got it.

        `sigma_source` defaulted to "atm_iv" unconditionally, so a premium
        priced off a market-wide GARCH forecast was still reported to the user
        as having come from the at-the-money option market. The caller knows
        the real provenance; this function does not, so it must be told.
        """
        # ── Per-strike IV from real chain (fixes flat-vol underpricing of OTM) ─
        # Skew is large: during verification SPY 85%-moneyness puts carried
        # IV 0.2473 against 0.0999 realized vol — a 2.5x pricing error if the
        # realized number is used. The vol actually used is reported back so the
        # candidate can disclose it.
        iv_source = sigma_source
        if expiry_str:
            try:
                from backend.data.options_chain import get_strike_iv
                strike_iv = get_strike_iv(ticker, expiry_str, K, option_type)
                if strike_iv:
                    sigma = strike_iv
                    iv_source = "contract_iv"
            except Exception as exc:
                logger.debug("Per-strike IV unavailable for %s %s K=%s: %s",
                             ticker, expiry_str, K, exc)

        if iv_source != "contract_iv":
            dq.record(f"strike_iv:{ticker}", dq.SOURCE_FALLBACK,
                      detail=(f"no per-strike IV for K={K}; priced on "
                              f"{iv_source} {sigma:.1%} with no skew adjustment"))

        try:
            if q > 0.01 and option_type == "put":
                eep = early_exercise_premium(S, K, r, q, sigma, T, option_type)
                if eep > 0.05:
                    out = crr_greeks(S, K, r, q, sigma, T, option_type, american=True)
                else:
                    out = bsm_greeks(S, K, r, q, sigma, T, option_type)
            else:
                out = bsm_greeks(S, K, r, q, sigma, T, option_type)
            out["sigma_used"] = sigma
            out["iv_source"]  = iv_source
            return out
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
    def _liquidity_score(
        open_interest: Optional[int],
        daily_volume: Optional[int],
        spread_pct: Optional[float],
        has_quote: bool,
    ) -> float:
        """
        0–1 tradeability score combining transaction cost and market depth.

        Returns 0.0 when no two-sided quote was obtainable: an unquotable
        contract is not demonstrably tradeable, and scoring it as neutral would
        let it outrank a contract with a real, tight market.

        The spread term dominates (60%) because it is a certain, immediate cost
        paid on entry, whereas thin depth is a risk rather than a charge.
        """
        if not has_quote:
            return 0.0

        if spread_pct is None or spread_pct <= 0:
            # No spread reported alongside a two-sided quote — treat depth alone
            # as the evidence rather than awarding free marks for a tight market.
            spread_component = 0.5
        else:
            spread_component = 1.0 - min(spread_pct / MAX_GOOD_SPREAD_PCT, 1.0)

        depth = max(int(open_interest or 0), int(daily_volume or 0))
        depth_component = min(depth / GOOD_OI_CONTRACTS, 1.0)

        return 0.6 * spread_component + 0.4 * depth_component

    @staticmethod
    def _score(
        delta_offset: float,
        target_delta: float,
        cost_pct: float,
        max_cost_pct: float,
        basis_r2: float,
        lambda_: float,
        liquidity: float = 0.0,
    ) -> float:
        """
        0–100 composite score.

        cost_pct may be negative for credit structures (a collar that collects
        net premium). The cost component is clamped to its weight so a credit
        earns full marks on cost without borrowing points from the
        effectiveness, basis-risk, liquidity and efficiency components.

        liquidity is the 0–1 output of `_liquidity_score`. It defaults to 0 so a
        caller that has not yet obtained a quote cannot accidentally score a
        contract as liquid; `_finalize_pricing` supplies the real value once the
        chain has been read.
        """
        eff_score    = min(delta_offset / target_delta, 1.0) * W_EFFECTIVENESS if target_delta > 0 else 0.0
        cost_ratio   = min(max(1 - cost_pct / max_cost_pct, 0.0), 1.0) if max_cost_pct > 0 else 0.0
        cost_score   = cost_ratio * W_COST
        basis_score  = min(max(basis_r2, 0.0), 1.0) * W_BASIS
        liq_score    = min(max(liquidity, 0.0), 1.0) * W_LIQUIDITY
        lambda_score = min(abs(lambda_) / 5.0, 1.0) * W_EFFICIENCY
        return round(eff_score + cost_score + basis_score + liq_score + lambda_score, 2)

    # ── Liquidity + earnings enrichment ──────────────────────────────────────

    @staticmethod
    def _diversify(
        ranked: List[InstrumentCandidate],
        limit: int,
    ) -> List[InstrumentCandidate]:
        """
        Trim to `limit` while keeping the shortlist comparable across strategies.

        A plain `ranked[:limit]` returned five variations of the same trade. In
        the 2026-10-28 MSFT run the book held all five advertised structures —
        protective put, direct OTM put, proxy put, collar (net debit $737.96)
        and bear put spread — and the top five were three single-leg put
        variants and nothing else. Single-leg puts win on the 35-point
        effectiveness term because their delta offset is uncapped, so they
        crowd out every structure that trades protection for cost, which is
        precisely the trade-off the product exists to show.

        Highest score still comes first and nothing is reordered within a
        family: one slot is reserved for each strategy family in rank order,
        then the remaining slots are filled by score. The user can therefore
        always compare a put against a spread and a collar when all three were
        actually constructed, without any candidate being promoted above a
        better one of its own kind.
        """
        if len(ranked) <= limit:
            return list(ranked)

        def family(c: InstrumentCandidate) -> str:
            # "Direct Put (3% OTM)" and "Direct Put" are the same structure.
            return (c.strategy or "").split(" (")[0].strip()

        best_of_family: List[InstrumentCandidate] = []
        seen: set = set()
        for c in ranked:                      # already sorted by score desc
            fam = family(c)
            if fam not in seen:
                seen.add(fam)
                best_of_family.append(c)

        picked = best_of_family[:limit]
        if len(picked) < limit:
            chosen = {id(c) for c in picked}
            picked += [c for c in ranked if id(c) not in chosen][
                : limit - len(picked)
            ]

        picked.sort(key=lambda c: c.score, reverse=True)
        return picked

    @staticmethod
    def _filter_executable(
        candidates: List[InstrumentCandidate],
        ticker: str = "",
    ) -> List[InstrumentCandidate]:
        """
        Keep only candidates the user could actually trade at the quoted price.

        A contract can be genuinely listed yet have no market. MSFT's
        2027-04-16 515 put had zero open interest and no two-sided quote, so its
        premium was a model number — and it still outranked three SPY puts with
        real 1%-wide markets, because the rest of the score knew nothing about
        tradeability.

        A quote can also be too wide to be a price. A spread above
        MAX_EXECUTABLE_SPREAD_PCT of mid is rejected the same way a missing
        quote is, because the displayed premium would not survive contact with
        the order book.

        When nothing in the window is quotable (an illiquid name, or — far more
        often — a provider that publishes no quotes outside market hours) the
        candidates are returned rather than leaving the user with silence, but
        they are flagged and the theoretical pricing is disclosed instead of
        being presented as a price they could pay.
        """
        if not candidates:
            return []

        quotable: List[InstrumentCandidate] = []
        too_wide: List[InstrumentCandidate] = []
        for c in candidates:
            if c.premium_source != "market_mid":
                continue
            if is_executably_quoted(c):
                quotable.append(c)
            else:
                too_wide.append(c)

        if too_wide:
            logger.info(
                "%s: dropped %d quoted candidate(s) with a spread wider than "
                "%.0f%% of mid: %s",
                ticker or "selector", len(too_wide),
                MAX_EXECUTABLE_SPREAD_PCT * 100,
                ", ".join(
                    f"{c.contract_symbol or c.ticker} {c.bid_ask_spread_pct:.0%}"
                    for c in too_wide
                ),
            )
            dq.record(
                f"spreads:{ticker}" if ticker else "spreads", dq.SOURCE_FALLBACK,
                detail=(
                    f"{len(too_wide)} contract(s) quoted with a bid-ask spread "
                    f"wider than {MAX_EXECUTABLE_SPREAD_PCT:.0%} of mid were "
                    f"excluded — the mid is not an attainable fill price"
                ),
            )

        if quotable:
            dropped = len(candidates) - len(quotable) - len(too_wide)
            if dropped > 0:
                logger.info(
                    "%s: dropped %d candidate(s) with no two-sided quote",
                    ticker or "selector", dropped,
                )
            return quotable

        logger.warning(
            "%s: no candidate has an executable two-sided quote — all premiums "
            "are theoretical", ticker or "selector",
        )
        dq.record(
            f"quotes:{ticker}" if ticker else "quotes", dq.SOURCE_UNAVAILABLE,
            detail=("no executable two-sided option quote available; premiums "
                    "shown are model values, not prices that can be traded"),
        )
        for c in candidates:
            c.liquidity_ok = False
            disclose(c, "No live two-sided quote — premium shown is "
                        "theoretical, not an executable price")
        return candidates

    @staticmethod
    def _enrich_candidate(candidate: "InstrumentCandidate") -> "InstrumentCandidate":
        """
        Attach liquidity metrics, market premium, and earnings-warning flag
        to a candidate in-place.

        Gap 3: Liquidity — sets `liquidity_ok` and flags thin markets in cons.

        Gap 9: Earnings — if the candidate's expiry window contains a known
               earnings date, sets earnings_warning=True and appends an IV-crush
               warning to cons.

        This no longer re-derives prices or liquidity figures. The builders now
        attach the chain quote through `_apply_market_quote` /
        `_apply_net_market_quote`, which also set `total_cost` to the executable
        price and feed `_rescore`. The earlier duplicate implementation here
        overwrote a two-leg structure's combined spread and depth with long-leg
        values, re-clamped a credit collar's net premium to zero via
        `max(long_mid - short_mid, 0.0)`, and left `total_cost` disagreeing with
        `market_total_cost`. Liquidity is also no longer penalised a second time
        with a score multiplier, because `_score` now carries a liquidity term.

        Returns the mutated candidate (same object).
        """
        from backend.data.options_chain import check_option_liquidity
        from backend.data.earnings_calendar import crosses_earnings

        # ── Liquidity flag ───────────────────────────────────────────────────
        if candidate.expiry_date and candidate.strike and candidate.option_type:
            liq = check_option_liquidity(
                ticker=candidate.ticker,
                expiry_str=candidate.expiry_date,
                strike=candidate.strike,
                option_type=candidate.option_type,
            )
            candidate.liquidity_ok = liq["passes"]

            # Only fill depth/spread the quote did not already establish, so a
            # multi-leg candidate keeps its combined two-leg transaction cost.
            # `available` is False for a chain we could not read at all, where
            # the zeros in the result are placeholders rather than readings —
            # those must stay None so they are not displayed as measurements.
            if candidate.premium_source != "market_mid" and liq["available"]:
                # A reported depth of 0 alongside no quote is indistinguishable
                # from depth that was never published: the provider zeroes
                # openInterest and volume outside market hours. Only positive
                # figures are treated as readings. A real post-close snapshot
                # (OI published, bid/ask cleared) still reports its OI here.
                oi  = int(liq["open_interest"] or 0)
                vol = int(liq["volume"] or 0)
                if oi > 0:
                    candidate.open_interest = oi
                if vol > 0:
                    candidate.daily_volume = vol
                spread = liq["spread_pct"]
                candidate.bid_ask_spread_pct = (
                    float(spread) if spread is not None else None
                )

            if not liq["passes"] and liq["reason"]:
                disclose(
                    candidate,
                    f"Low liquidity: {liq['reason']} — verify before trading",
                )

        # ── Earnings ─────────────────────────────────────────────────────────
        if candidate.expiry_date:
            try:
                ed = crosses_earnings(candidate.ticker, candidate.expiry_date)
                if ed:
                    candidate.earnings_warning = True
                    candidate.earnings_date    = ed
                    disclose(
                        candidate,
                        f"Expiry crosses earnings ({ed}) — IV crush risk: hedge "
                        f"may lose 30–50% of value the day after the announcement",
                    )
                    candidate.score = round(max(candidate.score * 0.85, 0.0), 2)
            except Exception:
                pass

        return candidate

    # ── Candidate disclosure ──────────────────────────────────────────────────
    #
    # Moneyness throughout this module is strike / spot (see the grids in
    # `_build_direct_candidates` and `_strike_moneyness_grid`). A protective
    # PUT is therefore out-of-the-money BELOW 1.0, and a hedging CALL above
    # it. The two helpers these replaced both tested moneyness as though a put
    # were OTM above 1.0:
    #
    #   `_put_pros`  called every put with moneyness <= 1.01 "ATM delta —
    #                highest sensitivity to decline". Since the put grid never
    #                exceeds 1.00, that fired on a 10%-OTM put as readily as
    #                on an ATM one and the OTM branch was unreachable. (It was
    #                also never called from anywhere.)
    #   `_put_cons`  guarded its OTM caveat on moneyness > 1.05, which no put
    #                in the grid can satisfy. The result was that the most
    #                material fact about an out-of-the-money protective put —
    #                that the first leg of the decline is not covered — was
    #                disclosed on no recommendation at all, while the pros
    #                list alongside it advertised "unlimited downside
    #                protection". It was also applied unchanged to call
    #                hedges, where "falls below strike" is the wrong
    #                direction.
    #
    # The caveat is now quantified in dollars and percent, because "no
    # protection until the underlying falls below strike" does not tell a user
    # how much of their own position is exposed.

    @staticmethod
    def _moneyness_caveat(
        moneyness: float, option_type: str, spot: float
    ) -> Optional[str]:
        """
        The uncovered move, stated as a number.

        For a put this is the decline the holder absorbs before the strike is
        reached — the deductible on the insurance. For a call hedge it is the
        rally the proxy must deliver before the leg pays anything. Returns
        None within half a percent of the money, where there is nothing to
        disclose.
        """
        gap = (1.0 - moneyness) if option_type == "put" else (moneyness - 1.0)

        if gap > 0.005:
            per_share = abs(spot * gap)
            if option_type == "put":
                return (
                    f"{gap * 100:.0f}% out-of-the-money: the first {gap * 100:.1f}% "
                    f"of a decline (${per_share:,.2f}/share) is not covered — "
                    f"protection starts below the strike, not at today's price"
                )
            return (
                f"{gap * 100:.0f}% out-of-the-money: pays nothing until the "
                f"underlying rallies {gap * 100:.1f}% (${per_share:,.2f}/share) "
                f"above today's price"
            )

        if gap < -0.005:
            depth = abs(gap)
            return (
                f"{depth * 100:.0f}% in-the-money: ${abs(spot * gap):,.2f}/share "
                f"of the premium is intrinsic value, not protection — that part "
                f"is returned only if the move happens"
            )

        return None

    @staticmethod
    def _option_pros(moneyness: float, option_type: str, regime: RegimeState) -> List[str]:
        """Generic pros for a single-leg option hedge."""
        pros = ["Simple single-leg hedge"]
        if option_type == "put":
            pros.append("Downside protection continues all the way to zero")

        gap = (1.0 - moneyness) if option_type == "put" else (moneyness - 1.0)
        if abs(gap) <= 0.005:
            pros.append("At the money — highest delta per contract")
        elif gap > 0.005:
            pros.append("Out of the money — lower premium per contract than ATM")

        if regime.regime_label in ("low_vol", "mid_vol"):
            pros.append("Low-vol regime: cheaper premium than high-stress periods")
        return pros

    @staticmethod
    def _option_cons(
        moneyness: float,
        option_type: str,
        spot: float,
        cost_pct: float,
        max_cost_pct: float,
        extra: Optional[List[str]] = None,
    ) -> List[str]:
        """Generic cons for a single-leg option hedge, strongest caveat first."""
        cons: List[str] = []

        caveat = OptionsSelector._moneyness_caveat(moneyness, option_type, spot)
        if caveat:
            cons.append(caveat)

        if cost_pct > max_cost_pct * 0.8:
            cons.append(
                f"Uses {cost_pct / max_cost_pct:.0%} of the hedge budget — "
                f"a spread or collar would cost less"
            )

        cons.append("Theta decay erodes value daily — monitor position")
        cons.extend(extra or [])
        return cons
