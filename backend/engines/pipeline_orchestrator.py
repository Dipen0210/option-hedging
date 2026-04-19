"""
Pipeline Orchestrator — wires all 12 layers end-to-end.

Single entry point:
    result = HedgeOrchestrator().run(portfolio_input, market_data)

Layer execution order:
  L1  InputParser          — validate & normalise PortfolioInput
  L2  MarketContextEngine  — regime detection (HDBSCAN + GARCH)
  L3  RiskAnalysisEngine   — per-holding beta, VaR, hedge ratios
  L4  InstrumentSelection  — rule-based candidate generation
  L5  LLMExplainerEngine   — preliminary explanations (pre-pricing)
  L6  PricingEngine        — asset-class pricing (BSM/B76/GK/IR/Credit)
  L7  PositionSizerEngine  — delta-adj / DV01 / CS01 / notional sizing
  L8  GreeksEngine         — portfolio-level Greeks scaling
  L9  SimulationEngine     — Monte Carlo + stress scenarios
  L10 EvaluatorEngine      — composite 0–100 scoring
  L11 FinalLLMExplainer    — final explanations with full quant context
  L12 OutputFormatter      — sort, rank, trim, stamp run_time

Each layer receives what it needs and returns HedgeOutput (or enriches in-place).
Errors in optional layers (L5, L9, L11) are caught — pipeline continues.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence

from backend.models.portfolio_models import PortfolioInput, StockPosition
from backend.models.hedge_models import HedgeOutput
from backend.models.risk_models import RegimeState, PortfolioRiskSummary

# L1: Input validation (pass-through — PortfolioInput is already a Pydantic model)
# L2
from backend.engines.layer_02_market_context import MarketContextEngine
# L3
from backend.engines.layer_03_risk_analysis import RiskAnalysisEngine
# L4
from backend.engines.layer_04_instrument_selection import InstrumentSelectionEngine
# L5
from backend.engines.layer_05_llm_explainer import LLMExplainerEngine
# L6
from backend.engines.layer_06_pricing_engine import PricingEngine
from backend.core.pricing.base_pricer import PricingContext
# L7
from backend.engines.layer_07_position_sizer import PositionSizerEngine
from backend.core.sizing.base_sizer import SizingContext
# L8
from backend.engines.layer_08_greeks_engine import GreeksEngine
# L9
from backend.engines.layer_09_simulation_engine import SimulationEngine
# L10
from backend.engines.layer_10_evaluator import EvaluatorEngine
# L11
from backend.engines.layer_11_llm_explainer import FinalLLMExplainerEngine
# L12
from backend.engines.layer_12_output_formatter import OutputFormatterEngine

logger = logging.getLogger(__name__)


class HedgeOrchestrator:
    """
    Full 12-layer pipeline.

    Usage:
        orchestrator = HedgeOrchestrator()
        result: HedgeOutput = orchestrator.run(portfolio, market_data)

    market_data format:
        {
          "SPY":  {"spot_price": 500.0, "sigma": 0.18, "notional": 50000},
          "AAPL": {"spot_price": 210.0, "sigma": 0.28, "notional": 25000},
          ...
          "_global": {
              "risk_free_rate": 0.053,
              "vix_level": 18.5,
              "yield_curve": {"2.0": 0.049, "10.0": 0.051},
              "fx_rates":    {"EURUSD": 1.085},
              "futures_prices": {"CL": 75.5, "GC": 2350.0},
              "credit_spreads": {"HYG": 360.0},
          }
        }
    """

    # Maps API-facing names → selector class names
    _SELECTOR_CLASS_NAMES = {
        "options":      "OptionsSelector",
        "futures":      "FuturesSelector",
        "forwards":     "ForwardsSelector",
        "swaps":        "SwapsSelector",
        "inverse_etfs": "InverseETFSelector",
    }

    def __init__(
        self,
        run_l5_prelim_llm: bool = False,   # L5 prelim LLM (off by default — L11 is richer)
        run_l9_simulation: bool = True,
        run_l11_llm: bool = True,
        llm_provider: Optional[str] = None,
        instrument_types: Optional[Sequence[str]] = None,  # e.g. ["options"] to restrict L4
    ):
        self.run_l5_prelim_llm = run_l5_prelim_llm
        self.run_l9_simulation = run_l9_simulation
        self.run_l11_llm = run_l11_llm
        self.llm_provider = llm_provider

        # Filter selectors if caller restricted instrument types
        if instrument_types is not None:
            from backend.instruments import ALL_SELECTORS
            allowed = {self._SELECTOR_CLASS_NAMES[t] for t in instrument_types if t in self._SELECTOR_CLASS_NAMES}
            selectors = [s for s in ALL_SELECTORS if s.__class__.__name__ in allowed]
        else:
            selectors = None  # InstrumentSelectionEngine defaults to ALL_SELECTORS

        # Instantiate all engines once (stateless — safe to reuse)
        self._l2  = MarketContextEngine()
        self._l3  = RiskAnalysisEngine()
        self._l4  = InstrumentSelectionEngine(selectors=selectors)
        self._l5  = LLMExplainerEngine()
        self._l6  = PricingEngine()
        self._l7  = PositionSizerEngine()
        self._l8  = GreeksEngine()
        self._l9  = SimulationEngine()
        self._l10 = EvaluatorEngine()
        self._l11 = FinalLLMExplainerEngine()
        self._l12 = OutputFormatterEngine()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_pricing_ctx(
        self, portfolio: PortfolioInput, market_data: Dict[str, Any], regime: RegimeState
    ) -> PricingContext:
        spot_prices = {
            ticker: md["spot_price"]
            for ticker, md in market_data.items()
            if ticker != "_global" and "spot_price" in md
        }
        g = market_data.get("_global", {})
        vol_overrides = {
            ticker: md["sigma"]
            for ticker, md in market_data.items()
            if ticker != "_global" and "sigma" in md
        }
        return PricingEngine.build_context(
            spot_prices=spot_prices,
            risk_free_rate=g.get("risk_free_rate", 0.053),
            regime_vol=regime.vol_forecast_garch,
            vix_level=regime.vix_level,
            vol_overrides=vol_overrides,
            dividend_yields=g.get("dividend_yields", {}),
            yield_curve=g.get("yield_curve", {}),
            fx_rates=g.get("fx_rates", {}),
            futures_prices=g.get("futures_prices", {}),
            credit_spreads=g.get("credit_spreads", {}),
        )

    def _build_sizing_contexts(
        self,
        portfolio: PortfolioInput,
        market_data: Dict[str, Any],
        risk_summary: PortfolioRiskSummary,
    ) -> Dict[str, SizingContext]:
        g = market_data.get("_global", {})
        contexts = {}
        risk_by_ticker = {p.ticker: p for p in risk_summary.risk_profiles}

        for holding in portfolio.stock_positions:
            ticker = holding.ticker
            md = market_data.get(ticker, {})
            rp = risk_by_ticker.get(ticker)

            spot = md.get("spot_price", holding.purchase_price)
            sigma = md.get("sigma", 0.20)
            notional = holding.shares * spot

            ctx = PositionSizerEngine.build_sizing_context(
                portfolio_notional=portfolio.total_notional,
                holding_notional=notional,
                beta=rp.beta_vs_spy if rp else 1.0,
                hedge_ratio=1.0,
                max_premium_pct=portfolio.max_hedge_cost_pct,
                correlation=rp.optimal_hedge_ratio if rp else 0.85,
                spot_vol=sigma,
                futures_vol=md.get("futures_sigma", sigma),
                portfolio_dv01=g.get("portfolio_dv01", 0.0),
                portfolio_cs01=g.get("portfolio_cs01", 0.0),
                fx_exposure=md.get("fx_exposure", 0.0),
                extended={"spot_price": spot, "holding_days": portfolio.hedge_horizon_days},
            )
            contexts[ticker] = ctx

        # Options positions — sizing is based on delta-adjusted exposure from L3
        for opt in portfolio.options_positions:
            ticker_key = f"{opt.ticker}_{opt.option_type[0].upper()}{int(opt.strike)}"
            rp = risk_by_ticker.get(ticker_key)
            if rp is None:
                continue

            md    = market_data.get(opt.ticker, {})
            spot  = md.get("spot_price", opt.strike)
            sigma = md.get("sigma", 0.20)

            ctx = PositionSizerEngine.build_sizing_context(
                portfolio_notional=portfolio.total_notional,
                holding_notional=rp.notional_value,   # effective delta $ exposure
                beta=rp.beta_vs_spy if rp else 1.0,
                hedge_ratio=1.0,
                max_premium_pct=portfolio.max_hedge_cost_pct,
                correlation=rp.optimal_hedge_ratio if rp else 1.0,
                spot_vol=sigma,
                futures_vol=md.get("futures_sigma", sigma),
                portfolio_dv01=g.get("portfolio_dv01", 0.0),
                portfolio_cs01=g.get("portfolio_cs01", 0.0),
                fx_exposure=md.get("fx_exposure", 0.0),
                extended={"spot_price": spot, "holding_days": portfolio.hedge_horizon_days},
            )
            contexts[ticker_key] = ctx

        return contexts

    def _enrich_pricing_ctx_with_candidates(
        self, pricing_ctx: PricingContext, hedge_output
    ) -> None:
        """
        After L4, fetch live spot/vol for any hedge tickers (e.g. QQQ, sector ETFs)
        that are referenced in candidates but missing from pricing_ctx.spot_prices.
        Mutates pricing_ctx in-place.
        """
        try:
            from backend.data.market_data import get_current_price, get_historical_volatility, get_dividend_yield
        except ImportError:
            return

        missing = set()
        for rec in hedge_output.recommendations:
            for cand in rec.candidates:
                t = cand.ticker.upper()
                if t not in pricing_ctx.spot_prices:
                    missing.add(t)

        for t in missing:
            try:
                spot = get_current_price(t)
                pricing_ctx.spot_prices[t] = spot
                if t not in pricing_ctx.vol_overrides:
                    pricing_ctx.vol_overrides[t] = get_historical_volatility(t)
                if t not in pricing_ctx.dividend_yields:
                    q = get_dividend_yield(t)
                    pricing_ctx.dividend_yields[t] = q
                logger.info("Enriched pricing_ctx with %s spot=%.2f", t, spot)
            except Exception as exc:
                logger.warning("Could not enrich %s: %s", t, exc)

    def _holding_notionals(
        self,
        portfolio: PortfolioInput,
        market_data: Dict[str, Any],
        risk_summary: Optional[PortfolioRiskSummary] = None,
    ) -> Dict[str, float]:
        result = {}
        for h in portfolio.stock_positions:
            spot = market_data.get(h.ticker, {}).get("spot_price", h.purchase_price)
            result[h.ticker] = h.shares * spot

        # Options positions: use the delta-adjusted notional from the risk profile
        if risk_summary:
            risk_by_ticker = {p.ticker: p for p in risk_summary.risk_profiles}
            for opt in portfolio.options_positions:
                ticker_key = f"{opt.ticker}_{opt.option_type[0].upper()}{int(opt.strike)}"
                rp = risk_by_ticker.get(ticker_key)
                if rp:
                    result[ticker_key] = rp.notional_value
        return result

    # ── main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        portfolio: PortfolioInput,
        market_data: Dict[str, Any],
    ) -> HedgeOutput:
        """
        Execute the full 12-layer pipeline.

        Returns:
            HedgeOutput — fully priced, sized, Greeks-computed,
                          simulated, scored, explained, and formatted.
        """
        pipeline_start = time.perf_counter()
        logger.info(
            "Pipeline start: %d holdings, notional=$%.0f",
            len(portfolio.holdings), portfolio.total_notional,
        )

        # ── Route by position type ────────────────────────────────────────────
        # Phase 1: stock + options positions → full 12-layer pipeline.
        # Options positions are analysed via their delta-adjusted equity exposure
        # (effective_delta × 100 × spot × contracts).  The same hedge candidate
        # generation and scoring logic applies; the UI labels them by their
        # position key (e.g. "SPY_C410") so they are easy to identify.
        if portfolio.options_positions:
            logger.info(
                "Pipeline includes %d options position(s) — "
                "computing delta-adjusted exposure for each.",
                len(portfolio.options_positions),
            )

        # ── L2: Market regime ─────────────────────────────────────────────────
        logger.info("L2: Detecting market regime...")
        regime: RegimeState = self._l2.get_regime(portfolio)
        logger.info("L2: regime=%s vix=%.1f anomaly=%s",
                    regime.regime_label, regime.vix_level, regime.is_anomaly)

        # ── L3: Per-holding risk analysis ─────────────────────────────────────
        logger.info("L3: Running risk analysis...")
        risk_summary: PortfolioRiskSummary = self._l3.analyze(portfolio, regime)
        logger.info("L3: portfolio beta=%.2f VaR=$%.0f",
                    risk_summary.portfolio_beta, risk_summary.portfolio_var_5pct)

        # ── L4: Instrument selection ──────────────────────────────────────────
        logger.info("L4: Selecting hedge candidates...")
        hedge_output: HedgeOutput = self._l4.select(portfolio, regime, risk_summary)
        total_candidates = sum(len(r.candidates) for r in hedge_output.recommendations)
        logger.info("L4: %d candidates across %d holdings",
                    total_candidates, len(hedge_output.recommendations))

        # ── L5: Preliminary LLM explanation (optional) ────────────────────────
        if self.run_l5_prelim_llm:
            logger.info("L5: Running preliminary LLM explanations...")
            try:
                hedge_output = self._l5.explain(hedge_output, portfolio, risk_summary, regime)
            except Exception as exc:
                logger.warning("L5 failed (non-fatal): %s", exc)

        # ── Build shared context objects ──────────────────────────────────────
        pricing_ctx = self._build_pricing_ctx(portfolio, market_data, regime)
        # Enrich pricing_ctx with live spot prices for any hedge ETF tickers
        # selected by L4 that weren't in market_data (e.g., QQQ, sector ETFs)
        self._enrich_pricing_ctx_with_candidates(pricing_ctx, hedge_output)
        sizing_ctxs = self._build_sizing_contexts(portfolio, market_data, risk_summary)
        holding_notionals = self._holding_notionals(portfolio, market_data, risk_summary)

        # ── L6: Pricing ───────────────────────────────────────────────────────
        logger.info("L6: Pricing %d candidates...", total_candidates)
        hedge_output = self._l6.run(hedge_output, pricing_ctx)

        # ── L7: Position sizing ───────────────────────────────────────────────
        logger.info("L7: Sizing positions...")
        hedge_output = self._l7.run(hedge_output, sizing_ctxs)

        # ── L8: Greeks ────────────────────────────────────────────────────────
        logger.info("L8: Computing portfolio Greeks...")
        hedge_output = self._l8.run(hedge_output, pricing_ctx)

        # ── L9: Monte Carlo simulation (optional) ─────────────────────────────
        if self.run_l9_simulation:
            logger.info("L9: Running simulations...")
            try:
                hedge_output = self._l9.run(
                    hedge_output,
                    market_data=market_data,
                    risk_free_rate=market_data.get("_global", {}).get("risk_free_rate", 0.053),
                    hedge_horizon_days=portfolio.hedge_horizon_days,
                )
            except Exception as exc:
                logger.warning("L9 failed (non-fatal): %s", exc)

        # ── L10: Scoring ──────────────────────────────────────────────────────
        logger.info("L10: Scoring candidates...")
        hedge_output = self._l10.run(hedge_output, regime, holding_notionals)

        # ── L11: Final LLM explanation (optional) ─────────────────────────────
        if self.run_l11_llm:
            logger.info("L11: Running final LLM explanations...")
            try:
                hedge_output = self._l11.run(
                    hedge_output, portfolio, risk_summary, regime,
                    provider=self.llm_provider,
                )
            except Exception as exc:
                logger.warning("L11 failed (non-fatal): %s", exc)

        # ── L12: Output formatting ────────────────────────────────────────────
        logger.info("L12: Formatting output...")
        hedge_output = self._l12.run(
            hedge_output,
            pipeline_start_time=pipeline_start,
            risk_summary=risk_summary,
        )

        logger.info(
            "Pipeline complete: %d recommendations, run_time=%.3fs",
            len(hedge_output.recommendations),
            hedge_output.run_time_seconds,
        )
        return hedge_output

    # ── Phase 2 stub ──────────────────────────────────────────────────────────

    def _run_options_hedge(
        self,
        portfolio: PortfolioInput,
        pipeline_start: float,
    ) -> HedgeOutput:
        """
        Phase 2: Greek-neutralisation pipeline for existing options positions.

        Will compute net delta/gamma/vega of the user's options book and
        recommend offsetting options to neutralise target Greeks.

        Not yet implemented — returns a clear not-supported response.
        """
        logger.info("Options-position hedging requested — Phase 2 stub.")
        elapsed = round(time.perf_counter() - pipeline_start, 3)
        return HedgeOutput(
            portfolio_notional=portfolio.total_notional,
            regime="unknown",
            is_anomaly=False,
            vix_level=0.0,
            recommendations=[],
            run_time_seconds=elapsed,
            portfolio_summary=(
                "Options position hedging (Greek neutralisation) is coming in Phase 2. "
                "Currently supported: stock position hedging via equity options."
            ),
            key_risks=[],
            regime_commentary="",
            top_recommendation="",
        )
