"""
Rolling hedge monitor (Gap 4).

Institutional platforms don't treat hedges as one-shot recommendations.
Once a hedge is placed, they monitor three triggers daily:

  1. Roll trigger   — DTE < 21 — theta decay accelerates; gamma risk spikes;
                      time to roll to next expiry before the hedge becomes worthless.

  2. Delta drift    — Hedge delta has drifted > 20% from the target delta.
                      As the underlying moves, a put's delta changes (gamma).
                      If a put bought at Δ=−0.35 is now Δ=−0.10, it provides
                      only 29% of the original protection.  Re-hedge.

  3. Spot move      — Underlying has moved > 5% since the hedge was initiated.
                      Even if delta hasn't drifted past the threshold yet, a
                      large spot move may have moved the strike deep OTM
                      (for puts) or ITM (for calls), changing the hedge economics.

Each trigger returns an Alert dict:
  {
      "trigger":   str,          # "roll" | "delta_drift" | "spot_move" | "earnings"
      "severity":  str,          # "low" | "medium" | "high"
      "message":   str,          # human-readable explanation
      "action":    str,          # what to do
      "metrics":   dict,         # supporting numbers
  }
"""
import logging
from datetime import date, timedelta
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class HedgeMonitor:
    """
    Stateless monitor — call check_triggers() with current hedge state.

    The caller is responsible for persisting hedge state between calls
    (entry_spot, initial_delta, etc.).  This class only computes and
    returns alerts given the current market data.
    """

    def check_triggers(
        self,
        # Hedge identification
        hedge_ticker: str,
        option_type: str,           # "put" | "call"
        strike: float,
        expiry_str: str,
        n_contracts: int,
        # Entry conditions (captured when recommendation was accepted)
        entry_spot: float,          # spot price when hedge was initiated
        initial_delta: float,       # delta at inception (e.g. -0.35 for a put)
        # Optional: underlying position context
        underlying_ticker: Optional[str] = None,
        # Thresholds (tunable per user)
        roll_dte_threshold: int = 21,
        delta_drift_threshold: float = 0.20,   # 20% relative drift
        spot_move_threshold: float = 0.05,      # 5% underlying move
    ) -> List[Dict]:
        """
        Evaluate all re-hedge triggers for an active hedge position.

        Returns a list of Alert dicts (may be empty if no triggers fire).
        """
        alerts = []

        # ── 1. Roll trigger (DTE) ─────────────────────────────────────────────
        try:
            expiry = date.fromisoformat(expiry_str)
            dte    = (expiry - date.today()).days

            if dte <= 0:
                alerts.append({
                    "trigger":  "roll",
                    "severity": "high",
                    "message":  f"Hedge on {hedge_ticker} has EXPIRED (DTE={dte}). "
                                f"Position is unhedged.",
                    "action":   "Immediately open a new hedge position. "
                                "Run OptionQ analysis for current recommendations.",
                    "metrics":  {"dte": dte, "expiry": expiry_str},
                })
            elif dte <= roll_dte_threshold:
                alerts.append({
                    "trigger":  "roll",
                    "severity": "high" if dte < 10 else "medium",
                    "message":  f"Hedge expires in {dte} days ({expiry_str}). "
                                f"Below the {roll_dte_threshold}-DTE roll threshold — "
                                f"theta decay is accelerating.",
                    "action":   "Roll to the next available expiry cycle. "
                                "Close current hedge and re-open at 30–45 DTE.",
                    "metrics":  {"dte": dte, "expiry": expiry_str,
                                 "roll_threshold": roll_dte_threshold},
                })
            elif dte <= 45:
                alerts.append({
                    "trigger":  "roll",
                    "severity": "low",
                    "message":  f"Hedge has {dte} DTE — approaching roll window.",
                    "action":   "Begin monitoring for roll. Act when DTE reaches "
                                f"{roll_dte_threshold}.",
                    "metrics":  {"dte": dte, "expiry": expiry_str},
                })
        except Exception as e:
            logger.warning("hedge_monitor: DTE check failed: %s", e)

        # ── 2. Delta drift trigger ─────────────────────────────────────────────
        try:
            current_delta = self._current_delta(
                hedge_ticker, option_type, strike, expiry_str
            )
            if current_delta is not None and abs(initial_delta) > 1e-6:
                drift = abs((current_delta - initial_delta) / initial_delta)
                if drift >= delta_drift_threshold:
                    severity = "high" if drift >= 0.40 else "medium"
                    alerts.append({
                        "trigger":  "delta_drift",
                        "severity": severity,
                        "message":  (
                            f"Hedge delta has drifted {drift:.0%} from inception. "
                            f"Initial Δ={initial_delta:+.3f} → Current Δ={current_delta:+.3f}. "
                            f"The hedge now provides only "
                            f"{abs(current_delta/initial_delta):.0%} of original protection."
                        ),
                        "action":   "Re-balance: add contracts or roll to an ATM strike "
                                    "to restore target delta coverage.",
                        "metrics":  {
                            "initial_delta":  round(initial_delta, 4),
                            "current_delta":  round(current_delta, 4),
                            "drift_pct":      round(drift, 4),
                            "threshold_pct":  delta_drift_threshold,
                        },
                    })
        except Exception as e:
            logger.warning("hedge_monitor: delta drift check failed: %s", e)

        # ── 3. Spot move trigger ───────────────────────────────────────────────
        try:
            underlying = underlying_ticker or hedge_ticker
            current_spot = self._current_spot(underlying)
            if current_spot and entry_spot > 0:
                spot_chg = (current_spot - entry_spot) / entry_spot
                if abs(spot_chg) >= spot_move_threshold:
                    direction_str = "rallied" if spot_chg > 0 else "fallen"
                    severity = "high" if abs(spot_chg) >= 0.10 else "medium"
                    alerts.append({
                        "trigger":  "spot_move",
                        "severity": severity,
                        "message":  (
                            f"{underlying} has {direction_str} "
                            f"{abs(spot_chg):.1%} since hedge inception "
                            f"(${entry_spot:.2f} → ${current_spot:.2f}). "
                            + (
                                "Put is now further OTM — reduced protection."
                                if (spot_chg > 0 and option_type == "put") else
                                "Call is now ITM — consider taking profits or rolling up."
                                if (spot_chg > 0 and option_type == "call") else
                                "Put is now ITM — hedge is working. Evaluate rolling down."
                                if (spot_chg < 0 and option_type == "put") else
                                "Call is now OTM — reduced upside protection."
                            )
                        ),
                        "action":   "Re-run OptionQ analysis with current market prices "
                                    "to get updated hedge recommendations.",
                        "metrics":  {
                            "entry_spot":   round(entry_spot, 2),
                            "current_spot": round(current_spot, 2),
                            "spot_change":  round(spot_chg, 4),
                            "strike":       strike,
                            "moneyness":    round(strike / current_spot, 4),
                        },
                    })
        except Exception as e:
            logger.warning("hedge_monitor: spot move check failed: %s", e)

        # ── 4. Earnings trigger ────────────────────────────────────────────────
        try:
            from backend.data.earnings_calendar import crosses_earnings
            ed = crosses_earnings(hedge_ticker, expiry_str, window_before_days=5)
            if ed:
                alerts.append({
                    "trigger":  "earnings",
                    "severity": "medium",
                    "message":  (
                        f"{hedge_ticker} has earnings on {ed}. "
                        f"IV will spike into the date then crush immediately after. "
                        f"A hedge expiring near earnings may lose 30–50% of value "
                        f"the day after the print."
                    ),
                    "action":   "Consider: (a) closing before earnings if IV premium "
                                "has raised the hedge's value, or (b) rolling to an "
                                "expiry well past the earnings date.",
                    "metrics":  {"earnings_date": ed, "expiry": expiry_str},
                })
        except Exception as e:
            logger.warning("hedge_monitor: earnings check failed: %s", e)

        return alerts

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _current_delta(
        self,
        ticker: str,
        option_type: str,
        strike: float,
        expiry_str: str,
    ) -> Optional[float]:
        """Recompute BSM delta at current spot and vol."""
        try:
            from backend.data.market_data import (
                get_current_price, get_risk_free_rate,
                get_historical_volatility, get_dividend_yield,
            )
            from backend.instruments.options.bsm_pricer import bsm_greeks
            from backend.data.options_chain import get_strike_iv

            spot  = get_current_price(ticker)
            r     = get_risk_free_rate()
            q     = get_dividend_yield(ticker)
            sigma = get_strike_iv(ticker, expiry_str, strike, option_type) \
                    or get_historical_volatility(ticker)

            expiry = date.fromisoformat(expiry_str)
            T = max((expiry - date.today()).days / 365.0, 1.0 / 365.0)

            g = bsm_greeks(spot, strike, r, q, sigma, T, option_type)
            return g["delta"]
        except Exception:
            return None

    @staticmethod
    def _current_spot(ticker: str) -> Optional[float]:
        try:
            from backend.data.market_data import get_current_price
            return get_current_price(ticker)
        except Exception:
            return None
