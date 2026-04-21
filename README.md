# OptionQ: Multi-Asset Derivative Hedge Engine

A full-stack quantitative platform that **detects market regimes**, **recommends optimal hedge instruments**, and **explains every decision in plain English**. Built around a 12-layer pipeline that goes from raw portfolio input to sized, priced, and scored hedge candidates — with a live Next.js UI and a FastAPI backend.

---

## What It Does

You describe your portfolio (stocks or existing options positions). OptionQ runs it through a multi-stage pipeline and returns ranked hedge candidates where each with a recommended instrument, strike, expiry, contract count, cost, Greeks, and an AI-generated explanation of why that hedge fits your position.

---

## Key Capabilities

**Regime-Aware Hedging**
Two ML models (10-year main + 6-month fast) continuously classify the market into volatility regimes using HDBSCAN clustering. Hedge recommendations adapt to whether the market is in a calm, stressed, or crisis regime.

**12-Layer Analysis Pipeline**
Each portfolio request flows through: input parsing → market context → risk analysis → stress testing → instrument selection → LLM explanation → pricing → position sizing → Greeks calculation → Monte Carlo simulation → scoring/ranking → output formatting.

**Multi-Instrument Support**
Selects from equity options, futures, inverse ETFs, forwards, swaps, and cross-hedges (e.g., hedging a tech stock with QQQ puts). Scores each candidate on cost efficiency, liquidity, coverage, and regime fit.

**Accurate Option Sizing**
Two-stage sizing: delta-target sizing first, then a budget gate (`max_hedge_cost_pct`) that caps contract count to what the portfolio can afford. Coverage % is shown on every candidate card.

**LLM Explainer Layer**
Pluggable explainer supports Claude (Anthropic), Ollama (local Llama), or HuggingFace models. Generates a plain-English rationale for each hedge recommendation.

**Hedge Monitor**
Active hedge monitoring checks for roll triggers (DTE < threshold), delta drift, spot moves, and earnings IV-crush risk. Alerts surface via the `/admin/hedge/check-triggers` endpoint.

**Live Execution**
Paper and live execution via Alpaca broker integration with TWAP order splitting.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 16 (Turbopack), React 19, TypeScript, Tailwind CSS |
| **Backend** | FastAPI, Python 3.13, Uvicorn, APScheduler |
| **ML / Regime** | HDBSCAN, scikit-learn, statsmodels, GARCH |
| **Options Pricing** | Black-Scholes-Merton, Black-76, Binomial, Garman-Kohlhagen, Monte Carlo |
| **Data** | yfinance, FRED API, NewsAPI |
| **LLM** | Claude (Anthropic API), Ollama, HuggingFace |
| **Execution** | Alpaca Markets API (paper + live) |
| **Caching** | SQLite-backed response cache |
| **NLP** | FinBERT sentiment scoring, earnings transcript parsing |

---

### Visual Representation
<img width="1470" height="835" alt="Screenshot 2026-04-19 at 12 05 11 PM" src="https://github.com/user-attachments/assets/3aca950d-3fe7-4220-9b73-293f7367f131" />

<img width="356" height="688" alt="Screenshot 2026-04-19 at 12 04 45 PM" src="https://github.com/user-attachments/assets/795ebb84-4f05-4962-9b58-e7d81e86487b" />

<img width="584" height="662" alt="Screenshot 2026-04-19 at 12 05 59 PM" src="https://github.com/user-attachments/assets/93d43510-e4cb-454a-a2f1-661528723cbf" />

<img width="860" height="589" alt="Screenshot 2026-04-19 at 12 06 22 PM" src="https://github.com/user-attachments/assets/3bae9eb5-e48c-4440-b44c-ff27762aeff2" />

<img width="1141" height="666" alt="Screenshot 2026-04-19 at 12 06 55 PM" src="https://github.com/user-attachments/assets/28e5ffcc-4a60-468a-85f6-09ad277c884c" />



## API Overview

| Method | Route | Description |
|---|---|---|
| `POST` | `/portfolio/analyze` | Full 12-layer hedge pipeline |
| `POST` | `/instruments/options` | Single-instrument options analysis |
| `POST` | `/instruments/futures` | Futures hedge analysis |
| `POST` | `/admin/hedge/check-triggers` | Check roll/delta-drift/spot-move alerts |
| `GET` | `/admin/earnings/{ticker}` | Next earnings date for a ticker |
| `GET` | `/admin/training/status` | ML model ages + next retrain schedule |
| `POST` | `/admin/retrain/main` | Force-retrain the 10-year regime model |
| `POST` | `/admin/retrain/fast` | Force-retrain the 6-month regime model |
| `GET` | `/health` | Liveness probe |

---

## Supported Asset Classes

| Asset Class | Status |
|---|---|
| **Equities** | Full pipeline — options, futures, inverse ETFs, cross-hedges |
| **FX** | Options pricing via Garman-Kohlhagen, forwards |
| **Commodities** | Futures + options (Black-76), sizing available |
| **Interest Rates** | IR derivatives, swap selectors |
| **Credit** | Credit derivative selectors (stub) |

---

## ML Regime Detection

Two models run in parallel and are blended at inference time:

- **Main model** (10-year window): trained monthly, captures long-cycle regimes
- **Fast model** (6-month window): retrained daily, captures short-cycle shifts

Both use HDBSCAN clustering on VIX-derived features. The regime label (calm / stressed / crisis) is used to filter and score hedge candidates — puts score higher in crisis regimes, collars score higher in calm regimes.

---

## Disclaimer

This project is for educational and research purposes only and does not constitute financial advice. Options trading involves substantial risk of loss.
