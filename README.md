# OptionQ — Adaptive Options Hedge Engine

A full-stack quantitative platform that **detects market regimes**, **recommends optimal hedge instruments**, and **explains every decision in plain English**. Built around a 12-layer pipeline that goes from raw portfolio input to sized, priced, and scored hedge candidates — with a live Next.js UI and a FastAPI backend.

---

## What It Does

You describe your portfolio (stocks or existing options positions). OptionQ runs it through a multi-stage pipeline and returns ranked hedge candidates — each with a recommended instrument, strike, expiry, contract count, cost, Greeks, and an AI-generated explanation of why that hedge fits your position.

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

## Repository Structure

```
optionQ/
├── backend/
│   ├── api/                    # FastAPI route handlers
│   │   ├── routes_portfolio.py # POST /portfolio/analyze
│   │   ├── routes_hedge.py     # Single-instrument pipelines
│   │   ├── routes_monitor.py   # Admin: training logs, hedge alerts
│   │   └── routes_execute.py   # Paper / live order execution
│   ├── config/                 # Settings, instrument universe
│   ├── core/
│   │   ├── explainer/          # LLM explainer (Claude / Ollama / HuggingFace)
│   │   ├── greeks/             # Greeks calculators per asset class
│   │   ├── pricing/            # BSM, Black-76, Binomial, MC, GK pricers
│   │   ├── risk/               # VaR, hedge monitor, factor decomposer
│   │   ├── simulation/         # Monte Carlo, payoff, scenario engine
│   │   └── sizing/             # Position sizers per asset class
│   ├── data/                   # Market data, options chain, rates, VIX
│   ├── engines/                # 12-layer pipeline (layer_01 → layer_12)
│   ├── execution/              # Alpaca broker, paper broker, TWAP executor
│   ├── instruments/            # Options, futures, forwards, swaps, inverse ETFs
│   ├── ml/
│   │   ├── regime/             # HDBSCAN main + fast regime detectors
│   │   ├── volatility/         # GARCH, IV surface, vol regime features
│   │   └── nlp/                # FinBERT, news ingester, earnings parser
│   ├── models/                 # Pydantic request/response models
│   ├── tests/                  # Pipeline, pricing, risk, instrument tests
│   └── main.py                 # FastAPI app entry point
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx            # Main portfolio hedge UI
│   │   └── layout.tsx
│   ├── lib/api.ts              # Typed API client
│   └── types/hedgeos.ts        # Shared TypeScript types
│
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.13
- Node.js 18+
- API keys: Anthropic (or Ollama running locally), Alpaca (optional), FRED (optional)

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Copy and fill in your keys
cp .env.example .env             # edit ANTHROPIC_API_KEY, etc.

# Run
uvicorn backend.main:app --reload --port 8000
```

Backend runs at `http://localhost:8000`. Interactive API docs at `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:3000`.

---

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
