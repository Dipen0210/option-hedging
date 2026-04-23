# OptionQ: Multi-Asset Derivative Hedge Engine

A full-stack quantitative platform that **detects market regimes**, **recommends optimal hedge instruments**, and **explains every decision in plain English**. Built around a 12-layer pipeline that goes from raw portfolio input to sized, priced, and scored hedge candidates — with a live Next.js UI and a FastAPI backend.

---

## What It Does

You describe your portfolio (stocks or existing options positions). OptionQ runs it through a multi-stage pipeline and returns ranked hedge candidates — each with a recommended instrument, strike, expiry, contract count, actual market premium, Greeks, and an AI-generated explanation of why that hedge fits your position.

---

## Key Capabilities

**Regime-Aware Hedging**
Two GMM models (10-year main + 6-month fast) classify the market into volatility regimes using Gaussian Mixture Models on 10 macro features (VIX level, VIX term slope, IV/RV ratio, SPY/TLT correlation, realized vol, and more). Hedge recommendations adapt to whether the market is in a low-vol, mid-vol, high-vol, or anomaly regime.

**12-Layer Analysis Pipeline**
Each portfolio request flows through: input parsing → market context → risk analysis → instrument selection → LLM explanation → pricing → position sizing → Greeks calculation → Monte Carlo simulation → scoring/ranking → output formatting.

**Accurate IV-Surface Pricing**
Every option leg is priced using the actual implied volatility at its specific strike — not a flat ATM vol. The system interpolates from the live options chain (volatility skew), so OTM puts are priced correctly instead of being systematically underpriced.

**Market Premium Display**
Each recommendation shows the actual market mid price `(bid+ask)/2` from the live chain alongside the BSM model price, so you see exactly what you'd pay at the broker vs what the model estimated.

**Portfolio-Level Greeks**
Aggregates delta, gamma, and vega across all recommended hedges (top candidate per holding) so you can see the net exposure your hedge portfolio provides at a glance.

**Multi-Strategy Options Selection**
- **Protective Put** — long put on underlying, full downside protection
- **Bear Put Spread** — long ATM put + short 10% OTM put, cheaper with capped protection
- **Collar** — long put + short call, near-zero cost with capped upside

**Liquidity + Earnings Filtering**
Candidates are filtered and scored by open interest, bid-ask spread, and whether the expiry crosses an earnings date (IV crush risk).

**LLM Explainer Layer**
Pluggable explainer supports Claude (Anthropic), Ollama (local Llama), or HuggingFace models. Generates plain-English rationale, pros, cons, and scenario analysis for each hedge candidate.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js (Turbopack), React, TypeScript, Tailwind CSS |
| **Backend** | FastAPI, Python 3.13, Uvicorn, APScheduler |
| **ML / Regime** | Gaussian Mixture Models (GMM), scikit-learn, GARCH |
| **Options Pricing** | Black-Scholes-Merton, Black-76, Binomial (CRR), Monte Carlo |
| **Data** | yfinance (options chain, prices, IV surface) |
| **LLM** | Claude (Anthropic API), Ollama, HuggingFace |
| **Caching** | SQLite-backed response cache |

---

## ML Regime Detection

Two GMM models run in parallel and are fused at inference time:

| Model | Window | Retrain | Components | VIX Weight |
|---|---|---|---|---|
| **Main** | 10 years | Monthly | 4 | 65% |
| **Fast** | 6 months | Daily | 3 | 50% |

Fusion: `combined = 0.65 × main + 0.35 × fast` (weighted by confidence)

**10 features**: `vix_level`, `vix_1d_change`, `vix_5d_change`, `realized_vol_20d`, `spy_return_5d`, `vix_term_slope`, `tlt_return_5d`, `hyg_return_5d`, `iv_rv_ratio`, `spy_tlt_corr_20d`

**Anomaly detection**: combined anomaly score ≥ 0.60, or either model individually ≥ 0.85 (hard override)

---

## API Overview

| Method | Route | Description |
|---|---|---|
| `POST` | `/portfolio/analyze` | Full 12-layer hedge pipeline |
| `POST` | `/instruments/options` | Options-only analysis |
| `GET` | `/admin/training/status` | ML model ages + next retrain schedule |
| `POST` | `/admin/retrain/main` | Force-retrain the 10-year GMM model |
| `POST` | `/admin/retrain/fast` | Force-retrain the 6-month GMM model |
| `GET` | `/health` | Liveness probe |

---

## Supported Strategies (Phase 1)

| Strategy | Cost | Downside Protection | Upside |
|---|---|---|---|
| Protective Put | High | Unlimited | Fully preserved |
| Bear Put Spread | Medium | Capped (ATM → −10%) | Preserved |
| Collar | Near-zero | Moderate | Capped |

Cross-hedges (positively correlated ETFs) and macro hedges (GLD, TLT) are generated alongside direct hedges, ranked by basis risk R² and regime fit.

---

## Running Locally

**Backend**
```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

**Frontend**
```bash
cd frontend
npm run dev
```

Frontend runs at `http://localhost:3000`, backend at `http://localhost:8000`.

---

## Visual Representation
<img width="1470" height="835" alt="Screenshot 2026-04-19 at 12 05 11 PM" src="https://github.com/user-attachments/assets/3aca950d-3fe7-4220-9b73-293f7367f131" />

<img width="356" height="688" alt="Screenshot 2026-04-19 at 12 04 45 PM" src="https://github.com/user-attachments/assets/795ebb84-4f05-4962-9b58-e7d81e86487b" />

<img width="584" height="662" alt="Screenshot 2026-04-19 at 12 05 59 PM" src="https://github.com/user-attachments/assets/93d43510-e4cb-454a-a2f1-661528723cbf" />

<img width="860" height="589" alt="Screenshot 2026-04-19 at 12 06 22 PM" src="https://github.com/user-attachments/assets/3bae9eb5-e48c-4440-b44c-ff27762aeff2" />

<img width="1141" height="666" alt="Screenshot 2026-04-19 at 12 06 55 PM" src="https://github.com/user-attachments/assets/28e5ffcc-4a60-468a-85f6-09ad277c884c" />

---

## Disclaimer

This project is for educational and research purposes only and does not constitute financial advice. Options trading involves substantial risk of loss.
