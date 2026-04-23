// Mirrors backend Pydantic models exactly

export type AssetClass    = "equity" | "commodity" | "bond" | "fx" | "crypto";
export type ExecutionMode = "analyze" | "paper" | "live";

export interface StockPosition {
  position_type: "stock";
  ticker: string;
  shares: number;
  purchase_price: number;
  purchase_date: string;     // ISO date "YYYY-MM-DD"
  asset_class: AssetClass;
}

export interface OptionsPosition {
  position_type: "option";
  ticker: string;
  option_type: "call" | "put";
  strike: number;
  expiry: string;            // ISO date "YYYY-MM-DD"
  contracts: number;
  direction: "long" | "short";
  premium_paid?: number;
  asset_class: "equity";
}

export type PositionInput = StockPosition | OptionsPosition;

// Backwards compat alias
export type HoldingInput = StockPosition;

export interface PortfolioInput {
  holdings: PositionInput[];
  total_notional: number;
  hedge_horizon_days: number;
  protection_level: number;    // 0.01–0.50
  max_hedge_cost_pct: number;  // 0.005–0.20
  upside_preservation: boolean;
  execution_mode: ExecutionMode;
}

export interface InstrumentCandidate {
  instrument_type: string;
  asset_class: string;
  ticker: string;
  strategy: string;

  /** Relationship to user's position:
   *  "direct_hedge" — same underlying, opposite direction
   *  "cross_hedge"  — correlated sector ETF / index
   *  "macro_hedge"  — uncorrelated macro asset (GLD, TLT …)
   */
  hedge_category: "direct_hedge" | "cross_hedge" | "macro_hedge";

  strike?: number;
  expiry_date?: string;
  option_type?: string;

  n_contracts: number;
  total_cost: number;
  max_protection: number;

  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
  lambda_leverage: number;

  basis_risk_r2: number;
  efficiency_ratio: number;
  score: number;

  rationale: string;
  pros: string[];
  cons: string[];
  when_works_best: string;
  when_fails: string;

  // Gap 8 — actual market mid price from options chain
  market_premium: number;       // (bid+ask)/2 per contract
  market_total_cost: number;    // market_premium × 100 × n_contracts

  extended_metrics: Record<string, unknown>;
}

export interface HedgeRecommendation {
  rank: number;
  asset_ticker: string;
  candidates: InstrumentCandidate[];
}

export interface HedgeOutput {
  portfolio_notional: number;
  regime: string;
  is_anomaly: boolean;
  vix_level: number;
  recommendations: HedgeRecommendation[];
  run_time_seconds: number;
  portfolio_summary: string;
  key_risks: string[];
  regime_commentary: string;
  top_recommendation: string;
  llm_provider: string;

  // Gap 2 — portfolio-level hedge Greeks (sum of top candidate per holding)
  hedge_portfolio_delta: number;
  hedge_portfolio_gamma: number;
  hedge_portfolio_vega: number;
}
