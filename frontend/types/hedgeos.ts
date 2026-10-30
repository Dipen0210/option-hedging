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
  /**
   * What `basis_risk_r2` rests on. "identity" is a direct hedge on the same
   * underlying, where R² = 1 is definitional and the sample fields are
   * correctly absent; "correlation" is measured over `basis_risk_window_days`
   * and the sample fields are present. null/undefined means not measured —
   * never zero.
   */
  basis_risk_source?: string;
  basis_risk_n_obs?: number | null;
  basis_risk_window_days?: number | null;
  basis_risk_stderr?: number | null;
  efficiency_ratio: number;
  score: number;

  rationale: string;
  pros: string[];
  cons: string[];
  /**
   * The subset of `cons` the backend asserted itself — no executable quote,
   * thin market, expiry crosses earnings — rather than taking from the LLM
   * explainer. Rendering `cons` is still correct and still shows all of
   * them; this is here so the contract is visible, and so a future UI can
   * style a disclosure differently from a caveat. Optional because a
   * candidate that carries none omits it.
   */
  mandatory_disclosures?: string[];
  when_works_best: string;
  when_fails: string;

  /** Second leg, for spreads and collars. Absent on a single-leg hedge. */
  short_strike?: number | null;
  short_option_type?: string | null;
  short_contract_symbol?: string | null;

  // ── Executable price vs theoretical price ──────────────────────────────────
  // The backend distinguishes these and the UI must not blur them. A user
  // quoted a theoretical premium as though it were tradeable has been
  // misinformed about what the hedge costs.

  /** "market_mid" when a two-sided quote existed, otherwise "model". */
  premium_source: "market_mid" | "model" | string;
  /** Mid of the live quote, per share. Zero when no quote was published. */
  market_premium: number;
  /** market_premium × 100 × n_contracts. */
  market_total_cost: number;
  /** Theoretical price per share (Black-Scholes / binomial). */
  model_premium: number;
  /** market_premium / model_premium − 1. Null when there is no quote. */
  premium_vs_model_pct?: number | null;
  bid?: number | null;
  ask?: number | null;

  // ── Contract identity ─────────────────────────────────────────────────────
  /** OCC symbol, e.g. "SPY261120P00475000". Null when unresolved. */
  contract_symbol?: string | null;
  /** False means the strike is not one the exchange lists — not placeable. */
  strike_is_listed: boolean;

  // ── Liquidity ─────────────────────────────────────────────────────────────
  /** Whether the contract passed the liquidity screen. */
  liquidity_ok: boolean;
  /** Null means the venue published no figure, NOT that the figure is zero. */
  open_interest?: number | null;
  daily_volume?: number | null;
  bid_ask_spread_pct?: number | null;

  // ── Volatility provenance ─────────────────────────────────────────────────
  /** The IV actually used to price this contract (contract-specific). */
  iv_used?: number | null;
  /** Where that number came from, e.g. "contract_iv", "skew_interp". */
  iv_source: string;

  // ── Event risk ────────────────────────────────────────────────────────────
  earnings_warning: boolean;
  earnings_date?: string | null;

  /** Cheaper partial-coverage alternatives produced by the sizer. */
  partial_hedge_options: Array<Record<string, unknown>>;

  extended_metrics: Record<string, unknown>;
}

export interface UnhedgedRisk {
  notional: number;
  /**
   * 95% VaR in dollars over `horizon_calendar_days` — the hedge horizon the
   * user chose, not a fixed month. This was commented "(1-month)" and the UI
   * label said "1mo VaR", which named a horizon the figure never had.
   */
  var_95: number;
  cvar_95: number;            // Expected Shortfall in dollars, same horizon
  /** Horizon of var_95 / cvar_95. 0 or absent means unknown — name no horizon. */
  horizon_calendar_days?: number;
  horizon_trading_days?: number;
  var_pct: number;            // VaR as % of notional
  beta: number;               // beta vs SPY
  stress_loss_10pct: number;  // dollar loss in -10% SPY shock
  stress_loss_20pct: number;  // dollar loss in -20% SPY shock
}

/**
 * What the risk figures for this holding were measured from.
 *
 * `null` in any numeric field means "not measured" — not zero. The backend
 * computed all of this and used to discard it before responding, so a beta
 * fitted on 130 observations was rendered exactly like one fitted on 2,500.
 */
export interface RiskEstimateProvenance {
  /** False means the whole profile is a placeholder; read no figure as risk. */
  data_available: boolean;
  unavailable_reason: string;

  beta_available: boolean;
  beta_r2?: number | null;
  beta_stderr?: number | null;
  beta_n_obs?: number | null;

  /** "parametric_cornish_fisher" is estimated; "assumed_vol_fallback" is not. */
  var_method: string;
  var_sigma_source: string;
  var_n_obs?: number | null;

  hedge_ratio_r2?: number | null;
  hedge_ratio_stderr?: number | null;
  hedge_ratio_n_obs?: number | null;

  /** Exposure name → why it could not be measured. */
  factors_unavailable: Record<string, string>;
}

export interface HedgeRecommendation {
  rank: number;
  asset_ticker: string;
  candidates: InstrumentCandidate[];
  unhedged_risk?: UnhedgedRisk;
  risk_provenance?: RiskEstimateProvenance | null;
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
  llm_notice: string;

  // Gap 2 — portfolio-level hedge Greeks (sum of top candidate per holding).
  // Position-level: the backend scales each candidate by n_contracts × 100.
  hedge_portfolio_delta: number;
  hedge_portfolio_gamma: number;
  hedge_portfolio_vega: number;

  // ── Portfolio hedge cost vs the user's budget ──────────────────────────────
  /** What the recommended set actually costs. Negative is a net credit. */
  hedge_portfolio_cost: number;
  /** max_hedge_cost_pct × portfolio_notional. */
  hedge_budget_total: number;
  /** cost / budget. 1.0 is exactly at the cap. */
  hedge_budget_utilization: number;

  // ── Valuation provenance ──────────────────────────────────────────────────
  // portfolio_notional is the server-computed market value (current prices ×
  // quantities), not whatever the client submitted. These expose what it was
  // derived from, so cost basis and current exposure stay distinguishable.
  portfolio_cost_basis: number;
  portfolio_unrealized_pnl: number;
  /** What the client submitted, if anything. Null when nothing was sent. */
  declared_notional?: number | null;

  // ── Regime provenance ─────────────────────────────────────────────────────
  /** "gmm_combined" for the full model, "vix_fallback" when degraded. */
  regime_source: string;
  regime_degraded: boolean;
  regime_degradation_reason: string;

  // ── Data quality ──────────────────────────────────────────────────────────
  /**
   * Per-request provenance ledger. An EMPTY object does not mean "everything
   * was live" — it means nothing was recorded. Render accordingly.
   */
  data_quality: DataQuality;

  /**
   * User-facing warnings raised anywhere in the pipeline: valuation
   * reconciliation, holdings left unhedged, candidates excluded for breaching
   * the budget, degraded regime detection, stale or unavailable data. These
   * are the product's disclosures and must be shown, not swallowed.
   */
  warnings: string[];
}

/**
 * One provenance record from the request-scoped ledger
 * (`DataPoint.to_dict` in backend/data/data_quality.py).
 */
export interface DataPoint {
  /** e.g. "SPY.spot", "AAPL.iv" — the value whose origin this describes. */
  name: string;
  source: "live" | "cache" | "fallback" | "unavailable" | string;
  degraded: boolean;
  /** ISO timestamp the value itself is as of — not when it was read. */
  as_of?: string;
  age_seconds?: number;
  detail?: string;
}

/**
 * Where every number in this response came from.
 *
 * Shaped to `DataQualityLog.summary()`, which is the authority — the previous
 * declaration here guessed the field names (`as_of` for `generated_at`,
 * strings for `degraded_fields`) and so every one of them read as
 * `undefined` at runtime while the compiler was satisfied.
 */
export interface DataQuality {
  /** When the snapshot was taken (NOT the age of the data in it). */
  generated_at?: string;
  /** How many values came from each source, e.g. { live: 98, cache: 4 }. */
  source_counts?: Record<string, number>;
  degraded_fields?: DataPoint[];
  warnings?: string[];
  /** True when any field was cached, substituted, or missing. */
  is_degraded?: boolean;
}
