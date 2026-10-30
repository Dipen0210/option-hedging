"use client";

import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { api, ApiError } from "@/lib/api";
import type {
  HedgeOutput,
  AssetClass,
  DataQuality,
  InstrumentCandidate,
  HedgeRecommendation,
  RiskEstimateProvenance,
} from "@/types/hedgeos";

// ─── Ticker universe ──────────────────────────────────────────────────────────

const TICKER_GROUPS: Record<string, string[]> = {
  "Index ETFs":     ["SPY", "QQQ", "IWM", "DIA", "VTI"],
  "Hedge Products": ["GLD", "TLT", "HYG", "LQD", "UUP"],
  "Sector ETFs":    ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE"],
  "Mega-Cap":       [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AMD", "TSLA", "NFLX", "CRM",
    "JPM",  "BAC",  "GS",   "MS",    "V",    "MA",
    "JNJ",  "UNH",  "PFE",  "ABBV",
    "XOM",  "CVX",  "BRKB", "WMT",   "KO",
  ],
};
const ALL_TICKERS = Object.values(TICKER_GROUPS).flat();

// ─── Greeks reference ─────────────────────────────────────────────────────────

const GREEK_INFO: Record<string, { label: string; desc: string }> = {
  delta: { label: "Δ Delta", desc: "Option price change per $1 move in the underlying. Negative delta = profits when the stock falls." },
  gamma: { label: "Γ Gamma", desc: "Rate at which delta changes. High gamma means hedge effectiveness shifts quickly as price moves." },
  theta: { label: "Θ Theta", desc: "Daily time-decay cost in dollars. Negative = option loses this amount of value each day held." },
  vega:  { label: "ν Vega",  desc: "Price change per 1% increase in implied volatility. Positive vega benefits when VIX rises." },
  rho:   { label: "ρ Rho",   desc: "Sensitivity to interest rate changes. More relevant for long-dated LEAPS." },
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

function fmt$(n: number) {
  return "$" + n.toLocaleString("en-US", { maximumFractionDigits: 0 });
}
function fmtPct(n: number, d = 1) {
  return (n * 100).toFixed(d) + "%";
}
function fmtDec(n: number, d = 3) {
  return (n >= 0 ? "+" : "") + n.toFixed(d);
}

type ParsedTicker =
  | { isOption: true;  underlying: string; optType: "call" | "put"; strike: number; displayLabel: string }
  | { isOption: false; ticker: string; displayLabel: string };

function parseAssetTicker(raw: string): ParsedTicker {
  const m = raw.match(/^([A-Z]+)_([CP])(\d+(?:\.\d+)?)$/);
  if (m) {
    const optType = m[2] === "C" ? "call" : "put";
    return { isOption: true, underlying: m[1], optType, strike: parseFloat(m[3]),
      displayLabel: `${m[1]} ${optType === "call" ? "Call" : "Put"} $${m[3]}` };
  }
  return { isOption: false, ticker: raw, displayLabel: raw };
}

// ─── Colour logic ─────────────────────────────────────────────────────────────

function scoreColor(s: number) {
  if (s >= 60) return { text: "text-emerald-400", bg: "bg-emerald-500", ring: "ring-emerald-500/40 bg-emerald-500/10 text-emerald-400" };
  if (s >= 40) return { text: "text-amber-400",   bg: "bg-amber-500",   ring: "ring-amber-500/40   bg-amber-500/10   text-amber-400"   };
  return         { text: "text-red-400",    bg: "bg-red-500",    ring: "ring-red-500/40    bg-red-500/10    text-red-400"    };
}
function regimeStyle(regime: string, anomaly: boolean) {
  if (anomaly)               return { cls: "bg-purple-500/15 text-purple-300 ring-1 ring-purple-500/40", dot: "bg-purple-400" };
  if (regime === "high_vol") return { cls: "bg-red-500/15    text-red-300    ring-1 ring-red-500/40",    dot: "bg-red-400"    };
  if (regime === "mid_vol")  return { cls: "bg-amber-500/15  text-amber-300  ring-1 ring-amber-500/40",  dot: "bg-amber-400"  };
  return                            { cls: "bg-sky-500/15    text-sky-300    ring-1 ring-sky-500/40",    dot: "bg-sky-400"    };
}
function costStyle(pct: number) {
  if (pct > 0.08) return "text-red-400";
  if (pct > 0.04) return "text-amber-400";
  return "text-emerald-400";
}

// ─── TickerSelect ─────────────────────────────────────────────────────────────

function TickerSelect({ value, onChange }: { value: string; onChange: (t: string) => void }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState(value);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const fn = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", fn);
    return () => document.removeEventListener("mousedown", fn);
  }, []);

  const filtered = query ? ALL_TICKERS.filter((t) => t.startsWith(query.toUpperCase())) : null;
  const groups: Record<string, string[]> = filtered ? { Results: filtered } : TICKER_GROUPS;

  const pick = (t: string) => { onChange(t); setQuery(t); setOpen(false); };

  return (
    <div ref={ref} className="relative">
      <input
        className="w-full bg-[#111318] border border-white/8 rounded-lg px-3 py-2 text-sm font-mono text-white placeholder-white/20 focus:outline-none focus:border-sky-500/60 focus:ring-1 focus:ring-sky-500/20 uppercase transition-all"
        placeholder="Ticker…"
        value={query}
        onChange={(e) => { setQuery(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
      />
      {open && (
        <div className="absolute z-50 top-full mt-1 left-0 min-w-[220px] w-full bg-[#0e0f14] border border-white/10 rounded-xl shadow-2xl shadow-black/60 max-h-72 overflow-y-auto">
          {Object.entries(groups).map(([grp, tickers]) => {
            const shown = tickers.filter((t) => !query || t.startsWith(query.toUpperCase()));
            if (!shown.length) return null;
            return (
              <div key={grp}>
                <p className="px-3 pt-2.5 pb-1 text-[9px] font-semibold text-white/30 uppercase tracking-wider sticky top-0 bg-[#0e0f14]">{grp}</p>
                <div className="grid grid-cols-3 gap-0.5 px-2 pb-2">
                  {shown.map((t) => (
                    <button key={t} onMouseDown={() => pick(t)}
                      className={`text-left px-2 py-1.5 rounded-lg text-xs font-mono transition-colors truncate ${t === value ? "bg-sky-600 text-white" : "text-white/60 hover:bg-white/8 hover:text-white"}`}>
                      {t}
                    </button>
                  ))}
                </div>
              </div>
            );
          })}
          {filtered?.length === 0 && <p className="px-3 py-3 text-xs text-white/30">No match — type any ticker.</p>}
        </div>
      )}
    </div>
  );
}

// ─── Greek badge ──────────────────────────────────────────────────────────────

function GreekBadge({ name, value }: { name: string; value: number }) {
  const [tip, setTip] = useState(false);
  const info = GREEK_INFO[name];
  const d = name === "gamma" ? 5 : 3;
  return (
    <div className="relative">
      <button
        onMouseEnter={() => setTip(true)} onMouseLeave={() => setTip(false)}
        className="flex flex-col items-center gap-0.5 px-3 py-2.5 rounded-xl bg-white/4 hover:bg-white/8 border border-white/8 transition-all cursor-help min-w-[72px]"
      >
        <span className="text-[10px] text-white/35 font-medium">{info?.label ?? name}</span>
        <span className={`font-mono text-sm font-bold ${value < 0 ? "text-red-400" : "text-emerald-400"}`}>
          {fmtDec(value, d)}
        </span>
      </button>
      {tip && info && (
        <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 w-60 bg-[#0e0f14] border border-white/10 rounded-xl p-3 text-xs text-white/60 shadow-2xl z-50 leading-relaxed pointer-events-none">
          <p className="font-semibold text-white mb-1">{info.label}</p>
          {info.desc}
        </div>
      )}
    </div>
  );
}

// ─── Hedge category badge ─────────────────────────────────────────────────────

function HedgeCategoryBadge({ category }: { category: string }) {
  const cfg: Record<string, { label: string; cls: string; tip: string }> = {
    direct_hedge: { label: "Direct",  cls: "bg-orange-500/15 text-orange-300 ring-1 ring-orange-500/30", tip: "Same underlying, opposite direction. Zero basis risk." },
    cross_hedge:  { label: "Cross",   cls: "bg-amber-500/15  text-amber-300  ring-1 ring-amber-500/30",  tip: "Correlated sector ETF — hedges via price correlation." },
    macro_hedge:  { label: "Macro",   cls: "bg-sky-500/15    text-sky-300    ring-1 ring-sky-500/30",    tip: "Uncorrelated macro asset (GLD, TLT…) — diversification." },
  };
  const [tip, setTip] = useState(false);
  const c = cfg[category] ?? cfg.cross_hedge;
  return (
    <div className="relative inline-block">
      <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded-md cursor-help ${c.cls}`}
        onMouseEnter={() => setTip(true)} onMouseLeave={() => setTip(false)}>
        {c.label}
      </span>
      {tip && (
        <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 w-52 bg-[#0e0f14] border border-white/10 rounded-xl p-2.5 text-xs text-white/60 shadow-2xl z-50 pointer-events-none leading-relaxed">
          {c.tip}
        </div>
      )}
    </div>
  );
}

// ─── Candidate card (replaces table row) ─────────────────────────────────────

/**
 * Whether this premium is one you could trade on.
 *
 * `market_mid` means a two-sided quote existed and the midpoint was used.
 * Anything else means the figure is a model output: the venue published no
 * quote (outside 09:30–16:00 ET it publishes none at all), so the number is
 * theoretical. The two were rendered identically, which is the difference
 * between "this costs $4,200" and "this would cost about $4,200 if it traded
 * at fair value" — and the budget check downstream is only as meaningful as
 * the price it ran against.
 */
function PremiumSourceBadge({ c }: { c: InstrumentCandidate }) {
  const market = c.premium_source === "market_mid";
  return (
    <span
      className={`text-[10px] px-2 py-0.5 rounded-full border font-medium ${
        market
          ? "bg-emerald-500/10 text-emerald-300/80 border-emerald-500/20"
          : "bg-amber-400/10 text-amber-300/80 border-amber-400/20"
      }`}
      title={market
        ? "Priced from the live bid/ask midpoint"
        : "No two-sided quote was available — this premium is a model estimate, not an executable price"}
    >
      {market ? "market price" : "model estimate"}
    </span>
  );
}

/**
 * Contract identity, tradability, and the vol the price was struck with.
 *
 * Every row here answers a question a user would otherwise have to take on
 * trust: is this a real listed contract (`contract_symbol`,
 * `strike_is_listed`), could I get out of it (`open_interest`,
 * `daily_volume`, `bid_ask_spread_pct`), and what volatility assumption is
 * the premium resting on (`iv_used`, `iv_source`).
 *
 * A null liquidity figure means the venue published nothing, which is NOT
 * zero. Rendering a missing open interest as "0 OI" would invent a fact — and
 * the most alarming one available — so missing stays visibly missing.
 */
function ProvenanceStrip({ c }: { c: InstrumentCandidate }) {
  const num = (v: number | null | undefined) =>
    v == null ? null : v.toLocaleString();

  const rows: { label: string; value: React.ReactNode; warn?: boolean }[] = [];

  if (c.contract_symbol) {
    rows.push({
      label: "Contract",
      value: <span className="font-mono">{c.contract_symbol}</span>,
    });
  }
  if (c.short_contract_symbol) {
    rows.push({
      label: "Short leg",
      value: <span className="font-mono">{c.short_contract_symbol}</span>,
    });
  }
  if (c.strike_is_listed === false) {
    rows.push({
      label: "Strike",
      value: "requested strike is not listed — the nearest listed strike was used",
      warn: true,
    });
  }

  if (c.bid != null && c.ask != null && (c.bid > 0 || c.ask > 0)) {
    rows.push({
      label: "Quote",
      value: (
        <span className="font-mono">
          {fmt$(c.bid)} / {fmt$(c.ask)}
          {c.bid_ask_spread_pct != null && (
            <span className="text-white/30"> ({(c.bid_ask_spread_pct * 100).toFixed(1)}% wide)</span>
          )}
        </span>
      ),
      warn: c.bid_ask_spread_pct != null && c.bid_ask_spread_pct > 0.10,
    });
  } else {
    rows.push({
      label: "Quote",
      value: "none published — the market for this contract was closed or untraded",
      warn: true,
    });
  }

  const oi  = num(c.open_interest);
  const vol = num(c.daily_volume);
  rows.push({
    label: "Liquidity",
    value:
      oi == null && vol == null
        ? "not published"
        : `${oi ?? "—"} open interest · ${vol ?? "—"} traded today`,
    warn: c.liquidity_ok === false,
  });

  if (c.iv_used != null && c.iv_used > 0) {
    // The backend emits exactly these four, in descending order of how much
    // they actually say about THIS contract (selector.py `_get_vol` /
    // `_price_option`). Only the first two come from the option market; the
    // other two are substitutes, and a realized or market-wide figure
    // standing in for implied vol can misprice a skewed strike badly — a
    // 85%-moneyness SPY put quoted 24.7% IV against 10.0% realized. Showing
    // the raw slug told the user nothing, so each is spelled out.
    const ivWording: Record<string, { text: string; warn: boolean }> = {
      contract_iv:    { text: "this strike's own implied vol", warn: false },
      atm_iv:         { text: "at-the-money implied vol — no skew adjustment for this strike", warn: false },
      // Realized vol is backward-looking, and index options trade at a
      // persistent premium to it — measured at 0.2473 implied against 0.0999
      // realized on a live SPY put chain. Naming the substitution is not
      // enough on its own; the direction of the error is what tells the user
      // the premium below is a floor rather than a quote.
      realized_vol:   { text: "substituted: 20-day realized vol, not the option market's view \u2014 options usually trade above realized vol, so this premium is likely an underestimate", warn: true },
      garch_forecast: { text: "substituted: market-wide GARCH forecast, not specific to this underlying", warn: true },
    };
    const iv = ivWording[c.iv_source] ?? {
      text: c.iv_source || "source not reported",
      warn: true,
    };
    rows.push({
      label: "Implied vol",
      value: (
        <>
          <span className="font-mono">{(c.iv_used * 100).toFixed(1)}%</span>
          <span className={iv.warn ? "" : "text-white/30"}> · {iv.text}</span>
        </>
      ),
      warn: iv.warn,
    });
  }

  // What the basis-risk R² was measured over. The tile above shows the
  // number and grades it "Low / Moderate / High basis risk"; that grade is a
  // claim about the hedge, and an R² of 0.62 from the 30-observation minimum
  // does not support it as well as one from a full year. The backend has
  // always returned the sample alongside the correlation and nothing read it.
  if (c.basis_risk_source === "identity") {
    rows.push({
      label: "Basis R²",
      value: "1.00 by identity — the hedge is the same underlying, so there is no tracking error to estimate",
    });
  } else if (c.basis_risk_source === "tail_correlation") {
    rows.push({
      label: "Basis R²",
      value: "bottom-decile correlation, not a full-sample R² — it describes behaviour in sell-offs only",
    });
  } else if (c.basis_risk_n_obs != null) {
    const se = c.basis_risk_stderr;
    rows.push({
      label: "Basis R²",
      value: (
        <>
          <span className="font-mono">{c.basis_risk_r2.toFixed(2)}</span>
          <span className={c.basis_risk_n_obs < 60 ? "" : "text-white/30"}>
            {" "}· {c.basis_risk_n_obs.toLocaleString()} overlapping sessions
            {c.basis_risk_window_days != null && ` in a ${c.basis_risk_window_days}-day window`}
            {se != null && `, ±${se.toFixed(3)} on the correlation`}
          </span>
        </>
      ),
      // A correlation from fewer than ~60 overlapping sessions is too thin to
      // grade a hedge on, so the grade beside it is flagged rather than shown
      // with the same confidence as one from a year of data.
      warn: c.basis_risk_n_obs < 60,
    });
  }

  if (c.earnings_warning) {
    rows.push({
      label: "Earnings",
      value: c.earnings_date
        ? `${c.earnings_warning} (${c.earnings_date})`
        : c.earnings_warning,
      warn: true,
    });
  }

  return (
    <div className="rounded-lg bg-white/3 border border-white/6 p-3">
      <p className="text-[10px] font-semibold text-white/30 uppercase tracking-widest mb-2">
        Contract &amp; data provenance
      </p>
      <dl className="space-y-1.5">
        {rows.map(({ label, value, warn }) => (
          <div key={label} className="flex gap-3 text-[11px] leading-relaxed">
            <dt className="text-white/30 w-20 shrink-0">{label}</dt>
            <dd className={warn ? "text-amber-300/80" : "text-white/55"}>{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/**
 * What the "Without hedge" figures above were measured from.
 *
 * The VaR, the expected shortfall and both stress losses are all produced by
 * the backend with a sample behind them, and beta multiplies straight through
 * the stress numbers. Until this was rendered, a VaR assuming a flat 20%
 * annual vol because the holding had 12 return observations looked exactly
 * like one estimated from two years of its own returns, and a beta explaining
 * 0.4% of a holding's variance was presented with the same authority as one
 * explaining 88%.
 */
function RiskBasisNote({ rp }: { rp: RiskEstimateProvenance }) {
  if (!rp.data_available) {
    return (
      <p className="mt-2 text-[10px] leading-relaxed text-amber-300/80">
        These figures are not measurements.{" "}
        {rp.unavailable_reason || "Risk analysis failed for this holding."}{" "}
        Nothing here may be read as this position&apos;s risk.
      </p>
    );
  }

  const parts: { text: string; warn: boolean }[] = [];

  // VaR method. "assumed_vol_fallback" is the one that matters: it is an
  // assumption about the position, not an estimate from it.
  if (rp.var_method === "assumed_vol_fallback") {
    parts.push({
      text:
        "VaR assumes a flat annual volatility — there were too few return " +
        "observations to estimate this holding’s own distribution, so it is " +
        "an assumption, not an estimate",
      warn: true,
    });
  } else if (rp.var_method === "normal_approximation") {
    parts.push({
      text:
        "VaR uses a normal approximation with no fat-tail correction, so it " +
        "understates the tail",
      warn: true,
    });
  } else if (rp.var_method === "parametric_cornish_fisher") {
    parts.push({
      text:
        `VaR estimated from ${rp.var_n_obs != null ? `${rp.var_n_obs.toLocaleString()} ` : ""}` +
        "daily returns with a fat-tail correction" +
        (rp.var_sigma_source === "garch_forecast"
          ? ", scaled by a GARCH volatility forecast"
          : ""),
      warn: false,
    });
  }

  // Beta, which the stress rows multiply through.
  if (!rp.beta_available) {
    parts.push({
      text: "beta could not be measured — the stress losses beside it are not reliable",
      warn: true,
    });
  } else if (rp.beta_r2 != null && rp.beta_n_obs != null) {
    const weak = rp.beta_r2 < 0.20;
    parts.push({
      text:
        `beta fitted on ${rp.beta_n_obs.toLocaleString()} sessions, ` +
        `R² ${rp.beta_r2.toFixed(2)}` +
        (rp.beta_stderr != null ? ` ±${rp.beta_stderr.toFixed(3)}` : "") +
        (weak
          ? " — SPY explains almost none of this holding’s moves, so the SPY stress rows say little about it"
          : ""),
      warn: weak,
    });
  }

  // Exposures that could not be measured at all. A populated entry is the
  // difference between "insensitive to rates" and "rate sensitivity unknown".
  const unmeasured = Object.keys(rp.factors_unavailable ?? {});
  if (unmeasured.length > 0) {
    parts.push({
      text: `not measured: ${unmeasured.join(", ")}`,
      warn: true,
    });
  }

  if (parts.length === 0) return null;

  return (
    <p className="mt-2 text-[10px] leading-relaxed text-white/30">
      <span className="text-white/25">Measured from: </span>
      {parts.map(({ text, warn }, i) => (
        <span key={i} className={warn ? "text-amber-300/80" : undefined}>
          {i > 0 && <span className="text-white/20"> · </span>}
          {text}
        </span>
      ))}
    </p>
  );
}

function CandidateCard({
  c, notional, rank, defaultOpen,
}: {
  c: InstrumentCandidate; notional: number; rank: number; defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen ?? false);
  const costPct      = notional > 0 ? c.total_cost / notional : 0;
  const coveragePct  = notional > 0 && c.max_protection > 0 ? c.max_protection / notional : 0;
  // L8 stores portfolio-level delta (per_contract_delta × n_contracts × 100); display per-contract
  const perContractDelta = c.n_contracts > 0 ? c.delta / (c.n_contracts * 100) : c.delta;
  const sc = scoreColor(c.score);

  return (
    <div className={`rounded-xl border transition-all ${open ? "border-white/12 bg-white/3" : "border-white/6 bg-white/[0.02] hover:border-white/10 hover:bg-white/[0.035]"}`}>

      {/* Summary row */}
      <button onClick={() => setOpen((o) => !o)} className="w-full text-left px-4 py-3.5">
        <div className="flex items-center gap-3">

          {/* Rank */}
          <span className="text-xs text-white/20 font-mono w-4 shrink-0">{rank}</span>

          {/* Score ring */}
          <div className="relative w-10 h-10 shrink-0">
            <svg className="w-10 h-10 -rotate-90" viewBox="0 0 40 40">
              <circle cx="20" cy="20" r="16" fill="none" stroke="white" strokeOpacity="0.06" strokeWidth="3" />
              <circle cx="20" cy="20" r="16" fill="none"
                stroke={c.score >= 60 ? "#10b981" : c.score >= 40 ? "#f59e0b" : "#ef4444"}
                strokeWidth="3" strokeLinecap="round"
                strokeDasharray={`${(c.score / 100) * 100.5} 100.5`} />
            </svg>
            <span className={`absolute inset-0 flex items-center justify-center text-[11px] font-bold ${sc.text}`}>
              {c.score.toFixed(0)}
            </span>
          </div>

          {/* Strategy + ticker */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm font-semibold text-white leading-tight">{c.strategy}</span>
              <HedgeCategoryBadge category={c.hedge_category} />
              <PremiumSourceBadge c={c} />
            </div>
            <div className="flex items-center gap-2 mt-0.5">
              <span className="text-xs font-mono text-white/35">{c.ticker}</span>
              {c.option_type && <span className="text-[10px] text-white/25">{c.option_type.toUpperCase()}</span>}
              {c.strike && <span className="text-[10px] text-white/25">@ {fmt$(c.strike)}</span>}
              {c.expiry_date && <span className="text-[10px] text-white/20">{c.expiry_date}</span>}
            </div>
          </div>

          {/* Cost + coverage */}
          <div className="text-right shrink-0">
            {c.market_total_cost > 0 ? (
              <>
                <p className="text-sm font-mono font-semibold text-white">{fmt$(c.market_total_cost)}</p>
                <p className="text-[10px] font-mono text-white/25">mkt mid · {fmt$(c.market_premium)}/contract</p>
              </>
            ) : (
              <>
                <p className="text-sm font-mono font-semibold text-white">{fmt$(c.total_cost)}</p>
                {/* Not a market price. Saying so here, not only in the
                    expanded panel, because this is the figure a user compares
                    across cards without ever opening one. */}
                <p className="text-[10px] font-mono text-amber-300/60">modelled · no live quote</p>
              </>
            )}
            <p className={`text-[11px] font-mono ${costStyle(costPct)}`}>{fmtPct(costPct)} of pos.</p>
            <p className={`text-[11px] font-mono mt-0.5 ${coveragePct >= 1 ? "text-emerald-400" : coveragePct >= 0.5 ? "text-amber-400" : "text-red-400"}`}>
              {Math.round(coveragePct * 100)}% covered
            </p>
          </div>

          {/* Delta */}
          <div className="text-right shrink-0 w-16">
            <p className="text-[10px] text-white/25 mb-0.5">delta</p>
            <p className={`text-sm font-mono font-bold ${perContractDelta < 0 ? "text-red-400" : "text-emerald-400"}`}>
              {fmtDec(perContractDelta, 3)}
            </p>
          </div>

          {/* Chevron */}
          <span className={`text-white/20 text-xs transition-transform duration-200 ${open ? "rotate-180" : ""}`}>▾</span>
        </div>
      </button>

      {/* Expanded detail */}
      {open && (
        <div className="px-4 pb-4 space-y-4 border-t border-white/6 pt-4">

          {/* Greeks */}
          <div>
            <p className="text-[10px] font-semibold text-white/30 uppercase tracking-widest mb-2">Greeks — hover for explanation</p>
            <div className="flex flex-wrap gap-2">
              <GreekBadge name="delta" value={perContractDelta} />
              <GreekBadge name="gamma" value={c.gamma} />
              <GreekBadge name="theta" value={c.theta} />
              <GreekBadge name="vega"  value={c.vega}  />
              {c.rho !== 0 && <GreekBadge name="rho" value={c.rho} />}
            </div>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-3 gap-2">
            {[
              // An R² of exactly 0 is what the backend emits when the
              // asset-vs-hedge regression could not be run at all (too few
              // overlapping sessions). A measured relationship, however weak,
              // never lands on exactly zero. Labelling it "High basis risk"
              // read as a measurement of a bad hedge instead of the absence
              // of a measurement; the Data Quality panel carries the reason.
              { label: "Basis Risk R²",
                value: c.basis_risk_r2 === 0 ? "n/a" : c.basis_risk_r2.toFixed(2),
                sub: c.basis_risk_r2 === 0
                  ? "Not measured — see data quality"
                  : c.basis_risk_r2 >= 0.7 ? "Low basis risk"
                  : c.basis_risk_r2 >= 0.5 ? "Moderate" : "High basis risk" },
              { label: "Capital Eff. λ", value: `${c.lambda_leverage.toFixed(2)}×`, sub: "Protection / $ premium" },
              { label: "Max Protection", value: c.max_protection > 0 ? fmt$(c.max_protection) : "Unlimited", sub: `${Math.round(coveragePct * 100)}% of position covered` },
            ].map(({ label, value, sub }) => (
              <div key={label} className="rounded-lg bg-white/3 border border-white/6 p-3">
                <p className="text-[10px] text-white/30 mb-1">{label}</p>
                <p className="text-lg font-mono font-bold text-white leading-none">{value}</p>
                <p className="text-[10px] text-white/25 mt-1">{sub}</p>
              </div>
            ))}
          </div>

          <ProvenanceStrip c={c} />

          {/* Premium breakdown */}
          {c.market_premium <= 0 && (
            <div className="rounded-lg bg-amber-400/6 border border-amber-400/20 p-3">
              <p className="text-[10px] font-semibold text-amber-300/70 uppercase tracking-widest mb-1.5">
                Premium is a model estimate
              </p>
              <p className="text-[11px] text-amber-100/70 leading-relaxed">
                No two-sided quote was available for this contract, so the{" "}
                {fmt$(c.total_cost)} above is what the pricing model says it is worth —
                not what it would cost to buy. Option quotes are only published while the
                market is open (09:30–16:00 ET); re-run during market hours for executable
                pricing before acting on this.
              </p>
            </div>
          )}

          {c.market_premium > 0 && (
            <div className="rounded-lg bg-white/3 border border-white/6 p-3">
              <p className="text-[10px] font-semibold text-white/30 uppercase tracking-widest mb-2">Premium — market vs model</p>
              <div className="flex items-center gap-6 flex-wrap">
                <div>
                  <p className="text-[10px] text-white/25 mb-0.5">Market mid</p>
                  <p className="text-base font-mono font-bold text-white">{fmt$(c.market_premium)}<span className="text-xs text-white/30 font-normal"> /contract</span></p>
                  <p className="text-[10px] text-white/35 mt-0.5">Total {fmt$(c.market_total_cost)} · {c.n_contracts}× lots</p>
                </div>
                <div className="w-px h-8 bg-white/8" />
                <div>
                  <p className="text-[10px] text-white/25 mb-0.5">BSM model</p>
                  <p className="text-base font-mono font-bold text-white/50">
                    {c.n_contracts > 0 ? fmt$(Math.round(c.total_cost / c.n_contracts / 100 * 100) / 100) : "—"}<span className="text-xs text-white/20 font-normal"> /contract</span>
                  </p>
                  <p className="text-[10px] text-white/25 mt-0.5">Total {fmt$(c.total_cost)}</p>
                </div>
                {c.market_total_cost > 0 && c.total_cost > 0 && (
                  <>
                    <div className="w-px h-8 bg-white/8" />
                    <div>
                      <p className="text-[10px] text-white/25 mb-0.5">Difference</p>
                      {(() => {
                        const diff = c.market_total_cost - c.total_cost;
                        const cls  = diff > 0 ? "text-red-400" : "text-emerald-400";
                        return (
                          <p className={`text-base font-mono font-bold ${cls}`}>
                            {diff >= 0 ? "+" : ""}{fmt$(Math.round(diff))}
                          </p>
                        );
                      })()}
                      <p className="text-[10px] text-white/25 mt-0.5">vs BSM estimate</p>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}

          {/* Rationale / Pros / Cons */}
          {(c.rationale || c.pros.length > 0 || c.cons.length > 0) && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
              {c.rationale && (
                <div className="rounded-lg bg-white/3 border border-white/6 p-3">
                  <p className="text-[10px] font-semibold text-white/40 uppercase tracking-widest mb-2">Rationale</p>
                  <p className="text-xs text-white/55 leading-relaxed">{c.rationale}</p>
                </div>
              )}
              {c.pros.length > 0 && (
                <div className="rounded-lg bg-emerald-500/5 border border-emerald-500/15 p-3">
                  <p className="text-[10px] font-semibold text-emerald-400/70 uppercase tracking-widest mb-2">Pros</p>
                  <ul className="space-y-1.5">
                    {c.pros.map((p, i) => (
                      <li key={i} className="text-xs text-white/55 flex gap-2 leading-relaxed">
                        <span className="text-emerald-500 shrink-0 mt-px">✓</span>{p}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {c.cons.length > 0 && (
                <div className="rounded-lg bg-red-500/5 border border-red-500/15 p-3">
                  <p className="text-[10px] font-semibold text-red-400/70 uppercase tracking-widest mb-2">Cons</p>
                  <ul className="space-y-1.5">
                    {c.cons.map((con, i) => (
                      <li key={i} className="text-xs text-white/55 flex gap-2 leading-relaxed">
                        <span className="text-red-400 shrink-0 mt-px">✗</span>{con}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* When works best / When fails — only shown when LLM populated them */}
          {(c.when_works_best || c.when_fails) && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {c.when_works_best && (
                <div className="rounded-lg bg-sky-500/5 border border-sky-500/15 p-3 flex gap-2.5">
                  <span className="text-sky-400 text-sm shrink-0 mt-px">↗</span>
                  <div>
                    <p className="text-[10px] font-semibold text-sky-400/70 uppercase tracking-widest mb-2">Works best when</p>
                    <ul className="space-y-1">
                      {c.when_works_best.split(" - ").map((part, i) => (
                        <li key={i} className="text-xs text-white/55 flex gap-1.5 leading-relaxed">
                          <span className="text-sky-400/60 shrink-0">•</span>{part}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
              {c.when_fails && (
                <div className="rounded-lg bg-orange-500/5 border border-orange-500/15 p-3 flex gap-2.5">
                  <span className="text-orange-400 text-sm shrink-0 mt-px">↘</span>
                  <div>
                    <p className="text-[10px] font-semibold text-orange-400/70 uppercase tracking-widest mb-2">Watch out when</p>
                    <ul className="space-y-1">
                      {c.when_fails.split(" - ").map((part, i) => (
                        <li key={i} className="text-xs text-white/55 flex gap-1.5 leading-relaxed">
                          <span className="text-orange-400/60 shrink-0">•</span>{part}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Holding card ─────────────────────────────────────────────────────────────

interface PositionMeta {
  isOption: boolean;
  label: string;
  detail: string;
  notional: number;
}

function HoldingCard({ rec, meta }: { rec: HedgeRecommendation; meta: PositionMeta }) {
  const sorted = [...rec.candidates].sort((a, b) => b.score - a.score);

  const best = sorted[0];
  const costPct = best && meta.notional > 0 ? best.total_cost / meta.notional : 0;
  const accentClass = meta.isOption ? "from-violet-500/10 to-transparent border-violet-500/20" : "from-sky-500/10 to-transparent border-sky-500/20";
  const tagClass    = meta.isOption ? "bg-violet-500/15 text-violet-300 ring-1 ring-violet-500/25" : "bg-sky-500/15 text-sky-300 ring-1 ring-sky-500/25";

  return (
    <div className="rounded-2xl border border-white/8 bg-[#0d0e13] overflow-hidden">

      {/* Position banner */}
      <div className={`bg-gradient-to-r ${accentClass} border-b px-5 py-4`}>
        <p className="text-[10px] font-semibold text-white/30 uppercase tracking-widest mb-2">Hedging your position</p>
        <div className="flex items-center gap-3 flex-wrap">
          <span className={`text-[11px] font-bold px-2.5 py-1 rounded-lg ${tagClass}`}>
            {meta.isOption ? "Option" : "Stock"}
          </span>
          <span className="text-xl font-bold font-mono text-white tracking-tight">{meta.label}</span>
          {meta.detail && <span className="text-xs text-white/35">{meta.detail}</span>}
          {meta.notional > 0 && (
            <div className="ml-auto flex items-center gap-1.5">
              <span className="text-[10px] text-white/30">Exposure</span>
              <span className="text-sm font-mono font-bold text-white">{fmt$(meta.notional)}</span>
            </div>
          )}
        </div>
      </div>

      {/* Without-Hedge Risk Panel */}
      {rec.unhedged_risk && (() => {
        const ur = rec.unhedged_risk;
        const topCost = best
          ? (best.market_total_cost > 0 ? best.market_total_cost : best.total_cost)
          : 0;
        // The VaR is scaled to the hedge horizon the user chose, so the label
        // is read from the response rather than asserted. It used to read
        // "1mo VaR (95%)" for what is by default a 180-day figure.
        const horizonDays = ur.horizon_calendar_days ?? 0;
        const varLabel = horizonDays > 0
          ? `${horizonDays}d VaR (95%)`
          : "VaR (95%)";
        const horizonSub = horizonDays > 0 ? ` over ${horizonDays}d` : "";
        const items = [
          { label: varLabel,          val: `-${fmt$(ur.var_95)}`,           sub: `${fmtPct(ur.var_pct)} of notional${horizonSub}` },
          { label: "Exp. Shortfall",  val: `-${fmt$(ur.cvar_95)}`,           sub: `tail loss beyond VaR${horizonSub}` },
          { label: "Stress  -10% SPY", val: `-${fmt$(ur.stress_loss_10pct)}`, sub: `β ${ur.beta.toFixed(2)} vs SPY` },
          { label: "Stress  -20% SPY", val: `-${fmt$(ur.stress_loss_20pct)}`, sub: "severe drawdown scenario" },
        ];
        return (
          <div className="px-5 py-3 border-b border-white/6 bg-red-950/20">
            <div className="flex items-center gap-2 mb-2.5">
              <span className="text-[10px] font-bold text-red-400/70 uppercase tracking-widest">Without hedge</span>
              {best && topCost > 0 && (
                <span className="ml-auto text-[10px] text-white/30">
                  Hedge costs <span className="text-emerald-400 font-mono font-semibold">{fmt$(topCost)}</span>
                  {" "}vs potential loss of{" "}
                  <span className="text-red-400 font-mono font-semibold">{fmt$(ur.var_95)}</span>
                </span>
              )}
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {items.map(({ label, val, sub }) => (
                <div key={label} className="rounded-lg bg-red-900/15 border border-red-500/10 px-3 py-2">
                  <p className="text-[10px] text-white/35 mb-0.5">{label}</p>
                  <p className="text-sm font-mono font-bold text-red-400">{val}</p>
                  <p className="text-[10px] text-white/25 mt-0.5">{sub}</p>
                </div>
              ))}
            </div>
            {rec.risk_provenance && <RiskBasisNote rp={rec.risk_provenance} />}
          </div>
        );
      })()}

      {/* Subheader */}
      <div className="px-5 py-3 border-b border-white/6 flex items-center gap-3 flex-wrap bg-white/[0.015]">
        <span className="text-xs text-white/30">
          {rec.candidates.length} candidate{rec.candidates.length !== 1 ? "s" : ""}
        </span>

        {best && (
          <>
            <span className={`text-xs font-bold px-2 py-0.5 rounded-full ring-1 ${scoreColor(best.score).ring}`}>
              Best score {best.score.toFixed(0)}
            </span>
            <span className="text-xs text-white/30 hidden sm:inline">
              {best.strategy}
              <span className="font-mono text-white/50 ml-1">{fmt$(best.total_cost)}</span>
              <span className={`ml-1 ${costStyle(costPct)}`}>({fmtPct(costPct)})</span>
            </span>

            {/* Per-position Greeks from top candidate */}
            <div className="ml-auto flex items-center gap-2">
              {(() => {
                const perDelta = best.n_contracts > 0 ? best.delta / (best.n_contracts * 100) : best.delta;
                return [
                  { k: "Δ", v: perDelta,   d: 3 },
                  { k: "Γ", v: best.gamma, d: 4 },
                  { k: "ν", v: best.vega,  d: 2 },
                ].map(({ k, v, d }) => (
                  <div key={k} className="flex items-center gap-1 px-2 py-0.5 rounded bg-white/4 border border-white/6">
                    <span className="text-[10px] text-white/25">{k}</span>
                    <span className={`text-[11px] font-mono font-semibold ${v < 0 ? "text-red-400" : "text-emerald-400"}`}>
                      {fmtDec(v, d)}
                    </span>
                  </div>
                ));
              })()}
            </div>
          </>
        )}
      </div>

      {/* Candidates */}
      <div className="p-3 space-y-2">
        {sorted.length === 0 ? (
          <div className="text-center py-10 text-white/25 text-sm">
            No hedge candidates found — try relaxing cost constraints or extending hedge horizon.
          </div>
        ) : sorted.map((c, i) => (
          <CandidateCard key={`${c.strategy}-${c.strike}-${i}`} c={c} notional={meta.notional} rank={i + 1} defaultOpen={i === 0} />
        ))}
      </div>
    </div>
  );
}

// ─── Disclosure panels ────────────────────────────────────────────────────────

/**
 * What the recommended hedge costs against what the user allowed.
 *
 * This is the one number that answers "can I afford this?", and until the
 * backend started summing the per-holding costs it was not in the response at
 * all — the UI showed a premium per candidate and left the addition to the
 * user. A net credit is shown as a credit rather than as negative spend,
 * because being paid to put a collar on is not 0% budget usage, it is income.
 */
function BudgetBar({ result }: { result: HedgeOutput }) {
  const { hedge_portfolio_cost: cost, hedge_budget_total: budget } = result;
  if (budget <= 0) return null;

  const credit = cost < 0;
  const pct    = Math.min(Math.abs(result.hedge_budget_utilization) * 100, 100);
  const overCap = result.hedge_budget_utilization > 1;

  return (
    <div className="border-t border-white/6 pt-4">
      <div className="flex items-baseline justify-between mb-2">
        <p className="text-[10px] font-semibold text-white/30 uppercase tracking-widest">
          Hedge cost vs budget
        </p>
        <p className="text-xs font-mono">
          <span className={credit ? "text-emerald-300" : overCap ? "text-red-300" : "text-white"}>
            {credit ? `+${fmt$(Math.abs(cost))} credit` : fmt$(cost)}
          </span>
          <span className="text-white/25"> / {fmt$(budget)}</span>
        </p>
      </div>

      <div className="h-1.5 rounded-full bg-white/6 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${
            credit ? "bg-emerald-500/70" : overCap ? "bg-red-500/70" : pct > 80 ? "bg-amber-500/70" : "bg-sky-500/70"
          }`}
          style={{ width: `${credit ? 100 : pct}%` }}
        />
      </div>

      <p className="text-[11px] text-white/30 mt-1.5">
        {credit
          ? "This set is a net credit — the short legs more than finance the long ones."
          : `${(result.hedge_budget_utilization * 100).toFixed(0)}% of the ${
              (budget / Math.max(result.portfolio_notional, 1) * 100).toFixed(1)
            }% cap you set. Premium is the mid-market estimate; fills may differ.`}
      </p>
    </div>
  );
}

/**
 * Market value, cost basis, and what the client claimed.
 *
 * The three are different numbers and were previously collapsed into one
 * "Notional" chip. The server values the portfolio itself from current
 * prices, so when the figure the client submitted disagrees the user should
 * see both rather than silently getting the server's.
 */
function ValuationNote({ result }: { result: HedgeOutput }) {
  const { portfolio_cost_basis: basis, portfolio_unrealized_pnl: pnl, declared_notional: declared } = result;
  if (!basis && !pnl) return null;

  const drift = declared != null && result.portfolio_notional > 0
    ? Math.abs(declared - result.portfolio_notional) / result.portfolio_notional
    : 0;

  return (
    <div className="border-t border-white/6 pt-4 space-y-2">
      <p className="text-[10px] font-semibold text-white/30 uppercase tracking-widest">Valuation</p>
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: "Market value", val: fmt$(result.portfolio_notional), cls: "text-white" },
          { label: "Cost basis",   val: fmt$(basis), cls: "text-white/60" },
          {
            label: "Unrealized",
            val: `${pnl >= 0 ? "+" : "−"}${fmt$(Math.abs(pnl))}`,
            cls: pnl >= 0 ? "text-emerald-300" : "text-red-300",
          },
        ].map(({ label, val, cls }) => (
          <div key={label}>
            <p className="text-[10px] text-white/30">{label}</p>
            <p className={`text-sm font-mono font-semibold ${cls}`}>{val}</p>
          </div>
        ))}
      </div>
      {drift > 0.01 && (
        <p className="text-[11px] text-amber-400/70 leading-relaxed">
          The value you submitted ({fmt$(declared!)}) differs by {(drift * 100).toFixed(1)}% from
          current market prices. Every figure here uses the market value.
        </p>
      )}
    </div>
  );
}

/**
 * The pipeline's own disclosures.
 *
 * Holdings left unhedged, candidates dropped for breaching the budget, stale
 * or substituted data, a degraded regime. These were being overwritten in the
 * orchestrator before they reached the response; they are the product's
 * honesty and belong at the top of the result, not in a collapsed footer.
 */
function WarningsPanel({ warnings }: { warnings: string[] }) {
  if (!warnings.length) return null;
  return (
    <div className="rounded-2xl border border-amber-400/20 bg-amber-400/6 p-4 space-y-2">
      <p className="text-[10px] font-semibold text-amber-300/70 uppercase tracking-widest">
        {warnings.length === 1 ? "Disclosure" : `${warnings.length} disclosures`}
      </p>
      <ul className="space-y-1.5">
        {warnings.map((w, i) => (
          <li key={i} className="text-xs text-amber-100/70 leading-relaxed flex gap-2">
            <span className="text-amber-400/50 shrink-0">•</span>
            <span>{w}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * Where the numbers came from.
 *
 * An empty ledger is reported as "not recorded", never as "all live" — the
 * absence of a provenance record is not evidence of freshness, and in a
 * product whose output is only as good as its market data, claiming the
 * stronger of the two would be the single most misleading thing on screen.
 */
function DataQualityPanel({ dq }: { dq: DataQuality }) {
  const counts   = dq.source_counts ?? {};
  const total    = Object.values(counts).reduce((a, b) => a + b, 0);
  const degraded = dq.degraded_fields ?? [];

  if (total === 0) {
    return (
      <div className="border-t border-white/6 pt-4">
        <p className="text-[10px] font-semibold text-white/30 uppercase tracking-widest mb-1.5">
          Data sources
        </p>
        <p className="text-[11px] text-white/35 leading-relaxed">
          No provenance was recorded for this run, so the freshness of these
          figures cannot be confirmed from the response.
        </p>
      </div>
    );
  }

  const tone: Record<string, string> = {
    live:        "bg-emerald-500/10 text-emerald-300/80 border-emerald-500/20",
    cache:       "bg-amber-400/10 text-amber-300/80 border-amber-400/20",
    fallback:    "bg-orange-500/10 text-orange-300/80 border-orange-500/20",
    unavailable: "bg-red-500/10 text-red-300/80 border-red-500/20",
  };
  const wording: Record<string, string> = {
    live: "live", cache: "cached", fallback: "substituted", unavailable: "unavailable",
  };

  return (
    <div className="border-t border-white/6 pt-4 space-y-2.5">
      <div className="flex items-baseline justify-between">
        <p className="text-[10px] font-semibold text-white/30 uppercase tracking-widest">Data sources</p>
        {dq.generated_at && (
          <p className="text-[10px] text-white/25 font-mono">
            snapshot {new Date(dq.generated_at).toLocaleTimeString()}
          </p>
        )}
      </div>

      <div className="flex flex-wrap gap-2">
        {Object.entries(counts)
          .sort(([, a], [, b]) => b - a)
          .map(([source, n]) => (
            <span key={source}
              className={`text-[11px] px-2.5 py-1 rounded-full border ${tone[source] ?? "bg-white/4 text-white/50 border-white/8"}`}>
              {n} {wording[source] ?? source}
            </span>
          ))}
      </div>

      {degraded.length > 0 && (
        <details className="group">
          <summary className="text-[11px] text-white/40 hover:text-white/60 cursor-pointer list-none">
            {degraded.length} value{degraded.length === 1 ? "" : "s"} were not live
            <span className="text-white/25 group-open:hidden"> — show</span>
          </summary>
          <ul className="mt-2 space-y-1">
            {degraded.map((d) => (
              <li key={d.name} className="text-[11px] text-white/40 font-mono leading-relaxed">
                {d.name}: {wording[d.source] ?? d.source}
                {d.age_seconds != null && ` (~${Math.round(d.age_seconds / 60)} min old)`}
                {d.detail && <span className="font-sans text-white/30"> — {d.detail}</span>}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

// ─── Portfolio summary ────────────────────────────────────────────────────────

function PortfolioSummary({ result }: { result: HedgeOutput }) {
  const rs = regimeStyle(result.regime, result.is_anomaly);
  const regimeLabel = result.is_anomaly ? "ANOMALY" : result.regime.replace("_", " ").toUpperCase();

  return (
    <div className="rounded-2xl border border-white/8 bg-[#0d0e13] p-5 space-y-4">

      {/* Top stats row */}
      <div className="flex flex-wrap items-center gap-3">
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold ${rs.cls}`}>
          <span className={`w-1.5 h-1.5 rounded-full ${rs.dot} animate-pulse`} />
          {regimeLabel}
        </div>

        {/* A regime inferred from a single VIX reading is not the same claim
            as one fitted by the GMM, and it changes the regime-fit component
            of every score below. The label alone read identically either
            way. */}
        {result.regime_degraded && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-amber-400/10 border border-amber-400/20"
               title={result.regime_degradation_reason}>
            <span className="text-[10px] text-amber-300/70">regime</span>
            <span className="text-xs font-semibold text-amber-200/90">estimated</span>
          </div>
        )}

        {[
          { label: "VIX",       val: result.vix_level.toFixed(1) },
          { label: "Value",     val: fmt$(result.portfolio_notional) },
          { label: "Holdings",  val: String(result.recommendations.length) },
          { label: "Computed",  val: result.run_time_seconds.toFixed(2) + "s" },
        ].map(({ label, val }) => (
          <div key={label} className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/4 border border-white/6">
            <span className="text-[10px] text-white/35">{label}</span>
            <span className="text-xs font-mono font-semibold text-white">{val}</span>
          </div>
        ))}
      </div>

      {result.regime_degraded && result.regime_degradation_reason && (
        <p className="text-[11px] text-amber-400/70 leading-relaxed">
          Regime detection fell back to {result.regime_source.replace(/_/g, " ")}:{" "}
          {result.regime_degradation_reason} Regime-dependent scores are less reliable than usual.
        </p>
      )}

      {/* LLM rate-limit notice */}
      {result.llm_notice && (
        <div className="border-t border-white/6 pt-4">
          <p className="text-xs text-amber-400/80 bg-amber-400/8 border border-amber-400/20 rounded-lg px-3 py-2">
            ⚠ {result.llm_notice}
          </p>
        </div>
      )}

      {/* LLM summary */}
      {result.portfolio_summary && (
        <p className="text-sm text-white/60 leading-relaxed border-t border-white/6 pt-4">
          {result.portfolio_summary}
        </p>
      )}

      {/* Key risks */}
      {result.key_risks.length > 0 && (
        <div className="border-t border-white/6 pt-4">
          <p className="text-[10px] font-semibold text-white/30 uppercase tracking-widest mb-2.5">Key Risks</p>
          <div className="flex flex-wrap gap-2">
            {result.key_risks.map((r, i) => (
              <span key={i} className="text-xs bg-red-500/8 text-red-300/80 border border-red-500/15 px-2.5 py-1 rounded-full">
                {r}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Top recommendation */}
      {result.top_recommendation && (
        <div className="border-t border-white/6 pt-4 flex gap-3">
          <span className="text-emerald-400 text-lg shrink-0">→</span>
          <p className="text-sm text-white/70 leading-relaxed">{result.top_recommendation}</p>
        </div>
      )}

      <ValuationNote result={result} />
      <BudgetBar result={result} />
      <DataQualityPanel dq={result.data_quality ?? {}} />

    </div>
  );
}

// ─── Spinner ──────────────────────────────────────────────────────────────────

function Spinner() {
  return (
    <div className="w-5 h-5 border-2 border-sky-500 border-t-transparent rounded-full animate-spin" />
  );
}

// ─── Error panel ──────────────────────────────────────────────────────────────

/**
 * What went wrong, and what the user can do about it.
 *
 * Every branch here says something different because the causes call for
 * different actions, and because two of them are reassurances the user needs:
 * a timeout and a server error both leave the question "did it do something?"
 * open, and in a hedging product the honest answer — nothing was ordered,
 * this is analysis only — belongs on screen rather than in the user's head.
 */
function ErrorPanel({ error, onRetry }: { error: UiError; onRetry: () => void }) {
  const fieldMessages = Object.values(error.fields ?? {});
  const serverFields  = Object.entries(error.serverFields ?? {});

  const title =
    error.kind === "timeout"     ? "Analysis timed out"
    : error.kind === "offline"   ? "Can't reach the analysis server"
    : error.kind === "validation"? "Check these positions"
    : error.kind === "auth"      ? "Not authorized"
    : error.kind === "unavailable" ? "A data source is unavailable"
    : "Analysis failed";

  const retryable = error.kind === "timeout" || error.kind === "offline" || error.kind === "server";

  // The reference id the backend puts in its sanitised message is the only
  // handle support has on the traceback, so it is pulled out and made
  // selectable instead of being buried mid-sentence.
  const ref = /Reference:\s*([0-9a-f]{6,})/i.exec(error.message)?.[1];
  const body = ref ? error.message.replace(/\s*Reference:\s*[0-9a-f]{6,}\.?/i, "") : error.message;

  return (
    <div className="p-3.5 bg-red-500/8 border border-red-500/15 rounded-2xl space-y-2.5">
      <p className="text-red-200/90 text-xs font-semibold">{title}</p>

      {fieldMessages.length > 1 ? (
        <ul className="text-red-300/75 text-xs leading-relaxed space-y-1 list-disc list-inside">
          {fieldMessages.map((m, i) => <li key={i}>{m}</li>)}
        </ul>
      ) : (
        <p className="text-red-300/75 text-xs leading-relaxed">{body}</p>
      )}

      {serverFields.length > 0 && (
        <ul className="text-red-300/75 text-xs leading-relaxed space-y-1">
          {serverFields.map(([path, msg]) => (
            <li key={path}><span className="font-mono text-red-200/70">{path}</span> — {msg}</li>
          ))}
        </ul>
      )}

      {(error.kind === "timeout" || error.kind === "server") && (
        <p className="text-white/35 text-[11px] leading-relaxed">
          Nothing was ordered or executed — OptionQ only analyzes and never places trades.
        </p>
      )}

      {ref && (
        <p className="text-white/30 text-[11px]">
          Reference <span className="font-mono text-white/50 select-all">{ref}</span> — quote this if you report the problem.
        </p>
      )}

      {retryable && (
        <button onClick={onRetry}
          className="text-xs font-semibold text-red-200/80 hover:text-red-100 underline underline-offset-2">
          Try again
        </button>
      )}
    </div>
  );
}

// ─── Form types ───────────────────────────────────────────────────────────────

interface StockFormRow {
  id: number; position_type: "stock";
  ticker: string; shares: string; purchase_price: string; purchase_date: string; asset_class: AssetClass;
}
interface OptionFormRow {
  id: number; position_type: "option";
  ticker: string; option_type: "call" | "put"; strike: string; expiry: string;
  contracts: string; direction: "long" | "short"; premium_paid: string;
}
type PositionFormRow = StockFormRow | OptionFormRow;

/**
 * A failure the user can read.
 *
 * `kind` comes from the API client and decides what the panel offers: a
 * timeout gets a retry, an unreachable backend gets the URL to check, a
 * validation failure gets the field list. The old handler flattened every
 * cause to one string, so "the server is not running" and "your strike is
 * negative" looked identical and neither suggested a next step.
 */
interface UiError {
  kind: string;
  message: string;
  /** Client-side, keyed by position row id. */
  fields?: Record<number, string>;
  /** Server-side (a 422), keyed by the backend's field path. */
  serverFields?: Record<string, string>;
}

// ─── Toggle ───────────────────────────────────────────────────────────────────

function Toggle<T extends string>({
  options, value, onChange, colorMap,
}: { options: readonly T[]; value: T; onChange: (v: T) => void; colorMap?: Partial<Record<T, string>> }) {
  return (
    <div className="flex rounded-lg overflow-hidden border border-white/8 bg-white/3">
      {options.map((opt) => {
        const active = opt === value;
        const activeCls = colorMap?.[opt] ?? "bg-sky-600/50 text-sky-200";
        return (
          <button key={opt} onClick={() => onChange(opt)}
            className={`flex-1 px-2 py-1.5 text-xs font-medium capitalize transition-all ${active ? activeCls : "text-white/35 hover:text-white/60"}`}>
            {opt}
          </button>
        );
      })}
    </div>
  );
}

// ─── Position card ────────────────────────────────────────────────────────────

function PositionCard({ pos, onReplace, onRemove }: {
  pos: PositionFormRow; onReplace: (n: PositionFormRow) => void; onRemove: () => void;
}) {
  const isOpt = pos.position_type === "option";
  const s = pos as StockFormRow;
  const o = pos as OptionFormRow;

  const switchToStock = () => !isOpt ? undefined : onReplace({
    id: pos.id, position_type: "stock", ticker: pos.ticker,
    shares: "", purchase_price: "", purchase_date: new Date().toISOString().split("T")[0], asset_class: "equity",
  });
  const switchToOption = () => isOpt ? undefined : onReplace({
    id: pos.id, position_type: "option", ticker: pos.ticker,
    option_type: "call", strike: "", expiry: "", contracts: "1", direction: "long", premium_paid: "",
  });

  const inputCls = "w-full bg-[#111318] border border-white/8 rounded-lg px-3 py-2 text-sm text-white placeholder-white/20 focus:outline-none focus:border-sky-500/50 transition-all";
  const purpleInputCls = "w-full bg-[#111318] border border-white/8 rounded-lg px-3 py-2 text-sm text-white placeholder-white/20 focus:outline-none focus:border-violet-500/50 transition-all";

  const notional = isOpt
    ? (parseInt(o.contracts, 10) || 0) * 100 * (parseFloat(o.strike) || 0)
    : (parseFloat(s.shares) || 0) * (parseFloat(s.purchase_price) || 0);

  return (
    <div className={`rounded-xl border p-3.5 space-y-3 ${isOpt ? "border-violet-500/15 bg-violet-500/[0.03]" : "border-white/6 bg-white/[0.02]"}`}>
      <div className="flex items-center gap-2.5">
        {/* Type toggle */}
        <div className="flex rounded-lg overflow-hidden border border-white/8 text-[11px] font-semibold shrink-0">
          <button onClick={switchToStock}
            className={`px-2.5 py-1.5 transition-all ${!isOpt ? "bg-sky-600 text-white" : "text-white/30 hover:text-white/60"}`}>
            Stock
          </button>
          <button onClick={switchToOption}
            className={`px-2.5 py-1.5 transition-all ${isOpt ? "bg-violet-600 text-white" : "text-white/30 hover:text-white/60"}`}>
            Option
          </button>
        </div>

        <div className="flex-1 min-w-0">
          <TickerSelect value={pos.ticker} onChange={(t) => onReplace({ ...pos, ticker: t })} />
        </div>

        {notional > 0 && (
          <span className="text-xs font-mono text-white/30 shrink-0">{fmt$(notional)}</span>
        )}

        <button onClick={onRemove}
          className="text-white/20 hover:text-red-400 w-6 h-6 flex items-center justify-center rounded-lg hover:bg-red-500/10 transition-all shrink-0 text-lg leading-none">
          ×
        </button>
      </div>

      {!isOpt && (
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-[10px] text-white/30 block mb-1 font-medium">Shares</label>
            <input type="number" placeholder="100" value={s.shares}
              onChange={(e) => onReplace({ ...s, shares: e.target.value })} className={inputCls} />
          </div>
          <div>
            <label className="text-[10px] text-white/30 block mb-1 font-medium">Avg price ($)</label>
            <input type="number" step="0.01" placeholder="450.00" value={s.purchase_price}
              onChange={(e) => onReplace({ ...s, purchase_price: e.target.value })} className={inputCls} />
          </div>
        </div>
      )}

      {isOpt && (
        <div className="space-y-2.5">
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-[10px] text-white/30 block mb-1 font-medium">Type</label>
              <Toggle options={["call", "put"] as const} value={o.option_type}
                onChange={(v) => onReplace({ ...o, option_type: v })}
                colorMap={{ call: "bg-emerald-600/40 text-emerald-300", put: "bg-red-600/40 text-red-300" }} />
            </div>
            <div>
              <label className="text-[10px] text-white/30 block mb-1 font-medium">Direction</label>
              <Toggle options={["long", "short"] as const} value={o.direction}
                onChange={(v) => onReplace({ ...o, direction: v })}
                colorMap={{ long: "bg-sky-700/50 text-sky-300", short: "bg-orange-700/40 text-orange-300" }} />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-[10px] text-white/30 block mb-1 font-medium">Strike ($)</label>
              <input type="number" step="0.5" placeholder="220.00" value={o.strike}
                onChange={(e) => onReplace({ ...o, strike: e.target.value })} className={purpleInputCls} />
            </div>
            <div>
              <label className="text-[10px] text-white/30 block mb-1 font-medium">Expiry</label>
              <input type="date" value={o.expiry}
                min={new Date().toISOString().slice(0, 10)}
                onChange={(e) => onReplace({ ...o, expiry: e.target.value })} className={purpleInputCls} />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-[10px] text-white/30 block mb-1 font-medium">Contracts</label>
              <input type="number" min="1" step="1" placeholder="1" value={o.contracts}
                onChange={(e) => onReplace({ ...o, contracts: e.target.value })} className={purpleInputCls} />
            </div>
            <div>
              <label className="text-[10px] text-white/30 block mb-1 font-medium">Premium paid ($)</label>
              <input type="number" step="0.01" placeholder="optional" value={o.premium_paid}
                onChange={(e) => onReplace({ ...o, premium_paid: e.target.value })} className={purpleInputCls} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

export default function Home() {
  const [positions, setPositions] = useState<PositionFormRow[]>([
    { id: 1, position_type: "stock", ticker: "AAPL", shares: "200", purchase_price: "210.00", purchase_date: "2024-01-15", asset_class: "equity" },
    { id: 2, position_type: "stock", ticker: "SPY",  shares: "150", purchase_price: "500.00", purchase_date: "2024-01-15", asset_class: "equity" },
    { id: 3, position_type: "stock", ticker: "MSFT", shares: "100", purchase_price: "390.00", purchase_date: "2024-01-15", asset_class: "equity" },
  ]);

  const totalNotional = useMemo(() =>
    positions.reduce((sum, p) => {
      if (p.position_type === "stock") return sum + (parseFloat(p.shares) || 0) * (parseFloat(p.purchase_price) || 0);
      return sum + (parseInt(p.contracts, 10) || 0) * 100 * (parseFloat(p.strike) || 0);
    }, 0), [positions]);

  const [hedgeHorizon,    setHedgeHorizon]    = useState("90");
  const [maxCostPct,      setMaxCostPct]      = useState("5");
  const [protectionLevel, setProtectionLevel] = useState("15");
  const [upsidePreserve,  setUpsidePreserve]  = useState(true);

  const [result,  setResult]  = useState<HedgeOutput | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<UiError | null>(null);

  // The exact inputs the displayed result was computed from. Every figure on
  // screen — VaR, premium, contract count, budget usage — belongs to one
  // specific portfolio, and the form it was typed into keeps changing
  // underneath it. Without this, editing 500 shares to 5,000 and not
  // re-running left a VaR for 500 shares sitting beside a form that said
  // 5,000, with nothing on screen to say which one the number described.
  const [analyzedInputs, setAnalyzedInputs] = useState<string | null>(null);

  // Lets a new run, or an unmount, cancel the one in flight. Without it a
  // superseded request still resolved and overwrote the newer result.
  const inFlight = useRef<AbortController | null>(null);
  useEffect(() => () => inFlight.current?.abort(), []);

  const nextId = useCallback(() => Math.max(0, ...positions.map((p) => p.id)) + 1, [positions]);

  const addStock = () => setPositions((prev) => [...prev, {
    id: nextId(), position_type: "stock", ticker: "", shares: "", purchase_price: "",
    purchase_date: new Date().toISOString().split("T")[0], asset_class: "equity",
  }]);
  const addOption = () => setPositions((prev) => [...prev, {
    id: nextId(), position_type: "option", ticker: "", option_type: "call",
    strike: "", expiry: "", contracts: "1", direction: "long", premium_paid: "",
  }]);
  const replacePosition = (n: PositionFormRow) => setPositions((prev) => prev.map((p) => (p.id === n.id ? n : p)));
  const removePosition  = (id: number) => setPositions((prev) => prev.filter((p) => p.id !== id));

  const optionCount = positions.filter((p) => p.position_type === "option").length;
  const stockCount  = positions.filter((p) => p.position_type === "stock").length;

  const buildPayload = () => {
    const valid = positions.filter((p) =>
      p.position_type === "stock" ? p.ticker && p.shares && p.purchase_price : p.ticker && p.strike && p.expiry && p.contracts
    );
    return {
      holdings: valid.map((p) => p.position_type === "stock"
        ? { position_type: "stock" as const, ticker: p.ticker.toUpperCase(), shares: parseFloat(p.shares),
            purchase_price: parseFloat(p.purchase_price), purchase_date: p.purchase_date, asset_class: p.asset_class }
        : { position_type: "option" as const, ticker: p.ticker.toUpperCase(), option_type: p.option_type,
            strike: parseFloat(p.strike), expiry: p.expiry, contracts: parseInt(p.contracts, 10),
            direction: p.direction, ...(p.premium_paid ? { premium_paid: parseFloat(p.premium_paid) } : {}),
            asset_class: "equity" as const }),
      total_notional: totalNotional,
      hedge_horizon_days: parseInt(hedgeHorizon, 10),
      max_hedge_cost_pct: parseFloat(maxCostPct) / 100,
      protection_level:   parseFloat(protectionLevel) / 100,
      upside_preservation: upsidePreserve,
      execution_mode: "analyze" as const,
    };
  };

  // Pure function of form state, so it is safe to serialise during render.
  const currentInputs = useMemo(
    () => JSON.stringify(buildPayload()),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [positions, totalNotional, hedgeHorizon, maxCostPct, protectionLevel, upsidePreserve],
  );
  const resultIsStale = result !== null && analyzedInputs !== null
                        && analyzedInputs !== currentInputs;

  /**
   * Check the form before anything is sent.
   *
   * The backend validates too — Pydantic rejects a non-positive quantity and
   * the pipeline refuses an empty portfolio — and these checks deliberately
   * mirror those rules rather than replacing them. What they add is the
   * timing and the wording: learning that a ticker box is blank should not
   * cost a multi-second round trip, and "holdings.2.shares" is not a field
   * the user can see. Anything that gets past here is still the server's call.
   *
   * `buildPayload` silently drops incomplete rows, so without this a
   * half-filled position was analysed as though it had never been entered —
   * the user got a hedge for two holdings after typing three.
   */
  const validate = (): UiError | null => {
    if (positions.length === 0) {
      return { kind: "validation", message: "Add at least one position to analyze." };
    }

    const today  = new Date().toISOString().split("T")[0];
    const fields: Record<number, string> = {};

    for (const pos of positions) {
      const label = pos.ticker.trim()
        ? pos.ticker.trim().toUpperCase()
        : "This position";

      if (!pos.ticker.trim()) {
        fields[pos.id] = "Enter a ticker.";
      } else if (pos.position_type === "stock") {
        if (!(parseFloat(pos.shares) > 0))
          fields[pos.id] = `${label}: share count must be greater than zero.`;
        else if (!(parseFloat(pos.purchase_price) > 0))
          fields[pos.id] = `${label}: purchase price must be greater than zero.`;
      } else {
        if (!(parseFloat(pos.strike) > 0))
          fields[pos.id] = `${label}: strike must be greater than zero.`;
        else if (!pos.expiry)
          fields[pos.id] = `${label}: choose an expiry date.`;
        else if (pos.expiry < today)
          fields[pos.id] =
            `${label}: expiry ${pos.expiry} has passed — an expired contract ` +
            `carries no exposure to hedge.`;
        else if (!(parseInt(pos.contracts, 10) > 0))
          fields[pos.id] = `${label}: contract count must be at least 1.`;
      }
    }

    const messages = Object.values(fields);
    if (messages.length > 0) {
      return {
        kind: "validation",
        message: messages.length === 1
          ? messages[0]
          : `${messages.length} positions need attention before the portfolio can be analyzed.`,
        fields,
      };
    }

    if (!(totalNotional > 0)) {
      return {
        kind: "validation",
        message: "The portfolio works out to $0 — check the quantities and prices.",
      };
    }
    return null;
  };

  const analyze = async () => {
    const invalid = validate();
    if (invalid) {
      setError(invalid);
      // The form can no longer be turned into a valid portfolio, so whatever
      // is on screen describes inputs the user has since broken. Leaving it
      // rendered under the validation message paired real figures with a
      // portfolio nobody can analyse.
      setResult(null);
      setAnalyzedInputs(null);
      return;
    }

    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;

    setLoading(true); setError(null); setResult(null); setAnalyzedInputs(null);
    try {
      const payload = buildPayload();
      const sent = JSON.stringify(payload);
      setResult(await api.analyzePortfolio(payload, controller.signal));
      // Set only after the result lands, and from the payload actually sent,
      // so a form edited while the request was in flight reads as stale
      // rather than as current.
      setAnalyzedInputs(sent);
    } catch (e: unknown) {
      // A run the user cancelled, or one superseded by a newer one, is not a
      // failure and must not paint an error over the next result.
      if (controller.signal.aborted) return;
      setError(
        e instanceof ApiError
          ? { kind: e.kind, message: e.message, serverFields: e.fieldErrors }
          : { kind: "server", message: e instanceof Error ? e.message : "Analysis failed." },
      );
    } finally {
      if (inFlight.current === controller) {
        inFlight.current = null;
        setLoading(false);
      }
    }
  };

  const cancelAnalysis = () => {
    inFlight.current?.abort();
    inFlight.current = null;
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-[#090a0d] text-white">

      {/* ── Header ── */}
      <header className="sticky top-0 z-20 border-b border-white/6 bg-[#090a0d]/90 backdrop-blur-xl px-6 py-3.5 flex items-center gap-4">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-sky-400 to-violet-500 flex items-center justify-center text-xs font-black">Q</div>
          <div>
            <span className="text-sm font-bold tracking-tight text-white">OptionQ</span>
            <span className="text-white/25 text-xs ml-2 hidden sm:inline">Hedge Engine</span>
          </div>
        </div>

        {result && (
          <div className="ml-auto flex items-center gap-2">
            <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold ${regimeStyle(result.regime, result.is_anomaly).cls}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${regimeStyle(result.regime, result.is_anomaly).dot}`} />
              {result.is_anomaly ? "ANOMALY" : result.regime.replace("_", " ").toUpperCase()}
            </div>
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-white/5 text-xs">
              <span className="text-white/30">VIX</span>
              <span className="font-mono font-bold text-white">{result.vix_level.toFixed(1)}</span>
            </div>
            <span className="text-white/20 text-xs font-mono hidden md:inline">{result.run_time_seconds.toFixed(2)}s</span>
          </div>
        )}
      </header>

      {/* ── Body ── */}
      <div className="max-w-screen-2xl mx-auto px-4 py-5 grid grid-cols-1 lg:grid-cols-[340px_1fr] gap-4">

        {/* ── Left sidebar ── */}
        <aside className="space-y-3 lg:sticky lg:top-[57px] lg:self-start lg:max-h-[calc(100vh-57px)] lg:overflow-y-auto lg:pb-4">

          {/* Positions */}
          <div className="rounded-2xl border border-white/6 bg-[#0d0e13] p-4">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-[10px] font-bold text-white/40 uppercase tracking-widest">Positions</h2>
              <div className="flex items-center gap-1.5">
                {stockCount > 0  && <span className="text-[10px] bg-sky-500/15 text-sky-400 px-1.5 py-0.5 rounded font-semibold">{stockCount}S</span>}
                {optionCount > 0 && <span className="text-[10px] bg-violet-500/15 text-violet-400 px-1.5 py-0.5 rounded font-semibold">{optionCount}O</span>}
              </div>
            </div>

            <div className="space-y-2">
              {positions.map((p) => (
                <PositionCard key={p.id} pos={p} onReplace={replacePosition} onRemove={() => removePosition(p.id)} />
              ))}
            </div>

            <div className="mt-3 grid grid-cols-2 gap-2">
              <button onClick={addStock}
                className="text-xs text-sky-400/70 hover:text-sky-300 border border-dashed border-white/8 hover:border-sky-500/30 rounded-xl py-2.5 transition-all hover:bg-sky-500/5">
                + Stock
              </button>
              <button onClick={addOption}
                className="text-xs text-violet-400/70 hover:text-violet-300 border border-dashed border-white/8 hover:border-violet-500/30 rounded-xl py-2.5 transition-all hover:bg-violet-500/5">
                + Option
              </button>
            </div>
          </div>

          {/* Constraints */}
          <div className="rounded-2xl border border-white/6 bg-[#0d0e13] p-4 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-[10px] font-bold text-white/40 uppercase tracking-widest">Constraints</h2>
              {totalNotional > 0 && (
                <span className="text-xs font-mono font-bold text-white/60">{fmt$(totalNotional)}</span>
              )}
            </div>

            {/* Sliders */}
            {[
              { label: "Hedge horizon", unit: "d",  value: hedgeHorizon, set: setHedgeHorizon, min: 30, max: 365, step: 30, lo: "30d",  hi: "1yr" },
              { label: "Max cost",      unit: "%",  value: maxCostPct,   set: setMaxCostPct,   min: 1,  max: 15,  step: 0.5, lo: "1%",   hi: "15%" },
            ].map(({ label, unit, value, set, min, max, step, lo, hi }) => (
              <label key={label} className="flex flex-col gap-1.5 cursor-pointer">
                <div className="flex justify-between">
                  <span className="text-xs text-white/45">{label}</span>
                  <span className="text-xs font-mono font-bold text-white/70">{value}{unit}</span>
                </div>
                <input type="range" min={min} max={max} step={step} value={value}
                  onChange={(e) => set(e.target.value)} className="accent-sky-500 w-full h-1" />
                <div className="flex justify-between text-[10px] text-white/20">
                  <span>{lo}</span><span>{hi}</span>
                </div>
              </label>
            ))}

            {stockCount > 0 && (
              <>
                <div className="border-t border-white/5 pt-3">
                  <p className="text-[10px] font-bold text-sky-400/50 uppercase tracking-widest mb-3">Stock hedging</p>

                  <label className="flex flex-col gap-1.5 cursor-pointer mb-3">
                    <div className="flex justify-between">
                      <span className="text-xs text-white/45">Protection level</span>
                      <span className="text-xs font-mono font-bold text-white/70">{protectionLevel}%</span>
                    </div>
                    <input type="range" min={5} max={30} step={1} value={protectionLevel}
                      onChange={(e) => setProtectionLevel(e.target.value)} className="accent-sky-500 w-full h-1" />
                    <div className="flex justify-between text-[10px] text-white/20"><span>5%</span><span>30%</span></div>
                  </label>

                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs text-white/45">Upside preservation</p>
                      <p className="text-[10px] text-white/25 mt-0.5">{upsidePreserve ? "OTM puts (keep upside)" : "Collars (cap upside, reduce cost)"}</p>
                    </div>
                    <button onClick={() => setUpsidePreserve((v) => !v)}
                      className={`relative w-9 h-5 rounded-full transition-colors shrink-0 ${upsidePreserve ? "bg-sky-600" : "bg-white/10"}`}>
                      <span className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${upsidePreserve ? "translate-x-4" : ""}`} />
                    </button>
                  </div>
                </div>
              </>
            )}

          </div>

          {/* Analyze button */}
          <button onClick={analyze} disabled={loading}
            className="w-full py-3.5 bg-gradient-to-r from-sky-600 to-violet-600 hover:from-sky-500 hover:to-violet-500 active:from-sky-700 active:to-violet-700 disabled:from-white/5 disabled:to-white/5 disabled:text-white/20 text-white font-bold rounded-2xl transition-all flex items-center justify-center gap-2.5 text-sm shadow-lg shadow-sky-900/30">
            {loading ? <><Spinner /><span>Analyzing…</span></> : "Analyze Portfolio"}
          </button>

          {/* A full run can take a minute or more. Leaving the user no way out
              but a page reload — which loses the whole portfolio they typed —
              is not an acceptable wait. */}
          {loading && (
            <button onClick={cancelAnalysis}
              className="w-full py-2 text-xs font-medium text-white/40 hover:text-white/70 transition-colors">
              Cancel
            </button>
          )}

          {error && <ErrorPanel error={error} onRetry={analyze} />}
        </aside>

        {/* ── Main results ── */}
        <main className="min-h-[60vh] space-y-4">

          {/* Empty state */}
          {!result && !loading && !error && (
            <div className="flex flex-col items-center justify-center h-96 gap-5 text-center">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-sky-500/20 to-violet-500/20 border border-white/8 flex items-center justify-center text-2xl">
                ⚡
              </div>
              <div>
                <p className="text-white/70 font-semibold text-lg">Ready to analyze</p>
                <p className="text-white/30 text-sm mt-1.5">Add positions on the left, then click Analyze Portfolio</p>
              </div>
              <div className="flex flex-wrap gap-2 justify-center">
                {["AAPL", "SPY", "MSFT", "NVDA", "QQQ", "GLD"].map((t) => (
                  <span key={t} className="text-xs font-mono bg-white/4 text-white/35 border border-white/6 px-3 py-1.5 rounded-full">{t}</span>
                ))}
                <span className="text-xs text-white/20 px-1 py-1.5">+ 39 more</span>
              </div>
            </div>
          )}

          {/* Loading */}
          {loading && (
            <div className="flex flex-col items-center justify-center h-96 gap-5">
              <div className="relative">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-sky-500/20 to-violet-500/20 border border-white/8 flex items-center justify-center">
                  <Spinner />
                </div>
              </div>
              <div className="text-center">
                <p className="text-white/70 font-semibold">Running analysis…</p>
                <p className="text-white/30 text-sm mt-1">Regime → Risk → Greeks → Scoring → LLM</p>
              </div>
            </div>
          )}

          {/* Results */}
          {result && !loading && (
            <>
              {/* Same reasoning as the warnings panel, one step stronger: if
                  the figures below no longer describe the portfolio in the
                  form, that has to be the first thing read, because it
                  invalidates all of them at once. */}
              {resultIsStale && (
                <div data-testid="stale-results"
                     className="rounded-xl border border-amber-400/30 bg-amber-400/8 px-4 py-3">
                  <p className="text-sm text-amber-200/90 font-semibold">
                    ⚠ These results describe the portfolio as it was when you ran the analysis
                  </p>
                  <p className="text-xs text-amber-200/60 mt-1 leading-relaxed">
                    The form has changed since. Every figure below — risk, premiums,
                    contract counts and budget usage — belongs to the earlier inputs.
                    Re-run the analysis to price the portfolio you have now.
                  </p>
                  <button onClick={analyze}
                    className="mt-2.5 text-xs font-semibold text-amber-200 hover:text-amber-100 underline underline-offset-2">
                    Re-run analysis
                  </button>
                </div>
              )}

              {/* Above the summary deliberately: a holding left unhedged or a
                  regime that had to be guessed changes how everything below
                  should be read, so it cannot sit underneath it. */}
              <WarningsPanel warnings={result.warnings ?? []} />
              <PortfolioSummary result={result} />

              {result.recommendations.length > 0 ? (
                result.recommendations.map((rec) => {
                  const parsed = parseAssetTicker(rec.asset_ticker);
                  let meta: PositionMeta;

                  if (parsed.isOption) {
                    const optPos = positions.find((p) =>
                      p.position_type === "option" &&
                      p.ticker.toUpperCase() === parsed.underlying &&
                      p.option_type === parsed.optType &&
                      Math.round(parseFloat(p.strike)) === Math.round(parsed.strike)
                    ) as OptionFormRow | undefined;
                    const contracts = optPos ? parseInt(optPos.contracts, 10) : 1;
                    meta = {
                      isOption: true,
                      label:  `${parsed.underlying} ${parsed.optType === "call" ? "Call" : "Put"} $${parsed.strike}`,
                      detail: optPos ? `${optPos.direction} · ${contracts}× · exp ${optPos.expiry}` : `${contracts}×`,
                      notional: contracts * 100 * parsed.strike,
                    };
                  } else {
                    const stockPos = positions.find((p) => p.position_type === "stock" && p.ticker.toUpperCase() === parsed.ticker) as StockFormRow | undefined;
                    const shares  = stockPos ? parseFloat(stockPos.shares) : 0;
                    const price   = stockPos ? parseFloat(stockPos.purchase_price) : 0;
                    meta = {
                      isOption: false,
                      label:  parsed.ticker,
                      detail: stockPos ? `${shares.toLocaleString()} shares · avg ${fmt$(price)}` : "",
                      notional: shares * price || result.portfolio_notional,
                    };
                  }

                  return <HoldingCard key={rec.asset_ticker} rec={rec} meta={meta} />;
                })
              ) : (
                <div className="rounded-2xl border border-white/6 bg-[#0d0e13] p-8 text-center">
                  <p className="text-white/40 text-sm">{result.portfolio_summary}</p>
                </div>
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}
