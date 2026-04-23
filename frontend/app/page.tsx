"use client";

import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { api } from "@/lib/api";
import type {
  HedgeOutput,
  AssetClass,
  InstrumentCandidate,
  HedgeRecommendation,
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
              <p className="text-sm font-mono font-semibold text-white">{fmt$(c.total_cost)}</p>
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
              { label: "Basis Risk R²", value: c.basis_risk_r2.toFixed(2),
                sub: c.basis_risk_r2 >= 0.7 ? "Low basis risk" : c.basis_risk_r2 >= 0.5 ? "Moderate" : "High basis risk" },
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

          {/* Premium breakdown */}
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

        {[
          { label: "VIX",       val: result.vix_level.toFixed(1) },
          { label: "Notional",  val: fmt$(result.portfolio_notional) },
          { label: "Holdings",  val: String(result.recommendations.length) },
          { label: "Computed",  val: result.run_time_seconds.toFixed(2) + "s" },
        ].map(({ label, val }) => (
          <div key={label} className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/4 border border-white/6">
            <span className="text-[10px] text-white/35">{label}</span>
            <span className="text-xs font-mono font-semibold text-white">{val}</span>
          </div>
        ))}
      </div>

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

    </div>
  );
}

// ─── Spinner ──────────────────────────────────────────────────────────────────

function Spinner() {
  return (
    <div className="w-5 h-5 border-2 border-sky-500 border-t-transparent rounded-full animate-spin" />
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
  const [error,   setError]   = useState("");

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

  const analyze = async () => {
    setLoading(true); setError(""); setResult(null);
    try   { setResult(await api.analyzePortfolio(buildPayload())); }
    catch (e: unknown) { setError(e instanceof Error ? e.message : "Analysis failed"); }
    finally { setLoading(false); }
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

          {error && (
            <div className="text-red-300/80 text-xs p-3.5 bg-red-500/8 border border-red-500/15 rounded-2xl leading-relaxed">
              {error}
            </div>
          )}
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
