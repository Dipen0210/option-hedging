# OptionQ — Product Requirements Document

**Status:** v1 (repair release) — internal, pre-external-testing
**Last updated:** 2026-10-29
**Owner:** Dipen Prajapati
**Repository:** `optionQ` (branch `main`, uncommitted working tree at the time of writing)

> **How to read this document.** Section 2 is history: what was built first,
> what was decided, and what the decisions turned out to cost. Section 3 is the
> product as it stands. Sections 4–8 are the requirements the current build is
> held to. Every claim about behaviour is tagged with how it was established:
> **[tested]** a regression test pins it, **[verified]** observed in a real
> end-to-end run, **[code]** read in the source but not exercised, and
> **[not verified]** asserted by neither. There are no unmarked claims about
> what works.

---

## 1. Product summary

### 1.1 What OptionQ is

OptionQ is a **portfolio hedging decision-support tool**. A user enters the
positions they hold, says how long they want protection for, how much downside
they want covered, and how much they are willing to pay. OptionQ measures the
portfolio's risk from market data, finds listed option contracts that would
reduce that risk, prices and sizes them, ranks them, and explains the
trade-offs in prose.

It is **not** an execution venue, a broker, or an advice service. It recommends;
the user decides and trades elsewhere.

### 1.2 What problem it solves

A retail or small-institutional investor who wants to hedge faces three
separate hard problems at once:

1. **Measurement** — how much is actually at risk, over what horizon, and from
   which factors? Portfolio VaR is not the sum of position VaRs; beta to a
   benchmark is an estimate with a standard error, not a number.
2. **Instrument selection** — the right hedge for a holding is often not an
   option on that holding. Many positions have no liquid options at all, and a
   proxy hedge trades basis risk for liquidity. Quantifying that trade-off
   requires a correlation estimate the investor does not have.
3. **Cost discipline** — hedging is a drag. Without a budget enforced at the
   portfolio level, a per-position hedge set that each looks cheap adds up to
   an unaffordable total.

Spreadsheets handle none of these. Broker option screeners handle the third
badly and the first two not at all.

### 1.3 Target user

| | |
|---|---|
| **Primary** | A self-directed investor with a $100K–$5M multi-asset portfolio who understands options well enough to place a spread order, and wants the measurement and screening done for them. |
| **Secondary** | A small RIA or family-office analyst preparing a hedging recommendation for a client, who needs the reasoning auditable. |
| **Explicitly not** | Anyone looking for a signal to follow without understanding it. OptionQ's output is dense with provenance and caveats by design; that is the product, not a flaw in it. |

### 1.4 Core product principle

**An unmeasured number must never be presented as a measured one.**

This is the single principle the v1 repair was organised around, because it is
the one the v0 build violated most often and most expensively. In a hedging
product, every number is load-bearing: a beta flows into contract count, an
implied volatility flows into premium, a VaR flows into protection level. A
plausible placeholder is not a neutral default — it is a specific false claim
that propagates into a dollar recommendation.

The operational form of the principle:

- A quantity that cannot be measured **raises or returns `None`**. It does not
  return a plausible value.
- A substituted value is **labelled at the point of use** and travels with its
  provenance all the way to the UI.
- A figure's **units and horizon travel with the figure.** A consumer derives
  its label; it never asserts one.
- Where no valid recommendation exists, the product **says so** rather than
  ranking the least-bad fabrication.

---

## 2. History: v0, the audit, and what the decisions cost

### 2.1 Origin and original design decisions

OptionQ began as **HedgeOS**, specified in `hedge_platform_prompt.md.pdf`. The
early architectural decisions, and their current status:

| Decision | Rationale at the time | Status now |
|---|---|---|
| **12-layer pipeline**, each layer a separate engine with a typed input and output | Each stage independently testable and replaceable | **Kept.** It is the main reason the v1 repair was possible without a rewrite: nearly every defect localised to one layer. |
| **Drop HMM; use HDBSCAN only** for regime detection | No fixed state count; `cluster_id = -1` gives anomaly detection for free | **Superseded.** No HDBSCAN detector was ever built. See 2.2. |
| **Pluggable LLM explainer** behind an abstract base class, provider chosen by env var | Avoid lock-in to one model vendor | **Kept and extended.** `claude` and `huggingface` providers exist; the critical addition in v1 is that *absence* of a provider is a first-class, disclosed path rather than a blank screen. |
| **Next.js + FastAPI**, not Streamlit | A real product surface, not a research notebook | **Kept.** |
| **One regime for the whole portfolio**, per-position hedging decisions | Regime is a property of the market, not of a holding | **Kept.** |
| **Alpaca paper trading / "paper trade" button** in the final deliverable | Close the loop from recommendation to execution | **Not built, and deliberately out of scope for v1.** See 7.1. |

### 2.2 The regime-detection decision was silently replaced

The architecture notes specify HDBSCAN as the sole regime detector, with
`ml/regime/hdbscan_detector.py` as the only implementation. **That file does not
exist and no module imports HDBSCAN.** What shipped is a dual-horizon Gaussian
Mixture ensemble **[code]**:

```
ml/regime/gmm_detector.py       GMMRegimeDetector, 3 components (base class)
ml/regime/main_detector.py      10-year window  — long-cycle baseline
ml/regime/fast_detector.py      6-month window  — retrained every 24h
ml/regime/combined_detector.py  weighted fusion: MAIN 0.65 / FAST 0.35
```

Labels are `low_vol | mid_vol | high_vol | anomaly`. `is_anomaly` fires when the
blended anomaly score reaches 0.60, or when either model alone exceeds 0.85
(hard override). The runtime reports its regime source as `gmm_combined`
**[verified]**.

What carried over from the original design: the 7-feature set (`vix_level`,
`vix_5d_change`, `realized_vol_20d`, `spy_return_5d`, `yield_curve_slope`,
`put_call_ratio`, `tlt_hyg_corr`), the SPY/VIX/TLT/HYG training universe, the
rolling retrain rather than a static model, and joblib persistence.

What was lost: HDBSCAN's `-1` cluster gave a *categorical* "never seen this
before". The GMM ensemble gives a *continuous* anomaly score, thresholded. That
is arguably better — it is tunable and it degrades smoothly — but it was never
written down as a decision, which is why the project's own architecture notes
described a system that did not exist. **The lesson recorded here: a deviation
that is not written down becomes a false claim in the documentation.**

### 2.3 The readiness audit

A full end-to-end product readiness audit was run against the v0 build under
these rules: inspect actual code rather than README claims, cite exact files,
distinguish implemented from tested from unverified, never equate a successful
build with a working product, and say "not verified" where evidence was absent.

**Audit verdict: 2.5 / 10. All four readiness questions answered "No."**

The build compiled, served, and returned a 200 with a plausible-looking JSON
body. It was not a working product. The gap between those two statements is the
whole subject of this document.

### 2.4 The defect classes — four found by the audit, two more by the repair

The individual defects (enumerated in §6) were not independent. They were four
recurring patterns, and naming them was what made the repair tractable:

**① Placeholder-as-measurement.** A quantity that could not be computed
returned a plausible number instead of failing.

- `return 1.0` for an unmeasurable beta is not a neutral default — it is the
  specific claim *"this holding moves one-for-one with the benchmark,"* and it
  sized contracts.
- `basis_r2 = 0.5` reads to the ranking scorer as a real, moderate
  relationship.
- `_fallback_profile` returned a **structurally complete** `RiskProfile`
  (`var_5pct = notional × 0.15`, `cvar_5pct = notional × 0.20`,
  `tail_correlation = 0.7`, `beta_vs_spy = 1.0`, both hedge ratios 1.0) with
  nothing in it derived from data. It was indistinguishable from a measured
  profile, so Layer 4 selected, priced, sized and ranked real contracts against
  an invented 15% VaR, and the user received a contract count and a premium
  for it.

**② Truncate-before-align.** Each return series was windowed to its own last
*N* rows *before* being joined, so two series ending on different dates were
compared over different calendar periods. A tz-aware/tz-naive index pair
collapsed the overlap to nothing at all. Every cross-asset number — beta,
correlation, hedge ratio, covariance — was affected.

**③ Solver-artifact IV.** Yahoo publishes `impliedVolatility` solved against a
stale `lastPrice` when no two-sided quote exists, producing values like
`0.00001` and a visible ladder of successive halvings of 1/16. Pricing on
`0.00001` IV produced a ~$0 premium and a delta near `1e-5`; the sizer, asked
to cover a position with a near-zero-delta hedge, requested **roughly 1.5
million contracts** — and presented that as a recommendation. The insight that
fixed it: **a two-sided quote is the precondition that makes an implied
volatility a measurement.** Without one, there is no IV, and clamping to a
plausible floor is still fabrication because a clamped 5% IV produces a priced,
tradeable-looking recommendation built on a quote nobody made.

**④ Asserted label.** A figure's units or horizon were asserted by its consumer
rather than reported by its producer. Layer 3 scales VaR to the user's hedge
horizon, but the models dropped the horizon, so the narrative called it
*"One-day 95% VaR"* and the UI labelled it *"1mo VaR (95%)"*. The same position
reads ≈$4.6K over one day and ≈$42.2K over 120 days: **the tail loss was
understated by roughly 9×**, and "1mo" was wrong for the 180-day default and
for every other input a user could choose.

This class recurred in the repair's own test suite, which is the instance worth
remembering. Five assertions across two test files protected the container
against shipping secrets. They read `assert ".env" in patterns`. All five
passed, and the first real build of the image contained `backend/.env` with
five live API keys — because a bare `.dockerignore` pattern matches only at the
build-context root, so the rule was present and matched nothing. **The label
was the rule's text; the thing being labelled was its effect; nothing connected
them.** A test that checks configuration is written, rather than what the
configuration does, is an asserted label wearing a test's clothes.

Two further patterns surfaced during the repair itself:

**⑤ Silent substitution.** A backstop that fills in for a failed or blank
upstream must report that it ran. Layer 11's deterministic narrative generator
ran unconditionally; when the language-model provider was reachable, raised
nothing, and returned *blank* fields, all four narrative sections were written
from templates and the response went out with `llm_provider="huggingface"` and
an **empty** notice. Template text was presented as model analysis.

The sharpest instance of this class was not in the narrative layer at all but
in the data layer, and it was found late: the two price-history paths
disagreed about whether a *"price"* includes dividends. The primary path
returned split-adjusted-only closes, the fallback returned total-return
closes, and the same ticker therefore produced a different history depending
on which path answered. It flipped the sign of HYG's annual return and moved
three of the five macro factors by about 4.5pp — and it passed every existing
plausibility gate, because annualised volatility barely moves under the
adjustment and the gates look at dispersion rather than drift. **A silent
substitution does not have to be a fallback; it can be two paths that were
never checked against each other.**

**⑥ Verified-here-only.** *A code path that behaves differently because
production's filesystem, permissions or installed dependencies differ from the
developer's is not a verified path, however green the suite is.* This class was
named after the lesson had been paid for twice. Defect 31 — the explanation
layer deleting the product's own disclosures — hid behind an unset API key,
which routed every verification run down the fallback branch so the merge code
never executed. Defect 34 — the cache crashing every analysis request on a
read-only filesystem — hid behind the fact that every directory a test touches
on a developer machine is writable; under the shipped container, where the
application tree is root-owned and the process is not, **every request would
have failed after its market-data fetch had already succeeded**. Neither had a
missing assertion. Both had a missing *condition*.

It was then paid for a third time, and the third time is the most instructive
because the missing condition was *the build itself*. An earlier pass of the
re-audit recorded that no container runtime existed on this machine, reproduced
the runtime in a clean venv and a hand-staged file tree, and stated the gap
honestly. The reproduction was faithful in every respect except the one that
mattered: the staged tree was built by applying `.dockerignore`'s rules the way
they were read, and Docker applies them differently. Building the image found
five live API keys in a layer, a 417MB host virtualenv, and the cache database
of defect 35 — a defect the register had already marked closed. **Reproducing
an environment you have reasoned about is not the same as running the
environment**, and the gap between them is precisely where this class lives.

### 2.5 Why the architecture survived

The repair preserved the 12-layer structure. Three reasons it was worth
preserving:

1. **Defects localised.** Of ~21 distinct defects, all but the horizon-label
   and provenance-plumbing changes were confined to a single layer or data
   module.
2. **The layer boundaries were where the lies crossed.** The fabricated
   `RiskProfile` was a *contract* violation between L3 and L4, visible only at
   the seam. Flattening the pipeline would have hidden it.
3. **Typed Pydantic models at each boundary** meant that adding a provenance
   field (horizon, source, R², observation count) propagated with compiler-like
   help rather than by grep.

The one structural addition is a **request-scoped data-provenance ledger**
(`backend/data/data_quality.py`), which cuts across all layers because
provenance does.

---

## 3. v1 architecture

### 3.1 System shape

```
┌──────────────────────────────────────────────────────────────┐
│ frontend/  Next.js 16.2.3 · React 19.2.4 · Tailwind v4       │
│   app/page.tsx       single-page 9-step hedging journey       │
│   lib/api.ts         typed client: per-route timeouts,        │
│                      AbortSignal composition, typed UiError   │
│   types/hedgeos.ts   response contract mirrored from Pydantic │
└───────────────────────────┬──────────────────────────────────┘
                            │ HTTP/JSON
┌───────────────────────────▼──────────────────────────────────┐
│ backend/  FastAPI · Python 3.13                               │
│   api/        routes_portfolio · routes_hedge · routes_monitor │
│               security.py  (require_admin / require_cron)     │
│   engines/    layer_01 … layer_12 + pipeline_orchestrator     │
│   core/       pricing · risk · sizing · explainer             │
│   instruments/options/  selector · position_sizer · greeks    │
│   ml/regime/  gmm · main · fast · combined · scheduler        │
│   data/       market_data · options_chain · vix_data ·        │
│               covariance · data_cache (sqlite) ·              │
│               data_quality · portfolio_valuation              │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 The 12-layer pipeline

| Layer | Responsibility | v1 notes |
|---|---|---|
| **L1** Input | Validate the portfolio; compute total notional from market prices | Valuation moved to `data/portfolio_valuation.py`; cost basis kept separate |
| **L2** Market context | Regime, VIX level, anomaly flag | `gmm_combined`; degraded state is explicit, not silently substituted |
| **L3** Risk analysis | Per-holding and portfolio VaR/CVaR, beta, hedge ratio, factors | Raises `InsufficientFactorData` rather than returning `1.0`; VaR carries its horizon |
| **L4** Instrument selection | Find listed contracts; direct, cross and macro hedges | Real expiries and strikes only; executability and liquidity screens |
| **L5** Preliminary LLM | Early narrative pass | Optional |
| **L6** Pricing | Theoretical premium vs actual bid/ask | No longer overwrites L4's market-quoted cost |
| **L7** Sizing | Contract counts, multipliers, budget gate | Single source of truth for the sizing formula |
| **L8** Greeks | Per-leg and net Greeks, collar composition | |
| **L9** Monte Carlo | Payoff simulation | |
| **L10** Scoring | Rank on effectiveness, basis risk, liquidity, cost, budget | Scores the **executable** price, not the model price |
| **L11** Final LLM | Narrative explanation | Deterministic fallback that **discloses itself** |
| **L12** Output formatting | Response assembly | Propagates horizon and provenance to the client |

### 3.3 Data provenance ledger

One ledger per request, via `contextvars`:

```python
with data_quality.capture() as dq:
    ...run the pipeline...
output.data_quality = dq.summary()
```

Every market-data read calls `dq.record(name, source, detail=…)` with one of
four sources: `live`, `cache`, `fallback`, `unavailable`. Points are named
`field:subject` (`atm_iv:AAPL`, `garch:SPY`). Modules deeper in the stack call
`record` unconditionally; with no active ledger the call is a no-op, so unit
tests and scheduler jobs are unaffected.

`summary()` returns source counts, every degraded point individually, a
grouped warning list, and `is_degraded`. **Warnings are grouped by condition**
rather than emitted one per point: a real out-of-hours run produced 77 lines,
in which the two that a user had to act on were indistinguishable from the 75
that said the same thing about different tickers. Grouping is presentational
only — the condition, the detail and every affected subject are still named,
and `degraded_fields` always carries every point. A cached group reports the
age of its **oldest** member, so the figure is never more reassuring than the
worst value behind it. **[tested]**

### 3.4 Caching

`data/data_cache.py` is **sqlite-backed and on disk**, so it survives process
restarts and deploys. Consequence for maintainers: **a semantic change to what
a cache key means requires a key version bump** (e.g. `strike_iv_v3` →
`strike_iv_v4`), or the new code will read values computed under the old
meaning. This bit the IV-plausibility work and is the single most important
operational footgun in the data layer.

**The cache is optional, and must stay optional.** Its location is *resolved*
at first use, not assumed: `OPTIONQ_CACHE_DB`, then a platform-specific
default, then the system temp directory — and each candidate is probed with a
**real write transaction**, because `os.access` and a bare `sqlite3.connect`
both succeed on a directory SQLite cannot actually write to. If no location
works, caching is **disabled and the provenance ledger says so** (every value
fetched live: slower, more rate limit, never stale). A failed *write* can
never fail the caller, because by then the caller already holds the value.

This is not defensive decoration. The fixed-path version crashed **every**
analysis request under the shipped `Dockerfile` — `/app/backend` is root-owned
and the container runs as `USER optionq` — and no test on a developer machine
could see it, because every directory a test touches locally is writable. The
container now gets a writable `/var/cache/optionq` owned by the runtime user
with `OPTIONQ_CACHE_DB` pointed at it, and `.dockerignore` keeps local `.db`
files out of the image. **[tested: `test_cache_resilience.py`, 14 tests, which
`chmod`s a real directory rather than mocking the error]**

### 3.5 Security model

| Control | Behaviour |
|---|---|
| `require_admin` | Guards every `/admin/*` route. **503 when `ADMIN_API_KEY` is unset or shorter than 16 characters** (fail closed), 401 when present but wrong. **[tested]** |
| `require_cron` | Guards `/admin/cron/retrain-*` only. Same fail-closed semantics with `CRON_SECRET`. **[tested]** |
| Credential scoping | The two credentials are scoped so neither unlocks the other. **[tested]** |
| Comparison | Constant-time; the previous `!=` comparison leaked the secret's prefix through response timing. **[tested]** |
| Error sanitisation | No handler forwards `str(exc)`. Nine did, exposing file paths, driver errors and provider URLs. **[tested]** |
| Startup validation | Configuration is validated at boot and missing secrets are logged as refusals, not warnings. **[tested]** |
| CORS | Reviewed; origin driven by `FRONTEND_URL`. **[tested]** |
| Secrets in the client | No secret reaches frontend code or logs. **[tested]** |

---

## 4. Functional requirements

### 4.1 Portfolio input

- **FR-1** Accept equity, bond, commodity and option positions with ticker,
  quantity, purchase price and purchase date. **[tested]**
- **FR-2** Accept three constraints: hedge horizon (30–365 days), protection
  level (5–30%), maximum hedge cost as a percentage of portfolio value
  (1–15%). **[tested]**
- **FR-3** Reject an unsupported ticker, a negative quantity, an empty
  portfolio and an out-of-range budget with **422 and a message containing no
  internal detail**. **[tested]**

### 4.2 Valuation

- **FR-4** Portfolio market value is computed from **current prices ×
  quantities**. Cost basis is carried separately and both are shown, with P&L
  derived. A real run reported market 337,116 against cost 234,000, P&L
  103,116. **[verified]**
- **FR-5** The cost budget is **the requested percentage of market value**, not
  of cost basis. **[tested] [verified]**

### 4.3 Risk measurement

- **FR-6** Per-holding and portfolio 95% VaR and expected shortfall, scaled to
  the user's hedge horizon, using Cornish-Fisher parametric VaR with a GARCH
  sigma. Method is reported as `parametric_cornish_fisher` or
  `assumed_vol_fallback`. **[tested]**
- **FR-7** Horizon conversion is **trading days, not calendar days**:
  `_trading_days(cal) = max(1.0, cal × 252 / 365.25)`. 120 calendar days ≈
  82.79 trading days; horizon VaR / 1-day VaR = √82.79 ≈ 9.1. **[tested]**
- **FR-8** **Every VaR figure carries its own horizon** in
  `var_horizon_calendar_days` / `var_horizon_trading_days`, and consumers
  derive their label from it. Where the horizon is unknown (0), **no horizon is
  named at all** rather than a default being assumed. **[tested] [verified]**
- **FR-9** Beta, hedge ratio and correlation are reported with their estimation
  quality: R², standard error, observation count, window length. A regression
  with fewer than `MIN_REGRESSION_OBS = 120` observations raises
  `InsufficientFactorData`. **[tested]**
- **FR-10** A holding whose risk cannot be measured is marked
  `data_available = False` and **no hedge is selected, priced, sized or ranked
  for it.** **[tested]**
- **FR-10a** **All return series are computed on a total-return basis** —
  split *and* dividend adjusted — and **every fetch path must agree**. The
  adjustment is applied as the ratio `adjclose / close` across all four OHLC
  columns, not by substituting the adjusted close for the close: the provider
  scales `adjclose` but not `high`/`low`, so substitution puts the close
  outside `[Low, High]` on roughly 8% of bars. Where the provider returns no
  adjusted close, the bars stay on a price basis and the provenance ledger
  records `history_adjustment:<ticker>` as **unavailable**. This matters most
  for the macro factor set, which is deliberately the high-yielding bond and
  credit ETFs where the dividend is most of the return. **[tested]
  [verified live]**

### 4.4 Instrument selection

- **FR-11** Every recommended contract must **correspond to an actual listed
  contract**: a real expiry, a real strike, and an OCC symbol whose encoded
  strike matches the strike claimed. Synthetic expiry and strike generation is
  removed. **[tested] [verified]** — a real run produced 14 candidates, all with
  OCC symbols matching their strikes and all verified against the chain.
- **FR-12** Supported strategies: **protective put**, **bear put spread**,
  **collar** (including **credit** collars, where the user is paid to establish
  the position), and **inverse/macro** structures, across **direct**, **cross**
  and **macro** hedge categories. **[code] [partly tested]**
- **FR-13** Expiry selection must cover the requested horizon. A 7-day expiry
  offered against a 180-day hedge request is a defect. **[tested]**
- **FR-14** Candidates failing executability or liquidity screens are excluded
  and the exclusion is disclosed. **[tested]**

### 4.5 Pricing

- **FR-15** Use **contract-specific implied volatility with skew**, read from a
  contract that has a **two-sided quote**. Realized volatility may be
  substituted only with `iv_source = realized_vol` labelled on the candidate.
  **[tested] [verified]**
- **FR-16** Implausible IV readings are **rejected, returning `None`** — never
  clamped to a plausible floor. **[tested]**
- **FR-17** Theoretical premium is compared against actual bid/ask, and
  `premium_source` (`market_mid` or `model`) is labelled on every candidate.
  **[tested] [verified]**
- **FR-18** Layer 6 must not overwrite a market-quoted cost with a theoretical
  price. **[tested]**

### 4.6 Sizing and budget

- **FR-19** Contract counts are whole positive integers, using the correct
  contract multiplier and the **hedge instrument's** spot (not the holding's).
  **[tested]**
- **FR-20** Both debit and credit structures are supported; a credit collar is
  not clamped to a $0.01 debit. **[tested]**
- **FR-21** **Portfolio-level** hedge cost is summed and enforced against the
  user's budget. A real run reported cost 7,328 against a budget of 10,113
  (72.5% utilisation), with the budget exactly 3% of market value.
  **[tested] [verified]**
- **FR-22** One sizing formula, in one place. L4, L6 and L7 must agree.
  **[tested]**

### 4.7 Ranking

- **FR-23** Rank on effectiveness, basis risk, liquidity, transaction cost and
  budget fit, scoring the **executable** price. **[tested]**
- **FR-24** Admission gates are stated in **R²**, not correlation. A 0.25
  correlation floor admits R² = 0.0625 — a hedge leaving 94% of the movement
  uncovered. The audit found XLP ranked best cross hedge for AAPL at R² = 0.10.
  **[tested]**
- **FR-25** **Where no valid hedge exists, no recommendation is forced.** The
  position is returned unhedged with a stated reason. A real run reported *"No
  hedge for TLT fits the cost budget and liquidity constraints; the position is
  shown unhedged."* **[tested] [verified]**

### 4.8 Explanation

- **FR-26** A narrative is produced on every run, with or without a language
  model. **[tested] [verified]**
- **FR-27** If any narrative section is written by the deterministic generator
  rather than the model, **`llm_notice` says so and names the substituted
  sections.** An all-template narrative gets the full-substitution notice; a
  partially written one names only the sections that were filled; a fully
  model-written narrative gets **no** notice, because a notice there would be a
  false warning. **[tested] [verified]**
- **FR-28** The model is given each VaR's horizon (`var_horizon_days`) in all
  three prompt payloads, so it cannot invent a label. **[code] [tested at the
  prompt-construction level]**
- **FR-29** **A disclosure the product asserts cannot be removed by the
  explanation layer.** The three hard disclosures — no executable two-sided
  quote, thin market, expiry crosses earnings — are recorded in
  `mandatory_disclosures` as well as in the rendered `cons`, and restored
  after every LLM merge in both L5 and L11. The model may rewrite prose; it
  may not delete a statement about whether a price is real. The LLM is an
  explanation layer, never a source of fact, and this is the boundary stated
  as code. **[tested] [verified over the live HTTP route]**
- **FR-30** An unrecognised `llm_provider` is logged at ERROR naming both the
  ignored value and the provider that will answer instead. Falling back is
  allowed; falling back silently is not. **[tested]**

### 4.9 Frontend

- **FR-29** Complete 9-step journey: enter positions → set constraints →
  submit → loading state → portfolio summary → per-holding cards → candidate
  cards with Greeks and provenance → budget bar → disclosures. **[code]
  [type-checked and lint-clean]**
- **FR-30** Per-route request timeouts via `AbortSignal.timeout`, composed with
  a caller's own signal through `AbortSignal.any`, so a cancelling component
  and the timeout coexist. Typed `UiError` discriminates `timeout` from other
  failures. **[code]**
- **FR-31** Every recommendation surfaces its data source, freshness, pricing
  assumption and risks via `ProvenanceStrip`, `PremiumSourceBadge`,
  `DataQualityPanel` and `WarningsPanel`. **[code] [verified in response
  payloads]**
- **FR-32** Disclosures render at the **top** of the result, not in a collapsed
  footer. **[code]**

---

## 5. Non-functional requirements

| | Requirement | Status |
|---|---|---|
| **NFR-1** | A request completes or fails within its timeout; `routes_portfolio` wraps the pipeline in `asyncio.timeout(ANALYSIS_TIMEOUT_SECONDS)` | **[tested]** |
| **NFR-2** | No response body contains a traceback, a library type name, a file path or an attribute repr | **[tested]** — the leak detector itself is pinned by two tests, one proving it fires on a real traceback and one proving it stays quiet on legitimate product prose |
| **NFR-3** | Privileged endpoints fail closed when their secret is unset | **[tested] [verified]** — all four refuse with 503 |
| **NFR-4** | Unavailable or stale data is disclosed, never silently substituted | **[tested] [verified]** |
| **NFR-5** | Deterministic tests must not depend on the network; live-data tests are opt-in and skip honestly | **[tested]** — `synthetic_market` fixture; 21 live tests skip out of hours with a stated reason |
| **NFR-6** | Every critical bug has a regression test | **[tested]** — see §8 |

---

## 6. v1 defect register

Every entry was found by the audit or by running the repaired product, and
every entry has at least one regression test. Grouped by the defect class from
§2.4.

### ① Placeholder-as-measurement

| # | Defect | What the user saw | Fix | Pinned by |
|---|---|---|---|---|
| 1 | `return 1.0` for an unmeasurable beta | Contract counts sized on an asserted one-for-one relationship | `InsufficientFactorData` raised; `beta_available` flag | `test_factor_decomposer.py::TestUnmeasurableRaises` |
| 2 | `_fallback_profile` returned a complete fabricated `RiskProfile` | Contract count, premium and risk-reduction estimate for an invented 15% VaR | Profile flagged `data_available = False`; L4 declines to select against it; portfolio beta excludes it | `test_degraded_risk_profile.py` (3 classes) |
| 3 | `basis_r2 = 0.5` placeholder | Scorer read a fabricated moderate relationship as real | Provenance-carrying estimates; unmeasured basis risk excluded | `test_covariance.py::TestEstimateProvenance` |
| 4 | Unmeasured liquidity treated as adequate | Illiquid contracts admitted | Explicit unmeasured-liquidity path | `test_selector.py::TestUnmeasuredLiquidity` |
| 4a | L10 `_basis_score` mapped an unmeasured basis R² **back to 0.5**, undoing defect 3's fix one layer downstream | At a 15% scoring weight, a hedge nobody could measure outranked one honestly measured as weak — the ranking was biased toward the candidates we knew least about | `basis_risk_source` separates measured from unmeasured: identity → 1.0, measured → its own value, unmeasured → 0.0 | `test_selector.py::TestBasisRiskIsScoredOnWhatWasMeasured` (6, revert-verified) |
| 4b | Selector default `basis_r2: float = 0.5`, formatted straight into user-facing rationale | "R²=0.50" presented as a measurement of tracking nobody performed | Default removed; unmeasured prints "Basis R² not measured." | `test_selector.py::TestBasisRiskProseMatchesTheMeasurement` |

### ② Truncate-before-align

| # | Defect | What the user saw | Fix | Pinned by |
|---|---|---|---|---|
| 5 | Each series windowed to its own last *N* rows before the join | Beta, correlation and hedge ratio computed over **different calendar periods** for the two legs | Align first, then window | `test_factor_decomposer.py::TestAlignBeforeWindow`, `test_covariance.py::TestWindowAlignment` |
| 6 | tz-aware / tz-naive index pair | Overlap collapsed to **zero rows** | `align_series(**named)` normalises timezones | `test_covariance.py::TestAlignSeries`, `test_live_market_data.py::TestAlignment` |
| 7 | Historical-period mapping wrong; empty arrays reaching the regime detector | Regime computed from nothing | Period mapping corrected; empty input refused | `test_live_market_data.py::TestPriceHistory` |

### ③ Solver-artifact IV

| # | Defect | What the user saw | Fix | Pinned by |
|---|---|---|---|---|
| 8 | `impliedVolatility: 0.00001` on never-traded contracts | ~$0 premium, delta ≈ 1e-5, **≈1.5 million contracts recommended** | Plausibility band; implausible readings **rejected**, returning `None`, never clamped | `test_iv_plausibility.py::TestPlausibilityPredicate`, `TestStrikeIV` |
| 9 | IV accepted without a two-sided quote | A solver artifact presented as an implied volatility | **Two-sided quote is the precondition for an IV**; `atm_iv` unavailable where no quoted strike sits within 10% of spot | `TestQuotePrecondition`, `TestAtmQuotePrecondition`, `TestContractQuoteIV` |
| 10 | Synthetic $5-grid strikes that were never listed | Recommendations for contracts that do not exist | `_resolve_strike` against the real chain | `test_selector.py::TestResolveStrike` |
| 11 | 7-day expiry offered as a 180-day hedge | Protection expiring long before the horizon | `_sample_expiries` horizon coverage | `TestExpiryHorizonCoverage` |
| 12 | 34.7%-spread / 20-OI contract ranked best in book | An unfillable contract as the top recommendation | `_liquidity_score` | `TestLiquidityScore` |
| 13 | Ranking computed on the model price, not the fill | Rank order reflected a price that was not for sale | `_rescore` on the executable price | `TestRescore`, `test_sizing.py::TestExecutablePremium` |
| 14 | Credit collar clamped to a $0.01 debit | A structure that pays the user shown as a cost | `_apply_net_market_quote` | `TestApplyNetMarketQuote`, `test_sizing.py::TestCreditStructures` |

### ④ Asserted label

| # | Defect | What the user saw | Fix | Pinned by |
|---|---|---|---|---|
| 15 | VaR horizon dropped by the models; L11 said "One-day", UI said "1mo VaR" | A 120-day tail loss of $42.2K presented as overnight — **understated ≈9.1×** | Horizon travels with the figure through `RiskProfile`, `PortfolioRiskSummary`, `UnhedgedRisk`, the LLM prompts and the UI label | `test_var_horizon_label.py` (6 classes, 22 tests) |
| 16 | Calendar-vs-trading-day VaR scaling | Wrong horizon conversion | `_trading_days()` | `test_risk.py`, `test_var_horizon_label.py::TestTheFigureIsHorizonScaled` |
| 17 | Misleading historical-vs-parametric VaR naming | A parametric figure labelled historical | `method` reported explicitly | `test_risk.py` |
| 18 | Moneyness read in the wrong direction | The OTM caveat appeared on **no** recommendation, while the pros list beside it advertised unlimited downside protection | `_put_cons` / `_put_pros` corrected | `test_disclosure_and_budget.py::TestMoneynessCaveat`, `TestOptionPros`, `TestOptionCons` |
| 18a | `"Basis R²={r2} — reasonable hedge tracking"` listed as a **pro** at any R² | A measured R² of 0.08 presented to the user as a point in the hedge's favour, contradicted by the number inside its own sentence | The pro is earned (≥0.50 "tracks closely", ≥0.25 "partial") or absent; the cons list already stated it honestly | `test_selector.py::TestBasisRiskProseMatchesTheMeasurement` |
| 18c | `covariance.py`'s docstring described the basis-risk estimate as *"signed, 60-day rolling"*; the code fits on `LOOKBACK_DAYS = 252` | Nothing, directly — but anyone auditing whether a reported R² of 0.08 was plausible was reasoning about a window roughly **four times noisier** than the one in use, which is exactly the mistake R7 invited | Docstring states the real window and records that it changed | — (documentation; the window itself is pinned by `test_covariance.py`) |
| 18b | Spread builder's `tail_correlation` fallback set a real measurement with **no source** | Under provenance-aware scoring it would have read as never measured | The fallback labels itself `tail_correlation` | `test_selector.py::TestBasisProvenanceIsAlwaysDeclared` |

### ⑤ Silent substitution

| # | Defect | What the user saw | Fix | Pinned by |
|---|---|---|---|---|
| 19 | Three L11 paths left all four narrative fields empty | A blank explanation panel; *"the model had nothing to say about your risk"* and *"the service is down"* looked identical | Deterministic narrative on every path | `test_llm_fallback.py::TestFallbackPaths` |
| 20 | Provider available, raising nothing, returning **blank** fields → template text with `llm_provider="huggingface"` and an **empty** notice | Template text presented as model analysis | `_deterministic_narrative` returns the fields it filled; `run()` sets a full or partial substitution notice naming them | `test_var_horizon_label.py::TestNarrativeSubstitutionIsDisclosed`, `TestEngineDisclosesSubstitution` |
| 21 | Nothing summed per-holding hedge costs | Each hedge passed its own gate; the **total** could exceed the user's budget | Portfolio-level budget enforcement | `test_disclosure_and_budget.py::TestPortfolioBudget`, `test_selector.py::TestHoldingBudget` |
| 31 | **Both LLM layers replaced `candidate.cons` wholesale, and `cons` is where the product's own disclosures live.** `run_l11_llm` defaults to `True`, so with any model configured a reply containing its own bullets **deleted** *"premium shown is theoretical, not an executable price"*, the thin-market warning and the earnings IV-crush warning from the API response | A hedge with a premium, a contract count and a cost, and nothing to say the premium was a model output. Proven live: the reverted build returned an MSFT put on a contract quoted **8.9% wide** described only as *"Premium is a cost drag"* | `backend/core/disclosures.py` — `cons` stays the rendered list, `mandatory_disclosures` records what the product said, `restore()` runs after every merge and reinstates anything dropped, ahead of the model's prose | `test_disclosures.py` (19), `test_pipeline.py::TestDisclosuresSurviveTheLLMLayers` (6), and a live HTTP run proven both ways |
| 33 | **The two price-history paths disagreed about whether a "price" includes dividends.** `_yahoo_history_direct` read `indicators.quote[0].close` — split-adjusted, **not** dividend-adjusted — while the yfinance fallback used `auto_adjust=True`, which is both. The same ticker returned a different history depending on which path answered, and nothing disclosed it. Impact over 1y: **HYG −4.25% → +1.63%, a sign flip**; TLT and LQD ~4.5pp; 2y price levels 9–12% apart. **Three of the five `MACRO_TICKERS` are the worst affected**, because the macro set is deliberately the high-yielding bond and credit ETFs where the dividend *is* most of the return. It survived every prior cycle because annualised vol moves ≤0.37pp under the adjustment, and the data-quality gates look at dispersion, not drift | Request `includeAdjustedClose=true` and apply the ratio `adjclose / close` to **all four** OHLC columns. Not substitution: `adjclose` is scaled while `high`/`low` are not, which puts the close outside `[Low, High]` on **107 of 1254** NVDA bars. No adjusted close from the provider → bars stay on a price basis and the ledger records `history_adjustment:<ticker>` as **unavailable** | `test_price_adjustment.py` (5 classes, 16 tests; **7 of 14 fail** on revert), `test_live_market_data.py::test_both_fetch_paths_return_the_same_adjusted_series`, `::test_the_adjusted_bars_are_internally_consistent` |
| 34 | **The cache crashed the request on a read-only filesystem — P0, production-only.** SQLite was opened at a fixed path inside the root-owned application tree and every error propagated, so `cache_set` (and `get_price_history` through it) raised `OperationalError: attempt to write a readonly database` **after the market-data fetch had already succeeded**. Under the shipped `Dockerfile` every analysis request on Cloud Run would have failed. Invisible to the entire suite | Path resolved and probed with a real transaction; all three public operations guarded; no writable location → caching disabled **and disclosed**. Writable `/var/cache/optionq` in the image, owned by the runtime user | `test_cache_resilience.py` (4 classes, 14 tests; **3 of 14 fail** on revert, including the one asserting a price fetch still returns) |
| 35 | 6.4MB of local cache database shipped inside the image by `COPY . .`, arriving **readable but not writable** — so reads returned a developer's months-old prices while writes failed, which is what made the failure look intermittent | `.dockerignore` excludes `*.db`, `*.sqlite*` and the journal/WAL sidecars. **This row previously read "Verified on the staged tree: 0 `.db` files, no secrets" — and the first real build showed `/app/backend/hedgeos_cache.db` present, 6.4MB, root-owned. The staged tree was built by applying the rules as I read them, not as Docker applies them; see defect 37.** Closed for real once the patterns were `**/`-anchored and a build-time guard added: verified absent from the rebuilt image | `test_deployment_contract.py`, `test_security.py` — both now evaluate the rules instead of asserting their text |
| 36 | **`wrapt-2.6.0.dev1`, a development prerelease, resolved into the runtime closure** and installed on both macOS and linux/amd64. Nothing in `backend/` imports it; it arrives via `arch` → `statsmodels` → `formulaic`, which declares `wrapt>=1.17.0rc1` for `python_version >= '3.13'`. A specifier that *names* a prerelease makes pip willing to resolve prereleases for that requirement, so a dev build beat the 2.5.2 stable release. A dev prerelease carries no release guarantees and can change behaviour between identical-looking builds — not something belonging in the dependency graph of a system that prices hedges | `wrapt==2.5.2` pinned in `api/requirements.txt` with the reason recorded; 2.5.2 satisfies formulaic's constraint. Confirmed 2.5.2 inside the built image | `test_deployment_contract.py::TestNoPinIsAPrerelease` (2; both fail on revert — one on the pin text, one on the installed version) |
| 37 | **Every `.dockerignore` rule aimed at something inside `backend/` matched nothing, so three fixes recorded as closed were never in effect.** `.dockerignore` is not `.gitignore`: Git re-applies a bare pattern at every directory level, Docker and Buildah match each pattern once against the whole path relative to the context root. `.env` excluded `./.env` and nothing else. `backend/tests/` was the only rule in the file that worked, because it names its path — which is exactly why the leak looked solved. The first real build of the image contained **`backend/.env` with five live API keys (P0)**, the **417MB** host virtualenv, and the 6.4MB SQLite cache of defect 35. Five test assertions across two files covered this and all five passed: they asserted the strings `.env`, `.venv/`, `*.pem` and `*.key` were *present in the file* — defect class ④ in its purest form | Three parts, because the pattern fix alone leaves the hole open to the next edit: (1) every nested rule `**/`-anchored, with the semantics documented in the file header; (2) a **build-time guard** in the `Dockerfile` that `find`s the copied tree and **fails the build** on any `.env`, `*.pem`, `*.key`, `*.db`, `.venv` or `tests/`; (3) the five string-presence assertions replaced by tests that **evaluate** the rules. Image rebuilt: **1.08GB → 668MB**, no secrets, no venv, no db | `backend/tests/fixtures/dockerignore.py` — a matcher calibrated against two throwaway builds, not a doc page; `test_deployment_contract.py` (4 tests), `test_security.py` (3). Revert-verified: with the bare patterns 3 fail, naming `backend/.env`. The build guard is revert-verified too — restoring the bare patterns fails the build at step 6 with `REFUSING TO BUILD:` and the paths listed |
| 32 | **`get_explainer()` returned `ClaudeExplainer` for every provider name it did not recognise**, and recognised only `"huggingface"`. `llm_provider=ollama` therefore sent the portfolio to Anthropic with nothing in the logs. `OllamaExplainer` was unreachable regardless: it read `settings.ollama_model` and `settings.ollama_base_url`, neither of which was declared, so constructing it raised `AttributeError` | A configured provider silently ignored; a whole provider that had never worked | `"ollama"` wired, both settings declared, and an unknown provider name logged at ERROR naming what was ignored and who answered instead | `test_disclosures.py::TestTheExplainerFactoryDoesNotSubstituteSilently` (4) |

### Security defects

| # | Defect | Fix | Pinned by |
|---|---|---|---|
| S1 | `/admin/retrain/main` — a blocking, CPU-bound retrain — had **no authentication at all** | `require_admin` | `test_security.py::TestAuthentication` |
| S2 | `_verify_cron_secret` returned early when `CRON_SECRET` was unset (*"no secret configured — open access"*), so a deployment that forgot the secret exposed model retraining to anyone who could reach the host | **Fails closed** with 503 | `TestFailsClosed` |
| S3 | Secret compared with `!=`, leaking its prefix through response timing | Constant-time comparison | `TestConstantTimeComparison` |
| S4 | Nine handlers raised `HTTPException(500, detail=str(exc))`, forwarding file paths, driver errors and provider URLs | Sanitised | `TestErrorSanitisation` |
| S5 | Admin and cron credentials not scoped | Neither unlocks the other | `TestLeastPrivilege` |
| S6 | No startup configuration validation | Validated at boot | `TestStartupValidation` |
| S7 | CORS unreviewed | Origin from `FRONTEND_URL` | `TestCorsPolicy` |
| S8 | Dependency incompatibilities; Docker/cloud config | `requirements.txt`, `Dockerfile`, `.dockerignore` | `TestDeploymentConfiguration` |

---
| 30 | **No `RequestValidationError` handler existed.** FastAPI returned pydantic's error objects verbatim: `"msg": "Unable to extract tag using discriminator 'position_type'"`, a `ctx` naming the discriminator, and an `input` echoing the submitted position data back. `frontend/lib/api.ts` copies `msg` into the rendered field error, so that sentence reached the user. The existing leak assertions searched for paths, stack frames and module names — none present in this body — so it passed them all. | `backend/api/validation_errors.py` translates each error to a sentence, keeps `loc` for field highlighting, drops `ctx`/`input`/`type`, logs the original server-side | `test_e2e_api.py::TestValidationErrorsDoNotLeakLibraryInternals` (6) |

## 7. Out of scope

### 7.1 Not built, deliberately

- **Live trade execution.** Not implemented and not to be implemented without
  explicit approval. The original spec's Alpaca "paper trade" button is
  deferred: a recommendation engine whose numbers were wrong in 21 distinct
  ways had no business holding an order ticket, and the trust bar for execution
  is higher than for advice.
- **Production deployment.** `Dockerfile` and Cloud Run configuration exist and
  are tested as *configuration*; **no deploy is to be performed without
  explicit approval.**
- **Multi-user accounts, persistence of portfolios, billing.** Single-session
  tool.
- **Advice.** OptionQ reports measurements and ranks instruments. It does not
  tell a user what to do.

### 7.2 Known-incomplete in v1

| Gap | Why it matters | Current state |
|---|---|---|
| ~~Market-hours verification~~ | ~~Every run executed outside US market hours.~~ | **CLOSED 2026-10-29 09:58 EDT.** 21/21 live tests pass; `premium_source=market_mid`, `iv_source=contract_iv`; the screen rejects 68/165 live SPY strikes (67 OI, 1 spread); SPY 95% put 10.30/10.37. Closing it first required fixing a DST bug in the market-hours gate, which had been skipping the decisive assertion during EDT. |
| ~~Risk provenance not exposed to the client~~ | ~~`beta_r2`, `beta_stderr`, `beta_n_obs`, hedge-ratio stats, `factors_unavailable` and the covariance `n_obs`/`window_days`/`corr_stderr` existed on `RiskProfile` but not on the response.~~ | **CLOSED.** `RiskEstimateProvenance` travels on every `HedgeRecommendation`; `basis_risk_source`/`_n_obs`/`_window_days`/`_stderr` on every `InstrumentCandidate`. Chosen over a sibling risk endpoint, because a sibling lets a client render a beta without ever fetching its standard error. Rendered and browser-verified. |
| ~~Single-stock correlation structure is weak and unexplained~~ | ~~Over 252 sessions AAPL/SPY R²=0.083, MSFT/SPY 0.154, and pairwise \|ρ\| across five large caps has a median of 0.11 — while ETF pairs over the same window are coherent.~~ | **CLOSED 2026-10-29 — and it was never a defect.** Two measurements settled it. **(a)** The product's return series is **byte-identical** to an independent recompute straight from the raw provider JSON — `maxdiff = 0.00e+00` on both legs — so nothing in the data stack distorts it. **(b)** The window is the whole story: `LOOKBACK_DAYS = 252` is what the product fits on, and over 500 sessions the same AAPL/SPY pair gives R²=**0.354**, ρ=**+0.595**, β=**1.03**. Single-stock/index correlation genuinely collapsed over the trailing year — XOM/SPY flips **+0.115 → −0.329** — while index-replication pairs cannot move (VOO/SPY 0.994 → 0.996). Mega-cap dispersion, correctly measured and correctly **gated**: SPY is *rejected* as an AAPL cross hedge at R²=0.084 < `MIN_CROSS_R2 = 0.30`. Found while root-causing it: `covariance.py`'s docstring still said *"60-day rolling"* (defect 18c). |
| ~~No valid-quote hard refusal~~ | ~~Open design question: refuse, or rank disclosed theoretical premiums?~~ | **CLOSED 2026-10-29.** Decided against both plain answers, because both answer the wrong question. `L4._refuse_unquotable_hedges` runs once across the whole run: if **some** instrument carries an executable two-sided quote, the venue is quoting, and a holding with none has no market — withheld, reported unhedged, and the warning distinguishes an *absent* market from a *closed* one. If **nothing** is quoted anywhere the provider is silent, and disclosed theoretical premiums stand. No market calendar, deliberately: one would need maintaining and would refuse hedges wrongly on every holiday it missed. The asymmetry is deliberate too — a missed refusal leaves a disclosed theoretical premium on screen, a wrong refusal silently removes a placeable hedge — so refusal fires only on positive evidence of quoting. 12 tests, 4 revert-verified; armed and correctly withholding nothing on the live market-hours run. |
| ~~Layer 5 preliminary LLM pass~~ | ~~Lightly exercised.~~ | **CLOSED.** It is off by default (`run_l5_prelim_llm = False`), so it was untested-by-default code, not a defect. Its docstring claimed *"Never raises — all LLM failures are caught and logged"* while the method has no `try/except`; corrected to state what is true (the orchestrator's guard is what makes a failure non-fatal), and that guard is now pinned by a test that raises from L5 and asserts the run still produces candidates. **Closing this is what surfaced defect 31** — reading L5's merge led to checking whether L11, which runs by default, did the same thing. It did. |
| ~~**The container image has never been built**~~ | **CLOSED — and closing it found a P0 the whole suite had missed.** The earlier entry here was wrong on one point: no container runtime was *present*, but Homebrew and hardware virtualization were available, so one could be *installed*. Podman 6.1.4 installed, VM started, and the image built for the deployment target (`--platform linux/amd64`, since this host is arm64): 11 steps, exit 0, all 73 distributions resolving to `cp313` linux x86-64 wheels with zero source builds. Running it showed `backend/.env` with five live API keys in the image, the 417MB host virtualenv, and the cache database of defect 35 — see defect 37. After the fix the image is 668MB and re-verified 11/11: no secrets, unprivileged `uid=10001`, `x86_64`/`glibc 2.41`/Python 3.13.18, every import resolving, fail-closed on all three bad configurations (exit 3), `/health` 200, the full auth matrix including both cells where a *valid* credential is correctly refused, a real `POST /portfolio/analyze` returning 200 in 31.2s with real OCC contracts and 0.854% of notional against a 2% cap, the cache confined to `/var/cache/optionq`, and zero tracebacks in the logs | **Closed.** What remains open is narrower and stated as such: Cloud Run's own networking, memory limits and cold start are untested; the Claude explainer path was not exercised in the container (no key passed, so the deterministic fallback ran and disclosed itself); and the container ran x86-64 under qemu emulation, so timing is not representative. Nothing has been deployed. |
| ~~Price history was dividend-unadjusted on the primary path~~ | ~~The same ticker returned a different history depending on which fetch path answered.~~ | **CLOSED 2026-10-29.** Defect 33 in §6. Found by diffing the two paths against each other rather than by any gate: annualised vol moves ≤0.37pp under the adjustment, so every dispersion-based plausibility check passed it. |
| ~~The cache crashed every request on a read-only filesystem~~ | ~~P0 under the shipped `Dockerfile`, and invisible to the whole suite.~~ | **CLOSED 2026-10-29.** Defects 34 and 35 in §6, and §3.4 for the contract the cache now holds to. |
| ~~Stale results stay on screen after an invalid edit~~ | ~~Figures computed for a different portfolio remain on screen.~~ | **CLOSED**, and root-caused past the reported symptom. The invalid-edit case was one instance of a general problem: *any* unre-run edit left real figures beside inputs they did not describe — changing 500 shares to 5,000 left a VaR for 500 shares on screen with nothing to say which portfolio it belonged to. The page now records the exact payload each result was computed from, raises a banner with a re-run button when the form diverges, and clears the result outright when the form can no longer be analysed. 5 browser checks, 28/0 overall. |

---

## 8. Verification

### 8.1 Test strategy

- **Deterministic by default.** Unit and integration tests use the
  `synthetic_market` fixture (`backend/tests/fixtures/synthetic_market.py`),
  which patches `yfinance.Ticker`, the market-data HTTP session, and disables
  every cache consumer. A test that depends on Yahoo answering is a test that
  fails for reasons unrelated to the code.
- **Live tests separated and opt-in.** `test_live_market_data.py` calls real
  providers, is marked opt-in, and skips with a stated reason out of hours —
  because *"a hedging product whose only evidence is mocked fetches has
  verified its mocks, not its data path."*
- **The product pipeline is never mocked.** The end-to-end test substitutes
  exactly two things: the market-data provider and the language model. Both are
  **network boundaries, not product logic**. The route, all twelve layers,
  valuation, selection, pricing, sizing, Greeks, the Monte Carlo pass, scoring,
  the narrative, the ledger and the response serialisation all run unmodified.
- **No assertion was weakened to make a test pass.** Where a test failed
  because a fixture was wrong, the fixture was fixed. One test was found
  passing *for the wrong reason* — reaching the generic exception path instead
  of the successful-but-blank path it was written for — and was corrected.

### 8.2 Current results

| Suite | Result |
|---|---|
| Full backend suite | **674 passed, 31 skipped, 0 failed** — 91.4s and 52.1s on two runs. The skips are the opt-in live tests, excluded by their gate rather than by failure |
| **Live-provider suite, market open** | **21 passed, 0 skipped** in 2.6s (`OPTIONQ_LIVE_TESTS=1`, 09:58 EDT 2026-10-29) |
| End-to-end API integration (`test_e2e_api.py`) | **59 passed**, offline |
| **Browser-driven UI journey** | **28 passed, 0 failed, 7 informational** — Chromium against the static production export and the live backend |
| **Live run with an LLM provider wired in** | **Fix proven both ways** over the real HTTP route: 5 disclosures deleted with the fix reverted, 0 with it in place. The real `OllamaExplainer`, its real JSON parser, the real merge and the real FastAPI route are all in the path; only the model's reply is substituted |
| **Deployment configuration contract** | **12 passed** — every third-party import in `backend/` is pinned, every pin matches the version the suite passes against, no secrets or tests in the image, non-root `USER`, `$PORT`/`0.0.0.0` in `CMD`, image Python matches the local interpreter. These assert the *config*, because a `pytest` run must not need a container runtime, a network and several minutes of build. **Asserting the config is also how a P0 hid for a cycle** — five of these assertions checked that `.dockerignore` rules were *written*, not what they matched (defect 37). They now evaluate the rules against the paths that actually leaked, and the image itself is verified out of band by a real build |
| Frontend | `tsc --noEmit` clean; `eslint app lib types` clean |
| End-to-end against a real `uvicorn` server | **45 passed, 0 failed, 4 informational** (2026-10-29, 05:50 ET — pre-market) |
| **End-to-end inside the built container** | **200, 41.6KB, 31.2s** through `optionq-api`, built `--platform linux/amd64` and run under podman. Real OCC contract `SPY261218P00769000`, 3 ranked assets / 11 candidates, notional $487,369.00, budget **$4,160.50 ≤ $9,747.38** (0.854% against the 2% requested), provenance `42 cache / 38 live / 2 unavailable / 2 fallback` with every degraded field named, 12 SPY contracts **excluded** for a spread wider than 50% of mid, `llm_notice` disclosing the deterministic fallback, cache confined to `/var/cache/optionq` (4.68MB, nothing under `/app`), zero tracebacks in the logs. Fail-closed fired on three bad configurations (no secrets, short secrets, `http://localhost` origin — exit 3 each); the auth matrix correct in all 8 cells including both valid-credential refusals. **What this does not prove:** Cloud Run's networking, memory limits and cold start; the Claude path (no key was passed); and the container ran x86-64 under qemu emulation, so the 31.2s is not representative timing |

### 8.3 What the real end-to-end run established

Against a live server with a four-asset portfolio (AAPL, IWM, TLT, GLD;
120-day horizon, 15% protection, 3% budget) **[verified]**:

- Market value 337,116 vs cost 234,000, P&L 103,116 — valued from current
  prices, cost basis separate
- 4 recommendation entries for 4 holdings; 14 candidates
- Every candidate carried an OCC symbol **matching its claimed strike** and
  verified against the chain
- `premium_source = {model: 14}`, `iv_source = {realized_vol: 14}` — expected
  out of market hours, and **disclosed rather than hidden**. Re-run during the
  open session on 2026-10-29, the same paths produce
  `premium_source = market_mid` and `iv_source = contract_iv`, so the
  disclosure tracks the data rather than being a permanent label
- Hedge cost 7,328 / budget 10,113 = 72.5%; budget exactly 3% of market value
- Regime `mid_vol` via `gmm_combined`, not degraded
- Narrative: *"$337.1K across 4 holdings. **Over the 120-day hedge horizon**,
  95% VaR is $42.2K and 95% expected shortfall $68.8K. Portfolio beta to SPY is
  0.73."*
- `llm_notice`: *"The language model returned no analysis for this run, so the
  narrative below was generated directly from the computed figures instead."*
- TLT returned **unhedged** with a stated reason rather than a forced
  recommendation
- All five malformed inputs → 422, sanitised; all four privileged endpoints →
  503 fail-closed; unknown route → 404; **no internals leaked anywhere**
- Disclosures: 51 degraded data points compressed to 36 grouped warning lines
  (37 user-facing warnings in total, down from 78 before grouping), with
  **every degraded field still named** — verified by the script rather than
  assumed

### 8.4 Definition of done

OptionQ is done when it completes the portfolio hedging workflow using valid
market data, produces financially defensible recommendations, enforces the
user's risk and cost constraints, and passes end-to-end testing.

**Against that bar, v1 is complete except for one open item in §7.2**, and
that item is a *build*, not a behaviour.
Market-hours verification — the item that most affected confidence, because
the product had never been observed pricing a hedge off a live two-sided quote
— is closed by observation: during the open session on 2026-10-29 the chain
returned 163 of 165 strikes two-sided, the liquidity screen rejected 68 of
them on the product's own thresholds, and premiums came from the market mid at
the contract's own IV. The no-valid-quote question is decided and implemented,
Layer 5 is accounted for, and stale results can no longer be read as current.

**The last data-confidence question is now answered.** The single-stock betas
are fit on genuinely weak correlation, and that weakness is *real*: the
product's series matches an independent recompute from raw provider JSON
exactly, and the weak R² reproduces or disappears purely as a function of
window length while index-replication pairs do not move. The product measures
it correctly, renders it, and **refuses the hedge** rather than sizing against
a beta it does not believe.

**Two caveats on "complete", stated rather than implied.** The image has now
been built and run, and doing so found a P0 the suite had coexisted with for a
cycle (defect 37) — so the caveat is narrower but sharper: the *image* is
verified, the *platform* is not. Cloud Run's networking, memory limits and
cold start are untested, the Claude explainer path was not exercised in the
container, and the container ran x86-64 under qemu emulation. And while the
repair is committed — on the local branch `repair/end-to-end-phase-2` —
nothing is pushed and nothing is deployed; both require explicit approval.

**Which bar is being claimed.** *Ready for external user testing* — the
workflow completes on live data, the figures are defensible and provenanced,
the failure paths are honest, the constraints are enforced, and the deployment
configuration fails closed. **Not** *ready for unsupervised public use*, which
would additionally require the image built and running, a monitored
deployment, and market-hours observation over more than a single session.

**One lesson is worth recording in the PRD itself**, because it is about how
this product gets verified rather than about any one defect, and it has now
been paid for **twice**.

Defect 31 — the LLM layer deleting the product's own disclosures — was live in
the default configuration and survived two verification cycles, including a
browser journey and 59 HTTP integration tests. It survived because no API key
was configured on the verifying machine, so every run took the
deterministic-fallback branch and never executed the merge at all. The
disclosures were genuinely verified, on the one path where nothing could
remove them.

Defect 34 — the cache crashing every request on a read-only filesystem — was
a **P0 that could only ever fire in production**, and it survived *every* test
ever run against this product. Not because the tests were weak, but because a
developer's directory is writable and a container's root-owned application
tree is not. There was no assertion to add; there was a condition to
reproduce.

**The rule, generalised: a code path that behaves differently because an
optional dependency, a permission, or a filesystem differs in production is
not a verified path, and a green suite says nothing about it.** Two practical
consequences for anyone working on this codebase:

1. Any provider or external boundary — a new explainer, a broker, a data
   vendor — needs at least one run with that boundary actually exercised, even
   if the far side is a stub.
2. Any environmental assumption — writable paths, available binaries, network
   egress, user identity — needs a test that **reproduces the hostile
   condition** rather than mocking the error it produces. `chmod`-ing a real
   directory found defect 34; a mocked `OperationalError` would have passed
   against the broken code.
3. Any artefact whose behaviour is defined by a **tool's** semantics rather
   than by this codebase's — a `.dockerignore`, a `Dockerfile`, a CI matrix, a
   dependency resolution — must be verified by running that tool, not by
   reimplementing its rules in a test. Five assertions reimplemented Docker's
   matching incorrectly and all five passed. Where a test must model such
   semantics, it should be **calibrated against the real tool first**:
   `backend/tests/fixtures/dockerignore.py` exists only because two throwaway
   builds established what the rules actually do.
4. "No runtime is installed" is a statement about the present, not a
   constraint. Before recording a verification as impossible, check whether the
   missing thing can be *installed* — here it could, in minutes, and the
   verification it unlocked was the one that found the P0.

---

## 9. Roadmap

| Priority | Item | Rationale |
|---|---|---|
| ~~P1~~ | ~~Explain or bound the single-stock correlation weakness~~ | **Closed 2026-10-29 — it was never a defect.** See §7.2. The product's return series is byte-identical to an independent recompute from the raw provider JSON (`maxdiff = 0.00e+00`), and the window was the whole story: over 500 sessions AAPL/SPY gives R²=0.354, ρ=+0.595, β=1.03 against 0.083/+0.288 over the 252 the product fits on. Real mega-cap dispersion, correctly measured and correctly gated — SPY is *rejected* as an AAPL cross hedge at R²=0.084 < `MIN_CROSS_R2 = 0.30`. **This row contradicted §7.2 for a day; the roadmap was not updated when the item closed.** |
| ~~P1~~ | ~~Commit the working tree~~ | **Done 2026-10-29/30** — `732cd56` (frontend), `e6a6e51` (docs), `b9e093b` (the `.dockerignore` P0). On `repair/end-to-end-phase-2`, **local only: nothing pushed, nothing deployed.** |
| ~~P1~~ | ~~Build and run the container image once~~ | **Done 2026-10-29.** Podman installed, image built `--platform linux/amd64` and run: 11/11 checks after the fix. It found defect 37 — five live API keys in the image, because every `.dockerignore` rule aimed inside `backend/` matched nothing and five test assertions checked the rules' text rather than their effect. Now 668MB, with a build-time guard that fails the build on any future leak |
| **P1** | Deploy to Cloud Run once, behind approval | The image is verified; the platform is not. Networking, memory limits, cold start and the real service account are untested, and `OPTIONQ_CACHE_DB` must point at a writable volume there |
| **P1** | Move the container build into CI or Cloud Build | The 2026-10-29 verification was real but is **no longer reproducible here**: podman was removed from the machine on 2026-10-30. A verification that depends on a tool somebody has to remember to install, on a laptop, is a verification that will quietly stop happening — and the one time it ran it found a P0. It belongs in a pipeline that builds `--platform linux/amd64` on every change and runs the Dockerfile's leak guard, so neither the build nor the guard depends on anyone's local setup |
| **P2** | Exercise the Claude explainer inside the container | The deterministic fallback is verified and discloses itself correctly; the LLM path in the container is not. Blocked on the item above — there is no container runtime on this machine now |
| ~~P0~~ | ~~Resolve the no-valid-quote question~~ | **Done 2026-10-29** — a conditional, self-calibrating refusal that distinguishes an absent market from a closed one. §7.2 |
| ~~P0~~ | ~~Re-run the full end-to-end verification during US market hours~~ | **Done 2026-10-29** — `market_mid` + `contract_iv` observed on live two-sided quotes |
| ~~P0~~ | ~~Stop the explanation layer from deleting the product's disclosures~~ | **Done 2026-10-29** — found while closing the Layer 5 item; proven both ways over the live HTTP route. Defect 31 |
| ~~P1~~ | ~~Expose risk provenance to the client~~ | **Done** — `RiskEstimateProvenance` travels with every recommendation |
| ~~P2~~ | ~~Verify Layer 5 / stale-results display~~ | **Done 2026-10-29** — §7.2 |
| **P2** | Historical hedge backtesting | Turns "this reduces modelled VaR" into "this reduced realised drawdown" |
| **P2** | Portfolio persistence and change tracking | |
| **P3** | Paper-trade execution via Alpaca | Only after the above, and only with explicit approval |

---

## Appendix A — Operating notes for maintainers

- **Always use the virtual environment.** `backend/.venv`, never system
  Python.
- **Frontend is Next.js 16.2.3 / React 19.2.4 / Tailwind v4.** Per
  `frontend/AGENTS.md`: *"This is NOT the Next.js you know."* Read
  `node_modules/next/dist/docs/` before writing frontend code.
- **The sqlite cache survives deploys.** A semantic change to a cached value
  needs a cache-key version bump.
- **`data_quality.record()` is a no-op without an active ledger**, so it is
  safe to call from anywhere.
- **OCC symbols encode strike in thousandths:**
  `sym.endswith(f"{int(round(strike*1000)):08d}")`.
- **Cron routes are GET, not POST:** `/admin/cron/retrain-fast`,
  `/admin/cron/retrain-main`.
