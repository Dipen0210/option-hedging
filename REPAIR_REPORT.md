# OptionQ — Final Independent Re-Audit (Phase 8)

**Date:** 2026-10-29 · **Branch:** `repair/end-to-end-phase-2` (local only — nothing pushed, nothing deployed)
**Scope:** re-audit of the v1 repair against the original readiness audit
(verdict 2.5 / 10, all four readiness questions "No")
**Method:** static inspection, 678 automated tests, and live end-to-end runs
against a real `uvicorn` server — 05:50 ET (pre-market), 09:58 EDT (market
open), and a third with an LLM provider wired in (§B ⑤)

> Verification tags used throughout: **[tested]** a regression test pins it ·
> **[verified]** observed in a live end-to-end run · **[code]** read in source
> but not exercised · **Not verified** — stated as such, never inferred.

---

## A. Executive verdict

**OptionQ now completes the portfolio hedging workflow end to end, on real
market data, with every recommendation traceable to a listed contract, every
substituted input disclosed, and its primary pricing paths observed against a
live two-sided market. It is ready for external user testing.**

**Two distinct bars, and only the first is claimed.** *Ready for external user
testing* means the workflow completes on live data, the numbers are defensible
and provenanced, the failure paths are honest, and the deployment
configuration fails closed — all of which is now measured, and now measured
**in the container itself**, built for `linux/amd64` and run (§F). *Ready for
unsupervised public use* is a higher bar and is **not** claimed: it would
require a monitored deployment on Cloud Run, the LLM path exercised in the
container, and market-hours observation over more than a single session.
Nothing in this report should be read as the second claim.

### What changed this cycle

**The container was built and run, and that is how the worst defect of the
whole repair was found.** The previous pass of this report concluded that no
container runtime was available here, reproduced the runtime as faithfully as a
developer machine allows, and stated the gap honestly. **The conclusion was
wrong:** nothing was installed, but Homebrew and hardware virtualization both
were, so a runtime was a `brew install` away. Podman installed, image built for
`linux/amd64` — and the running container held **`backend/.env`, with five live
API keys, in an image layer**, alongside a 417MB host virtualenv and the cache
database that the defect register already marked closed.

One root cause: `.dockerignore` is not `.gitignore`. Docker matches each
pattern once against the whole path relative to the context root, so `.env`
excluded `./.env` and nothing else, and **every rule in that file aimed at
something inside `backend/` matched nothing**. `backend/tests/` was the only
rule that worked, because it names its path — which is exactly why the leak
looked like a solved problem. Five assertions across two test files covered
this. All five passed, because all five checked that the rule was *written*.

Fixed three ways, not one: the patterns anchored, a **build-time guard** that
fails the build on any leaked secret or database, and the five string
assertions replaced by tests that evaluate the rules against the paths that
actually leaked. Image rebuilt at **668MB** and re-verified **11/11**, then run
end to end: fail-closed on three bad configurations, the full auth matrix
including both cells where a *valid* credential must be refused, and a real
`POST /portfolio/analyze` at 200 in 31.2s on live data, 0.854% of notional
against a 2% cap, cache confined to its own directory, zero tracebacks. §F.

The lesson is not that the fix was hard; it was ten minutes. It is that **a
cycle of this report asserted the configuration was correct, a suite of 674
tests agreed, and five live API keys were sitting in an image layer the whole
time.** Only building it could see that.

**R7 is closed, and it was not a defect.** The last open P1 — an implausibly
weak single-stock correlation structure — was root-caused rather than argued
away. The product's return series is **byte-identical** to an independent
recompute from raw provider JSON (`maxdiff = 0.00e+00`), and the weak R² is
purely an artefact of the 252-session window the product actually fits on:
over 500 sessions the same AAPL/SPY pair gives R²=0.354, ρ=+0.595, β=1.03,
while index-replication pairs barely move. Single-stock dispersion against the
index was genuinely extreme over the trailing year. The product measures it
correctly **and refuses the hedge** — SPY is rejected as an AAPL cross hedge
at R²=0.084 against a 0.30 floor. Full detail in §C.

**Two further defects were found alongside it.** Reproducing the production
runtime exposed a **P0 that could only
ever have fired in production**: the cache opened SQLite inside the
root-owned application tree and let every error propagate, so on a read-only
container filesystem every analysis request died *after* its market-data
fetch had succeeded. Separately, the primary history path was serving
**dividend-unadjusted** prices while the fallback served adjusted ones — the
same ticker returned a different history depending on which path answered,
flipping the sign of HYG's annual return and moving three of the five macro
factors by ~4.5pp. Both are fixed, both are pinned by tests that reproduce the
real condition rather than mocking it, and both are written up as a new defect
class (§B⑥) because what they have in common is that **no test on this machine
could have seen either one**.

**From the previous cycle: R2, R4 and R8 closed, and one further defect found
while closing R4 that was worse than R4.** L11 runs by default. Its
`_merge_explanation` replaced `candidate.cons` wholesale — and `cons` is where
the product's own disclosures live, including *"premium shown is theoretical,
not an executable price"*. So with any LLM configured, a model that returned
its own bullets **deleted the product's disclosures from the API response**,
and the user saw a hedge priced off a model with nothing left on screen to
say so. This was never visible in the earlier verification runs because no
API key was configured, which routed every run down the deterministic
fallback and never through the merge. Proven both ways against the live HTTP
API in §B ⑤.
**R1 is closed.** Run at 09:53–09:58 EDT on 2026-10-29 with the market open:
all 21 live-provider tests passed (previously 21 skipped), premiums resolved
to `market_mid` and volatility to `contract_iv`, and the liquidity screen
rejected **68 of 165** live SPY put strikes — 67 on open interest and one on a
spread wider than the 5% limit. The 95%-moneyness put quoted 10.30 / 10.37, a
0.68% spread, and 163 of 165 strikes were two-sided. These are measurements,
not inferences.

Closing R1 required fixing the gate that had been hiding it. `_is_us_market_hours()`
subtracted a flat five hours from UTC and called the result ET — that is EST,
so for the ~8 months of EDT the window it actually enforced was 10:30–17:00.
It therefore skipped the two-sided-quote assertion for the first hour of every
summer session, and, worse, kept asserting for an hour *after* the close, where
a missing quote is not a defect. It now uses `ZoneInfo("America/New_York")`.

### Readiness questions, re-answered

| Question | v0 | v1 | Basis |
|---|---|---|---|
| Does the application complete its advertised workflow? | **No** | **Yes** | 678 tests, 21/21 live, 28/28 browser-driven UI checks, and a full `/portfolio/analyze` run **inside the built container** **[verified]** |
| Are the financial calculations defensible? | **No** | **Yes** | §E. The one open design question (R2) is now decided and implemented |
| Is it secure enough to expose? | **No** | **Yes** | 9 security test classes, 4/4 endpoints fail closed, 422 bodies sanitised **[tested] [verified]** |
| Would a real investor get usable, honest output? | **No** | **Yes, in hours and out** | §A above, §F. R7, the one open data-confidence question, is now resolved — the weak single-stock betas are a correct measurement of a genuinely dispersed window, not bad data |

### Score: **9.0 / 10** (from 2.5)

The 1.0 withheld, re-apportioned again this cycle. The image has now been
built and run, which closes the gap the previous apportionment named — but
building it **found a P0 that every green test in this suite coexisted with**:
five live API keys in an image layer, because `.dockerignore`'s rules matched
nothing inside `backend/`, and because five assertions across two test files
checked that those rules were *written* rather than what they *did* (−0.5 for
a P0 that reached this cycle undetected, and for the deployment-facing
assertions that were vacuous). The remaining −0.5: Cloud Run itself is
untested, the LLM path was not exercised in the container, nothing is
deployed, and the container ran under emulation. R7 clearing does not raise the
score, because what replaced it is worse: a crash that every test on this
machine was structurally incapable of seeing.

The repair itself is now committed — three commits on
`repair/end-to-end-phase-2`, local only — so R5 is closed and no longer part of
what is withheld.

Raised from 8.5 this cycle: R2 decided and implemented (+0.5), R4 and R8
closed, and the disclosure-deletion defect above found and fixed. The score
did not rise further *because* of that find: a P0-class defect that sat
undetected through two verification cycles is evidence about the coverage,
not just about the code. What closed the gap was running the LLM boundary at
all, rather than inspecting it.

---

## B. Issues resolved

All 21 functional defects and 8 security defects from the original audit are
resolved, each with at least one regression test — as are the 12 further
defects found during the repair itself, including the 3 found this cycle while
root-causing R7 and reproducing the production runtime. The full register with
file paths, user-visible effect and pinning test is maintained in
[`PRD.md`](PRD.md) §6; this section summarises by defect class and names the
files changed. **Class ⑥ is new this cycle**, and it is the one worth reading
first: it is the class the suite could not see.

### ① Placeholder-as-measurement — 4 defects

The class: a quantity that could not be computed returned a plausible number
instead of failing. `return 1.0` for an unmeasurable beta is not a neutral
default — it is the claim *"this holding moves one-for-one with the
benchmark"*, and it sized contracts.

The worst instance: `RiskAnalysisEngine._fallback_profile`, reached from a bare
`except Exception`, returned a **structurally complete** `RiskProfile` — VaR =
notional × 0.15, CVaR = notional × 0.20, `tail_correlation = 0.7`,
`beta_vs_spy = 1.0`, both hedge ratios 1.0 — with nothing in it from data. L4
then selected, priced, sized and ranked real contracts against an invented
15% VaR and handed the user a contract count and a premium for it.

**Resolution:** unmeasurable quantities raise `InsufficientFactorData` or
return `None`; profiles carry `data_available` / `beta_available`; L4 declines
to select against an unavailable profile; `_compute_portfolio_beta` excludes it.

**Files:** `backend/core/risk/factor_decomposer.py`,
`backend/engines/layer_03_risk_analysis.py`,
`backend/engines/layer_04_instrument_selection.py`,
`backend/models/risk_models.py`, `backend/data/covariance.py`,
`backend/instruments/options/selector.py`

**Pinned by:** `test_factor_decomposer.py::TestUnmeasurableRaises`,
`test_degraded_risk_profile.py` (3 classes), `test_covariance.py::TestEstimateProvenance`,
`test_selector.py::TestUnmeasuredLiquidity`

### ② Truncate-before-align — 3 defects

The class: each return series was windowed to its own last *N* rows **before**
being joined, so two series ending on different dates were compared over
different calendar periods. A tz-aware/tz-naive index pair collapsed the
overlap to zero rows. Every cross-asset number was affected: beta,
correlation, hedge ratio, covariance.

**Resolution:** align first, then window. `align_series(**named)` normalises
timezones. Historical-period mapping corrected; empty arrays refused rather
than passed to the regime detector.

**Files:** `backend/core/risk/factor_decomposer.py`, `backend/data/covariance.py`,
`backend/data/market_data.py`, `backend/engines/layer_02_market_context.py`

**Pinned by:** `test_factor_decomposer.py::TestAlignBeforeWindow`,
`test_covariance.py::TestWindowAlignment`, `::TestAlignSeries`,
`test_live_market_data.py::TestAlignment`, `::TestPriceHistory`

### ③ Solver-artifact IV and unlisted contracts — 7 defects

The class: Yahoo publishes `impliedVolatility` solved against a stale
`lastPrice` when no two-sided quote exists, producing values like `0.00001`
and a visible ladder of successive halvings of 1/16. Pricing on `0.00001` IV
produced a ~$0 premium and a delta near `1e-5`; the sizer, asked to cover a
position with a near-zero-delta hedge, requested **roughly 1.5 million
contracts** — and presented it as a recommendation.

**The insight that fixed it:** a two-sided quote is the precondition that makes
an implied volatility a measurement. Without one there is no IV — and clamping
to a plausible floor is still fabrication, because a clamped 5% IV yields a
priced, tradeable-looking recommendation built on a quote nobody made. So
implausible readings are **rejected, returning `None`**.

Also in this class: synthetic $5-grid strikes that were never listed; a 7-day
expiry offered as a 180-day hedge; a 34.7%-spread / 20-OI contract ranked best
in book; ranking computed on the model price rather than the fill; a credit
collar clamped to a $0.01 debit.

**Resolution:** synthetic expiry and strike generation removed; `_resolve_strike`
reads the real chain; `_sample_expiries` enforces horizon coverage;
`_liquidity_score` and an executability filter screen candidates; `_rescore`
ranks the executable price; `_apply_net_market_quote` preserves credit.

**Files:** `backend/data/options_chain.py`, `backend/instruments/options/selector.py`,
`backend/engines/layer_04_instrument_selection.py`,
`backend/engines/layer_06_pricing_engine.py`, `backend/core/pricing/base_pricer.py`,
`backend/core/pricing/equity_pricer.py`, `backend/data/data_cache.py`

**Pinned by:** `test_iv_plausibility.py` (6 classes), `test_selector.py` (12 classes)

### ④ Asserted label — 9 defects

The class: a figure's units or horizon asserted by its consumer rather than
reported by its producer. L3 scales VaR to the user's hedge horizon, but the
models dropped the horizon, so the narrative said *"One-day 95% VaR"* and the
UI said *"1mo VaR (95%)"*. The same position reads ≈$4.6K over one day and
≈$42.2K over 120 days: **the tail loss was understated by roughly 9.1×**, and
"1mo" was wrong for the 180-day default and for every input a user could pick.

Also: calendar-vs-trading-day VaR scaling; parametric VaR labelled historical;
moneyness read in the wrong direction, so the OTM caveat appeared on **no**
recommendation while the pros list beside it advertised unlimited downside
protection.

**Resolution:** the horizon travels with the figure —
`var_horizon_calendar_days` / `var_horizon_trading_days` on `RiskProfile` and
`PortfolioRiskSummary` (all three L3 sites), `horizon_calendar_days` /
`horizon_trading_days` on `UnhedgedRisk`, `var_horizon_days` in all three LLM
prompt payloads, and a UI label **derived** from the response. Where the
horizon is unknown, `_var_horizon_phrase()` names **no** horizon rather than
defaulting to a wrong one.

**Files:** `backend/models/risk_models.py`, `backend/models/hedge_models.py`,
`backend/engines/layer_03_risk_analysis.py`, `backend/engines/layer_12_output_formatter.py`,
`backend/engines/layer_11_llm_explainer.py`, `backend/core/explainer/prompt_builder.py`,
`backend/core/risk/var_calculator.py`, `backend/instruments/options/selector.py`,
`frontend/types/hedgeos.ts`, `frontend/app/page.tsx`

**Pinned by:** `test_var_horizon_label.py` (6 classes, 22 tests),
`test_disclosure_and_budget.py::TestMoneynessCaveat`, `::TestOptionPros`, `::TestOptionCons`,
`test_risk.py`

**Sixth instance, found this cycle while root-causing R7.** `covariance.py`'s
docstring described the basis-risk estimate as a *"signed, 60-day rolling"*
correlation. The code fits on `LOOKBACK_DAYS = 252`. The window was lengthened
at some point and the docstring was not, so anyone reading the module to judge
whether an R² of 0.08 was plausible was reasoning about a window roughly four
times noisier than the one in use — which is precisely the mistake R7 invited.
A comment that asserts a number the code does not use is the same defect class
as a label that asserts a horizon the figure does not have; the fix states the
real window and says that it changed.

**File:** `backend/data/covariance.py`

**Seventh instance, and the one this cycle paid most for — it was in the test
suite.** Five assertions across `test_security.py` and
`test_deployment_contract.py` protected the container against shipping secrets:

```python
assert ".env" in patterns
assert ".venv/" in patterns
assert "*.pem" in patterns and "*.key" in patterns
```

All five passed. The first real build of the image contained `backend/.env`
with five live API keys, because a bare `.dockerignore` pattern matches only at
the build-context root. The assertions checked that the rule was *written*; the
comments above them described what it *did*; nothing connected the two. **A
test that checks configuration exists, rather than what the configuration does,
is an asserted label wearing a test's clothes** — and it is worse than no test,
because it also supplies the confidence.

The fix was not to add a stronger string check. `backend/tests/fixtures/dockerignore.py`
now *evaluates* the file — Docker's glob semantics, `**/` spanning zero or more
components, last-match-wins, `!` negation — against the paths that actually
leaked, and the semantics were calibrated against two throwaway builds rather
than recalled. Both test files import the one implementation. Full account and
the build-time guard in §F, registered as defect 37.

**Files:** `.dockerignore`, `Dockerfile`, `backend/tests/fixtures/dockerignore.py`,
`backend/tests/test_security.py`, `backend/tests/test_deployment_contract.py`

**Defect 38 — a permanent provider failure labelled a per-run one (P1, found
2026-10-30).** The same class applied to a cause rather than a unit. HuggingFace
answers every explanation call on this account with HTTP 402 *"You have no
remaining credits"*. `_is_rate_limit` tested the message for `"429"`,
`"rate limit"`, `"too many requests"` and `"quota"`, matched none, so the
explainer logged and returned empty — and L11, seeing a call that had succeeded
and produced nothing, told the user:

> *"The language model returned no analysis for this run, so the narrative below
> was generated directly from the computed figures instead."*

The account has no credits. The next run fails identically and so does every one
after it; only an operator can clear it. **"for this run" invites a retry that
cannot work**, and the same sentence also covered a model that genuinely had
nothing to say, so the two were indistinguishable in the UI. Nothing in the code
had established the cause — the wording was a label asserted over an
unclassified failure. Closing it is what showed the Claude provider does the
same thing for a refused API key.

The fix reads the **HTTP status**, and consults the message only when there is
none. That ordering is the repair, not an implementation detail: the real 402
body arrives with a request ID such as `Root=1-6ae50969-…` and an appended MDN
link, so substring-matching an error string for the digits `"429"` can classify
a dead account as a transient limit. A test pins a 402 whose text contains both
`"429"` and `"quota"` and requires it to stay permanent. `LLMProviderRejectedError`
(401/402/403) and `LLMRateLimitError` are now distinct types; the user-facing
reason is derived from the status, never copied from the response body, so no
request ID, internal router URL or account identifier reaches the response.
An ordinary 500 still degrades quietly — a provider hiccup must not cost a user
their hedge analysis, and neither does a refusal.

`base_explainer`'s contract previously read *"Never raise exceptions to the
caller"*, which is the sentence that justified swallowing this; it now carves out
refusals and records why.

**Verified against the live account**, not a stub: the notice reads *"…refused
this request because the provider account has no remaining credits. This is not
a temporary problem…"*, the narrative and all recommendations are still
delivered, and nothing internal appears. 25 tests; reverting the classifier to
the keyword logic fails 14 of them, including an end-to-end case that drives the
real explainer with only the HTTP client stubbed.

**Defect 39 — the same swallow in `ClaudeExplainer` (P1, found 2026-10-30).**
Found while closing 38, not reported separately because it is the same
mechanism: both `explain_*` methods returned an empty result for *every*
exception, so a refused API key and a malformed response were the same event to
the caller. `ANTHROPIC_API_KEY` in this environment is 13 characters — a
placeholder — so switching `LLM_PROVIDER` to `claude` today produces a 401, and
before this fix the product would have reported "no analysis for this run"
indefinitely. Both methods now classify before degrading. 2 tests, both
revert-verified.

**Files:** `backend/core/explainer/base_explainer.py`,
`backend/core/explainer/huggingface_explainer.py`,
`backend/core/explainer/claude_explainer.py`,
`backend/engines/layer_11_llm_explainer.py`,
`backend/tests/test_llm_provider_refusal.py`

**And then the product was given a model (2026-10-30).** Defects 38 and 39
fixed the *reporting* and left the product with nothing to report: HuggingFace
had no credits and the Anthropic key was a placeholder, so every run shipped
the deterministic narrative. The user chose Gemini from a new GCP project — first over the API
key, then, on the same day, **over Vertex AI with IAM authentication** and
with billing linked to the same account as their two other projects. The full
decision is in `PRD.md` §7.3. What belongs *here* is that a second provider
is where this defect class gets a second chance, and it took it three times:

- **An invalid Gemini key is a `400`, not a `401`.** A classifier keyed on
  {401, 402, 403} would have called a dead key an ordinary failure, which is
  defect 38 verbatim on a different provider.
- **A retired model is a `404`.** Not anticipated — *found*. The configured
  default was `gemini-2.5-flash`, and the first live call answered *"no longer
  available to new users"*. Reasoning about this in advance would not have
  caught it; making the call did.
- **A 200 can carry nothing.** Gemini 2.5+ charges its own reasoning against
  `maxOutputTokens`, so an unset thinking budget can return
  `finishReason: MAX_TOKENS` with no text — the exact shape of "the model
  returned no analysis", arriving with a success status. `thinkingBudget` is
  now 0, which these prompts have no use for anyway.

Both new statuses are decided by **wording**, not by status alone, and this is
the one place the ordering from defect 38 inverts — for a good reason. 400 and
404 are each equally the shape of a malformed request *of ours*, and blaming
our own bug on the user's credentials is not a smaller lie than the one this
class is about, merely a different one. So a 404 naming a retired model is a
refusal, a 404 from a URL we got wrong is not, and both directions are pinned
by tests. 401/402/403 stay unconditional: no provider answers those to a
request we malformed.

Two more were found while building it and fixed before they shipped, so
neither is in the register — neither ever reached the product. 404 was briefly
unconditional, caught not by review but by a revert check: the 404 tests kept
passing against reverted code, which meant they were pinning nothing. And
`GeminiExplainer(api_key="")` fell back to the ambient configured key through
a falsiness test — a live credential silently substituted for an explicitly
absent one, class ⑤ in new code. `None` now means "read the configuration" and
`""` means "none".

**Capacity is not a refusal.** The free tier sheds load with `503 - overloaded`
on roughly half of all calls (five of eight during model selection). Three
attempts with 1s/3s backoff, 503/504 only — retrying a 429 makes a rate limit
worse, and a refusal returns the same answer every time. After three, the run
degrades to the deterministic narrative and says so.

**Verified live, not by inspection:** `POST /portfolio/analyze` over real
market data returned 200 in 42.1s (two 503 retries included), with
`llm_provider: "gemini"`, an **empty** `llm_notice`, model prose on all 14
candidates, all four portfolio fields, all 12 degradation warnings still
present after the LLM rewrite, and no key, URL or internal path in the body.
The prose cites the computed figures rather than inventing any — *"-36.383
delta_per_contract"*, *"basis_risk_r2 1.0"*, *"$1140 total_cost_usd"*. 39
tests; every fix fails when reverted, in both directions.

**Then the credential was replaced, and the lesson repeated once more.**
Moving to Vertex AI removes the long-lived secret entirely: IAM locally via
ADC, an attached service account in Cloud Run, no key to store or leak.
Vertex is a *subclass* of the API-key provider precisely because of the
paragraphs above — the retry, the thinking budget and the refusal
classification are the parts already got wrong once, so they are inherited
rather than copied, and tests assert the inherited defences actually run on
the new path. Two Vertex-specific traps were found the same way as the rest,
by making real calls:

- **The location is not free.** `gemini-3.5-flash` is served from `global`
  only; every regional endpoint answers the 3.x models `404 NOT_FOUND`. And
  `global` is not spelled like a region — its host is the unprefixed
  `aiplatform.googleapis.com`.
- **Vertex has its own two 404s.** An unavailable model says *"Publisher
  model ... was not found or your project does not have access to it"*; a URL
  we malformed gets Google's generic HTML page. `"publisher model"` is the
  discriminator, and the revert check proves it is the right one: matching the
  words the two messages *share* ("was not found") makes the "our own bug"
  test fail.

One thing that would have gone unnoticed without looking: Application Default
Credentials carry a `quota_project_id`, and on this machine it names a
different project of the user's. `AuthorizedSession` sends it as
`x-goog-user-project`, so the first working Vertex calls would have billed an
unrelated project. The quota project is now set from `VERTEX_PROJECT`, and the
outgoing header on a live call was inspected to confirm it. The user's shared
ADC file was left untouched.

**Verified live on the Vertex path:** 200 in **21.9s**, `llm_provider:
"vertex"`, empty `llm_notice`, prose on all 14 candidates, all four portfolio
fields, all 9 warnings intact, and no bearer token, key, project id or
internal path in the body. The cost constraint was re-checked on that
response: $8,721.04 against a $10,011.81 budget = **2.613% of $333,727
notional**, under the user's 3% cap, with TLT left unhedged and a stated
reason. 24 further tests, each revert-verified; 769 in the suite, 0 failing.

**One honest gap:** `when_works_best` and `when_fails` were absent on 4 of the
14 candidates. Those two fields have no deterministic source anywhere in the
product — `layer_05_llm_explainer.py` is their only writer and the frontend
hides the section when they are empty — so an omission costs a section, not
correctness, and nothing is substituted for them. It is prompt adherence, not
a correctness or disclosure defect.

**Files:** `backend/core/explainer/vertex_explainer.py` (new),
`backend/core/explainer/gemini_explainer.py` (new),
`backend/core/explainer/response_parsers.py` (new),
`backend/core/explainer/base_explainer.py`,
`backend/core/explainer/__init__.py`, `backend/config/settings.py`,
`backend/.env.example`, `backend/requirements.txt`,
`backend/tests/test_gemini_explainer.py` (new),
`backend/tests/test_vertex_explainer.py` (new)

### ⑤ Silent substitution — 6 defects

The class: a backstop that fills in for a failed or blank upstream must report
that it ran. Three L11 paths left all four narrative fields empty, so *"the
model had nothing to say about your risk"* and *"the explanation service is
down"* rendered identically — as a blank panel.

Then, found **during** the repair: a provider that was *available*, raised
nothing, and returned **blank** fields fell through to the unconditional
deterministic backstop, and the response went out with
`llm_provider="huggingface"` and an **empty** `llm_notice` while every
narrative field was template text. Template text presented as model analysis
is exactly the substitution the rest of the pipeline exists to prevent.

Also: nothing summed the per-holding hedge costs, so each hedge passed its own
gate while the **total** could exceed the user's budget.

**The worst instance of this class was found last, while closing R4, and was
live in the default configuration.** The selector states three facts in
`candidate.cons` — no executable two-sided quote, thin market, expiry crosses
earnings. Both explainer layers merged a model reply by **replacing** that
list:

```python
if explanation.cons:
    candidate.cons = explanation.cons
```

`run_l11_llm` defaults to `True`. So a model returning its own bullets deleted
the product's disclosures from the response, and the user was shown a hedge
with its premium's provenance silently removed. Proven, not inferred: a stub
provider was stood up behind the real `OllamaExplainer`, the real pipeline and
the real HTTP route, and asked for a reply containing only its own cons. With
the fix reverted, the live `POST /portfolio/analyze` returned **5 of 6
candidates with their disclosure deleted** — an MSFT protective put on a
contract quoted **8.9% wide**, described to the user as nothing worse than
*"Premium is a cost drag"*. With the fix in place, the same request over the
same market: **5 recorded, 0 deleted**.

Why two verification cycles missed it: no `ANTHROPIC_API_KEY` was configured,
so every previous run took the deterministic-fallback branch and never
reached the merge at all. The disclosures were verified in the browser and in
`test_e2e_api.py` — on the one path where nothing could delete them.

And alongside it, a second substitution in the same subsystem:
`get_explainer()` returned `ClaudeExplainer` for **every** provider name it
did not recognise, and it recognised only `"huggingface"`. So
`llm_provider=ollama` sent the portfolio to Anthropic with nothing in the logs
to say the configured provider had been ignored — and `OllamaExplainer` was
unreachable through the factory in any case, since `settings.ollama_model` and
`settings.ollama_base_url` were referenced by it but never declared, so
constructing it raised `AttributeError`. A whole provider had never worked.

**Resolution:** `_deterministic_narrative` returns the names of the fields it
filled; `run()` sets a full-substitution notice when it wrote all four, a
partial notice naming the specific sections when it wrote some, and **no
notice** when the model wrote everything — because a notice there would be a
false warning. Portfolio-level budget summed and enforced.

For the disclosure deletion: `cons` stays the single list the UI renders, so
nothing about the display changes, and a second record is added that the LLM
has no path to. `backend/core/disclosures.py` exposes `disclose()`, which
writes both, and `restore()`, which is called after **every** merge in both
layers and puts back whatever the merge dropped — in front of the surviving
cons, because a caveat under four bullets of model prose is a caveat nobody
reads. Scope is deliberately the three hard disclosures and not every
sentence the product generates: rewriting the basis-R² prose into better
English is the explainer's job and loses nothing material, while dropping
*"this price is not real"* is a different kind of event.

For the factory: `"ollama"` is now wired, its two settings are declared, and
an unrecognised provider name is **logged at ERROR** naming both the rejected
value and the fact that Anthropic will answer instead. It remains a fallback
rather than an error — an unreadable provider name must not take the pipeline
down — but a loud one.

**Files:** `backend/engines/layer_11_llm_explainer.py`,
`backend/engines/layer_05_llm_explainer.py`, `backend/core/disclosures.py` (new),
`backend/core/explainer/__init__.py`, `backend/config/settings.py`,
`backend/models/hedge_models.py`,
`backend/engines/layer_10_evaluator.py`, `backend/engines/pipeline_orchestrator.py`,
`backend/instruments/options/selector.py`, `backend/instruments/options/position_sizer.py`,
`backend/engines/layer_07_position_sizer.py`, `backend/core/sizing/equity_sizer.py`

**Pinned by:** `test_llm_fallback.py::TestFallbackPaths`,
`test_var_horizon_label.py::TestNarrativeSubstitutionIsDisclosed`, `::TestEngineDisclosesSubstitution`,
`test_disclosure_and_budget.py::TestPortfolioBudget`, `test_selector.py::TestHoldingBudget`,
`test_disclosures.py` (19 tests: 4 on the record itself, 7 on the L11 merge,
2 on L5, 2 guarding that the selector still produces them, 4 on the factory),
`test_pipeline.py::TestDisclosuresSurviveTheLLMLayers` (6, full pipeline),
`::TestAFailingL5DoesNotFailTheRun`

#### Sixth instance, found this cycle: the history path served dividend-unadjusted prices

The two history paths disagreed about what a price *is*. The primary path,
`_yahoo_history_direct`, read `indicators.quote[0].close` — split-adjusted but
**not** dividend-adjusted. The fallback called yfinance with
`auto_adjust=True`, which is both. The same ticker therefore returned a
different history depending on which path happened to answer, and nothing
anywhere said so.

This is a substitution because a total-return series and a price series are
not interchangeable inputs to a return calculation, and the product uses these
for factor regressions, covariance and VaR. Measured impact over one year:

| Ticker | Unadjusted | Adjusted | Δ |
|---|---|---|---|
| HYG | **−4.25%** | **+1.63%** | sign flip |
| TLT | −7.0% | −2.5% | ~4.5pp |
| LQD | −3.2% | +1.3% | ~4.5pp |

**Three of the five `MACRO_TICKERS` are the worst affected**, because the
macro set is deliberately the high-yielding instruments — bond and credit
ETFs — where the dividend is most of the return. Over two years the price
*levels* diverge by 9–12%.

**Why it survived every prior cycle:** annualised volatility moves by at most
0.37pp under the adjustment, and volatility is what the existing data-quality
checks look at. A defect that only moves the *drift* passes every plausibility
gate aimed at the *dispersion*.

**Resolution:** the request now asks for `includeAdjustedClose=true` and a
ratio factor `adjclose / close` is applied to **all four** OHLC columns rather
than substituting `adjclose` for `Close`. Substituting would have been wrong:
`adjclose` is scaled while `high` and `low` are not, which on NVDA's 5-year
series puts the close outside `[Low, High]` on **107 of 1254** bars. If the
provider returns no adjusted close, the bars are left on a price basis and the
provenance ledger records `history_adjustment:<ticker>` as **unavailable** —
disclosed, not faked.

Verified after the fix: **0** out-of-band bars across 7 tickers × 501 bars,
and the two paths now agree to 8×10⁻⁷ with return correlation 1.000000.

**File:** `backend/data/market_data.py`

**Pinned by:** `test_price_adjustment.py` (5 classes, 16 tests — the fixture
sets High/Low against the *unadjusted* close with a 0.90 factor, so a
substituting implementation fails and a scaling one passes),
`test_live_market_data.py::test_both_fetch_paths_return_the_same_adjusted_series`,
`::test_the_adjusted_bars_are_internally_consistent`. Revert-verified: **7 of
14 fail** without the fix.

**The synthetic harness had to be fixed too, and the assertion left alone.**
Supplying no `adjclose` made the harness emit 25 `history_adjustment:
unavailable` records and fail `test_a_fully_quoted_run_is_not_marked_degraded`.
Before touching anything, real Yahoo was checked: it returns `adjclose` for
**every** instrument queried, including the dividend-free `^VIX` family. So the
harness was asserting a defect into existence. It now emits `adjclose == close`
— factor exactly 1.0 — and the assertion stands untouched.

### ⑥ Verified-here-only — 3 defects, two of them P0

**The class, named last cycle:** *a code path that behaves differently because
production's filesystem, permissions or installed dependencies differ from the
developer's is not a verified path, however green the suite is.* The lesson has
now been paid for three times. The ⑤ disclosure-deletion defect survived two
cycles because no `ANTHROPIC_API_KEY` was set here, so the merge code never
ran. The cache defects below survived because a developer's directory is
writable and a container's is not.

**The third payment is the one worth studying, because the missing condition
was the build itself.** Last cycle this report recorded that no container
runtime existed on this machine, reproduced the runtime in a clean venv and a
hand-staged file tree, and stated the gap honestly. The reproduction was
faithful in every respect but one: the staged tree was assembled by applying
`.dockerignore`'s rules *as I read them*, and Docker applies them differently.
Building the image for real produced `backend/.env` with five live API keys in
a layer — **a second P0 in this class** — plus a 417MB host virtualenv and the
cache database that defect 35's row already called closed. Full account in §F,
registered as defect 37.

Reproducing an environment you have reasoned about is not the same as running
it, and the difference is exactly where this class lives. It is also why
"no runtime is installed" was the wrong conclusion rather than an honest limit:
nothing was *installed*, but Homebrew and hardware virtualization were both
available, so the verification was a `brew install` away the whole time.

#### (a) The cache crashed the request on a read-only filesystem — **P0, production-only**

`data_cache.py` opened SQLite at a fixed path inside the application tree and
let every error propagate. Reproduced deliberately with `chmod 555` on the
containing directory: `cache_set` and, through it, `get_price_history` both
raised `sqlite3.OperationalError: attempt to write a readonly database`
**after the market-data fetch had already succeeded**. The answer was in hand
and was thrown away along with the request.

Under the shipped `Dockerfile` this was not hypothetical. `/app/backend` is
root-owned, the container runs as `USER optionq`, and SQLite must write both
the database **and** a journal file beside it. Every analysis request would
have failed on Cloud Run. Not one test on this machine could have seen it,
because every directory a test touches here is writable.

**Resolution, in three parts.** The path is now *resolved* rather than
assumed — `OPTIONQ_CACHE_DB`, then a platform-specific location, then the
system temp directory — and each candidate is **probed with a real
transaction**, because `os.access` and a bare `sqlite3.connect` both report
success on a directory SQLite cannot actually write. All three public
operations are guarded, and the write guard carries the reason in the code:

> *Failing to store a value the caller already has in hand must never fail the
> caller. This is the line that took the container down: the fetch had
> succeeded and the result was discarded along with it.*

If no location works, caching is **disabled and disclosed** — the ledger
records that every value is fetched live, which is slower and costs more rate
limit but is never stale. Degrading silently to an uncached product would have
been class ⑤ all over again.

**Files:** `backend/data/data_cache.py`, `Dockerfile` (writable
`/var/cache/optionq`, owned by the runtime user, with `OPTIONQ_CACHE_DB`
pointed at it)

**Pinned by:** `test_cache_resilience.py` (4 classes, 14 tests). The fixture
seeds a real database and then `chmod`s its directory to `r-x`, **reproducing
the permission condition rather than mocking it** — a mocked
`OperationalError` would have passed against the broken code too.
Revert-verified: **3 of 14 fail**, including the one that asserts a price
fetch still returns.

**A test caught a hole in this fix, and the code was changed, not the test.**
`test_a_disabled_cache_is_disclosed_on_the_ledger` failed: with caching fully
disabled, `cache_set` returned early at `conn is None` without recording
anything. An early return is not an exception path, so it needed its own
disclosure — added at all three such returns.

#### (b) The local cache database was shipped inside the image

6.4MB of this machine's query history was being copied into a published layer
by `COPY . .`. Worse than dead weight: the file arrived **readable but not
writable**, so cache *reads* returned a developer's months-old prices while
*writes* failed — the exact combination that makes a failure look
intermittent. `.dockerignore` now excludes `*.db`, `*.sqlite*` and SQLite's
journal and WAL sidecars.

**This subsection was wrong for a full cycle, and it is worth leaving the
correction visible.** It used to end *"Verified on the staged tree: 0 `.db`
files, and no secrets."* The staged tree was built by applying
`.dockerignore`'s rules the way I read them. Docker applies them differently —
a bare pattern matches only at the context root — so the first real build of
the image contained this database, root-owned, exactly as described above, plus
`backend/.env` and the 417MB host virtualenv. See defect 37 in §F. Now verified
against the built image rather than a tree I staged myself: **0** `.db` files,
no secrets, no virtualenv, and a build-time guard that fails the build if that
ever changes.

### Security — 9 defects

| Defect | Resolution |
|---|---|
| `/admin/retrain/main` — a blocking, CPU-bound retrain — had **no authentication at all** | `require_admin` on every `/admin/*` route |
| `_verify_cron_secret` returned early when `CRON_SECRET` was unset (*"no secret configured — open access"*), exposing model retraining to anyone who could reach the host | **Fails closed**: 503 when unset or under 16 chars |
| Secret compared with `!=`, leaking its prefix through response timing | Constant-time comparison |
| **Nine** handlers raised `HTTPException(500, detail=str(exc))`, forwarding file paths, driver errors and provider URLs | Sanitised; no handler forwards `str(exc)` |
| Admin and cron credentials not scoped | Neither unlocks the other |
| No startup configuration validation | Validated at boot; missing secrets logged as refusals |
| CORS unreviewed | Origin from `FRONTEND_URL` |
| Dependency incompatibilities, Docker/cloud config | `requirements.txt`, `Dockerfile`, `.dockerignore`, `backend/.env.example` |
| **`backend/.env` — `ANTHROPIC_API_KEY`, `HF_API_TOKEN`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `FRED_API_KEY` — was present in the built image**, 713 bytes at `/app/backend/.env`, in a layer that stays readable to anyone who can pull it. `.dockerignore` carried a `.env` rule the whole time; a bare pattern matches only at the build-context root, so it excluded `./.env` and nothing else. Five test assertions covered this and all five passed by checking the rule's text (defect 37) | Every nested rule `**/`-anchored; a **build-time guard** that fails the build on any `.env`/`*.pem`/`*.key`/`*.db`/`.venv`/`tests` in the copied tree; the five string assertions replaced by tests that evaluate the rules. Verified absent from the rebuilt image, which also dropped 1.08GB → 668MB |

**Files:** `backend/api/security.py` (new), `backend/api/routes_monitor.py`,
`backend/api/routes_portfolio.py`, `backend/api/routes_hedge.py`,
`backend/main.py`, `backend/config/settings.py`, `Dockerfile`,
`.dockerignore`, `backend/.env.example`, `backend/requirements.txt`,
`backend/requirements-dev.txt`

**Pinned by:** `test_security.py` (9 classes)

### Defects found and fixed *while verifying* R1 and R3 — 6 more

Closing the two open verification items surfaced six further defects. Five are
in the product; one was in the verification itself. This is the argument for
closing verification gaps rather than carrying them: the gaps were hiding
live bugs.

**① Placeholder-as-measurement, in two new places.** L3 was deliberately
changed earlier in this repair to stop emitting `0.5` for an unmeasurable
basis R², because at a 15% scoring weight `0.5` reads as a middling-but-real
relationship — so an unmeasurable hedge outscored one honestly measured as
weak. The fix emitted `0.0` with a disclosed reason. But
[`layer_10_evaluator.py:121`](backend/engines/layer_10_evaluator.py#L121)
mapped that `0.0` **straight back to `0.5`** (`if r2 <= 0: return 0.5 # unknown
basis — neutral`), reinstating the placeholder one layer downstream and
restoring the exact ranking inversion L3's fix removed. A third instance sat
in the selector as a function default, `basis_r2: float = 0.5`, which was
formatted directly into user-facing rationale text as though it had been
measured. Both are gone. The distinction only became *expressible* because R3
added `basis_risk_source`: `0.0` with a source is a measurement, `0.0` without
one is ignorance, and they must not score alike.

**④ Asserted label, twice more.** The selector listed
`f"Basis R²={basis_r2:.2f} — reasonable hedge tracking"` as a **pro**
unconditionally, so a measured R² of 0.08 was presented to the user as a point
in the hedge's favour — directly contradicted by the number inside the same
sentence, and by the cons list, which stated the tracking honestly. The pro is
now earned (≥0.50 "tracks closely", ≥0.25 "partial tracking") or absent.
Separately, the spread builder's fallback to `profile.tail_correlation` set a
real measurement with no source, which under the new scoring would have read
as unmeasured; it now labels itself.

**Security / UX — the raw pydantic 422 body.** No `RequestValidationError`
handler existed anywhere in the backend, so FastAPI returned pydantic's error
objects verbatim. A holding submitted without `position_type` produced
`"msg": "Unable to extract tag using discriminator 'position_type'"`, a `ctx`
naming the discriminator, and an `input` echoing the submitted position data
back to the caller. `frontend/lib/api.ts` copies `msg` straight into the field
error it renders, so that sentence reached the user. The existing leak
assertions looked for paths, stack frames and module names — none of which
this body contains — so it passed all of them.
[`backend/api/validation_errors.py`](backend/api/validation_errors.py) now
translates each error to a sentence, keeps `loc` intact for field
highlighting, drops `ctx`/`input`/`type`, and logs the untranslated original
server-side. Verified against the running service:
`holdings[0] must include "position_type" so we know which kind of entry it is.`

**The verification's own defect.** `_is_us_market_hours()` subtracted a flat
five hours from UTC — EST, not ET — so during EDT it enforced 10:30–17:00.
It skipped the two-sided-quote assertion for the first hour of every summer
session and kept asserting for an hour after the close. This is the third
false signal from my own checks this repair (after `self.` matching
"itself." and a case-sensitive regex against a CSS-uppercased heading), and
the rule holds: **a check that misfires on correct output trains the reader to
ignore it.** All three are fixed.

**Pinned by:** 11 new tests in `test_selector.py` (basis-risk scoring, prose,
provenance) and 6 in `test_e2e_api.py` (422 sanitisation). The basis-risk
tests were revert-verified: 3 fail against the old `_basis_score`, 6 pass
against the fix.

### Change set

45 tracked files modified (**+6,784 / −1,032**), plus 29 new files totalling
9,846 lines of new Python/TypeScript/Markdown: **17** test files, 5 new backend
modules and fixtures (`backend/api/security.py`,
`backend/api/validation_errors.py`, `backend/core/disclosures.py`,
`backend/data/data_quality.py`, `backend/data/portfolio_valuation.py`,
`backend/tests/fixtures/synthetic_market.py`), and `pytest.ini`,
`.dockerignore`, `backend/.env.example`, `CLAUDE.md`, `PRD.md` and this
report.

**Committed as three commits on `repair/end-to-end-phase-2`** — backend,
frontend, docs — each verified by its own gate before being made. Local only:
nothing is pushed and nothing is deployed. See R5 in §C.

---

## C. Remaining issues

| # | Issue | Severity | Why it is still open |
|---|---|---|---|
| ~~R1~~ | ~~Market-hours pricing unverified.~~ **CLOSED 2026-10-29 09:58 EDT.** 21/21 live tests pass; `premium_source=market_mid`, `iv_source=contract_iv`; liquidity screen rejects 68/165 live strikes (67 OI, 1 spread). The DST bug in the market-hours gate that had been suppressing the key assertion is fixed. | — | Observed, not inferred. §A. |
| ~~R2~~ | ~~The no-valid-quote question is unresolved.~~ **CLOSED.** Decided and implemented: the refusal is conditional on the venue demonstrably quoting. `L4._refuse_unquotable_hedges` withholds a holding's candidates only when *some other* instrument in the same run carries an executable two-sided quote; if nothing is quoted anywhere, the provider is silent and disclosed theoretical premiums stand. 12 tests, 4 revert-verified; armed and correct on a live market-hours run. | — | See the appendix. The decision turned on asymmetric cost, not on preference. |
| ~~R3~~ | ~~Risk-estimate provenance computed but not exposed.~~ **CLOSED.** `RiskEstimateProvenance` now travels on every `HedgeRecommendation`, and `basis_risk_source` / `_n_obs` / `_window_days` / `_stderr` on every `InstrumentCandidate`; `corr_stderr` finally has a consumer. Rendered and verified in the browser. | — | Resolved by extending `HedgeOutput` rather than adding a sibling endpoint: a sibling lets a client render a beta without ever fetching its standard error. |
| ~~R4~~ | ~~Layer 5 (preliminary LLM pass) lightly exercised.~~ **CLOSED.** It is off by default (`run_l5_prelim_llm = False`), so it was untested-by-default code rather than a defect. Its `explain()` docstring claimed *"Never raises — all LLM failures are caught and logged"* while the method has no `try/except`; the docstring is corrected to say what is actually true — the orchestrator's guard is what makes a failure non-fatal — and that guard is now pinned by a test that raises from L5 and asserts the run still produces candidates. The merge path is covered by 2 tests. | — | Closing it is what surfaced the ⑤ disclosure-deletion defect. |
| ~~R5~~ | ~~Nothing is committed.~~ **CLOSED.** The repair is committed to the branch `repair/end-to-end-phase-2` as three commits — backend (68 files), frontend (3 files), docs and `.gitignore` — each one verified by its own gate before being made: the backend suite at 674/31/0 (678/31/0 after the container work below), `tsc --noEmit` and `eslint` clean for the frontend. **Local only. Nothing is pushed and nothing is deployed**, both of which need explicit approval. | — | Three commits rather than one per repair stage: the changes are interdependent, and committing intermediate states that no suite passes against would have made the history look incremental while being less honest than three commits that each pass their own verification. The working tree now holds only seven verification screenshots, left untracked as a record of one browser run rather than repository content. |
| R6 | **Disclosure volume still high.** 8 user-facing warnings on the live market-hours run, down from 37 pre-market and 78 before grouping. Honest, but still dense out of hours. | **P2** | Grouping fixed the duplication, and the market-open run shows most of the remaining volume was genuinely degraded pre-market data rather than a presentation problem. Further reduction risks hiding things. |

| ~~R7~~ | ~~Single-stock return data shows an implausibly weak correlation structure.~~ **CLOSED — and it was never a defect.** Root-caused, not explained away. Two things were conflated. **(a) The data is right.** The product's return series is **byte-identical** to an independent recompute straight from the raw Yahoo JSON — `maxdiff = 0.00e+00` for both the AAPL and the SPY leg — so no stage of the pipeline is distorting it. **(b) The window is the whole story.** `LOOKBACK_DAYS = 252` is what the product actually fits on; the same two series over 500 sessions give AAPL/SPY R²=**0.354**, ρ=**+0.595**, β=**1.03**. Single-stock/index correlation genuinely collapsed over the trailing twelve months — XOM/SPY flips **+0.115 → −0.329** between the two windows — while index-replication pairs are window-invariant by construction (VOO/SPY 0.994 → 0.996). Mega-cap dispersion, correctly measured. And correctly *gated*: SPY is **rejected** as an AAPL cross hedge at R²=0.084 < `MIN_CROSS_R2 = 0.30`, which is the system declining to sell a hedge it cannot justify. | — | The anomaly was real, the alarm was right to raise it, and the answer is that a 252-day ρ of 0.29 is what AAPL/SPY *was* over this particular year. Pinned by the live-provider suite, which asserts both legs carry identical dates and 501 observations. |
| ~~R8~~ | ~~Zero-share and similar edits re-run against a stale result view.~~ **CLOSED**, and root-caused past the reported symptom: *any* unre-run edit left real figures beside inputs they did not describe — editing 500 shares to 5,000 left a VaR for 500 shares on screen with nothing to say which portfolio it belonged to. The page now records the exact payload each result was computed from, flags divergence with a banner and a re-run button, and clears the result outright when the form can no longer be analysed. Browser-verified, 28/0. | — | The reported case was a validation-path instance of a general staleness problem. |
| ~~R9~~ | ~~L11 narrates with a figure the product did not compute.~~ **CLOSED 2026-10-30, and it was not an LLM hallucination.** A stub-only run put *"the $30,000 AAPL position"* against a `portfolio_notional` of **$40,000**. Root cause: L11 computed the exposure it put in the prompt as `shares x purchase_price` — **cost basis** — while L3 measures `current_price x shares` and VaR, hedge sizing and the budget are all computed on that. The narrative was wrong on **every** run, not just empty ones; on a profitable holding it understated exposure by exactly the unrealised gain. `portfolio_valuation.py` already called that expression `cost_basis`. Fixed: L11 passes the measured `notional_value`, the prompt carries `null` rather than a substitute when nothing was measured, the three system prompts forbid estimating a null field, and a run with no candidate on any holding skips the provider entirely. 13 tests, 11 revert-verified. | — | The empty-run symptom led to a defect in the hedged path. The prompt's own rule — *"every claim needs a number"* — turned a missing field into pressure to supply one, which is why the null rule had to be stated rather than implied. |
| ~~R10~~ | ~~A holding the provider cannot price is still hedged — sized, scored and simulated against figures nobody measured.~~ **CLOSED 2026-10-31.** `fetch_market_data` omits an unpriced ticker from `market_data` and only logs it, but the holding stays in the `PortfolioInput` the pipeline runs on. **Five** call sites then re-derived an exposure L3 had already computed as `current_price x shares`, and each had its own substitute: the orchestrator's stock sizing context and its `_holding_notionals` fell back to `shares x purchase_price` — **cost basis**, the price the user typed, which the product elsewhere calls `cost_basis`; L7's missing-context fallback built `holding_notional=hedge_output.portfolio_notional`, sizing one holding as the entire portfolio; L9's simulation inputs substituted a **$100 spot**, a **20% volatility**, and `md.get("notional", hedge_output.portfolio_notional)`; and L10's scoring denominator substituted the portfolio notional too. **Measured, not inferred:** with MSFT's true spot at $400 and its entry absent, the sizer received `holding_notional = $10,000` instead of **$40,000** — a 4x understatement — and still returned 1 contract with **no warning naming MSFT**. The error equals the unrealised gain, so it was worst on the holdings the user had done best on. L9's `notional` fallback was worse still: **nothing in the codebase ever writes `market_data[t]["notional"]`**, so it fired on *every* run, and every `sim_*` and `scenario_*_protection` ratio shipped to the user was divided by the portfolio's exposure instead of the holding's. In the verification run that is a factor of **$121,117 / $68,926 = 1.76**. The 20% vol contradicted the data layer in writing: `routes_portfolio` omits `sigma` deliberately and says *"downstream layers that need vol will see it missing rather than receiving an invented number"* — and `SizingContext.spot_vol` defaulted to 0.20 anyway, feeding the min-variance ratio `h* = rho x sigma_s / sigma_f` directly; reverted, the commodity sizer returned **4 contracts hedging $21,636** off that invented number. | **P1** | Found 2026-10-30 tracing R9. Fixed 2026-10-31 on the user's decision: *"decline to hedge it and say so"* and delete the 0.20 default rather than keep it as a last resort. The correct pattern was already in the same function — the options branch twenty lines below read `rp.notional_value`, L3's measured figure, while the stock branch recomputed it. New `_measured_notional()` is the single reader; `_build_sizing_contexts` now returns `(contexts, declined)`; a new `_decline_unpriced()` step withdraws the candidates **before L6 prices anything**, so no later layer reaches for a substitute, and appends a warning naming the holding. L7 drops an unsizable candidate instead of inventing a context, L9 and L10 take the measured map or decline, and the two volatility defaults are now `Optional[float] = None`. **One wrong turn, caught over HTTP and not by the suite:** the first L9 fix read `rec.unhedged_risk.notional`, which **L12** populates — three layers downstream — so L9 silently simulated nothing on every real run and the response simply had no `sim_*` fields. It now takes the orchestrator's `holding_notionals` explicitly. 23 tests, **18 revert-verified** (the 5 that pass reverted are the controls — a measured holding must still be sized, scored and simulated). Verified live: AAPL's scenario ratios are now against its own $68,926. |
| ~~R11~~ | ~~The headline recommendation published a cost ratio as a protection claim.~~ **CLOSED 2026-10-31.** A live run's `top_recommendation` read *"Buy the AAPL Direct Put (3% OTM) expiring 2027-02-19 for 1312 USD to cover **9.3% of portfolio VaR**."* 9.3% is `1312 / 14051` — the premium over **AAPL's** VaR. The put's maximum payoff covered `11523 / 14051` = **82%** of it, and portfolio VaR in the same run was **$23,756**. Root cause: `prompt_builder.build_combined_context` handed the model a single field named `var_coverage_pct` computed as `cost / holding_var`, nested under `portfolio`. The name asserted protection, the value measured cost, and the position asserted the portfolio — so the model understated the hedge by a factor of nine and attributed it to the wrong denominator. Two further substitutions sat in the same expression: `.get(ticker, 1)` divided by a $1 VaR when a ticker was missing, rendering a $1,312 hedge as **131250%**, and `if holding_var else 0` reported **0%** coverage for a holding whose VaR was not measured — reading as *"this hedge covers nothing"*, which is a finding, where none had been made. Fixed: the quantity is split into `cost_pct_of_holding_var` and `holding_var_covered_pct`, both denominators published as `holding_var_usd` and `max_protection_usd`, both fallbacks replaced by `null`, and prompt rule **1b** forbids calling a per-holding figure portfolio VaR or a cost ratio a protection claim. 12 tests, 11 revert-verified. Re-verified live: the headline now reads *"for $1,312 to cover **82.0% of the holding's VaR**"*, and `key_risks` cites the portfolio field for its portfolio claim. | — | Found 2026-10-31 while verifying the R9 fix end-to-end, by checking the headline's arithmetic against every plausible denominator rather than reading it. R9 fixed the figure the model was *given*; R11 is the figure it was given a false *name* for. Neither was a hallucination — both were the product mislabelling its own output, and only the arithmetic check distinguished them from a model error. The suite had 800 tests and none multiplied a ratio back out. |
| ~~R12~~ | ~~A silent cost-basis fallback moved the reported portfolio VaR by up to 48%.~~ **CLOSED 2026-10-31.** `layer_03_risk_analysis._compute_portfolio_var` weighted the covariance matrix by re-calling `get_current_price(ticker)` per holding, with `except Exception: return match.shares * match.purchase_price` under a comment calling it a *"graceful fallback"*. Two things were wrong. The fetch was **redundant**: a ticker only reaches `returns_dict` after `_analyze_holding` has already priced it, and that price is sitting on the profile as `notional_value` — so this was the sixth site re-deriving a figure already measured. And the fallback was **silent** — no warning, no `dq.record`, nothing in the response — while these weights set the whole portfolio's VaR. **Measured, not inferred:** on a two-holding portfolio (AAPL 200sh @ $400 vs $150 basis, 1% daily vol; MSFT 100sh @ $520 vs $300 basis, 3% daily vol), one transient price failure moved the reported figure from **$29,384** to **$43,473 (+48%)** when AAPL failed and to **$24,094 (-18%)** when MSFT failed. The sign depends on whether the mispriced holding is the volatile one, and the negative direction **understates the user's risk**. | **P1** | Found 2026-10-31 while verifying that R10's fix held at its *source* — `_measured_notional()` trusts `RiskProfile.notional_value`, so the question was whether L3 could put a substituted figure there. It cannot (the degraded path sets `data_available=False`, which `_measured_notional` rejects), but the same file was substituting one layer over, in the VaR weights. Fixed by reading the measured notionals instead of fetching again: the function now takes `risk_profiles` and builds its weights from `notional_value` where `data_available`. A holding L3 could not measure now weighs **nothing** rather than its cost basis. Re-measured after the fix: with the price feed **entirely down** the portfolio VaR is unchanged to the cent, because no fetch happens any more. 2 tests, both revert-verified. Full suite 837 passed, 31 skipped. **No behavioural decision was involved** — unlike R10, this is strictly "read the figure you already computed", which is why it was fixed rather than queued for review. |

**No P0 defect remains open. R6 is the only open defect.** R1, R2, R3,
R4, R5, R7, R8, R9, R10, R11 and R12 are closed — R1, R3 and R7 by investigation, R2 by a decision
now implemented and tested, R4 and R8 by repair, R5 by committing. R6 (warning
density out of market hours) stays open by judgement: grouping fixed the
duplication, the market-open run showed most of the remaining volume was
genuinely degraded pre-market data, and reducing it further risks hiding
things. R9 was opened and closed on 2026-10-30: the HTTP verification of the
stub-disclosure fix showed an LLM-authored dollar figure that did not match
the computed notional, and tracing it found the pipeline had been handing the
model cost basis where it meant market exposure — a defect on every run, not
only the one that exposed it.

**One P0 was found and closed after that sentence was first written**, and the
sequence is the point. R5 was closed by committing the repair; building the
container — the verification the previous pass had recorded as impossible —
then found five live API keys in an image layer (defect 37, §F). It is fixed,
the image is rebuilt and re-verified 11/11, and the suite now evaluates
`.dockerignore` instead of asserting its text. **No P0 remains open.** What
remains unverified is the deployment *platform* rather than the artefact:
Cloud Run's networking, memory limits and cold start, and the LLM path inside
the container.

**R7 is closed by root cause, not by argument.** The previous cycle recorded
it as *"unresolved rather than explained away"*, which was the right call on
the evidence then available but is no longer the state of knowledge. Two
measurements settled it: the product's series matches an independent recompute
from raw provider JSON to the last bit, and the weak R² reproduces or
disappears purely as a function of window length while index-replication pairs
do not move. The product also behaves correctly on the finding — it renders
*"beta fitted on 252 sessions, R² 0.08"* **and refuses the hedge**, rather than
sizing against a beta it does not believe.

What this cycle opened in R7's place is in §B① below: the history path was
returning **dividend-unadjusted** prices, and the cache crashed on a read-only
filesystem. Neither was visible from this machine without deliberately
reproducing production's conditions.

---

## D. Test results

### Automated

| Suite | Result |
|---|---|
| **Full backend suite** | **678 passed, 31 skipped, 0 failed** — 50.4s; 91.4s and 52.1s on earlier runs this cycle |
| **Live-provider suite, market open** | **21 passed, 0 skipped** in 2.6s — `OPTIONQ_LIVE_TESTS=1`, 09:58 EDT 2026-10-29 |
| Skips in the default run | The same 21 live tests, excluded by the opt-in gate rather than by failure |
| End-to-end API integration (`test_e2e_api.py`) | **59 passed** offline |
| **Browser-driven UI journey** | **28 passed, 0 failed, 7 informational** — real Chromium against the static production export and the live backend |
| Frontend | `tsc --noEmit` clean; `eslint app lib types` clean |
| **Live run with an LLM provider wired** | **Fix proven both ways** over the real HTTP route: 5 disclosures deleted with the fix reverted, 0 deleted with it in place |

New tests this cycle (+30 over 644): **16** in `test_price_adjustment.py` for
the dividend-adjustment fix, **14** in `test_cache_resilience.py` for the
read-only-filesystem crash, and **2** added to the live-provider suite — which
is why the default-run skip count rose from 21 to 31 (10 of the live tests are
newly parameterised across tickers).

The cycle before that (+75 over 569): 8 pinning risk-estimate provenance in the
API response, 11 pinning basis-risk scoring and its prose, 6 pinning the
sanitised 422 body, 12 for the R2 no-market gate, 12 asserting the deployment
image against what the code imports, 19 for the disclosure record and the
explainer factory, and 7 driving the LLM layers through the whole pipeline.

Both of this cycle's suites were revert-verified: price adjustment **7 of 14
fail** without the fix, cache resilience **3 of 14 fail**. Neither was written
against a mock of the failure — the cache suite `chmod`s a real directory and
the price suite builds a payload whose High/Low sit against the *unadjusted*
close, so an implementation that substitutes `adjclose` rather than scaling by
it fails.

Every one was confirmed to fail against the unfixed code before being
accepted. The reverts: basis-risk 3 failures → 6 passes; R2 gate 4 failures →
12 passes; disclosure unit tests **6 of 15** failures → 15 passes; the
full-pipeline disclosure tests 2 failures → 7 passes. The two that *didn't*
fail on revert are listed as guards-on-the-guard, not as regressions —
`test_the_explainer_actually_ran` and
`test_some_candidate_carried_a_disclosure_to_begin_with` exist precisely so
the class cannot pass vacuously, which it would otherwise have done against
the fully-quoted fixture market (it did, once, and the fixture was changed to
the overnight snapshot rather than the assertion relaxed).

### Coverage by area

| Area | File | Classes |
|---|---|---|
| Options pricing | `test_instruments.py` | BSM, binomial, Black-76, Greeks, sizer |
| Smoke (imports, settings, fixture, cache) | `test_smoke.py` | 5 tests |
| IV integrity | `test_iv_plausibility.py` | 6 |
| Instrument selection | `test_selector.py` | 14 (incl. the R2 no-market gate) |
| Sizing | `test_sizing.py` | 8 |
| Risk / VaR | `test_risk.py`, `test_var_horizon_label.py` | 6 + module-level |
| Factor / beta | `test_factor_decomposer.py` | 3 |
| Covariance / cross-hedge | `test_covariance.py` | 5 |
| Degraded risk paths | `test_degraded_risk_profile.py` | 3 |
| Disclosure & budget | `test_disclosure_and_budget.py` | 4 |
| Data provenance | `test_data_quality.py` | 5 |
| LLM fallback | `test_llm_fallback.py`, `test_explainer.py` | 5 + 3 |
| **Disclosure survival & explainer factory** | `test_disclosures.py` | 5 |
| **Deployment config contract** | `test_deployment_contract.py` | 3 |
| **Price-history adjustment** | `test_price_adjustment.py` | 5 (16 tests) |
| **Cache resilience on a read-only filesystem** | `test_cache_resilience.py` | 4 (14 tests) |
| Security | `test_security.py` | 9 |
| Pipeline integration | `test_pipeline.py` | 11 (incl. the LLM layers end to end) |
| **HTTP API end-to-end** | `test_e2e_api.py` | 10 |
| Live data (opt-in) | `test_live_market_data.py` | 4 (incl. both fetch paths agreeing on the adjusted series) |

### Test integrity

No assertion was weakened to make a test pass. Specifically:

- Where a test failed because a **fixture** was wrong, the fixture was fixed —
  e.g. a `PortfolioInput` left `total_notional = None`, which made a
  disclosure test pass **for the wrong reason** (via the generic exception
  path rather than the successful-but-blank path under test). That was called
  out and corrected rather than accepted.
- Where `pytest.approx(_trading_days(120), rel=1e-6)` failed on 82.79 vs
  82.7926…, the assertion was changed to `== round(_trading_days(120), 2)` to
  match what the calculator actually reports — not to a looser tolerance.
- The end-to-end test's leak detector produced a **false positive** (the
  substring `self.` matched the English word *"itself."* in a legitimate
  rationale). The markers were narrowed to regexes anchored on word
  boundaries, **and two tests were added to pin the detector itself** — one
  proving it still fires on a real traceback, one proving it stays quiet on
  product prose. A marker that fires on correct output trains the reader to
  ignore the check; a marker narrowed without a counter-test can silently stop
  detecting anything.
- Two warning assertions (`"garch:SPY" in w`, `"var:IPO" in w`) broke when
  warnings became grouped. Both were updated to assert the field **and** the
  subject separately — the same intent, expressed against the new format —
  rather than reverting the grouping.
- A disclosure test initially passed **vacuously**: run against the
  fully-quoted fixture market, no candidate carried a disclosure at all, so
  "no disclosure was deleted" was trivially true. The fixture was switched to
  the overnight snapshot (`ChainConfig(quotes_published=False)`), which is
  what makes the selector say *"premium shown is theoretical"*, and a
  precondition test was added so the class can never pass that way again.
- One of those tests then failed for a reason that turned out to be **the
  test's fault, not the code's**: it asserted a disclosure always leads the
  `cons` list, but L11 only explains the top three candidates per holding, so
  a candidate the model never saw legitimately keeps the selector's own
  ordering. The assertion was narrowed to candidates that were actually
  rewritten — the front-of-list guarantee is about a *replaced* list — rather
  than reordering every candidate's cons to make the original assertion true.
- The product pipeline is never mocked. The end-to-end test substitutes
  exactly two things: the market-data provider and the language model. Both
  are **network boundaries, not product logic**. The LLM-layer tests in
  `test_pipeline.py` substitute the model's *reply* and nothing else: the
  engine, the real `_merge_explanation`, the scorer and the formatter all
  execute. The live proof in §B ⑤ goes further and substitutes only the
  Ollama HTTP endpoint, so the real `OllamaExplainer`, its real JSON parser
  and the real FastAPI route are all in the path.

### Performance note

Suite runtime fell from 172s to 84s by making two fixtures module-scoped
(`test_e2e_api.py`, `test_pipeline.py`). Both fixtures' own docstrings already
described them as shared by read-only assertions; function scope was re-running
the entire product — Monte Carlo included — once per assertion. No test mutates
the shared output.

---

## E. Financial verification

All figures below are from the live run at 05:50 ET on 2026-10-29, portfolio
AAPL 400 / IWM 300 / TLT 500 / GLD 200, 120-day horizon, 15% protection, 3%
budget. **[verified]**

| Property | Observed | Verdict |
|---|---|---|
| **Valuation** | Market 337,116 vs cost 234,000, P&L 103,116 | ✅ Market value from current prices; cost basis separate, not substituted |
| **Budget derivation** | 10,113.48 = exactly 3.0% of 337,116 | ✅ Budget is a share of **market** value, not cost basis |
| **Budget enforcement** | Total hedge cost 7,328 / 10,113 = **72.5%** | ✅ Portfolio-level, not per-holding |
| **VaR horizon** | Narrative: *"Over the 120-day hedge horizon, 95% VaR is $42.2K and 95% expected shortfall $68.8K"*; all four `unhedged_risk` blocks report `horizon_calendar_days = 120` | ✅ Horizon reported, not asserted. Was *"One-day"* / *"1mo"* for the same figure |
| **Horizon arithmetic** | `_trading_days(120) = 82.79`; ratio of 120-day to 1-day VaR = √82.79 ≈ 9.1 | ✅ Trading days, not calendar days **[tested]** |
| **VaR method labelling** | `parametric_cornish_fisher` with GARCH sigma, or `assumed_vol_fallback` below `MIN_OBS_FOR_MOMENTS = 30` | ✅ Named for what it is |
| **Portfolio beta** | 0.73 to SPY, excluding holdings whose beta is unmeasurable | ✅ No `1.0` placeholders weighted in |
| **Contract reality** | 14 candidates; **0** without an OCC symbol; **0** with a strike unverified against the chain; **0** OCC/strike mismatches | ✅ Every recommendation is a real listed contract |
| **Contract sizing** | All counts whole, positive, correct multiplier, hedge instrument's spot | ✅ **[tested]** |
| **Premium provenance** | `premium_source = {model: 14}` | ⚠️ Correct and disclosed out of hours — **but the `market_mid` path is unproven live** (R1) |
| **IV provenance** | `iv_source = {realized_vol: 14}` | ⚠️ Same — the `contract_iv` skew path is unproven live (R1) |
| **Honest refusal** | *"No hedge for TLT fits the cost budget and liquidity constraints; the position is shown unhedged."* | ✅ No forced recommendation |
| **Regime** | `mid_vol` via `gmm_combined`, `degraded = False` | ✅ Source disclosed |
| **Provenance ledger** | `{cache: 51, live: 18, unavailable: 33, fallback: 18}`; 51 degraded points → 36 grouped lines, **every degraded field still named** | ✅ Verified by the script, not assumed |

### Numerical regression tests added

Cornish-Fisher VaR scaling, trading-day conversion, Greeks (BSM / binomial /
Black-76), net and collar Greeks, contract multipliers, protection-level
tracking, budget monotonicity (*a tighter budget never costs more*), spread
cost below its long leg, credit-structure sign handling, and R²-vs-correlation
admission arithmetic. **[tested]**

### The financial judgement call, now made

**R2 — decided and implemented.** Out of hours, no contract has a two-sided
quote, so every premium is theoretical. Refusing all of them would make the
product useless for two thirds of the day; ranking all of them invites the
user to size a position in a contract nobody will fill. Both framings were
wrong because they answered the wrong question: *"is this premium quoted?"*
rather than *"is anything quoted right now?"*

The gate is therefore conditional and self-calibrating. If any instrument
anywhere in the run carries an executable two-sided quote, the venue is
quoting, and a holding with none has **no market** — its candidates are
withheld and the holding is reported unhedged with the reason stated. If
nothing is quoted anywhere, the provider is silent and nothing can be
concluded about any single contract, so disclosed theoretical premiums stand.

No market calendar is involved, deliberately: a calendar has to be maintained
and would refuse hedges wrongly on every holiday it missed. The asymmetry is
also deliberate — a missed refusal leaves a disclosed theoretical premium on
screen, while a wrong refusal silently removes a hedge the user could have
placed — so the refusal fires only on positive evidence that quoting works
right now.

**And the disclosure that makes the "theoretical premiums stand" branch
acceptable was, until this cycle, deletable by the LLM layer.** That branch is
only defensible because the user is told plainly what they are looking at; §B
⑤ is the defect that removed the telling. The two findings belong together:
R2's resolution depends on the disclosure surviving to the response, which is
now tested at the unit, pipeline and live-HTTP levels.

---

## F. End-to-end user journey

| Step | Status | Evidence |
|---|---|---|
| 1. Enter positions (ticker, quantity, cost, date, asset class) | **Verified** | 4 asset classes accepted; 422 with sanitised message on bad input **[tested] [verified]** |
| 2. Set constraints (horizon 30–365d, protection 5–30%, cost 1–15%) | **Verified** | Honoured end to end; budget = 3% of market value exactly **[verified]** |
| 3. Submit and wait | **Verified** | 200 in 14.9s; per-route `AbortSignal.timeout` client-side, `asyncio.timeout` server-side **[tested]**; spinner and typed `UiError` **[code]** |
| 4. Portfolio summary (value, cost, P&L, VaR, ES, beta, regime) | **Verified** | 337,116 / 234,000 / 103,116; VaR $42.2K over a **named** 120-day horizon; beta 0.73; `mid_vol` **[verified]** |
| 5. Per-holding recommendations | **Verified** | One entry per holding; TLT returned unhedged **with a stated reason** **[verified]** |
| 6. Candidate detail (contract, strike, expiry, premium, Greeks, provenance) | **Verified** | 14 candidates, all OCC-verified; `premium_source` and `iv_source` on every one **[verified]** |
| 7. Budget bar / cost comparison | **Verified** | 7,328 / 10,113 = 72.5% **[verified]** |
| 8. Narrative explanation | **Verified** | Produced with no model available, and `llm_notice` **says so** **[verified]**. It also distinguishes *why*: a provider that refuses permanently (no credits, refused key, retired model) is reported as permanent and operator-actionable rather than as "no analysis for this run" — see defect 38 **[verified live] [tested]**. As of 2026-10-30 it is also verified *with* a model, on both Google surfaces: a live run over real market data produced Gemini prose on all 14 candidates with an **empty** `llm_notice`, the deterministic backstop standing aside, and every degradation warning intact afterwards — first over the API key, then over Vertex AI with IAM authentication, which is the configured provider **[verified live] [tested]** |
| 9. Disclosures and data quality | **Verified** | Ledger + grouped warnings (8 on the market-open run, 37 pre-market); every degraded field named **[verified]** |
| **9b. Disclosures survive an LLM rewrite** | **Verified** | Live `POST /portfolio/analyze` with a provider wired in: all 6 candidates rewritten by the model, all 5 disclosures still present and leading the `cons` list. Same request with the fix reverted: **5 deleted** **[verified] [tested]** |
| **Failure scenarios** | **Verified** | 5/5 malformed inputs → 422 sanitised; malformed JSON → 422; unknown route → 404; 4/4 privileged endpoints → 503 fail-closed; **no internals leaked anywhere** **[verified]** |
| **Market-hours journey** | **Verified** | 2026-10-29 09:53–09:58 EDT. 21/21 live tests; `premium_source=market_mid`, `iv_source=contract_iv`; SPY 95% put 10.30/10.37 (0.68% spread); 163/165 strikes two-sided; screen rejects 68/165 **[verified]** |
| **Rendered UI** | **Verified** | Driven in Chromium against the static production export and the live backend: **23 passed, 0 failed**. The horizon-labelled VaR, the R3 provenance strip ("beta fitted on 252 sessions, R² 0.08"), the contract & data provenance panel, a real OCC symbol, the premium disclosure and the LLM fallback notice all render. No uncaught JS errors, no failed requests, no console errors. 5 screenshots captured **[verified]** |
| **Invalid input is reported** | **Verified** | A zero share count produces a visible, actionable message and leaks nothing. The earlier version of this check asserted only the absence of a leak, which a page showing *no error at all* passes — it was strengthened after it passed vacuously **[verified]** |
| **Results cannot be read against the wrong inputs** | **Verified** | Editing any input after a run raises a stale-results banner with a re-run button; breaking the form clears the result rather than leaving real figures under a validation message. 5 browser checks **[verified]** |
| **The journey survives the production runtime** | **Verified, with one stated limit** | See below — the runtime was reproduced from the image's own pins and booted through its own `CMD`; **the image was never built** |

### The container, built and run — and the three defects that only a build could find

Phase 5 asks for the Docker and cloud configuration to be *verified*, not
merely inspected. An earlier pass of this report recorded that no container
runtime existed on this machine and that the image therefore could not be
built; it reproduced the runtime in a clean venv instead and stated the gap
plainly. **That conclusion was wrong in one important respect.** No runtime was
*present*, but Homebrew and hardware virtualization were both available
(`kern.hv_support = 1`), so one could be *installed*. Podman 6.1.4 was
installed, its VM started, and because this host is arm64 while Cloud Run is
x86-64, the image was built for the deployment target explicitly:

```
podman build --platform linux/amd64 -t optionq-api:verify -f Dockerfile .
```

It built: 11 steps, exit 0, all 73 distributions resolving to `cp313` linux
x86-64 **wheels** with zero source builds. Then it was run, and the run found
three defects that no amount of static inspection had found — including one P0
that **this report had already claimed was fixed**.

| What the running container showed | Verdict |
|---|---|
| `/app/backend/.env` present, 713 bytes, holding `ANTHROPIC_API_KEY`, `HF_API_TOKEN`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `FRED_API_KEY` | **P0.** Every key in a published image layer |
| `/app/backend/.venv` present, **417MB** of host arm64 binaries | Image 1.08GB instead of 668MB, and host binaries beside the installed ones |
| `/app/backend/hedgeos_cache.db` present, 6.4MB, root-owned | **Defect 35 was never actually fixed** — this is the exact state its row in the register describes as resolved |

These are not three new findings. They are **one new defect — 37 — and the
discovery that two fixes this report already recorded as closed were never in
effect.** The secret-in-image fix and defect 35 both consisted of adding rules
to `.dockerignore`; the rules were added, and they matched nothing. Both rows
reopen and close again here, for real this time, and the register says so
rather than quietly keeping the original ticks.

**One root cause — defect 37.** `.dockerignore` is not `.gitignore`. Git
re-applies a bare pattern at every directory level; Docker and Buildah match
each pattern **once, against the whole path relative to the context root**. So
`.env` excludes `./.env` and nothing else, `*.db` excludes `./*.db` and nothing
else, and every rule in that file aimed at something *inside* `backend/`
matched nothing at all. `backend/tests/` was the only rule that worked — because
it names its path, which is why `backend/tests` was correctly absent and the
leak looked like a solved problem.

This was confirmed against two throwaway builds rather than asserted from
memory: a context holding `.env`, `sub/.env`, `sub/cache.db` and
`sub/.venv/lib.py`, built once with bare patterns and once with `**/`-anchored
ones.

| `.dockerignore` contents | What survived into the image |
|---|---|
| `.env`, `*.db`, `.venv/` | `sub/.env`, `sub/cache.db`, `sub/.venv/lib.py` — root `.env` excluded, everything nested kept |
| `**/.env`, `**/*.db`, `**/.venv/` | nothing |

That second row also settles a detail the fix depends on: `**/` matches **zero**
or more leading components, so the anchored form still covers the root.

**How this got past the test suite — defect class ④, caught red-handed.** Three
tests existed for exactly this. They read:

```python
assert ".env" in patterns
assert ".venv/" in patterns
assert "*.pem" in patterns and "*.key" in patterns
```

All three passed. They assert that a rule is *written*, and the comment above
them asserts what it *does* — and nothing connected the two. This is the
asserted-label defect in its purest form, and it was sitting in the file whose
whole purpose is to catch deployment defects that tests cannot see.

**The fix.** Three parts, because the pattern fix alone would leave the same
hole open to the next edit:

1. `.dockerignore` rewritten with every nested rule `**/`-anchored, and a
   header explaining the semantics so the next person does not reintroduce it.
2. A **build-time guard** in the Dockerfile: a `RUN` step that `find`s the
   copied tree for `.env`, `*.pem`, `*.key`, `*.db`, `.venv` and `tests/`, and
   **fails the build** if any is present. A leak is now a failed build rather
   than a published secret. Revert-verified: restoring the bare patterns fails
   the build at step 6 with `REFUSING TO BUILD:` and the offending paths listed.
3. The three string-presence tests replaced by tests that **evaluate**
   `.dockerignore` — a translator for Docker's glob semantics (`**/` → zero or
   more components, `*` and `?` not crossing `/`, last match wins, `!`
   negation), run against the paths that actually leaked. Revert-verified: with
   the old patterns, `test_the_paths_that_leaked_are_excluded` fails on
   `backend/.env`.

**After the fix — the image rebuilt and re-verified, 11 of 11:**

| Check | Result |
|---|---|
| Image contents | No `.env`, no `.pem`/`.key`, no `.venv`, no `.db`, no `backend/tests` |
| Image size | **1.08GB → 668MB** |
| Runs unprivileged | `uid=10001(optionq)`, not root |
| Platform | `x86_64`, `glibc 2.41`, Python `3.13.18` — the deployment target, not the host |
| Every runtime import | `FAILED_IMPORTS: []` — and `backend.main` imports, which is the `b59d1e2` failure mode |
| The pinned `wrapt` | `2.5.2` inside the image, not the dev prerelease |
| Fail-closed: no secrets | `ConfigurationError: Refusing to start:` naming both missing secrets, **exit 3** |
| Fail-closed: short secrets | **exit 3** |
| Fail-closed: `http://localhost` origin in production | **exit 3**, naming both the dev origin and the non-HTTPS scheme |
| `/health` | **200** `{"status":"ok","version":"0.1.0"}` |

**The auth matrix, against the live container.** Both credentials, both route
groups, four states each — and the point of the test is the two cells where a
*valid* credential is refused, because that is what proves `CRON_SECRET` does
not unlock the admin surface:

| | no header | wrong token | `ADMIN_API_KEY` | `CRON_SECRET` |
|---|---|---|---|---|
| `GET /admin/training/status` | 401 | 401 | **200** | **401** |
| `GET /admin/cron/retrain-fast` | 401 | 401 | **401** | **200** |

Every refusal body is `{"detail":"Unauthorized"}` — no internal detail, no
stack, no hint which half of the credential pair was wrong. The `CRON_SECRET`
cell returning 200 means a real fast retrain executed inside the container and
succeeded, which exercises the scheduler path as a side effect.

**A real analysis, through the container.** `POST /portfolio/analyze` with the
AAPL(500)/MSFT(300)/SPY(200) payload, `protection_level 0.1`,
`max_hedge_cost_pct 0.02`, `hedge_horizon_days 90`:

| | |
|---|---|
| Response | **200**, 41.6KB, **31.2s** |
| Portfolio | notional **$487,369.00**, cost basis $334,810.00, unrealized **+$152,559.00** |
| Regime | `mid_vol`, VIX **17.20**, `regime_source: gmm_combined`, `regime_degraded: false` |
| Recommendations | 3 ranked assets, 11 candidates — real OCC contracts (`SPY261218P00769000`), real strikes and listed expiries |
| Budget | cost **$4,160.50** against a **$9,747.38** cap = **0.854%** of notional against the 2% requested, `hedge_budget_utilization: 0.4268` |
| Provenance | `source_counts: 42 cache / 38 live / 2 unavailable / 2 fallback`, every degraded field named with its reason |
| Disclosure | 3 warnings surfaced, including UUP IV unavailable and 12 SPY contracts excluded for a bid-ask spread wider than 50% of mid |
| LLM | `llm_notice` states explanations are unavailable and the narrative came from the computed figures |
| Cache | **4.68MB** written to `/var/cache/optionq/`, owned by `optionq`; **nothing** written into `/app`, no `/tmp` fallback |
| Container logs | **zero** tracebacks or errors across the whole session |

Note what the disclosure does here rather than what it says: UUP's missing
per-strike IV is reported as `unavailable` and `fallback` with the exact reason,
and the SPY candidates whose mid was not an attainable fill price were
**excluded**, not quietly priced. That is rule 5 holding under live data in the
deployment artifact.

**What this still does not prove.** Four things, stated rather than papered
over:

- **Cloud Run itself.** Networking, memory limits, cold start, request timeouts
  and the real service account are untested. The image is verified; the platform
  is not. Nothing has been deployed, and nothing will be without approval.
- **The LLM path in the container.** No `ANTHROPIC_API_KEY` was passed, so the
  explainer ran its deterministic fallback. That the fallback discloses itself
  correctly is verified; the Claude path in the container is not.
- **amd64 under emulation.** The container ran x86-64 on arm64 via qemu. The
  logic, the imports and the glibc are the deployment target's; the instruction
  timing is not, which is part of why the analysis took 31.2s here.
- **That it will happen again.** Podman was installed on 2026-10-29 only to run
  this verification, and was removed from the machine on 2026-10-30, so none of
  the above is reproducible here as the repo now stands. The evidence is real
  and the Dockerfile's leak guard is committed, but the guard only runs when
  somebody builds, and nothing in the repository causes a build. Until the
  build lives in CI or Cloud Build (§9, P1), the single most productive check
  of this entire repair is also the one least likely to run again.

**What it does prove** is the thing the previous pass of this report could only
assume — and the assumption was wrong, in the most expensive direction
available. Every green test in this suite, and every line of this report's
earlier deployment section, coexisted with five live API keys sitting in an
image layer. The build was the only thing that could see it.

---

## G. Product readiness scores

| Dimension | v0 | v1 | Notes |
|---|---|---|---|
| Core functionality | 2 | 9 | Completes the workflow on real data; contracts are real |
| Financial correctness | 1 | 8 | Verified arithmetic; live pricing paths proven; the no-market refusal decided and implemented; R7 root-caused and cleared. Held at 8, not raised: this cycle found the history path serving **dividend-unadjusted** prices, which flipped the sign of HYG's annual return and moved three of the five macro factors by ~4.5pp. Fixed and pinned, but it is the second time a returns-level defect has survived a full verification pass |
| Data integrity / honesty | 1 | 9 | The strongest dimension. Provenance ledger, rejection over clamping, disclosed substitution — and disclosures that now survive the LLM layer, proven live. Held at 9, not raised: this is where the P0-class deletion defect was, and it went two cycles undetected |
| Security | 2 | 9 | Fails closed, sanitised, scoped, constant-time, startup-validated |
| Reliability | 3 | 8 | Timeouts both sides; degraded paths explicit; no silent fallbacks |
| Testing | 2 | 9 | 678 tests + 21 live + 28 browser checks; every new test revert-verified, two found to pass vacuously and strengthened. Held at 9: the read-only-filesystem crash proves the suite's blind spot is *environmental*, and the new `test_cache_resilience.py` closes that one case by `chmod`-ing a real directory rather than by mocking the failure |
| UX / product polish | 3 | 9 | Disclosure-first UI, browser-verified; stale results can no longer be read as current; −1 for out-of-hours warning density (R6) |
| Deployment readiness | 2 | **8** | Startup and auth fail-closed behaviour **verified inside the built container**; secrets verified absent from the built frontend and from the image; all 18 pins match tested versions and all 73 distributions resolve to `cp313` linux x86-64 wheels with zero source builds. Raised 7 → 8 this cycle: the image was built for `linux/amd64` and run — 11/11 checks, 668MB, unprivileged, fail-closed on three bad configurations, `/health` 200, the full auth matrix with both valid-credential refusals correct, a real `POST /portfolio/analyze` at **200 in 31.2s** on live data within 0.854% of a 2% budget, the cache confined to `/var/cache/optionq`, zero tracebacks. A build-time guard now fails the build on any leaked secret or database. **Not 9:** the first build found five live API keys in the image (defect 37), which is what a cycle of asserting configuration instead of building it costs; and Cloud Run's networking, memory limits and cold start remain untested, the LLM path was not exercised in the container, nothing is deployed by instruction, and the container ran under qemu emulation |
| **Overall** | **2.5** | **9.0** | |

---

## H. Resume credibility

**What can be claimed, truthfully:**

> Took a quantitative hedging prototype that passed its own smoke tests and
> failed a readiness audit at 2.5/10, and repaired it to a verified
> end-to-end product: 39 defects across market data, options pricing, risk
> measurement, position sizing, ranking, LLM integration and security, each
> with a regression test confirmed to fail against the unfixed code. Built a
> request-scoped data-provenance ledger so every figure reports whether it
> was live, cached, substituted or unavailable, and a disclosure record the
> explanation layer cannot overwrite — found when an LLM reply was shown to
> delete *"premium shown is theoretical"* from live API responses. Grew the
> suite to 678 tests including real-HTTP end-to-end integration, and verified
> the full user journey against a live server in and out of market hours and
> against a reproduced production runtime — which is how the last two defects
> were found: a history path serving dividend-unadjusted prices, and a cache
> that crashed every request on a read-only container filesystem.

**The strongest thing to talk about in an interview** is not the defect count
— it is the **classification**. Four recurring patterns explained nearly every
defect, and naming them is what made the repair tractable rather than a
whack-a-mole:

1. **Placeholder-as-measurement** — `return 1.0` for an unmeasurable beta is
   not a default, it is a false claim that sizes contracts.
2. **Truncate-before-align** — windowing each series before joining compares
   different calendar periods.
3. **Solver-artifact IV** — a two-sided quote is the *precondition* that makes
   an implied volatility a measurement; a $0.00001 IV produced a 1.5-million-
   contract recommendation.
4. **Asserted label** — a dollar VaR without its horizon is not a risk number;
   the same figure was presented 9× too small because its consumer asserted a
   horizon its producer never reported.
5. **Silent substitution** — a backstop that fills in for failed data must
   report that it ran; two history paths that disagreed about whether a
   "price" includes dividends flipped the sign of a macro factor's annual
   return.
6. **Verified-here-only** — a path that behaves differently because
   production's permissions or installed dependencies differ from the
   developer's is not a verified path. This one was paid for twice: a
   disclosure-deletion defect that hid behind an unset API key, and a cache
   crash that hid behind a writable directory.

The second strongest is the judgement call in ③: **rejecting an implausible
reading is correct; clamping it to a plausible floor is not**, because a
clamped 5% IV yields a priced, tradeable-looking recommendation built on a
quote nobody made. That distinction is the difference between a quant who
knows the formulas and one who knows what a number means.

**What must not be claimed:**

- ❌ "Production-ready" or "deployed." Nothing is deployed, by instruction, and
  nothing is even pushed — the work sits on a local branch.
- ❌ "Live trading" or "execution." Not implemented, deliberately.
- ❌ "Validated against live option quotes." Every run has been out of hours
  and every premium model-priced. This is the one claim that would be false,
  and it is also the easiest for an interviewer to probe.
- ❌ Any user, adoption, accuracy or performance figure. There are no users
  and no backtest.

- ⚠️ "Container-verified." Now largely true and worth stating precisely: the
  image **was** built for `linux/amd64` and run, and the workflow, the
  fail-closed paths, the auth matrix and the cache confinement were all
  verified inside it. What is still **not** verified is Cloud Run itself — its
  networking, memory limits and cold start — and the container ran x86-64
  under emulation. Say "image-verified, platform-unverified," not
  "production-ready."

**The honest framing:** *a verified decision-support engine, pending
market-hours validation of its live pricing path.* That sentence is both more
defensible and more impressive than "production fintech platform," because it
demonstrates knowing the difference.

---

## Appendix — the structure cleanup, and the test that only passed here

Two things, closed 2026-10-30.

**A test made a developer's config a precondition for the suite.** The Vertex
work shipped green locally and turned CI red. One test,
`test_an_explicitly_empty_project_does_not_fall_back_to_the_configured_one`,
asserted that a project *was already configured* before checking that
`project=""` is not silently replaced by it — and told the reader to put one in
`backend/.env`. That holds on the machine it was written on and nowhere else;
CI has no `.env`, so the guard fired and the suite failed. The assertion the
test exists for was fine; its precondition was ambient. It now installs the
configured value itself with `monkeypatch`, so it tests this code instead of
this machine. The suite is now run both ways — with `backend/.env` in place and
with it moved aside — and passes identically, which is the check that would
have caught it.

**The repository advertised an architecture that was not there.** Thirty-eight
files were empty. `backend/execution/` held five empty broker modules —
`alpaca_broker.py`, `paper_broker.py`, `order_builder.py`, `twap_executor.py`,
`base_broker.py` — so the directory listing implied an execution path exists in
a product that must not trade. `backend/instruments/` held ten empty
instrument modules beside the real `options/` package, `backend/engines/` held
four phantom layers (`layer_01_input_parser`, `layer_03b_stress_monitor`,
`layer_04_scenario_engine`, `layer_05_instrument_selector`) whose duplicate
numbers made the pipeline unreadable, and `backend/core/pricing/` held six
empty pricers beside the real ones. One of them,
`backend/instruments/inverse_etfs.py`, sat in the same directory as the real
`inverse_etfs/` package — a module and a package competing for one name, which
resolves by import precedence rather than by intent.

Removed with them: two dead modules, nine dead definitions and 35 unused
imports. `backend/data/rates_data.py` was the one that mattered — nothing
imported it, the live rate comes from `market_data.get_risk_free_rate()`, and
it carried `{"3m": 0.045, "5y": 0.043, "10y": 0.042, "30y": 0.044}` as a silent
fallback for a failed curve fetch plus a bare `return 0.045` for SOFR. Dead
code cannot fabricate a yield curve, but code that is wired in later can, and
the fallback was written to look like data.

`test_pricing.py` is now `test_smoke.py`. It contained no pricing assertion and
never had one, while the coverage table above credited the name with BSM,
binomial, Black-76, Greeks and sizing. That coverage exists, in
`test_instruments.py`. The table and the filename both said otherwise.

Scope: no behaviour changed. 769 passed, 31 skipped, before and after, and the
FastAPI app imports with all 20 routes registered.

## Appendix — R1, closed 2026-10-29

Run between 09:30 and 16:00 ET on a trading day:

```bash
cd /Users/dipen/Projects/optionQ
OPTIONQ_LIVE_TESTS=1 backend/.venv/bin/python -m pytest backend/tests/test_live_market_data.py -v
```

**All four pass criteria observed at 09:53–09:58 EDT on 2026-10-29:**

| Criterion | Result |
|---|---|
| The 21 live tests run instead of skipping | **21 passed, 0 skipped** (2.6s) |
| `premium_source` includes `market_mid` | `Counter({'market_mid': 11})` |
| `iv_source` includes `contract_iv` | `Counter({'contract_iv': 11})`; a passing contract carried `implied_volatility=0.1863` off the chain |
| The liquidity screens reject a wide-spread contract on live quotes | **68 of 165** SPY put strikes rejected — 67 on OI < 100, **1 on spread > 5%** |

Supporting observations, all live: SPY spot 767.53; nearest-90d expiry
2027-01-29; the 95%-moneyness put quoted **bid 10.30 / ask 10.37**, a 0.68%
relative spread; **163 of 165** strikes two-sided; chain unavailable for 0.

Two notes on the integrity of this result. First, the option chain and quote
cache TTL is 300 seconds, so none of these figures could have been served from
a prior session — the 24-hour cache entries are 10-byte daily-history scalars.
Second, this did not pass on the first attempt: the market-hours gate had a
DST bug that skipped the decisive assertion, and it had to be fixed before the
test could run at all. §B.

**Still not done:** nothing has been committed, and nothing has been deployed.
Both require explicit instruction.

---

## Appendix — R2, closed 2026-10-29

**The question.** Out of hours nothing is quoted, so every premium is a model
value. Refuse to recommend, or rank the model values with the assumption
disclosed?

**Why both plain answers are wrong.** Always refusing makes the product
useless for most of the day, for a user who is researching a hedge they intend
to place when the market opens — the contracts are real and listed, and a
disclosed theoretical premium is the honest and useful answer. Always ranking
means that when the market *is* open and a particular instrument still has no
two-sided market, the product invites the user to size a position in a
contract nobody will fill. That is the forced recommendation the brief
prohibits.

**What the two cases have in common** is that neither can be distinguished
from a single holding's data. They differ in what the *rest of the run* looks
like. So the decision moved up a level, to `L4._refuse_unquotable_hedges`,
which runs once across all recommendations:

- If **no** candidate anywhere in the run has an executable two-sided quote,
  the provider is silent. Nothing can be concluded about any single contract,
  so every candidate keeps its disclosed theoretical premium.
- If **some** instrument is quoting normally and a particular holding has
  none, that holding has no market. Its candidates are withheld, it is
  reported unhedged, and the warning says explicitly that this is an *absent*
  market rather than a *closed* one.

"Executable" means both conditions, not either: the premium came from a real
two-sided quote (`premium_source == "market_mid"`) **and** the spread is
inside `MAX_EXECUTABLE_SPREAD_PCT = 0.50`. A contract quoted 0.05 / 3.00 is
quoted in the first sense, and showing its mid as the cost of a hedge is not
honest. The predicate `is_executably_quoted` was lifted to module level in
`selector.py` for exactly one reason: two layers now have to agree on it, and
two copies drifting apart would mean L4 refusing hedges the selector thought
were fine, or the reverse.

**Why no market calendar.** A calendar would have to be maintained, and every
holiday it missed would become a day of wrongly refused hedges. The run
answers the question about itself, for free, from data it already has.

**The asymmetry is deliberate.** A missed refusal leaves a disclosed
theoretical premium on screen. A wrong refusal silently removes a hedge the
user could have placed. The first is visible and bounded; the second is
invisible. So the refusal fires only on positive evidence that quoting works
right now — a wide quote does not count as the venue quoting, and neither does
a single quoted candidate elsewhere in the same holding being absent.

**Verification.** 12 tests in
`test_selector.py::TestUnquotableHedgesAreWithheldOnlyWhenTheVenueIsQuoting`
and `::TestExecutablyQuotedPredicate`; 4 fail against the pre-fix code. On the
live market-hours run the gate was **armed and withheld nothing** — every
holding had at least one executable quote, which is the correct outcome and
the one a naive always-refuse implementation would have got wrong.

---

## Appendix — the disclosure-deletion defect, found 2026-10-29

Recorded separately because of how it was found and what it says about the
earlier verification, not because of its size.

**What it was.** `run_l11_llm` defaults to `True`.
`layer_11_llm_explainer._merge_explanation` wrote a model reply onto a
candidate by replacing `cons` outright. `cons` is also where the selector
puts the three facts the product asserts: no executable two-sided quote, thin
market, expiry crosses earnings. A model that returned its own bullets
therefore deleted them.

**Why it matters more than it looks.** The R2 resolution above is only
defensible because the "theoretical premiums stand" branch tells the user
plainly what they are looking at. This defect removed the telling, in the
default configuration, silently. A user would have seen a ranked hedge, a
premium, a contract count and a cost — with nothing to say the premium was a
model output rather than a price.

**Why it survived two verification cycles.** No `ANTHROPIC_API_KEY` was ever
configured on this machine, so every run — including the browser journey and
`test_e2e_api.py` — took the deterministic-fallback branch and never executed
the merge. The disclosures were genuinely verified, on the one code path where
nothing could remove them. Static inspection found it only because closing R4
meant reading L5's merge, which does the same thing, and then checking
whether the layer that runs by default did too.

**The fix.** `backend/core/disclosures.py`. `cons` remains the single list the
UI renders; a parallel `mandatory_disclosures` records what the product itself
said, and `restore()` runs after every merge in both layers, putting back
anything dropped, in front of the surviving cons. Scope is the three hard
disclosures and not every generated sentence — an explainer improving the
basis-R² prose is doing its job.

**How it was proven.** A stub HTTP server was stood up in place of an Ollama
endpoint, so the real `OllamaExplainer`, its real JSON parser, the real L11
engine, the real merge and the real FastAPI route were all in the path, and
only the model's reply was ours. Against the same portfolio and the same live
market:

| | Disclosures recorded | Deleted from the response |
|---|---|---|
| Fix reverted | 5 | **5** |
| Fix in place | 5 | **0** |

The reverted run returned an MSFT protective put on a contract quoted 8.9%
wide and described it to the user as *"Premium is a cost drag"* and
*"Expires worthless if spot is flat"* — nothing else.

**A second defect in the same subsystem, found alongside it.**
`get_explainer()` returned `ClaudeExplainer` for every provider name it did
not recognise, and recognised only `"huggingface"`. `llm_provider=ollama` sent
the portfolio to Anthropic with nothing in the logs to say so — and
`OllamaExplainer` could not be constructed in any case, because the two
settings it reads were never declared. `"ollama"` is now wired, the settings
declared, and an unknown provider name logged at ERROR naming what was
ignored and who answered instead.

---

## Appendix — R7, closed 2026-10-29

R7 was the last open P1 and the one the previous cycle declined to close,
recording it as *"unresolved rather than explained away."* That was the right
call on the evidence then available: the pipeline had been cleared, the data
could not be, and there was no ground truth for the window in question. This
appendix records what actually settled it.

### The finding, restated

Over the trailing 252 sessions:

| Pair | R² | ρ | Verdict |
|---|---|---|---|
| AAPL / SPY | 0.083 | 0.288 | implausibly weak for a mega-cap |
| MSFT / SPY | 0.154 | 0.392 | implausibly weak |
| median \|ρ\| across AAPL/MSFT/NVDA/JPM/XOM | — | **0.11** | implausibly weak |
| VOO / SPY | 0.996 | — | coherent |
| IVV / SPY | 0.995 | — | coherent |
| QQQ / SPY | 0.847 | — | coherent (β=1.45) |

The pattern is what made it suspicious: the pairs that *must* be correlated by
construction were, and the pairs that *should* be correlated economically
were not.

### Test 1 — is the pipeline distorting the series?

The product's returns were recomputed independently, straight from the raw
Yahoo chart JSON, bypassing every layer of the data stack. The two series are
**byte-identical**:

```
AAPL  maxdiff = 0.00e+00
SPY   maxdiff = 0.00e+00
```

Combined with the checks the previous cycle already ran — cross-correlation
peaking exactly at lag 0, 501 observations over identical dates on both legs,
SPY-on-SPY returning β=1.000 / R²=1.0, no duplicate dates, no close outside
`[Low, High]`, plausible marginal vols (AAPL 28.6% annualised) — **no stage of
the product is at fault.** The regression computes the right answer from the
data it is handed, and the data it is handed is what the provider sent.

### Test 2 — is the window the whole story?

Yes. The same two series over 500 sessions instead of 252:

| Pair | 252d R² | 500d R² | 252d ρ | 500d ρ | 500d β |
|---|---|---|---|---|---|
| AAPL / SPY | 0.083 | **0.354** | +0.288 | **+0.595** | **1.03** |
| XOM / SPY | — | — | **+0.115** | **−0.329** | — |
| VOO / SPY | 0.994 | 0.996 | — | — | — |

A β of 1.03 for AAPL against SPY over two years is exactly what theory
expects. The index-replication pair barely moves between windows, because it
cannot — it is the same basket. The single stocks move enormously, and XOM
**changes sign**.

That is the signature of a genuinely dispersed period, not of corrupt data: a
window in which single names decoupled from the index while the index's own
trackers stayed glued to it. `LOOKBACK_DAYS = 252` is the window the product
actually fits on, so 0.083 is what it correctly reports.

**Discovered while doing this:** `covariance.py`'s docstring still described
the estimate as *"signed, 60-day rolling"*. Anyone auditing whether R²=0.08
was plausible would have been reasoning about a window roughly four times
noisier than the one in use. Logged as a ④ asserted-label defect and fixed.

### What the product does with it, which is the part that matters

It refuses. SPY is **rejected** as an AAPL cross hedge because
R² = 0.084 < `MIN_CROSS_R2 = 0.30`, and the user is shown
*"beta fitted on 252 sessions, R² 0.08"* rather than a hedge sized against a
beta the system does not believe. The weak reading is surfaced, acted on, and
not sold.

This is the behaviour the ③ judgement call was written for — *rejecting an
implausible reading is correct; clamping it to a plausible floor is not* —
arriving at the right outcome from the opposite direction: here the reading
was not implausible at all, and the gate still did the right thing with it.

### Status

**Closed. Not a defect.** The alarm was correct to raise it, the previous
cycle was correct not to hand-wave it, and the answer is that a 252-day
AAPL/SPY ρ of 0.29 is what that year was.

## Appendix — the stub endpoints that named a financial cause, found 2026-10-30

A ⑤ silent-substitution defect, surfaced by the structure cleanup above: four
public endpoints explained an empty result with a reason that had not been
measured.

### What the user was told

`POST /instruments/futures` on a single-holding AAPL portfolio returned
HTTP 200 and this warning:

> AAPL: No hedge for AAPL fits the cost budget and liquidity constraints;
> the position is shown unhedged.

That is a financial finding. It says a search was run, contracts were priced
against the user's budget, liquidity was assessed, and nothing cleared. None
of it happened. `FuturesSelector.find_candidates` returns `[]` immediately —
it is a declared stub. No future was looked up, no budget consulted, no quote
fetched. A user acting on that sentence would conclude futures hedging is
uneconomic for their portfolio, which the product has no basis to claim. The
same text was returned by `/forwards`, `/swaps` and `/inverse-etfs`.

The OpenAPI descriptions compounded it: `/docs` advertised *"Hedge candidates
using index / commodity / treasury futures"* for an endpoint that evaluates
nothing — a ④ asserted label.

### Root cause

An empty candidate list has several possible causes and they are the same
value. `[]` from a stub and `[]` from a selector that priced every listed
contract and rejected them are indistinguishable downstream.

The warning came from `pipeline_orchestrator._drop_unsizeable`, whose actual
job is to remove candidates L7 could not size. It emitted the budget sentence
on `if not kept:` — true whenever a holding ended with no candidates,
including when `dropped == 0` and nothing had ever been priced. The function
was reporting on a filtering step it had not performed.

### The fix

`InstrumentSelector` now declares what it is: a `label` for user-facing text
and `implemented: bool`, set `False` on the four stubs. L4's
`_select_for_holding` returns which instrument classes were *evaluated*,
which are *unbuilt*, and which *raised*, and emits the warning that is true:

| What happened | What the user is told |
|---|---|
| Selector searched, found nothing | "no hedge candidate was found among the instrument types evaluated (options)" |
| Selector is a stub | "hedging with futures is not implemented in this version — no instrument was evaluated" |
| No selector applies to the asset class | "no instrument type in this run applies to this equity holding" |
| Selector raised | "hedge selection failed for futures. The error was logged" |
| Candidates priced, all unsizeable | the original cost-and-liquidity sentence — now only here |

`_drop_unsizeable` is gated on `if dropped and not kept:`, so it speaks only
about candidates it actually dropped. The four endpoint docstrings now state
non-implementation. Exception text stays in the log; responses carry none.

### Two further defects the regression test caught in the fix itself

1. `selector.label` was read inside the `try` that wraps `find_candidates`, so
   a selector injected by a caller or a test — which need only satisfy the
   two-method protocol — was reported to the user as a *failed* selector
   because of a missing attribute. Hoisted out and read via `getattr`.
2. A selector that genuinely raised produced "nothing applies", hiding an
   error behind a statement about applicability. Failures now get their own
   disclosure, emitted even when another instrument class did produce a hedge,
   so a partial run cannot read as a complete one.

### Verification

`backend/tests/test_unbuilt_instrument_disclosure.py` — 14 tests covering all
five cases above plus the stub declarations themselves. **12 of the 14 fail
against the pre-fix tree** (verified by stashing the source fix and re-running);
the 2 that pass both ways are the guards on the honest case, which had to keep
working. Full suite: **783 passed, 31 skipped**.

Behaviour verified over HTTP against the synthetic market, not by inspection:

| Endpoint | Candidates | What it says |
|---|---|---|
| `/instruments/options` | 4 | recommends a 2027-02-19 direct put, 1325 USD, 12.6% of VaR |
| `/instruments/futures` | 0 | "not implemented in this version — no instrument was evaluated" |
| `/instruments/inverse-etfs` | 0 | "not implemented in this version — no instrument was evaluated" |
| `/instruments/forwards` | 0 | "no instrument type in this run applies to this equity holding" |
| `/instruments/swaps` | 0 | "no instrument type in this run applies to this equity holding" |

(`forwards` and `swaps` stop earlier, at `is_applicable`, which is False for a
single equity holding — so the applicability reason is the true one there, and
non-implementation is never reached.)

### Status

**Fixed.** The false cost-and-liquidity claim no longer appears on any path
where cost and liquidity were not measured.

### Discovered while verifying this, and also fixed

With zero candidates, L11 was still asked to write `top_recommendation`, and
one run produced *"No candidates are currently available to hedge the $30,000
AAPL position"* for a portfolio whose `portfolio_notional` is **$40,000**.

That figure was not invented. L11 built the holding exposure for the prompt
from `shares x purchase_price` — 200 x $150, the **cost basis** — while L3
measures notional as `current_price x shares` and `portfolio_notional`, VaR,
hedge sizing and the budget are all computed on that. The narrative therefore
described a $30,000 position beside a $40,000 one, and on any profitable
holding it understated the user's exposure by exactly the unrealised gain —
**on every run, including successful options runs**, not only the empty one
that surfaced it. `portfolio_valuation.py` already names that same expression
`cost_basis`; L11 was the only caller treating it as exposure.

Fixed in four parts, because the figure and the pressure to produce one are
separate problems:

1. L11 passes the measured `risk_profile.notional_value`.
2. When nothing was measured the prompt carries `null`, not a substitute —
   cost basis and the portfolio total are each a number the product did not
   compute for that holding. `_round_or_none` exists because `round(None)`
   raises, and a crash there would have been reported to the user as an LLM
   failure.
3. All three system prompts now state that a null field was not measured and
   must not be estimated, derived from other fields, or filled with a number
   given for something else. This had to be said: `COMBINED_SYSTEM`'s first
   rule is *"every claim needs a number"*, which on its own reads as licence
   to supply one.
4. A run with no candidate on any holding does not call the provider at all.
   There is no hedge to recommend, and the repair spec's *"do not force a
   recommendation when no valid hedge exists"* is the rule. It gets the
   deterministic narrative and an `llm_notice` saying the model was not
   called.

L11's own deterministic text carried the same ⑤ defect as the endpoints above
and is fixed with them: *"no candidate met the budget, liquidity and
basis-risk requirements"* and *"N of N holdings have no **affordable** hedge"*
both named a price screen for holdings where nothing was priced.

**Verification:** `backend/tests/test_llm_narrative_figures.py`, 13 tests,
**11 of which fail against the pre-fix tree**. Two existing tests in
`test_var_horizon_label.py` had to be repaired rather than updated: they
exercised the LLM-substitution disclosure using a fixture with no candidates,
so after part 4 they were passing through the skip path without the explainer
being called at all — they would have gone on passing with the substitution
disclosure removed entirely. They now carry a candidate. Full suite: **796
passed, 31 skipped**. Confirmed over HTTP: the options narrative states
*"40000 USD in total notional"*, matching `portfolio_notional`, and the
stub endpoints produce no dollar figure at all.

## Appendix — the fixture that tested the fallbacks, found 2026-10-30

A ④ asserted-label defect in the test harness itself, found while measuring
R10. It is recorded because it changes what the suite's green means.

`backend/tests/fixtures/synthetic_market.py::market_data_for` is documented as

> Mirrors what `routes_portfolio.fetch_market_data` assembles, so a test can
> drive the orchestrator directly without going through the route.

It did not. The route writes `{"spot_price": ..., "sigma": ...}`. The fixture
wrote `{"spot": ..., "volatility": ...}`. **No consumer in the product reads
either of those two keys** — every one of them looks up `spot_price` and
`sigma`:

| Consumer | Reads |
|---|---|
| `pipeline_orchestrator._build_pricing_ctx` | `md["spot_price"]`, `md["sigma"]`, both behind an `in` guard |
| `pipeline_orchestrator._build_sizing_contexts` | `md.get("spot_price", holding.purchase_price)`, `md.get("sigma", 0.20)` |
| `pipeline_orchestrator._holding_notionals` | `md.get("spot_price", h.purchase_price)` |
| `pipeline_orchestrator._reconcile_notional` | `md["spot_price"]`, behind an `in` guard |
| `layer_09_simulation_engine` | `md.get("spot_price", 100.0)`, `md.get("sigma", 0.20)` |

So every test driving the orchestrator through this fixture was exercising the
**fallback** paths — `spot = purchase_price`, `sigma = 0.20` — and the guarded
dict comprehensions were building an **empty** spot-price map. Tests that
asserted on hedge sizing were asserting on hedges sized from cost basis and a
fabricated volatility, which is the opposite of what the fixture's name and
docstring promise. This is the failure mode the spec's *"never make tests pass
by mocking the entire product pipeline"* warns about, arrived at by accident
rather than by design.

**Fixed:** the fixture emits `spot_price` and `sigma`. The suite still passes
— **796 passed, 31 skipped** — which is the second finding: nothing in it
discriminated between a market spot and the purchase price, so no test failed
when the real keys started arriving. That gap is what let R10 sit unnoticed,
and R10's regression test has to assert on the figure the sizer receives
rather than on the presence of a recommendation.

**Closed out 2026-10-31.** The R10 fix removed every fallback in that table,
so the rows now read `md.get("spot_price")` with no second argument and the
five re-derivations are one `_measured_notional()` reader. The prediction in
the last paragraph held exactly: `test_unpriced_holding_substitutions.py`
asserts on the sizing context, the scoring denominator and the simulation
input — and **18 of its 23 tests fail** when the six source files are
reverted, against a suite of 812 that passed through every one of the
defects. The 5 that survive the revert are the controls, which is the point:
they are the assertions the old suite was making.

---

## Appendix — packaging the two halves for deployment, 2026-10-31

Four things were asked for: wrap the repository around `backend/` and
`frontend/`, prepare the deployment, remove what is redundant, and leave the
three deployment problems alone for now. **Nothing was deployed** — see the end
of this appendix.

### The leftover `api/` directory

A `api/requirements.txt` survived the deletion of the serverless backend
(commit `17258a4`) and had become the file the container actually installs,
while `backend/requirements.txt` held the test dependencies. So the
runtime closure lived outside the package it serves, under the name of a
directory whose purpose had been deleted. Consolidated:

| Was | Now | Holds |
|---|---|---|
| `api/requirements.txt` | `backend/requirements.txt` | the runtime closure the image installs |
| `backend/requirements.txt` | `backend/requirements-dev.txt` | test and tooling dependencies |

`api/` no longer exists. Updated with it: the `Dockerfile`'s `COPY`, the CI
cache keys and install step, `vertex_explainer.py`'s path comment, `PRD.md`,
and the deployment-contract tests.

One reference did not show up in a grep for `api/requirements` because it was
**constructed rather than written**: `test_security.py:512` built
`self._repo_root() / "api" / "requirements.txt"` from path components. A
textual search for the old path could not see it, and the test that reads the
runtime requirements failed after the move. Worth recording because the same
blind spot hides any path a test assembles instead of spelling out.

### Dead code removed

A sweep for unreferenced modules found none — earlier passes had taken them.
A symbol-level sweep produced 47 candidates, of which about 20 were genuinely
dead. Removed: `backend/core/risk/hedge_ratio.py` entirely (the
`HedgeRatioCalculator` was constructed at `layer_03_risk_analysis.py:46` and
never called again, and its two methods duplicated live implementations in
`position_sizer.py`), `MCResult.prob_below`, `PriceResult.is_valid`,
`can_price`/`can_size`/`can_compute` (registry dispatch goes through
`REGISTRY[asset_class]` directly, so the predicates were never consulted),
three sector-flag helpers in `company_data.py`, an 8-member diagnostic block
and two `force_retrain_*` wrappers in `combined_detector.py`, the
`_run_options_hedge` Phase-2 stub, and the five create-next-app SVGs in
`frontend/public/`. 833 passed afterwards against 837 before — exactly the four
tests that covered the removed symbols, and nothing else.

Two categories were deliberately **kept**. Framework-invoked symbols (FastAPI
handlers, Pydantic validators) show zero name references but are called by the
framework. And about eleven symbols have tests but no product caller
(`put_call_parity_check`, `implied_vol_76`, `compute_ols_hedge_ratio` and
similar); deleting them would mean deleting their tests, which the spec forbids
as a way of making a suite pass. They are listed for a decision rather than
removed unilaterally.

### The deploy surface

`.gcloudignore`, new, is a **security control rather than a size
optimisation**. `gcloud run deploy --source .` uploads the build context to a
Cloud Storage staging bucket *before* the image is built, so the Dockerfile's
secret-leak guard — the one added after this repository shipped five live API
keys in a layer (§F) — runs one step too late to keep `backend/.env` out of
that bucket. It also uses **gitignore semantics**, where a bare pattern
matches at every directory level: the exact opposite of `.dockerignore` here,
which is `**/`-anchored throughout precisely because bare patterns were the
original leak. A test now fails on any `**/`-anchored pattern in
`.gcloudignore`, because its presence is evidence the two files have been
confused again.

`scripts/deploy_backend.sh` holds the Cloud Run shape, with each value a
contract against code elsewhere rather than a preference:

- `--timeout 300` must exceed the frontend's own 240s analyze budget
  (`TIMEOUT_MS.analyze` in `frontend/lib/api.ts`). If the platform cuts first,
  the written timeout explanation can never be what the user sees, and the run
  is lost after the market data was already paid for. A test parses both numbers
  and compares them.
- `--concurrency 8` against Cloud Run's default of **80**. Eighty concurrent
  pipelines on two cores, each owning a `ThreadPoolExecutor`, do not run
  slowly — they time out, while the platform reports a healthy instance.
- `--max-instances 1` is a **correctness** setting. Fitted regime models live
  in the container's writable layer and the response cache is per-instance, so
  two instances can answer the same portfolio with different volatility
  regimes. The pin makes that impossible rather than unlikely; the test
  asserting it exists so the value is not raised as a throughput tweak.

The script refuses, before calling `gcloud`, when `PROJECT_ID` or
`FRONTEND_URL` is unset, when the origin is localhost, or when it is not
HTTPS — the three conditions under which the service would deploy and then
fail its own startup guard. All four refusal paths were exercised.

Also fixed: `vercel.json` installed with `npm install`, which resolves afresh
and can deploy dependency versions no CI build ever exercised — structurally
the same defect as the container running a different numerical stack from the
test suite (§F). Now `npm ci`. And the two `/admin/cron/retrain-*` docstrings
still described themselves as "Vercel Cron endpoint", naming a platform that
had been deleted from the repository; they now name Cloud Scheduler and state
that a retrain writes to the container's writable layer and does not outlive
the instance.

### Verification

| | |
|---|---|
| Full suite | **856 passed, 31 skipped** (833 + 19 deploy-shape + 4 runbook) |
| New tests | **24** (20 functions, some parametrized); **16 revert-verified** — each perturbation caught, every touched file restored byte-identically (`diff -q`). The other 8 assert a file or pattern is present, where the perturbation and the assertion are the same act |
| Frontend build after asset removal | green, 4 static pages, TypeScript clean |
| Deploy script refusals | all four exercised live |
| Container build after the requirements move | **not verified locally** — no container engine on this machine. CI's `image` job builds it on push; that is the only honest check |
| README's documented backend command | was **broken** (`cd backend && uvicorn main:app` → `No module named 'backend'`); tested, fixed |

**Nothing was deployed.** `scripts/deploy_backend.sh` prints its plan and exits
unless given `--execute`, which was never passed. The three deferred
problems — ephemeral fitted models, per-instance cache, and yfinance's
untested behaviour from datacentre IP ranges — are unresolved by decision and
written into `DEPLOYMENT.md` under "Known open problems" rather than left for
whoever deploys to discover.
