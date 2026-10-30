"""
End-to-end integration test through the HTTP API.

Everything else in this suite stops short of the boundary the product
actually exposes. `test_pipeline.py` drives `HedgeOrchestrator.run` directly,
and its `TestInputValidation` class exercises the pydantic models rather than
a request, so nothing covered `POST /portfolio/analyze`: not the route's
market-data assembly, not its timeout and exception handling, not the
`HedgeOutput` response contract the frontend reads, and not the sanitisation
of a failure. Four defects found during verification lived in exactly that
gap — the VaR horizon labels, the silent deterministic-narrative
substitution, and the admin/cron authorization matrix — and a unit test on
any single layer would have missed all of them.

The market data provider is replaced with `synthetic_market` and the language
model with an unconfigured provider. Both are network boundaries, not product
logic: the route, all twelve layers, valuation, selection, pricing, sizing,
Greeks, the Monte Carlo pass, scoring, the deterministic narrative, the
data-quality ledger and the response serialisation all run unmodified. That
keeps the test deterministic and offline while still asserting the real
contract. The live-provider counterpart is `test_live_market_data.py`, which
is opt-in because it cannot be deterministic.
"""
import json
import re

import pytest
from fastapi.testclient import TestClient

from backend.tests.fixtures import synthetic_market as sm

pytestmark = pytest.mark.integration


# A multi-asset portfolio: four asset classes, so cross-asset hedging, the
# bond and commodity proxy paths and the per-class benchmark map all run.
PORTFOLIO = {
    "holdings": [
        {"position_type": "stock", "ticker": "AAPL", "shares": 400,
         "purchase_price": 185.0, "purchase_date": "2025-01-15",
         "asset_class": "equity"},
        {"position_type": "stock", "ticker": "IWM", "shares": 300,
         "purchase_price": 220.0, "purchase_date": "2025-03-02",
         "asset_class": "equity"},
        {"position_type": "stock", "ticker": "TLT", "shares": 500,
         "purchase_price": 92.0, "purchase_date": "2024-11-20",
         "asset_class": "bond"},
        {"position_type": "stock", "ticker": "GLD", "shares": 200,
         "purchase_price": 240.0, "purchase_date": "2025-05-08",
         "asset_class": "commodity"},
    ],
    "hedge_horizon_days": 120,
    "protection_level": 0.15,
    "max_hedge_cost_pct": 0.03,
}

# Patterns that must never reach a client. A traceback, a library type name or
# an attribute repr in a response body is both an information leak and
# unreadable to a user.
#
# These are regexes, not plain substrings, because the response legitimately
# contains English prose: a rationale reads "Direct hedge on AAPL itself. R2=1.0",
# and a substring search for "self." matched the word "itself". A marker that
# fires on correct output is worse than no marker — it trains the reader to
# ignore the check. So the attribute-repr markers are anchored on a word
# boundary and require an identifier character after the dot.
INTERNAL_MARKERS = (
    r"Traceback",
    r'File "/',
    r"yfinance",
    r"\bpandas\b",
    r"\bnumpy\b",
    r"sqlite3",
    r"psycopg",
    r"\b(?:Key|Value|Attribute|Type|Index|ZeroDivision)Error\b",
    r"site-packages",
    r"backend/",
    r"\bbackend\.\w",
    r"__init__",
    r"\bself\.\w",
)


def _leaks(text: str) -> list:
    return [m for m in INTERNAL_MARKERS if re.search(m, text)]


# Module-scoped: a single pipeline run (Monte Carlo included) takes several
# seconds, and every assertion below reads the same response rather than
# mutating it. A function-scoped fixture re-ran the whole product once per
# test. `monkeypatch` is function-scoped by construction, so the patches are
# installed through an explicit context that lives as long as this fixture.
@pytest.fixture(scope="module")
def client():
    """
    The real app, with only its two outbound network boundaries replaced.
    """
    with pytest.MonkeyPatch.context() as mp:
        sm.install(mp)

    # No language model configured — the common deployment, and the one that
    # must still produce a narrative and say where it came from. Stubbing the
    # provider keeps the test offline; the narrative itself is still generated
    # by L11 from the computed figures.
        import backend.engines.layer_11_llm_explainer as l11

        class _NoProvider:
            provider_name = "none"

            def is_available(self) -> bool:
                return False

        mp.setattr(l11, "get_explainer", lambda provider=None: _NoProvider())

        from backend.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture(scope="module")
def analysis(client):
    """
    One analysis of PORTFOLIO, shared by every read-only assertion. Tests must
    not mutate it; a test that needs a different request body posts its own.
    """
    r = client.post("/portfolio/analyze", json=PORTFOLIO)
    assert r.status_code == 200, r.text
    return r.json()


# ── The request completes and returns the documented shape ───────────────────

class TestTheRequestSucceeds:

    def test_health_is_served(self, client):
        assert client.get("/health").status_code == 200

    def test_analysis_returns_200(self, analysis):
        assert analysis["recommendations"]

    def test_one_recommendation_entry_per_holding(self, analysis):
        tickers = {r["asset_ticker"] for r in analysis["recommendations"]}
        assert tickers == {"AAPL", "IWM", "TLT", "GLD"}

    def test_the_response_carries_the_run_time(self, analysis):
        assert analysis["run_time_seconds"] > 0


# ── Valuation: market value, not cost basis ──────────────────────────────────

class TestValuation:

    def test_notional_is_valued_from_current_prices(self, analysis):
        declared_cost = sum(
            h["shares"] * h["purchase_price"] for h in PORTFOLIO["holdings"]
        )
        notional = analysis["portfolio_notional"]
        assert notional > 0
        # Priced from the fixture's current prices, so it must not simply
        # echo the cost basis back.
        assert notional != pytest.approx(declared_cost)

    def test_the_budget_is_the_requested_share_of_market_value(self, analysis):
        expected = analysis["portfolio_notional"] * PORTFOLIO["max_hedge_cost_pct"]
        assert analysis["hedge_budget_total"] == pytest.approx(expected, rel=1e-6)


# ── Every recommended contract is a real listed contract ─────────────────────

class TestContractsAreReal:

    @staticmethod
    def _candidates(analysis):
        return [c for r in analysis["recommendations"] for c in r["candidates"]]

    def test_candidates_were_produced(self, analysis):
        assert self._candidates(analysis)

    def test_every_option_candidate_names_a_contract(self, analysis):
        missing = [
            c for c in self._candidates(analysis)
            if c.get("option_type") and not c.get("contract_symbol")
        ]
        assert missing == []

    def test_the_occ_symbol_encodes_the_strike_it_claims(self, analysis):
        wrong = []
        for c in self._candidates(analysis):
            sym, strike = c.get("contract_symbol"), c.get("strike")
            if not sym or strike is None:
                continue
            if not sym.endswith(f"{int(round(strike * 1000)):08d}"):
                wrong.append((sym, strike))
        assert wrong == []

    def test_expiries_are_dated_in_the_future(self, analysis):
        past = [
            c["expiry_date"] for c in self._candidates(analysis)
            if c.get("expiry_date") and c["expiry_date"] <= sm.AS_OF.isoformat()
        ]
        assert past == []


# ── Pricing and volatility provenance is stated ──────────────────────────────

class TestProvenanceIsDisclosed:

    @staticmethod
    def _candidates(analysis):
        return [c for r in analysis["recommendations"] for c in r["candidates"]]

    def test_every_candidate_states_where_its_premium_came_from(self, analysis):
        sources = {c.get("premium_source") for c in self._candidates(analysis)}
        assert sources and None not in sources and "" not in sources

    def test_every_candidate_states_which_volatility_was_used(self, analysis):
        sources = {c.get("iv_source") for c in self._candidates(analysis)}
        assert sources and None not in sources and "" not in sources

    def test_a_fully_quoted_market_prices_off_the_quote(self, analysis):
        # The fixture publishes two-sided quotes, so the market-mid path is
        # the one that must be taken. Out of hours this is realized_vol /
        # model, which is why the live test can only skip here.
        sources = {c.get("premium_source") for c in self._candidates(analysis)}
        assert "market_mid" in sources

    def test_the_data_quality_ledger_is_populated(self, analysis):
        dq = analysis.get("data_quality") or {}
        assert dq.get("source_counts")


# ── Risk estimates reach the client with their samples ───────────────────────

class TestRiskEstimateProvenanceIsExposed:
    """
    L3 measures what every risk figure rests on — the beta regression's R2,
    standard error and observation count, the VaR method and its sample, the
    hedge-ratio regression, and which factor exposures could not be measured
    at all. L12 dropped all of it, so the response stated a beta and a VaR
    with no way to tell a fit on 130 observations from one on 2,500, or an
    assumed volatility from an estimated one. These numbers size the
    contracts, so the qualification has to travel in the same response rather
    than behind a second request a client may never make.
    """

    @staticmethod
    def _candidates(analysis):
        return [c for r in analysis["recommendations"] for c in r["candidates"]]

    def test_every_recommendation_carries_risk_provenance(self, analysis):
        missing = [
            r["asset_ticker"] for r in analysis["recommendations"]
            if not r.get("risk_provenance")
        ]
        assert missing == [], f"no risk_provenance for {missing}"

    def test_the_var_names_the_method_that_produced_it(self, analysis):
        # An assumed flat volatility and a Cornish-Fisher estimate are
        # different claims about the position and must not be indistinguishable.
        known = {
            "parametric_cornish_fisher",
            "assumed_vol_fallback",
            "normal_approximation",
        }
        for rec in analysis["recommendations"]:
            rp = rec["risk_provenance"]
            if rp["data_available"]:
                assert rp["var_method"] in known, (
                    f"{rec['asset_ticker']}: var_method={rp['var_method']!r}"
                )

    def test_an_estimated_var_reports_its_sample(self, analysis):
        for rec in analysis["recommendations"]:
            rp = rec["risk_provenance"]
            if rp["var_method"] == "parametric_cornish_fisher":
                assert rp["var_n_obs"] and rp["var_n_obs"] > 0, rec["asset_ticker"]

    def test_a_measured_beta_reports_r2_stderr_and_sample(self, analysis):
        for rec in analysis["recommendations"]:
            rp = rec["risk_provenance"]
            if not rp["beta_available"]:
                continue
            for field in ("beta_r2", "beta_stderr", "beta_n_obs"):
                assert rp[field] is not None, f"{rec['asset_ticker']}: {field} is None"
            assert rp["beta_n_obs"] > 0

    def test_an_unmeasurable_estimate_is_null_and_never_zero(self, analysis):
        # Zero R2 and zero standard error both read as measurements. The
        # pipeline's rule is that an unmeasured quantity is absent, not 0.
        for rec in analysis["recommendations"]:
            rp = rec["risk_provenance"]
            if not rp["beta_available"]:
                assert rp["beta_stderr"] is None, rec["asset_ticker"]
                assert rp["beta_n_obs"] is None, rec["asset_ticker"]

    def test_every_candidate_says_what_its_basis_r2_rests_on(self, analysis):
        # The UI grades basis_risk_r2 as "Low / Moderate / High basis risk".
        # That grade is a claim about the hedge, so the response has to say
        # whether the number is an estimate, a tail statistic, or definitional.
        known = {"correlation", "tail_correlation", "identity"}
        bad = [
            (c["ticker"], c.get("basis_risk_source"))
            for c in self._candidates(analysis)
            if c.get("basis_risk_source") not in known
        ]
        assert bad == [], f"candidates with no usable basis_risk_source: {bad}"

    def test_a_correlation_based_basis_r2_carries_its_sample(self, analysis):
        measured = [
            c for c in self._candidates(analysis)
            if c.get("basis_risk_source") == "correlation"
        ]
        assert measured, "the fixture produces cross/macro hedges; none were found"
        for c in measured:
            assert c["basis_risk_n_obs"] and c["basis_risk_n_obs"] > 0, c["ticker"]
            assert c["basis_risk_window_days"], c["ticker"]
            assert c["basis_risk_stderr"] is not None, c["ticker"]

    def test_a_direct_hedge_claims_identity_not_a_measurement(self, analysis):
        # R2 = 1 on the same underlying is definitional. Reporting a sample
        # for it would imply a regression nobody ran.
        direct = [
            c for c in self._candidates(analysis)
            if c["hedge_category"] == "direct_hedge" and c["basis_risk_r2"] == 1.0
        ]
        assert direct, "the fixture portfolio produces direct hedges"
        for c in direct:
            assert c["basis_risk_source"] == "identity", c["ticker"]
            assert c["basis_risk_n_obs"] is None, c["ticker"]
            assert c["basis_risk_stderr"] is None, c["ticker"]


# ── The budget constraint is enforced, not just reported ─────────────────────

class TestBudgetIsEnforced:

    def test_the_selected_hedges_fit_the_budget(self, analysis):
        assert analysis["hedge_portfolio_cost"] <= analysis["hedge_budget_total"]

    def test_utilization_is_consistent_with_the_figures(self, analysis):
        total = analysis["hedge_budget_total"]
        if total > 0:
            assert analysis["hedge_budget_utilization"] == pytest.approx(
                analysis["hedge_portfolio_cost"] / total, rel=1e-3
            )

    def test_contract_counts_are_whole_and_positive(self, analysis):
        bad = [
            c["n_contracts"]
            for r in analysis["recommendations"] for c in r["candidates"]
            if not isinstance(c["n_contracts"], int) or c["n_contracts"] <= 0
        ]
        assert bad == []


# ── The VaR horizon is reported, never asserted ──────────────────────────────

class TestVarHorizonReachesTheClient:

    def test_every_unhedged_risk_block_carries_the_requested_horizon(self, analysis):
        horizons = [
            r["unhedged_risk"]["horizon_calendar_days"]
            for r in analysis["recommendations"] if r.get("unhedged_risk")
        ]
        assert horizons
        assert set(horizons) == {PORTFOLIO["hedge_horizon_days"]}

    def test_the_narrative_does_not_call_it_a_one_day_var(self, analysis):
        # It was "One-day 95% VaR" for a 120-day figure: understated ~9x.
        assert "one-day" not in (analysis["portfolio_summary"] or "").lower()

    def test_the_narrative_names_the_horizon_it_used(self, analysis):
        summary = analysis["portfolio_summary"] or ""
        if "VaR" in summary:
            assert f"{PORTFOLIO['hedge_horizon_days']}-day hedge horizon" in summary


# ── A narrative that the model did not write says so ─────────────────────────

class TestNarrativeDisclosure:

    def test_a_narrative_is_produced_without_a_language_model(self, analysis):
        assert (analysis["portfolio_summary"] or "").strip()
        assert (analysis["top_recommendation"] or "").strip()

    def test_and_it_is_disclosed_as_not_coming_from_the_model(self, analysis):
        notice = (analysis["llm_notice"] or "").strip()
        assert notice
        assert "computed figures" in notice


# ── Nothing is forced, and what is missing is said out loud ──────────────────

class TestHonestGaps:

    def test_a_holding_with_no_viable_hedge_is_reported_unhedged(self, analysis):
        for rec in analysis["recommendations"]:
            if not rec["candidates"]:
                # Must be explained somewhere the user will see.
                blob = json.dumps(analysis["warnings"])
                assert rec["asset_ticker"] in blob, (
                    f"{rec['asset_ticker']} has no candidates and no warning"
                )

    def test_warnings_are_a_list_of_strings(self, analysis):
        assert all(isinstance(w, str) for w in analysis["warnings"])


# ── Nothing internal leaves the process ──────────────────────────────────────

class TestNoInternalsLeak:

    # The leak checks below are only worth as much as the detector is. These
    # two tests pin it: it must fire on a real traceback and stay quiet on the
    # prose the product legitimately emits. Without them, a marker that had
    # been narrowed to silence a false positive could silently stop detecting
    # anything, and every assertion in this class would pass vacuously.
    def test_the_detector_fires_on_a_real_traceback(self):
        leaked = '''
Traceback (most recent call last):
  File "/Users/x/optionQ/backend/data/market_data.py", line 88, in fetch
    return self._session.get(url).json()
KeyError: 'Close'
'''
        found = _leaks(leaked)
        for expected in (r"Traceback", r'File "/', r"\bself\.\w",
                         r"\b(?:Key|Value|Attribute|Type|Index|ZeroDivision)Error\b",
                         r"backend/"):
            assert expected in found, f"detector missed {expected}"

    def test_the_detector_does_not_fire_on_product_prose(self):
        # Verbatim shapes the pipeline really produces.
        for benign in (
            "Direct hedge on AAPL itself. R2=1.0 - zero basis risk.",
            "Puts protect against downside; the position itself is unchanged.",
            "atm_iv: unavailable (AAPL, GLD) - no quoted strike within 10% of spot",
            "No hedge for TLT fits the cost budget and liquidity constraints.",
        ):
            assert _leaks(benign) == [], benign

    def test_a_successful_response_leaks_nothing(self, analysis):
        # `backend.` and `backend/` appear in module paths only, so a clean
        # response proves no traceback or repr was serialised into it.
        assert _leaks(json.dumps(analysis)) == []

    @pytest.mark.parametrize("payload,label", [
        ({"holdings": [{"position_type": "stock", "ticker": "NOTREAL",
                        "shares": 10, "purchase_price": 100.0,
                        "purchase_date": "2025-01-01"}]}, "unsupported ticker"),
        ({"holdings": [{"position_type": "stock", "ticker": "AAPL",
                        "shares": -5, "purchase_price": 100.0,
                        "purchase_date": "2025-01-01"}]}, "negative shares"),
        ({"holdings": []}, "empty portfolio"),
        ({"holdings": [{"position_type": "stock", "ticker": "AAPL",
                        "shares": 10, "purchase_price": 100.0,
                        "purchase_date": "2025-01-01"}],
          "max_hedge_cost_pct": 50.0}, "absurd budget"),
    ])
    def test_a_rejected_request_is_rejected_cleanly(self, client, payload, label):
        r = client.post("/portfolio/analyze", json=payload)
        assert r.status_code == 422, f"{label}: got {r.status_code}"
        assert _leaks(r.text) == [], f"{label} leaked: {_leaks(r.text)}"

    def test_malformed_json_is_rejected_cleanly(self, client):
        r = client.post(
            "/portfolio/analyze",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422
        assert _leaks(r.text) == []

    def test_an_unknown_route_is_a_plain_404(self, client):
        r = client.get("/no/such/route")
        assert r.status_code == 404
        assert _leaks(r.text) == []


# ── Privileged endpoints fail closed ─────────────────────────────────────────

class TestPrivilegedEndpointsFailClosed:
    """
    With no secret configured these must refuse, not run. 503 is the
    fail-closed answer when the credential is unset; 401/403 when it is set
    and the caller did not present it. Never 200.
    """

    ENDPOINTS = [
        ("GET", "/admin/training/status"),
        ("POST", "/admin/retrain/fast"),
        ("GET", "/admin/cron/retrain-fast"),
        ("GET", "/admin/cron/retrain-main"),
    ]

    @pytest.mark.parametrize("method,path", ENDPOINTS)
    def test_no_credential_is_refused(self, client, method, path):
        r = client.request(method, path)
        assert r.status_code in (401, 403, 503), f"{path} -> {r.status_code}"

    @pytest.mark.parametrize("method,path", ENDPOINTS)
    def test_a_wrong_credential_is_refused(self, client, method, path):
        r = client.request(
            method, path,
            headers={"X-Admin-Key": "wrong", "X-Cron-Secret": "wrong"},
        )
        assert r.status_code in (401, 403, 503), f"{path} -> {r.status_code}"

    @pytest.mark.parametrize("method,path", ENDPOINTS)
    def test_a_refusal_explains_nothing_internal(self, client, method, path):
        r = client.request(method, path)
        assert _leaks(r.text) == []


# ── 422 bodies are written for users, not for pydantic ───────────────────────
#
# FastAPI's default handler returned pydantic's error objects verbatim, so a
# holding submitted without `position_type` produced
# `"msg": "Unable to extract tag using discriminator 'position_type'"` plus a
# `ctx` naming the discriminator and an `input` echoing the submitted
# portfolio. `frontend/lib/api.ts` copies `msg` into the field error it
# renders, so that sentence reached the user. The earlier leak assertions
# looked for paths, stack frames and module names, none of which this body
# contains, so it passed them all.

_LIBRARY_LEAKS = [
    "discriminator", "union_tag", "pydantic", "ctx", "validation error",
    "Unable to extract tag", "value_error", "model_type", "literal_error",
]


class TestValidationErrorsDoNotLeakLibraryInternals:

    def _post(self, client, body):
        r = client.post("/portfolio/analyze", json=body)
        assert r.status_code == 422, r.status_code
        return r.json()

    def test_a_missing_discriminator_is_explained_in_plain_words(self, client):
        body = self._post(client, {"holdings": [{"ticker": "AAPL", "quantity": 100}]})
        msg = body["detail"][0]["msg"]
        assert "position_type" in msg, msg          # names the field to supply
        assert "discriminator" not in msg.lower(), msg

    def test_no_error_body_mentions_the_validation_library(self, client):
        for payload in (
            {"holdings": [{"ticker": "AAPL", "quantity": 100}]},
            {"holdings": []},
            {"holdings": [{"position_type": "stock", "ticker": "AAPL"}]},
            {"holdings": [{"position_type": "nonsense", "ticker": "AAPL"}]},
        ):
            raw = json.dumps(self._post(client, payload)).lower()
            for leak in _LIBRARY_LEAKS:
                assert leak.lower() not in raw, f"{leak!r} leaked for {payload}: {raw}"

    def test_the_submitted_portfolio_is_not_echoed_back(self, client):
        """
        `input` returned the caller's own position data in the error body.
        A malformed holding does not need to travel back out to be rejected.
        """
        body = self._post(client, {
            "holdings": [{"ticker": "SECRET_TICKER", "quantity": 12345}]
        })
        raw = json.dumps(body)
        assert "SECRET_TICKER" not in raw, raw
        assert "12345" not in raw, raw
        assert all("input" not in e and "ctx" not in e for e in body["detail"]), body

    def test_the_field_location_is_preserved_for_highlighting(self, client):
        """
        The prose drops pydantic's union-variant tag, but `loc` must keep the
        full path or the client cannot mark the offending input.
        """
        body = self._post(client, {
            "holdings": [{"position_type": "stock", "ticker": "AAPL"}]
        })
        locs = [e["loc"] for e in body["detail"]]
        assert all(l[0] == "body" for l in locs), locs
        assert any("shares" in l for l in locs), locs
        # ...and the sentence does not show the variant tag to the user.
        shares = next(e for e in body["detail"] if "shares" in e["loc"])
        assert ".stock." not in shares["msg"], shares["msg"]

    def test_a_range_violation_states_the_actual_limit(self, client):
        body = self._post(client, {
            "holdings": [{"position_type": "stock", "ticker": "AAPL", "shares": 10,
                          "purchase_price": 100.0, "purchase_date": "2024-01-02"}],
            "protection_level": 5.0,
        })
        msgs = " ".join(e["msg"] for e in body["detail"])
        assert "protection_level" in msgs, msgs
        assert "0.5" in msgs, msgs

    def test_every_message_is_a_sentence(self, client):
        for payload in (
            {"holdings": [{"ticker": "AAPL", "quantity": 100}]},
            {"holdings": [{"position_type": "stock", "ticker": "AAPL"}]},
        ):
            for e in self._post(client, payload)["detail"]:
                msg = e["msg"]
                assert msg and msg[0].isupper() or "." in msg, msg
                assert msg.endswith("."), msg
