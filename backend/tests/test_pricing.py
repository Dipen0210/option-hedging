"""
Phase 1 smoke tests — run once BSM pricer is implemented.
These are stubs now; assertions are filled in as each pricer is built.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_models_import():
    from backend.models.portfolio_models import PortfolioInput, HoldingInput
    from backend.models.hedge_models import InstrumentCandidate, HedgeOutput
    from backend.models.risk_models import RegimeState, RiskProfile
    from backend.models.execution_models import OrderInput, ExecutionResult
    assert True


def test_settings_loads():
    from backend.config.settings import settings
    assert settings is not None
    assert settings.environment in ("development", "production", "test")


def test_sample_portfolio_parses():
    import json
    from backend.models.portfolio_models import PortfolioInput

    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "sample_portfolio.json")
    with open(fixture_path) as f:
        data = json.load(f)

    portfolio = PortfolioInput(**data)
    assert len(portfolio.holdings) == 3
    assert portfolio.holdings[0].ticker == "IWM"
    assert portfolio.total_notional == 600000


def test_cache_set_get():
    from backend.data.data_cache import cache_set, cache_get
    cache_set("test_ns", "key1", value={"foo": 42}, ttl=60)
    result = cache_get("test_ns", "key1")
    assert result == {"foo": 42}


def test_cache_expiry():
    from backend.data.data_cache import cache_set, cache_get
    import time
    cache_set("test_ns", "expire_key", value="should_expire", ttl=1)
    time.sleep(2)
    result = cache_get("test_ns", "expire_key")
    assert result is None
