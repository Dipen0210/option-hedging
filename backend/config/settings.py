import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

# Resolve .env relative to this file so it works regardless of CWD
_ENV_FILE = os.path.join(os.path.dirname(__file__), "..", ".env")


class Settings(BaseSettings):
    # Anthropic / LLM
    anthropic_api_key: str = ""
    llm_provider: str = "claude"       # "claude" | "huggingface" | "ollama"
    llm_model: str = "claude-sonnet-4-6"

    # HuggingFace Inference API
    hf_api_token: str = ""
    hf_model: str = "meta-llama/Llama-3.3-70B-Instruct"

    # Ollama (local). These two were referenced by
    # backend/core/explainer/ollama_explainer.py and never declared here, so
    # constructing that explainer raised AttributeError — the provider has
    # never been reachable.
    ollama_model: str = "llama3"
    ollama_base_url: str = "http://localhost:11434"

    # Alpaca
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # FRED
    fred_api_key: str = ""

    # NewsAPI
    news_api_key: str = ""

    # Administrative access.
    # Empty means NOT CONFIGURED, which the auth dependencies treat as "refuse
    # the request", never as "allow anyone". There is deliberately no default
    # value: a baked-in default key is a published key.
    admin_api_key: str = ""     # guards every /admin/* route
    cron_secret: str = ""       # guards /admin/cron/retrain-* only

    # App
    environment: str = "development"
    log_level: str = "INFO"
    cache_ttl_seconds: int = 3600

    # CORS. Comma-separated list; the first entry is the canonical frontend.
    # Credentialed requests cannot be served to a wildcard origin, so this must
    # name the real deployed origin(s) in production.
    frontend_url: str = "http://localhost:3000"

    @property
    def allowed_origins(self) -> list[str]:
        """Parsed `frontend_url`, blank entries dropped."""
        return [o.strip() for o in self.frontend_url.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.strip().lower() in ("production", "prod")

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",   # silently drop unknown .env keys (e.g. stale OLLAMA_*)
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
