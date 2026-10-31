import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

# Resolve .env relative to this file so it works regardless of CWD
_ENV_FILE = os.path.join(os.path.dirname(__file__), "..", ".env")


class Settings(BaseSettings):
    # Anthropic / LLM
    anthropic_api_key: str = ""
    llm_provider: str = "claude"       # "claude" | "vertex" | "gemini" | "huggingface" | "ollama"
    llm_model: str = "claude-sonnet-4-6"

    # Google Gemini, via the Generative Language REST API.
    #
    # Pinned to an exact stable version on purpose: the API also serves
    # `-preview` builds and shifting `-latest` aliases, and a hedging system
    # should not have the thing writing its narratives change underneath it
    # between runs. (This repo already has a defect — 36 — about a dev
    # prerelease reaching the dependency closure.)
    #
    # gemini-3.5-flash was chosen by measurement against the real combined
    # prompt captured from a live hedge run, not by reputation. Over three
    # trials each it explained all 7 candidates and filled all 4 portfolio
    # fields whenever the call got through, at 9-20s; gemini-3.6-flash matched
    # it at 9-25s and gemini-3.8-flash took 17-56s and timed out once, which is
    # too slow for a call on the request path. gemini-2.5-flash, -flash-lite
    # and -pro all answer a new project with 404: retired for new users.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"

    # Google Gemini, via Vertex AI. The configured provider.
    #
    # Same model as above, reached through Vertex instead of an API key, so the
    # bake-off that chose gemini-3.5-flash carries over unchanged. What changes
    # is the credential: Vertex authenticates with IAM, so there is no
    # long-lived secret to store in an environment variable, bake into an image
    # or leak through a log, and access is revoked in IAM rather than by
    # rotating a string. Locally that is `gcloud auth application-default
    # login`; in Cloud Run it is the attached service account, with no
    # configuration difference between the two.
    #
    # The location is `global`, measured rather than chosen: the 3.x Gemini
    # publisher models are served only from `global`, and every regional
    # endpoint tried (us-central1 among them) answers them with 404 NOT_FOUND.
    # An empty vertex_project disables the provider rather than guessing one.
    vertex_project: str = ""
    vertex_location: str = "global"
    vertex_model: str = "gemini-3.5-flash"

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
