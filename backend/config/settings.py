from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    # Anthropic / LLM
    anthropic_api_key: str = ""
    llm_provider: str = "claude"       # "claude" | "huggingface"
    llm_model: str = "claude-sonnet-4-6"

    # HuggingFace Inference API
    hf_api_token: str = ""
    hf_model: str = "meta-llama/Llama-3.3-70B-Instruct"

    # Alpaca
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # FRED
    fred_api_key: str = ""

    # NewsAPI
    news_api_key: str = ""

    # App
    environment: str = "development"
    log_level: str = "INFO"
    cache_ttl_seconds: int = 3600

    # CORS
    frontend_url: str = "http://localhost:3000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",   # silently drop unknown .env keys (e.g. stale OLLAMA_*)
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
