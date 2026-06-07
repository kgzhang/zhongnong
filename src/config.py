"""Configuration for zhongnong-kg v2 pipeline.

All settings load from .env with ZN_ prefix.
Also accepts unprefixed aliases for backward compatibility.
"""
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM
    llm_model: str = Field(
        default="deepseek-chat",
        validation_alias="ZN_LLM_MODEL",
        description="Model ID for the LLM provider",
    )
    llm_api_key: str = Field(
        default="",
        validation_alias="ZN_LLM_API_KEY",
        description="API key for the LLM provider",
    )
    llm_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        validation_alias="ZN_LLM_BASE_URL",
        description="Base URL for the LLM API",
    )
    llm_max_retries: int = 3
    llm_temperature: float = 0.1
    llm_max_tokens: int = 16384
    llm_thinking_enabled: bool = Field(
        default=False,
        validation_alias="ZN_LLM_THINKING_ENABLED",
        description="Enable chain-of-thought thinking mode (disable for batch processing)",
    )

    # Pipeline
    max_char_buffer: int = 16000
    batch_length: int = 32
    max_workers: int = 256
    extraction_passes: int = 1
    context_window_chars: int | None = None

    # Cache
    cache_enabled: bool = True
    cache_dir: Path = Path("data/cache")

    # Paths，生产一定要改
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    checkpoint_dir: Path = Path("data/intermediates")
    schema_dir: Path = Path("schemas")

    # Debug
    debug: bool = False
    show_progress: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ZN_",
        extra="allow",  # accept unprefixed keys from .env as fallback
    )

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings,
                                    dotenv_settings, file_secret_settings):
        """Ensure .env file is read and unprefixed keys work as fallback."""
        return (init_settings, env_settings, dotenv_settings, file_secret_settings)


def _apply_env_fallbacks(settings: Settings) -> Settings:
    """Apply unprefixed .env keys as fallbacks for ZN_-prefixed settings.

    Reads .env file directly to avoid OS env pollution (expired keys in shell).
    Prioritizes .env values over OS env for unprefixed keys.
    """
    import os

    # Read .env file directly
    dotenv_path = Path(".env")
    dotenv_vars: dict[str, str] = {}
    if dotenv_path.exists():
        for line in dotenv_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val:
                dotenv_vars[key] = val

    def _get_val(*keys: str) -> str | None:
        """Get first non-empty value: .env file → OS ZN_ prefixed → OS unprefixed."""
        for key in keys:
            # Prefer .env file values over OS env (avoids expired shell keys)
            if key in dotenv_vars:
                return dotenv_vars[key]
        for key in keys:
            if key.startswith("ZN_"):
                val = os.getenv(key, "")
                if val:
                    return val
        return None

    fallbacks = {
        "llm_model": ["ZN_LLM_MODEL", "llm_model", "LLM_MODEL"],
        "llm_api_key": ["ZN_LLM_API_KEY", "llm_api_key", "DEEPSEEK_API_KEY",
                         "OPENAI_API_KEY"],
        "llm_base_url": ["ZN_LLM_BASE_URL", "llm_base_url", "LLM_BASE_URL"],
        "llm_temperature": ["ZN_LLM_TEMPERATURE", "llm_temperature"],
        "llm_max_tokens": ["ZN_LLM_MAX_TOKENS", "llm_max_tokens"],
        "llm_max_retries": ["ZN_LLM_MAX_RETRIES", "llm_max_retries"],
    }
    for attr, env_keys in fallbacks.items():
        current = getattr(settings, attr)
        default = Settings.model_fields[attr].default
        if current == default:
            val = _get_val(*env_keys)
            if val:
                field_type = type(default)
                if field_type is int:
                    val = int(val)  # type: ignore[assignment]
                elif field_type is float:
                    val = float(val)  # type: ignore[assignment]
                object.__setattr__(settings, attr, val)
    return settings


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the pipeline.

    Output format: ``HH:MM:SS [LEVEL] module: message``
    """
    import logging
    import sys

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root = logging.getLogger("src")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False


settings = Settings()
settings = _apply_env_fallbacks(settings)

# Normalize model IDs: strip provider prefixes like "deepseek/" or "openai/"
if "/" in settings.llm_model:
    settings.llm_model = settings.llm_model.split("/", 1)[1]

# Configure logging at module load time
setup_logging()
