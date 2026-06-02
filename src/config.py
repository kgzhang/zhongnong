"""Configuration for zhongnong-kg v2 pipeline."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM
    llm_model: str = "deepseek-chat"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_max_retries: int = 3
    llm_temperature: float = 0.1
    llm_max_tokens: int = 16384

    # Pipeline
    max_char_buffer: int = 8000
    batch_length: int = 1
    max_workers: int = 4
    extraction_passes: int = 1
    context_window_chars: int | None = None

    # Paths
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    checkpoint_dir: Path = Path("data/intermediates")
    alternative_tsv: Path = Path("ALTERNATIVE.tsv")
    schema_dir: Path = Path("schemas")

    # Gate
    skip_if_no_known_alternative: bool = True

    # Debug
    debug: bool = False
    show_progress: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_prefix="ZN_", extra="ignore")


settings = Settings()
