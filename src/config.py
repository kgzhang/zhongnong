"""Application configuration via pydantic-settings with .env support."""
from pathlib import Path
import logging
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Paths (derived from PROJECT_ROOT, not env-overridable) -----------
    project_root: Path = Field(default=PROJECT_ROOT, frozen=True)

    @property
    def data_dir(self) -> Path: return self.project_root / "data"

    @property
    def xml_dir(self) -> Path: return self.data_dir / "xml"

    @property
    def sections_dir(self) -> Path: return self.data_dir / "structured_sections"

    @property
    def entities_dir(self) -> Path: return self.data_dir / "entities"

    @property
    def relations_dir(self) -> Path: return self.data_dir / "relationships"

    @property
    def output_tsv_dir(self) -> Path: return self.data_dir / "output" / "tsv"

    @property
    def output_neo4j_dir(self) -> Path: return self.data_dir / "output" / "neo4j"

    @property
    def schemas_dir(self) -> Path: return self.project_root / "schemas"

    @property
    def alternative_tsv(self) -> Path: return self.project_root / "ALTERNATIVE.tsv"

    @property
    def schema_tsv(self) -> Path: return self.project_root / "SCHEMA.tsv"

    # -- LLM settings ----------------------------------------------------
    deepseek_api_key: str = Field(default="", description="DeepSeek API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key (fallback)")
    llm_model: str = Field(default="deepseek/deepseek-chat")
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    batch_size: int = Field(default=50, ge=1, le=500)
    max_concurrent: int = Field(default=12, ge=1, le=50)

    # -- Entrez settings -------------------------------------------------
    entrez_email: str = Field(default="", description="NCBI Entrez email (required)")
    entrez_api_key: str = Field(default="")


# Singleton instance
settings = Settings()


# ---------------------------------------------------------------------------
# Backward-compatible module-level aliases (existing code uses these)
# ---------------------------------------------------------------------------
DATA_DIR = settings.data_dir
XML_DIR = settings.xml_dir
SECTIONS_DIR = settings.sections_dir
ENTITIES_DIR = settings.entities_dir
RELATIONS_DIR = settings.relations_dir
OUTPUT_TSV_DIR = settings.output_tsv_dir
OUTPUT_NEO4J_DIR = settings.output_neo4j_dir
SCHEMAS_DIR = settings.schemas_dir

ALTERNATIVE_TSV = settings.alternative_tsv
SCHEMA_TSV = settings.schema_tsv

ANTHROPIC_API_KEY = settings.anthropic_api_key
DEEPSEEK_API_KEY = settings.deepseek_api_key
LLM_MODEL = settings.llm_model
LLM_MAX_RETRIES = settings.llm_max_retries
BATCH_SIZE = settings.batch_size
MAX_CONCURRENT = settings.max_concurrent

ENTREZ_EMAIL = settings.entrez_email
ENTREZ_API_KEY = settings.entrez_api_key


def ensure_dirs() -> None:
    """Create all data directories if they don't exist."""
    dirs = [
        DATA_DIR, XML_DIR, SECTIONS_DIR, ENTITIES_DIR, RELATIONS_DIR,
        OUTPUT_TSV_DIR, OUTPUT_NEO4J_DIR, SCHEMAS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def validate() -> None:
    """Print warnings for misconfigured settings."""
    if not settings.entrez_email:
        logger.warning("ENTREZ_EMAIL not set. PubMed API calls may be rate-limited.")
    if not settings.deepseek_api_key and not settings.anthropic_api_key:
        logger.warning("No LLM API key set (DEEPSEEK_API_KEY or ANTHROPIC_API_KEY). LLM calls will fail.")
