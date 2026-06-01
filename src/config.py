from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
XML_DIR = DATA_DIR / "xml"
SECTIONS_DIR = DATA_DIR / "structured_sections"
ENTITIES_DIR = DATA_DIR / "entities"
RELATIONS_DIR = DATA_DIR / "relationships"
OUTPUT_TSV_DIR = DATA_DIR / "output" / "tsv"
OUTPUT_NEO4J_DIR = DATA_DIR / "output" / "neo4j"
SCHEMAS_DIR = PROJECT_ROOT / "schemas"

ALTERNATIVE_TSV = PROJECT_ROOT / "ALTERNATIVE.tsv"
SCHEMA_TSV = PROJECT_ROOT / "SCHEMA.tsv"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
LLM_MAX_RETRIES = 2
BATCH_SIZE = 50
MAX_CONCURRENT = 12

ENTREZ_EMAIL = os.environ.get("ENTREZ_EMAIL", "")
ENTREZ_API_KEY = os.environ.get("ENTREZ_API_KEY", "")

for d in [DATA_DIR, XML_DIR, SECTIONS_DIR, ENTITIES_DIR, RELATIONS_DIR,
          OUTPUT_TSV_DIR, OUTPUT_NEO4J_DIR, SCHEMAS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
