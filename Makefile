.PHONY: help install test test-gate test-all stage1 stage2 stage3 stage4 pipeline dspy clean-data clean-all lint

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

# ---------------------------------------------------------------------------
# Default
# ---------------------------------------------------------------------------
help:
	@echo "Knowledge Graph Extraction Pipeline"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install dependencies"
	@echo "  make sync          Sync dependencies from pyproject.toml"
	@echo ""
	@echo "Tests:"
	@echo "  make test          Run all tests"
	@echo "  make test-gate     Run gate regression tests"
	@echo "  make test-dspy     Run DSPy extraction tests"
	@echo "  make test-watch    Run tests in watch mode"
	@echo ""
	@echo "Pipeline (legacy litellm):"
	@echo "  make pipeline      Run full pipeline (stage0 → stage4)"
	@echo "  make stage1        Parse XML files only"
	@echo "  make stage2        Entity extraction (LLM)"
	@echo "  make stage3        Result extraction (LLM)"
	@echo "  make stage4        Validation + TSV + Neo4j export"
	@echo ""
	@echo "Pipeline (DSPy — recommended):"
	@echo "  make dspy          Run DSPy pipeline (all stages)"
	@echo "  make dspy-resume   Resume DSPy pipeline (skip XML parse)"
	@echo "  make dspy-limit-%  Run DSPy pipeline with limit (e.g. make dspy-limit-5)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean-data    Remove extracted data (entities, sections, output)"
	@echo "  make clean-all     Remove data + cache + venv"
	@echo ""

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
install: $(VENV)/bin/activate
	$(PIP) install -e ".[dev]"
	$(PIP) install litellm dspy-ai pydantic-settings
	@echo "Dependencies installed."

sync:
	uv pip install -e ".[dev]"
	uv pip install litellm dspy-ai pydantic-settings

$(VENV)/bin/activate:
	uv venv
	@echo "Virtual environment created."

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
test:
	$(PYTEST) tests/ -v

test-gate:
	$(PYTEST) tests/test_dspy_extract.py::TestAlternativeGate -v

test-dspy:
	$(PYTEST) tests/test_dspy_extract.py -v

test-watch:
	$(PYTEST) tests/ -v --tb=short -x

# ---------------------------------------------------------------------------
# Legacy litellm pipeline
# ---------------------------------------------------------------------------
pipeline:
	$(PYTHON) -m src.run_pipeline --from stage0

stage1:
	$(PYTHON) -c "\
from src.stage1_xml_parser import parse_xml_to_sections; \
import json; \
from pathlib import Path; \
out_dir = Path('data/structured_sections'); \
out_dir.mkdir(parents=True, exist_ok=True); \
for f in sorted(Path('data/xml').glob('*.xml')): \
    print(f'Parsing {f.name}...'); \
    art = parse_xml_to_sections(str(f)); \
    doi = art['doi'].replace('/','_').replace(':','_'); \
    (out_dir / f'{doi}.json').write_text(json.dumps(art, ensure_ascii=False, indent=2)); \
    mm = art['sections'].get('materials_and_methods',{}); \
    print(f'  DOI={art[\"doi\"]} M&M={len(mm.get(\"full_text\",\"\"))}ch')"

stage2:
	$(PYTHON) -m src.run_pipeline --from stage2

stage3:
	$(PYTHON) -m src.run_pipeline --from stage3

stage4:
	$(PYTHON) -m src.run_pipeline --from stage4

# ---------------------------------------------------------------------------
# DSPy pipeline (recommended)
# ---------------------------------------------------------------------------
dspy:
	@echo "Running DSPy pipeline (all stages)..."
	$(PYTHON) -m src.run_dspy_pipeline

dspy-resume:
	@echo "Resuming DSPy pipeline (skip XML parse)..."
	$(PYTHON) -c "from src.run_dspy_pipeline import run_full_pipeline; run_full_pipeline(skip_stage1=True)"

dspy-limit-%:
	@echo "Running DSPy pipeline with limit $*..."
	$(PYTHON) -c "\
from src.run_dspy_pipeline import run_full_pipeline; \
from src.dspy_extract import _get_glossary; \
import logging; \
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'); \
import json; \
from pathlib import Path; \
from src.config import settings; \
articles = [json.loads(f.read_text()) for f in sorted(settings.sections_dir.glob('*.json')) if not f.name.startswith('_')]; \
articles = articles[:$*]; \
print(f'Processing {len(articles)} articles (limit=$*)'); \
for art in articles: \
    from src.run_dspy_pipeline import stage1_parse_xml, stage2_extract_entities, stage3_extract_results, stage4_export, _clean_entity_dir; \
    all_entities = []; all_results = []; \
    entities = stage2_extract_entities(art); \
    if entities is None: continue; \
    all_entities.append(entities); \
    results = stage3_extract_results(art, entities); \
    for r in results: r['doi'] = art['doi']; \
    all_results.extend(results); \
    summary = stage4_export(all_entities, all_results); \
    print('Summary:', summary)"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean-data:
	@echo "Removing extracted data..."
	rm -rf data/entities/*/
	rm -rf data/structured_sections/*.json
	rm -rf data/output/tsv/*.tsv
	rm -rf data/output/neo4j/*.cypher
	rm -f data/literature_pool.tsv
	rm -f data/entities/_stage3_results.json
	rm -f data/relationships/_stage3_*.json
	rm -rf data/sections/
	@echo "Data cleaned."

clean-all: clean-data
	@echo "Removing cache and virtual environment..."
	rm -rf $(VENV)
	rm -rf .pytest_cache
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Full cleanup complete."

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
lint:
	$(PYTHON) -m ruff check src/ tests/ 2>/dev/null || $(PYTHON) -m flake8 src/ tests/ --max-line-length=120 2>/dev/null || echo "No linter configured. Install ruff or flake8."
