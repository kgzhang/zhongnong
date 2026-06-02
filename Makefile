.PHONY: help install sync test test-cov test-watch debug extract export clean clean-all

VENV := .venv
PYTEST := $(VENV)/bin/pytest
CLI := $(VENV)/bin/python -m src.cli

# ---------------------------------------------------------------------------
# Default
# ---------------------------------------------------------------------------
help:
	@echo "zhongnong-kg v2 — Knowledge Graph Extraction Pipeline"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install dependencies via uv sync"
	@echo "  make sync          Alias for install"
	@echo ""
	@echo "Tests:"
	@echo "  make test          Run all tests (203 tests)"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make test-watch    Run tests in fail-fast mode"
	@echo ""
	@echo "Extraction:"
	@echo "  make debug XML=    Debug: extract from single article"
	@echo "                     e.g. make debug XML=data/xml/PMC12183824.xml"
	@echo "  make extract XML=  Extract and export graph for single article"
	@echo "  make run-all       Run pipeline on all articles in data/xml/"
	@echo ""
	@echo "Export:"
	@echo "  make export DIR=   Export Neo4j CSV from checkpoints"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Remove extracted data and cache"
	@echo "  make clean-all     Remove data + cache + venv"
	@echo ""

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
install:
	uv sync
	@echo "Dependencies installed."

sync: install

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
test:
	$(PYTEST) tests/ -v

test-cov:
	$(PYTEST) tests/ --cov=src --cov-report=term-missing -v

test-watch:
	$(PYTEST) tests/ -v --tb=short -x

# ---------------------------------------------------------------------------
# Extraction (v2 pipeline via CLI)
# ---------------------------------------------------------------------------
debug:
	@test -n "$(XML)" || { echo "Usage: make debug XML=data/xml/PMC12183824.xml"; exit 1; }
	$(CLI) debug --xml $(XML)

extract:
	@test -n "$(XML)" || { echo "Usage: make extract XML=data/xml/PMC12183824.xml"; exit 1; }
	$(VENV)/bin/python -c "\
from src.extraction import extract; \
from src.graph import build_graph, export_neo4j_csv; \
from src.schema_registry import SchemaRegistry; \
from pathlib import Path; \
result = extract('$(XML)', skip_if_no_known_alternative=False); \
print(f'Entities: {len(result.extractions)}, Skipped: {result.skipped}'); \
if result.skipped: print(f'Reason: {result.skip_reason}'); \
registry = SchemaRegistry(); \
graph = build_graph([result], registry); \
print(f'Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges'); \
out = Path('output'); out.mkdir(exist_ok=True); \
export_neo4j_csv(graph, out); \
print(f'Exported nodes.csv + edges.csv to output/')"

run-all:
	@echo "Running pipeline on all articles in data/xml/..."
	$(VENV)/bin/python -c "\
from src.extraction import extract; \
from src.graph import build_graph, export_neo4j_csv; \
from src.schema_registry import SchemaRegistry; \
from pathlib import Path; \
registry = SchemaRegistry(); \
all_results = []; \
xml_dir = Path('data/xml'); \
for f in sorted(xml_dir.glob('*.xml')): \
    print(f'=== {f.name} ==='); \
    result = extract(str(f), skip_if_no_known_alternative=False); \
    all_results.append(result); \
    print(f'  Entities: {len(result.extractions)} {"SKIPPED: "+result.skip_reason if result.skipped else ""}'); \
graph = build_graph(all_results, registry); \
print(f'\nTotal: {len(graph.nodes)} nodes, {len(graph.edges)} edges across {len(all_results)} articles'); \
out = Path('output'); out.mkdir(exist_ok=True); \
export_neo4j_csv(graph, out); \
print(f'Exported to output/nodes.csv + output/edges.csv')"

# ---------------------------------------------------------------------------
# Export (from checkpoints)
# ---------------------------------------------------------------------------
export:
	@test -n "$(DIR)" || { echo "Usage: make export DIR=data/intermediates"; exit 1; }
	$(CLI) export --checkpoints $(DIR) --output output

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean:
	@echo "Removing extracted data and cache..."
	rm -rf output/
	rm -rf data/intermediates/
	rm -rf .pytest_cache
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."

clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	rm -rf .worktrees
	@echo "Full cleanup complete."
