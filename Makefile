.PHONY: help install sync test test-cov test-watch debug batch batch-dir batch-all clean clean-all

CLI := uv run python -m src.cli
STANDALONE := python scripts/batch_extract.py

# ---------------------------------------------------------------------------
# Default
# ---------------------------------------------------------------------------
help:
	@echo "llm-extract v2 — Knowledge Graph Extraction Pipeline"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install dependencies via uv sync"
	@echo ""
	@echo "Tests:"
	@echo "  make test          Run all tests"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make test-watch    Run tests in fail-fast mode"
	@echo ""
	@echo "Batch Extraction (via CLI):"
	@echo "  make batch XML=data/xml/PMC12188611.xml"
	@echo "                     Single file → output/<filename>/"
	@echo "  make batch-dir DIR=data/xml/"
	@echo "                     All *.xml in directory → output/batch/"
	@echo "  make batch-all     All *.xml in data/xml/ → output/batch/"
	@echo ""
	@echo "Batch Extraction (standalone — no CLI install needed):"
	@echo "  make batch-py XML=data/xml/PMC12188611.xml"
	@echo "                     Single file via scripts/batch_extract.py"
	@echo "  make batch-py-dir DIR=data/xml/"
	@echo "                     Directory via scripts/batch_extract.py"
	@echo ""
	@echo "Debugging:"
	@echo "  make debug XML=data/xml/PMC12188611.xml"
	@echo "                     Extract + inspect a single file"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Remove output + cache"
	@echo "  make clean-all     Remove output + cache + .venv"
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
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ --cov=src --cov-report=term-missing -v

test-watch:
	uv run pytest tests/ -v --tb=short -x

# ---------------------------------------------------------------------------
# Batch extraction (via CLI)
# ---------------------------------------------------------------------------
batch:
	@test -n "$(XML)" || { echo "Usage: make batch XML=data/xml/PMC12188611.xml"; exit 1; }
	$(CLI) batch --input $(XML)

batch-dir:
	@test -n "$(DIR)" || { echo "Usage: make batch-dir DIR=data/xml/"; exit 1; }
	$(CLI) batch --input $(DIR)

batch-all:
	$(CLI) batch --input data/all/

# ---------------------------------------------------------------------------
# Batch extraction (standalone script — no CLI install needed)
# ---------------------------------------------------------------------------
batch-py:
	@test -n "$(XML)" || { echo "Usage: make batch-py XML=data/xml/PMC12188611.xml"; exit 1; }
	$(STANDALONE) --input $(XML)

batch-py-dir:
	@test -n "$(DIR)" || { echo "Usage: make batch-py-dir DIR=data/xml/"; exit 1; }
	$(STANDALONE) --input $(DIR)

# ---------------------------------------------------------------------------
# Debugging
# ---------------------------------------------------------------------------
debug:
	@test -n "$(XML)" || { echo "Usage: make debug XML=data/xml/PMC12188611.xml"; exit 1; }
	$(CLI) debug --input $(XML)

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean:
	@echo "Removing output and cache..."
	rm -rf output/
	rm -rf .pytest_cache
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."

clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf .venv
	@echo "Full cleanup complete."
