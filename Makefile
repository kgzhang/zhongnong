.PHONY: help install sync test test-cov test-watch debug extract run-all clean clean-all

CLI := uv run python -m src.cli

# ---------------------------------------------------------------------------
# Default
# ---------------------------------------------------------------------------
help:
	@echo "zhongnong-kg v2 — Knowledge Graph Extraction Pipeline"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install dependencies via uv sync"
	@echo ""
	@echo "Tests (203 total):"
	@echo "  make test          Run all tests"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make test-watch    Run tests in fail-fast mode"
	@echo ""
	@echo "Extraction:"
	@echo "  make debug XML=data/xml/PMC12183824.xml"
	@echo "                     Debug: extract from single article"
	@echo "  make extract XML=data/xml/PMC12183824.xml"
	@echo "                     Extract and export Neo4j CSV for one article"
	@echo "  make run-all       Run pipeline on all articles in data/xml/"
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
# Extraction
# ---------------------------------------------------------------------------
debug:
	@test -n "$(XML)" || { echo "Usage: make debug XML=data/xml/PMC12183824.xml"; exit 1; }
	$(CLI) debug --xml $(XML)

extract:
	@test -n "$(XML)" || { echo "Usage: make extract XML=data/xml/PMC12183824.xml"; exit 1; }
	$(CLI) extract --xml $(XML) --no-skip-gate

run-all:
	$(CLI) run --dir data/xml --no-skip-gate

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
