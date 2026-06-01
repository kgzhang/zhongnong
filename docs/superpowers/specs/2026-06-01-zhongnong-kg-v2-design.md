# zhongnong-kg v2: Production Knowledge Graph Extraction Pipeline

**Date:** 2026-06-01
**Status:** Spec — awaiting review
**Context:** Rewrite of the zhongnong-kg pipeline from scratch, informed by langextract's schema-based extraction patterns but built independently for production-grade stability.

---

## 1. Motivation

The current pipeline (`src/dspy_extract.py`, `src/run_dspy_pipeline.py`) relies on DSPy's `Predict` abstraction with string-based JSON parsing from LLM output. This is unstable — malformed JSON, missing fields, and inconsistent entity names cause cascading failures downstream. The pipeline also lacks checkpointing (crash = restart from zero), has no concurrency, and mixes concerns across modules.

**Goal:** A production-grade pipeline that:
- Uses structured output (`response_format` with JSON Schema) for reliable LLM extraction
- Processes articles concurrently with per-article checkpointing for resumability
- Co-extracts entities and relationships in single LLM calls (langextract pattern)
- Produces deterministic, deduplicated `nodes.tsv` + `edges.tsv` output
- Is fully testable with mocked LLM responses

## 2. Architecture

```
src/
  cli.py            # Click/Typer CLI entry point
  runner.py         # Pipeline orchestrator: load → process → merge → export
  chunking.py       # PMC XML → section-level text chunks
  extract.py        # Per-article extraction sequence (4 LLM calls)
  prompts.py        # Jinja2 prompt templates
  glossary.py       # TSV glossary loader + exact/fuzzy matcher
  graph.py          # Entity dedup, edge resolution, TSV export
  entity_id.py      # Deterministic global ID generation
  config.py         # Pydantic Settings (env vars, paths, model config)

  providers/
    __init__.py
    base.py         # BaseLLMProvider ABC
    openai_compat.py # httpx async client for OpenAI-compatible APIs
    factory.py      # create_provider(model_id, api_key, base_url) → Provider

  schemas/
    __init__.py
    entities.py     # Pydantic models for all entity types
    relations.py    # Relation types enum + edge model
    article.py      # ArticleExtraction — top-level per-article output

tests/
  test_chunking.py
  test_extract.py
  test_glossary.py
  test_graph.py
  test_provider.py
  test_schemas.py
  test_integration.py
  fixtures/
    article_1.xml
    article_2.xml
    llm_response_alternatives.json
    llm_response_design.json
    llm_response_indicators.json
    llm_response_results.json
```

### 2.1 Data Flow

```
ALTERNATIVE.tsv ──→ GlossaryIndex (loaded once at startup)

literature_pool.tsv (DOI, PMID, XML path)
  │
  ▼
Runner.process_all(articles, max_workers=N)
  │
  ├─ For each article (asyncio, up to N concurrent):
  │   ├─ 1. Check checkpoint: if data/intermediates/{pmid}.json exists → skip
  │   ├─ 2. Parse XML → {abstract, methods, results, discussion}
  │   ├─ 3. extract_alternatives(methods) → AlternativeEntities + Composites
  │   │      GATE: if no known-class Alternative → skip article, write skip.txt
  │   ├─ 4. extract_experiment_design(methods) → Swine + Interventions + Controls
  │   ├─ 5. extract_indicators(methods) → TissueSites + Indicators + Methods
  │   ├─ 6. extract_results(results, discussion, indicators, controls, tissues)
  │   │      → ResultEntities (with inline foreign keys)
  │   ├─ 7. Normalize + validate (Pydantic)
  │   ├─ 8. Glossary enrich (match extracted alternatives against glossary)
  │   └─ 9. Write checkpoint JSON
  │
  ├─ 10. Collect all checkpoint JSONs
  ├─ 11. Deduplicate entities across articles → global IDs
  ├─ 12. Resolve inline relationships → graph edges
  └─ 13. Export nodes.tsv + edges.tsv
```

### 2.2 Streaming & Memory Model

Each article is processed independently. At no point are all articles' extractions in memory simultaneously until the final merge step (step 10). The runner uses `asyncio.Semaphore(max_workers)` to bound concurrent LLM calls. Checkpoint files are written atomically (write to temp file, rename) to avoid corruption on crash.

## 3. Component Specifications

### 3.1 Provider Layer

**`BaseLLMProvider`** (ABC):
```python
class BaseLLMProvider(ABC):
    @abstractmethod
    async def infer(
        self,
        prompt: str,
        schema: type[BaseModel],
        system_prompt: str | None = None,
    ) -> BaseModel:
        """Send prompt to LLM with structured output schema.
        Returns a validated Pydantic model instance.
        """
```

**`OpenAICompatProvider`**:
- Async httpx client targeting `{base_url}/chat/completions`
- Sends `response_format: { type: "json_schema", json_schema: { name: "...", schema: model.model_json_schema() } }`
- Validates response JSON against the Pydantic model — returns model instance directly
- Retries: 3 attempts with exponential backoff (1s, 2s, 4s) on 429/5xx
- Logs token usage per call for cost tracking
- Raises `ProviderError` on exhaustion (caught by runner, article marked failed)

**`create_provider(model_id, api_key, base_url) → BaseLLMProvider`**:
- If model_id contains "deepseek" → OpenAICompatProvider with DeepSeek base URL
- Otherwise → generic OpenAICompatProvider
- Reads env vars for defaults if api_key/base_url not provided

### 3.2 Schema Layer

All entities defined as Pydantic `BaseModel` subclasses in `src/schemas/entities.py`. Each model's `model_json_schema()` generates the JSON Schema passed to the LLM. The models also serve as the runtime validation layer — LLM output is parsed directly into model instances.

**Entity types** (matching BACKGROUND.md):

| Model | Key Fields |
|-------|-----------|
| `AlternativeEntity` | standard_name, alternative_class (enum), abbreviation, cas_number, source_organism, subclass, match_source, original_text, evidence_text, source_location |
| `CompositeProductEntity` | product_name, manufacturer, is_commercial, components (list[ComponentRef]), evidence_text, source_location |
| `ComponentRef` | standard_name, entity_type — co-extracted within CompositeProduct |
| `SwineEntity` | breed, sex (enum), age, physiological_stage, initial_body_weight, sample_size, evidence_text, source_location |
| `InterventionEntity` | intervention_target, dose_value, dose_unit_original, dose_unit_standard, administration_route (enum), duration, basal_diet, positive_control, evidence_text, source_location |
| `ControlGroupEntity` | group_name, group_type (enum: negative_control/positive_control/basal_control/sham), description, evidence_text, source_location |
| `TissueSiteEntity` | site_name, site_category (enum: content/mucosa/serum/tissue/feces), evidence_text, source_location |
| `IndicatorEntity` | standard_name, abbreviation, unit, indicator_category (enum: macro_phenotype/microbiome/metabolome/molecular), measurement_method, measured_in (str — TissueSite.site_name), evidence_text, source_location |
| `MethodEntity` | method_name, description, evidence_text, source_location |
| `ResultEntity` | indicator_abbreviation (str → IndicatorEntity.abbreviation), tissue_site (str → TissueSite.site_name), direction (enum: increased/decreased/no_significant_change), relation_type (enum), significance_level (enum), p_value, p_value_original_text, corrected_significance, effect_size, time_point, subgroup, compared_to_group (str → ControlGroupEntity.group_name), evidence_text, source_location |

**Relation types** (enum in `src/schemas/relations.py`):
- `has_component` — CompositeProduct → Alternative/CompositeProduct
- `measured_in` — Indicator → TissueSite
- `uses_method` — Indicator → Method
- `corresponds_to` — Result → Indicator
- `occurs_in` — Result → TissueSite
- `compared_to` — Result → ControlGroup
- `increases / decreases` — Intervention → Result (numerical)
- `upregulates / downregulates` — Intervention → Result (gene/protein)
- `enriches / depletes` — Intervention → Result (microbial)
- `affects` — Intervention → Result (fallback)

**Top-level extraction outputs** (`src/schemas/article.py`):
```python
class AlternativeExtractionOutput(BaseModel):
    doi: str
    extraction_target: str = "alternatives"
    alternatives: list[AlternativeEntity] = []
    composite_products: list[CompositeProductEntity] = []
    warnings: list[str] = []

class ExperimentDesignOutput(BaseModel):
    doi: str
    swine: list[SwineEntity] = []
    interventions: list[InterventionEntity] = []
    control_groups: list[ControlGroupEntity] = []

class IndicatorExtractionOutput(BaseModel):
    doi: str
    tissue_sites: list[TissueSiteEntity] = []
    indicators: list[IndicatorEntity] = []
    methods: list[MethodEntity] = []

class ResultExtractionOutput(BaseModel):
    doi: str
    results: list[ResultEntity] = []

class ArticleExtraction(BaseModel):
    """Final per-article checkpoint format."""
    doi: str
    pmid: str
    alternatives: list[AlternativeEntity]
    composite_products: list[CompositeProductEntity]
    swine: list[SwineEntity]
    interventions: list[InterventionEntity]
    control_groups: list[ControlGroupEntity]
    tissue_sites: list[TissueSiteEntity]
    indicators: list[IndicatorEntity]
    methods: list[MethodEntity]
    results: list[ResultEntity]
    abstract_conclusion: str
    warnings: list[str] = []
    skipped: bool = False
    skip_reason: str = ""
```

### 3.3 Extraction Engine

**`extract.py`** contains four async functions, each making one LLM call:

```python
async def extract_alternatives(
    provider: BaseLLMProvider,
    methods_text: str,
    glossary_text: str,
    doi: str,
) -> AlternativeExtractionOutput

async def extract_experiment_design(
    provider: BaseLLMProvider,
    methods_text: str,
    doi: str,
) -> ExperimentDesignOutput

async def extract_indicators(
    provider: BaseLLMProvider,
    methods_text: str,
    doi: str,
) -> IndicatorExtractionOutput

async def extract_results(
    provider: BaseLLMProvider,
    results_text: str,
    discussion_text: str,
    indicators: list[IndicatorEntity],
    control_groups: list[ControlGroupEntity],
    tissue_sites: list[TissueSiteEntity],
    doi: str,
) -> ResultExtractionOutput
```

**Gate logic** (applied after `extract_alternatives`):
```python
KNOWN_CLASSES = {
    "Plant_Extract", "Trace_Element", "Organic_Acid", "Probiotic",
    "Polysaccharides_and_Oligosaccharides", "Enzyme", "Bioactive_Peptides",
}

def has_known_alternative(output: AlternativeExtractionOutput) -> bool:
    return any(
        a.alternative_class in KNOWN_CLASSES
        for a in output.alternatives
    )
```

If `has_known_alternative` is False → article skipped, no further calls made.

**Prompts** (`src/prompts.py`):
- Each extraction function has a corresponding Jinja2 template
- Templates include: system prompt, extraction instructions, few-shot examples (embedded in template), glossary (for alternatives), context (for results)
- Templates are versioned — the template string itself is the source of truth, no runtime assembly

### 3.4 Chunking

`chunking.py` handles PMC XML parsing:
```python
def parse_article_sections(xml_path: str | Path) -> dict[str, str]:
    """Returns {abstract, methods, results, discussion, conclusion}."""
```

- Uses `lxml` to parse PMC XML
- Extracts Abstract, Methods/Materials, Results, Discussion sections by tag
- Handles nested sections, supplementary material exclusion
- Strips references/citations
- Implements `abstract_conclusion()` helper: finds last 1-3 sentences with conclusion markers ("In conclusion", "These results suggest", "Overall", "Therefore", "In summary", "Collectively"), returns verbatim copy

### 3.5 Glossary Engine

`glossary.py` loads `ALTERNATIVE.tsv`:
```python
class GlossaryIndex:
    def __init__(self, tsv_path: Path):
        # Builds exact-match dict and subclass index
    
    def lookup(self, name: str) -> GlossaryMatch | None:
        """Exact match first, then fuzzy."""
    
    def fuzzy_match(self, name: str) -> GlossaryMatch | None:
        """Normalize → substring containment → Levenshtein ≤ 3."""
```

- Pre-loaded once at startup
- TSV columns: standard_name, class, subclass, synonyms (pipe-separated)
- Exact match: case-insensitive, whitespace-normalized
- Fuzzy match: lowercase, strip punctuation, check substring containment both ways, Levenshtein distance ≤ 3 as fallback
- Returns `GlossaryMatch(standard_name, class, subclass, match_source)` where match_source is one of: `词表精确匹配`, `词表同义映射`, `词表模糊匹配`, `Other_未匹配`

### 3.6 Pipeline Runner

```python
class Pipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider = create_provider(...)
        self.glossary = GlossaryIndex(settings.alternative_tsv)
    
    async def process_all(self, articles: list[ArticleRef]) -> PipelineResult:
        """Process all articles with concurrency control."""
    
    async def _process_one(self, article: ArticleRef) -> ArticleExtraction | None:
        """Full extraction sequence for one article."""
    
    def merge_and_export(self, extractions: list[ArticleExtraction]) -> None:
        """Dedup, resolve edges, write nodes.tsv + edges.tsv."""
```

**Concurrency model:**
- `asyncio.Semaphore(settings.max_workers)` bounds parallel article processing
- Within an article, the 4 LLM calls are sequential (call 4 depends on call 3's output)
- Failed articles logged to `data/failures.jsonl`, processing continues

**Checkpoint format:** `data/intermediates/{pmid}.json` — serialized `ArticleExtraction` model. Written atomically via temp-file + rename.

**Resume logic:** At startup, `runner.py` scans `data/intermediates/` for existing checkpoints. Articles with valid checkpoint files are skipped. A checkpoint is "valid" if it deserializes successfully to `ArticleExtraction` and has `skipped == False`.

### 3.7 Graph Builder

```python
def build_graph(extractions: list[ArticleExtraction]) -> Graph:
    """Dedup entities, resolve edges, return nodes + edges."""

class Graph:
    nodes: list[GraphNode]
    edges: list[GraphEdge]

def export_graph(graph: Graph, output_dir: Path) -> None:
    """Write nodes.tsv and edges.tsv."""
```

**Entity deduplication:**
- Global ID = `sha256(f"{entity_type}:{normalized_name}")[:16]` — deterministic, no registry needed
- `normalized_name = standard_name.lower().strip()`
- Same Alternative appearing in multiple articles → same global ID
- `source_pmids` accumulates across articles

**Edge resolution:**
- Co-extracted relationships (has_component, ComponentRef) → direct edges
- Foreign-key relationships resolved within each article's entity set:
  - `IndicatorEntity.measured_in` → matches `TissueSiteEntity.site_name` → edge `(Indicator, measured_in, TissueSite)`
  - `ResultEntity.indicator_abbreviation` → matches `IndicatorEntity.abbreviation` → edge `(Result, corresponds_to, Indicator)`
  - `ResultEntity.tissue_site` → matches `TissueSiteEntity.site_name` → edge `(Result, occurs_in, TissueSite)`
  - `ResultEntity.compared_to_group` → matches `ControlGroupEntity.group_name` → edge `(Result, compared_to, ControlGroup)`
  - `ResultEntity.direction + relation_type` → edge `(Intervention, relation_type, Result)`
- Matching is case-insensitive, whitespace-normalized within the same article
- Unresolved references logged as warnings, edge omitted

**Output format:**

`nodes.tsv`:
```
id\ttype\tname\tattributes_json\tsource_pmids
```

`edges.tsv`:
```
source_id\ttarget_id\trelation_type\tevidence_text\tsource_pmids
```

### 3.8 CLI

```bash
# Full pipeline
zhongnong-kg run --input data/literature_pool.tsv --output output/

# Resume from checkpoints
zhongnong-kg run --resume

# Single article (debug)
zhongnong-kg debug --xml data/xml/PMC12178903.xml

# Export only (skip extraction, use existing checkpoints)
zhongnong-kg export --checkpoints data/intermediates/ --output output/
```

## 4. Error Handling

| Failure Mode | Behavior |
|-------------|----------|
| LLM rate limit (429) | Retry up to 3x with exponential backoff |
| LLM server error (5xx) | Retry up to 3x with exponential backoff |
| LLM returns invalid JSON | Log raw response, retry once with error context in prompt |
| LLM output fails Pydantic validation | Log validation errors, retry once with error description |
| LLM retries exhausted | Mark article failed, write to `data/failures.jsonl`, continue |
| XML parse error | Log, skip article, continue |
| Checkpoint write failure | Log warning, continue (re-extract on next run) |
| No known alternative (gate) | Skip article cleanly, write `skipped: true` checkpoint |
| Empty section (no methods text) | Skip article, log reason |
| Entity reference unresolved | Log warning, omit edge, include entity anyway |

## 5. Configuration

Single `src/config.py` with `pydantic-settings`:

```python
class Settings(BaseSettings):
    # LLM
    llm_model: str = "deepseek-chat"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_max_retries: int = 3
    llm_temperature: float = 0.1
    llm_max_tokens: int = 16384

    # Pipeline
    max_workers: int = 4
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    checkpoint_dir: Path = Path("data/intermediates")
    alternative_tsv: Path = Path("ALTERNATIVE.tsv")

    # Gate
    skip_if_no_known_alternative: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_prefix="ZN_")
```

## 6. Testing Strategy

| Layer | Test Type | What it verifies |
|-------|----------|-----------------|
| Schemas | Unit | Pydantic validation (valid/invalid), JSON Schema generation, enum constraints |
| Provider | Unit (mocked httpx) | Retry logic, rate-limit backoff, schema validation on response, error handling |
| Glossary | Unit | Exact match, fuzzy match algorithms, edge cases (empty, unicode, short strings) |
| Chunking | Unit (real XML fixtures) | Section extraction, conclusion detection, reference stripping |
| Extraction | Integration (recorded responses) | Full 4-call sequence with mocked LLM, gate logic, empty-article handling |
| Graph | Unit | Dedup determinism, edge resolution, cross-article merge |
| Pipeline | Integration | End-to-end with recorded responses, output shape validation |
| CLI | Smoke | --help, basic invocation |

**Golden files:** `tests/fixtures/llm_response_*.json` — recorded real LLM responses used for deterministic regression testing. Never hit real APIs in CI.

## 7. Dependencies

```toml
[project]
name = "zhongnong-kg"
version = "2.0.0"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.28",           # Async HTTP client
    "pydantic>=2.0",         # Schema models + validation
    "pydantic-settings>=2.0", # Env-based config
    "lxml>=5.3",             # XML parsing
    "jinja2>=3.0",           # Prompt templates
    "click>=8.0",            # CLI
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.25",
    "pytest-httpx>=0.30",    # HTTP mocking
]
```

**Removed from v1:** `dspy`, `litellm`, `pandas`, `biopython`, `jsonschema` (replaced by Pydantic).

## 8. Migration from v1

1. New pipeline built in `src/` — same package, restructured modules
2. Old modules (`dspy_extract.py`, `run_dspy_pipeline.py`, `stage*.py`) remain until new pipeline validates
3. `ALTERNATIVE.tsv` and `data/` directory reused as-is
4. Once new pipeline produces correct output on 3+ real XML fixtures, old modules deleted
5. No dual-running, no compatibility shims

## 9. Non-Goals

- No real-time/streaming extraction API — batch processing only
- No web UI or dashboard
- No multi-format input beyond PMC XML
- No incremental graph updates — each run produces a fresh graph
- No citation network analysis or bibliometric features
