# zhongnong-kg v2: Knowledge Graph Extraction Pipeline

**Date:** 2026-06-02
**Status:** Spec — awaiting review

**Context:** Full rewrite of zhongnong-kg following langextract's architecture faithfully — same layer decomposition, same pipeline pattern, same abstractions — adapted to the swine-nutrition knowledge graph domain.

---

## 1. Architecture Overview

The pipeline faithfully reproduces langextract's layered architecture. Each layer is an isolated, testable abstraction with a single purpose:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Extraction API                                │
│  src/extraction.py  —  extract() main entry point               │
│  Configures all layers, runs the full pipeline                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                     ▼
┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   Factory     │  │   Annotation     │  │   Graph Builder   │
│ src/factory  │  │ src/annotation  │  │ src/graph.py      │
│              │  │                  │  │                   │
│ ModelConfig  │  │ Annotator:       │  │ merge entities    │
│ create_model │  │ chunk→prompt→    │  │ dedup globally    │
│ provider     │  │ infer→resolve    │  │ resolve edges     │
│ routing      │  │ →align→emit      │  │ export TSV        │
└──────┬───────┘  └────────┬─────────┘  └──────────────────┘
       │                   │
       │    ┌──────────────┼──────────────┐
       │    │              │              │
       ▼    ▼              ▼              ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Providers   │  │   Prompting  │  │   Resolver    │
│ src/providers │  │src/prompting│  │src/resolver  │
│              │  │              │  │              │
│ BaseLanguage  │  │ Prompt       │  │ Abstract      │
│ Model ABC     │  │ Template     │  │ Resolver ABC  │
│              │  │ Structured   │  │              │
│ OpenAICompat  │  │ QAPrompt     │  │ Resolver:     │
│ Provider      │  │ Generator    │  │ parse + align │
│              │  │              │  │              │
│ apply_schema  │  │ PromptBuilder│  │ WordAligner:  │
│ infer(batch)  │  │ ContextAware │  │ difflib exact │
│ fence_output  │  │ PromptBuilder│  │ + LCS fuzzy   │
└──────┬───────┘  └──────────────┘  └──────────────┘
       │
       ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│    Schema     │  │   Chunking   │  │  Tokenizer    │
│ src/schema.py │  │src/chunking │  │src/tokenizer │
│              │  │              │  │              │
│ BaseSchema   │  │ ChunkIterator│  │ Tokenizer ABC │
│ ABC          │  │ (token-level │  │              │
│              │  │  sentence-   │  │ RegexTokenizer│
│ FormatMode   │  │  boundary    │  │ TokenizedText │
│ Schema       │  │  aware)      │  │ Token, Token  │
│              │  │              │  │ Interval,     │
│ from_examples│  │ TextChunk    │  │ CharInterval  │
│ to_provider  │  │ make_batches │  │              │
│ _config      │  │              │  │ tokens_text() │
└──────────────┘  └──────────────┘  └──────────────┘
       │
       ▼
┌──────────────┐  ┌──────────────┐
│ FormatHandler │  │  Data Model  │
│src/format    │  │ src/data.py  │
│ _handler.py  │  │              │
│              │  │ Document     │
│ JSON/YAML    │  │ ExampleData  │
│ fence detect │  │ Extraction   │
│ wrapper mgmt │  │ AnnotatedDoc │
│ parse_output │  │ CharInterval │
│ format_ex    │  │ FormatType   │
└──────────────┘  └──────────────┘
```

### Layer Summary

| Layer | Module | Role (exactly as langextract) |
|-------|--------|------|
| **Data Model** | `src/data.py` | `Document`, `ExampleData`, `Extraction`, `AnnotatedDocument`, `CharInterval`, `FormatType` — the core data types flowing through the pipeline |
| **Tokenizer** | `src/tokenizer.py` | `Tokenizer` ABC, `RegexTokenizer`, `TokenizedText`, `Token`, `TokenInterval`, `CharInterval` — regex/unicode tokenization for alignment |
| **Chunking** | `src/chunking.py` | `ChunkIterator` (token-level, sentence-boundary-aware), `SentenceIterator`, `TextChunk`, `make_batches_of_textchunk` |
| **Format Handler** | `src/format_handler.py` | `FormatHandler` — centralized JSON/YAML format, fence extraction, wrapper key management, `format_extraction_example()`, `parse_output()` |
| **Schema** | `src/schema.py` | `BaseSchema` ABC, `FormatModeSchema` — generates provider-specific structured output config from extraction targets |
| **Providers** | `src/providers/` | `BaseLanguageModel` ABC, `OpenAICompatProvider` — `infer(batch_prompts)`, `apply_schema()`, `requires_fence_output`, `set_fence_output()` |
| **Prompting** | `src/prompting.py` | `PromptTemplateStructured`, `QAPromptGenerator`, `PromptBuilder`, `ContextAwarePromptBuilder` — Q/A-format few-shot prompting with cross-chunk context |
| **Resolver** | `src/resolver.py` | `AbstractResolver` ABC, `Resolver`, `WordAligner` — parse LLM output into `Extraction` objects, align to source text (difflib exact + LCS fuzzy) |
| **Annotation** | `src/annotation.py` | `Annotator` — orchestrates chunk→prompt→infer→resolve→align→emit, streaming with batch parallelism, multi-pass extraction |
| **Factory** | `src/factory.py` | `ModelConfig`, `create_model()` — provider resolution, environment-based defaults, schema constraint injection |
| **Extraction API** | `src/extraction.py` | `extract()` — main entry: configure model, schema, format handler, resolver; run annotation; return annotated documents |
| **Glossary** | `src/glossary.py` | `GlossaryIndex` — TSV loader, exact+fuzzy matching (domain-specific, not in langextract) |
| **Graph** | `src/graph.py` | Entity dedup, edge resolution, `nodes.tsv` + `edges.tsv` export (domain-specific output format) |

---

## 2. Data Model (`src/data.py`)

Faithful reproduction of langextract's `core/data.py` types, adapted for the zhongnong domain.

```python
# FormatType — enum for output format
class FormatType(enum.Enum):
    JSON = "json"
    YAML = "yaml"

# CharInterval — character position in source text
@dataclass
class CharInterval:
    start_pos: int | None = None
    end_pos: int | None = None

# AlignmentStatus — how well extraction matched source
class AlignmentStatus(enum.Enum):
    MATCH_EXACT = "match_exact"
    MATCH_GREATER = "match_greater"
    MATCH_LESSER = "match_lesser"
    MATCH_FUZZY = "match_fuzzy"

# Extraction — extracted entity with position + attributes
@dataclass
class Extraction:
    extraction_class: str          # entity type (e.g. "Alternative", "Result")
    extraction_text: str           # primary display text
    char_interval: CharInterval | None = None
    alignment_status: AlignmentStatus | None = None
    extraction_index: int | None = None
    group_index: int | None = None
    description: str | None = None
    attributes: dict[str, Any] | None = None  # all other fields
    _token_interval: TokenInterval | None = None

# Document — input to the pipeline
@dataclass
class Document:
    text: str
    document_id: str | None = None
    additional_context: str | None = None

# ExampleData — few-shot example for prompting
@dataclass
class ExampleData:
    text: str
    extractions: list[Extraction] = field(default_factory=list)

# AnnotatedDocument — pipeline output
@dataclass
class AnnotatedDocument:
    extractions: list[Extraction] | None = None
    text: str | None = None
    document_id: str | None = None
```

**Key design choice:** `Extraction.attributes` is a `dict[str, Any]` carrying ALL entity-specific fields beyond `extraction_class` + `extraction_text`. This matches langextract's model where a single `Extraction` type carries heterogeneous entity data via attributes. The alternative (per-entity-type Pydantic models) would fragment the resolver, alignment, and annotation layers that all operate on the uniform `Extraction` type.

### Domain Entity Types (within Extraction)

Each entity type is encoded as an `Extraction` with `extraction_class` set to the type name and `attributes` carrying the type-specific fields:

| extraction_class | extraction_text | Key attributes |
|---|---|---|
| `Alternative` | standard_name | alternative_class, abbreviation, cas_number, source_organism, subclass, match_source, original_text, evidence_text, source_location |
| `Composite_Product` | product_name | manufacturer, is_commercial, components (JSON list), evidence_text, source_location |
| `Swine` | breed | sex, age, physiological_stage, initial_body_weight, sample_size, evidence_text, source_location |
| `Intervention` | intervention_target | dose_value, dose_unit_original, dose_unit_standard, administration_route, duration, basal_diet, positive_control, evidence_text, source_location |
| `Control_Group` | group_name | group_type, description, evidence_text, source_location |
| `Tissue_Site` | site_name | site_category, evidence_text, source_location |
| `Indicator` | abbreviation | standard_name, unit, indicator_category, measurement_method, measured_in, evidence_text, source_location |
| `Method` | method_name | description, evidence_text, source_location |
| `Result` | indicator_abbreviation | tissue_site, direction, relation_type, significance_level, p_value, p_value_original_text, corrected_significance, effect_size, time_point, subgroup, compared_to_group, evidence_text, source_location |
| `Abstract_Conclusion` | conclusion_text | has_marker_phrase |

### Relation Types

Relations are defined as enums, resolved during graph building from co-extracted data:

```python
class RelationType(str, Enum):
    HAS_COMPONENT = "has_component"
    MEASURED_IN = "measured_in"
    USES_METHOD = "uses_method"
    CORRESPONDS_TO = "corresponds_to"
    OCCURS_IN = "occurs_in"
    COMPARED_TO = "compared_to"
    INCREASES = "increases"
    DECREASES = "decreases"
    UPREGULATES = "upregulates"
    DOWNREGULATES = "downregulates"
    ENRICHES = "enriches"
    DEPLETES = "depletes"
    AFFECTS = "affects"
```

---

## 3. Tokenizer (`src/tokenizer.py`)

Faithful reproduction of langextract's `core/tokenizer.py`. Tokenizers are used for chunking (sentence boundary detection) and alignment (matching extraction text to source text), NOT for LLM token counting.

```python
# Core types (identical to langextract)
@dataclass(slots=True)
class CharInterval:
    start_pos: int
    end_pos: int

@dataclass(slots=True)
class TokenInterval:
    start_index: int = 0  # inclusive
    end_index: int = 0    # exclusive

class TokenType(enum.IntEnum):
    WORD = 0
    NUMBER = 1
    PUNCTUATION = 2

@dataclass(slots=True)
class Token:
    index: int
    token_type: TokenType
    char_interval: CharInterval
    first_token_after_newline: bool = False

@dataclass
class TokenizedText:
    text: str
    tokens: list[Token] = field(default_factory=list)

# Abstract tokenizer
class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> TokenizedText: ...

# Concrete implementations
class RegexTokenizer(Tokenizer): ...
class UnicodeTokenizer(Tokenizer): ...

# Utilities
def tokenize(text: str, tokenizer: Tokenizer = DEFAULT) -> TokenizedText: ...
def tokens_text(tokenized: TokenizedText, interval: TokenInterval) -> str: ...
def find_sentence_range(text, tokens, start_idx) -> TokenInterval: ...
```

**`RegexTokenizer`** (default): regex-based, splits on `[^\W\d_]+` (words), `\d+` (numbers), `([^\w\s]|_)\1*` (punctuation). Tracks `first_token_after_newline`. Fast for English text.

**`UnicodeTokenizer`**: grapheme-cluster-based via `\X` pattern. Script-aware merging (CJK fragments, Latin merges). Correct for non-English text. Slower.

**`tokenize()`**: convenience function using the default `RegexTokenizer`.

**`tokens_text()`**: reconstructs original text substring from a `TokenInterval` by mapping to `CharInterval` boundaries.

**`find_sentence_range()`**: detects sentence boundaries via end-of-sentence punctuation (`[.?!。！？।]`), excluding known abbreviations (`Dr.`, `Mr.`, etc.), and newline-then-capital-letter breaks. Used by `SentenceIterator` in chunking.

---

## 4. Chunking (`src/chunking.py`)

Faithful reproduction of langextract's `chunking.py`.

### TextChunk

```python
@dataclass
class TextChunk:
    token_interval: TokenInterval      # position in source document
    document: Document | None = None   # source document reference
    
    # Lazy-computed
    chunk_text: str                    # text within token_interval
    sanitized_chunk_text: str          # whitespace-normalized
    char_interval: CharInterval        # character position
    document_id: str | None            # from document
    additional_context: str | None     # from document
```

### SentenceIterator

Iterates through sentences of a `TokenizedText`. Uses `find_sentence_range()` to detect boundaries. Starts from a given token position — supports resuming mid-sentence after a chunk split.

### ChunkIterator

Token-level sliding window with sentence-boundary awareness. Three behaviors:

1. **Normal**: Fits whole sentences into chunks up to `max_char_buffer`. If multiple sentences fit within buffer, they're combined.
2. **Long sentence**: If a single sentence exceeds buffer, splits at newline boundaries within the sentence. Falls back to token-level split if no newlines.
3. **Huge token**: If a single token exceeds buffer, it becomes its own chunk.

```python
class ChunkIterator:
    def __init__(self, text: str | TokenizedText, max_char_buffer: int,
                 tokenizer_impl: Tokenizer, document: Document | None = None): ...
    def __iter__(self) -> Iterator[TextChunk]: ...

def make_batches_of_textchunk(
    chunk_iter: Iterator[TextChunk], batch_length: int
) -> Iterable[Sequence[TextChunk]]: ...
```

### Domain Adaptation: Section-Based Input

For zhongnong, PMC XML articles are parsed into sections BEFORE chunking. Each section becomes a `Document`:

```python
def parse_article_sections(xml_path: str | Path) -> dict[str, Document]:
    """Parse PMC XML → {abstract, methods, results, discussion} as Documents.
    
    Each Document carries the section text + document_id = PMID + section name.
    """
```

This is a thin pre-processing step. The actual chunking pipeline (ChunkIterator → TextChunk → batches) operates on each section's `Document` exactly as langextract does.

---

## 5. Format Handler (`src/format_handler.py`)

Faithful reproduction of langextract's `core/format_handler.py`.

Centralizes ALL format-specific logic: JSON vs YAML output, code fence detection/extraction, wrapper key management, attribute suffix convention.

```python
ExtractionValueType = str | int | float | dict | list | None

class FormatHandler:
    format_type: FormatType          # JSON or YAML
    use_wrapper: bool                # True: {"extractions": [...]}, False: [...]
    wrapper_key: str | None          # key name for wrapper (default "extractions")
    use_fences: bool                 # whether to expect ```json/```yaml
    attribute_suffix: str            # "_attributes"
    strict_fences: bool              # strict fence validation
    allow_top_level_list: bool       # accept [...] without wrapper
    
    def format_extraction_example(self, extractions: list[Extraction]) -> str:
        """Format extractions for a prompt few-shot example (JSON or YAML)."""
    
    def parse_output(self, text: str, *, strict: bool = False
                     ) -> Sequence[Mapping[str, ExtractionValueType]]:
        """Parse LLM output → list of extraction dicts.
        
        Extracts content from ```json/```yaml fences if use_fences=True.
        Handles wrapper unwrapping. Handles <think> tag stripping for
        reasoning models (DeepSeek-R1, QwQ). Validates structure.
        """
    
    # Internal
    def _extract_content(self, text: str) -> str: ...
    def _parse_with_fallback(self, content: str, strict: bool): ...
    def _add_fences(self, content: str) -> str: ...
```

**Fence extraction regex:** ` ```(?P<lang>[A-Za-z0-9_+-]+)?\s*\n(?P<body>[\s\S]*?)``` `

**Think tag stripping:** `<think>...</think>` removed before parsing (for DeepSeek-R1 reasoning output).

**Wrapper handling:** If `use_wrapper=True` and `wrapper_key="extractions"`, expects `{"extractions": [...]}`. Falls back to top-level list if `allow_top_level_list=True`.

---

## 6. Schema Layer (`src/schema.py`)

Faithful reproduction of langextract's `core/schema.py`.

```python
class BaseSchema(ABC):
    """Abstract base for generating structured output config from extraction targets."""
    
    @classmethod
    @abstractmethod
    def from_examples(
        cls, examples_data: Sequence[ExampleData],
        attribute_suffix: str = "_attributes"
    ) -> BaseSchema: ...
    
    @abstractmethod
    def to_provider_config(self) -> dict[str, Any]: ...
    
    @property
    @abstractmethod
    def requires_raw_output(self) -> bool: ...
    
    def validate_format(self, format_handler: FormatHandler) -> None: ...
    def sync_with_provider_kwargs(self, kwargs: dict[str, Any]) -> None: ...

class FormatModeSchema(BaseSchema):
    """Schema for providers that support format modes (JSON/YAML) without
    field-level constraints. Wraps format_type for provider config."""
    format_type: FormatType = FormatType.JSON
    
    def from_examples(...) -> FormatModeSchema: ...
    def to_provider_config(self) -> dict: ...
    @property
    def requires_raw_output(self) -> bool: ...
```

### Domain Adaptation: OpenAI JSON Schema

For the OpenAI-compatible provider, we extend `BaseSchema` with a provider-specific subclass:

```python
# src/providers/schemas/openai.py
class OpenAISchema(BaseSchema):
    """Generates OpenAI response_format json_schema from extraction targets.
    
    Maps each extraction_class in examples to a JSON Schema object variant
    in an anyOf union within {"extractions": [anyOf variants]}. Uses
    OpenAI strict mode by default.
    """
    schema_dict: dict[str, Any]
    schema_name: str = "zhongnong_extraction"
    strict: bool = True
    
    @classmethod
    def from_examples(cls, examples_data, attribute_suffix="_attributes",
                      strict=True) -> OpenAISchema: ...
    
    def to_provider_config(self) -> dict: ...
    
    @property
    def response_format(self) -> dict:
        """Returns the Chat Completions response_format payload."""
    
    @property
    def requires_raw_output(self) -> bool: ...
    
    def validate_format(self, format_handler: FormatHandler) -> None: ...
```

**How `from_examples` works:**
1. Collect all extraction classes and their attribute types from examples
2. For each extraction class, build a strict JSON Schema object variant with:
   - `{extraction_class: {type: "string"}}` — the main text field
   - `{extraction_class}_attributes: {type: "object", properties: {attr: type}, ...}` — attributes object
3. Wrap variants in `anyOf` inside `{"type": "object", "properties": {"extractions": {"type": "array", "items": {"anyOf": variants}}}}`
4. This produces the `response_format.json_schema` payload OpenAI's API accepts

---

## 7. Provider Layer (`src/providers/`)

Faithful reproduction of langextract's `core/base_model.py` + `providers/openai.py`.

### Base Language Model (`src/providers/base.py`)

```python
class BaseLanguageModel(ABC):
    """Abstract inference class for LLM inference — identical to langextract."""
    
    @classmethod
    def get_schema_class(cls) -> type[BaseSchema] | None: ...
    
    def apply_schema(self, schema_instance: BaseSchema | None) -> None: ...
    
    def set_fence_output(self, fence_output: bool | None) -> None: ...
    
    @property
    def requires_fence_output(self) -> bool:
        """Returns True if model output needs code fences for parsing.
        Computed from schema.requires_raw_output if no explicit override."""
    
    @property
    def schema(self) -> BaseSchema | None: ...
    
    def merge_kwargs(self, runtime_kwargs=None) -> dict: ...
    
    @abstractmethod
    def infer(self, batch_prompts: Sequence[str], **kwargs
              ) -> Iterator[Sequence[ScoredOutput]]:
        """Batch inference. Yields one Sequence[ScoredOutput] per prompt."""
    
    def infer_batch(self, prompts, batch_size=32) -> list[list[ScoredOutput]]: ...
```

### ScoredOutput (`src/providers/base.py`)

```python
@dataclass(frozen=True)
class ScoredOutput:
    score: float | None = None
    output: str | None = None
```

### OpenAI-Compatible Provider (`src/providers/openai_compat.py`)

```python
class OpenAICompatProvider(BaseLanguageModel):
    """OpenAI-compatible Chat Completions provider via httpx.
    
    Supports: OpenAI, DeepSeek, and any /v1/chat/completions endpoint.
    Uses ThreadPoolExecutor for parallel inference (like langextract's OpenAI provider).
    Applies OpenAISchema as response_format.json_schema.
    """
    model_id: str
    api_key: str | None = None
    base_url: str | None = None
    format_type: FormatType = FormatType.JSON
    temperature: float | None = None
    max_workers: int = 10
    
    @classmethod
    def get_schema_class(cls) -> type[BaseSchema]:
        return OpenAISchema
    
    def apply_schema(self, schema_instance): ...
    
    def infer(self, batch_prompts, **kwargs) -> Iterator[Sequence[ScoredOutput]]:
        """Sends prompts via httpx. Parallelizes with ThreadPoolExecutor when
        batch_prompts > 1 and max_workers > 1."""
    
    def _process_single_prompt(self, prompt, config) -> ScoredOutput:
        """Builds Chat Completions request with response_format, sends via httpx."""
```

**Request building:**
1. If `OpenAISchema` is applied, set `response_format` from `schema.response_format`
2. Otherwise, if `format_type == JSON`, set `response_format: {type: "json_object"}`
3. System message: "You are a helpful assistant that responds in JSON format."
4. Messages: `[system, user(prompt)]`

**Error handling:**
- 429: retry with exponential backoff (1s, 2s, 4s)
- 5xx: retry with exponential backoff
- Parse failure: wrap in `InferenceRuntimeError`
- Config errors: raise `InferenceConfigError`

### Factory (`src/factory.py`)

```python
@dataclass(slots=True, frozen=True)
class ModelConfig:
    model_id: str | None = None
    provider: str | None = None
    provider_kwargs: dict[str, Any] = field(default_factory=dict)

def create_model(
    config: ModelConfig,
    examples: Sequence[ExampleData] | None = None,
    use_schema_constraints: bool = False,
    fence_output: bool | None = None,
) -> BaseLanguageModel:
    """Create a model from config with optional schema constraints.
    
    1. Resolve provider class from model_id (model_id → provider routing)
    2. If use_schema_constraints + examples: create schema via from_examples()
    3. Instantiate provider with merged kwargs
    4. apply_schema() + set_fence_output()
    5. Return configured model
    """

def _kwargs_with_environment_defaults(model_id, kwargs) -> dict: ...
```

---

## 8. Prompting Layer (`src/prompting.py`)

Faithful reproduction of langextract's `prompting.py`.

```python
@dataclass
class PromptTemplateStructured:
    """Structured prompt template with description + few-shot examples."""
    description: str
    examples: list[ExampleData] = field(default_factory=list)

@dataclass
class QAPromptGenerator:
    """Generates Question/Answer format prompts from template + FormatHandler."""
    template: PromptTemplateStructured
    format_handler: FormatHandler
    examples_heading: str = "Examples"
    question_prefix: str = "Q: "
    answer_prefix: str = "A: "
    
    def format_example_as_text(self, example: ExampleData) -> str:
        """Q: {example.text}\nA: {format_handler.format_extraction_example(example.extractions)}\n"""
    
    def render(self, question: str, additional_context: str | None = None) -> str:
        """Render full prompt: description + context + examples + Q/A"""

class PromptBuilder:
    """Builds prompts for text chunks using QAPromptGenerator."""
    def __init__(self, generator: QAPromptGenerator): ...
    def build_prompt(self, chunk_text, document_id, additional_context=None) -> str: ...

class ContextAwarePromptBuilder(PromptBuilder):
    """Prompt builder with cross-chunk context tracking.
    
    Injects text from the previous chunk's tail (context_window_chars)
    to help LLMs resolve coreferences across chunk boundaries.
    Context tracked per document_id.
    """
    def __init__(self, generator, context_window_chars=None): ...
    def build_prompt(self, chunk_text, document_id, additional_context=None) -> str: ...
```

### Domain Adaptation: Extraction-Specific Templates

The `description` field of `PromptTemplateStructured` carries domain-specific extraction instructions (from BACKGROUND.md). Each extraction target (Alternatives, Experiment Design, Indicators, Results) has its own template with few-shot examples specific to that target.

---

## 9. Resolver Layer (`src/resolver.py`)

Faithful reproduction of langextract's `resolver.py`.

### AbstractResolver

```python
class AbstractResolver(ABC):
    """Resolves LLM text outputs into structured Extractions."""
    
    @abstractmethod
    def resolve(self, input_text: str, **kwargs) -> Sequence[Extraction]:
        """Parse LLM output string → list of Extraction objects."""
    
    @abstractmethod
    def align(
        self, extractions: Sequence[Extraction], source_text: str,
        token_offset: int, char_offset: int | None = None,
        enable_fuzzy_alignment: bool = True,
        fuzzy_alignment_threshold: float = 0.75,
        accept_match_lesser: bool = True, *,
        fuzzy_alignment_algorithm: str = "lcs",
        fuzzy_alignment_min_density: float = 1/3,
        **kwargs
    ) -> Iterator[Extraction]:
        """Align extractions to source text, setting token/char intervals
        and alignment_status. Uses exact matching first (difflib),
        then LCS fuzzy alignment fallback."""
```

### Resolver

```python
class Resolver(AbstractResolver):
    """Resolver using FormatHandler for parsing, WordAligner for alignment.
    
    resolve():
      1. Parse input_text via format_handler.parse_output()
      2. Extract ordered extractions from parsed dicts
      3. Handle index keys, attribute keys, type validation
      4. Return Sequence[Extraction]
    
    align():
      1. Tokenize source_text + extraction texts
      2. WordAligner.exact_match (difflib SequenceMatcher)
      3. WordAligner.lcs_fuzzy_match for unmatched extractions
      4. Set token_interval, char_interval, alignment_status
      5. Yield aligned Extractions
    """
```

### WordAligner

```python
class WordAligner:
    """Aligns extraction text to source text tokens.
    
    Phase 1 — Exact matching:
      Uses difflib.SequenceMatcher on tokenized extraction texts (joined with
      delimiter) against tokenized source text. Matching blocks with full
      extraction token count → MATCH_EXACT. Partial matches → MATCH_LESSER
      (if accept_match_lesser=True).
    
    Phase 2 — LCS fuzzy alignment:
      For unmatched extractions, uses an O(n * m²) DP to find the tightest
      source span containing each achievable match count k (1..m). Applies:
      - Coverage gate: matches >= ceil(extraction_len * threshold)
      - Density gate: matches / span_len >= min_density
      Matching span → MATCH_FUZZY. No match → None (extraction kept, unaligned).
    """
```

### Key Constants (from langextract)

```python
_FUZZY_ALIGNMENT_MIN_THRESHOLD = 0.75
_FUZZY_ALIGNMENT_MIN_DENSITY = 1/3
DEFAULT_INDEX_SUFFIX = "_index"
```

---

## 10. Annotation Layer (`src/annotation.py`)

Faithful reproduction of langextract's `annotation.py`.

### Annotator

```python
class Annotator:
    """Orchestrates the full extraction pipeline for documents.
    
    Pipeline:
      Documents → ChunkIterator → make_batches → 
        for each batch:
          build prompts (ContextAwarePromptBuilder) →
          model.infer(batch_prompts) →
          resolver.resolve(scored_output) →
          resolver.align(extractions, chunk_text) →
          accumulate per-document →
        emit completed AnnotatedDocuments (streaming)
    """
    
    def __init__(self, language_model, prompt_template, format_handler): ...
    
    def annotate_documents(
        self, documents: Iterable[Document], resolver,
        max_char_buffer=200, batch_length=1, debug=True,
        extraction_passes=1, context_window_chars=None,
        show_progress=True, tokenizer=None, **kwargs
    ) -> Iterator[AnnotatedDocument]:
        """Annotate documents with streaming emission.
        
        If extraction_passes == 1:
          Single pass: streaming, emit documents as they complete.
        
        If extraction_passes > 1:
          Sequential passes: reprocess each document multiple times,
          merge non-overlapping extractions (first-pass wins for overlaps).
          All passes complete before emission.
        """
    
    def annotate_text(self, text, resolver, ...) -> AnnotatedDocument:
        """Convenience: annotate single text string."""
```

### Single-Pass Streaming Algorithm (from langextract)

```
1. Capture document order + text lazily as chunks are produced
2. For each batch of TextChunks:
   a. Build prompts (ContextAwarePromptBuilder with optional cross-chunk context)
   b. model.infer(batch_prompts) → ScoredOutputs
   c. For each (chunk, output):
      - resolver.resolve(output) → Extractions
      - resolver.align(extractions, chunk_text, offsets) → aligned Extractions
      - Append to per_doc[chunk.document_id]
   d. Emit any documents that have no remaining chunks (streaming)
3. Emit remaining documents
4. Return Iterator[AnnotatedDocument]
```

### Multi-Pass Merge Algorithm (from langextract)

```
1. Run single-pass extraction N times
2. For each document, collect extractions from all passes
3. Merge: extractions from pass 1 are baseline
4. For each later pass's extraction:
   - Check char_interval overlap against existing extractions
   - If no overlap → add to merged set
   - If overlap → keep first-pass extraction (first-pass wins)
5. Return merged AnnotatedDocuments in original input order
```

### Domain Adaptation: Per-Article Extraction Sequence

The zhongnong pipeline uses 4 sequential extraction calls within annotate_text per article section (alternatives, design, indicators, results). This is implemented as **separate `Annotator` instances** with different `PromptTemplateStructured` configs, all operating on the same article text:

```
Article text (methods section) 
  → Annotator(alt_template).annotate_text() → Alternative + Composite extractions
  → GATE CHECK: if no known Alternative → skip article
  → Annotator(design_template).annotate_text() → Swine + Intervention + Control extractions
  → Annotator(indicator_template).annotate_text() → Tissue + Indicator + Method extractions
  → Annotator(result_template).annotate_text(results_text, additional_context={indicator_list, ...})
     → Result extractions
  → Combine all extractions into one AnnotatedDocument
```

This is a thin orchestration layer in `src/extraction.py`, not a modification to `Annotator`.

---

## 11. Extraction API (`src/extraction.py`)

Main entry point, analogous to langextract's `extraction.extract()`.

```python
def extract(
    article_xml: str | Path,
    *,
    model: BaseLanguageModel | None = None,
    model_id: str = "deepseek-chat",
    api_key: str | None = None,
    glossary: GlossaryIndex | None = None,
    max_char_buffer: int = 8000,
    batch_length: int = 1,
    temperature: float | None = 0.1,
    fence_output: bool | None = None,
    use_schema_constraints: bool = True,
    extraction_passes: int = 1,
    context_window_chars: int | None = None,
    show_progress: bool = True,
    debug: bool = False,
) -> ArticleExtractionResult:
    """Extract structured knowledge graph entities + relations from a PMC XML article.
    
    Pipeline:
    1. Parse article XML → sections (abstract, methods, results, discussion)
    2. Extract Alternatives (Gate): if no known Alternative → skip
    3. Extract Experiment Design
    4. Extract Indicators + Tissue Sites + Methods
    5. Extract Results (with indicator/control/tissue context)
    6. Enrich with glossary lookups
    7. Return combined ArticleExtractionResult
    """

def extract_batch(
    article_xmls: Iterable[str | Path],
    max_workers: int = 4,
    **kwargs,
) -> Iterator[ArticleExtractionResult]:
    """Extract from multiple articles with concurrency."""
```

---

## 12. Glossary, Graph, CLI (Domain-Specific Layers)

### Glossary (`src/glossary.py`)

Not in langextract — domain-specific to zhongnong.

```python
class GlossaryIndex:
    def __init__(self, tsv_path: Path): ...
    def lookup(self, name: str) -> GlossaryMatch | None: ...
    def fuzzy_match(self, name: str) -> GlossaryMatch | None: ...

@dataclass
class GlossaryMatch:
    standard_name: str
    class_: str
    subclass: str | None
    match_source: str  # 词表精确匹配|词表同义映射|词表模糊匹配|Other_未匹配
```

### Graph Builder (`src/graph.py`)

```python
@dataclass
class Graph:
    nodes: list[GraphNode]
    edges: list[GraphEdge]

def build_graph(extractions: list[ArticleExtractionResult]) -> Graph: ...
def export_graph(graph: Graph, output_dir: Path) -> None: ...

# Entity dedup: global ID = sha256(f"{entity_type}:{normalized_name}")[:16]
# Edge resolution: inline foreign keys resolved within each article's entity set
```

### CLI (`src/cli.py`)

```bash
zhongnong-kg run --input data/literature_pool.tsv --output output/
zhongnong-kg run --resume
zhongnong-kg debug --xml data/xml/PMC12178903.xml
zhongnong-kg export --checkpoints data/intermediates/ --output output/
```

---

## 13. File Structure

```
src/
  __init__.py
  data.py            # Document, Extraction, AnnotatedDocument, ExampleData, CharInterval, FormatType
  tokenizer.py       # Tokenizer ABC, RegexTokenizer, UnicodeTokenizer, TokenizedText, Token, TokenInterval
  chunking.py        # ChunkIterator, SentenceIterator, TextChunk, make_batches_of_textchunk
  format_handler.py  # FormatHandler — JSON/YAML, fences, wrapper, parse_output, format_example
  schema.py          # BaseSchema ABC, FormatModeSchema
  prompting.py       # PromptTemplateStructured, QAPromptGenerator, PromptBuilder, ContextAwarePromptBuilder
  resolver.py        # AbstractResolver, Resolver, WordAligner
  annotation.py      # Annotator — chunk→prompt→infer→resolve→align→emit
  extraction.py      # extract() — main entry, configures all layers
  factory.py         # ModelConfig, create_model
  
  providers/
    __init__.py
    base.py          # BaseLanguageModel ABC, ScoredOutput
    openai_compat.py # OpenAICompatProvider — httpx async client
    schemas/
      __init__.py
      openai.py      # OpenAISchema — from_examples → response_format json_schema
  
  glossary.py        # GlossaryIndex (domain-specific)
  graph.py           # build_graph, export_graph (domain-specific)
  cli.py             # CLI entry point
  config.py          # Settings (pydantic-settings)

tests/
  test_data.py
  test_tokenizer.py
  test_chunking.py
  test_format_handler.py
  test_schema.py
  test_prompting.py
  test_resolver.py
  test_annotation.py
  test_extraction.py
  test_factory.py
  test_provider_openai_compat.py
  test_glossary.py
  test_graph.py
  test_integration.py
  fixtures/
    article_1.xml
    article_2.xml
```

---

## 14. Data Flow (End-to-End)

```
ALTERNATIVE.tsv ──→ GlossaryIndex (loaded once)

Article XML
  │
  ▼
extract():
  ├─ parse_article_sections(xml) → {abstract, methods, results, discussion} Documents
  │
  ├─ Phase 1: Alternatives (Gate)
  │   ├─ PromptTemplateStructured(alt_description, alt_examples)
  │   ├─ Annotator(model, template, format_handler).annotate_text(methods_text)
  │   │   ├─ ChunkIterator(methods_text) → TextChunks
  │   │   ├─ make_batches(chunks, batch_length)
  │   │   ├─ For each batch:
  │   │   │   ├─ ContextAwarePromptBuilder.build_prompt(chunk)
  │   │   │   ├─ model.infer([prompt]) → ScoredOutput
  │   │   │   ├─ resolver.resolve(scored_output.output) → Extractions
  │   │   │   └─ resolver.align(extractions, chunk_text, offsets)
  │   │   └─ Return AnnotatedDocument(extractions=[...])
  │   ├─ Glossary lookup on each Alternative extraction
  │   └─ GATE: has_known_alternative? → continue or skip
  │
  ├─ Phase 2: Experiment Design
  │   └─ Same Annotator pattern with design_template → Swine, Intervention, Control extractions
  │
  ├─ Phase 3: Indicators
  │   └─ Same Annotator pattern with indicator_template → Tissue, Indicator, Method extractions
  │
  ├─ Phase 4: Results
  │   └─ Same Annotator pattern with result_template + indicator_list context → Result extractions
  │
  └─ Return ArticleExtractionResult (combined AnnotatedDocuments)

Graph Build:
  ├─ Collect all ArticleExtractionResults
  ├─ Deduplicate entities → global IDs
  ├─ Resolve inline relationships → edges
  └─ Export nodes.tsv + edges.tsv
```

---

## 15. Error Handling (langextract pattern)

| Layer | Error | Handling |
|-------|-------|----------|
| **Provider** | API error (429, 5xx) | Retry with exponential backoff; `InferenceRuntimeError` on exhaustion |
| **Provider** | Config error (bad API key, wrong format) | `InferenceConfigError` — raised immediately, no retry |
| **Resolver** | Parse error (malformed JSON, missing wrapper) | `ResolverParsingError`; if `suppress_parse_errors=True`, log warning and return `[]` |
| **Resolver** | Schema error (wrong type, missing index) | `ValueError` → `ResolverParsingError` |
| **WordAligner** | No exact match, no fuzzy match | Extraction kept with `alignment_status=None`, `char_interval=None` |
| **Chunking** | Empty text, no sentence boundaries | `StopIteration` (empty iterator) |
| **Chunking** | TokenUtil returns empty string | `TokenUtilError` |
| **Annotation** | No scored outputs from model | `InferenceOutputError` |
| **Annotation** | Duplicate document_id | `InvalidDocumentError` |
| **Extraction** | Empty examples | `ValueError` — examples required for schema generation |

---

## 16. Configuration

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
    max_char_buffer: int = 8000
    batch_length: int = 1           # chunks per batch (1 = one LLM call per chunk)
    max_workers: int = 4            # parallel article processing (extraction layer)
    extraction_passes: int = 1      # multi-pass for improved recall
    context_window_chars: int | None = None  # cross-chunk context for coreference
    
    # Paths
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    alternative_tsv: Path = Path("ALTERNATIVE.tsv")
    
    # Gate
    skip_if_no_known_alternative: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_prefix="ZN_")
```

---

## 17. Dependencies

```toml
[project]
name = "zhongnong-kg"
version = "2.0.0"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.28",            # Async HTTP for OpenAI-compatible API
    "pydantic>=2.0",          # Settings management
    "pydantic-settings>=2.0", # Env-based config
    "lxml>=5.3",              # PMC XML parsing
    "pyyaml>=6.0",            # YAML parsing in FormatHandler
    "regex>=2024",            # Unicode tokenization
    "click>=8.0",             # CLI
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.25",
]
```

**Removed from v1:** `dspy`, `litellm`, `pandas`, `biopython`, `jsonschema`, `jinja2` (prompts are Python f-string templates, not Jinja2).

---

## 18. Testing Strategy

| Layer | Test Type | Approach |
|-------|----------|----------|
| **Tokenizer** | Unit | Test boundary cases, sentence detection, newline tracking, abbreviation exclusion |
| **Chunking** | Unit | Test sentence grouping, long-sentence splitting, huge-token handling |
| **FormatHandler** | Unit | Test JSON/YAML parsing, fence extraction, wrapper unwrapping, think-tag stripping |
| **Schema** | Unit | Test from_examples → valid OpenAI response_format, requires_raw_output |
| **Prompting** | Unit | Test Q/A format generation, example formatting, cross-chunk context injection |
| **Resolver** | Unit | Test parse_output → Extractions, align → token intervals, exact + fuzzy |
| **Provider** | Unit (mocked httpx) | Test retry logic, schema injection, error handling |
| **Annotation** | Integration (mocked model) | Test full chunk→prompt→infer→resolve→align pipeline with mock model that returns known JSON |
| **Extraction** | Integration (mocked model) | Test 4-phase extraction per article, gate logic |
| **Glossary** | Unit | Test exact + fuzzy matching |
| **Graph** | Unit | Test entity dedup, edge resolution |
| **End-to-End** | Integration | Real XML fixtures → mocked LLM → full pipeline → validate output shape |

---

## 19. Non-Goals

- Real-time/streaming API — batch processing only
- Web UI or dashboard
- Multi-format input beyond PMC XML
- Incremental graph updates — each run produces fresh graph
- Citation network or bibliometric features
