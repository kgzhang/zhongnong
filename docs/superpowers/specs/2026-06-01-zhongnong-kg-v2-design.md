# zhongnong-kg v2: Knowledge Graph Extraction Pipeline

**Date:** 2026-06-02
**Status:** Spec — awaiting review

**Context:** Full rewrite of zhongnong-kg following langextract's architecture faithfully — same layer decomposition, same pipeline pattern, same abstractions — adapted to the swine-nutrition knowledge graph domain.

---

## 1. Architecture Overview

The pipeline faithfully reproduces langextract's layered architecture. Each layer is an isolated, testable abstraction with a single purpose:

```
                    ┌──────────────────────────────────┐
                    │  schemas/                         │
                    │  entities.yaml + relations.yaml   │
                    │  + extraction_phases.yaml         │
                    │  (EXTERNAL DOMAIN CONFIG)          │
                    └──────────────┬───────────────────┘
                                   │ loads at startup
                                   ▼
                    ┌──────────────────────────────────┐
                    │  SchemaRegistry                   │
                    │  src/schema_registry.py           │
                    │  (BRIDGE: config → typed API)     │
                    └──────────────┬───────────────────┘
                                   │ used by all layers
        ┌──────────────┬───────────┼───────────┬──────────────┐
        ▼              ▼           ▼           ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│   Factory     │ │   Annotation │ │   Graph       │ │  Extraction API  │
│ src/factory  │ │src/annotation│ │ src/graph.py  │ │ src/extraction.py│
│              │ │              │ │               │ │                  │
│ ModelConfig  │ │ Annotator:   │ │ merge→dedup   │ │ extract() entry  │
│ create_model │ │ chunk→prompt→│ │ resolve edges │ │ 4-phase pipeline │
│ provider     │ │ infer→resolve│ │ export Neo4j   │ │ per article      │
│ routing      │ │ →align→emit  │ │ CSV            │ │                  │
└──────┬───────┘ └──────┬───────┘ └──────────────┘ └──────────────────┘
       │                │
       ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Providers   │ │   Prompting  │ │   Resolver    │ │   Chunking    │
│ src/providers │ │src/prompting│ │ src/resolver │ │ src/chunking │
│              │ │              │ │              │ │              │
│ BaseLanguage  │ │ PromptTemplate│ │ Abstract      │ │ ChunkIterator │
│ Model ABC     │ │ Structured   │ │ Resolver ABC  │ │ (token-level  │
│              │ │              │ │              │ │  sentence-    │
│ OpenAICompat  │ │ QAPrompt     │ │ Resolver:     │ │  boundary     │
│ Provider      │ │ Generator    │ │ parse + align │ │  aware)       │
│              │ │              │ │              │ │              │
│ Capability    │ │ PromptBuilder│ │ WordAligner:  │ │ TextChunk     │
│ Detection     │ │ ContextAware │ │ difflib exact │ │ make_batches  │
│ Fallback Chain│ │ PromptBuilder│ │ + LCS fuzzy   │ │              │
└──────┬───────┘ └──────────────┘ └──────────────┘ └──────────────┘
       │
       ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│    Schema     │ │ FormatHandler │ │  Tokenizer    │
│ src/schema.py │ │src/format    │ │src/tokenizer │
│              │ │ _handler.py  │ │              │
│ BaseSchema   │ │              │ │ Tokenizer ABC │
│ ABC          │ │ JSON/YAML    │ │              │
│ OpenAISchema │ │ fence detect │ │ RegexTokenizer│
│              │ │ wrapper mgmt │ │ TokenizedText │
│ from_registry│ │ parse_output │ │ Token, Char   │
│ to_provider  │ │ format_ex    │ │ Interval      │
│ _config      │ │              │ │              │
└──────────────┘ └──────────────┘ └──────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  Data Model (src/data.py)                             │
│  Document, Extraction, AnnotatedDocument, ExampleData │
└──────────────────────────────────────────────────────┘
```

### Layer Summary

| # | Layer | Module | Role |
|---|-------|--------|------|
| — | **Schema Config** | `schemas/*.yaml` | External domain model — entities, attributes, relations (NOT code) |
| 3 | **Schema Registry** | `src/schema_registry.py` | Bridge: loads YAML config → typed API for all layers |
| 2 | **Data Model** | `src/data.py` | `Document`, `Extraction`, `AnnotatedDocument`, `ExampleData` — all config-driven |
| 4 | **Tokenizer** | `src/tokenizer.py` | `Tokenizer` ABC, `RegexTokenizer`, `TokenizedText`, `Token` — regex/unicode tokenization |
| 5 | **Chunking** | `src/chunking.py` | `ChunkIterator` (token-level, sentence-boundary-aware), `TextChunk`, batching |
| 6 | **Format Handler** | `src/format_handler.py` | `FormatHandler` — JSON/YAML, fence extraction, wrapper keys |
| 7 | **Schema** | `src/schema.py` | `BaseSchema` ABC → `OpenAISchema` — from_registry() → response_format |
| 8 | **Providers** | `src/providers/` | `BaseLanguageModel` ABC, `OpenAICompatProvider` + capability detection + fallback |
| 9 | **Prompting** | `src/prompting.py` | `PromptTemplateStructured`, `QAPromptGenerator`, `ContextAwarePromptBuilder` |
| 10 | **Resolver** | `src/resolver.py` | `AbstractResolver`, `WordAligner` — parse + align (difflib + LCS fuzzy) |
| 11 | **Annotation** | `src/annotation.py` | `Annotator` — chunk→prompt→infer→resolve→align→emit, streaming, multi-pass |
| 12 | **Factory** | `src/factory.py` | `ModelConfig`, `create_model()` — provider routing, env defaults |
| 13 | **Extraction API** | `src/extraction.py` | `extract()` — 4-phase per-article pipeline |
| 13 | **Glossary** | `src/glossary.py` | `GlossaryIndex` — TSV loader, exact+fuzzy match |
| 13 | **Graph** | `src/graph.py` | Entity dedup, edge resolution, Neo4j CSV export |

---

## 2. Data Model (`src/data.py`)

Faithful reproduction of langextract's `core/data.py` types. Entity types and relations are NOT hardcoded — they are loaded from external configuration files (Section 3).

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
    extraction_class: str          # entity type name (from config, e.g. "Alternative")
    extraction_text: str           # primary display text
    char_interval: CharInterval | None = None
    alignment_status: AlignmentStatus | None = None
    extraction_index: int | None = None
    group_index: int | None = None
    description: str | None = None
    attributes: dict[str, Any] | None = None  # typed according to config
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

**Key design choice:** `Extraction` is a uniform generic type. Entity-specific fields live in `attributes: dict[str, Any]`. The `extraction_class` field holds the entity type name (e.g. `"Alternative"`, `"Result"`). All typing, validation, and schema generation rules come from the external configuration — NOT from per-entity Python classes. This keeps the entire pipeline domain-agnostic and config-driven.

---

## 3. Schema Registry — External Configuration Layer

Entity types, their attributes, extraction guidance, vocabulary bindings, and relationship definitions live in external YAML files. A `SchemaRegistry` loads these at startup and serves as the single source of truth for every layer that needs entity/relation metadata.

### 3.0 Field Classification: LLM vs. Derived

Each attribute in an entity definition is classified by its **source**:

| Source | Meaning | Examples |
|--------|---------|---------|
| `llm` | Extracted by the LLM — the model must identify and type this value | `standard_name`, `dose_value`, `direction`, `p_value` |
| `align` | Derived from alignment position in source text — verbatim copy | `evidence_text` |
| `structure` | Derived from document section/chunk metadata | `source_location` |
| `post` | Computed in post-processing (vocabulary match, unit conversion) | `match_source`, `dose_unit_standard` |

**Critical rule:** `evidence_text` and `source_location` are NEVER LLM output fields. The LLM cannot be trusted to copy-paste verbatim — it paraphrases, truncates, and hallucinates. These fields are populated by the pipeline AFTER alignment determines WHERE in the source text the entity was found.

### 3.1 Configuration Files

**`schemas/entities.yaml`** — defines all entity types with rich metadata:

```yaml
entities:
  Alternative:
    display_name: "抗生素替代物物质"
    description: >
      抗生素替代物物质（标准分类中的具体有效成分）。
      Alternative 与 Composite_Product 分开，因为复合产品本身不是单一物质，
      但其组分可以是 Alternative 或其他 Composite_Product（嵌套复合）。
    primary_text: standard_name
    source: llm                # values extracted by LLM
    extraction_guidance: |
      ## 提取要求
      1. 扫描材料与方法部分，识别所有用作抗生素替代物的物质
      2. 优先将物质名称与下方提供的《Alternative分类》词表进行匹配
      3. 若物质在词表中 → 使用词表内的标准名称和分类
      4. 若物质不在词表中 → 标记为 Other，但保留原文原始名称
      5. 复合制剂按三步处理：拆分组分 → 创建复合节点 → 建立has_component关系
      6. 严禁将多个实体拼接成一个长字符串作为单一实体名
    examples:
      - text: "Pigs were fed a basal diet supplemented with 500 mg/kg thymol (THY, purity ≥ 99%, Sigma-Aldrich)..."
        extractions:
          - extraction_class: Alternative
            extraction_text: thymol
            attributes:
              standard_name: thymol
              abbreviation: THY
              alternative_class: Plant_Extract
              subclass: "Volatile oils and Terpenoids"
              original_text: "thymol (THY, purity ≥ 99%, Sigma-Aldrich)"
    notes: |
      重要约束：
      - 不要拼接字符串创造不存在的复合实体
      - 原封不动保留原文中的原始英文全称及缩写
      - 每个组分分别向词表对齐
    # Vocabulary binding — injected into prompt + used for post-processing
    vocabulary:
      source: ALTERNATIVE.tsv
      key_field: standard_name
      class_field: Alternative_Class
      subclass_field: Subclass
      match_fields: [standard_name, synonyms]   # fields to match against
      match_mode: fuzzy                          # exact → fuzzy → Other
      inject_in_prompt: true                     # include full vocabulary in LLM prompt
    attributes:
      - name: standard_name
        type: string
        required: true
        source: llm
      - name: abbreviation
        type: string
        required: false
        source: llm
      - name: cas_number
        type: string
        required: false
        source: llm
      - name: source_organism
        type: string
        required: false
        source: llm
      - name: is_synthetic
        type: boolean
        required: false
        source: llm
      - name: alternative_class
        type: enum
        values: [Plant_Extract, Trace_Element, Organic_Acid, Probiotic,
                 Polysaccharides_and_Oligosaccharides, Enzyme, Bioactive_Peptides, Other]
        required: true
        source: post              # populated by vocabulary match, not LLM
      - name: subclass
        type: string
        required: false
        source: post              # populated by vocabulary match
      - name: match_source
        type: enum
        values: [词表精确匹配, 词表同义映射, 词表模糊匹配, Other_未匹配]
        required: true
        source: post              # populated by vocabulary match
      - name: original_text
        type: string
        required: true
        source: llm               # the verbatim name as written in paper
      # --- DERIVED FIELDS (not in LLM output schema) ---
      - name: evidence_text
        type: text
        required: true
        source: align             # ← populated from alignment position
      - name: source_location
        type: string
        required: true
        source: structure         # ← populated from document section

  Composite_Product:
    display_name: "复合制剂产品"
    description: "复合制剂产品作为独立实体，通过 has_component 关联各组分"
    primary_text: product_name
    source: llm
    extraction_guidance: |
      ## 复合制剂处理规则（三步法）
      第一步 — 拆分组分：将复合制剂拆分为多个独立的组分实体
      第二步 — 创建复合节点：为复合制剂本身创建独立实体，使用文献中的产品名
      第三步 — 建立关系关联：通过 has_component 连接复合节点与各组分
      严禁将各组分名称通过字符串拼接的方式命名
    examples:
      - text: "Piglets received a commercial multi-strain probiotic (Lactobacillus plantarum + Bacillus subtilis + xylanase, Product X, Novozymes)..."
        extractions:
          - extraction_class: Composite_Product
            extraction_text: "Product X"
            attributes:
              product_name: "Product X"
              manufacturer: "Novozymes"
              is_commercial: true
              components:
                - {standard_name: "Lactobacillus plantarum", entity_type: Alternative}
                - {standard_name: "Bacillus subtilis", entity_type: Alternative}
                - {standard_name: "xylanase", entity_type: Alternative}
    notes: "禁止生成'植物乳杆菌+枯草芽孢杆菌+木聚糖酶'这样的长字符串作为节点名"
    attributes:
      - name: product_name
        type: string
        required: true
        source: llm
      - name: manufacturer
        type: string
        required: false
        source: llm
      - name: is_commercial
        type: boolean
        required: false
        source: llm
      - name: components
        type: json_array
        items: ComponentRef
        required: true
        source: llm
      - name: evidence_text
        type: text
        required: true
        source: align
      - name: source_location
        type: string
        required: true
        source: structure
    inline_relations:
      - name: has_component
        target: [Alternative, Composite_Product]
        via_field: components
        multiple: true

  Result:
    display_name: "显著性结果"
    description: "三元组：部位+指标+变化方向。全量抽取所有报告了统计学比较结果的指标变化。"
    primary_text: indicator_abbreviation
    source: llm
    extraction_guidance: |
      ## 提取规则
      1. 全量抽取：所有在原文中明确报告了统计学比较结果的指标变化均需抽取
      2. 不判断显著性：如实记录原文提供的统计信息，无论P值大小
      3. 对比基准：所有对比必须是"试验组与对照组"的横向比较
      4. 关系类型选择：
         - 生长性能/消化率/肠道形态/代谢物/血液生化 → increases/decreases
         - 基因/蛋白表达 → upregulates/downregulates
         - 微生物相对丰度 → enriches/depletes
         - 无法明确归类 → affects
      5. direction 必须与 relation_type 语义一致
    notes: |
      - 若原文同时报告了绝对空白组与模型攻毒组，对比基准必须是模型攻毒组
      - 同一句话描述多个指标变化时，拆分为多行独立记录
    attributes:
      - name: indicator_abbreviation
        type: string
        required: true
        source: llm
      - name: tissue_site
        type: string
        required: false
        source: llm
      - name: direction
        type: enum
        values: [increased, decreased, no_significant_change]
        required: true
        source: llm
      - name: relation_type
        type: enum
        values: [increases, decreases, upregulates, downregulates, enriches, depletes, affects]
        required: true
        source: llm
      - name: p_value
        type: float
        required: false
        source: llm
      - name: p_value_original_text
        type: string
        required: false
        source: llm
      - name: corrected_significance
        type: string
        required: false
        source: llm
      - name: effect_size
        type: string
        required: false
        source: llm
      - name: time_point
        type: string
        required: false
        source: llm
      - name: subgroup
        type: string
        required: false
        source: llm
      - name: significance_level
        type: enum
        values: [p_less_0.01, p_less_0.05, trend_0.05_0.1, not_significant]
        required: true
        source: llm
      - name: compared_to_group
        type: string
        required: false
        source: llm
      # --- DERIVED FIELDS (not in LLM output schema) ---
      - name: evidence_text
        type: text
        required: true
        source: align
      - name: source_location
        type: string
        required: true
        source: structure
    references:
      - name: indicator_abbreviation
        target_entity: Indicator
        target_field: abbreviation
        edge_type: corresponds_to
      - name: tissue_site
        target_entity: Tissue_Site
        target_field: site_name
        edge_type: occurs_in
      - name: compared_to_group
        target_entity: Control_Group
        target_field: group_name
        edge_type: compared_to

  # ... (all 13 entity types with the same metadata pattern)
```

**Key structural change:** Every attribute now has a `source` field (`llm` | `align` | `structure` | `post`). When generating the LLM's `response_format` JSON Schema, only `source: llm` fields are included. The `source: align`, `source: structure`, and `source: post` fields are populated by the pipeline after extraction + alignment.

**`schemas/relations.yaml`** — defines all relationship types with source/target constraints:

```yaml
relations:
  # 2.1 分类与归属关系
  belongs_to:
    description: "物质属于某个一级分类"
    source: Alternative
    target: Alternative_Class
    cardinality: many_to_one

  has_component:
    description: "复合产品包含某个组分"
    source: Composite_Product
    target: [Alternative, Composite_Product]   # union type
    cardinality: one_to_many
    co_extracted: true   # resolved from inline data, not separate LLM call

  # 2.2 文献与试验设计关系
  contains:
    source: Literature
    target: Experiment
    cardinality: one_to_many

  uses_model:
    source: Experiment
    target: Swine_Model
    cardinality: many_to_one

  uses_animal:
    source: Experiment
    target: Swine
    cardinality: many_to_one

  has_intervention:
    source: Experiment
    target: Intervention
    cardinality: one_to_many

  measures_indicator:
    source: Experiment
    target: Indicator
    cardinality: one_to_many

  uses_control:
    source: Experiment
    target: Control_Group
    cardinality: one_to_many

  uses:
    source: Intervention
    target: [Alternative, Composite_Product]
    cardinality: many_to_one

  applied_to:
    source: Intervention
    target: Swine_Model
    cardinality: many_to_one

  # 2.3 指标与组织关系
  measured_in:
    source: Indicator
    target: Tissue_Site
    cardinality: many_to_one
    co_extracted: true   # resolved from Indicator.measured_in field

  uses_method:
    source: Indicator
    target: Method
    cardinality: many_to_one
    co_extracted: true   # resolved from Indicator.measurement_method field

  # 2.4 结果关系
  increases:
    source: Intervention
    target: Result
    description: "导致指标数值或活性升高"

  decreases:
    source: Intervention
    target: Result
    description: "导致指标数值或活性降低"

  upregulates:
    source: Intervention
    target: Result
    description: "导致基因或蛋白表达上调"

  downregulates:
    source: Intervention
    target: Result
    description: "导致基因或蛋白表达下调"

  enriches:
    source: Intervention
    target: Result
    description: "导致微生物丰度增加"

  depletes:
    source: Intervention
    target: Result
    description: "导致微生物丰度减少"

  corresponds_to:
    source: Result
    target: Indicator
    co_extracted: true   # resolved from Result.indicator_abbreviation reference

  occurs_in:
    source: Result
    target: Tissue_Site
    co_extracted: true   # resolved from Result.tissue_site reference

  compared_to:
    source: Result
    target: Control_Group
    co_extracted: true   # resolved from Result.compared_to_group reference

  # 2.5 属性关系
  has_synonym:
    source: Alternative
    target: Alternative
    description: "物质之间的同义关系"

  # 2.6 指标间关系
  leads_to:
    source: Indicator
    target: Indicator
    description: "一个指标的变化导致另一个指标的变化"

  correlates_with:
    source: Indicator
    target: Indicator
    description: "两个指标之间存在统计相关性"

  part_of:
    source: Indicator
    target: Indicator
    description: "子指标属于父指标（层级关系）"
```

**`schemas/extraction_phases.yaml`** — defines which entities are extracted in each LLM call:

```yaml
phases:
  phase_1_alternatives:
    description: "Extract antibiotic alternatives (GATE)"
    prompt_file: prompts/alternatives.txt
    extracts:
      - Alternative
      - Alternative_Class
      - Composite_Product    # co-extracted with component refs
    gate:
      entity: Alternative
      condition: "alternative_class not in ['Other']"
      on_fail: skip_article

  phase_2_experiment:
    description: "Extract experiment design parameters"
    prompt_file: prompts/experiment.txt
    extracts:
      - Literature
      - Experiment
      - Swine_Model
      - Swine
      - Intervention
      - Control_Group

  phase_3_indicators:
    description: "Extract indicators, tissue sites, methods"
    prompt_file: prompts/indicators.txt
    extracts:
      - Tissue_Site
      - Indicator
      - Method

  phase_4_results:
    description: "Extract statistically evaluated results"
    prompt_file: prompts/results.txt
    extracts:
      - Result
    context_from: [phase_2_experiment, phase_3_indicators]  # feed forward
```

### 3.2 Schema Registry (`src/schema_registry.py`)

The mapping layer that loads external config and provides typed access. All entity/relation metadata — including extraction guidance, examples, vocabulary bindings — flows through this single interface.

```python
@dataclass
class EntityDef:
    name: str
    display_name: str
    description: str
    primary_text: str
    extraction_guidance: str       # instructions for LLM prompt
    examples: list[dict]           # few-shot examples
    notes: str                     # constraints / caveats
    attributes: list[AttributeDef]
    references: list[ReferenceDef]
    inline_relations: list[InlineRelationDef]
    vocabulary: VocabularyBinding | None  # candidate wordlist binding

@dataclass
class AttributeDef:
    name: str
    type: str                     # string | integer | float | boolean | text | enum | json_array
    required: bool
    enum_values: list[str] | None = None
    source: str = "llm"           # llm | align | structure | post

@dataclass
class VocabularyBinding:
    source_path: str              # e.g. "ALTERNATIVE.tsv"
    key_field: str                # field for primary name
    class_field: str | None       # field for classification
    subclass_field: str | None    # field for subclass
    match_fields: list[str]       # fields to match against
    match_mode: str               # exact | fuzzy
    inject_in_prompt: bool        # include vocabulary text in LLM prompt

class Vocabulary:
    """Loaded vocabulary from TSV, keyed by lookup fields."""
    entries: list[dict]           # raw rows
    by_name: dict[str, dict]      # exact-match index (lowercase → row)
    class_hierarchy: dict         # class → subclass → [names]
    
    def format_for_prompt(self) -> str:
        """Render vocabulary as prompt text (classified list with descriptions)."""
    
    def lookup(self, name: str) -> dict | None:
        """Exact match (case-insensitive, whitespace-normalized)."""
    
    def fuzzy_match(self, name: str) -> dict | None:
        """Normalize → substring containment → Levenshtein ≤ 3."""

class SchemaRegistry:
    def __init__(self, config_dir: str | Path = "schemas"): ...
    
    # Entity/relation access
    def entity_def(self, name: str) -> EntityDef: ...
    def relation_def(self, name: str) -> RelationDef: ...
    def phase_defs(self) -> list[ExtractionPhase]: ...
    def all_entity_names(self) -> list[str]: ...
    def all_relation_names(self) -> list[str]: ...
    
    # Vocabulary
    def vocabulary(self, entity_name: str) -> Vocabulary | None: ...
    
    # LLM output schema generation (LLM fields ONLY)
    def llm_output_fields(self, entity_name: str) -> list[AttributeDef]:
        """Return only source=llm attributes. Excludes align/structure/post fields."""
    
    def generate_json_schema(
        self, entity_names: list[str], strict: bool = True
    ) -> dict:
        """Generate OpenAI response_format json_schema.
        ONLY includes source=llm attributes. evidence_text and source_location
        are excluded — they are derived post-alignment.
        """
    
    # Prompt construction from metadata
    def build_extraction_prompt(
        self, entity_names: list[str], include_examples: bool = True
    ) -> str:
        """Build extraction prompt from entity metadata.
        Composes: entity descriptions + extraction_guidance + examples + notes.
        Injects vocabulary text for entities with inject_in_prompt=True.
        """
    
    # Validation
    def validate_extraction(self, extraction: Extraction) -> list[str]:
        """Validate extraction against entity definition. Returns errors."""
    
    # Post-processing (after alignment)
    def post_process(
        self, extraction: Extraction, aligned_text: str | None = None
    ) -> Extraction:
        """Apply post-processing: vocabulary match, unit conversion, etc.
        Populates source=post fields like match_source, alternative_class.
        """
    
    # Edge resolution
    def edge_from_reference(
        self, source_extraction: Extraction, ref: ReferenceDef,
        entity_id_map: dict
    ) -> tuple[str, str, str] | None: ...
    
    def edges_from_inline(
        self, extraction: Extraction, rel: InlineRelationDef
    ) -> list[tuple[str, str, str]]: ...
```

### 3.3 How Config Metadata Flows Into the Pipeline

```
schemas/entities.yaml
  ├─ entity.description ──────────→ PromptTemplateStructured.description
  ├─ entity.extraction_guidance ──→ Prompt: "## 提取要求" section
  ├─ entity.examples ─────────────→ ExampleData for few-shot Q/A format
  ├─ entity.notes ────────────────→ Prompt: "## 重要约束" section
  ├─ entity.vocabulary ───────────→ Vocabulary.format_for_prompt() injected into prompt
  │                                 + Vocabulary.lookup() in post-processing
  ├─ attr.source=llm ─────────────→ OpenAISchema.generate_json_schema()
  │                                 (ONLY these fields in response_format)
  ├─ attr.source=align ───────────→ EvidenceExtractor (Section 10.3)
  │                                 populates evidence_text from aligned char_interval
  ├─ attr.source=structure ────────→ SourceLocationResolver (Section 10.4)
  │                                 populates source_location from section/chunk metadata
  └─ attr.source=post ────────────→ SchemaRegistry.post_process()
                                    populates match_source, alternative_class, etc.
```

### 3.4 Vocabulary Injection Example

For the `Alternative` entity, the `ALTERNATIVE.tsv` vocabulary is:
1. **Loaded at startup** into `Vocabulary` — builds exact-match index + class hierarchy
2. **Injected into prompt** via `Vocabulary.format_for_prompt()`:
   ```
   ## 候选词表（Alternative分类）
   
   ### Plant_Extract — 植物提取物
   [Volatile oils and Terpenoids] thymol (THY), carvacrol (CAR), eugenol (EUG), ...
   [Phenols and Flavonoids] curcumin, quercetin, kaempferol, ...
   [Alkaloids] berberine, sanguinarine, ...
   ...
   
   ### Trace_Element — 微量元素
   [Inorganic trace elements] Iron (Fe): ferrous sulfate (FeSO₄), ...
   ...
   ```
3. **Matched post-extraction** — after LLM returns entity values, `Vocabulary.lookup()` + `Vocabulary.fuzzy_match()` classify each extracted substance, populating `alternative_class`, `subclass`, and `match_source`.

### 3.5 Field Source Classification (Complete)

| Attribute | Source | Populated By | When |
|-----------|--------|-------------|------|
| `standard_name`, `abbreviation`, `original_text` | `llm` | LLM extraction | During `Annotator.annotate_text()` |
| `dose_value`, `dose_unit_original`, `administration_route` | `llm` | LLM extraction | During `Annotator.annotate_text()` |
| `direction`, `p_value`, `significance_level`, `relation_type` | `llm` | LLM extraction | During `Annotator.annotate_text()` |
| `evidence_text` | `align` | `EvidenceExtractor` | After `resolver.align()` — verbatim from source |
| `source_location` | `structure` | `SourceLocationResolver` | After alignment — section/table/figure metadata |
| `alternative_class`, `subclass`, `match_source` | `post` | `Vocabulary.lookup()` | After LLM extraction — vocabulary matching |
| `dose_unit_standard` | `post` | Unit normalizer | After LLM extraction — unit conversion |

---

## 4. Tokenizer (`src/tokenizer.py`)

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

## 5. Chunking (`src/chunking.py`)

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

## 6. Format Handler (`src/format_handler.py`)

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

## 7. Schema Layer (`src/schema.py`)

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

## 8. Provider Layer (`src/providers/`)

Faithful reproduction of langextract's `core/base_model.py` + `providers/openai.py`.

Extended with **model capability detection** and **fallback chain** to support DeepSeek, Qwen, and other OpenAI-compatible models that differ in their structured-output support levels.

### Model Capability Profiles

Different models support different levels of structured output. The provider detects capabilities from the model ID and adjusts behavior:

| Model Family | `json_schema` (strict) | `json_object` mode | native JSON |
|-------------|----------------------|-------------------|-------------|
| OpenAI (gpt-4o, gpt-4o-mini) | ✅ full support | ✅ | ✅ |
| DeepSeek (deepseek-chat, deepseek-reasoner) | ⚠️ limited (no strict mode) | ✅ | ✅ |
| Qwen (qwen-max, qwen-plus via DashScope) | ❌ | ✅ (some models) | ⚠️ fences needed |
| Qwen (qwen2.5 via Ollama/vLLM) | ❌ | ❌ | ❌ fence-only |

**Capability detection** — model ID prefix matching:

```python
@dataclass
class ModelCapabilities:
    supports_json_schema: bool       # response_format json_schema
    supports_json_schema_strict: bool # strict mode within json_schema
    supports_json_object: bool       # response_format {type: json_object}
    supports_system_message: bool    # system role in messages
    requires_fence_output: bool      # need ```json fences in prompt
    max_context_tokens: int          # approximate context window size

def detect_capabilities(model_id: str) -> ModelCapabilities:
    """Detect model capabilities from model ID prefix.
    
    DeepSeek: supports json_object, supports json_schema without strict.
    Qwen DashScope: supports json_object on qwen-max/plus.
    Qwen open-source: no structured output — fence-only mode.
    OpenAI: full json_schema with strict.
    """
```

### Fallback Chain

When the provider is asked to apply a schema, it tries the best available mode:

```
1. json_schema (strict)     ← best: model CANNOT deviate from schema
2. json_schema (non-strict) ← good: schema constraint without strict validation
3. json_object mode         ← ok: model MUST output valid JSON, enforced by prompt
4. fence-only mode          ← fallback: prompt asks for ```json, FormatHandler extracts it
```

```python
class SchemaApplication:
    """Result of applying a schema to a provider."""
    mode: str                   # "json_schema_strict" | "json_schema" | "json_object" | "fence"
    response_format: dict | None  # API payload for response_format, or None for fence mode
    prompt_addition: str | None   # Extra prompt text for fence mode ("Output valid JSON in ```json fences.")
```

The `apply_schema()` method:
1. Checks `detect_capabilities(model_id)`
2. Selects the best available mode
3. If fence-only: sets `requires_fence_output = True`, adds JSON instruction to system prompt
4. If json_object: sets `response_format: {type: "json_object"}` 
5. If json_schema: sets `response_format` from schema
6. Synchronizes `FormatHandler.use_fences` to match

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

## 9. Prompting Layer (`src/prompting.py`)

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

## 10. Resolver Layer (`src/resolver.py`)

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

### EvidenceExtractor — Verbatim Evidence from Alignment

The LLM is NOT asked to produce `evidence_text`. Instead, after `resolver.align()` has set `char_interval` on each extraction (pointing to the extraction's position in the source text), the `EvidenceExtractor` derives evidence_text by extracting the surrounding sentence(s) from the source document — guaranteeing verbatim text.

```python
class EvidenceExtractor:
    """Derives evidence_text from aligned char_interval in source text.
    
    Why not LLM? LLMs paraphrase, truncate, and hallucinate when asked to
    "copy the exact sentence." Alignment gives us the exact character position
    — we extract the sentence directly from the source.
    """
    
    def extract_evidence(
        self,
        extraction: Extraction,
        document_text: str,            # full section text
        tokenized_text: TokenizedText, # tokenized version
        context_sentences: int = 2,    # include N surrounding sentences
    ) -> str:
        """Extract verbatim evidence sentence(s) from source text.
        
        Algorithm:
        1. If extraction.char_interval is None → return "" (unaligned)
        2. Find the sentence(s) containing the char_interval span
           using find_sentence_range() from tokenizer
        3. Expand to include context_sentences before and after
        4. Extract the exact text from document_text using char intervals
        5. Return verbatim text (no modification, no truncation)
        
        This guarantees evidence_text is an EXACT copy from the source.
        """
    
    def extract_evidence_batch(
        self,
        extractions: list[Extraction],
        document_text: str,
        tokenized_text: TokenizedText,
        context_sentences: int = 2,
    ) -> None:
        """Mutate extractions in-place, setting evidence_text on each."""


class SourceLocationResolver:
    """Derives source_location from document structure + alignment position.
    
    source_location indicates the structured location in the article:
    - Section-level: "Methods 2.3", "Results 3.1", "Discussion"
    - Table/Figure: "Table 2", "Figure 3" (detected from nearby text)
    - Paragraph-level: "Abstract", "Introduction"
    
    NOT an LLM output field. Determined from:
    1. The section Document the extraction came from (document_id metadata)
    2. The char_interval cross-referenced with article section boundaries
    3. Table/figure mentions near the aligned span
    """
    
    def resolve_location(
        self,
        extraction: Extraction,
        section_id: str,              # e.g. "methods", "results"
        section_text: str,
        char_offset: int,             # chunk's char offset in section
        table_figure_patterns: list[re.Pattern] | None = None,
    ) -> str:
        """Determine structured source location.
        
        Algorithm:
        1. Base location = section_id (e.g. "Methods")
        2. If section has numbered subsections, detect subsection from
           char_interval position (e.g. "Methods 2.3")
        3. Scan text near char_interval for table/figure references
           (e.g. "Table 2", "Figure 3") using regex
        4. Return the most specific location string
        
        Examples:
        - "Materials and Methods, §2.3"
        - "Results, Table 2"
        - "Discussion, §4.1"
        """
    
    def resolve_batch(
        self,
        extractions: list[Extraction],
        section_id: str,
        section_text: str,
        char_offset: int,
    ) -> None:
        """Mutate extractions in-place, setting source_location on each."""
```

### Updated Resolver Pipeline (resolve → align → evidence → location)

```
resolver.resolve(llm_output)          → Extractions (values only)
resolver.align(extractions, text)     → Extractions with char_interval set
EvidenceExtractor.extract_evidence()  → Extractions with evidence_text set
SourceLocationResolver.resolve()      → Extractions with source_location set
SchemaRegistry.post_process()         → Extractions with match_source, etc.
```

## 11. Annotation Layer (`src/annotation.py`)

Faithful reproduction of langextract's `annotation.py`.

### Annotator

```python
class Annotator:
    """Orchestrates the full extraction pipeline for documents.
    
    Pipeline (extended from langextract with evidence + location derivation):
      Documents → ChunkIterator → make_batches → 
        for each batch:
          build prompts (ContextAwarePromptBuilder) →
          model.infer(batch_prompts) →
          resolver.resolve(scored_output) →       # parse LLM → Extractions
          resolver.align(extractions, chunk_text) → # set char_interval
          evidence_extractor.extract(extractions) → # derive evidence_text
          source_location_resolver.resolve(extractions) → # derive source_location
          accumulate per-document →
        emit completed AnnotatedDocuments (streaming)
    """
    
    def __init__(self, language_model, prompt_template, format_handler,
                 evidence_extractor=None, source_location_resolver=None): ...
    
    def annotate_documents(
        self, documents: Iterable[Document], resolver,
        max_char_buffer=200, batch_length=1, debug=True,
        extraction_passes=1, context_window_chars=None,
        show_progress=True, tokenizer=None, **kwargs
    ) -> Iterator[AnnotatedDocument]:
        """Annotate documents with streaming emission.
        
        Each chunk flows through:
        resolve → align → evidence → source_location → accumulate
        
        If extraction_passes == 1:
          Single pass: streaming, emit documents as they complete.
        
        If extraction_passes > 1:
          Sequential passes, merge non-overlapping (first-pass wins).
        """
    
    def annotate_text(self, text, resolver, ...) -> AnnotatedDocument:
        """Convenience: annotate single text string."""
```

### Single-Pass Streaming Algorithm (from langextract, extended)

```
1. Capture document order + text lazily as chunks are produced
2. For each batch of TextChunks:
   a. Build prompts (ContextAwarePromptBuilder with optional cross-chunk context)
   b. model.infer(batch_prompts) → ScoredOutputs
   c. For each (chunk, output):
      - resolver.resolve(scored_output.output) → Extractions (values only)
      - resolver.align(extractions, chunk_text, token_offset, char_offset)
        → Extractions with char_interval + alignment_status set
      - evidence_extractor.extract_evidence(extractions, document_text, tokenized_text)
        → Extractions with evidence_text set (verbatim from source, 1-3 sentences)
      - source_location_resolver.resolve(extractions, section_id, section_text, char_offset)
        → Extractions with source_location set (e.g. "Methods 2.3", "Table 2")
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

## 12. Extraction API (`src/extraction.py`)

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

## 13. Glossary, Graph, CLI (Domain-Specific Layers)

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
class GraphNode:
    id: str                     # deterministic global ID
    labels: list[str]           # Neo4j labels (entity type + "Entity")
    properties: dict[str, Any]  # all attributes
    source_pmids: list[str]     # PMIDs this entity appeared in

@dataclass  
class GraphEdge:
    source_id: str
    target_id: str
    type: str                   # relation type
    properties: dict[str, Any]  # evidence_text, source_pmids, etc.

@dataclass
class Graph:
    nodes: list[GraphNode]
    edges: list[GraphEdge]


def build_graph(
    extractions: list[ArticleExtractionResult],
    registry: SchemaRegistry,
) -> Graph:
    """Phase 1: Collect all extractions from all articles.
    Phase 2: Deduplicate entities → assign global IDs.
    Phase 3: Resolve edges (references + inline relations + cross-article).
    Phase 4: Validate against relation definitions in registry.
    """


def export_neo4j_csv(graph: Graph, output_dir: Path) -> None:
    """Write nodes.csv and edges.csv in Neo4j CSV import format.
    
    These files can be directly imported with:
      LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
      CREATE (n) SET n = row;
      
      LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
      MATCH (a) WHERE a.id = row.source_id
      MATCH (b) WHERE b.id = row.target_id
      CREATE (a)-[r:REL_TYPE]->(b) SET r = properties(row);
    """
```

### Neo4j CSV Format

**`nodes.csv`** — Neo4j-compatible with `:ID` and `:LABEL` columns:

```csv
entity_id:ID,entity_type:LABEL,name,doi,pmid,title,journal,abstract_conclusion,publication_year,study_design,standard_name,abbreviation,alternative_class,cas_number,...other_properties...,evidence_text,source_location,source_pmids
alt_0001,Alternative;Entity,thymol,,,,,,,thymol,THY,Plant_Extract,89-83-8,...,"Thymol was added to the diet...","Table 1","36789012"
alt_0002,Alternative;Entity,zinc oxide,,,,,,,zinc oxide,ZnO,Trace_Element,1314-13-2,...,"Zinc oxide was supplemented...","Methods 2.3","36789012;36789013"
lit_0001,Literature;Entity,,10.1016/j.x.2023.100123,36789012,"Effects of thymol on...","Journal of Animal Science","In conclusion, thymol...",2023,completely_randomized,,,,,,,,,"36789012"
```

**Key Neo4j conventions:**
- `entity_id:ID` — unique node identifier (use `:ID` suffix in header)
- `entity_type:LABEL` — semi-colon separated labels for multi-label nodes (use `:LABEL` suffix)
- All entity properties as columns (sparse: empty string when not applicable to that entity type)
- `source_pmids` — pipe-separated PMID list for dedup tracking
- `evidence_text` — quoted with standard CSV escaping

**`edges.csv`** — Neo4j-compatible with `:START_ID`, `:END_ID`, `:TYPE`:

```csv
source_id:START_ID,target_id:END_ID,relation_type:TYPE,evidence_text,source_pmids,doi,pmid
alt_0001,altclass_0001,belongs_to,"Thymol (THY) was used as a plant extract...","36789012","10.1016/j.x.2023.100123","36789012"
comp_0001,alt_0001,has_component,"Product X contained thymol...","36789012","10.1016/j.x.2023.100123","36789012"
exp_0001_36789012,alt_0001,uses,"Thymol was administered in the diet...","36789012","10.1016/j.x.2023.100123","36789012"
int_0001_36789012,res_0001_36789012,increases,"Compared with the control group, thymol significantly increased ADG (P<0.05)","36789012","10.1016/j.x.2023.100123","36789012"
```

**Key Neo4j conventions:**
- `:START_ID` and `:END_ID` reference `entity_id:ID` values from `nodes.csv`
- `:TYPE` — the relationship type (Neo4j will create `[:REL_TYPE]` with this value)
- All edge properties as additional columns (evidence_text, source_pmids, etc.)
- `source_pmids` and `doi`/`pmid` for provenance tracking

### Entity ID Scheme

Deterministic, no registry needed:

```python
def entity_global_id(entity_type: str, primary_text: str, 
                     article_pmid: str | None = None) -> str:
    """Generate deterministic global entity ID.
    
    For article-scoped entities (Result, Experiment, Intervention):
      → sha256(f"{entity_type}:{article_pmid}:{primary_text}")[:12]
    
    For global entities (Alternative, Alternative_Class, Tissue_Site, 
                       Indicator, Method):
      → sha256(f"{entity_type}:{normalized_name}")[:12]
      where normalized_name = primary_text.lower().strip()
    
    Prefix: first 4 chars of entity type + "_" + 8 hex chars
    Example: alt_a1b2c3d4, res_e5f6g7h8
    """
```

### Dedup Strategy

Two-tier:

1. **Within article**: `extraction_class` + `extraction_text` + `extraction_index` — Resolver handles ordering, duplicates within a chunk naturally rare.

2. **Across articles** (graph build):
   - **Global entities** (Alternative, Indicator, Method, Tissue_Site): matched by `(entity_type, normalized_name)`. Same entity appearing in multiple articles → single node, accumulated `source_pmids`.
   - **Article-scoped entities** (Result, Experiment, Intervention, Swine, Swine_Model, Control_Group, Literature): unique per article. `article_pmid` is part of the ID. Two articles' "Experiment 1" remain separate nodes.

### Edge Resolution Pipeline

```python
def resolve_edges(
    article_extractions: list[ArticleExtractionResult],
    entity_id_map: dict[tuple[str, str, str | None], str],  # → global ID
    registry: SchemaRegistry,
) -> list[GraphEdge]:
    """For each article:
    1. Co-extracted inline relations (has_component from Composite_Product.components)
    2. Reference-field edges (Result.indicator_abbreviation → Indicator)
    3. Explicit relation edges (Intervention → Result via increases/decreases/etc.)
    4. Validate all edges against relation definitions (source/target type check)
    5. Drop edges with unresolved targets + log warning
    """
```

### CLI (`src/cli.py`)

```bash
zhongnong-kg run --input data/literature_pool.tsv --output output/
zhongnong-kg run --resume
zhongnong-kg debug --xml data/xml/PMC12178903.xml
zhongnong-kg export --checkpoints data/intermediates/ --output output/
```

---

## 14. File Structure

```
zhongnong-kg/
├── schemas/                         # EXTERNAL CONFIG — domain model, not code
│   ├── entities.yaml                # 13 entity types: descriptions, extraction_guidance,
│   │                                #   examples, notes, vocabulary bindings, field source
│   ├── relations.yaml               # 20+ relation types: source/target, cardinality, co_extracted
│   └── extraction_phases.yaml       # 4 phases: entities per LLM call, gate rules, context feeds
│
├── prompts/                         # Prompt templates (alternate: auto-generated from entity metadata)
│   ├── alternatives.txt             #   Or: SchemaRegistry.build_extraction_prompt() at runtime
│   ├── experiment.txt
│   ├── indicators.txt
│   └── results.txt
│
├── src/
│   ├── __init__.py
│   ├── data.py                      # Document, Extraction, AnnotatedDocument, ExampleData, CharInterval
│   ├── tokenizer.py                 # Tokenizer ABC, RegexTokenizer, UnicodeTokenizer, TokenizedText, Token, TokenInterval
│   ├── chunking.py                  # ChunkIterator, SentenceIterator, TextChunk, make_batches_of_textchunk
│   ├── format_handler.py            # FormatHandler — JSON/YAML, fences, wrapper, parse_output
│   ├── schema_registry.py           # SchemaRegistry — loads YAML, Vocabulary, entity/relation defs,
│   │                                #   llm_output_fields(), generate_json_schema(),
│   │                                #   build_extraction_prompt(), post_process()
│   ├── schema.py                    # BaseSchema ABC, FormatModeSchema
│   ├── prompting.py                 # PromptTemplateStructured, QAPromptGenerator, PromptBuilder, ContextAwarePromptBuilder
│   ├── resolver.py                  # AbstractResolver, Resolver, WordAligner — parse + align (difflib + LCS fuzzy)
│   ├── evidence.py                  # EvidenceExtractor — verbatim sentences from aligned char_interval
│   ├── source_location.py           # SourceLocationResolver — section/table/figure from structure
│   ├── annotation.py                # Annotator — chunk→prompt→infer→resolve→align→evidence→location→emit
│   ├── extraction.py                # extract() — main entry: configures layers, 4-phase pipeline
│   ├── factory.py                   # ModelConfig, create_model — provider resolution + env defaults
│   │
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseLanguageModel ABC, ScoredOutput, ModelCapabilities
│   │   ├── capabilities.py          # detect_capabilities(), fallback chain
│   │   ├── openai_compat.py         # OpenAICompatProvider — httpx async client
│   │   └── schemas/
│   │       ├── __init__.py
│   │       └── openai.py            # OpenAISchema — from_registry() → response_format json_schema
│   │
│   ├── graph.py                     # build_graph, resolve_edges, export_neo4j_csv
│   ├── cli.py                       # CLI entry point (click)
│   └── config.py                    # Settings (pydantic-settings)
│
├── tests/
│   ├── test_data.py
│   ├── test_tokenizer.py
│   ├── test_chunking.py
│   ├── test_format_handler.py
│   ├── test_evidence.py             # EvidenceExtractor: verbatim extraction, sentence boundaries
│   ├── test_source_location.py      # SourceLocationResolver: section/table/figure detection
│   ├── test_schema_registry.py
│   ├── test_schema.py
│   ├── test_prompting.py
│   ├── test_resolver.py
│   ├── test_annotation.py
│   ├── test_extraction.py
│   ├── test_factory.py
│   ├── test_provider_capabilities.py
│   ├── test_provider_openai_compat.py
│   ├── test_glossary.py
│   ├── test_graph.py
│   ├── test_integration.py
│   └── fixtures/
│       ├── article_1.xml
│       ├── article_2.xml
│       └── llm_responses/           # Recorded LLM outputs for deterministic tests
│
├── ALTERNATIVE.tsv                  # Glossary data (reused from v1)
├── pyproject.toml
├── .env
└── Makefile
```

**Key structural principles:**
- `schemas/` is the domain model — entity types, attributes, relations are ALL here, NOT in Python source
- `prompts/` is the LLM instruction layer — separate from code for easy iteration by domain experts
- `src/schema_registry.py` is the BRIDGE — loads config, exposes typed API to all other modules
- Domain code (`glossary.py`, `graph.py`) depends on the registry, not on hardcoded entity lists

---

## 15. Data Flow (End-to-End)

```
ALTERNATIVE.tsv ──→ Vocabulary.load() ──→ SchemaRegistry.vocabulary("Alternative")
                                          │
schemas/entities.yaml ──┐                 │
schemas/relations.yaml   ─┤                │
schemas/extraction_phases.yaml ─┘         │
         │                                │
         ▼                                │
   SchemaRegistry (loaded once at startup)│
         │                                │
         │  entity_def() → extraction_guidance + examples + notes
         │  generate_json_schema() → response_format (LLM fields only)
         │  vocabulary() → formatted glossary text injected in prompt
         │
         ▼
Article XML
  │
  ▼
extract():
  ├─ 1. parse_article_sections(xml) 
  │     → {abstract, methods, results, discussion} as Documents
  │       each with section_id for source_location derivation
  │
  ├─ 2. Phase 1 — Alternatives (Gate)
  │   ├─ PromptTemplate from entity metadata (Alternatives + Composites)
  │   ├─ Vocabulary injected: formatted ALTERNATIVE.tsv glossary
  │   ├─ Annotator.annotate_text(methods_text)
  │   │   └─ chunk → prompt(build) → model.infer() 
  │   │       → resolver.resolve() → Extractions (LLM values only)
  │   │       → resolver.align() → char_interval set
  │   │       → evidence_extractor.extract() → evidence_text verbatim
  │   │       → source_location_resolver.resolve() → "Methods 2.3"
  │   ├─ SchemaRegistry.post_process()
  │   │   → Vocabulary.lookup("thymol") → alternative_class=Plant_Extract
  │   │   → Vocabulary.fuzzy_match(...) → match_source=词表精确匹配
  │   └─ GATE: has_known_alternative?
  │
  ├─ 3. Phase 2 — Experiment Design
  │   └─ Same pipe: prompt(metadata) → infer → resolve → align 
  │       → evidence → location → post_process
  │
  ├─ 4. Phase 3 — Indicators + Tissue Sites + Methods
  │   └─ Same pipe: prompt(metadata) → infer → resolve → align 
  │       → evidence → location → post_process
  │
  ├─ 5. Phase 4 — Results
  │   ├─ Context injected: indicator_list + control_group_list + tissue_list
  │   └─ Same pipe: prompt(metadata+context) → infer → resolve → align 
  │       → evidence → location → post_process
  │
  └─ 6. Return ArticleExtractionResult
        (all extractions with ALL fields populated)

Graph Build:
  ├─ Collect all ArticleExtractionResults
  ├─ Deduplicate entities → global IDs (schema-driven ID scheme)
  ├─ Resolve edges: references + inline_relations (schema-driven)
  └─ Export nodes.csv + edges.csv (Neo4j format)
```

### Field Population Timeline

| Step | Fields Populated | Source |
|------|-----------------|--------|
| LLM infer + resolve | `standard_name`, `abbreviation`, `dose_value`, `direction`, `p_value`, `significance_level`, ... | `source: llm` |
| resolver.align() | `char_interval`, `token_interval`, `alignment_status` | alignment engine |
| evidence_extractor | `evidence_text` (verbatim 1-3 sentences from source) | `source: align` |
| source_location_resolver | `source_location` ("Methods 2.3", "Table 2", etc.) | `source: structure` |
| SchemaRegistry.post_process() | `alternative_class`, `subclass`, `match_source`, `dose_unit_standard` | `source: post` |

---

## 16. Error Handling (langextract pattern)

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

## 17. Configuration

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

## 18. Dependencies

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
    "pyyaml>=6.0",            # YAML config loading (entities.yaml, relations.yaml)
    "regex>=2024",            # Unicode tokenization
    "click>=8.0",             # CLI
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.25",
]
```

**Removed from v1:** `dspy`, `litellm`, `pandas`, `biopython`, `jsonschema`, `jinja2`. Prompts are plain text files loaded at runtime. Schema validation is via `SchemaRegistry` not `jsonschema`.

---

## 19. Testing Strategy

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

## 20. Non-Goals

- Real-time/streaming API — batch processing only
- Web UI or dashboard
- Multi-format input beyond PMC XML
- Incremental graph updates — each run produces fresh graph
- Citation network or bibliometric features
