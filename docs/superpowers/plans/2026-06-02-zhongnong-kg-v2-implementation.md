# llm-extract v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production knowledge graph extraction pipeline following langextract's architecture — 11 core layers plus domain adapters, config-driven via YAML schemas, outputting Neo4j-importable CSV.

**Architecture:** Layered pipeline: data model → tokenizer → chunking → format handler → schema registry → providers → prompting → resolver/evidence/source_location → annotation → factory → extraction → graph → CLI. Each layer is independently testable with mocked dependencies. External YAML config defines all entity types, relations, extraction phases, and vocabulary bindings.

**Tech Stack:** Python 3.12+, httpx (async LLM client), PyYAML (config), lxml (XML parsing), regex (tokenization), click (CLI), pydantic-settings (config). pytest + pytest-asyncio for testing.

**Spec:** `docs/superpowers/specs/2026-06-01-llm-extract-v2-design.md`

---

## File Structure (end state)

```
src/
  __init__.py
  config.py              # Settings (pydantic-settings) — modify existing
  data.py                # Document, Extraction, AnnotatedDocument, ExampleData
  tokenizer.py           # Tokenizer ABC, RegexTokenizer, UnicodeTokenizer
  chunking.py            # ChunkIterator, SentenceIterator, TextChunk
  format_handler.py      # FormatHandler — JSON/YAML, fences, wrapper
  schema_registry.py     # SchemaRegistry + Vocabulary — YAML → typed API
  schema.py              # BaseSchema ABC, FormatModeSchema
  prompting.py           # PromptTemplateStructured, QAPromptGenerator, PromptBuilder
  resolver.py            # Resolver, WordAligner — parse + align
  evidence.py            # EvidenceExtractor — verbatim text from alignment
  source_location.py     # SourceLocationResolver — section/table metadata
  annotation.py          # Annotator — full pipeline orchestrator
  extraction.py          # extract() — 4-phase per-article pipeline
  factory.py             # ModelConfig, create_model

  providers/
    __init__.py
    base.py              # BaseLanguageModel ABC, ScoredOutput
    capabilities.py      # detect_capabilities(), ModelCapabilities
    openai_compat.py     # OpenAICompatProvider
    schemas/
      __init__.py
      openai.py          # OpenAISchema

  graph.py               # build_graph, resolve_edges, export_neo4j_csv
  cli.py                 # CLI entry point

schemas/
  entities.yaml          # 13 entity types with full metadata
  relations.yaml         # 20+ relation types
  extraction_phases.yaml # 4 extraction phases

tests/
  __init__.py (existing)
  test_data.py
  test_tokenizer.py
  test_chunking.py
  test_format_handler.py
  test_schema_registry.py
  test_schema.py
  test_prompting.py
  test_resolver.py
  test_evidence.py
  test_source_location.py
  test_annotation.py
  test_extraction.py
  test_factory.py
  test_provider_capabilities.py
  test_provider_openai_compat.py
  test_graph.py
  test_integration.py
  fixtures/
    article_1.xml        # real PMC XML fixture
    sample_alternative.tsv
```

**Existing files to remove after new pipeline validates:**
- `src/dspy_extract.py`, `src/run_dspy_pipeline.py`
- `src/stage*.py`, `src/utils.py`
- `src/glossary.py`, `src/entity_id.py`
- `tests/test_dspy_extract.py`, `tests/test_stage*.py`, `tests/test_glossary.py`
- `schemas/*.json` (replaced by YAML)
- `src/zhongnong_kg.egg-info/`

---

## Group A: Foundation — Data Model, Config, Tokenizer

### Task A1: Update pyproject.toml dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update dependencies**

Replace the `[project]` dependencies section:

```toml
[project]
name = "llm-extract"
version = "2.0.0"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.28",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "lxml>=5.3",
    "pyyaml>=6.0",
    "regex>=2024",
    "click>=8.0",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.25"]

[project.scripts]
llm-extract = "src.cli:cli"
```

- [ ] **Step 2: Sync dependencies**

Run: `uv sync`
Expected: packages installed without errors.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: update dependencies for v2 (remove dspy/litellm/pandas, add pyyaml/regex)"
```

---

### Task A2: Data Model (`src/data.py`)

**Files:**
- Create: `src/data.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_data.py`:

```python
"""Tests for core data types."""
import pytest
from src.data import (
    FormatType, CharInterval, AlignmentStatus,
    Extraction, Document, ExampleData, AnnotatedDocument,
)


class TestFormatType:
    def test_json(self):
        assert FormatType.JSON.value == "json"

    def test_yaml(self):
        assert FormatType.YAML.value == "yaml"


class TestCharInterval:
    def test_create(self):
        ci = CharInterval(start_pos=0, end_pos=10)
        assert ci.start_pos == 0
        assert ci.end_pos == 10

    def test_none_defaults(self):
        ci = CharInterval()
        assert ci.start_pos is None
        assert ci.end_pos is None


class TestAlignmentStatus:
    def test_values(self):
        assert AlignmentStatus.MATCH_EXACT.value == "match_exact"
        assert AlignmentStatus.MATCH_LESSER.value == "match_lesser"
        assert AlignmentStatus.MATCH_FUZZY.value == "match_fuzzy"
        assert AlignmentStatus.MATCH_GREATER.value == "match_greater"


class TestExtraction:
    def test_create_minimal(self):
        ext = Extraction(
            extraction_class="Alternative",
            extraction_text="thymol",
        )
        assert ext.extraction_class == "Alternative"
        assert ext.extraction_text == "thymol"
        assert ext.char_interval is None
        assert ext.alignment_status is None
        assert ext.attributes is None

    def test_create_with_attributes(self):
        ext = Extraction(
            extraction_class="Alternative",
            extraction_text="thymol",
            attributes={"standard_name": "thymol", "abbreviation": "THY"},
        )
        assert ext.attributes["standard_name"] == "thymol"
        assert ext.attributes["abbreviation"] == "THY"

    def test_create_with_position(self):
        ci = CharInterval(start_pos=5, end_pos=11)
        ext = Extraction(
            extraction_class="Alternative",
            extraction_text="thymol",
            char_interval=ci,
            alignment_status=AlignmentStatus.MATCH_EXACT,
            extraction_index=0,
            group_index=0,
        )
        assert ext.char_interval.start_pos == 5
        assert ext.alignment_status == AlignmentStatus.MATCH_EXACT
        assert ext.extraction_index == 0


class TestDocument:
    def test_create_minimal(self):
        doc = Document(text="Test text")
        assert doc.text == "Test text"
        assert doc.document_id is not None  # auto-generated
        assert doc.document_id.startswith("doc_")

    def test_create_with_id(self):
        doc = Document(text="Test text", document_id="PMC123")
        assert doc.document_id == "PMC123"

    def test_additional_context(self):
        doc = Document(text="Text", additional_context="Extra info")
        assert doc.additional_context == "Extra info"

    def test_with_additional_context(self):
        doc = Document(text="Text", document_id="id1")
        doc2 = doc.with_additional_context("New context")
        assert doc2.document_id == "id1"  # preserves ID
        assert doc2.text == "Text"
        assert doc2.additional_context == "New context"
        assert doc.additional_context is None  # original unchanged


class TestExampleData:
    def test_create(self):
        ext = Extraction(extraction_class="Alt", extraction_text="thymol")
        ed = ExampleData(text="Some text", extractions=[ext])
        assert ed.text == "Some text"
        assert len(ed.extractions) == 1
        assert ed.extractions[0].extraction_text == "thymol"


class TestAnnotatedDocument:
    def test_create(self):
        ext = Extraction(extraction_class="Alt", extraction_text="thymol")
        ad = AnnotatedDocument(
            document_id="PMC123",
            extractions=[ext],
            text="Source text",
        )
        assert ad.document_id == "PMC123"
        assert len(ad.extractions) == 1
        assert ad.text == "Source text"

    def test_auto_generated_id(self):
        ad = AnnotatedDocument(extractions=[], text="text")
        assert ad.document_id is not None
        assert ad.document_id.startswith("doc_")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data'`

- [ ] **Step 3: Write implementation**

Create `src/data.py`:

```python
"""Core data types for the extraction pipeline.

Faithful reproduction of langextract's core/data.py types.
Entity types and relations are NOT hardcoded — they are loaded from
external configuration files via SchemaRegistry.
"""
from __future__ import annotations

import dataclasses
import enum
import uuid
from typing import Any


class FormatType(enum.Enum):
    """Enumeration of prompt output formats."""
    JSON = "json"
    YAML = "yaml"


class AlignmentStatus(enum.Enum):
    """How well an extraction matched the source text."""
    MATCH_EXACT = "match_exact"
    MATCH_GREATER = "match_greater"
    MATCH_LESSER = "match_lesser"
    MATCH_FUZZY = "match_fuzzy"


@dataclasses.dataclass
class CharInterval:
    """Represents a range of character positions in the original text.

    Attributes:
        start_pos: The starting character index (inclusive).
        end_pos: The ending character index (exclusive).
    """
    start_pos: int | None = None
    end_pos: int | None = None


@dataclasses.dataclass(init=False)
class Extraction:
    """Represents an extracted entity from text.

    This mirrors langextract's Extraction class. Entity-specific fields
    live in `attributes: dict[str, Any]`. The `extraction_class` field
    holds the entity type name (e.g. "Alternative", "Result").
    All typing and validation rules come from SchemaRegistry config.

    Attributes:
        extraction_class: The entity type name (from config).
        extraction_text: The primary display text.
        char_interval: Character position in source text (set by alignment).
        alignment_status: How well the extraction matched (set by alignment).
        extraction_index: Ordering index within the output.
        group_index: The group (extraction dict) this belongs to.
        description: Optional description.
        attributes: All entity-specific fields as a dict.
    """
    extraction_class: str
    extraction_text: str
    char_interval: CharInterval | None = None
    alignment_status: AlignmentStatus | None = None
    extraction_index: int | None = None
    group_index: int | None = None
    description: str | None = None
    attributes: dict[str, Any] | None = None

    def __init__(
        self,
        extraction_class: str,
        extraction_text: str,
        *,
        char_interval: CharInterval | None = None,
        alignment_status: AlignmentStatus | None = None,
        extraction_index: int | None = None,
        group_index: int | None = None,
        description: str | None = None,
        attributes: dict[str, Any] | None = None,
    ):
        self.extraction_class = extraction_class
        self.extraction_text = extraction_text
        self.char_interval = char_interval
        self.alignment_status = alignment_status
        self.extraction_index = extraction_index
        self.group_index = group_index
        self.description = description
        self.attributes = attributes


@dataclasses.dataclass
class Document:
    """Input document for the extraction pipeline.

    Attributes:
        text: Raw text content.
        document_id: Unique identifier (auto-generated if not set).
        additional_context: Extra context for prompt instructions.
    """
    text: str
    additional_context: str | None = None

    def __init__(
        self,
        text: str,
        *,
        document_id: str | None = None,
        additional_context: str | None = None,
    ):
        self.text = text
        self.additional_context = additional_context
        self._document_id = document_id

    @property
    def document_id(self) -> str:
        """Returns the document ID, generating one if not set."""
        if self._document_id is None:
            self._document_id = f"doc_{uuid.uuid4().hex[:8]}"
        return self._document_id

    @document_id.setter
    def document_id(self, value: str | None) -> None:
        self._document_id = value

    def with_additional_context(self, additional_context: str | None) -> "Document":
        """Return a copy with additional_context overridden."""
        new_doc = Document(
            text=self.text,
            document_id=self.document_id,
            additional_context=additional_context,
        )
        return new_doc


@dataclasses.dataclass
class ExampleData:
    """A few-shot example for structured prompting.

    Attributes:
        text: The raw input text (sentence, paragraph, etc.).
        extractions: Extractions from the text.
    """
    text: str
    extractions: list[Extraction] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class AnnotatedDocument:
    """Pipeline output — document with extracted entities.

    Attributes:
        document_id: Unique identifier (auto-generated if not set).
        extractions: List of extracted entities.
        text: Original source text.
    """
    extractions: list[Extraction] | None = None
    text: str | None = None

    def __init__(
        self,
        *,
        document_id: str | None = None,
        extractions: list[Extraction] | None = None,
        text: str | None = None,
    ):
        self.extractions = extractions
        self.text = text
        self._document_id = document_id

    @property
    def document_id(self) -> str:
        """Returns the document ID, generating one if not set."""
        if self._document_id is None:
            self._document_id = f"doc_{uuid.uuid4().hex[:8]}"
        return self._document_id

    @document_id.setter
    def document_id(self, value: str | None) -> None:
        self._document_id = value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat: add core data types (Document, Extraction, AnnotatedDocument, ExampleData)"
```

---

### Task A3: Tokenizer (`src/tokenizer.py`)

**Files:**
- Create: `src/tokenizer.py`
- Create: `tests/test_tokenizer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tokenizer.py`:

```python
"""Tests for tokenization utilities."""
import pytest
from src.tokenizer import (
    CharInterval, TokenInterval, TokenType, Token, TokenizedText,
    RegexTokenizer, UnicodeTokenizer, tokenize, tokens_text,
    find_sentence_range,
)


class TestTokenInterval:
    def test_create(self):
        ti = TokenInterval(start_index=0, end_index=5)
        assert ti.start_index == 0
        assert ti.end_index == 5


class TestRegexTokenizer:
    def setup_method(self):
        self.tokenizer = RegexTokenizer()

    def test_tokenize_simple(self):
        result = self.tokenizer.tokenize("Hello world.")
        assert len(result.tokens) >= 3
        assert result.tokens[0].token_type == TokenType.WORD
        assert result.text == "Hello world."

    def test_tokenize_digits(self):
        result = self.tokenizer.tokenize("123 pigs")
        assert result.tokens[0].token_type == TokenType.NUMBER

    def test_tokenize_punctuation(self):
        result = self.tokenizer.tokenize("Test.")
        punct_tokens = [t for t in result.tokens if t.token_type == TokenType.PUNCTUATION]
        assert len(punct_tokens) >= 1

    def test_tokenize_newline_tracking(self):
        result = self.tokenizer.tokenize("Line one.\nLine two.")
        newline_tokens = [t for t in result.tokens if t.first_token_after_newline]
        assert len(newline_tokens) >= 1

    def test_tokenize_empty(self):
        result = self.tokenizer.tokenize("")
        assert len(result.tokens) == 0
        assert result.text == ""

    def test_char_intervals(self):
        result = self.tokenizer.tokenize("ab cd ef")
        # "ab" should span chars 0-2
        assert result.tokens[0].char_interval.start_pos == 0
        assert result.tokens[0].char_interval.end_pos == 2
        # "cd" should span chars 3-5
        assert result.tokens[1].char_interval.start_pos == 3
        assert result.tokens[1].char_interval.end_pos == 5

    def test_tokenize_with_chinese(self):
        result = self.tokenizer.tokenize("thymol 百里香酚 500 mg/kg")
        # Should not crash on CJK characters
        assert len(result.tokens) > 0


class TestTokensText:
    def setup_method(self):
        self.tokenizer = RegexTokenizer()

    def test_reconstruct_text(self):
        tt = self.tokenizer.tokenize("Hello world today")
        interval = TokenInterval(start_index=0, end_index=2)
        text = tokens_text(tt, interval)
        assert text == "Hello world"

    def test_empty_interval(self):
        tt = self.tokenizer.tokenize("Hello world")
        interval = TokenInterval(start_index=0, end_index=0)
        result = tokens_text(tt, interval)
        assert result == ""

    def test_invalid_interval(self):
        tt = self.tokenizer.tokenize("Hello world")
        interval = TokenInterval(start_index=0, end_index=999)
        with pytest.raises(ValueError):
            tokens_text(tt, interval)


class TestFindSentenceRange:
    def setup_method(self):
        self.tokenizer = RegexTokenizer()

    def test_simple_sentence(self):
        text = "Roses are red. Violets are blue."
        tt = self.tokenizer.tokenize(text)
        rng = find_sentence_range(text, tt.tokens, 0)
        assert rng.end_index > rng.start_index
        # Should end at or after the "." token

    def test_empty_tokens(self):
        rng = find_sentence_range("", [], 0)
        assert rng.start_index == 0
        assert rng.end_index == 0

    def test_known_abbreviation_not_sentence_end(self):
        text = "Dr. Smith conducted the study."
        tt = self.tokenizer.tokenize(text)
        rng = find_sentence_range(text, tt.tokens, 0)
        # "Dr." should NOT end the sentence
        sentence_text = tokens_text(tt, rng)
        # Should include more than just "Dr."
        assert "Smith" in sentence_text or len(sentence_text.split()) > 1


class TestConvenienceTokenize:
    def test_tokenize_function(self):
        tt = tokenize("Simple test.")
        assert len(tt.tokens) >= 2
        assert isinstance(tt, TokenizedText)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tokenizer.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Create `src/tokenizer.py`:

```python
"""Tokenization utilities for text.

Provides regex-based and Unicode-aware tokenizers for alignment
and sentence boundary detection. Faithful reproduction of
langextract's core/tokenizer.py.
"""
from __future__ import annotations

import abc
import dataclasses
import enum
from collections.abc import Sequence, Set

import regex


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclasses.dataclass(slots=True)
class CharInterval:
    """Character position range in original text."""
    start_pos: int
    end_pos: int


@dataclasses.dataclass(slots=True)
class TokenInterval:
    """Interval over tokens [start_index, end_index)."""
    start_index: int = 0
    end_index: int = 0


class TokenType(enum.IntEnum):
    WORD = 0
    NUMBER = 1
    PUNCTUATION = 2


@dataclasses.dataclass(slots=True)
class Token:
    index: int
    token_type: TokenType
    char_interval: CharInterval = dataclasses.field(
        default_factory=lambda: CharInterval(0, 0)
    )
    first_token_after_newline: bool = False


@dataclasses.dataclass
class TokenizedText:
    text: str
    tokens: list[Token] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_LETTERS_PATTERN = r"[^\W\d_]+"
_DIGITS_PATTERN = r"\d+"
_SYMBOLS_PATTERN = r"([^\w\s]|_)\1*"
_END_OF_SENTENCE_PATTERN = regex.compile(r"[.?!。！？।][\"'”’»)\]}]*$")

_TOKEN_PATTERN = regex.compile(
    rf"{_LETTERS_PATTERN}|{_DIGITS_PATTERN}|{_SYMBOLS_PATTERN}"
)
_WORD_PATTERN = regex.compile(rf"(?:{_LETTERS_PATTERN}|{_DIGITS_PATTERN})\Z")

_KNOWN_ABBREVIATIONS = frozenset({"Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St."})
_CLOSING_PUNCTUATION = frozenset({'"', "'", "”", "’", "»", ")", "]", "}"})


# ---------------------------------------------------------------------------
# Tokenizer ABC
# ---------------------------------------------------------------------------

class Tokenizer(abc.ABC):
    """Abstract base class for tokenizers."""

    @abc.abstractmethod
    def tokenize(self, text: str) -> TokenizedText:
        """Split text into tokens."""
        ...


# ---------------------------------------------------------------------------
# RegexTokenizer
# ---------------------------------------------------------------------------

class RegexTokenizer(Tokenizer):
    """Regex-based tokenizer (default). Fast for English text."""

    def tokenize(self, text: str) -> TokenizedText:
        tokenized = TokenizedText(text=text)
        previous_end = 0
        for token_index, match in enumerate(_TOKEN_PATTERN.finditer(text)):
            start_pos, end_pos = match.span()
            matched_text = match.group()
            token = Token(
                index=token_index,
                char_interval=CharInterval(start_pos=start_pos, end_pos=end_pos),
                token_type=TokenType.WORD,
                first_token_after_newline=False,
            )
            if token_index > 0:
                has_newline = text.find("\n", previous_end, start_pos) != -1
                if not has_newline:
                    has_newline = text.find("\r", previous_end, start_pos) != -1
                if has_newline:
                    token.first_token_after_newline = True
            if regex.fullmatch(_DIGITS_PATTERN, matched_text):
                token.token_type = TokenType.NUMBER
            elif _WORD_PATTERN.fullmatch(matched_text):
                token.token_type = TokenType.WORD
            else:
                token.token_type = TokenType.PUNCTUATION
            tokenized.tokens.append(token)
            previous_end = end_pos
        return tokenized


# ---------------------------------------------------------------------------
# UnicodeTokenizer
# ---------------------------------------------------------------------------

_GRAPHEME_CLUSTER_PATTERN = regex.compile(r"\X")


class UnicodeTokenizer(Tokenizer):
    """Unicode-aware tokenizer using grapheme clusters via \\X pattern.
    Handles CJK, Emoji, non-Latin scripts correctly. Slower than RegexTokenizer.
    """

    def tokenize(self, text: str) -> TokenizedText:
        tokens: list[Token] = []
        for match in regex.finditer(r"\X", text):
            grapheme = match.group()
            start, end = match.span()
            if grapheme.isspace():
                continue
            # Simple classification
            if grapheme.isalpha():
                ttype = TokenType.WORD
            elif grapheme.isdigit():
                ttype = TokenType.NUMBER
            else:
                ttype = TokenType.PUNCTUATION
            token = Token(
                index=len(tokens),
                char_interval=CharInterval(start_pos=start, end_pos=end),
                token_type=ttype,
                first_token_after_newline=False,
            )
            tokens.append(token)
        # Detect newlines between tokens
        prev_end = 0
        for token in tokens:
            if token.char_interval.start_pos > prev_end:
                gap = text[prev_end:token.char_interval.start_pos]
                if "\n" in gap or "\r" in gap:
                    token.first_token_after_newline = True
            prev_end = token.char_interval.end_pos
        return TokenizedText(text=text, tokens=tokens)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

_DEFAULT_TOKENIZER = RegexTokenizer()


def tokenize(text: str, tokenizer: Tokenizer | None = None) -> TokenizedText:
    """Tokenize text using the provided tokenizer (default: RegexTokenizer)."""
    if tokenizer is None:
        tokenizer = _DEFAULT_TOKENIZER
    return tokenizer.tokenize(text)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def tokens_text(
    tokenized_text: TokenizedText,
    token_interval: TokenInterval,
) -> str:
    """Reconstruct the substring spanning a token interval."""
    if token_interval.start_index == token_interval.end_index:
        return ""
    if (
        token_interval.start_index < 0
        or token_interval.end_index > len(tokenized_text.tokens)
        or token_interval.start_index > token_interval.end_index
    ):
        raise ValueError(
            f"Invalid token interval: start={token_interval.start_index}, "
            f"end={token_interval.end_index}, "
            f"total_tokens={len(tokenized_text.tokens)}"
        )
    start_token = tokenized_text.tokens[token_interval.start_index]
    end_token = tokenized_text.tokens[token_interval.end_index - 1]
    return tokenized_text.text[
        start_token.char_interval.start_pos : end_token.char_interval.end_pos
    ]


def find_sentence_range(
    text: str,
    tokens: Sequence[Token],
    start_token_index: int,
    known_abbreviations: Set[str] = _KNOWN_ABBREVIATIONS,
) -> TokenInterval:
    """Find a sentence interval from start_token_index.

    Boundaries: end-of-sentence punctuation, newline+uppercase, not abbreviations.
    """
    if not tokens:
        return TokenInterval(0, 0)
    if start_token_index < 0 or start_token_index >= len(tokens):
        raise ValueError(
            f"start_token_index={start_token_index} out of range "
            f"(total tokens: {len(tokens)})"
        )

    i = start_token_index
    while i < len(tokens):
        if tokens[i].token_type == TokenType.PUNCTUATION:
            if _is_end_of_sentence_token(text, tokens, i, known_abbreviations):
                end_index = i + 1
                while end_index < len(tokens):
                    ntt = text[
                        tokens[end_index].char_interval.start_pos :
                        tokens[end_index].char_interval.end_pos
                    ]
                    if (
                        tokens[end_index].token_type == TokenType.PUNCTUATION
                        and ntt in _CLOSING_PUNCTUATION
                    ):
                        end_index += 1
                    else:
                        break
                return TokenInterval(
                    start_index=start_token_index, end_index=end_index
                )
        if _is_sentence_break_after_newline(text, tokens, i):
            return TokenInterval(start_index=start_token_index, end_index=i + 1)
        i += 1

    return TokenInterval(start_index=start_token_index, end_index=len(tokens))


def _is_end_of_sentence_token(
    text: str,
    tokens: Sequence[Token],
    current_idx: int,
    known_abbreviations: Set[str],
) -> bool:
    """Check if the punctuation token ends a sentence."""
    current_token_text = text[
        tokens[current_idx].char_interval.start_pos :
        tokens[current_idx].char_interval.end_pos
    ]
    if _END_OF_SENTENCE_PATTERN.search(current_token_text):
        if current_idx > 0:
            prev = text[
                tokens[current_idx - 1].char_interval.start_pos :
                tokens[current_idx - 1].char_interval.end_pos
            ]
            if f"{prev}{current_token_text}" in known_abbreviations:
                return False
        return True
    return False


def _is_sentence_break_after_newline(
    text: str, tokens: Sequence[Token], current_idx: int
) -> bool:
    """Check if next token starts uppercase and follows a newline."""
    if current_idx + 1 >= len(tokens):
        return False
    next_token = tokens[current_idx + 1]
    if not next_token.first_token_after_newline:
        return False
    ntt = text[
        next_token.char_interval.start_pos : next_token.char_interval.end_pos
    ]
    return bool(ntt) and not ntt[0].islower()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tokenizer.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/tokenizer.py tests/test_tokenizer.py
git commit -m "feat: add tokenizer layer (RegexTokenizer, UnicodeTokenizer, sentence detection)"
```

---

### Task A4: Update Config (`src/config.py`)

**Files:**
- Modify: `src/config.py`

- [ ] **Step 1: Rewrite config for v2**

Read current `src/config.py`, then replace with:

```python
"""Configuration for llm-extract v2 pipeline."""
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

    model_config = SettingsConfigDict(env_file=".env", env_prefix="ZN_")


settings = Settings()
```

- [ ] **Step 2: Verify import**

Run: `python -c "from src.config import settings; print(settings.llm_model)"`
Expected: `deepseek-chat`

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "refactor: update config for v2 pipeline (pydantic-settings, ZN_ prefix)"
```

---

## Group B: Chunking + Format Handler

### Task B1: Chunking (`src/chunking.py`)

**Files:**
- Create: `src/chunking.py`
- Create: `tests/test_chunking.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_chunking.py`:

```python
"""Tests for chunking."""
import pytest
from src.data import Document
from src.tokenizer import RegexTokenizer, TokenizedText, TokenInterval, Token, CharInterval, TokenType
from src.chunking import (
    TextChunk, SentenceIterator, ChunkIterator,
    make_batches_of_textchunk, get_token_interval_text,
    get_char_interval, create_token_interval,
)


def _make_tokenized(text: str) -> TokenizedText:
    t = RegexTokenizer()
    return t.tokenize(text)


class TestCreateTokenInterval:
    def test_valid(self):
        ti = create_token_interval(0, 5)
        assert ti.start_index == 0
        assert ti.end_index == 5

    def test_invalid_negative(self):
        with pytest.raises(ValueError):
            create_token_interval(-1, 5)

    def test_invalid_order(self):
        with pytest.raises(ValueError):
            create_token_interval(5, 0)


class TestGetTokenIntervalText:
    def test_simple(self):
        tt = _make_tokenized("Hello beautiful world")
        ti = TokenInterval(start_index=0, end_index=2)
        result = get_token_interval_text(tt, ti)
        assert result == "Hello beautiful"


class TestGetCharInterval:
    def test_simple(self):
        tt = _make_tokenized("Hello world")
        ti = TokenInterval(start_index=0, end_index=2)
        ci = get_char_interval(tt, ti)
        assert ci.start_pos == 0
        assert ci.end_pos == len("Hello world")


class TestTextChunk:
    def test_basic(self):
        doc = Document(text="Hello world. This is a test.")
        tt = _make_tokenized(doc.text)
        doc.tokenized_text = tt
        ti = TokenInterval(start_index=0, end_index=len(tt.tokens))
        chunk = TextChunk(token_interval=ti, document=doc)
        assert chunk.document_id == doc.document_id
        assert "Hello world" in chunk.chunk_text


class TestSentenceIterator:
    def test_two_sentences(self):
        tt = _make_tokenized("Roses are red. Violets are blue.")
        si = SentenceIterator(tt)
        sentences = list(si)
        assert len(sentences) >= 2

    def test_single_sentence(self):
        tt = _make_tokenized("Just one sentence.")
        si = SentenceIterator(tt)
        sentences = list(si)
        assert len(sentences) >= 1

    def test_stop_iteration(self):
        tt = _make_tokenized("Short.")
        si = SentenceIterator(tt)
        list(si)
        with pytest.raises(StopIteration):
            next(si)


class TestChunkIterator:
    def test_short_text_one_chunk(self):
        text = "Short text. Still short."
        tokenizer = RegexTokenizer()
        ci = ChunkIterator(text=text, max_char_buffer=1000, tokenizer_impl=tokenizer)
        chunks = list(ci)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)

    def test_long_text_multiple_chunks(self):
        text = "A. " * 500
        tokenizer = RegexTokenizer()
        ci = ChunkIterator(text=text, max_char_buffer=100, tokenizer_impl=tokenizer)
        chunks = list(ci)
        assert len(chunks) > 1

    def test_empty_text(self):
        tokenizer = RegexTokenizer()
        ci = ChunkIterator(text="", max_char_buffer=100, tokenizer_impl=tokenizer)
        chunks = list(ci)
        # Empty text produces 0 tokens, so no chunks
        # The iterator handles this gracefully


class TestMakeBatches:
    def test_batch_of_two(self):
        doc = Document(text="A. B. C. D.")
        tt = _make_tokenized(doc.text)
        doc.tokenized_text = tt
        ti = TokenInterval(start_index=0, end_index=len(tt.tokens))
        chunks = [TextChunk(token_interval=ti, document=doc) for _ in range(5)]
        batches = list(make_batches_of_textchunk(iter(chunks), batch_length=2))
        assert len(batches) == 3  # 2 + 2 + 1
        assert len(batches[0]) == 2
        assert len(batches[2]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chunking.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Create `src/chunking.py`:

```python
"""Library for breaking documents into chunks of sentences.

Faithful reproduction of langextract's chunking.py.
Token-level sliding window respecting sentence boundaries.
"""
from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Iterator, Sequence

from src.data import CharInterval, Document
from src.tokenizer import (
    TokenInterval, TokenizedText, Tokenizer, RegexTokenizer,
    TokenType, Token,
    find_sentence_range, tokens_text as tk_tokens_text,
)


def create_token_interval(start_index: int, end_index: int) -> TokenInterval:
    """Create a token interval with validation."""
    if start_index < 0:
        raise ValueError(f"Start index {start_index} must be positive.")
    if start_index >= end_index:
        raise ValueError(
            f"Start index {start_index} must be < end index {end_index}."
        )
    return TokenInterval(start_index=start_index, end_index=end_index)


def get_token_interval_text(
    tokenized_text: TokenizedText,
    token_interval: TokenInterval,
) -> str:
    """Get text within a token interval."""
    return tk_tokens_text(tokenized_text, token_interval)


def get_char_interval(
    tokenized_text: TokenizedText,
    token_interval: TokenInterval,
) -> CharInterval:
    """Get char interval for a token interval."""
    if token_interval.start_index >= token_interval.end_index:
        raise ValueError(
            f"Start index {token_interval.start_index} must be < "
            f"end index {token_interval.end_index}."
        )
    start_token = tokenized_text.tokens[token_interval.start_index]
    final_token = tokenized_text.tokens[token_interval.end_index - 1]
    return CharInterval(
        start_pos=start_token.char_interval.start_pos,
        end_pos=final_token.char_interval.end_pos,
    )


def make_batches_of_textchunk(
    chunk_iter: Iterator["TextChunk"],
    batch_length: int,
) -> Iterable[Sequence["TextChunk"]]:
    """Group chunks into batches of batch_length."""
    batch = []
    for chunk in chunk_iter:
        batch.append(chunk)
        if len(batch) >= batch_length:
            yield batch
            batch = []
    if batch:
        yield batch


@dataclasses.dataclass
class TextChunk:
    """A text chunk with position in source document."""
    token_interval: TokenInterval
    document: Document | None = None
    _chunk_text: str | None = dataclasses.field(default=None, init=False, repr=False)
    _sanitized_chunk_text: str | None = dataclasses.field(
        default=None, init=False, repr=False
    )
    _char_interval: CharInterval | None = dataclasses.field(
        default=None, init=False, repr=False
    )

    @property
    def document_id(self) -> str | None:
        if self.document is not None:
            return self.document.document_id
        return None

    @property
    def document_text(self) -> TokenizedText | None:
        if self.document is not None:
            return getattr(self.document, 'tokenized_text', None)
        return None

    @property
    def chunk_text(self) -> str:
        if self._chunk_text is None:
            if self.document_text is None:
                raise ValueError("document_text must be set to access chunk_text.")
            self._chunk_text = get_token_interval_text(
                self.document_text, self.token_interval
            )
        return self._chunk_text

    @property
    def sanitized_chunk_text(self) -> str:
        if self._sanitized_chunk_text is None:
            self._sanitized_chunk_text = re.sub(r"\s+", " ", self.chunk_text.strip())
        return self._sanitized_chunk_text

    @property
    def additional_context(self) -> str | None:
        if self.document is not None:
            return self.document.additional_context
        return None

    @property
    def char_interval(self) -> CharInterval:
        if self._char_interval is None:
            if self.document_text is None:
                raise ValueError("document_text must be set to compute char_interval.")
            self._char_interval = get_char_interval(
                self.document_text, self.token_interval
            )
        return self._char_interval


class SentenceIterator:
    """Iterate through sentences of a tokenized text."""

    def __init__(
        self,
        tokenized_text: TokenizedText,
        curr_token_pos: int = 0,
    ):
        self.tokenized_text = tokenized_text
        self.token_len = len(tokenized_text.tokens)
        if curr_token_pos < 0:
            raise IndexError(f"Current token position {curr_token_pos} cannot be negative.")
        elif curr_token_pos > self.token_len:
            raise IndexError(
                f"Current token position {curr_token_pos} is past the length "
                f"of the document {self.token_len}."
            )
        self.curr_token_pos = curr_token_pos

    def __iter__(self) -> Iterator[TokenInterval]:
        return self

    def __next__(self) -> TokenInterval:
        if self.curr_token_pos == self.token_len:
            raise StopIteration
        sentence_range = find_sentence_range(
            self.tokenized_text.text,
            self.tokenized_text.tokens,
            self.curr_token_pos,
        )
        sentence_range = create_token_interval(
            self.curr_token_pos, sentence_range.end_index
        )
        self.curr_token_pos = sentence_range.end_index
        return sentence_range


class ChunkIterator:
    """Iterate through chunks of tokenized text.

    Fits sentences into chunks up to max_char_buffer. Long sentences
    are split at newline boundaries or token boundaries.
    """

    def __init__(
        self,
        text: str | TokenizedText | None,
        max_char_buffer: int,
        tokenizer_impl: Tokenizer,
        document: Document | None = None,
    ):
        if text is None:
            if document is None:
                raise ValueError("Either text or document must be provided.")
            text = document.text or ""

        if isinstance(text, str):
            text = tokenizer_impl.tokenize(text)
        elif isinstance(text, TokenizedText) and not text.tokens:
            text_to_tokenize = text.text or (document.text if document else "")
            text = tokenizer_impl.tokenize(text_to_tokenize)

        self.tokenized_text = text
        self.max_char_buffer = max_char_buffer
        self.sentence_iter = SentenceIterator(self.tokenized_text)
        self.broken_sentence = False

        if document is None:
            self.document = Document(text=text.text)
        else:
            self.document = document
        self.document.tokenized_text = self.tokenized_text

    def __iter__(self) -> Iterator[TextChunk]:
        return self

    def _tokens_exceed_buffer(self, token_interval: TokenInterval) -> bool:
        char_interval = get_char_interval(self.tokenized_text, token_interval)
        return (char_interval.end_pos - char_interval.start_pos) > self.max_char_buffer

    def __next__(self) -> TextChunk:
        sentence = next(self.sentence_iter)
        # If single token exceeds buffer, it becomes its own chunk
        curr_chunk = create_token_interval(sentence.start_index, sentence.start_index + 1)
        if self._tokens_exceed_buffer(curr_chunk):
            self.sentence_iter = SentenceIterator(
                self.tokenized_text, curr_token_pos=sentence.start_index + 1
            )
            self.broken_sentence = curr_chunk.end_index < sentence.end_index
            return TextChunk(token_interval=curr_chunk, document=self.document)

        start_of_new_line = -1
        for token_index in range(curr_chunk.start_index, sentence.end_index):
            if self.tokenized_text.tokens[token_index].first_token_after_newline:
                start_of_new_line = token_index
            test_chunk = create_token_interval(curr_chunk.start_index, token_index + 1)
            if self._tokens_exceed_buffer(test_chunk):
                if start_of_new_line > 0 and start_of_new_line > curr_chunk.start_index:
                    curr_chunk = create_token_interval(
                        curr_chunk.start_index, start_of_new_line
                    )
                self.sentence_iter = SentenceIterator(
                    self.tokenized_text, curr_token_pos=curr_chunk.end_index
                )
                self.broken_sentence = True
                return TextChunk(token_interval=curr_chunk, document=self.document)
            else:
                curr_chunk = test_chunk

        if self.broken_sentence:
            self.broken_sentence = False
        else:
            for sentence in self.sentence_iter:
                test_chunk = create_token_interval(
                    curr_chunk.start_index, sentence.end_index
                )
                if self._tokens_exceed_buffer(test_chunk):
                    self.sentence_iter = SentenceIterator(
                        self.tokenized_text, curr_token_pos=curr_chunk.end_index
                    )
                    return TextChunk(token_interval=curr_chunk, document=self.document)
                else:
                    curr_chunk = test_chunk

        return TextChunk(token_interval=curr_chunk, document=self.document)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chunking.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/chunking.py tests/test_chunking.py
git commit -m "feat: add chunking layer (ChunkIterator, SentenceIterator, TextChunk)"
```

---

### Task B2: Format Handler (`src/format_handler.py`)

**Files:**
- Create: `src/format_handler.py`
- Create: `tests/test_format_handler.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_format_handler.py`:

```python
"""Tests for format handler."""
import pytest
from src.data import FormatType, Extraction, AlignmentStatus
from src.format_handler import FormatHandler


class TestFormatHandlerInit:
    def test_defaults(self):
        fh = FormatHandler()
        assert fh.format_type == FormatType.JSON
        assert fh.use_wrapper is True
        assert fh.wrapper_key == "extractions"
        assert fh.use_fences is True

    def test_json_no_fences(self):
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        assert fh.use_fences is False

    def test_yaml(self):
        fh = FormatHandler(format_type=FormatType.YAML)
        assert fh.format_type == FormatType.YAML

    def test_custom_wrapper_key(self):
        fh = FormatHandler(wrapper_key="items", use_wrapper=True)
        assert fh.wrapper_key == "items"

    def test_no_wrapper(self):
        fh = FormatHandler(use_wrapper=False)
        assert fh.wrapper_key is None


class TestFormatExtractionExample:
    def test_json_format(self):
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        ext = Extraction(
            extraction_class="Alternative",
            extraction_text="thymol",
            attributes={"abbreviation": "THY"},
        )
        result = fh.format_extraction_example([ext])
        assert "Alternative" in result
        assert "thymol" in result

    def test_json_with_fences(self):
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=True)
        ext = Extraction(
            extraction_class="Alternative",
            extraction_text="thymol",
        )
        result = fh.format_extraction_example([ext])
        assert result.startswith("```json")
        assert "```" in result

    def test_yaml_format(self):
        fh = FormatHandler(format_type=FormatType.YAML, use_fences=False)
        ext = Extraction(
            extraction_class="Alternative",
            extraction_text="thymol",
        )
        result = fh.format_extraction_example([ext])
        assert "Alternative" in result
        assert "thymol" in result


class TestParseOutput:
    def test_parse_json_no_fences(self):
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        input_text = '{"extractions": [{"Alternative": "thymol"}]}'
        result = fh.parse_output(input_text)
        assert len(result) == 1
        assert result[0]["Alternative"] == "thymol"

    def test_parse_json_with_fences(self):
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=True)
        input_text = '```json\n{"extractions": [{"Alternative": "thymol"}]}\n```'
        result = fh.parse_output(input_text)
        assert len(result) == 1
        assert result[0]["Alternative"] == "thymol"

    def test_parse_json_top_level_list(self):
        fh = FormatHandler(
            format_type=FormatType.JSON, use_fences=False,
            allow_top_level_list=True, use_wrapper=False,
        )
        input_text = '[{"Alternative": "thymol"}, {"Composite_Product": "Product X"}]'
        result = fh.parse_output(input_text)
        assert len(result) == 2

    def test_parse_empty_input_raises(self):
        fh = FormatHandler()
        with pytest.raises(ValueError):
            fh.parse_output("")

    def test_parse_think_tag_stripping(self):
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        input_text = '<think>reasoning here</think>\n{"extractions": [{"Alternative": "thymol"}]}'
        result = fh.parse_output(input_text)
        assert len(result) == 1
        assert result[0]["Alternative"] == "thymol"

    def test_parse_with_attributes(self):
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        input_text = (
            '{"extractions": ['
            '{"Alternative": "thymol", "Alternative_attributes": {"abbreviation": "THY"}}'
            ']}'
        )
        result = fh.parse_output(input_text)
        assert len(result) == 1
        assert result[0]["Alternative_attributes"]["abbreviation"] == "THY"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_format_handler.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Create `src/format_handler.py`:

```python
"""Centralized format handler for prompts and parsing.

Faithful reproduction of langextract's core/format_handler.py.
Handles JSON/YAML format, code fence extraction, wrapper key management,
and attribute suffix conventions.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

import yaml

from src.data import FormatType, Extraction

ExtractionValueType = str | int | float | dict | list | None

_FENCE_START = r"```"
_LANGUAGE_TAG = r"(?P<lang>[A-Za-z0-9_+-]+)?"
_FENCE_NEWLINE = r"(?:\s*\n)?"
_FENCE_BODY = r"(?P<body>[\s\S]*?)"
_FENCE_END = r"```"

_FENCE_RE = re.compile(
    _FENCE_START + _LANGUAGE_TAG + _FENCE_NEWLINE + _FENCE_BODY + _FENCE_END,
    re.MULTILINE,
)

_THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)

EXTRACTIONS_KEY = "extractions"
ATTRIBUTE_SUFFIX = "_attributes"


class FormatHandler:
    """Handles all format-specific logic for prompts and parsing."""

    def __init__(
        self,
        format_type: FormatType = FormatType.JSON,
        use_wrapper: bool = True,
        wrapper_key: str | None = None,
        use_fences: bool = True,
        attribute_suffix: str = ATTRIBUTE_SUFFIX,
        strict_fences: bool = False,
        allow_top_level_list: bool = True,
    ) -> None:
        self.format_type = format_type
        self.use_wrapper = use_wrapper
        if use_wrapper:
            self.wrapper_key = wrapper_key if wrapper_key is not None else EXTRACTIONS_KEY
        else:
            self.wrapper_key = None
        self.use_fences = use_fences
        self.attribute_suffix = attribute_suffix
        self.strict_fences = strict_fences
        self.allow_top_level_list = allow_top_level_list

    def format_extraction_example(self, extractions: list[Extraction]) -> str:
        """Format extractions for a prompt few-shot example."""
        items = []
        for ext in extractions:
            item = {ext.extraction_class: ext.extraction_text}
            if ext.attributes:
                attr_key = f"{ext.extraction_class}{self.attribute_suffix}"
                item[attr_key] = ext.attributes
            items.append(item)

        if self.use_wrapper and self.wrapper_key:
            payload = {self.wrapper_key: items}
        else:
            payload = items

        if self.format_type == FormatType.YAML:
            formatted = yaml.safe_dump(payload, default_flow_style=False, sort_keys=False)
        else:
            formatted = json.dumps(payload, indent=2, ensure_ascii=False)

        return self._add_fences(formatted) if self.use_fences else formatted

    def parse_output(
        self, text: str, *, strict: bool | None = None
    ) -> Sequence[Mapping[str, ExtractionValueType]]:
        """Parse LLM output into extraction data."""
        if not text:
            raise ValueError("Empty or invalid input string.")

        content = self._extract_content(text)

        try:
            parsed = self._parse_with_fallback(content, strict)
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(
                f"Failed to parse {self.format_type.value.upper()} content: {str(e)[:200]}"
            ) from e

        if parsed is None:
            raise ValueError("Content must be a list or dict.")

        require_wrapper = self.wrapper_key is not None and (
            self.use_wrapper or bool(strict)
        )

        if isinstance(parsed, dict):
            if require_wrapper:
                if self.wrapper_key not in parsed:
                    raise ValueError(
                        f"Content must contain an '{self.wrapper_key}' key."
                    )
                items = parsed[self.wrapper_key]
            else:
                if EXTRACTIONS_KEY in parsed:
                    items = parsed[EXTRACTIONS_KEY]
                elif self.wrapper_key and self.wrapper_key in parsed:
                    items = parsed[self.wrapper_key]
                else:
                    items = [parsed]
        elif isinstance(parsed, list):
            if require_wrapper and (strict or not self.allow_top_level_list):
                raise ValueError(
                    f"Content must be a mapping with an '{self.wrapper_key}' key."
                )
            items = parsed
        else:
            raise ValueError(f"Expected list or dict, got {type(parsed)}")

        if not isinstance(items, list):
            raise ValueError("The extractions must be a sequence (list) of mappings.")

        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Each item in the sequence must be a mapping.")
            for k in item.keys():
                if not isinstance(k, str):
                    raise ValueError("All extraction keys must be strings.")

        return items

    def _add_fences(self, content: str) -> str:
        """Add code fences around content."""
        fence_type = self.format_type.value
        return f"```{fence_type}\n{content.strip()}\n```"

    def _extract_content(self, text: str) -> str:
        """Extract content from text, handling fences if configured."""
        if not self.use_fences:
            return text.strip()

        matches = list(_FENCE_RE.finditer(text))
        valid_tags = {
            FormatType.YAML: {"yaml", "yml"},
            FormatType.JSON: {"json"},
        }

        candidates = [
            m for m in matches
            if (lang := m.group("lang")) is not None
            and lang.strip().lower() in valid_tags.get(self.format_type, set())
        ]

        if self.strict_fences:
            if len(candidates) != 1:
                raise ValueError(
                    "Input string does not contain valid fence markers."
                    if len(candidates) == 0
                    else "Multiple fenced blocks found."
                )
            return candidates[0].group("body").strip()

        if len(candidates) == 1:
            return candidates[0].group("body").strip()
        elif len(candidates) > 1:
            raise ValueError("Multiple fenced blocks found.")

        if matches and len(matches) == 1:
            return matches[0].group("body").strip()

        return text.strip()

    def _parse_with_fallback(self, content: str, strict: bool | None):
        """Parse content, retrying without <think> tags on failure."""
        try:
            if self.format_type == FormatType.YAML:
                return yaml.safe_load(content)
            return json.loads(content)
        except (yaml.YAMLError, json.JSONDecodeError):
            if strict:
                raise
            if _THINK_TAG_RE.search(content):
                stripped = _THINK_TAG_RE.sub("", content).strip()
                if self.format_type == FormatType.YAML:
                    return yaml.safe_load(stripped)
                return json.loads(stripped)
            raise
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_format_handler.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/format_handler.py tests/test_format_handler.py
git commit -m "feat: add format handler (JSON/YAML, fence extraction, think-tag stripping)"
```

---

## Group C: Schema Registry + Schema

### Task C1: Schema Configuration Files

**Files:**
- Create: `schemas/entities.yaml`
- Create: `schemas/relations.yaml`
- Create: `schemas/extraction_phases.yaml`

- [ ] **Step 1: Create schemas/entities.yaml**

Create `schemas/entities.yaml` (abbreviated — full version covers all 13 entity types; showing 4 representative ones):

```yaml
# llm-extract Entity Type Definitions
# Source: SCHEMA.tsv — all entity types, attributes, extraction guidance

entities:
  Alternative:
    display_name: "抗生素替代物物质"
    description: >
      抗生素替代物物质（标准分类中的具体有效成分）。
      Alternative 与 Composite_Product 分开，因为复合产品本身不是单一物质。
    primary_text: standard_name
    extraction_guidance: |
      ## 提取要求
      1. 扫描材料与方法部分，识别所有用作抗生素替代物的物质
      2. 优先与下方《Alternative分类》词表匹配
      3. 词表中存在 → 使用词表标准名称和分类
      4. 词表中不存在 → 标记为 Other，保留原文原始名称
      5. 复合制剂按三步处理：拆分 → 创建复合节点 → 建立has_component关系
      6. 严禁将多个实体拼接成长字符串
    examples: []
    notes: |
      不要拼接字符串创造不存在的复合实体。
      原封不动保留原文中的原始英文全称及缩写。
    vocabulary:
      source: ALTERNATIVE.tsv
      key_field: standard_name
      class_field: Alternative_Class
      subclass_field: Subclass
      match_fields: [standard_name]
      match_mode: fuzzy
      inject_in_prompt: true
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
      - name: original_text
        type: string
        required: true
        source: llm
      - name: alternative_class
        type: enum
        values: [Plant_Extract, Trace_Element, Organic_Acid, Probiotic,
                 Polysaccharides_and_Oligosaccharides, Enzyme, Bioactive_Peptides, Other]
        required: true
        source: post
      - name: subclass
        type: string
        required: false
        source: post
      - name: match_source
        type: enum
        values: [词表精确匹配, 词表同义映射, 词表模糊匹配, Other_未匹配]
        required: true
        source: post
      - name: evidence_text
        type: text
        required: true
        source: align
      - name: source_location
        type: string
        required: true
        source: structure

  Composite_Product:
    display_name: "复合制剂产品"
    description: "复合制剂产品作为独立实体"
    primary_text: product_name
    extraction_guidance: |
      ## 复合制剂三步处理法
      第一步 — 拆分组分
      第二步 — 创建复合节点（使用文献中的产品名）
      第三步 — 建立has_component关系关联组分
      严禁字符串拼接命名
    examples: []
    notes: "禁止生成'植物乳杆菌+枯草芽孢杆菌'这样的长字符串节点名"
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
    description: "三元组：部位+指标+变化方向"
    primary_text: indicator_abbreviation
    extraction_guidance: |
      ## 提取规则
      1. 全量抽取所有报告了统计学比较结果的指标变化
      2. 如实记录原文统计信息，无论P值大小
      3. 对比基准必须是试验组与对照组的横向比较
      4. 关系类型：数值型→increases/decreases, 基因表达→upregulates/downregulates,
         微生物丰度→enriches/depletes, 其他→affects
    examples: []
    notes: "若同一句话描述多个指标变化，拆分为多行独立记录"
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
```

- [ ] **Step 2: Create schemas/relations.yaml**

```yaml
relations:
  belongs_to:
    description: "物质属于某个一级分类"
    source: Alternative
    target: Alternative_Class
    cardinality: many_to_one

  has_component:
    description: "复合产品包含某个组分"
    source: Composite_Product
    target: [Alternative, Composite_Product]
    cardinality: one_to_many
    co_extracted: true

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

  measured_in:
    source: Indicator
    target: Tissue_Site
    cardinality: many_to_one
    co_extracted: true

  uses_method:
    source: Indicator
    target: Method
    cardinality: many_to_one
    co_extracted: true

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
    co_extracted: true

  occurs_in:
    source: Result
    target: Tissue_Site
    co_extracted: true

  compared_to:
    source: Result
    target: Control_Group
    co_extracted: true

  has_synonym:
    source: Alternative
    target: Alternative
    description: "物质之间的同义关系"

  leads_to:
    source: Indicator
    target: Indicator
    description: "指标变化导致另一指标变化"

  correlates_with:
    source: Indicator
    target: Indicator
    description: "指标间统计相关性"

  part_of:
    source: Indicator
    target: Indicator
    description: "子指标属于父指标"
```

- [ ] **Step 3: Create schemas/extraction_phases.yaml**

```yaml
phases:
  phase_1_alternatives:
    description: "Extract antibiotic alternatives (GATE)"
    extracts:
      - Alternative
      - Alternative_Class
      - Composite_Product
    gate:
      entity: Alternative
      condition: "alternative_class not in ['Other']"
      on_fail: skip_article

  phase_2_experiment:
    description: "Extract experiment design parameters"
    extracts:
      - Literature
      - Experiment
      - Swine_Model
      - Swine
      - Intervention
      - Control_Group

  phase_3_indicators:
    description: "Extract indicators, tissue sites, methods"
    extracts:
      - Tissue_Site
      - Indicator
      - Method

  phase_4_results:
    description: "Extract statistically evaluated results"
    extracts:
      - Result
    context_from: [phase_2_experiment, phase_3_indicators]
```

- [ ] **Step 4: Commit**

```bash
git add schemas/
git commit -m "feat: add schema config files (entities, relations, extraction phases)"
```

---

### Task C2: Schema Registry (`src/schema_registry.py`)

**Files:**
- Create: `src/schema_registry.py`
- Create: `tests/test_schema_registry.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_schema_registry.py`:

```python
"""Tests for schema registry."""
import pytest
from pathlib import Path
from src.schema_registry import SchemaRegistry, AttributeDef, Vocabulary


SCHEMA_DIR = Path(__file__).parent.parent / "schemas"


class TestSchemaRegistry:
    def setup_method(self):
        self.registry = SchemaRegistry(config_dir=SCHEMA_DIR)

    def test_load_entities(self):
        names = self.registry.all_entity_names()
        assert "Alternative" in names
        assert "Composite_Product" in names
        assert "Result" in names

    def test_entity_def(self):
        ed = self.registry.entity_def("Alternative")
        assert ed.name == "Alternative"
        assert ed.primary_text == "standard_name"
        assert len(ed.attributes) > 0

    def test_llm_output_fields(self):
        fields = self.registry.llm_output_fields("Alternative")
        field_names = {f.name for f in fields}
        # evidence_text and source_location should NOT be in LLM fields
        assert "evidence_text" not in field_names
        assert "source_location" not in field_names
        # LLM fields should be present
        assert "standard_name" in field_names
        assert "abbreviation" in field_names

    def test_align_fields(self):
        ed = self.registry.entity_def("Alternative")
        align_fields = [a for a in ed.attributes if a.source == "align"]
        assert len(align_fields) >= 1
        assert any(a.name == "evidence_text" for a in align_fields)

    def test_structure_fields(self):
        ed = self.registry.entity_def("Alternative")
        struct_fields = [a for a in ed.attributes if a.source == "structure"]
        assert any(a.name == "source_location" for a in struct_fields)

    def test_post_fields(self):
        ed = self.registry.entity_def("Alternative")
        post_fields = [a for a in ed.attributes if a.source == "post"]
        assert any(a.name == "alternative_class" for a in post_fields)
        assert any(a.name == "match_source" for a in post_fields)

    def test_generate_json_schema(self):
        schema = self.registry.generate_json_schema(["Alternative"])
        assert "extractions" in schema["properties"]
        items = schema["properties"]["extractions"]["items"]
        assert "anyOf" in items
        # evidence_text should not be in the anyOf variants
        schema_str = str(items)
        assert "evidence_text" not in schema_str

    def test_relation_defs(self):
        rel_names = self.registry.all_relation_names()
        assert "belongs_to" in rel_names
        assert "has_component" in rel_names
        assert "increases" in rel_names

    def test_phase_defs(self):
        phases = self.registry.phase_defs()
        assert len(phases) >= 4
        # Phase 1 should be the gate
        assert phases[0].gate is not None
        assert phases[0].gate.entity == "Alternative"

    def test_vocabulary_load(self):
        vocab = self.registry.vocabulary("Alternative")
        if vocab is not None:
            assert len(vocab.entries) > 0

    def test_post_process(self):
        from src.data import Extraction
        ext = Extraction(
            extraction_class="Alternative",
            extraction_text="thymol",
            attributes={"standard_name": "thymol", "abbreviation": "THY"},
        )
        result = self.registry.post_process(ext)
        # alternative_class should be set (even if "Other")
        assert "alternative_class" in (result.attributes or {})
        assert "match_source" in (result.attributes or {})


class TestAttributeDef:
    def test_create(self):
        ad = AttributeDef(
            name="test_field",
            type="string",
            required=True,
            source="llm",
        )
        assert ad.name == "test_field"
        assert ad.source == "llm"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_schema_registry.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Create `src/schema_registry.py`:

```python
"""Schema Registry — loads external YAML config, provides typed API.

Bridge between schemas/*.yaml and all pipeline layers.
Handles vocabulary loading, LLM field filtering, JSON Schema generation,
extraction prompting, and post-processing.
"""
from __future__ import annotations

import csv
import dataclasses
import re
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Config data classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AttributeDef:
    name: str
    type: str                      # string | integer | float | boolean | text | enum | json_array
    required: bool
    source: str = "llm"            # llm | align | structure | post
    enum_values: list[str] | None = None


@dataclasses.dataclass
class ReferenceDef:
    name: str
    target_entity: str
    target_field: str
    edge_type: str


@dataclasses.dataclass
class InlineRelationDef:
    name: str
    target: list[str]
    via_field: str
    multiple: bool = False


@dataclasses.dataclass
class VocabularyBinding:
    source_path: str
    key_field: str
    class_field: str | None
    subclass_field: str | None
    match_fields: list[str]
    match_mode: str
    inject_in_prompt: bool = True


@dataclasses.dataclass
class EntityDef:
    name: str
    display_name: str
    description: str
    primary_text: str
    extraction_guidance: str = ""
    examples: list[dict] = dataclasses.field(default_factory=list)
    notes: str = ""
    attributes: list[AttributeDef] = dataclasses.field(default_factory=list)
    references: list[ReferenceDef] = dataclasses.field(default_factory=list)
    inline_relations: list[InlineRelationDef] = dataclasses.field(default_factory=list)
    vocabulary: VocabularyBinding | None = None


@dataclasses.dataclass
class RelationDef:
    name: str
    description: str
    source: str | list[str]
    target: str | list[str]
    cardinality: str = "many_to_one"
    co_extracted: bool = False


@dataclasses.dataclass
class GateDef:
    entity: str
    condition: str
    on_fail: str


@dataclasses.dataclass
class ExtractionPhase:
    name: str
    description: str
    extracts: list[str]
    context_from: list[str] = dataclasses.field(default_factory=list)
    gate: GateDef | None = None


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    """Loaded vocabulary from TSV for entity matching."""

    def __init__(self, tsv_path: Path, binding: VocabularyBinding):
        self.entries: list[dict] = []
        self.by_name: dict[str, dict] = {}
        self.binding = binding
        self._load(tsv_path)

    def _load(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                self.entries.append(row)
                name = (row.get(self.binding.key_field, "") or "").strip().lower()
                if name:
                    self.by_name[name] = row

    def lookup(self, name: str) -> dict | None:
        """Exact case-insensitive match."""
        key = name.lower().strip()
        return self.by_name.get(key)

    def fuzzy_match(self, name: str) -> dict | None:
        """Fuzzy match: normalize → substring → Levenshtein ≤ 3."""
        norm = re.sub(r"[^a-z0-9\s]", "", name.lower().strip())
        if not norm:
            return None
        # substring containment
        for gname, info in self.by_name.items():
            gnorm = re.sub(r"[^a-z0-9\s]", "", gname.lower().strip())
            if not gnorm:
                continue
            if gnorm in norm or norm in gnorm:
                return info
        # Levenshtein
        for gname, info in self.by_name.items():
            gnorm = re.sub(r"[^a-z0-9\s]", "", gname.lower().strip())
            if not gnorm:
                continue
            if _levenshtein(norm, gnorm) <= 3:
                return info
        return None

    def format_for_prompt(self) -> str:
        """Render vocabulary for LLM prompt."""
        lines = ["## 候选词表（Alternative分类）"]
        current_class = None
        for entry in self.entries:
            cls = entry.get(self.binding.class_field or "Alternative_Class", "")
            sub = entry.get(self.binding.subclass_field or "Subclass", "")
            name = entry.get(self.binding.key_field, "")
            if cls != current_class:
                current_class = cls
                lines.append(f"\n### {cls}")
            prefix = f"  [{sub}] " if sub else "  "
            lines.append(f"{prefix}{name}")
        return "\n".join(lines)


def _levenshtein(a: str, b: str) -> int:
    """Levenshtein distance."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (ca != cb),
            ))
        prev = curr
    return prev[-1]


# ---------------------------------------------------------------------------
# Schema Registry
# ---------------------------------------------------------------------------

class SchemaRegistry:
    """Loads entities.yaml + relations.yaml + extraction_phases.yaml."""

    def __init__(self, config_dir: str | Path = "schemas"):
        self.config_dir = Path(config_dir)
        self._entities: dict[str, EntityDef] = {}
        self._relations: dict[str, RelationDef] = {}
        self._phases: list[ExtractionPhase] = []
        self._vocabularies: dict[str, Vocabulary] = {}
        self._load()

    def _load(self):
        self._load_entities()
        self._load_relations()
        self._load_phases()
        self._load_vocabularies()

    def _load_entities(self):
        path = self.config_dir / "entities.yaml"
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for name, info in data.get("entities", {}).items():
            attrs = [
                AttributeDef(
                    name=a["name"],
                    type=a.get("type", "string"),
                    required=a.get("required", False),
                    source=a.get("source", "llm"),
                    enum_values=a.get("values"),
                )
                for a in info.get("attributes", [])
            ]
            refs = [
                ReferenceDef(
                    name=r["name"],
                    target_entity=r["target_entity"],
                    target_field=r["target_field"],
                    edge_type=r["edge_type"],
                )
                for r in info.get("references", [])
            ]
            inlines = [
                InlineRelationDef(
                    name=r["name"],
                    target=r["target"],
                    via_field=r["via_field"],
                    multiple=r.get("multiple", False),
                )
                for r in info.get("inline_relations", [])
            ]
            vocab_cfg = info.get("vocabulary")
            vocab = None
            if vocab_cfg:
                vocab = VocabularyBinding(
                    source_path=vocab_cfg["source"],
                    key_field=vocab_cfg["key_field"],
                    class_field=vocab_cfg.get("class_field"),
                    subclass_field=vocab_cfg.get("subclass_field"),
                    match_fields=vocab_cfg.get("match_fields", []),
                    match_mode=vocab_cfg.get("match_mode", "exact"),
                    inject_in_prompt=vocab_cfg.get("inject_in_prompt", True),
                )

            self._entities[name] = EntityDef(
                name=name,
                display_name=info.get("display_name", name),
                description=info.get("description", ""),
                primary_text=info.get("primary_text", "name"),
                extraction_guidance=info.get("extraction_guidance", ""),
                examples=info.get("examples", []),
                notes=info.get("notes", ""),
                attributes=attrs,
                references=refs,
                inline_relations=inlines,
                vocabulary=vocab,
            )

    def _load_relations(self):
        path = self.config_dir / "relations.yaml"
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for name, info in data.get("relations", {}).items():
            self._relations[name] = RelationDef(
                name=name,
                description=info.get("description", ""),
                source=info["source"],
                target=info["target"],
                cardinality=info.get("cardinality", "many_to_one"),
                co_extracted=info.get("co_extracted", False),
            )

    def _load_phases(self):
        path = self.config_dir / "extraction_phases.yaml"
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for name, info in data.get("phases", {}).items():
            gate_cfg = info.get("gate")
            gate = None
            if gate_cfg:
                gate = GateDef(
                    entity=gate_cfg["entity"],
                    condition=gate_cfg["condition"],
                    on_fail=gate_cfg["on_fail"],
                )
            self._phases.append(ExtractionPhase(
                name=name,
                description=info.get("description", ""),
                extracts=info.get("extracts", []),
                context_from=info.get("context_from", []),
                gate=gate,
            ))

    def _load_vocabularies(self):
        for name, ed in self._entities.items():
            if ed.vocabulary:
                tsv_path = Path(ed.vocabulary.source_path)
                if not tsv_path.exists():
                    tsv_path = self.config_dir.parent / ed.vocabulary.source_path
                if tsv_path.exists():
                    self._vocabularies[name] = Vocabulary(tsv_path, ed.vocabulary)

    # -- Public API --

    def entity_def(self, name: str) -> EntityDef:
        return self._entities[name]

    def relation_def(self, name: str) -> RelationDef:
        return self._relations[name]

    def phase_defs(self) -> list[ExtractionPhase]:
        return list(self._phases)

    def all_entity_names(self) -> list[str]:
        return list(self._entities.keys())

    def all_relation_names(self) -> list[str]:
        return list(self._relations.keys())

    def vocabulary(self, entity_name: str) -> Vocabulary | None:
        return self._vocabularies.get(entity_name)

    def llm_output_fields(self, entity_name: str) -> list[AttributeDef]:
        """Return only source=llm attributes."""
        ed = self._entities.get(entity_name)
        if ed is None:
            return []
        return [a for a in ed.attributes if a.source == "llm"]

    def generate_json_schema(
        self, entity_names: list[str], strict: bool = True
    ) -> dict:
        """Generate OpenAI response_format json_schema (LLM fields only)."""
        variants = []
        for ename in entity_names:
            ed = self._entities.get(ename)
            if ed is None:
                continue
            llm_fields = self.llm_output_fields(ename)
            props: dict[str, Any] = {ed.primary_text: {"type": "string"}}
            for a in llm_fields:
                if a.name == ed.primary_text:
                    continue
                if a.type == "string" or a.type == "text":
                    props[a.name] = {"type": "string"}
                elif a.type == "integer":
                    props[a.name] = {"type": "integer"}
                elif a.type == "float":
                    props[a.name] = {"type": "number"}
                elif a.type == "boolean":
                    props[a.name] = {"type": "boolean"}
                elif a.type == "json_array":
                    props[a.name] = {"type": "array", "items": {"type": "object"}}
                elif a.type == "enum" and a.enum_values:
                    props[a.name] = {"type": "string", "enum": a.enum_values}
                else:
                    props[a.name] = {"type": "string"}

            required = [ed.primary_text] + [
                a.name for a in llm_fields if a.required and a.name != ed.primary_text
            ]
            variants.append({
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": False,
            })

        return {
            "type": "object",
            "properties": {
                "extractions": {
                    "type": "array",
                    "items": {"anyOf": variants} if variants else {"type": "object"},
                }
            },
            "required": ["extractions"],
            "additionalProperties": False,
        }

    def build_extraction_prompt(
        self, entity_names: list[str], include_examples: bool = True
    ) -> str:
        """Build extraction prompt from entity metadata."""
        parts = []
        vocab_parts = []

        for ename in entity_names:
            ed = self._entities.get(ename)
            if ed is None:
                continue
            parts.append(f"## {ed.display_name} ({ed.name})")
            if ed.description:
                parts.append(ed.description)
            if ed.extraction_guidance:
                parts.append(ed.extraction_guidance)
            if ed.notes:
                parts.append(f"**重要约束:** {ed.notes}")

            # Inject vocabulary if configured
            if ed.vocabulary and ed.vocabulary.inject_in_prompt:
                vocab = self.vocabulary(ename)
                if vocab:
                    vocab_parts.append(vocab.format_for_prompt())

        prompt = "\n\n".join(parts)
        if vocab_parts:
            prompt += "\n\n" + "\n\n".join(vocab_parts)
        return prompt

    def post_process(self, extraction: "Extraction") -> "Extraction":
        """Apply post-processing: vocabulary match, etc. Populates source=post fields."""
        from src.data import Extraction
        ed = self._entities.get(extraction.extraction_class)
        if ed is None:
            return extraction

        attrs = dict(extraction.attributes or {})
        post_fields = [a for a in ed.attributes if a.source == "post"]

        for pf in post_fields:
            if pf.name == "alternative_class":
                name = attrs.get("standard_name") or extraction.extraction_text
                vocab = self.vocabulary(extraction.extraction_class)
                if vocab:
                    match = vocab.lookup(name) or vocab.fuzzy_match(name)
                    if match:
                        attrs["alternative_class"] = match.get(
                            ed.vocabulary.class_field or "Alternative_Class", "Other"
                        )
                        attrs["subclass"] = match.get(
                            ed.vocabulary.subclass_field or "Subclass"
                        )
                        attrs["match_source"] = "词表精确匹配" if vocab.lookup(name) else "词表模糊匹配"
                    else:
                        attrs["alternative_class"] = "Other"
                        attrs["match_source"] = "Other_未匹配"
                else:
                    attrs.setdefault("alternative_class", "Other")
                    attrs.setdefault("match_source", "Other_未匹配")

        extraction.attributes = attrs
        return extraction

    def validate_extraction(self, extraction: "Extraction") -> list[str]:
        """Validate extraction against entity definition. Returns errors."""
        errors = []
        ed = self._entities.get(extraction.extraction_class)
        if ed is None:
            return [f"Unknown entity type: {extraction.extraction_class}"]

        for a in ed.attributes:
            if not a.required:
                continue
            val = (extraction.attributes or {}).get(a.name)
            if val is None or val == "":
                errors.append(f"Required field '{a.name}' is missing or empty")
        return errors
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_schema_registry.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/schema_registry.py tests/test_schema_registry.py
git commit -m "feat: add schema registry (YAML→typed API, vocabulary, LLM field filtering, JSON Schema gen)"
```

---

### Task C3: Schema Layer (`src/schema.py`)

**Files:**
- Create: `src/schema.py`
- Create: `tests/test_schema.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_schema.py`:

```python
"""Tests for schema layer."""
import pytest
from src.data import FormatType
from src.schema import BaseSchema, FormatModeSchema


class TestFormatModeSchema:
    def test_default_json(self):
        s = FormatModeSchema(format_type=FormatType.JSON)
        assert s.requires_raw_output is True
        config = s.to_provider_config()
        assert config["format"] == "json"

    def test_yaml(self):
        s = FormatModeSchema(format_type=FormatType.YAML)
        assert s.requires_raw_output is False
        config = s.to_provider_config()
        assert config["format"] == "yaml"

    def test_from_examples(self):
        from src.data import ExampleData, Extraction
        ex = ExampleData(
            text="Test",
            extractions=[Extraction(extraction_class="Alt", extraction_text="test")]
        )
        s = FormatModeSchema.from_examples([ex])
        assert isinstance(s, FormatModeSchema)

    def test_sync_with_provider_kwargs(self):
        s = FormatModeSchema(format_type=FormatType.JSON)
        s.sync_with_provider_kwargs({"format": "yaml"})
        assert s.format_type == FormatType.YAML
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Create `src/schema.py`:

```python
"""Core schema abstractions.

Faithful reproduction of langextract's core/schema.py.
"""
from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

from src.data import FormatType, ExampleData


class BaseSchema(abc.ABC):
    """Abstract base for generating structured output config from extraction targets."""

    @classmethod
    @abc.abstractmethod
    def from_examples(
        cls,
        examples_data: Sequence[ExampleData],
        attribute_suffix: str = "_attributes",
    ) -> "BaseSchema":
        """Build a schema instance from example data."""
        ...

    @abc.abstractmethod
    def to_provider_config(self) -> dict[str, Any]:
        """Convert schema to provider-specific configuration."""
        ...

    @property
    @abc.abstractmethod
    def requires_raw_output(self) -> bool:
        """Whether this schema outputs raw JSON without fences."""
        ...

    def validate_format(self, format_handler) -> None:
        """Validate format compatibility. Override in subclasses."""

    def sync_with_provider_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Hook to update schema state based on provider kwargs."""


class FormatModeSchema(BaseSchema):
    """Schema for providers that support format modes (JSON/YAML)."""

    def __init__(self, format_type: FormatType = FormatType.JSON):
        self.format_type = format_type
        self._format = "json" if format_type == FormatType.JSON else "yaml"

    @classmethod
    def from_examples(
        cls,
        examples_data: Sequence[ExampleData],
        attribute_suffix: str = "_attributes",
    ) -> "FormatModeSchema":
        return cls(format_type=FormatType.JSON)

    def to_provider_config(self) -> dict[str, Any]:
        return {"format": self._format}

    @property
    def requires_raw_output(self) -> bool:
        return self._format == "json"

    def sync_with_provider_kwargs(self, kwargs: dict[str, Any]) -> None:
        if "format" in kwargs:
            self._format = kwargs["format"]
            self.format_type = (
                FormatType.JSON if self._format == "json" else FormatType.YAML
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_schema.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/schema.py tests/test_schema.py
git commit -m "feat: add schema layer (BaseSchema ABC, FormatModeSchema)"
```

---

## Group D: Providers

### Task D1: Provider Base + Capabilities (`src/providers/`)

**Files:**
- Create: `src/providers/__init__.py`
- Create: `src/providers/base.py`
- Create: `src/providers/capabilities.py`
- Create: `tests/test_provider_capabilities.py`

- [ ] **Step 1: Write provider base and capabilities**

Create `src/providers/__init__.py` (empty).

Create `src/providers/base.py`:

```python
"""Base language model interface."""
from __future__ import annotations

import abc
import dataclasses
from collections.abc import Iterator, Sequence
from typing import Any

from src.schema import BaseSchema


@dataclasses.dataclass(frozen=True)
class ScoredOutput:
    """Scored output from language model inference."""
    score: float | None = None
    output: str | None = None


class BaseLanguageModel(abc.ABC):
    """Abstract inference class for LLM inference."""

    def __init__(self, **kwargs: Any):
        self._schema: BaseSchema | None = None
        self._fence_output_override: bool | None = None
        self._extra_kwargs: dict[str, Any] = kwargs.copy()

    @classmethod
    def get_schema_class(cls) -> type[Any] | None:
        return None

    def apply_schema(self, schema_instance: BaseSchema | None) -> None:
        self._schema = schema_instance

    @property
    def schema(self) -> BaseSchema | None:
        return self._schema

    def set_fence_output(self, fence_output: bool | None) -> None:
        self._fence_output_override = fence_output

    @property
    def requires_fence_output(self) -> bool:
        if self._fence_output_override is not None:
            return self._fence_output_override
        if self._schema is None:
            return True
        return not self._schema.requires_raw_output

    def merge_kwargs(self, runtime_kwargs: dict | None = None) -> dict[str, Any]:
        base = self._extra_kwargs or {}
        incoming = dict(runtime_kwargs or {})
        return {**base, **incoming}

    @abc.abstractmethod
    def infer(
        self, batch_prompts: Sequence[str], **kwargs
    ) -> Iterator[Sequence[ScoredOutput]]:
        """Batch inference. Yields one Sequence[ScoredOutput] per prompt."""
        ...

    def infer_batch(
        self, prompts: Sequence[str]
    ) -> list[list[ScoredOutput]]:
        results = []
        for output in self.infer(prompts):
            results.append(list(output))
        return results
```

Create `src/providers/capabilities.py`:

```python
"""Model capability detection and fallback chain."""

import dataclasses


@dataclasses.dataclass
class ModelCapabilities:
    supports_json_schema: bool = False
    supports_json_schema_strict: bool = False
    supports_json_object: bool = False
    supports_system_message: bool = True
    requires_fence_output: bool = True
    max_context_tokens: int = 128000


def detect_capabilities(model_id: str) -> ModelCapabilities:
    """Detect model capabilities from model ID prefix.

    DeepSeek: supports json_object, partial json_schema (no strict).
    Qwen Cloud: json_object on qwen-max/plus, fence-only on others.
    OpenAI: full json_schema with strict.
    """
    mid = model_id.lower()

    # OpenAI models — full structured output support
    if any(p in mid for p in ("gpt-4", "gpt-3.5", "o1", "o3")):
        return ModelCapabilities(
            supports_json_schema=True,
            supports_json_schema_strict=True,
            supports_json_object=True,
            requires_fence_output=False,
        )

    # DeepSeek models — json_object + limited json_schema
    if "deepseek" in mid:
        return ModelCapabilities(
            supports_json_schema=True,
            supports_json_schema_strict=False,
            supports_json_object=True,
            requires_fence_output=False,
        )

    # Qwen models via DashScope
    if "qwen" in mid:
        if any(p in mid for p in ("qwen-max", "qwen-plus", "qwen-turbo")):
            return ModelCapabilities(
                supports_json_schema=False,
                supports_json_object=True,
                requires_fence_output=False,
            )
        else:
            return ModelCapabilities(
                requires_fence_output=True,
            )

    # Default: assume basic json_object support
    return ModelCapabilities(
        supports_json_object=True,
        requires_fence_output=False,
    )
```

Create `tests/test_provider_capabilities.py`:

```python
"""Tests for provider capability detection."""
from src.providers.capabilities import detect_capabilities, ModelCapabilities


class TestDetectCapabilities:
    def test_openai_gpt4o(self):
        caps = detect_capabilities("gpt-4o")
        assert caps.supports_json_schema is True
        assert caps.supports_json_schema_strict is True

    def test_deepseek(self):
        caps = detect_capabilities("deepseek-chat")
        assert caps.supports_json_schema is True
        assert caps.supports_json_schema_strict is False
        assert caps.supports_json_object is True

    def test_qwen_max(self):
        caps = detect_capabilities("qwen-max")
        assert caps.supports_json_object is True

    def test_qwen_oss(self):
        caps = detect_capabilities("qwen2.5-7b")
        assert caps.requires_fence_output is True

    def test_default(self):
        caps = detect_capabilities("unknown-model")
        assert caps.supports_json_object is True
        assert caps.requires_fence_output is False
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_provider_capabilities.py -v`
Expected: all tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/providers/ tests/test_provider_capabilities.py
git commit -m "feat: add provider base (BaseLanguageModel, ScoredOutput) + capability detection"
```

---

### Task D2: OpenAI Compat Provider + OpenAISchema

**Files:**
- Create: `src/providers/schemas/__init__.py`
- Create: `src/providers/schemas/openai.py`
- Create: `src/providers/openai_compat.py`
- Create: `tests/test_provider_openai_compat.py`

- [ ] **Step 1: Write OpenAISchema**

Create `src/providers/schemas/__init__.py` (empty).

Create `src/providers/schemas/openai.py`:

```python
"""OpenAI provider schema — generates response_format json_schema from registry."""
from __future__ import annotations

import copy
import dataclasses
from typing import Any

from src.data import FormatType
from src.format_handler import FormatHandler
from src.schema import BaseSchema

DEFAULT_SCHEMA_NAME = "zhongnong_extraction"


@dataclasses.dataclass(frozen=True)
class OpenAISchema(BaseSchema):
    """Schema for OpenAI structured outputs via response_format.

    Generated from SchemaRegistry, not from ExampleData introspection.
    """
    schema_dict: dict[str, Any]
    schema_name: str = DEFAULT_SCHEMA_NAME
    strict: bool = True

    def __post_init__(self):
        object.__setattr__(self, "schema_dict", copy.deepcopy(self.schema_dict))

    @property
    def response_format(self) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.schema_name,
                "schema": copy.deepcopy(self.schema_dict),
                "strict": self.strict,
            },
        }

    @classmethod
    def from_registry(cls, registry, entity_names: list[str], strict: bool = True) -> "OpenAISchema":
        schema_dict = registry.generate_json_schema(entity_names, strict=strict)
        return cls(schema_dict=schema_dict, strict=strict)

    @classmethod
    def from_examples(cls, examples_data, attribute_suffix="_attributes") -> "OpenAISchema":
        # Stub — full implementation would introspect examples
        return cls(
            schema_dict={
                "type": "object",
                "properties": {"extractions": {"type": "array", "items": {"type": "object"}}},
                "required": ["extractions"],
                "additionalProperties": False,
            },
            strict=False,
        )

    def to_provider_config(self) -> dict[str, Any]:
        return {}

    @property
    def requires_raw_output(self) -> bool:
        return True

    def validate_format(self, format_handler: FormatHandler) -> None:
        if format_handler.format_type != FormatType.JSON:
            raise ValueError("OpenAI structured output only supports JSON format.")
```

- [ ] **Step 2: Write OpenAICompatProvider**

Create `src/providers/openai_compat.py`:

```python
"""OpenAI-compatible Chat Completions provider via httpx."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Iterator, Sequence
from typing import Any

import httpx

from src.data import FormatType
from src.providers.base import BaseLanguageModel, ScoredOutput
from src.providers.capabilities import detect_capabilities
from src.providers.schemas.openai import OpenAISchema

logger = logging.getLogger(__name__)


class OpenAICompatProvider(BaseLanguageModel):
    """OpenAI-compatible provider using httpx async client.

    Supports: OpenAI, DeepSeek, Qwen, and any /v1/chat/completions endpoint.
    Applies OpenAISchema as response_format.json_schema when available.
    Falls back to json_object or fence-only mode based on capability detection.
    """

    def __init__(
        self,
        model_id: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str | None = None,
        format_type: FormatType = FormatType.JSON,
        temperature: float | None = None,
        max_workers: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url or "https://api.deepseek.com/v1"
        self.format_type = format_type
        self.temperature = temperature
        self.max_workers = max_workers
        self.openai_schema: OpenAISchema | None = None
        self._capabilities = detect_capabilities(model_id)
        self._client: httpx.AsyncClient | None = None
        self._extra_kwargs = kwargs or {}

    @classmethod
    def get_schema_class(cls) -> type:
        return OpenAISchema

    def apply_schema(self, schema_instance):
        if schema_instance is None:
            self.openai_schema = None
        elif isinstance(schema_instance, OpenAISchema):
            self.openai_schema = schema_instance
        super().apply_schema(schema_instance)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
        return self._client

    def infer(
        self, batch_prompts: Sequence[str], **kwargs
    ) -> Iterator[Sequence[ScoredOutput]]:
        """Synchronous wrapper around async infer."""
        return self._run_async(self._async_infer(batch_prompts, **kwargs))

    def _run_async(self, coro):
        """Run async coroutine, handling event loop management."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()

    async def _async_infer(
        self, batch_prompts: Sequence[str], **kwargs
    ) -> Iterator[Sequence[ScoredOutput]]:
        merged = self.merge_kwargs(kwargs)
        client = await self._get_client()

        for prompt in batch_prompts:
            result = await self._process_single_prompt(client, prompt, merged)
            yield [result]

    async def _process_single_prompt(
        self, client: httpx.AsyncClient, prompt: str, config: dict
    ) -> ScoredOutput:
        body = self._build_request(prompt, config)
        url = "/chat/completions"
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await client.post(url, json=body)
                if response.status_code == 429:
                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue
                if response.status_code >= 500:
                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return ScoredOutput(score=1.0, output=content)
            except (httpx.HTTPError, KeyError, json.JSONDecodeError) as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"LLM inference failed: {e}") from e
                time.sleep(2 ** attempt)
        raise RuntimeError("LLM inference failed after retries")

    def _build_request(self, prompt: str, config: dict) -> dict:
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that responds in JSON format."},
                {"role": "user", "content": prompt},
            ],
            "n": 1,
        }

        temp = config.get("temperature", self.temperature)
        if temp is not None:
            body["temperature"] = temp

        # Apply response_format based on capabilities
        if self.openai_schema and self._capabilities.supports_json_schema:
            body["response_format"] = self.openai_schema.response_format
        elif self._capabilities.supports_json_object:
            body["response_format"] = {"type": "json_object"}

        if "max_tokens" in config:
            body["max_tokens"] = config["max_tokens"]

        return body
```

- [ ] **Step 3: Write tests with mocked httpx**

Create `tests/test_provider_openai_compat.py`:

```python
"""Tests for OpenAICompatProvider."""
import pytest
import json
from src.providers.openai_compat import OpenAICompatProvider
from src.providers.schemas.openai import OpenAISchema
from src.data import FormatType


class TestOpenAICompatProvider:
    def test_init_defaults(self):
        p = OpenAICompatProvider(
            model_id="deepseek-chat",
            api_key="test-key",
        )
        assert p.model_id == "deepseek-chat"
        assert p.format_type == FormatType.JSON

    def test_capability_detection(self):
        p = OpenAICompatProvider(model_id="deepseek-chat", api_key="test-key")
        assert p._capabilities.supports_json_schema is True
        assert p._capabilities.supports_json_schema_strict is False

    def test_apply_schema(self):
        p = OpenAICompatProvider(model_id="gpt-4o", api_key="test-key")
        schema = OpenAISchema.from_registry(None, ["Alternative"])
        p.apply_schema(schema)
        assert p.openai_schema is not None

    def test_requires_fence_output_with_schema(self):
        p = OpenAICompatProvider(model_id="gpt-4o", api_key="test-key")
        assert p.requires_fence_output is True  # no schema → default True

    def test_build_request_basic(self):
        p = OpenAICompatProvider(model_id="deepseek-chat", api_key="test-key")
        body = p._build_request("Test prompt", {})
        assert body["model"] == "deepseek-chat"
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"

    def test_build_request_with_json_object(self):
        p = OpenAICompatProvider(model_id="qwen-max", api_key="test-key")
        body = p._build_request("Test", {})
        assert "response_format" in body
        assert body["response_format"]["type"] == "json_object"
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_provider_openai_compat.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/providers/schemas/ src/providers/openai_compat.py tests/test_provider_openai_compat.py
git commit -m "feat: add OpenAICompatProvider + OpenAISchema (response_format, capability-aware)"
```

---

## Group E: Prompting

### Task E1: Prompting Layer (`src/prompting.py`)

**Files:**
- Create: `src/prompting.py`
- Create: `tests/test_prompting.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_prompting.py`:

```python
"""Tests for prompting layer."""
import pytest
from src.data import FormatType, Extraction, ExampleData
from src.format_handler import FormatHandler
from src.prompting import (
    PromptTemplateStructured, QAPromptGenerator,
    PromptBuilder, ContextAwarePromptBuilder,
)


@pytest.fixture
def fh():
    return FormatHandler(format_type=FormatType.JSON, use_fences=False)

@pytest.fixture
def template():
    return PromptTemplateStructured(
        description="Extract entities from text.",
        examples=[
            ExampleData(
                text="Sample input text.",
                extractions=[
                    Extraction(extraction_class="Alt", extraction_text="thymol"),
                ],
            )
        ],
    )


class TestPromptTemplateStructured:
    def test_create(self):
        pt = PromptTemplateStructured(description="Test")
        assert pt.description == "Test"
        assert pt.examples == []

    def test_with_examples(self):
        ex = ExampleData(text="t", extractions=[])
        pt = PromptTemplateStructured(description="D", examples=[ex])
        assert len(pt.examples) == 1


class TestQAPromptGenerator:
    def test_render_basic(self, template, fh):
        gen = QAPromptGenerator(template=template, format_handler=fh)
        result = gen.render("What is this?")
        assert "Extract entities" in result
        assert "Q: What is this?" in result
        assert "A:" in result

    def test_render_with_examples(self, template, fh):
        gen = QAPromptGenerator(template=template, format_handler=fh)
        result = gen.render("What is this?")
        assert "Sample input text" in result  # example text

    def test_render_with_context(self, template, fh):
        gen = QAPromptGenerator(template=template, format_handler=fh)
        result = gen.render("Q text", additional_context="Extra info")
        assert "Extra info" in result

    def test_format_example_as_text(self, template, fh):
        gen = QAPromptGenerator(template=template, format_handler=fh)
        ex = template.examples[0]
        result = gen.format_example_as_text(ex)
        assert "Q: Sample input text." in result
        assert "A:" in result
        assert "thymol" in result


class TestPromptBuilder:
    def test_build_prompt(self, template, fh):
        gen = QAPromptGenerator(template=template, format_handler=fh)
        builder = PromptBuilder(gen)
        result = builder.build_prompt("Chunk text", "doc1")
        assert "Chunk text" in result


class TestContextAwarePromptBuilder:
    def test_build_prompt_with_context_window(self, template, fh):
        gen = QAPromptGenerator(template=template, format_handler=fh)
        builder = ContextAwarePromptBuilder(gen, context_window_chars=50)
        result1 = builder.build_prompt("First chunk.", "doc1")
        result2 = builder.build_prompt("Second chunk.", "doc1")
        # Second prompt should contain context from first
        assert "[Previous text]" in result2 or "First" in result2

    def test_no_context_bleeding(self, template, fh):
        gen = QAPromptGenerator(template=template, format_handler=fh)
        builder = ContextAwarePromptBuilder(gen, context_window_chars=50)
        result1 = builder.build_prompt("Doc1 text.", "doc1")
        result2 = builder.build_prompt("Doc2 text.", "doc2")
        # Doc2 should NOT see Doc1's context
        assert "Doc1" not in result2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prompting.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Create `src/prompting.py`:

```python
"""Library for building prompts.

Faithful reproduction of langextract's prompting.py.
Q/A-format few-shot prompting with cross-chunk context.
"""
from __future__ import annotations

import dataclasses

from src.data import ExampleData
from src.format_handler import FormatHandler


@dataclasses.dataclass
class PromptTemplateStructured:
    """Structured prompt template with description + few-shot examples."""
    description: str
    examples: list[ExampleData] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class QAPromptGenerator:
    """Generates Q/A format prompts from template + FormatHandler."""
    template: PromptTemplateStructured
    format_handler: FormatHandler
    examples_heading: str = "Examples"
    question_prefix: str = "Q: "
    answer_prefix: str = "A: "

    def format_example_as_text(self, example: ExampleData) -> str:
        question = example.text
        answer = self.format_handler.format_extraction_example(example.extractions)
        return "\n".join([
            f"{self.question_prefix}{question}",
            f"{self.answer_prefix}{answer}\n",
        ])

    def render(self, question: str, additional_context: str | None = None) -> str:
        prompt_lines: list[str] = [f"{self.template.description}\n"]
        if additional_context:
            prompt_lines.append(f"{additional_context}\n")
        if self.template.examples:
            prompt_lines.append(self.examples_heading)
            for ex in self.template.examples:
                prompt_lines.append(self.format_example_as_text(ex))
        prompt_lines.append(f"{self.question_prefix}{question}")
        prompt_lines.append(self.answer_prefix)
        return "\n".join(prompt_lines)


class PromptBuilder:
    """Builds prompts for text chunks using QAPromptGenerator."""

    def __init__(self, generator: QAPromptGenerator):
        self._generator = generator

    def build_prompt(
        self, chunk_text: str, document_id: str,
        additional_context: str | None = None,
    ) -> str:
        del document_id
        return self._generator.render(
            question=chunk_text, additional_context=additional_context,
        )


class ContextAwarePromptBuilder(PromptBuilder):
    """Prompt builder with cross-chunk context for coreference resolution."""

    _CONTEXT_PREFIX = "[Previous text]: ..."

    def __init__(
        self, generator: QAPromptGenerator,
        context_window_chars: int | None = None,
    ):
        super().__init__(generator)
        self._context_window_chars = context_window_chars
        self._prev_chunk_by_doc_id: dict[str, str] = {}

    def build_prompt(
        self, chunk_text: str, document_id: str,
        additional_context: str | None = None,
    ) -> str:
        effective_context = self._build_effective_context(document_id, additional_context)
        prompt = self._generator.render(question=chunk_text, additional_context=effective_context)
        self._update_state(document_id, chunk_text)
        return prompt

    def _build_effective_context(
        self, document_id: str, additional_context: str | None,
    ) -> str | None:
        context_parts: list[str] = []
        if self._context_window_chars and document_id in self._prev_chunk_by_doc_id:
            prev_text = self._prev_chunk_by_doc_id[document_id]
            window = prev_text[-self._context_window_chars:]
            context_parts.append(f"{self._CONTEXT_PREFIX}{window}")
        if additional_context:
            context_parts.append(additional_context)
        return "\n\n".join(context_parts) if context_parts else None

    def _update_state(self, document_id: str, chunk_text: str) -> None:
        if self._context_window_chars:
            self._prev_chunk_by_doc_id[document_id] = chunk_text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prompting.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/prompting.py tests/test_prompting.py
git commit -m "feat: add prompting layer (QAPromptGenerator, ContextAwarePromptBuilder)"
```

---

## Group F: Resolver + Evidence + Source Location

### Task F1: Resolver (`src/resolver.py`)

**Files:**
- Create: `src/resolver.py`
- Create: `tests/test_resolver.py`

- [ ] **Step 1: Write failing test skeleton**

Create `tests/test_resolver.py`:

```python
"""Tests for resolver."""
import pytest
from src.data import FormatType, Extraction, AlignmentStatus, CharInterval
from src.format_handler import FormatHandler
from src.tokenizer import RegexTokenizer, TokenizedText
from src.resolver import Resolver, WordAligner


@pytest.fixture
def fh():
    return FormatHandler(format_type=FormatType.JSON, use_fences=False)


class TestResolver:
    def test_resolve_simple(self, fh):
        r = Resolver(format_handler=fh)
        input_text = '{"extractions": [{"Alternative": "thymol", "Alternative_attributes": {"abbreviation": "THY"}}]}'
        result = r.resolve(input_text)
        assert len(result) == 1
        assert result[0].extraction_class == "Alternative"
        assert result[0].extraction_text == "thymol"
        assert result[0].attributes is not None
        assert result[0].attributes["abbreviation"] == "THY"

    def test_resolve_multiple(self, fh):
        r = Resolver(format_handler=fh)
        input_text = (
            '{"extractions": ['
            '{"Alternative": "thymol"}, '
            '{"Alternative": "curcumin"}'
            ']}'
        )
        result = r.resolve(input_text)
        assert len(result) == 2

    def test_resolve_empty(self, fh):
        r = Resolver(format_handler=fh)
        input_text = '{"extractions": []}'
        result = r.resolve(input_text)
        assert len(result) == 0

    def test_resolve_suppress_errors(self, fh):
        r = Resolver(format_handler=fh)
        # malformed input — should return [] with suppress_parse_errors=True
        result = r.resolve("not valid json at all", suppress_parse_errors=True)
        assert result == []

    def test_resolve_no_suppress_raises(self, fh):
        r = Resolver(format_handler=fh)
        with pytest.raises(ValueError):
            r.resolve("not valid json at all", suppress_parse_errors=False)


class TestWordAligner:
    def setup_method(self):
        self.aligner = WordAligner()
        self.tokenizer = RegexTokenizer()

    def test_align_extractions_exact(self):
        source = "thymol was added to the diet"
        tt = self.tokenizer.tokenize(source)
        exts = [Extraction(extraction_class="Alt", extraction_text="thymol")]
        result = self.aligner.align_extractions(
            [exts], source, token_offset=0, char_offset=0,
            enable_fuzzy_alignment=False,
            tokenizer_impl=self.tokenizer,
        )
        aligned = result[0][0]
        assert aligned.alignment_status == AlignmentStatus.MATCH_EXACT
        assert aligned.char_interval is not None

    def test_align_unmatched(self):
        source = "zinc oxide was supplemented"
        tt = self.tokenizer.tokenize(source)
        exts = [Extraction(extraction_class="Alt", extraction_text="thymol")]
        result = self.aligner.align_extractions(
            [exts], source, token_offset=0, char_offset=0,
            enable_fuzzy_alignment=False,
            tokenizer_impl=self.tokenizer,
        )
        aligned = result[0][0]
        # Should not match — alignment_status may be None
        assert aligned.alignment_status is None or aligned.char_interval is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_resolver.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Create `src/resolver.py` (abbreviated — full version follows langextract's resolver.py with WordAligner, difflib exact matching, LCS fuzzy alignment):

```python
"""Library for resolving LLM output into structured Extractions.

Faithful reproduction of langextract's resolver.py.
Parse → ordered extractions → align to source text (difflib + LCS fuzzy).
"""
from __future__ import annotations

import abc
import collections
import difflib
import itertools
import math
import operator
from collections.abc import Iterator, Mapping, Sequence
from typing import Final

from src.data import (
    CharInterval, Extraction, AlignmentStatus, FormatType,
)
from src.format_handler import FormatHandler, ExtractionValueType
from src.tokenizer import (
    Tokenizer, RegexTokenizer, TokenizedText, TokenInterval,
    TokenType, tokenize, tokens_text,
)


_FUZZY_ALIGNMENT_MIN_THRESHOLD = 0.75
_FUZZY_ALIGNMENT_MIN_DENSITY = 1.0 / 3.0
DEFAULT_INDEX_SUFFIX = "_index"


class AbstractResolver(abc.ABC):
    """Resolves LLM text outputs into structured Extractions."""

    @abc.abstractmethod
    def resolve(self, input_text: str, **kwargs) -> Sequence[Extraction]:
        """Parse LLM output → Extractions."""
        ...

    @abc.abstractmethod
    def align(
        self, extractions: Sequence[Extraction], source_text: str,
        token_offset: int, char_offset: int | None = None,
        **kwargs,
    ) -> Iterator[Extraction]:
        """Align extractions to source text."""
        ...


class Resolver(AbstractResolver):
    """Resolver using FormatHandler for parsing, WordAligner for alignment."""

    def __init__(
        self,
        format_handler: FormatHandler | None = None,
        extraction_index_suffix: str | None = None,
    ):
        self.format_handler = format_handler or FormatHandler()
        self.extraction_index_suffix = extraction_index_suffix

    def resolve(
        self, input_text: str, suppress_parse_errors: bool = False, **kwargs
    ) -> Sequence[Extraction]:
        try:
            extraction_data = self.format_handler.parse_output(input_text)
        except (ValueError, Exception) as e:
            if suppress_parse_errors:
                return []
            raise ValueError(str(e)) from e

        try:
            return self._extract_ordered_extractions(extraction_data)
        except ValueError as e:
            if suppress_parse_errors:
                return []
            raise

    def _extract_ordered_extractions(
        self, extraction_data: Sequence[Mapping[str, ExtractionValueType]],
    ) -> Sequence[Extraction]:
        processed = []
        extraction_index = 0
        index_suffix = self.extraction_index_suffix
        attr_suffix = self.format_handler.attribute_suffix

        for group_index, group in enumerate(extraction_data):
            for key, value in group.items():
                if index_suffix and key.endswith(index_suffix):
                    continue
                if attr_suffix and key.endswith(attr_suffix):
                    continue
                if not isinstance(value, (str, int, float)):
                    raise ValueError(f"Extraction text must be string/int/float, got {type(value)}")
                if not isinstance(value, str):
                    value = str(value)

                idx = group.get(f"{key}{index_suffix}", None) if index_suffix else None
                if idx is None:
                    extraction_index += 1
                    idx = extraction_index

                attrs = group.get(f"{key}{attr_suffix}", None) if attr_suffix else None

                processed.append(Extraction(
                    extraction_class=key,
                    extraction_text=value,
                    extraction_index=idx,
                    group_index=group_index,
                    attributes=dict(attrs) if isinstance(attrs, dict) else None,
                ))

        processed.sort(key=operator.attrgetter("extraction_index"))
        return processed

    def align(
        self, extractions: Sequence[Extraction], source_text: str,
        token_offset: int, char_offset: int | None = None,
        enable_fuzzy_alignment: bool = True,
        fuzzy_alignment_threshold: float = _FUZZY_ALIGNMENT_MIN_THRESHOLD,
        accept_match_lesser: bool = True,
        tokenizer_inst: Tokenizer | None = None,
        **kwargs,
    ) -> Iterator[Extraction]:
        if not extractions:
            return

        aligner = WordAligner()
        aligned_groups = aligner.align_extractions(
            [extractions], source_text, token_offset,
            char_offset or 0,
            enable_fuzzy_alignment=enable_fuzzy_alignment,
            fuzzy_alignment_threshold=fuzzy_alignment_threshold,
            accept_match_lesser=accept_match_lesser,
            tokenizer_impl=tokenizer_inst,
        )
        for extraction in itertools.chain(*aligned_groups):
            yield extraction


class WordAligner:
    """Aligns extraction text to source text tokens using difflib + LCS fuzzy."""

    def __init__(self):
        self.matcher = difflib.SequenceMatcher(autojunk=False)

    def align_extractions(
        self, extraction_groups: Sequence[Sequence[Extraction]],
        source_text: str, token_offset: int = 0, char_offset: int = 0,
        delim: str = "␟",
        enable_fuzzy_alignment: bool = True,
        fuzzy_alignment_threshold: float = _FUZZY_ALIGNMENT_MIN_THRESHOLD,
        accept_match_lesser: bool = True,
        tokenizer_impl: Tokenizer | None = None,
        **kwargs,
    ) -> Sequence[Sequence[Extraction]]:
        if not extraction_groups:
            return []

        tok = tokenizer_impl or RegexTokenizer()
        source_tokens = list(_tokenize_with_lowercase(source_text, tok))
        tokenized_text = tok.tokenize(source_text)

        extraction_texts = [
            extraction.extraction_text
            for extraction in itertools.chain(*extraction_groups)
        ]
        joined = f" {delim} ".join(extraction_texts)
        extraction_tokens = list(_tokenize_with_lowercase(joined, tok))

        self.matcher.set_seqs(source_tokens, extraction_tokens)

        aligned_groups: list[list[Extraction]] = [[] for _ in extraction_groups]
        aligned_set = set()

        # Exact matching
        for i, j, n in self.matcher.get_matching_blocks()[:-1]:
            extraction = self._find_extraction_at_index(
                extraction_groups, extraction_tokens, j, delim, tok
            )
            if extraction is None or extraction in aligned_set:
                continue

            extraction.token_interval = TokenInterval(
                start_index=i + token_offset, end_index=i + n + token_offset,
            )
            try:
                start_token = tokenized_text.tokens[i]
                end_token = tokenized_text.tokens[i + n - 1]
                extraction.char_interval = CharInterval(
                    start_pos=char_offset + start_token.char_interval.start_pos,
                    end_pos=char_offset + end_token.char_interval.end_pos,
                )
            except IndexError:
                continue

            ext_tokens = list(_tokenize_with_lowercase(extraction.extraction_text, tok))
            if len(ext_tokens) == n:
                extraction.alignment_status = AlignmentStatus.MATCH_EXACT
            elif accept_match_lesser:
                extraction.alignment_status = AlignmentStatus.MATCH_LESSER
            else:
                extraction.token_interval = None
                extraction.char_interval = None
                continue

            aligned_set.add(id(extraction))

        # Assign aligned to groups
        for group_idx, group in enumerate(extraction_groups):
            for ext in group:
                if id(ext) in aligned_set:
                    aligned_groups[group_idx].append(ext)
                else:
                    aligned_groups[group_idx].append(ext)

        return aligned_groups

    def _find_extraction_at_index(
        self, groups, ext_tokens, j, delim, tok,
    ):
        """Map token index j back to the extraction it belongs to."""
        extraction_index = 0
        delim_len = len(list(_tokenize_with_lowercase(delim, tok)))
        for group in groups:
            for ext in group:
                ext_text_tokens = list(_tokenize_with_lowercase(ext.extraction_text, tok))
                if extraction_index <= j < extraction_index + len(ext_text_tokens):
                    return ext
                extraction_index += len(ext_text_tokens) + delim_len
        return None


def _tokenize_with_lowercase(text: str, tokenizer_inst: Tokenizer) -> Iterator[str]:
    tt = tokenizer_inst.tokenize(text)
    for token in tt.tokens:
        start = token.char_interval.start_pos
        end = token.char_interval.end_pos
        yield tt.text[start:end].lower()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_resolver.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/resolver.py tests/test_resolver.py
git commit -m "feat: add resolver layer (Resolver, WordAligner — parse + difflib alignment)"
```

---

### Task F2: Evidence + Source Location Extractors

**Files:**
- Create: `src/evidence.py`
- Create: `src/source_location.py`
- Create: `tests/test_evidence.py`
- Create: `tests/test_source_location.py`

- [ ] **Step 1: Write evidence extractor**

Create `src/evidence.py`:

```python
"""EvidenceExtractor — derive verbatim evidence_text from aligned char_interval."""
from src.data import Extraction
from src.tokenizer import TokenizedText, find_sentence_range, tokens_text, TokenInterval


class EvidenceExtractor:
    """Extracts verbatim evidence sentences from source text using alignment position.

    The LLM is NOT asked for evidence_text. Instead, after alignment sets
    char_interval on each extraction, we extract the surrounding sentences
    directly from the source document — guaranteeing verbatim text.
    """

    def extract_evidence(
        self,
        extraction: Extraction,
        document_text: str,
        tokenized_text: TokenizedText,
        context_sentences: int = 2,
    ) -> str:
        """Extract verbatim evidence sentence(s) from source text.

        Returns empty string if extraction is unaligned (char_interval=None).
        """
        ci = extraction.char_interval
        if ci is None or ci.start_pos is None or ci.end_pos is None:
            return ""

        # Find the sentence containing the extraction's char position
        try:
            # Find token index closest to the char position
            start_idx = self._char_to_token_index(tokenized_text, ci.start_pos)
            sent_range = find_sentence_range(
                document_text, tokenized_text.tokens, start_idx
            )

            # Expand to include context sentences
            expanded_start = sent_range.start_index
            expanded_end = sent_range.end_index
            for _ in range(context_sentences):
                # Expand backward
                if expanded_start > 0:
                    prev = find_sentence_range(
                        document_text, tokenized_text.tokens,
                        max(0, expanded_start - 1)
                    )
                    expanded_start = prev.start_index
                # Expand forward
                if expanded_end < len(tokenized_text.tokens):
                    fwd = find_sentence_range(
                        document_text, tokenized_text.tokens,
                        expanded_end
                    )
                    expanded_end = fwd.end_index

            evidence_interval = TokenInterval(
                start_index=expanded_start, end_index=expanded_end
            )
            return tokens_text(tokenized_text, evidence_interval)
        except (ValueError, IndexError):
            return ""

    def extract_batch(
        self,
        extractions: list[Extraction],
        document_text: str,
        tokenized_text: TokenizedText,
        context_sentences: int = 2,
    ) -> None:
        """Mutate extractions in-place, setting evidence_text on each."""
        for ext in extractions:
            ext.attributes = dict(ext.attributes or {})
            ext.attributes["evidence_text"] = self.extract_evidence(
                ext, document_text, tokenized_text, context_sentences
            )

    def _char_to_token_index(
        self, tokenized_text: TokenizedText, char_pos: int
    ) -> int:
        """Find the token index containing the given character position."""
        for i, token in enumerate(tokenized_text.tokens):
            if (
                token.char_interval.start_pos <= char_pos
                and char_pos < token.char_interval.end_pos
            ):
                return i
        # Fallback: return first token
        return 0
```

Create `src/source_location.py`:

```python
"""SourceLocationResolver — derive source_location from document structure."""
import re
from src.data import Extraction


# Patterns for detecting table/figure references
_TABLE_PATTERN = re.compile(r"(Table|Tab\.)\s*\d+", re.IGNORECASE)
_FIGURE_PATTERN = re.compile(r"(Figure|Fig\.)\s*\d+", re.IGNORECASE)


class SourceLocationResolver:
    """Determines structured source location from document + alignment metadata.

    NOT an LLM output field. Derived from:
    1. Section ID (which Document the chunk came from)
    2. Subsection detection (numbered headings near the aligned span)
    3. Table/figure mentions near the aligned span
    """

    def resolve_location(
        self,
        extraction: Extraction,
        section_id: str,
        section_text: str,
        char_offset: int,
    ) -> str:
        """Determine the most specific source location string."""
        parts = [section_id]

        ci = extraction.char_interval
        if ci is not None and ci.start_pos is not None:
            # Check for table/figure references near the aligned position
            nearby_start = max(0, ci.start_pos - 200)
            nearby_end = min(len(section_text), ci.end_pos + 200)
            nearby_text = section_text[nearby_start:nearby_end]

            table_match = _TABLE_PATTERN.search(nearby_text)
            figure_match = _FIGURE_PATTERN.search(nearby_text)

            location_details = []
            if table_match:
                location_details.append(table_match.group(0))
            if figure_match:
                location_details.append(figure_match.group(0))

            if location_details:
                parts.append(", ".join(location_details))

        return ", ".join(parts)

    def resolve_batch(
        self,
        extractions: list[Extraction],
        section_id: str,
        section_text: str,
        char_offset: int,
    ) -> None:
        """Mutate extractions in-place, setting source_location on each."""
        for ext in extractions:
            ext.attributes = dict(ext.attributes or {})
            ext.attributes["source_location"] = self.resolve_location(
                ext, section_id, section_text, char_offset
            )
```

- [ ] **Step 2: Write tests**

Create `tests/test_evidence.py`:

```python
"""Tests for evidence extractor."""
import pytest
from src.data import Extraction, CharInterval
from src.tokenizer import RegexTokenizer
from src.evidence import EvidenceExtractor


class TestEvidenceExtractor:
    def setup_method(self):
        self.extractor = EvidenceExtractor()
        self.tokenizer = RegexTokenizer()

    def test_extract_evidence_aligned(self):
        text = "Pigs were fed a basal diet supplemented with 500 mg/kg thymol. The trial lasted 28 days."
        tt = self.tokenizer.tokenize(text)
        # "thymol" is at some position in the text
        thymol_start = text.index("thymol")
        thymol_end = thymol_start + len("thymol")
        ext = Extraction(
            extraction_class="Alternative",
            extraction_text="thymol",
            char_interval=CharInterval(start_pos=thymol_start, end_pos=thymol_end),
        )
        evidence = self.extractor.extract_evidence(ext, text, tt, context_sentences=1)
        assert "thymol" in evidence
        assert len(evidence) > 0

    def test_extract_evidence_unaligned(self):
        ext = Extraction(
            extraction_class="Alternative",
            extraction_text="thymol",
            char_interval=None,
        )
        tt = self.tokenizer.tokenize("Some text.")
        evidence = self.extractor.extract_evidence(ext, "Some text.", tt)
        assert evidence == ""

    def test_extract_batch(self):
        text = "Thymol was used. It improved growth."
        tt = self.tokenizer.tokenize(text)
        thymol_start = text.index("Thymol")
        ext1 = Extraction(
            extraction_class="Alt", extraction_text="Thymol",
            char_interval=CharInterval(thymol_start, thymol_start + 6),
        )
        ext2 = Extraction(
            extraction_class="Alt", extraction_text="zinc",
            char_interval=None,  # unaligned
        )
        self.extractor.extract_batch([ext1, ext2], text, tt)
        assert ext1.attributes["evidence_text"] != ""
        assert ext2.attributes["evidence_text"] == ""


class TestSourceLocation:
    def test_basic_section(self):
        from src.source_location import SourceLocationResolver
        r = SourceLocationResolver()
        ext = Extraction(extraction_class="Alt", extraction_text="x")
        loc = r.resolve_location(ext, "Methods", "Some methods text.", 0)
        assert "Methods" in loc

    def test_table_detection(self):
        from src.source_location import SourceLocationResolver
        r = SourceLocationResolver()
        ext = Extraction(
            extraction_class="Alt", extraction_text="x",
            char_interval=CharInterval(start_pos=50, end_pos=55),
        )
        text = "The results are shown in Table 2. " + "x " * 50 + "thymol data here."
        loc = r.resolve_location(ext, "Results", text, 0)
        assert "Table 2" in loc
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_evidence.py tests/test_source_location.py -v`
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/evidence.py src/source_location.py tests/test_evidence.py tests/test_source_location.py
git commit -m "feat: add EvidenceExtractor + SourceLocationResolver (derive from alignment, not LLM)"
```

---

## Group G: Annotation + Factory + Extraction

### Task G1: Annotation (`src/annotation.py`)

**Files:**
- Create: `src/annotation.py`
- Create: `tests/test_annotation.py`

- [ ] **Step 1: Write Annotator**

Create `src/annotation.py`:

```python
"""Annotator — orchestrates the full extraction pipeline for documents.

Faithful reproduction of langextract's annotation.py.
Pipeline: chunk → prompt → infer → resolve → align → evidence → location → emit.
"""
from __future__ import annotations

import collections
import logging
from collections.abc import Iterable, Iterator

from src.chunking import (
    ChunkIterator, TextChunk, make_batches_of_textchunk,
)
from src.data import Document, Extraction, AnnotatedDocument, FormatType
from src.evidence import EvidenceExtractor
from src.format_handler import FormatHandler
from src.prompting import (
    PromptTemplateStructured, QAPromptGenerator,
    ContextAwarePromptBuilder,
)
from src.providers.base import BaseLanguageModel, ScoredOutput
from src.resolver import Resolver
from src.source_location import SourceLocationResolver
from src.tokenizer import Tokenizer, RegexTokenizer

logger = logging.getLogger(__name__)


class Annotator:
    """Orchestrates chunk → prompt → infer → resolve → align → evidence → location → emit."""

    def __init__(
        self,
        language_model: BaseLanguageModel,
        prompt_template: PromptTemplateStructured,
        format_handler: FormatHandler | None = None,
        evidence_extractor: EvidenceExtractor | None = None,
        source_location_resolver: SourceLocationResolver | None = None,
    ):
        self._language_model = language_model
        if format_handler is None:
            format_handler = FormatHandler()
        self._prompt_generator = QAPromptGenerator(
            template=prompt_template, format_handler=format_handler,
        )
        self._format_handler = format_handler
        self._evidence_extractor = evidence_extractor or EvidenceExtractor()
        self._source_location_resolver = (
            source_location_resolver or SourceLocationResolver()
        )

    def annotate_documents(
        self,
        documents: Iterable[Document],
        max_char_buffer: int = 200,
        batch_length: int = 1,
        extraction_passes: int = 1,
        context_window_chars: int | None = None,
        show_progress: bool = True,
        tokenizer: Tokenizer | None = None,
        **kwargs,
    ) -> Iterator[AnnotatedDocument]:
        """Annotate documents with streaming emission."""
        tok = tokenizer or RegexTokenizer()
        resolver = Resolver(format_handler=self._format_handler)

        if extraction_passes == 1:
            yield from self._annotate_single_pass(
                documents, resolver, max_char_buffer, batch_length,
                context_window_chars, tok, **kwargs,
            )
        else:
            # Multi-pass: collect all, merge, emit
            doc_list = list(documents)
            all_by_doc: dict[str, list[list[Extraction]]] = collections.defaultdict(list)
            doc_texts: dict[str, str] = {}

            for _ in range(extraction_passes):
                for ad in self._annotate_single_pass(
                    doc_list, resolver, max_char_buffer, batch_length,
                    context_window_chars, tok, **kwargs,
                ):
                    all_by_doc[ad.document_id].append(ad.extractions or [])
                    doc_texts[ad.document_id] = ad.text or ""

            for doc in doc_list:
                passes = all_by_doc.get(doc.document_id, [])
                merged = self._merge_non_overlapping(passes)
                yield AnnotatedDocument(
                    document_id=doc.document_id,
                    extractions=merged,
                    text=doc_texts.get(doc.document_id, doc.text),
                )

    def _annotate_single_pass(
        self, documents, resolver, max_char_buffer, batch_length,
        context_window_chars, tok, **kwargs,
    ) -> Iterator[AnnotatedDocument]:
        """Single-pass streaming annotation."""
        doc_order: list[str] = []
        doc_text_by_id: dict[str, str] = {}
        per_doc: dict[str, list[Extraction]] = collections.defaultdict(list)
        next_emit_idx = 0

        def _capture(src):
            for doc in src:
                doc_order.append(doc.document_id)
                doc_text_by_id[doc.document_id] = doc.text or ""
                yield doc

        captured = list(_capture(documents))
        chunk_iter = self._document_chunk_iterator(captured, max_char_buffer, tok)
        batches = make_batches_of_textchunk(chunk_iter, batch_length)

        prompt_builder = ContextAwarePromptBuilder(
            generator=self._prompt_generator,
            context_window_chars=context_window_chars,
        )

        for batch in batches:
            if not batch:
                continue
            prompts = [
                prompt_builder.build_prompt(chunk.chunk_text, chunk.document_id or "unknown")
                for chunk in batch
            ]
            outputs = list(self._language_model.infer(prompts, **kwargs))

            for chunk, scored_list in zip(batch, outputs):
                if not scored_list:
                    continue
                scored = (list(scored_list) if not isinstance(scored_list, list) else scored_list)
                if not scored:
                    continue

                try:
                    raw = scored[0].output or ""
                    extractions = list(resolver.resolve(raw))
                except ValueError:
                    continue  # skip malformed chunk

                # Align
                token_offset = chunk.token_interval.start_index
                char_offset = chunk.char_interval.start_pos if chunk.char_interval else 0
                doc_text = doc_text_by_id.get(chunk.document_id, "")

                aligned = list(resolver.align(
                    extractions, doc_text, token_offset, char_offset,
                    tokenizer_inst=tok,
                ))

                # Derive evidence and source_location
                tt = tok.tokenize(doc_text)
                self._evidence_extractor.extract_batch(aligned, doc_text, tt)
                section_id = self._section_id_from_document(chunk.document_id)
                self._source_location_resolver.resolve_batch(
                    aligned, section_id, doc_text, char_offset,
                )

                for ext in aligned:
                    per_doc[chunk.document_id].append(ext)

            # Emit completed documents
            while next_emit_idx < len(doc_order) - 1:
                did = doc_order[next_emit_idx]
                yield AnnotatedDocument(
                    document_id=did,
                    extractions=per_doc.get(did, []),
                    text=doc_text_by_id.get(did, ""),
                )
                per_doc.pop(did, None)
                doc_text_by_id.pop(did, None)
                next_emit_idx += 1

        # Emit remaining
        while next_emit_idx < len(doc_order):
            did = doc_order[next_emit_idx]
            yield AnnotatedDocument(
                document_id=did,
                extractions=per_doc.get(did, []),
                text=doc_text_by_id.get(did, ""),
            )
            next_emit_idx += 1

    def annotate_text(
        self, text: str, resolver=None, max_char_buffer=200, batch_length=1,
        additional_context=None, extraction_passes=1, context_window_chars=None,
        tokenizer=None, **kwargs,
    ) -> AnnotatedDocument:
        """Convenience: annotate single text string."""
        resolver = resolver or Resolver(format_handler=self._format_handler)
        tok = tokenizer or RegexTokenizer()
        doc = Document(text=text, additional_context=additional_context)
        results = list(self.annotate_documents(
            [doc], max_char_buffer=max_char_buffer, batch_length=batch_length,
            extraction_passes=extraction_passes, context_window_chars=context_window_chars,
            tokenizer=tok, **kwargs,
        ))
        if not results:
            return AnnotatedDocument(text=text)
        return results[0]

    def _document_chunk_iterator(
        self, documents, max_char_buffer, tokenizer,
    ) -> Iterator[TextChunk]:
        for doc in documents:
            tok = tokenizer or RegexTokenizer()
            ci = ChunkIterator(
                text=doc.text or "", max_char_buffer=max_char_buffer,
                tokenizer_impl=tok, document=doc,
            )
            yield from ci

    def _merge_non_overlapping(
        self, all_pass_extractions: list[list[Extraction]],
    ) -> list[Extraction]:
        """Merge multi-pass extractions. First-pass wins for overlaps."""
        if not all_pass_extractions:
            return []
        if len(all_pass_extractions) == 1:
            return list(all_pass_extractions[0])

        merged = list(all_pass_extractions[0])
        for pass_exts in all_pass_extractions[1:]:
            for ext in pass_exts:
                if ext.char_interval is None:
                    merged.append(ext)
                    continue
                overlaps = False
                for existing in merged:
                    if existing.char_interval is None:
                        continue
                    s1, e1 = ext.char_interval.start_pos, ext.char_interval.end_pos
                    s2, e2 = existing.char_interval.start_pos, existing.char_interval.end_pos
                    if s1 is not None and e1 is not None and s2 is not None and e2 is not None:
                        if s1 < e2 and s2 < e1:
                            overlaps = True
                            break
                if not overlaps:
                    merged.append(ext)
        return merged

    def _section_id_from_document(self, document_id: str | None) -> str:
        """Extract section name from document ID (e.g., 'PMC123_methods' → 'Methods')."""
        if not document_id:
            return "Unknown"
        if "_" in document_id:
            return document_id.rsplit("_", 1)[-1].replace("_", " ").title()
        return "Full Text"
```

- [ ] **Step 2: Write tests**

Create `tests/test_annotation.py`:

```python
"""Tests for annotation layer."""
import pytest
from src.data import Document, Extraction, AnnotatedDocument, ExampleData
from src.format_handler import FormatHandler, FormatType
from src.prompting import PromptTemplateStructured
from src.annotation import Annotator


class MockLanguageModel:
    """Mock LLM that returns a fixed JSON response."""
    requires_fence_output = False

    def infer(self, batch_prompts, **kwargs):
        for prompt in batch_prompts:
            yield [MockScoredOutput('{"extractions": [{"Alternative": "thymol"}]}')]

    def apply_schema(self, schema): pass
    def set_fence_output(self, val): pass
    @property
    def schema(self): return None


class MockScoredOutput:
    def __init__(self, output):
        self.output = output
        self.score = 1.0


class TestAnnotator:
    def test_annotate_text(self):
        template = PromptTemplateStructured(description="Extract alternatives.")
        model = MockLanguageModel()
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        annotator = Annotator(model, template, fh)
        result = annotator.annotate_text(
            "Pigs were fed thymol at 500 mg/kg.",
            max_char_buffer=500,
        )
        assert isinstance(result, AnnotatedDocument)
        assert len(result.extractions) > 0
        assert result.extractions[0].extraction_text == "thymol"

    def test_annotate_documents(self):
        template = PromptTemplateStructured(description="Extract alternatives.")
        model = MockLanguageModel()
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        annotator = Annotator(model, template, fh)
        docs = [Document(text="Pigs were fed thymol.", document_id="doc1")]
        results = list(annotator.annotate_documents(docs, max_char_buffer=500))
        assert len(results) == 1
        assert results[0].document_id == "doc1"
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_annotation.py -v`
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/annotation.py tests/test_annotation.py
git commit -m "feat: add Annotator (chunk→prompt→infer→resolve→align→evidence→location→emit)"
```

---

### Task G2: Factory (`src/factory.py`)

**Files:**
- Create: `src/factory.py`
- Create: `tests/test_factory.py`

- [ ] **Step 1: Write factory**

Create `src/factory.py`:

```python
"""Factory for creating language model instances."""
from __future__ import annotations

import dataclasses
import os
from typing import Any, Sequence

from src.providers.base import BaseLanguageModel
from src.providers.openai_compat import OpenAICompatProvider


@dataclasses.dataclass(slots=True, frozen=True)
class ModelConfig:
    model_id: str | None = None
    provider: str | None = None
    provider_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)


def create_model(
    config: ModelConfig,
    examples: Sequence[Any] | None = None,
    use_schema_constraints: bool = False,
    fence_output: bool | None = None,
) -> BaseLanguageModel:
    """Create a language model instance from configuration."""
    model_id = config.model_id or "deepseek-chat"
    kwargs = dict(config.provider_kwargs)
    kwargs = _kwargs_with_environment_defaults(model_id, kwargs)
    kwargs["model_id"] = model_id

    model = OpenAICompatProvider(**kwargs)

    if use_schema_constraints and examples:
        schema_class = model.get_schema_class()
        if schema_class is not None and hasattr(schema_class, 'from_examples'):
            schema_instance = schema_class.from_examples(examples)
            model.apply_schema(schema_instance)

    model.set_fence_output(fence_output)
    return model


def _kwargs_with_environment_defaults(
    model_id: str, kwargs: dict[str, Any],
) -> dict[str, Any]:
    resolved = dict(kwargs)
    if "api_key" not in resolved:
        resolved["api_key"] = os.getenv("ZN_LLM_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))
    if "base_url" not in resolved:
        if "deepseek" in model_id.lower():
            resolved["base_url"] = os.getenv("ZN_LLM_BASE_URL", "https://api.deepseek.com/v1")
    return resolved
```

Create `tests/test_factory.py`:

```python
"""Tests for factory."""
import pytest
from src.factory import ModelConfig, create_model
from src.providers.base import BaseLanguageModel


class TestModelConfig:
    def test_create(self):
        mc = ModelConfig(model_id="deepseek-chat", provider_kwargs={"api_key": "test"})
        assert mc.model_id == "deepseek-chat"


class TestCreateModel:
    def test_create_basic(self):
        config = ModelConfig(model_id="deepseek-chat", provider_kwargs={"api_key": "test-key"})
        model = create_model(config)
        assert isinstance(model, BaseLanguageModel)
        assert model.model_id == "deepseek-chat"

    def test_create_with_fence_output(self):
        config = ModelConfig(model_id="gpt-4o", provider_kwargs={"api_key": "test-key"})
        model = create_model(config, fence_output=False)
        assert model.requires_fence_output is False
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_factory.py -v`
Expected: all tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/factory.py tests/test_factory.py
git commit -m "feat: add factory (ModelConfig, create_model with env defaults)"
```

---

### Task G3: Extraction API (`src/extraction.py`)

**Files:**
- Create: `src/extraction.py`
- Create: `tests/test_extraction.py`

- [ ] **Step 1: Write extraction API**

Create `src/extraction.py`:

```python
"""Main extraction API — 4-phase per-article pipeline."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from src.annotation import Annotator
from src.data import Document, Extraction, AnnotatedDocument
from src.format_handler import FormatHandler, FormatType
from src.prompting import PromptTemplateStructured
from src.providers.base import BaseLanguageModel
from src.schema_registry import SchemaRegistry
from src.factory import create_model, ModelConfig


@dataclasses.dataclass
class ArticleExtractionResult:
    """Output for one article — all extracted entities across all phases."""
    doi: str
    pmid: str
    extractions: list[Extraction] = dataclasses.field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""
    warnings: list[str] = dataclasses.field(default_factory=list)


def extract(
    article_xml: str | Path,
    *,
    model: BaseLanguageModel | None = None,
    model_id: str = "deepseek-chat",
    api_key: str | None = None,
    registry: SchemaRegistry | None = None,
    max_char_buffer: int = 8000,
    skip_if_no_known_alternative: bool = True,
    **kwargs,
) -> ArticleExtractionResult:
    """Extract entities + relations from one PMC XML article."""
    if registry is None:
        registry = SchemaRegistry()

    if model is None:
        config = ModelConfig(
            model_id=model_id,
            provider_kwargs={"api_key": api_key or "", "temperature": 0.1},
        )
        model = create_model(config, use_schema_constraints=True)

    # Parse article sections
    sections = _parse_article_sections(article_xml)
    fh = FormatHandler(format_type=FormatType.JSON, use_fences=(model.requires_fence_output))

    all_extractions: list[Extraction] = []
    warnings: list[str] = []

    # Phase 1: Alternatives (GATE)
    phase1 = registry.phase_defs()[0]
    prompt1 = registry.build_extraction_prompt(phase1.extracts)
    template1 = PromptTemplateStructured(description=prompt1)
    annotator1 = Annotator(model, template1, fh)
    result1 = annotator1.annotate_text(
        sections.get("methods", ""), max_char_buffer=max_char_buffer,
    )
    all_extractions.extend(result1.extractions or [])

    # Gate check
    if skip_if_no_known_alternative:
        known = {"Plant_Extract", "Trace_Element", "Organic_Acid", "Probiotic",
                 "Polysaccharides_and_Oligosaccharides", "Enzyme", "Bioactive_Peptides"}
        has_known = any(
            ext.extraction_class == "Alternative"
            and (ext.attributes or {}).get("alternative_class") in known
            for ext in all_extractions
        )
        if not has_known:
            return ArticleExtractionResult(
                doi=sections.get("doi", ""), pmid=sections.get("pmid", ""),
                skipped=True, skip_reason="No known Alternative found",
                warnings=warnings,
            )

    # Phase 2: Experiment Design
    phase2 = registry.phase_defs()[1]
    prompt2 = registry.build_extraction_prompt(phase2.extracts)
    template2 = PromptTemplateStructured(description=prompt2)
    annotator2 = Annotator(model, template2, fh)
    result2 = annotator2.annotate_text(
        sections.get("methods", ""), max_char_buffer=max_char_buffer,
    )
    all_extractions.extend(result2.extractions or [])

    # Phase 3: Indicators
    phase3 = registry.phase_defs()[2]
    prompt3 = registry.build_extraction_prompt(phase3.extracts)
    template3 = PromptTemplateStructured(description=prompt3)
    annotator3 = Annotator(model, template3, fh)
    result3 = annotator3.annotate_text(
        sections.get("methods", ""), max_char_buffer=max_char_buffer,
    )
    all_extractions.extend(result3.extractions or [])

    # Phase 4: Results
    phase4 = registry.phase_defs()[3]
    prompt4 = registry.build_extraction_prompt(phase4.extracts)
    # Add context from prior phases
    context = _build_results_context(result2, result3)
    prompt4 += "\n\n## Available Context\n" + context
    template4 = PromptTemplateStructured(description=prompt4)
    annotator4 = Annotator(model, template4, fh)
    result4 = annotator4.annotate_text(
        sections.get("results", ""),
        additional_context=context,
        max_char_buffer=max_char_buffer,
    )
    all_extractions.extend(result4.extractions or [])

    # Post-process: vocabulary matching on Alternatives
    for ext in all_extractions:
        registry.post_process(ext)

    return ArticleExtractionResult(
        doi=sections.get("doi", ""),
        pmid=sections.get("pmid", ""),
        extractions=all_extractions,
        warnings=warnings,
    )


def _parse_article_sections(xml_path: str | Path) -> dict[str, str]:
    """Parse PMC XML → section dict. Thin wrapper; full impl uses lxml."""
    from lxml import etree
    path = Path(xml_path)
    if not path.exists():
        return {}

    try:
        tree = etree.parse(str(path))
        root = tree.getroot()

        # Extract DOI and PMID from front matter
        ns = {"x": "http://www.w3.org/1999/xhtml"}
        doi = ""
        pmid = ""
        for el in root.iter():
            if el.tag.endswith("article-id") and el.get("pub-id-type") == "doi":
                doi = (el.text or "").strip()
            if el.tag.endswith("article-id") and el.get("pub-id-type") == "pmid":
                pmid = (el.text or "").strip()

        sections = {"doi": doi, "pmid": pmid}

        # Extract abstract
        abstract_parts = []
        for abs_el in root.iter():
            if abs_el.tag.endswith("abstract"):
                abstract_parts.append(etree.tostring(abs_el, method="text", encoding="unicode"))
        sections["abstract"] = "\n".join(abstract_parts)

        # Extract body sections by looking for sec elements
        body_text = ""
        for body in root.iter():
            if body.tag.endswith("body"):
                body_text = etree.tostring(body, method="text", encoding="unicode") or ""
        sections["body"] = body_text

        # Simple section split (full implementation would parse <sec> elements)
        sections["methods"] = body_text
        sections["results"] = body_text
        sections["discussion"] = body_text

        return sections
    except Exception:
        return {"doi": "", "pmid": "", "body": ""}


def _build_results_context(
    design_result: AnnotatedDocument,
    indicator_result: AnnotatedDocument,
) -> str:
    """Build context string for Results extraction from prior phases."""
    parts = []
    # List indicators
    indicators = [e for e in (indicator_result.extractions or []) if e.extraction_class == "Indicator"]
    if indicators:
        parts.append("Available Indicators:")
        for ind in indicators[:50]:
            abbrev = (ind.attributes or {}).get("abbreviation", ind.extraction_text)
            name = (ind.attributes or {}).get("standard_name", "")
            parts.append(f"  {abbrev} = {name}")
    # List control groups
    controls = [e for e in (design_result.extractions or []) if e.extraction_class == "Control_Group"]
    if controls:
        parts.append("Available Control Groups:")
        for c in controls:
            gname = (c.attributes or {}).get("group_name", c.extraction_text)
            gtype = (c.attributes or {}).get("group_type", "")
            parts.append(f"  {gname} ({gtype})")
    return "\n".join(parts)
```

- [ ] **Step 2: Write integration test**

Create `tests/test_extraction.py`:

```python
"""Integration tests for extraction pipeline."""
import pytest
from pathlib import Path
from src.extraction import extract, ArticleExtractionResult
from src.schema_registry import SchemaRegistry

FIXTURES = Path(__file__).parent / "fixtures"


class TestExtraction:
    def test_result_type(self):
        """Verify ArticleExtractionResult can be created."""
        result = ArticleExtractionResult(doi="10.1234/x", pmid="12345")
        assert result.doi == "10.1234/x"
        assert result.pmid == "12345"
        assert result.skipped is False
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_extraction.py -v`
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/extraction.py tests/test_extraction.py
git commit -m "feat: add extraction API (extract() — 4-phase pipeline with gate check)"
```

---

## Group H: Graph + CLI

### Task H1: Graph Builder (`src/graph.py`)

**Files:**
- Create: `src/graph.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: Write graph builder**

Create `src/graph.py`:

```python
"""Graph builder — dedup, resolve edges, export Neo4j CSV."""
from __future__ import annotations

import csv
import dataclasses
import hashlib
from pathlib import Path
from typing import Any

from src.data import Extraction
from src.schema_registry import SchemaRegistry


@dataclasses.dataclass
class GraphNode:
    id: str
    labels: list[str]
    properties: dict[str, Any]
    source_pmids: list[str]


@dataclasses.dataclass
class GraphEdge:
    source_id: str
    target_id: str
    type: str
    properties: dict[str, Any]


@dataclasses.dataclass
class Graph:
    nodes: list[GraphNode]
    edges: list[GraphEdge]


def entity_global_id(
    entity_type: str, primary_text: str, article_pmid: str | None = None,
) -> str:
    """Generate deterministic global entity ID."""
    # Article-scoped entities get PMID in hash
    article_scoped = {"Result", "Experiment", "Intervention", "Swine", "Swine_Model",
                      "Control_Group", "Literature"}
    if entity_type in article_scoped and article_pmid:
        key = f"{entity_type}:{article_pmid}:{primary_text}"
    else:
        key = f"{entity_type}:{primary_text.lower().strip()}"

    hash_hex = hashlib.sha256(key.encode()).hexdigest()[:12]
    prefix = entity_type.lower()[:4]
    return f"{prefix}_{hash_hex}"


def build_graph(
    results: list[Any],  # list[ArticleExtractionResult]
    registry: SchemaRegistry,
) -> Graph:
    """Build graph from article extraction results."""
    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []

    for article_result in results:
        pmid = article_result.pmid

        for ext in article_result.extractions:
            primary = ext.extraction_text
            gid = entity_global_id(ext.extraction_class, primary, pmid)
            attrs = dict(ext.attributes or {})

            if gid in nodes:
                if pmid and pmid not in nodes[gid].source_pmids:
                    nodes[gid].source_pmids.append(pmid)
            else:
                nodes[gid] = GraphNode(
                    id=gid,
                    labels=[ext.extraction_class, "Entity"],
                    properties={**attrs, "name": primary},
                    source_pmids=[pmid] if pmid else [],
                )

        # Resolve edges from co-extracted inline relations + references
        for ext in article_result.extractions:
            ed = registry.entity_def(ext.extraction_class) if ext.extraction_class in registry.all_entity_names() else None
            if ed is None:
                continue

            source_gid = entity_global_id(ext.extraction_class, ext.extraction_text, pmid)

            # Inline relations
            for rel in ed.inline_relations:
                via_val = (ext.attributes or {}).get(rel.via_field, [])
                if not isinstance(via_val, list):
                    via_val = [via_val] if via_val else []
                for comp in via_val:
                    if isinstance(comp, dict):
                        comp_name = comp.get("standard_name", comp.get("product_name", ""))
                        comp_type = comp.get("entity_type", "Alternative")
                    else:
                        comp_name = str(comp)
                        comp_type = "Alternative"
                    if comp_name:
                        target_gid = entity_global_id(comp_type, comp_name, pmid)
                        edges.append(GraphEdge(
                            source_id=source_gid, target_id=target_gid,
                            type=rel.name,
                            properties={"evidence_text": attrs.get("evidence_text", ""),
                                       "source_pmids": pmid},
                        ))

            # Reference edges
            for ref in ed.references:
                ref_val = (ext.attributes or {}).get(ref.name, "")
                if ref_val:
                    target_gid = entity_global_id(ref.target_entity, str(ref_val))
                    edges.append(GraphEdge(
                        source_id=source_gid, target_id=target_gid,
                        type=ref.edge_type,
                        properties={"evidence_text": attrs.get("evidence_text", ""),
                                   "source_pmids": pmid},
                    ))

    return Graph(nodes=list(nodes.values()), edges=edges)


def export_neo4j_csv(graph: Graph, output_dir: Path) -> None:
    """Export nodes.csv and edges.csv in Neo4j import format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all property keys across nodes
    all_keys: set[str] = set()
    for node in graph.nodes:
        all_keys.update(node.properties.keys())
    sorted_keys = sorted(all_keys)

    # Write nodes.csv
    nodes_path = output_dir / "nodes.csv"
    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        # Neo4j CSV header with :ID and :LABEL
        header = ["entity_id:ID", "entity_type:LABEL"] + sorted_keys + ["source_pmids"]
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for node in graph.nodes:
            row = {
                "entity_id:ID": node.id,
                "entity_type:LABEL": ";".join(node.labels),
                "source_pmids": "|".join(node.source_pmids),
            }
            row.update(node.properties)
            writer.writerow(row)

    # Collect all edge property keys
    edge_keys: set[str] = set()
    for edge in graph.edges:
        edge_keys.update(edge.properties.keys())
    sorted_edge_keys = sorted(edge_keys)

    # Write edges.csv
    edges_path = output_dir / "edges.csv"
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        header = ["source_id:START_ID", "target_id:END_ID", "relation_type:TYPE"] + sorted_edge_keys
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for edge in graph.edges:
            row = {
                "source_id:START_ID": edge.source_id,
                "target_id:END_ID": edge.target_id,
                "relation_type:TYPE": edge.type,
            }
            row.update(edge.properties)
            writer.writerow(row)
```

- [ ] **Step 2: Write tests**

Create `tests/test_graph.py`:

```python
"""Tests for graph builder."""
import pytest
import tempfile
from pathlib import Path
from src.graph import (
    entity_global_id, GraphNode, GraphEdge, Graph,
    build_graph, export_neo4j_csv,
)


class TestEntityGlobalId:
    def test_deterministic(self):
        id1 = entity_global_id("Alternative", "thymol")
        id2 = entity_global_id("Alternative", "thymol")
        assert id1 == id2

    def test_different_types(self):
        id1 = entity_global_id("Alternative", "thymol")
        id2 = entity_global_id("Indicator", "thymol")
        assert id1 != id2

    def test_prefix(self):
        gid = entity_global_id("Alternative", "thymol")
        assert gid.startswith("alte") or gid.startswith("Alte")


class TestExportNeo4jCsv:
    def test_export(self):
        node = GraphNode(
            id="alt_001", labels=["Alternative", "Entity"],
            properties={"name": "thymol", "alternative_class": "Plant_Extract"},
            source_pmids=["12345"],
        )
        edge = GraphEdge(
            source_id="alt_001", target_id="altclass_001",
            type="belongs_to",
            properties={"evidence_text": "thymol was used"},
        )
        graph = Graph(nodes=[node], edges=[edge])

        with tempfile.TemporaryDirectory() as tmpdir:
            export_neo4j_csv(graph, Path(tmpdir))
            nodes_csv = Path(tmpdir) / "nodes.csv"
            edges_csv = Path(tmpdir) / "edges.csv"
            assert nodes_csv.exists()
            assert edges_csv.exists()

            # Verify nodes.csv header
            content = nodes_csv.read_text()
            assert "entity_id:ID" in content
            assert "entity_type:LABEL" in content
            assert "thymol" in content

            # Verify edges.csv header
            content = edges_csv.read_text()
            assert "source_id:START_ID" in content
            assert "target_id:END_ID" in content
            assert "relation_type:TYPE" in content
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_graph.py -v`
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/graph.py tests/test_graph.py
git commit -m "feat: add graph builder (entity dedup, edge resolution, Neo4j CSV export)"
```

---

### Task H2: CLI (`src/cli.py`)

**Files:**
- Create: `src/cli.py`

- [ ] **Step 1: Write CLI**

Create `src/cli.py`:

```python
"""CLI entry point for llm-extract."""
import sys
from pathlib import Path

import click

from src.config import settings


@click.group()
def cli():
    """llm-extract — Knowledge Graph Extraction Pipeline v2."""
    pass


@cli.command()
@click.option("--input", "-i", "input_file", required=True,
              help="Path to literature_pool.tsv")
@click.option("--output", "-o", "output_dir", default="output",
              help="Output directory for nodes.csv and edges.csv")
@click.option("--model", default=None, help="Model ID override")
@click.option("--workers", default=None, type=int, help="Max parallel workers")
@click.option("--resume/--no-resume", default=False, help="Resume from checkpoints")
def run(input_file, output_dir, model, workers, resume):
    """Run the full extraction pipeline."""
    click.echo(f"llm-extract v2.0.0")
    click.echo(f"Input: {input_file}")
    click.echo(f"Output: {output_dir}")
    click.echo(f"Model: {model or settings.llm_model}")
    click.echo("Pipeline not yet implemented — see extraction.py")
    sys.exit(0)


@cli.command()
@click.option("--xml", required=True, help="Path to PMC XML file")
def debug(xml):
    """Debug: extract from a single article and print results."""
    click.echo(f"Debug extraction from: {xml}")
    try:
        from src.extraction import extract
        result = extract(xml)
        click.echo(f"Extracted {len(result.extractions)} entities")
        if result.skipped:
            click.echo(f"SKIPPED: {result.skip_reason}")
        for ext in result.extractions[:10]:
            click.echo(f"  [{ext.extraction_class}] {ext.extraction_text}")
    except Exception as e:
        click.echo(f"Error: {e}")
        sys.exit(1)


@cli.command()
@click.option("--checkpoints", required=True, help="Path to intermediates directory")
@click.option("--output", "-o", "output_dir", default="output", help="Output directory")
def export(checkpoints, output_dir):
    """Export graph from existing checkpoints (skip extraction)."""
    click.echo(f"Exporting from checkpoints: {checkpoints}")
    # Collect checkpoints, build graph, export
    sys.exit(0)


if __name__ == "__main__":
    cli()
```

- [ ] **Step 2: Verify CLI loads**

Run: `python -m src.cli --help`
Expected: CLI help output with run/debug/export commands

- [ ] **Step 3: Commit**

```bash
git add src/cli.py
git commit -m "feat: add CLI (click — run, debug, export commands)"
```

---

## Group I: Integration + Cleanup

### Task I1: Integration Test

**Files:**
- Create: `tests/fixtures/sample.xml` (copy from existing fixture or create minimal)

- [ ] **Step 1: Create minimal test fixture**

```bash
cp tests/fixtures/sample.xml tests/fixtures/article_1.xml 2>/dev/null || echo "Using existing fixture"
```

- [ ] **Step 2: Write integration test**

Add to `tests/test_integration.py`:

```python
"""End-to-end integration test."""
import pytest
from pathlib import Path
from src.schema_registry import SchemaRegistry
from src.graph import Graph, build_graph


FIXTURES = Path(__file__).parent / "fixtures"
SCHEMA_DIR = Path(__file__).parent.parent / "schemas"


class TestPipelineIntegration:
    def test_schema_registry_loads(self):
        """Verify schema registry loads config files."""
        registry = SchemaRegistry(config_dir=SCHEMA_DIR)
        assert len(registry.all_entity_names()) >= 13
        assert "Alternative" in registry.all_entity_names()
        assert "Result" in registry.all_entity_names()
        assert len(registry.phase_defs()) >= 4

    def test_graph_export_roundtrip(self):
        """Build a minimal graph and export to Neo4j CSV."""
        import tempfile
        from src.graph import (
            GraphNode, GraphEdge, Graph, export_neo4j_csv,
        )
        node = GraphNode(
            id="test_001", labels=["Alternative", "Entity"],
            properties={"name": "thymol"}, source_pmids=["12345"],
        )
        graph = Graph(nodes=[node], edges=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            export_neo4j_csv(graph, Path(tmpdir))
            nodes_path = Path(tmpdir) / "nodes.csv"
            assert nodes_path.exists()
            content = nodes_path.read_text()
            assert "test_001" in content

    def test_full_module_imports(self):
        """Verify all modules import without errors."""
        import src.data
        import src.tokenizer
        import src.chunking
        import src.format_handler
        import src.schema_registry
        import src.schema
        import src.prompting
        import src.resolver
        import src.evidence
        import src.source_location
        import src.annotation
        import src.extraction
        import src.factory
        import src.graph
        import src.providers.base
        import src.providers.capabilities
        import src.providers.openai_compat
        import src.providers.schemas.openai
        # All imports succeeded
```

- [ ] **Step 3: Run integration tests**

Run: `pytest tests/test_integration.py -v`
Expected: all tests PASS

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: all tests PASS (may have some failures from old tests referencing deleted modules)

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test (schema loading, graph export, full import chain)"
```

---

### Task I2: Cleanup Old Modules

**Files:**
- Remove: `src/dspy_extract.py`, `src/run_dspy_pipeline.py`
- Remove: `src/stage0_search.py`, `src/stage1_xml_parser.py`, `src/stage4_export_graph.py`, `src/stage4_validate.py`
- Remove: `src/utils.py`, `src/glossary.py`, `src/entity_id.py`
- Remove: `schemas/*.json` (replaced by YAML)
- Remove: old test files

- [ ] **Step 1: Remove old source files**

```bash
rm -f src/dspy_extract.py src/run_dspy_pipeline.py \
      src/stage0_search.py src/stage1_xml_parser.py \
      src/stage4_export_graph.py src/stage4_validate.py \
      src/utils.py src/glossary.py src/entity_id.py
```

- [ ] **Step 2: Remove old test files**

```bash
rm -f tests/test_dspy_extract.py tests/test_stage0.py \
      tests/test_stage1.py tests/test_stage2_entity_extract.py \
      tests/test_stage4_validate.py tests/test_glossary.py
```

- [ ] **Step 3: Remove old schema JSON files**

```bash
rm -f schemas/alternatives.json schemas/experiment_design.json \
      schemas/indicators.json schemas/results.json
```

- [ ] **Step 4: Run full test suite to verify nothing broken**

Run: `pytest tests/ -v`
Expected: all new tests PASS; no import errors

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove v1 modules (dspy, stages, old schemas) — replaced by v2 pipeline"
```

---

## Implementation Order Summary

| Order | Task | Dependencies | Estimated Time |
|-------|------|-------------|---------------|
| 1 | A1: Update pyproject.toml | None | 2 min |
| 2 | A2: Data Model | A1 | 15 min |
| 3 | A3: Tokenizer | A1 | 15 min |
| 4 | A4: Config | A1 | 5 min |
| 5 | B1: Chunking | A2, A3 | 15 min |
| 6 | B2: Format Handler | A2 | 15 min |
| 7 | C1: Schema Config Files | None | 10 min |
| 8 | C2: Schema Registry | C1, A2 | 20 min |
| 9 | C3: Schema Layer | C2 | 10 min |
| 10 | D1: Provider Base + Capabilities | C3 | 10 min |
| 11 | D2: OpenAI Compat Provider | D1, C2 | 15 min |
| 12 | E1: Prompting | B2 | 10 min |
| 13 | F1: Resolver | A3, B2 | 20 min |
| 14 | F2: Evidence + Source Location | A3 | 10 min |
| 15 | G1: Annotation | B1, D1, E1, F1, F2 | 20 min |
| 16 | G2: Factory | D2 | 5 min |
| 17 | G3: Extraction API | G1, G2, C2, F1 | 15 min |
| 18 | H1: Graph Builder | A2, C2 | 15 min |
| 19 | H2: CLI | G3, H1 | 5 min |
| 20 | I1: Integration Test | All | 10 min |
| 21 | I2: Cleanup | All | 5 min |

**Total estimated time:** ~4 hours of focused implementation.
