"""Core data types for the extraction pipeline.

Matches the original langextract data model:
- Extraction has both char_interval and token_interval for alignment
- Document caches tokenized_text
- AnnotatedDocument carries the full source text for evidence derivation
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.tokenizer import TokenInterval, TokenizedText, RegexTokenizer, Tokenizer


class FormatType(str, Enum):
    JSON = "json"
    YAML = "yaml"


class AlignmentStatus(str, Enum):
    MATCH_EXACT = "match_exact"
    MATCH_GREATER = "match_greater"
    MATCH_LESSER = "match_lesser"
    MATCH_FUZZY = "match_fuzzy"


@dataclass
class CharInterval:
    start_pos: int | None = None
    end_pos: int | None = None


@dataclass(init=False)
class Extraction:
    """Represents an extraction from text.

    Attributes:
        extraction_class: The entity type (e.g. "Chemical", "Measurement").
        extraction_text: The primary text value returned by the LLM.
        char_interval: Character position in the source document (set by alignment).
        token_interval: Token position in the source document (set by alignment).
        alignment_status: How the extraction was aligned (exact, fuzzy, etc.).
        extraction_index: Order index within the extraction list.
        group_index: Group index in the model output.
        description: Optional description.
        attributes: Additional key-value fields (LLM output + post-processing).
    """

    extraction_class: str
    extraction_text: str
    char_interval: CharInterval | None = None
    alignment_status: AlignmentStatus | None = None
    extraction_index: int | None = None
    group_index: int | None = None
    description: str | None = None
    attributes: dict[str, Any] | None = None
    evidence_text: str = ""  # engineering-derived verbatim source text at char_interval
    source_location: str = ""  # engineering-derived section/table/figure reference
    _token_interval: TokenInterval | None = field(default=None, repr=False)

    def __init__(
        self,
        extraction_class: str,
        extraction_text: str,
        char_interval: CharInterval | None = None,
        alignment_status: AlignmentStatus | None = None,
        extraction_index: int | None = None,
        group_index: int | None = None,
        description: str | None = None,
        attributes: dict[str, Any] | None = None,
        token_interval: TokenInterval | None = None,
        evidence_text: str = "",
        source_location: str = "",
    ) -> None:
        self.extraction_class = extraction_class
        self.extraction_text = extraction_text
        self.char_interval = char_interval
        self.alignment_status = alignment_status
        self.extraction_index = extraction_index
        self.group_index = group_index
        self.description = description
        self.attributes = attributes
        self._token_interval = token_interval
        self.evidence_text = evidence_text
        self.source_location = source_location

    @property
    def token_interval(self) -> TokenInterval | None:
        return self._token_interval

    @token_interval.setter
    def token_interval(self, value: TokenInterval | None) -> None:
        self._token_interval = value


@dataclass(init=False)
class Document:
    """Document class for annotating documents.

    Caches tokenized_text (like the original langextract) so that downstream
    chunking and alignment can reuse the same tokenization.
    """

    text: str
    additional_context: str | None = None
    _document_id: str = field(init=False, repr=False)
    _tokenized_text: TokenizedText | None = field(init=False, default=None, repr=False)

    def __init__(
        self,
        text: str,
        additional_context: str | None = None,
        document_id: str | None = None,
    ) -> None:
        self.text = text
        self.additional_context = additional_context
        if document_id is not None:
            self._document_id = document_id
        else:
            self._document_id = "doc_" + uuid.uuid4().hex[:8]

    @property
    def document_id(self) -> str:
        return self._document_id

    @document_id.setter
    def document_id(self, value: str) -> None:
        self._document_id = value

    @property
    def tokenized_text(self) -> TokenizedText:
        """Lazily tokenize and cache the document text."""
        if self._tokenized_text is None:
            self._tokenized_text = RegexTokenizer().tokenize(self.text)
        return self._tokenized_text

    @tokenized_text.setter
    def tokenized_text(self, value: TokenizedText) -> None:
        self._tokenized_text = value

    def with_additional_context(self, context: str) -> Document:
        """Return a copy with *additional_context* overridden.

        Preserves cached tokenization to avoid redundant re-tokenization.
        """
        new_doc = Document(
            text=self.text,
            additional_context=context,
            document_id=self._document_id,
        )
        if self._tokenized_text is not None:
            new_doc.tokenized_text = self._tokenized_text
        return new_doc


@dataclass
class ExampleData:
    text: str
    extractions: list[Extraction] = field(default_factory=list)


@dataclass(init=False)
class AnnotatedDocument:
    """Result of annotating a document.

    Carries the full source text so that evidence_text can be derived
    from char_intervals without needing to pass text separately.
    """

    extractions: list[Extraction] | None = None
    text: str | None = None
    _document_id: str = field(init=False, repr=False)
    _tokenized_text: TokenizedText | None = field(init=False, default=None, repr=False)

    def __init__(
        self,
        extractions: list[Extraction] | None = None,
        text: str | None = None,
        document_id: str | None = None,
    ) -> None:
        self.extractions = extractions
        self.text = text
        if document_id is not None:
            self._document_id = document_id
        else:
            self._document_id = "doc_" + uuid.uuid4().hex[:8]

    @property
    def document_id(self) -> str:
        return self._document_id

    @property
    def tokenized_text(self) -> TokenizedText | None:
        """Lazily tokenize the document text if available."""
        if self._tokenized_text is None and self.text is not None:
            self._tokenized_text = RegexTokenizer().tokenize(self.text)
        return self._tokenized_text

    @tokenized_text.setter
    def tokenized_text(self, value: TokenizedText) -> None:
        self._tokenized_text = value
