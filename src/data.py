"""Core data types for the extraction pipeline."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
        char_interval: CharInterval | None = None,
        alignment_status: AlignmentStatus | None = None,
        extraction_index: int | None = None,
        group_index: int | None = None,
        description: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.extraction_class = extraction_class
        self.extraction_text = extraction_text
        self.char_interval = char_interval
        self.alignment_status = alignment_status
        self.extraction_index = extraction_index
        self.group_index = group_index
        self.description = description
        self.attributes = attributes


@dataclass(init=False)
class Document:
    text: str
    additional_context: str | None = None
    _document_id: str = field(init=False, repr=False)

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

    def with_additional_context(self, context: str) -> Document:
        return Document(
            text=self.text,
            additional_context=context,
            document_id=self._document_id,
        )


@dataclass
class ExampleData:
    text: str
    extractions: list[Extraction] = field(default_factory=list)


@dataclass(init=False)
class AnnotatedDocument:
    extractions: list[Extraction] | None = None
    text: str | None = None
    _document_id: str = field(init=False, repr=False)

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
