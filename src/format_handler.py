"""Centralized format handling for JSON/YAML prompts and parsing."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import yaml

from src.data import Extraction, FormatType


EXTRACTIONS_KEY = "extractions"
ATTRIBUTE_SUFFIX = "_attributes"
_FENCE_RE = re.compile(
    r"```(?P<lang>[A-Za-z0-9_+-]+)?(?:\s*\n)?(?P<body>[\s\S]*?)```", re.MULTILINE
)
_THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)

ExtractionValueType: TypeAlias = Any


class FormatHandler:
    """Handles formatting and parsing of extraction prompts in JSON or YAML."""

    def __init__(
        self,
        format_type: FormatType = FormatType.JSON,
        use_wrapper: bool = True,
        wrapper_key: str | None = None,
        use_fences: bool = True,
        attribute_suffix: str = "_attributes",
        strict_fences: bool = False,
        allow_top_level_list: bool = True,
    ) -> None:
        self.format_type = format_type
        self.use_wrapper = use_wrapper
        if wrapper_key is not None:
            self.wrapper_key = wrapper_key
        elif use_wrapper:
            self.wrapper_key = EXTRACTIONS_KEY
        else:
            self.wrapper_key = None
        self.use_fences = use_fences
        self.attribute_suffix = attribute_suffix
        self.strict_fences = strict_fences
        self.allow_top_level_list = allow_top_level_list

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_fences(self, content: str) -> str:
        """Wrap content in code fences with the appropriate language tag."""
        lang = "json" if self.format_type == FormatType.JSON else "yaml"
        return f"```{lang}\n{content}\n```"

    def _extract_content(self, text: str) -> str:
        """Extract content from fenced code blocks.

        If no fence is found, returns the original text.  When
        *strict_fences* is True, validates that the language tag matches
        the expected format type.
        """
        match = _FENCE_RE.search(text)
        if not match:
            return text

        lang = match.group("lang")
        body = match.group("body")

        if self.strict_fences:
            valid_langs = {"json"} if self.format_type == FormatType.JSON else {"yaml", "yml"}
            if lang is not None and lang.lower() not in valid_langs:
                raise ValueError(
                    f"Invalid fence language tag '{lang}' "
                    f"for format type {self.format_type.value}"
                )

        return body.strip() if body else ""

    def _parse_content(self, content: str) -> Any:
        """Parse *content* as JSON or YAML."""
        stripped = content.strip()
        if not stripped:
            raise ValueError("Empty content")
        if self.format_type == FormatType.JSON:
            return json.loads(stripped)
        return yaml.safe_load(stripped)

    def _parse_with_fallback(self, content: str, strict: bool | None = None) -> Any:
        """Parse content, falling back to stripping <think>…</think> tags.

        Tries parsing first; if parsing fails *and* think tags are
        present, strips the tags and retries once.
        """
        try:
            return self._parse_content(content)
        except (json.JSONDecodeError, yaml.YAMLError, ValueError):
            if _THINK_TAG_RE.search(content):
                stripped = _THINK_TAG_RE.sub("", content).strip()
                try:
                    return self._parse_content(stripped)
                except (json.JSONDecodeError, yaml.YAMLError, ValueError):
                    raise ValueError(
                        "Failed to parse output after stripping think tags"
                    ) from None
            raise ValueError("Failed to parse output") from None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format_extraction_example(self, extractions: list[Extraction]) -> str:
        """Build a formatted example string from a list of *Extraction* objects."""
        extraction_list: list[dict[str, Any]] = []
        for ext in extractions:
            item: dict[str, Any] = {ext.extraction_class: ext.extraction_text}
            if ext.attributes:
                attrs_key = f"{ext.extraction_class}{self.attribute_suffix}"
                item[attrs_key] = ext.attributes
            extraction_list.append(item)

        if self.use_wrapper and self.wrapper_key is not None:
            output: Any = {self.wrapper_key: extraction_list}
        else:
            output = extraction_list

        if self.format_type == FormatType.JSON:
            formatted = json.dumps(output, indent=2, ensure_ascii=False)
        else:
            formatted = yaml.safe_dump(output, indent=2, allow_unicode=True)

        if self.use_fences:
            formatted = self._add_fences(formatted)

        return formatted

    def parse_output(
        self, text: str, *, strict: bool | None = None
    ) -> Sequence[Mapping[str, ExtractionValueType]]:
        """Parse model output into a sequence of extraction mappings.

        Handles fence extraction, think-tag stripping, and structure
        validation.
        """
        if not text or not text.strip():
            raise ValueError("Empty output text")

        # 1. Extract content from fences (or just strip)
        if self.use_fences:
            content = self._extract_content(text)
        else:
            content = text.strip()

        # 2. Parse
        parsed = self._parse_with_fallback(content, strict=strict)

        # 3. Unwrap / normalise to a list of dicts
        if isinstance(parsed, dict):
            if self.use_wrapper and self.wrapper_key is not None:
                if self.wrapper_key in parsed:
                    parsed = parsed[self.wrapper_key]
                else:
                    # Treat the dict itself as a single extraction
                    parsed = [parsed]
            else:
                parsed = [parsed]
        elif isinstance(parsed, list):
            if self.use_wrapper and not self.allow_top_level_list:
                raise ValueError(
                    "Expected a dict with wrapper key but got a list"
                )
        else:
            raise ValueError(
                f"Expected a dict or list, got {type(parsed).__name__}"
            )

        # 4. Validate structure
        validated: list[Mapping[str, ExtractionValueType]] = []
        for item in parsed:
            if not isinstance(item, dict):
                raise ValueError(
                    f"Expected each extraction to be a dict, "
                    f"got {type(item).__name__}"
                )
            for key in item:
                if not isinstance(key, str):
                    raise ValueError(
                        f"Expected string keys, got {type(key).__name__}"
                    )
            validated.append(item)

        return validated
