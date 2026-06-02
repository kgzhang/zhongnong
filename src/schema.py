"""Core schema abstractions."""
from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

from src.data import FormatType, ExampleData


class BaseSchema(abc.ABC):
    """Abstract base for output format schemas."""

    @classmethod
    @abc.abstractmethod
    def from_examples(
        cls,
        examples_data: Sequence[ExampleData],
        attribute_suffix: str = "_attributes",
    ) -> "BaseSchema":
        ...

    @abc.abstractmethod
    def to_provider_config(self) -> dict[str, Any]:
        ...

    @property
    @abc.abstractmethod
    def requires_raw_output(self) -> bool:
        ...

    def validate_format(self, format_handler: Any) -> None:
        pass

    def sync_with_provider_kwargs(self, kwargs: dict[str, Any]) -> None:
        pass


class FormatModeSchema(BaseSchema):
    """Schema that delegates everything to provider-level format param."""

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
            self.format_type = FormatType.JSON if self._format == "json" else FormatType.YAML
