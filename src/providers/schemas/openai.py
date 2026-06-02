"""OpenAI provider schema — generates response_format json_schema from registry."""
from __future__ import annotations
import copy, dataclasses
from typing import Any
from src.data import FormatType
from src.format_handler import FormatHandler
from src.schema import BaseSchema

DEFAULT_SCHEMA_NAME = "zhongnong_extraction"

@dataclasses.dataclass(frozen=True)
class OpenAISchema(BaseSchema):
    schema_dict: dict[str, Any]
    schema_name: str = DEFAULT_SCHEMA_NAME
    strict: bool = True

    def __post_init__(self):
        object.__setattr__(self, "schema_dict", copy.deepcopy(self.schema_dict))

    @property
    def response_format(self) -> dict[str, Any]:
        return {"type": "json_schema", "json_schema": {
            "name": self.schema_name,
            "schema": copy.deepcopy(self.schema_dict),
            "strict": self.strict,
        }}

    @classmethod
    def from_registry(cls, registry, entity_names: list[str], strict: bool = True) -> "OpenAISchema":
        schema_dict = registry.generate_json_schema(entity_names, strict=strict)
        return cls(schema_dict=schema_dict, strict=strict)

    @classmethod
    def from_examples(cls, examples_data, attribute_suffix="_attributes") -> "OpenAISchema":
        return cls(schema_dict={
            "type": "object", "properties": {"extractions": {"type": "array", "items": {"type": "object"}}},
            "required": ["extractions"], "additionalProperties": False,
        }, strict=False)

    def to_provider_config(self) -> dict[str, Any]: return {}

    @property
    def requires_raw_output(self) -> bool: return True

    def validate_format(self, format_handler: FormatHandler) -> None:
        if format_handler.format_type != FormatType.JSON:
            raise ValueError("OpenAI structured output only supports JSON format.")
