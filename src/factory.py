"""Factory for creating language model instances."""
from __future__ import annotations

import dataclasses
import os
from typing import Any

from src.providers.base import BaseLanguageModel
from src.providers.openai_compat import OpenAICompatProvider


@dataclasses.dataclass(slots=True, frozen=True)
class ModelConfig:
    model_id: str | None = None
    provider: str | None = None
    provider_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)


def create_model(
    config: ModelConfig,
    examples=None,
    use_schema_constraints: bool = False,
    fence_output: bool | None = None,
) -> BaseLanguageModel:
    model_id = config.model_id or "deepseek-chat"
    kwargs = dict(config.provider_kwargs)
    kwargs = _kwargs_with_environment_defaults(model_id, kwargs)
    kwargs["model_id"] = model_id
    model = OpenAICompatProvider(**kwargs)
    if use_schema_constraints and examples:
        schema_class = model.get_schema_class()
        if schema_class is not None and hasattr(schema_class, "from_examples"):
            model.apply_schema(schema_class.from_examples(examples))
    model.set_fence_output(fence_output)
    return model


def _kwargs_with_environment_defaults(model_id: str, kwargs: dict) -> dict:
    resolved = dict(kwargs)
    if "api_key" not in resolved:
        resolved["api_key"] = os.getenv(
            "ZN_LLM_API_KEY", os.getenv("DEEPSEEK_API_KEY", "")
        )
    if "base_url" not in resolved:
        if "deepseek" in model_id.lower():
            resolved["base_url"] = os.getenv(
                "ZN_LLM_BASE_URL", "https://api.deepseek.com/v1"
            )
    return resolved
