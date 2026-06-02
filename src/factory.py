"""Factory for creating language model instances."""
from __future__ import annotations

import dataclasses
from typing import Any

from src.config import settings
from src.providers.base import BaseLanguageModel


@dataclasses.dataclass(slots=True, frozen=True)
class ModelConfig:
    model_id: str | None = None
    provider: str | None = None
    provider_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)


def create_model(
    config: ModelConfig | None = None,
    examples=None,
    use_schema_constraints: bool = False,
    fence_output: bool | None = None,
) -> BaseLanguageModel:
    """Create a language model from config (or global settings).

    All defaults come from ``src.config.settings``.
    """
    from src.providers.openai_compat import OpenAICompatProvider

    if config is None:
        config = ModelConfig()

    model_id = config.model_id or settings.llm_model
    api_key = config.provider_kwargs.get("api_key") or settings.llm_api_key
    base_url = config.provider_kwargs.get("base_url") or settings.llm_base_url

    kwargs = {
        "model_id": model_id,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": settings.llm_temperature,
    }
    # Merge any extra provider kwargs from config
    kwargs.update({k: v for k, v in config.provider_kwargs.items()
                   if k not in ("api_key", "base_url")})

    model = OpenAICompatProvider(**kwargs)

    if use_schema_constraints and examples:
        schema_class = model.get_schema_class()
        if schema_class is not None and hasattr(schema_class, "from_examples"):
            model.apply_schema(schema_class.from_examples(examples))

    model.set_fence_output(fence_output)
    return model
