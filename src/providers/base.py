"""Base language model interface."""
from __future__ import annotations
import abc, dataclasses
from collections.abc import Iterator, Sequence
from typing import Any
from src.schema import BaseSchema

@dataclasses.dataclass(frozen=True)
class ScoredOutput:
    score: float | None = None
    output: str | None = None

class BaseLanguageModel(abc.ABC):
    def __init__(self, **kwargs: Any):
        self._schema: BaseSchema | None = None
        self._fence_output_override: bool | None = None
        self._extra_kwargs: dict[str, Any] = kwargs.copy()

    @classmethod
    def get_schema_class(cls) -> type[Any] | None: return None

    def apply_schema(self, schema_instance: BaseSchema | None) -> None:
        self._schema = schema_instance

    @property
    def schema(self) -> BaseSchema | None: return self._schema

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
        return {**base, **(runtime_kwargs or {})}

    @abc.abstractmethod
    def infer(self, batch_prompts: Sequence[str], **kwargs) -> Iterator[Sequence[ScoredOutput]]: ...

    def infer_batch(self, prompts: Sequence[str]) -> list[list[ScoredOutput]]:
        results = []
        for output in self.infer(prompts):
            results.append(list(output))
        return results
