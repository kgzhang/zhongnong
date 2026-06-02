"""OpenAI-compatible Chat Completions provider via httpx.

Includes LLM response caching to avoid redundant API calls.
"""
from __future__ import annotations
import json, time, logging
from collections.abc import Iterator, Sequence
from typing import Any
import httpx
from src.config import settings
from src.data import FormatType
from src.providers.base import BaseLanguageModel, ScoredOutput
from src.providers.capabilities import detect_capabilities
from src.providers.schemas.openai import OpenAISchema

logger = logging.getLogger(__name__)


class OpenAICompatProvider(BaseLanguageModel):
    """OpenAI-compatible provider with response caching."""

    def __init__(self, model_id: str = "deepseek-chat", api_key: str | None = None,
                 base_url: str | None = None, format_type: FormatType = FormatType.JSON,
                 temperature: float | None = None, max_workers: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url or settings.llm_base_url
        self.format_type = format_type
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_workers = max_workers
        self.openai_schema: OpenAISchema | None = None
        self._capabilities = detect_capabilities(model_id)
        self._client: httpx.Client | None = None
        self._cache_enabled = settings.cache_enabled
        self._cache = None
        if self._cache_enabled:
            from src.cache import LLMCache
            self._cache = LLMCache(settings.cache_dir)

    @classmethod
    def get_schema_class(cls) -> type: return OpenAISchema

    def apply_schema(self, schema_instance):
        if schema_instance is None:
            self.openai_schema = None
        elif isinstance(schema_instance, OpenAISchema):
            self.openai_schema = schema_instance
        super().apply_schema(schema_instance)

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                timeout=120.0,
            )
        return self._client

    def infer(self, batch_prompts: Sequence[str], **kwargs) -> Iterator[Sequence[ScoredOutput]]:
        merged = self.merge_kwargs(kwargs)
        client = self._get_client()
        for prompt in batch_prompts:
            result = self._process_single(client, prompt, merged)
            yield [result]

    def _process_single(self, client: httpx.Client, prompt: str, config: dict) -> ScoredOutput:
        body = self._build_request(prompt, config)
        system_msg = body["messages"][0]["content"]

        # Check cache
        if self._cache is not None:
            cached = self._cache.get(self.model_id, system_msg, prompt)
            if cached is not None:
                return ScoredOutput(score=1.0, output=cached)

        # API call with retry
        for attempt in range(settings.llm_max_retries):
            try:
                resp = client.post("/chat/completions", json=body)
                if resp.status_code in (429, 500, 502, 503):
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]

                # Store in cache
                if self._cache is not None:
                    self._cache.set(self.model_id, system_msg, prompt, content)

                return ScoredOutput(score=1.0, output=content)
            except (httpx.HTTPError, KeyError, json.JSONDecodeError) as e:
                if attempt == settings.llm_max_retries - 1:
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
        }
        temp = config.get("temperature", self.temperature)
        if temp is not None:
            body["temperature"] = temp
        if self.openai_schema and self._capabilities.supports_json_schema:
            body["response_format"] = self.openai_schema.response_format
        elif self._capabilities.supports_json_object:
            body["response_format"] = {"type": "json_object"}
        if "max_tokens" in config:
            body["max_tokens"] = config["max_tokens"]
        return body
