"""OpenAI-compatible Chat Completions provider via httpx.

Includes LLM response caching to avoid redundant API calls.

Single-prompt batches (95 %+ of all calls in this pipeline) use a direct
sync ``httpx.Client`` stored in thread-local storage — zero asyncio overhead.
Multi-prompt batches still use ``asyncio.gather`` so all prompts hit the API
concurrently.
"""
from __future__ import annotations
import asyncio, json, logging, threading
from collections.abc import Iterator, Sequence
from typing import Any
import httpx
from src.config import settings
from src.data import FormatType
from src.providers.base import BaseLanguageModel, ScoredOutput
from src.providers.capabilities import detect_capabilities
from src.providers.schemas.openai import OpenAISchema

logger = logging.getLogger(__name__)

# Per-worker-thread sync httpx.Client — avoids asyncio.run() overhead
# for the common single-prompt case.
_tl = threading.local()


def _thread_sync_client() -> httpx.Client:
    """Return a thread-local ``httpx.Client`` (lazy, created once per thread)."""
    c = getattr(_tl, "client", None)
    if c is not None:
        return c
    c = httpx.Client(
        base_url=settings.llm_base_url,
        headers={
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(120.0),
    )
    _tl.client = c
    return c


class OpenAICompatProvider(BaseLanguageModel):
    """OpenAI-compatible provider — thread-local sync client for single prompts,
    async gather for multi-prompt batches."""

    def __init__(self, model_id: str = "deepseek-chat", api_key: str | None = None,
                 base_url: str | None = None, format_type: FormatType = FormatType.JSON,
                 temperature: float | None = None, thinking_enabled: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url or settings.llm_base_url
        self.format_type = format_type
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.thinking_enabled = thinking_enabled
        self.openai_schema: OpenAISchema | None = None
        self._capabilities = detect_capabilities(model_id)
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(self, batch_prompts: Sequence[str], **kwargs) -> Iterator[Sequence[ScoredOutput]]:
        """Concurrently run *batch_prompts* through the LLM.

        Single prompt → direct sync call (zero asyncio overhead).
        Multiple prompts → ``asyncio.gather`` (amortized overhead).
        """
        if not batch_prompts:
            return
        merged = self.merge_kwargs(kwargs)

        if len(batch_prompts) == 1:
            # Fast path — 95 %+ of calls, zero framing overhead.
            client = _thread_sync_client()
            yield [self._process_single(client, batch_prompts[0], merged)]
            return

        # Multi-prompt: fire all concurrently.
        async def _run_all():
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(120.0),
            ) as ac:
                tasks = [self._process_single_async(ac, p, merged)
                         for p in batch_prompts]
                return await asyncio.gather(*tasks)

        results = asyncio.run(_run_all())
        for r in results:
            if r is not None:
                yield [r]

    # ------------------------------------------------------------------
    # Sync single-request path (no asyncio)
    # ------------------------------------------------------------------

    def _process_single(self, client: httpx.Client, prompt: str,
                        config: dict) -> ScoredOutput:
        """Sync LLM call with caching and retry."""
        body = self._build_request(prompt, config)
        system_msg = body["messages"][0]["content"]

        if self._cache is not None:
            cached = self._cache.get(self.model_id, system_msg, prompt)
            if cached is not None:
                return ScoredOutput(score=1.0, output=cached)

        for attempt in range(settings.llm_max_retries):
            try:
                resp = client.post("/chat/completions", json=body)
                if resp.status_code in (429, 500, 502, 503):
                    import time
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]

                if self._cache is not None:
                    self._cache.set(self.model_id, system_msg, prompt, content)

                return ScoredOutput(score=1.0, output=content)
            except Exception as e:
                if attempt == settings.llm_max_retries - 1:
                    logger.error("LLM inference failed: %s", e)
                    return ScoredOutput(score=0.0, output="")
                import time
                time.sleep(2 ** attempt)
        return ScoredOutput(score=0.0, output="")

    # ------------------------------------------------------------------
    # Async path (for multi-prompt batches only)
    # ------------------------------------------------------------------

    async def _process_single_async(self, client: httpx.AsyncClient,
                                    prompt: str, config: dict) -> ScoredOutput | None:
        """Async LLM call with caching and retry."""
        body = self._build_request(prompt, config)
        system_msg = body["messages"][0]["content"]

        if self._cache is not None:
            cached = self._cache.get(self.model_id, system_msg, prompt)
            if cached is not None:
                return ScoredOutput(score=1.0, output=cached)

        for attempt in range(settings.llm_max_retries):
            try:
                resp = await client.post("/chat/completions", json=body)
                if resp.status_code in (429, 500, 502, 503):
                    await asyncio.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]

                if self._cache is not None:
                    self._cache.set(self.model_id, system_msg, prompt, content)

                return ScoredOutput(score=1.0, output=content)
            except Exception as e:
                if attempt == settings.llm_max_retries - 1:
                    logger.error("LLM async inference failed: %s", e)
                    return ScoredOutput(score=0.0, output="")
                await asyncio.sleep(2 ** attempt)
        return ScoredOutput(score=0.0, output="")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

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
        if not self.thinking_enabled:
            body["thinking"] = {"type": "disabled"}
        return body
