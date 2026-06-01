"""LLM client with JSON Schema enforcement via litellm gateway.

Supports Anthropic, OpenAI, and any litellm-compatible provider through a
unified interface with automatic model name resolution.
"""
import json
import logging
import re
import time
from typing import Optional

import litellm
from jsonschema import validate, ValidationError

from src.config import settings

logger = logging.getLogger(__name__)

# litellm global config
litellm.drop_params = True
litellm.suppress_debug_info = True


def resolve_model(model_name: str) -> str:
    """Resolve shorthand model names to litellm provider/model format.

    If the model already has a provider prefix (contains '/'), return as-is.
    Otherwise, prefix with 'deepseek/' as the default provider.

    Examples:
        deepseek-chat              ->  deepseek/deepseek-chat
        openai/gpt-4o              ->  openai/gpt-4o
        anthropic/claude-sonnet    ->  anthropic/claude-sonnet
    """
    if "/" in model_name:
        return model_name
    return f"deepseek/{model_name}"


class LLMClient:
    """Unified LLM client via litellm with JSON Schema validation and retry."""

    def __init__(self, model: Optional[str] = None):
        self.model = resolve_model(model or settings.llm_model)
        # Use provider-specific API key based on model prefix
        if self.model.startswith("deepseek/"):
            self.api_key = settings.deepseek_api_key
        elif self.model.startswith("anthropic/"):
            self.api_key = settings.anthropic_api_key
        else:
            self.api_key = settings.deepseek_api_key or settings.anthropic_api_key

    @staticmethod
    def load_schema(schema_path: str) -> dict:
        """Load a JSON Schema from file."""
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def validate_output(data: dict, schema: dict) -> list[str]:
        """Validate data against JSON Schema. Returns list of error messages (empty = valid)."""
        errors: list[str] = []
        try:
            validate(instance=data, schema=schema)
        except ValidationError as e:
            path = "/".join(str(p) for p in e.absolute_path)
            errors.append(f"{e.message} (at {path})" if path else e.message)
        return errors

    def extract_json(
        self,
        prompt: str,
        output_schema: dict,
        system_prompt: str = "You are a scientific literature data extraction expert.",
        temperature: float = 0.1,
    ) -> dict:
        """Call LLM via litellm, validate JSON output, retry on failure.

        Returns:
            Validated JSON dict with _model and _retry_count metadata.

        Raises:
            RuntimeError: If extraction fails after all retries.
        """
        last_error = None
        for attempt in range(settings.llm_max_retries + 1):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=8192,
                    api_key=self.api_key or None,
                )
                raw_text = response.choices[0].message.content
                if not raw_text:
                    raise ValueError("LLM returned empty response")

                data = parse_llm_json_response(raw_text)
                errors = self.validate_output(data, output_schema)
                if not errors:
                    data["_model"] = self.model
                    data["_retry_count"] = attempt
                    return data

                last_error = "; ".join(errors)
                logger.warning(
                    "Schema validation failed (attempt %d/%d): %s",
                    attempt + 1, settings.llm_max_retries + 1, last_error,
                )
                prompt = (
                    f"{prompt}\n\n"
                    f"[PREVIOUS OUTPUT HAD VALIDATION ERRORS: {last_error}. "
                    f"Fix the JSON output to match the schema.]"
                )

            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e}"
                logger.warning("JSON parse failed (attempt %d): %s", attempt + 1, e)
                prompt = (
                    f"{prompt}\n\n"
                    f"[PREVIOUS OUTPUT WAS NOT VALID JSON: {e}. Output ONLY valid JSON.]"
                )

            except Exception as e:
                last_error = str(e)
                if attempt < settings.llm_max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "LLM call failed (attempt %d), retrying in %ds: %s",
                        attempt + 1, wait, e,
                    )
                    time.sleep(wait)
                else:
                    logger.error("LLM call failed after all retries: %s", e)

        raise RuntimeError(
            f"LLM extraction failed after {settings.llm_max_retries + 1} attempts: {last_error}"
        )


def parse_llm_json_response(raw: str) -> dict:
    """Parse LLM response text that may be wrapped in markdown code fences.

    Handles:
    - Pure JSON:  {"key": "value"}
    - Fenced:     ```json\\n{"key": "value"}\\n```
    - No-lang:    ```\\n{"key": "value"}\\n```
    """
    raw = raw.strip()

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Extract from markdown fences
    for pattern in [r'```json\s*\n(.*?)\n```', r'```\s*\n(.*?)\n```']:
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            return json.loads(m.group(1).strip())

    raise json.JSONDecodeError("Could not parse JSON from response", raw, 0)
