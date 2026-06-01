"""LLM client with JSON Schema enforcement and retry logic."""
import json
import re
import time
from typing import Optional
from pathlib import Path
from anthropic import Anthropic, RateLimitError, APIError
from jsonschema import validate, ValidationError
from src.config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_MAX_RETRIES


class LLMClient:
    """Anthropic API client with JSON Schema validation and automatic retry."""

    def __init__(self, model: Optional[str] = None):
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = model or LLM_MODEL

    @staticmethod
    def load_schema(schema_path: str) -> dict:
        """Load a JSON Schema from a file path."""
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def validate_output(data: dict, schema: dict) -> list[str]:
        """Validate LLM output against JSON Schema. Returns list of error messages (empty = valid)."""
        errors = []
        try:
            validate(instance=data, schema=schema)
        except ValidationError as e:
            errors.append(f"{e.message} (at {'/'.join(str(p) for p in e.absolute_path)})")
        return errors

    def extract_json(
        self,
        prompt: str,
        output_schema: dict,
        system_prompt: str = "You are a scientific literature data extraction expert.",
        temperature: float = 0.1,
    ) -> dict:
        """Call LLM with JSON output, validate against schema, retry on failure.

        Args:
            prompt: The user prompt with extraction instructions.
            output_schema: JSON Schema the response must conform to.
            system_prompt: System-level instruction for the model.
            temperature: LLM temperature (low for deterministic extraction).

        Returns:
            Parsed and validated JSON dict. The dict will have _model and _retry_count metadata keys.

        Raises:
            RuntimeError: If extraction fails after all retries.
        """
        last_error = None
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=8192,
                    temperature=temperature,
                    system=system_prompt + "\n\nYou MUST respond with valid JSON that matches the schema exactly. Do NOT wrap in markdown fences unless absolutely necessary.",
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                )
                raw_text = response.content[0].text
                data = parse_llm_json_response(raw_text)

                errors = self.validate_output(data, output_schema)
                if not errors:
                    data["_model"] = self.model
                    data["_retry_count"] = attempt
                    return data

                last_error = "; ".join(errors)
                # Append error context for retry
                prompt = f"{prompt}\n\n[PREVIOUS OUTPUT HAD VALIDATION ERRORS: {last_error}. Fix the JSON output to match the schema.]"

            except (RateLimitError, APIError) as e:
                last_error = str(e)
                if attempt < LLM_MAX_RETRIES:
                    wait = 2 ** attempt
                    time.sleep(wait)
            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e}"
                prompt = f"{prompt}\n\n[PREVIOUS OUTPUT WAS NOT VALID JSON: {e}. Output ONLY valid JSON.]"

        raise RuntimeError(f"LLM extraction failed after {LLM_MAX_RETRIES + 1} attempts: {last_error}")


def parse_llm_json_response(raw: str) -> dict:
    """Parse LLM response text that may be wrapped in markdown code fences.

    Handles:
    - Pure JSON: '{"key": "value"}'
    - Fenced JSON: '```json\\n{"key": "value"}\\n```'
    - Fenced without language: '```\\n{"key": "value"}\\n```'
    """
    raw = raw.strip()

    # Try direct parsing first (most common case)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Extract from markdown fences: try ```json ... ``` then ``` ... ```
    for fence_pattern in [r'```json\s*\n(.*?)\n```', r'```\s*\n(.*?)\n```']:
        m = re.search(fence_pattern, raw, re.DOTALL)
        if m:
            return json.loads(m.group(1).strip())

    raise json.JSONDecodeError(f"Could not parse JSON from response", raw, 0)
