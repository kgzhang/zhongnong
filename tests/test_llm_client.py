"""Tests for LLM Client with JSON Schema enforcement (litellm backend)."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from llm_client import LLMClient, parse_llm_json_response, resolve_model


def test_resolve_model_anthropic():
    """Short model names get anthropic/ prefix for litellm."""
    assert resolve_model("claude-sonnet-4-20250514") == "anthropic/claude-sonnet-4-20250514"


def test_resolve_model_prefixed_passthrough():
    """Already-prefixed model names pass through unchanged."""
    assert resolve_model("openai/gpt-4o") == "openai/gpt-4o"
    assert resolve_model("anthropic/claude-opus-4-20250514") == "anthropic/claude-opus-4-20250514"


def test_parse_valid_json():
    raw = '{"doi": "10.1016/x", "results": []}'
    result = parse_llm_json_response(raw)
    assert result["doi"] == "10.1016/x"
    assert result["results"] == []


def test_parse_json_with_markdown_wrapper():
    raw = '```json\n{"key": "value"}\n```'
    result = parse_llm_json_response(raw)
    assert result["key"] == "value"


def test_parse_json_no_fence():
    raw = '{"key": "value"}'
    result = parse_llm_json_response(raw)
    assert result["key"] == "value"


def test_validate_against_schema():
    schema = {
        "type": "object",
        "properties": {
            "doi": {"type": "string"},
            "alternatives": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "standard_name": {"type": "string"},
                        "alternative_class": {"enum": ["Plant_Extract", "Other"]},
                    },
                    "required": ["standard_name", "alternative_class"],
                }
            },
        },
        "required": ["doi"],
    }
    data = {
        "doi": "10.1016/test",
        "alternatives": [{"standard_name": "thymol", "alternative_class": "Plant_Extract"}],
    }
    errors = LLMClient.validate_output(data, schema)
    assert len(errors) == 0


def test_validate_invalid_enum():
    schema = {
        "type": "object",
        "properties": {"direction": {"enum": ["increased", "decreased", "no_significant_change"]}},
        "required": ["direction"],
    }
    data = {"direction": "up"}
    errors = LLMClient.validate_output(data, schema)
    assert len(errors) > 0


def test_validate_missing_required():
    schema = {
        "type": "object",
        "properties": {"doi": {"type": "string"}},
        "required": ["doi"],
    }
    data = {}
    errors = LLMClient.validate_output(data, schema)
    assert len(errors) > 0


def test_load_schema(tmp_path):
    schema_path = tmp_path / "test_schema.json"
    schema_path.write_text('{"type": "object", "required": ["name"]}')
    schema = LLMClient.load_schema(str(schema_path))
    assert schema["type"] == "object"
    assert "name" in schema["required"]
