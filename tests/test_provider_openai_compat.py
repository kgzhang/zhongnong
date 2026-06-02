import pytest
from src.providers.openai_compat import OpenAICompatProvider
from src.providers.schemas.openai import OpenAISchema
from src.data import FormatType

class TestOpenAICompatProvider:
    def test_init(self):
        p = OpenAICompatProvider(model_id="deepseek-chat", api_key="test-key")
        assert p.model_id == "deepseek-chat"
        assert p.format_type == FormatType.JSON

    def test_capability_deepseek(self):
        p = OpenAICompatProvider(model_id="deepseek-chat", api_key="test-key")
        assert p._capabilities.supports_json_schema is True
        assert p._capabilities.supports_json_schema_strict is False

    def test_apply_schema(self):
        p = OpenAICompatProvider(model_id="gpt-4o", api_key="test-key")
        schema = OpenAISchema(schema_dict={"type": "object", "properties": {}, "additionalProperties": False})
        p.apply_schema(schema)
        assert p.openai_schema is not None

    def test_requires_fence_output_default(self):
        p = OpenAICompatProvider(model_id="qwen2.5-7b", api_key="test-key")
        assert p.requires_fence_output is True

    def test_build_request_basic(self):
        p = OpenAICompatProvider(model_id="deepseek-chat", api_key="test-key")
        body = p._build_request("Test prompt", {})
        assert body["model"] == "deepseek-chat"
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"

    def test_build_request_qwen_json_object(self):
        p = OpenAICompatProvider(model_id="qwen-max", api_key="test-key")
        body = p._build_request("Test", {})
        assert "response_format" in body
        assert body["response_format"]["type"] == "json_object"
