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

    def test_fence_output_override(self):
        """set_fence_output should override default."""
        p = OpenAICompatProvider(model_id="gpt-4o", api_key="test-key")
        assert p.requires_fence_output is True  # no schema -> default True
        p.set_fence_output(False)
        assert p.requires_fence_output is False

    def test_merge_kwargs(self):
        """merge_kwargs should merge stored and runtime kwargs."""
        p = OpenAICompatProvider(model_id="deepseek-chat", api_key="test-key", extra="val")
        merged = p.merge_kwargs({"runtime": "val2"})
        assert merged.get("extra") == "val"
        assert merged.get("runtime") == "val2"

    def test_apply_schema_none_clears(self):
        """Applying None schema should clear it."""
        p = OpenAICompatProvider(model_id="gpt-4o", api_key="test-key")
        schema = OpenAISchema(schema_dict={"type": "object", "properties": {}, "additionalProperties": False})
        p.apply_schema(schema)
        assert p.openai_schema is not None
        p.apply_schema(None)
        assert p.openai_schema is None

    def test_requires_fence_output_with_schema(self):
        """With OpenAISchema applied, should not require fences."""
        p = OpenAICompatProvider(model_id="gpt-4o", api_key="test-key")
        schema = OpenAISchema(schema_dict={"type": "object", "properties": {}, "additionalProperties": False})
        p.apply_schema(schema)
        assert p.requires_fence_output is False

    def test_build_request_with_schema(self):
        """When schema is applied and supported, response_format should use json_schema."""
        p = OpenAICompatProvider(model_id="gpt-4o", api_key="test-key")
        schema = OpenAISchema(schema_dict={"type": "object", "properties": {"extractions": {"type": "array", "items": {"type": "object"}}}, "required": ["extractions"], "additionalProperties": False})
        p.apply_schema(schema)
        body = p._build_request("Test", {})
        assert "response_format" in body
        assert body["response_format"]["type"] == "json_schema"

    def test_build_request_with_temperature(self):
        """Temperature should be included in request when set."""
        p = OpenAICompatProvider(model_id="deepseek-chat", api_key="test-key", temperature=0.5)
        body = p._build_request("Test", {})
        assert body["temperature"] == 0.5

    def test_build_request_default_model_in_body(self):
        """Request body should use the model_id."""
        p = OpenAICompatProvider(model_id="custom-model", api_key="test-key")
        body = p._build_request("Test", {})
        assert body["model"] == "custom-model"

    def test_get_schema_class(self):
        """get_schema_class should return OpenAISchema."""
        assert OpenAICompatProvider.get_schema_class() == OpenAISchema
