from src.providers.capabilities import detect_capabilities

class TestDetectCapabilities:
    def test_openai(self):
        caps = detect_capabilities("gpt-4o")
        assert caps.supports_json_schema is True
        assert caps.supports_json_schema_strict is True
    def test_deepseek(self):
        caps = detect_capabilities("deepseek-chat")
        assert caps.supports_json_schema is True
        assert caps.supports_json_schema_strict is False
    def test_qwen_max(self):
        caps = detect_capabilities("qwen-max")
        assert caps.supports_json_object is True
    def test_qwen_oss(self):
        caps = detect_capabilities("qwen2.5-7b")
        assert caps.requires_fence_output is True
