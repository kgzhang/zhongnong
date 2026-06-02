"""Model capability detection and fallback chain."""
import dataclasses

@dataclasses.dataclass
class ModelCapabilities:
    supports_json_schema: bool = False
    supports_json_schema_strict: bool = False
    supports_json_object: bool = False
    supports_system_message: bool = True
    requires_fence_output: bool = True
    max_context_tokens: int = 128000

def detect_capabilities(model_id: str) -> ModelCapabilities:
    mid = model_id.lower()
    # OpenAI
    if any(p in mid for p in ("gpt-4", "gpt-3.5", "o1", "o3")):
        return ModelCapabilities(supports_json_schema=True, supports_json_schema_strict=True,
                                supports_json_object=True, requires_fence_output=False)
    # DeepSeek
    if "deepseek" in mid:
        return ModelCapabilities(supports_json_schema=True, supports_json_schema_strict=False,
                                supports_json_object=True, requires_fence_output=False)
    # Qwen DashScope
    if "qwen" in mid:
        if any(p in mid for p in ("qwen-max", "qwen-plus", "qwen-turbo")):
            return ModelCapabilities(supports_json_schema=False, supports_json_object=True,
                                    requires_fence_output=False)
        return ModelCapabilities(requires_fence_output=True)
    # Default
    return ModelCapabilities(supports_json_object=True, requires_fence_output=False)
