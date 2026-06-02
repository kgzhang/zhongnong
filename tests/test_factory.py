"""Tests for factory module."""
from src.factory import ModelConfig, create_model
from src.providers.base import BaseLanguageModel


class TestModelConfig:
    def test_create(self):
        mc = ModelConfig(model_id="deepseek-chat", provider_kwargs={"api_key": "test"})
        assert mc.model_id == "deepseek-chat"

    def test_default_provider(self):
        mc = ModelConfig()
        assert mc.model_id is None
        assert mc.provider is None
        assert mc.provider_kwargs == {}


class TestCreateModel:
    def test_create_basic(self):
        config = ModelConfig(model_id="deepseek-chat", provider_kwargs={"api_key": "test-key"})
        model = create_model(config)
        assert isinstance(model, BaseLanguageModel)
        assert model.model_id == "deepseek-chat"

    def test_create_with_examples(self):
        config = ModelConfig(model_id="deepseek-chat", provider_kwargs={"api_key": "test-key"})
        model = create_model(config, examples=[])
        assert isinstance(model, BaseLanguageModel)
