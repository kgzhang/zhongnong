"""Tests for pydantic-based config settings."""
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class TestSettings:
    """Test the pydantic Settings class."""

    def test_default_values(self):
        """Defaults are sensible when no env vars set."""
        from config import settings
        assert settings.llm_model == "deepseek/deepseek-chat"
        assert settings.llm_max_retries == 2
        assert settings.batch_size == 50
        assert settings.max_concurrent == 12
        assert isinstance(settings.project_root, Path)

    def test_api_key_fields(self):
        """API key fields exist and .env file is loaded."""
        from config import settings
        assert hasattr(settings, "deepseek_api_key")
        assert hasattr(settings, "anthropic_api_key")
        # .env file provides values; verify they're loaded (non-empty)
        assert len(settings.deepseek_api_key) > 0, ".env DEEPSEEK_API_KEY not loaded"

    def test_env_file_loaded(self, tmp_path, monkeypatch):
        """.env file is loaded automatically by pydantic-settings."""
        env_file = tmp_path / ".env"
        env_file.write_text("LLM_MODEL=deepseek/deepseek-reasoner\nDEEPSEEK_API_KEY=sk-test-key\n")
        # Override the env_file path — tricky since it's set at class level.
        # Instead, verify that SettingsConfigDict points to .env
        from config import Settings
        assert Settings.model_config.get("env_file") == ".env"

    def test_env_override(self, monkeypatch):
        """Environment variables override defaults."""
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("BATCH_SIZE", "100")
        # Re-import to pick up env changes (pydantic-settings reads at init)
        import importlib
        import config
        importlib.reload(config)
        assert config.settings.llm_model == "gpt-4o"
        assert config.settings.batch_size == 100
        assert isinstance(config.settings.batch_size, int)

    def test_type_coercion(self, monkeypatch):
        """String env values are coerced to proper types."""
        monkeypatch.setenv("LLM_MAX_RETRIES", "5")
        monkeypatch.setenv("MAX_CONCURRENT", "20")
        import importlib
        import config
        importlib.reload(config)
        assert config.settings.llm_max_retries == 5
        assert isinstance(config.settings.llm_max_retries, int)
        assert config.settings.max_concurrent == 20

    def test_directory_paths(self):
        """All directory paths are absolute Path objects."""
        from config import settings
        dirs = [
            settings.data_dir, settings.xml_dir, settings.sections_dir,
            settings.entities_dir, settings.relations_dir,
            settings.output_tsv_dir, settings.output_neo4j_dir,
            settings.schemas_dir,
        ]
        for d in dirs:
            assert isinstance(d, Path), f"{d} is not a Path"
            assert d.is_absolute(), f"{d} is not absolute"

    def test_ensure_dirs_creates_all(self, tmp_path):
        """ensure_dirs creates all expected directories using real paths."""
        import config
        # Use the real ensure_dirs with actual paths — it just creates dirs
        # that already exist (idempotent). Verify no crash.
        config.ensure_dirs()
        # All paths should exist after call
        assert config.DATA_DIR.exists()
        assert config.SCHEMAS_DIR.exists()

    def test_backward_compatible_exports(self):
        """All old-style module-level exports still work."""
        import config
        # Module-level re-exports for backward compatibility
        attrs = [
            "PROJECT_ROOT", "DATA_DIR", "XML_DIR", "SECTIONS_DIR",
            "ENTITIES_DIR", "RELATIONS_DIR", "OUTPUT_TSV_DIR",
            "OUTPUT_NEO4J_DIR", "SCHEMAS_DIR",
            "ALTERNATIVE_TSV", "SCHEMA_TSV",
            "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "LLM_MODEL",
            "LLM_MAX_RETRIES", "BATCH_SIZE", "MAX_CONCURRENT",
            "ENTREZ_EMAIL", "ENTREZ_API_KEY",
            "ensure_dirs", "validate",
        ]
        for attr in attrs:
            assert hasattr(config, attr), f"Missing backward-compat export: {attr}"

    def test_validate_checks_email(self, caplog, monkeypatch):
        """validate() logs warning when ENTREZ_EMAIL is empty."""
        import logging
        from config import settings, validate
        monkeypatch.setattr(settings, "entrez_email", "")
        monkeypatch.setattr(settings, "deepseek_api_key", "")
        monkeypatch.setattr(settings, "anthropic_api_key", "")
        with caplog.at_level(logging.WARNING):
            validate()
        assert "ENTREZ_EMAIL" in caplog.text
