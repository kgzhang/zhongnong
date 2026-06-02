"""Tests for schema registry."""
import pytest
from pathlib import Path
from src.data import Extraction
from src.schema_registry import SchemaRegistry, AttributeDef

SCHEMA_DIR = Path(__file__).parent.parent / "schemas"


class TestSchemaRegistry:
    def setup_method(self):
        self.registry = SchemaRegistry(config_dir=SCHEMA_DIR)

    def test_load_entities(self):
        names = self.registry.all_entity_names()
        assert "Alternative" in names
        assert "Composite_Product" in names
        assert "Result" in names
        assert len(names) >= 13

    def test_entity_def(self):
        ed = self.registry.entity_def("Alternative")
        assert ed.name == "Alternative"
        assert ed.primary_text == "standard_name"
        assert len(ed.attributes) > 0

    def test_llm_output_fields_excludes_evidence(self):
        fields = self.registry.llm_output_fields("Alternative")
        field_names = {f.name for f in fields}
        assert "evidence_text" not in field_names
        assert "source_location" not in field_names
        assert "standard_name" in field_names

    def test_generate_json_schema(self):
        schema = self.registry.generate_json_schema(["Alternative"])
        assert "extractions" in schema["properties"]
        items = schema["properties"]["extractions"]["items"]
        assert "anyOf" in items
        schema_str = str(items)
        assert "evidence_text" not in schema_str  # LLM fields only

    def test_relation_defs(self):
        assert "belongs_to" in self.registry.all_relation_names()
        assert "increases" in self.registry.all_relation_names()

    def test_phase_defs(self):
        phases = self.registry.phase_defs()
        assert len(phases) >= 4
        assert phases[0].gate is not None
        assert phases[0].gate.entity == "Alternative"

    def test_post_process_sets_match_source(self):
        ext = Extraction(extraction_class="Alternative", extraction_text="thymol",
                        attributes={"standard_name": "thymol"})
        result = self.registry.post_process(ext)
        assert "alternative_class" in (result.attributes or {})
        assert "match_source" in (result.attributes or {})

    def test_validate_extraction(self):
        ext = Extraction(extraction_class="Alternative", extraction_text="thymol",
                        attributes={})
        errors = self.registry.validate_extraction(ext)
        # standard_name is required for Alternative, it's missing
        assert len(errors) > 0

    def test_build_extraction_prompt(self):
        prompt = self.registry.build_extraction_prompt(["Alternative"])
        assert "提取要求" in prompt or "Alternative" in prompt
        assert len(prompt) > 0
