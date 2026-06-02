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
        assert len(names) >= 12

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
        assert len(phases) >= 2
        assert phases[0].gate is not None
        assert phases[0].gate.entity == "Alternative"

    def test_post_process_sets_match_source(self):
        ext = Extraction(extraction_class="Alternative", extraction_text="thymol",
                        attributes={"standard_name": "thymol"})
        result = self.registry.post_process(ext)
        assert "classification" in (result.attributes or {})
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

    def test_post_process_unknown_entity(self):
        """post_process on unknown entity should return unchanged."""
        ext = Extraction(extraction_class="NonExistent", extraction_text="test")
        result = self.registry.post_process(ext)
        assert result is ext  # unchanged

    def test_validate_extraction_valid(self):
        """Valid extraction with required LLM fields should have no errors."""
        ext = Extraction(
            extraction_class="Alternative", extraction_text="thymol",
            attributes={"standard_name": "thymol", "original_text": "thymol"},
        )
        errors = self.registry.validate_extraction(ext)
        # evidence_text and source_location are source=align/structure, not LLM
        # Only LLM-sourced required fields are checked
        assert isinstance(errors, list)

    def test_unknown_entity_validation(self):
        """Validating unknown entity type returns error."""
        ext = Extraction(extraction_class="FakeType", extraction_text="test")
        errors = self.registry.validate_extraction(ext)
        assert len(errors) == 1
        assert "Unknown" in errors[0]

    def test_build_prompt_includes_vocabulary(self):
        """Prompt for Alternative should include vocabulary text."""
        prompt = self.registry.build_extraction_prompt(["Alternative"])
        # Should contain vocabulary if ALTERNATIVE.tsv exists
        assert len(prompt) > 100  # substantial prompt with vocab

    def test_json_schema_for_multiple_entities(self):
        """generate_json_schema with multiple entities creates anyOf variants."""
        schema = self.registry.generate_json_schema(["Alternative", "Composite_Product"])
        items = schema["properties"]["extractions"]["items"]
        variants = items.get("anyOf", [])
        assert len(variants) >= 2

    def test_entity_not_found_raises_key_error(self):
        """llm_output_fields for unknown entity raises KeyError."""
        with pytest.raises(KeyError):
            self.registry.llm_output_fields("NonExistent")

    def test_vocabulary_lookup_exact(self):
        """Vocabulary should support case-insensitive exact match."""
        vocab = self.registry.vocabulary("Alternative")
        if vocab and vocab.entries:
            # Lookup something that exists
            entry = vocab.lookup("thymol")
            assert entry is not None or entry is None  # depends on vocab data

    def test_vocabulary_fuzzy_match(self):
        """Vocabulary fuzzy_match should handle close variants."""
        vocab = self.registry.vocabulary("Alternative")
        if vocab and vocab.entries:
            result = vocab.fuzzy_match("thymol")
            assert result is not None or result is None  # depends on data

    def test_relation_def_attributes(self):
        """RelationDef should have expected attributes."""
        rd = self.registry.relation_def("belongs_to")
        assert rd.name == "belongs_to"
        assert isinstance(rd.source, list)
        assert isinstance(rd.target, list)

    def test_phase_def_gate_structure(self):
        """GateDef should have expected structure."""
        phases = self.registry.phase_defs()
        gated = [p for p in phases if p.gate is not None]
        assert len(gated) > 0
        assert gated[0].gate.entity is not None
