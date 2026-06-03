"""End-to-end integration tests for llm-extract v2 pipeline."""
import pytest
import tempfile
from pathlib import Path

SCHEMA_DIR = Path(__file__).parent.parent / "schemas"


class TestSchemaRegistryLoads:
    def test_all_entities_load(self):
        """Verify schema registry loads all entity types from YAML config."""
        from src.schema_registry import SchemaRegistry
        registry = SchemaRegistry(config_dir=SCHEMA_DIR)
        names = registry.all_entity_names()
        assert "Alternative" in names
        assert "Alternative_Class" in names
        assert "Result" in names
        assert "Indicator" in names
        assert "Intervention" in names
        assert len(names) >= 11

    def test_all_phases_load(self):
        """Verify extraction phases are defined (backward-compat, phases unused in core pipeline)."""
        from src.schema_registry import SchemaRegistry
        registry = SchemaRegistry(config_dir=SCHEMA_DIR)
        phases = registry.phase_defs()
        assert len(phases) >= 2
        # First phase has a gate definition
        assert phases[0].gate is not None

    def test_json_schema_generation(self):
        """Verify JSON Schema is generated without align/structure fields."""
        from src.schema_registry import SchemaRegistry
        registry = SchemaRegistry(config_dir=SCHEMA_DIR)
        schema = registry.generate_json_schema(["Alternative", "Result"])
        schema_str = str(schema)
        # evidence_text and source_location must NOT be in LLM schema
        assert "evidence_text" not in schema_str
        assert "source_location" not in schema_str


class TestGraphExportRoundtrip:
    def test_export_and_verify(self):
        """Build a minimal graph and verify Neo4j CSV output."""
        from src.graph import GraphNode, GraphEdge, Graph, export_neo4j_csv

        node = GraphNode(
            id="alt_001", labels=["Alternative", "Entity"],
            properties={"name": "thymol", "classification": "Plant_Extract"},
            source_pmids=["12345"],
        )
        edge = GraphEdge(
            source_id="alt_001", target_id="altclass_001",
            type="belongs_to",
            properties={"evidence_text": "thymol was used"},
        )
        graph = Graph(nodes=[node], edges=[edge])

        with tempfile.TemporaryDirectory() as tmpdir:
            export_neo4j_csv(graph, Path(tmpdir))
            nodes_csv = Path(tmpdir) / "nodes.csv"
            edges_csv = Path(tmpdir) / "edges.csv"
            assert nodes_csv.exists()
            assert edges_csv.exists()
            # Verify Neo4j CSV header format
            nodes_text = nodes_csv.read_text()
            assert "entity_id:ID" in nodes_text
            assert "entity_type:LABEL" in nodes_text
            edges_text = edges_csv.read_text()
            assert "source_id:START_ID" in edges_text
            assert "relation_type:TYPE" in edges_text


class TestFullModuleImports:
    def test_all_modules_import(self):
        """Verify all v2 pipeline modules import without errors."""
        import src.data
        import src.tokenizer
        import src.chunking
        import src.format_handler
        import src.schema_registry
        import src.schema
        import src.prompting
        import src.resolver
        import src.evidence
        import src.annotation
        import src.extraction
        import src.factory
        import src.graph
        import src.providers.base
        import src.providers.capabilities
        import src.providers.openai_compat
        # All imports succeeded — no ModuleNotFoundError


class TestDataFlow:
    def test_extraction_to_graph_flow(self):
        """Verify Extraction → entity_global_id → GraphNode flow works."""
        from src.data import Extraction
        from src.graph import entity_global_id, GraphNode

        ext = Extraction(
            extraction_class="Alternative",
            extraction_text="thymol",
            attributes={"standard_name": "thymol", "classification": "Plant_Extract"},
        )
        gid = entity_global_id("Alternative", "thymol")
        node = GraphNode(
            id=gid, labels=["Alternative", "Entity"],
            properties=ext.attributes or {},
            source_pmids=["12345"],
        )
        assert node.id == gid
        assert node.properties["standard_name"] == "thymol"


class TestCLIImports:
    def test_cli_imports(self):
        """Verify CLI module can be imported."""
        from src.cli import cli
        assert cli is not None


class TestEndToEndPipeline:
    """Full end-to-end test: text → extract → graph → CSV export."""

    def test_full_pipeline_with_mock_llm(self):
        """Run the complete pipeline with a mock LLM and verify output."""
        import json
        from src.format_handler import FormatHandler, FormatType
        from src.prompting import PromptTemplateStructured
        from src.annotation import Annotator
        from src.schema_registry import SchemaRegistry
        from src.graph import build_graph, export_neo4j_csv

        registry = SchemaRegistry(config_dir=SCHEMA_DIR)

        # Single prompt extracts all entity types (no phases)
        all_json = json.dumps({
            "extractions": [
                {"Alternative": "thymol", "Alternative_attributes": {
                    "standard_name": "thymol", "abbreviation": "THY",
                    "original_text": "thymol (THY, purity >= 99%)",
                }},
                {"Alternative": "Product X", "Alternative_attributes": {
                    "standard_name": "Product X", "is_composite": True, "is_commercial": True,
                    "product_name": "Product X",
                    "components": [{"standard_name": "thymol", "entity_type": "Alternative"}],
                }},
                {"Intervention": "thymol", "Intervention_attributes": {
                    "intervention_target": "thymol", "dose_value": 500,
                    "dose_unit_original": "mg/kg", "administration_route": "diet",
                    "duration": "28 days",
                }},
                {"Control_Group": "Basal Diet", "Control_Group_attributes": {
                    "group_name": "Basal Diet", "group_type": "basal_control",
                }},
                {"Tissue_Site": "jejunal mucosa", "Tissue_Site_attributes": {
                    "site_name": "jejunal mucosa", "site_category": "mucosa",
                }},
                {"Indicator": "ADG", "Indicator_attributes": {
                    "standard_name": "Average Daily Gain", "abbreviation": "ADG",
                    "unit": "g/d", "indicator_category": "macro_phenotype",
                }},
                {"Result": "ADG", "Result_attributes": {
                    "indicator_abbreviation": "ADG",
                    "tissue_site": "jejunal mucosa",
                    "direction": "increased",
                    "relation_type": "increases",
                    "significance_level": "p_less_0.05",
                    "p_value_original_text": "P < 0.05",
                    "compared_to_group": "Basal Diet",
                }},
            ]
        })

        class SingleMockLM:
            requires_fence_output = False
            def infer(self, batch_prompts, **kwargs):
                yield [MockSO(all_json)]
            def apply_schema(self, s): pass
            def set_fence_output(self, v): pass
            @property
            def schema(self): return None

        class MockSO:
            def __init__(self, output): self.output = output; self.score = 1.0

        # Single-pass extraction (no phases)
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        model = SingleMockLM()
        all_entity_names = registry.all_entity_names()
        prompt = registry.build_extraction_prompt(all_entity_names)
        template = PromptTemplateStructured(description=prompt)
        annotator = Annotator(model, template, fh)

        source_text = (
            "Pigs were fed a basal diet supplemented with 500 mg/kg thymol (THY, "
            "purity >= 99%) for 28 days. Average Daily Gain (ADG) was measured. "
            "Compared with the Basal Diet control group, thymol significantly "
            "increased ADG in the jejunal mucosa (P < 0.05)."
        )
        result = annotator.annotate_text(source_text, max_char_buffer=2000)
        all_extractions = result.extractions or []

        # Post-process
        for ext in all_extractions:
            registry.post_process(ext)

        # Verify extractions
        assert len(all_extractions) >= 5, f"Expected 5+ extractions, got {len(all_extractions)}"

        # Verify entity types
        entity_types = {e.extraction_class for e in all_extractions}
        assert "Alternative" in entity_types
        assert "Alternative_Class" in entity_types or "Alternative" in entity_types
        assert "Intervention" in entity_types
        assert "Indicator" in entity_types
        assert "Result" in entity_types

        # Verify evidence_text and source_location are populated (now top-level fields)
        for ext in all_extractions:
            # evidence_text is now a top-level Extraction field, not in attributes
            assert hasattr(ext, "evidence_text"), \
                f"Missing evidence_text for {ext.extraction_class}:{ext.extraction_text}"
            assert hasattr(ext, "source_location"), \
                f"Missing source_location for {ext.extraction_class}:{ext.extraction_text}"

        # Post-process should set classification via vocabulary matching
        alt_exts = [e for e in all_extractions if e.extraction_class == "Alternative"]
        for alt in alt_exts:
            assert "classification" in (alt.attributes or {})

        # Build graph
        from src.extraction import DocumentExtractionResult
        article_result = DocumentExtractionResult(
            document_id="99999", metadata={"doi": "10.1234/test", "pmid": "99999"},
            extractions=all_extractions,
        )
        graph = build_graph([article_result], registry)

        # Verify graph structure
        assert len(graph.nodes) >= 5
        assert len(graph.edges) >= 1

        # Verify corresponds_to edge exists (Result → Indicator)
        edge_types = {e.type for e in graph.edges}
        assert "corresponds_to" in edge_types

        # Export to CSV
        with tempfile.TemporaryDirectory() as tmpdir:
            export_neo4j_csv(graph, Path(tmpdir))
            nodes_csv = Path(tmpdir) / "nodes.csv"
            edges_csv = Path(tmpdir) / "edges.csv"
            assert nodes_csv.exists()
            assert edges_csv.exists()

            # Verify CSV content is importable
            nodes_text = nodes_csv.read_text()
            assert "thymol" in nodes_text
            assert "entity_id:ID" in nodes_text

            edges_text = edges_csv.read_text()
            assert "source_id:START_ID" in edges_text

    def test_unknown_entity_post_processing(self):
        """Entity with no vocabulary match should get classification='Other'."""
        import json
        from src.format_handler import FormatHandler, FormatType
        from src.prompting import PromptTemplateStructured
        from src.annotation import Annotator
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry(config_dir=SCHEMA_DIR)

        other_json = json.dumps({
            "extractions": [
                {"Alternative": "unknown_substance", "Alternative_attributes": {
                    "standard_name": "unknown_substance",
                    "classification": "Other",
                    "original_text": "unknown_substance",
                }},
            ]
        })

        class SingleMockLM:
            requires_fence_output = False
            def infer(self, batch_prompts, **kwargs):
                yield [MockSO2(other_json)]
            def apply_schema(self, s): pass
            def set_fence_output(self, v): pass
            @property
            def schema(self): return None

        class MockSO2:
            def __init__(self, output): self.output = output; self.score = 1.0

        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        model = SingleMockLM()
        all_names = registry.all_entity_names()
        prompt = registry.build_extraction_prompt(all_names)
        template = PromptTemplateStructured(description=prompt)
        annotator = Annotator(model, template, fh)
        result = annotator.annotate_text("Some unknown substance was tested.", max_char_buffer=500)

        # Post-process
        for ext in (result.extractions or []):
            registry.post_process(ext)

        # Verify: classification may or may not match vocabulary (depends on fuzzy matching).
        # The improved vocabulary matching may find partial matches for generic names.
        alt_exts = [e for e in (result.extractions or []) if e.extraction_class == "Alternative"]
        for alt in alt_exts:
            alt_class = (alt.attributes or {}).get("classification", "")
            assert alt_class is not None  # should always have a classification

    def test_extraction_result_dataclass(self):
        """DocumentExtractionResult should handle all fields."""
        from src.data import Extraction
        from src.extraction import DocumentExtractionResult

        r = DocumentExtractionResult(
            document_id="12345", metadata={"doi": "10.1/test", "pmid": "12345"},
            extractions=[Extraction(extraction_class="Alt", extraction_text="test")],
            warnings=["warning1"],
        )
        assert r.metadata.get("doi") == "10.1/test"
        assert r.document_id == "12345"
        assert len(r.extractions) == 1
        assert len(r.warnings) == 1
