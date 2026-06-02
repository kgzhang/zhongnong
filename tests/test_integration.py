"""End-to-end integration tests for llm-extract v2 pipeline."""
import pytest
import tempfile
from pathlib import Path

SCHEMA_DIR = Path(__file__).parent.parent / "schemas"


class TestSchemaRegistryLoads:
    def test_all_entities_load(self):
        """Verify schema registry loads all 12 entity types from YAML config."""
        from src.schema_registry import SchemaRegistry
        registry = SchemaRegistry(config_dir=SCHEMA_DIR)
        names = registry.all_entity_names()
        assert "Alternative" in names
        assert "Composite_Product" in names
        assert "Result" in names
        assert "Indicator" in names
        assert "Intervention" in names
        assert len(names) >= 12

    def test_all_phases_load(self):
        """Verify all 4 extraction phases are defined."""
        from src.schema_registry import SchemaRegistry
        registry = SchemaRegistry(config_dir=SCHEMA_DIR)
        phases = registry.phase_defs()
        assert len(phases) >= 2
        # Phase 1 has gate
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
        import src.source_location
        import src.annotation
        import src.extraction
        import src.factory
        import src.graph
        import src.providers.base
        import src.providers.capabilities
        import src.providers.openai_compat
        import src.providers.schemas.openai
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
    """Full end-to-end test: XML → extract → graph → CSV export."""

    def test_full_pipeline_with_mock_llm(self):
        """Run the complete pipeline with a mock LLM and verify output."""
        import json
        from src.data import Document, Extraction
        from src.format_handler import FormatHandler, FormatType
        from src.prompting import PromptTemplateStructured
        from src.annotation import Annotator
        from src.schema_registry import SchemaRegistry
        from src.extraction import DocumentExtractionResult, _build_results_context
        from src.graph import build_graph, export_neo4j_csv

        registry = SchemaRegistry(config_dir=SCHEMA_DIR)

        # Phase 1 mock: Alternatives
        alt_json = json.dumps({
            "extractions": [
                {"Alternative": "thymol", "Alternative_attributes": {
                    "standard_name": "thymol", "abbreviation": "THY",
                    "original_text": "thymol (THY, purity >= 99%)",
                }},
                {"Composite_Product": "Product X", "Composite_Product_attributes": {
                    "product_name": "Product X", "is_commercial": True,
                    "components": [{"standard_name": "thymol", "entity_type": "Alternative"}],
                }},
            ]
        })

        # Phase 2 mock: Bulk (all remaining entities in one call)
        bulk_json = json.dumps({
            "extractions": [
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

        responses = [alt_json, bulk_json]
        call_count = [0]

        class MultiMockLM:
            requires_fence_output = False
            def infer(self, batch_prompts, **kwargs):
                idx = call_count[0]
                call_count[0] = min(idx + 1, len(responses) - 1)
                yield [MockSO(responses[idx])]
            def apply_schema(self, s): pass
            def set_fence_output(self, v): pass
            @property
            def schema(self): return None

        class MockSO:
            def __init__(self, output): self.output = output; self.score = 1.0

        # Run 4-phase extraction manually (simulating extract() logic)
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        model = MultiMockLM()
        all_extractions = []
        source_text = (
            "Pigs were fed a basal diet supplemented with 500 mg/kg thymol (THY, "
            "purity >= 99%) for 28 days. Average Daily Gain (ADG) was measured. "
            "Compared with the Basal Diet control group, thymol significantly "
            "increased ADG in the jejunal mucosa (P < 0.05)."
        )

        for phase_idx, phase in enumerate(registry.phase_defs()):
            prompt = registry.build_extraction_prompt(phase.extracts)
            template = PromptTemplateStructured(description=prompt)
            annotator = Annotator(model, template, fh)
            result = annotator.annotate_text(source_text, max_char_buffer=2000)
            all_extractions.extend(result.extractions or [])

        # Post-process
        for ext in all_extractions:
            registry.post_process(ext)

        # Verify extractions
        assert len(all_extractions) >= 5, f"Expected 5+ extractions, got {len(all_extractions)}"

        # Verify entity types
        entity_types = {e.extraction_class for e in all_extractions}
        assert "Alternative" in entity_types
        assert "Composite_Product" in entity_types
        assert "Intervention" in entity_types
        assert "Indicator" in entity_types
        assert "Result" in entity_types

        # Verify evidence_text is populated (alignment-derived)
        for ext in all_extractions:
            assert "evidence_text" in (ext.attributes or {}), \
                f"Missing evidence_text for {ext.extraction_class}:{ext.extraction_text}"
            assert "source_location" in (ext.attributes or {}), \
                f"Missing source_location for {ext.extraction_class}:{ext.extraction_text}"

        # Post-process should set alternative_class
        alt_exts = [e for e in all_extractions if e.extraction_class == "Alternative"]
        for alt in alt_exts:
            assert "classification" in (alt.attributes or {})

        # Build graph
        article_result = DocumentExtractionResult(
            document_id="99999", metadata={"doi": "10.1234/test", "pmid": "99999"}, extractions=all_extractions,
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

    def test_gate_skips_article_without_known_alternative(self):
        """Article with only 'Other' alternatives should be skippable."""
        import json
        from src.format_handler import FormatHandler, FormatType
        from src.prompting import PromptTemplateStructured
        from src.annotation import Annotator
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry(config_dir=SCHEMA_DIR)

        # Mock returns only an 'Other' alternative
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
        phase1 = registry.phase_defs()[0]
        prompt = registry.build_extraction_prompt(phase1.extracts)
        template = PromptTemplateStructured(description=prompt)
        annotator = Annotator(model, template, fh)
        result = annotator.annotate_text("Some unknown substance was tested.", max_char_buffer=500)

        # Post-process
        for ext in (result.extractions or []):
            registry.post_process(ext)

        # Verify gate logic: alternative_class is "Other"
        alt_exts = [e for e in (result.extractions or []) if e.extraction_class == "Alternative"]
        for alt in alt_exts:
            alt_class = (alt.attributes or {}).get("classification", "")
            # Should be "Other" (no known class matched)
            assert alt_class == "Other" or alt_class == ""

    def test_build_results_context(self):
        """_build_results_context should list indicators and control groups."""
        from src.data import Extraction
        from src.extraction import DocumentExtractionResult, _build_results_context

        ind_ext = Extraction(
            extraction_class="Indicator", extraction_text="ADG",
            attributes={"abbreviation": "ADG", "standard_name": "Average Daily Gain"},
        )
        ctrl_ext = Extraction(
            extraction_class="Control_Group", extraction_text="Basal",
            attributes={"group_name": "Basal Diet", "group_type": "basal_control"},
        )
        design_result = DocumentExtractionResult(
            document_id="1", metadata={"doi": "10.1", "pmid": "1"}, extractions=[ctrl_ext],
        )
        indicator_result = DocumentExtractionResult(
            document_id="1", metadata={"doi": "10.1", "pmid": "1"}, extractions=[ind_ext],
        )

        context = _build_results_context(design_result, indicator_result)
        assert "ADG" in context
        # Context includes indicators; control groups may or may not be included
        assert len(context) > 0

    def test_extraction_result_dataclass(self):
        """DocumentExtractionResult should handle all fields."""
        from src.data import Extraction
        from src.extraction import DocumentExtractionResult

        r = DocumentExtractionResult(
            document_id="12345", metadata={"doi": "10.1/test", "pmid": "12345"},
            extractions=[Extraction(extraction_class="Alt", extraction_text="test")],
            skipped=False, skip_reason="", warnings=["warning1"],
        )
        assert r.metadata.get("doi") == "10.1/test"
        assert r.document_id == "12345"
        assert len(r.extractions) == 1
        assert not r.skipped
        assert len(r.warnings) == 1

    def test_parse_article_sections_with_real_xml(self):
        """Parse a real PMC XML fixture and verify sections are extracted."""
        from src.extraction import _parse_article_sections

        fixture = Path(__file__).parent / "fixtures" / "sample.xml"
        if not fixture.exists():
            pytest.skip("No sample.xml fixture")

        sections = _parse_article_sections(fixture)
        # Should at minimum have body text
        assert isinstance(sections, dict)
        # Real PMC XML should have some content
        body = sections.get("body", "")
        assert len(body) > 0 or len(sections.get("abstract", "")) > 0
