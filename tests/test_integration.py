"""End-to-end integration tests for zhongnong-kg v2 pipeline."""
import tempfile
from pathlib import Path

SCHEMA_DIR = Path(__file__).parent.parent / "schemas"


class TestSchemaRegistryLoads:
    def test_all_entities_load(self):
        """Verify schema registry loads all 13 entity types from YAML config."""
        from src.schema_registry import SchemaRegistry
        registry = SchemaRegistry(config_dir=SCHEMA_DIR)
        names = registry.all_entity_names()
        assert "Alternative" in names
        assert "Composite_Product" in names
        assert "Result" in names
        assert "Indicator" in names
        assert "Intervention" in names
        assert len(names) >= 13

    def test_all_phases_load(self):
        """Verify all 4 extraction phases are defined."""
        from src.schema_registry import SchemaRegistry
        registry = SchemaRegistry(config_dir=SCHEMA_DIR)
        phases = registry.phase_defs()
        assert len(phases) >= 4
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
            properties={"name": "thymol", "alternative_class": "Plant_Extract"},
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
            attributes={"standard_name": "thymol", "alternative_class": "Plant_Extract"},
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
