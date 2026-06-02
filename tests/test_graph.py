"""Tests for graph builder."""
import pytest
import tempfile
from pathlib import Path

from src.graph import (
    entity_global_id,
    GraphNode,
    GraphEdge,
    Graph,
    build_graph,
    export_neo4j_csv,
)


class TestEntityGlobalId:
    def test_deterministic(self):
        id1 = entity_global_id("Alternative", "thymol")
        id2 = entity_global_id("Alternative", "thymol")
        assert id1 == id2

    def test_different_types(self):
        id1 = entity_global_id("Alternative", "thymol")
        id2 = entity_global_id("Indicator", "thymol")
        assert id1 != id2

    def test_article_scoped_includes_pmid(self):
        id1 = entity_global_id("Result", "ADG increased", "12345")
        id2 = entity_global_id("Result", "ADG increased", "67890")
        assert id1 != id2  # different PMIDs → different IDs

    def test_article_scoped_same_pmid(self):
        id1 = entity_global_id("Result", "ADG increased", "12345")
        id2 = entity_global_id("Result", "ADG increased", "12345")
        assert id1 == id2

    def test_prefix(self):
        gid = entity_global_id("Alternative", "thymol")
        assert gid.startswith("alte")

    def test_global_types_ignore_pmid(self):
        id1 = entity_global_id("Alternative", "thymol", "12345")
        id2 = entity_global_id("Alternative", "thymol", "67890")
        assert id1 == id2  # same name, different PMIDs → same ID

    def test_case_sensitive_name(self):
        id1 = entity_global_id("Alternative", "Thymol")
        id2 = entity_global_id("Alternative", "thymol")
        assert id1 == id2  # both normalized to lowercase

    def test_whitespace_normalized(self):
        id1 = entity_global_id("Alternative", "  thymol  ")
        id2 = entity_global_id("Alternative", "thymol")
        assert id1 == id2


class TestGraphNode:
    def test_basic_creation(self):
        node = GraphNode(
            id="alt_001",
            labels=["Alternative", "Entity"],
            properties={"name": "thymol", "alternative_class": "Plant_Extract"},
            source_pmids=["12345"],
        )
        assert node.id == "alt_001"
        assert "Alternative" in node.labels


class TestGraphEdge:
    def test_basic_creation(self):
        edge = GraphEdge(
            source_id="alt_001",
            target_id="ind_001",
            type="indicates",
            properties={"evidence_text": "thymol improved ADG"},
            source_pmids=["12345"],
        )
        assert edge.source_id == "alt_001"
        assert edge.type == "indicates"


class TestGraph:
    def test_basic_creation(self):
        graph = Graph(nodes=[], edges=[])
        assert graph.nodes == []
        assert graph.edges == []


class TestExportNeo4jCsv:
    def test_export_nodes_and_edges(self):
        node = GraphNode(
            id="alt_001",
            labels=["Alternative", "Entity"],
            properties={"name": "thymol", "alternative_class": "Plant_Extract"},
            source_pmids=["12345"],
        )
        edge = GraphEdge(
            source_id="alt_001",
            target_id="altclass_001",
            type="belongs_to",
            properties={"evidence_text": "thymol was used as a plant extract"},
            source_pmids=["12345"],
        )
        graph = Graph(nodes=[node], edges=[edge])

        with tempfile.TemporaryDirectory() as tmpdir:
            export_neo4j_csv(graph, Path(tmpdir))
            nodes_csv = Path(tmpdir) / "nodes.csv"
            edges_csv = Path(tmpdir) / "edges.csv"
            assert nodes_csv.exists()
            assert edges_csv.exists()

            nodes_content = nodes_csv.read_text()
            assert "entity_id:ID" in nodes_content
            assert "entity_type:LABEL" in nodes_content
            assert "thymol" in nodes_content

            edges_content = edges_csv.read_text()
            assert "source_id:START_ID" in edges_content
            assert "target_id:END_ID" in edges_content
            assert "relation_type:TYPE" in edges_content
            assert "belongs_to" in edges_content

    def test_csv_empty_graph(self):
        graph = Graph(nodes=[], edges=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            export_neo4j_csv(graph, Path(tmpdir))
            nodes_csv = Path(tmpdir) / "nodes.csv"
            edges_csv = Path(tmpdir) / "edges.csv"
            assert nodes_csv.exists()
            assert edges_csv.exists()
            # Should have headers but no data rows
            lines = nodes_csv.read_text().strip().split("\n")
            assert len(lines) == 1
            lines = edges_csv.read_text().strip().split("\n")
            assert len(lines) == 1

    def test_create_output_dir(self):
        graph = Graph(nodes=[], edges=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nested" / "dir"
            export_neo4j_csv(graph, output_dir)
            assert output_dir.exists()


class TestBuildGraph:
    def test_build_empty_results(self):
        from src.extraction import ArticleExtractionResult
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry()
        results = [
            ArticleExtractionResult(doi="10.1", pmid="12345", extractions=[]),
        ]
        graph = build_graph(results, registry)
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_creates_nodes_from_extractions(self):
        """Each Extraction becomes a GraphNode with entity_global_id."""
        from src.data import Extraction
        from src.extraction import ArticleExtractionResult
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry()
        ext = Extraction(
            extraction_class="Alternative",
            extraction_text="thymol",
            attributes={"standard_name": "thymol", "alternative_class": "Plant_Extract"},
        )
        result = ArticleExtractionResult(
            doi="10.1", pmid="12345", extractions=[ext],
        )
        graph = build_graph([result], registry)
        assert len(graph.nodes) == 1
        node = graph.nodes[0]
        assert node.id.startswith("alte")
        assert "Alternative" in node.labels
        assert node.properties["name"] == "thymol"
        assert "12345" in node.source_pmids

    def test_deduplicates_global_entity_across_articles(self):
        """Same Alternative in two articles → one node with merged source_pmids."""
        from src.data import Extraction
        from src.extraction import ArticleExtractionResult
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry()
        ext1 = Extraction(
            extraction_class="Alternative", extraction_text="thymol",
            attributes={"standard_name": "thymol"},
        )
        ext2 = Extraction(
            extraction_class="Alternative", extraction_text="thymol",
            attributes={"standard_name": "thymol"},
        )
        results = [
            ArticleExtractionResult(doi="10.1", pmid="111", extractions=[ext1]),
            ArticleExtractionResult(doi="10.2", pmid="222", extractions=[ext2]),
        ]
        graph = build_graph(results, registry)
        assert len(graph.nodes) == 1  # deduped
        node = graph.nodes[0]
        assert "111" in node.source_pmids
        assert "222" in node.source_pmids

    def test_article_scoped_entities_not_deduped(self):
        """Result entities from different articles get different IDs."""
        from src.data import Extraction
        from src.extraction import ArticleExtractionResult
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry()
        ext1 = Extraction(
            extraction_class="Result", extraction_text="ADG increased",
            attributes={"direction": "increased"},
        )
        ext2 = Extraction(
            extraction_class="Result", extraction_text="ADG increased",
            attributes={"direction": "increased"},
        )
        results = [
            ArticleExtractionResult(doi="10.1", pmid="111", extractions=[ext1]),
            ArticleExtractionResult(doi="10.2", pmid="222", extractions=[ext2]),
        ]
        graph = build_graph(results, registry)
        # Result is article-scoped → two separate nodes
        assert len(graph.nodes) == 2
        ids = {n.id for n in graph.nodes}
        assert len(ids) == 2  # different IDs

    def test_resolves_reference_edges(self):
        """Result.indicator_abbreviation reference creates corresponds_to edge."""
        from src.data import Extraction
        from src.extraction import ArticleExtractionResult
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry()
        ind_ext = Extraction(
            extraction_class="Indicator", extraction_text="ADG",
            attributes={"abbreviation": "ADG", "standard_name": "Average Daily Gain"},
        )
        res_ext = Extraction(
            extraction_class="Result", extraction_text="ADG",
            attributes={
                "indicator_abbreviation": "ADG",
                "direction": "increased",
                "significance_level": "p_less_0.05",
                "evidence_text": "ADG significantly increased",
                "source_location": "Results",
            },
        )
        result = ArticleExtractionResult(
            doi="10.1", pmid="12345", extractions=[ind_ext, res_ext],
        )
        graph = build_graph([result], registry)
        assert len(graph.nodes) == 2
        # Should have a corresponds_to edge from Result to Indicator
        ref_edges = [e for e in graph.edges if e.type == "corresponds_to"]
        assert len(ref_edges) == 1
        edge = ref_edges[0]
        # source is Result node, target is Indicator node
        result_node = next(n for n in graph.nodes if "Result" in n.labels)
        indicator_node = next(n for n in graph.nodes if "Indicator" in n.labels)
        assert edge.source_id == result_node.id
        assert edge.target_id == indicator_node.id

    def test_skips_empty_extraction_text(self):
        """Extractions with empty text should be skipped."""
        from src.data import Extraction
        from src.extraction import ArticleExtractionResult
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry()
        ext = Extraction(extraction_class="Alternative", extraction_text="")
        result = ArticleExtractionResult(doi="10.1", pmid="12345", extractions=[ext])
        graph = build_graph([result], registry)
        assert len(graph.nodes) == 0

    def test_merge_properties_on_dedup(self):
        """First article's properties take priority on dedup."""
        from src.data import Extraction
        from src.extraction import ArticleExtractionResult
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry()
        ext1 = Extraction(
            extraction_class="Alternative", extraction_text="thymol",
            attributes={"standard_name": "thymol", "abbreviation": "THY"},
        )
        ext2 = Extraction(
            extraction_class="Alternative", extraction_text="thymol",
            attributes={"standard_name": "thymol", "cas_number": "89-83-8"},
        )
        results = [
            ArticleExtractionResult(doi="10.1", pmid="111", extractions=[ext1]),
            ArticleExtractionResult(doi="10.2", pmid="222", extractions=[ext2]),
        ]
        graph = build_graph(results, registry)
        assert len(graph.nodes) == 1
        node = graph.nodes[0]
        # First article's properties kept; second fills missing
        assert node.properties.get("abbreviation") == "THY"
        assert node.properties.get("cas_number") == "89-83-8"
