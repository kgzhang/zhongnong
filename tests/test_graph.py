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
        from src.data import Extraction
        from src.extraction import ArticleExtractionResult
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry()
        results = [
            ArticleExtractionResult(doi="10.1", pmid="12345", extractions=[]),
        ]
        graph = build_graph(results, registry)
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
