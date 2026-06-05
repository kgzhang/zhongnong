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
    _normalize_name,
    _deduplicate_global_nodes,
)


def _data_nodes(graph: Graph) -> int:
    """Count nodes that are NOT pre-built classification nodes."""
    return sum(1 for n in graph.nodes
               if not (n.properties or {}).get("_prebuilt", False))


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
        # With global_types set, PMID is ignored → same ID
        global_types = frozenset({"Alternative"})
        id1 = entity_global_id("Alternative", "thymol", "12345", global_types=global_types)
        id2 = entity_global_id("Alternative", "thymol", "67890", global_types=global_types)
        assert id1 == id2  # same name, different PMIDs → same ID

    def test_article_scoped_when_not_global(self):
        # Without global_types, all types are article-scoped → different IDs
        id1 = entity_global_id("Alternative", "thymol", "12345")
        id2 = entity_global_id("Alternative", "thymol", "67890")
        assert id1 != id2  # different PMIDs → different IDs

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
        from src.extraction import DocumentExtractionResult
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry()
        results = [
            DocumentExtractionResult(document_id="12345", metadata={"doi": "10.1", "pmid": "12345"}, extractions=[]),
        ]
        graph = build_graph(results, registry)
        # Pre-built classification nodes always exist
        prebuilt = [n for n in graph.nodes if (n.properties or {}).get("_prebuilt")]
        assert len(prebuilt) >= 1, "Pre-built classification nodes should exist"
        assert len([n for n in graph.nodes if not (n.properties or {}).get("_prebuilt")]) == 0
        assert len(graph.edges) == 0

    def test_creates_nodes_from_extractions(self):
        """Each Extraction becomes a GraphNode with entity_global_id."""
        from src.data import Extraction
        from src.extraction import DocumentExtractionResult
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry()
        ext = Extraction(
            extraction_class="Alternative",
            extraction_text="thymol",
            attributes={"standard_name": "thymol", "alternative_class": "Plant_Extract"},
        )
        result = DocumentExtractionResult(
            document_id="12345", metadata={"doi": "10.1", "pmid": "12345"}, extractions=[ext],
        )
        graph = build_graph([result], registry)
        assert _data_nodes(graph) == 1
        node = graph.nodes[0]
        assert node.id.startswith("alte")
        assert "Alternative" in node.labels
        assert node.properties["name"] == "thymol"
        assert "12345" in node.source_pmids

    def test_deduplicates_global_entity_across_articles(self):
        """Same Alternative in two articles → one node with merged source_pmids."""
        from src.data import Extraction
        from src.extraction import DocumentExtractionResult
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry()
        global_types = frozenset({"Alternative"})
        ext1 = Extraction(
            extraction_class="Alternative", extraction_text="thymol",
            attributes={"standard_name": "thymol"},
        )
        ext2 = Extraction(
            extraction_class="Alternative", extraction_text="thymol",
            attributes={"standard_name": "thymol"},
        )
        results = [
            DocumentExtractionResult(document_id="111", metadata={"doi": "10.1", "pmid": "111"}, extractions=[ext1]),
            DocumentExtractionResult(document_id="222", metadata={"doi": "10.2", "pmid": "222"}, extractions=[ext2]),
        ]
        graph = build_graph(results, registry, global_types=global_types)
        assert _data_nodes(graph) == 1  # deduped because Alternative is global
        node = graph.nodes[0]
        assert "111" in node.source_pmids
        assert "222" in node.source_pmids

    def test_article_scoped_entities_not_deduped(self):
        """Result entities from different articles get different IDs."""
        from src.data import Extraction
        from src.extraction import DocumentExtractionResult
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
            DocumentExtractionResult(document_id="111", metadata={"doi": "10.1", "pmid": "111"}, extractions=[ext1]),
            DocumentExtractionResult(document_id="222", metadata={"doi": "10.2", "pmid": "222"}, extractions=[ext2]),
        ]
        graph = build_graph(results, registry)
        # Result is article-scoped → two separate nodes
        assert _data_nodes(graph) == 2
        data_ids = {n.id for n in graph.nodes
                    if not (n.properties or {}).get("_prebuilt")}
        assert len(data_ids) == 2  # different IDs

    def test_resolves_reference_edges(self):
        """Result.indicator_abbreviation reference creates corresponds_to edge."""
        from src.data import Extraction
        from src.extraction import DocumentExtractionResult
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
        result = DocumentExtractionResult(
            document_id="12345", metadata={"doi": "10.1", "pmid": "12345"}, extractions=[ind_ext, res_ext],
        )
        graph = build_graph([result], registry)
        assert _data_nodes(graph) == 2
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
        from src.extraction import DocumentExtractionResult
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry()
        ext = Extraction(extraction_class="Alternative", extraction_text="")
        result = DocumentExtractionResult(document_id="12345", metadata={"doi": "10.1", "pmid": "12345"}, extractions=[ext])
        graph = build_graph([result], registry)
        assert _data_nodes(graph) == 0

    def test_merge_properties_on_dedup(self):
        """First article's properties take priority on dedup."""
        from src.data import Extraction
        from src.extraction import DocumentExtractionResult
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry()
        global_types = frozenset({"Alternative"})
        ext1 = Extraction(
            extraction_class="Alternative", extraction_text="thymol",
            attributes={"standard_name": "thymol", "abbreviation": "THY"},
        )
        ext2 = Extraction(
            extraction_class="Alternative", extraction_text="thymol",
            attributes={"standard_name": "thymol", "cas_number": "89-83-8"},
        )
        results = [
            DocumentExtractionResult(document_id="111", metadata={"doi": "10.1", "pmid": "111"}, extractions=[ext1]),
            DocumentExtractionResult(document_id="222", metadata={"doi": "10.2", "pmid": "222"}, extractions=[ext2]),
        ]
        graph = build_graph(results, registry, global_types=global_types)
        assert _data_nodes(graph) == 1
        node = graph.nodes[0]
        # First article's properties kept; second fills missing
        assert node.properties.get("abbreviation") == "THY"
        assert node.properties.get("cas_number") == "89-83-8"


class TestNormalizeName:
    """Tests for the _normalize_name helper (Issue 5A)."""

    def test_strips_whitespace(self):
        assert _normalize_name("  thymol  ") == "thymol"

    def test_lowercases(self):
        assert _normalize_name("Thymol") == "thymol"
        assert _normalize_name("THYMOL") == "thymol"

    def test_replaces_hyphens(self):
        assert _normalize_name("microbe-derived") == "microbe derived"

    def test_replaces_underscores(self):
        assert _normalize_name("microbe_derived") == "microbe derived"

    def test_collapses_internal_whitespace(self):
        assert _normalize_name("microbe   derived") == "microbe derived"

    def test_combined_normalization(self):
        """Hyphens + capitals + whitespace all normalized together."""
        result = _normalize_name("  Microbe-Derived_Antioxidants  ")
        assert result == "microbe derived antioxidants"


class TestEntityGlobalIdNormalization:
    """Issue 5A: hyphens and case variants map to the same ID."""

    def test_hyphen_vs_space(self):
        gid1 = entity_global_id("Alternative", "microbe-derived antioxidants")
        gid2 = entity_global_id("Alternative", "microbe derived antioxidants")
        assert gid1 == gid2

    def test_case_variants(self):
        gid1 = entity_global_id("Alternative", "Microbe-derived Antioxidants")
        gid2 = entity_global_id("Alternative", "microbe-derived antioxidants")
        assert gid1 == gid2

    def test_hyphen_space_case_mix(self):
        gid1 = entity_global_id("Alternative", "Microbe-Derived Antioxidants")
        gid2 = entity_global_id("Alternative", "microbe derived antioxidants")
        assert gid1 == gid2


class TestBuildGraphPrimaryTextIdentity:
    """Issue 5B: use primary_text (standard_name) as identity key."""

    def test_uses_standard_name_for_id(self):
        """When extraction_text is an abbreviation but standard_name is the
        full name, the node ID should be based on standard_name."""
        from src.data import Extraction
        from src.extraction import DocumentExtractionResult
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry()
        global_types = frozenset({"Alternative"})
        ext = Extraction(
            extraction_class="Alternative",
            extraction_text="MA",
            attributes={
                "standard_name": "microbe-derived antioxidants",
                "abbreviation": "MA",
            },
        )
        result = DocumentExtractionResult(
            document_id="12345", metadata={"doi": "10.1", "pmid": "12345"},
            extractions=[ext],
        )
        graph = build_graph([result], registry, global_types=global_types)
        assert _data_nodes(graph) == 1
        node = graph.nodes[0]
        # Properties keep the original extraction_text as "name"
        assert node.properties["name"] == "MA"
        # But the node ID should be derived from standard_name, not "MA"
        expected_id = entity_global_id(
            "Alternative", "microbe-derived antioxidants",
            global_types=global_types,
        )
        assert node.id == expected_id

    def test_abbreviation_and_full_name_same_node(self):
        """Two extractions of the same entity — one using full name as primary
        text, one using abbreviation — should produce a single node."""
        from src.data import Extraction
        from src.extraction import DocumentExtractionResult
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry()
        global_types = frozenset({"Alternative"})

        # Chunk A: full name
        ext1 = Extraction(
            extraction_class="Alternative",
            extraction_text="microbe-derived antioxidants",
            attributes={
                "standard_name": "microbe-derived antioxidants",
                "abbreviation": "MA",
            },
        )
        # Chunk B: abbreviation as primary, but standard_name given
        ext2 = Extraction(
            extraction_class="Alternative",
            extraction_text="MA",
            attributes={
                "standard_name": "microbe-derived antioxidants",
                "abbreviation": "MA",
            },
        )
        results = [
            DocumentExtractionResult(
                document_id="111", metadata={"doi": "10.1", "pmid": "111"},
                extractions=[ext1],
            ),
            DocumentExtractionResult(
                document_id="222", metadata={"doi": "10.2", "pmid": "222"},
                extractions=[ext2],
            ),
        ]
        graph = build_graph(results, registry, global_types=global_types)
        assert _data_nodes(graph) == 1


class TestBuildGraphDedup:
    """Issue 5C: post-processing dedup merges abbreviation-only extractions."""

    def test_abbreviation_dedup(self):
        """When one extraction uses the full name and another uses the
        abbreviation WITHOUT a standard_name attribute, the dedup step
        should merge them."""
        from src.data import Extraction
        from src.extraction import DocumentExtractionResult
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry()
        global_types = frozenset({"Alternative"})

        # Chunk A: full name with abbreviation field
        ext1 = Extraction(
            extraction_class="Alternative",
            extraction_text="microbe-derived antioxidants",
            attributes={
                "standard_name": "microbe-derived antioxidants",
                "abbreviation": "MA",
            },
        )
        # Chunk B: abbreviation only, NO standard_name
        ext2 = Extraction(
            extraction_class="Alternative",
            extraction_text="MA",
            attributes={},
        )
        results = [
            DocumentExtractionResult(
                document_id="111", metadata={"doi": "10.1", "pmid": "111"},
                extractions=[ext1],
            ),
            DocumentExtractionResult(
                document_id="222", metadata={"doi": "10.2", "pmid": "222"},
                extractions=[ext2],
            ),
        ]
        graph = build_graph(results, registry, global_types=global_types)
        # Should be 1 node after dedup, because "MA" matches ext1's abbreviation
        assert _data_nodes(graph) == 1
        node = graph.nodes[0]
        assert "111" in node.source_pmids
        assert "222" in node.source_pmids

    def test_dedup_merges_edge_pmids(self):
        """After dedup merge, edges should have correct source_pmids and
        no stale references."""
        from src.data import Extraction
        from src.extraction import DocumentExtractionResult
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry()
        global_types = frozenset({"Alternative"})

        ext1 = Extraction(
            extraction_class="Alternative",
            extraction_text="thymol",
            attributes={"standard_name": "thymol", "abbreviation": "THY"},
        )
        ext2 = Extraction(
            extraction_class="Alternative",
            extraction_text="THY",
            attributes={},
        )
        results = [
            DocumentExtractionResult(
                document_id="111", metadata={"doi": "10.1", "pmid": "111"},
                extractions=[ext1],
            ),
            DocumentExtractionResult(
                document_id="222", metadata={"doi": "10.2", "pmid": "222"},
                extractions=[ext2],
            ),
        ]
        graph = build_graph(results, registry, global_types=global_types)
        assert _data_nodes(graph) == 1
