"""Graph builder — deduplicates entities across articles, resolves edges, and
exports Neo4j-compatible CSV files.
"""
from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.extraction import ArticleExtractionResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GraphNode:
    id: str
    labels: list[str]           # Neo4j labels e.g. ["Alternative", "Entity"]
    properties: dict[str, Any]  # all attributes
    source_pmids: list[str] = field(default_factory=list)


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    type: str                   # relation type
    properties: dict[str, Any] = field(default_factory=dict)
    source_pmids: list[str] = field(default_factory=list)


@dataclass
class Graph:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Entity types
# ---------------------------------------------------------------------------

# Entity types that are scoped to a single article (PMID-dependent)
_ARTICLE_SCOPED_TYPES = frozenset({
    "Result",
    "Experiment",
    "Intervention",
    "Swine",
    "Swine_Model",
    "Control_Group",
    "Literature",
})

# Entity types that are global (name-based, shared across articles)
_GLOBAL_TYPES = frozenset({
    "Alternative",
    "Alternative_Class",
    "Composite_Product",
    "Tissue_Site",
    "Indicator",
    "Method",
})


def _is_article_scoped(entity_type: str) -> bool:
    """Return True if the entity type is article-scoped (PMID-dependent)."""
    return entity_type in _ARTICLE_SCOPED_TYPES


def _normalize_name(name: str) -> str:
    """Strip whitespace and collapse internal whitespace."""
    return re.sub(r"\s+", " ", name.strip())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def entity_global_id(entity_type: str, primary_text: str, article_pmid: str | None = None) -> str:
    """Generate a deterministic global ID for an entity.

    Article-scoped types (Result, Experiment, Intervention, Swine, Swine_Model,
    Control_Group, Literature) include the PMID in the hash so that the same
    text from different articles produces different IDs.

    Global types (Alternative, Alternative_Class, Composite_Product, Tissue_Site,
    Indicator, Method) produce the same ID regardless of PMID — the same name
    always maps to the same node.

    Parameters
    ----------
    entity_type : str
        The entity class name (e.g. ``"Alternative"``, ``"Result"``).
    primary_text : str
        The primary identifying text (name) of the entity.
    article_pmid : str or None
        PMID of the article.  Required for article-scoped types, ignored for
        global types.

    Returns
    -------
    str
        Global ID like ``"alte_a1b2c3d4e5f6"``.
    """
    norm_name = _normalize_name(primary_text).lower()

    if _is_article_scoped(entity_type):
        pmid = article_pmid or ""
        key = f"{entity_type}:{norm_name}:{pmid}"
    else:
        key = f"{entity_type}:{norm_name}"

    h = hashlib.sha256(key.encode()).hexdigest()[:12]
    prefix = entity_type.lower()[:4]
    return f"{prefix}_{h}"


def _build_node_key(node: GraphNode) -> str:
    """Return a comparable key for deduplication purposes."""
    return node.id


def build_graph(
    results: list[ArticleExtractionResult],
    registry: Any = None,
) -> Graph:
    """Build a deduplicated Graph from a list of article extraction results.

    Parameters
    ----------
    results : list[ArticleExtractionResult]
        Extraction results from one or more articles.
    registry : SchemaRegistry or None
        Schema registry for reference/inline-relation metadata.  When ``None``
        edge resolution is skipped.

    Returns
    -------
    Graph
        Deduplicated graph with nodes and edges.
    """
    nodes_by_id: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []

    # First pass: create nodes from all extractions
    for article_result in results:
        pmid = article_result.pmid

        for ext in article_result.extractions:
            entity_type = ext.extraction_class
            primary_text = ext.extraction_text
            if not primary_text:
                continue

            node_id = entity_global_id(entity_type, primary_text, pmid)

            # Build properties from extraction attributes
            properties: dict[str, Any] = {
                "name": primary_text,
                "entity_type": entity_type,
            }
            if ext.attributes:
                properties.update(ext.attributes)

            labels = [entity_type]
            if "Entity" not in labels:
                labels.append("Entity")

            if node_id in nodes_by_id:
                # Merge source_pmids
                existing = nodes_by_id[node_id]
                if pmid not in existing.source_pmids:
                    existing.source_pmids.append(pmid)
                # Merge properties (existing takes priority)
                for k, v in properties.items():
                    if k not in existing.properties:
                        existing.properties[k] = v
            else:
                nodes_by_id[node_id] = GraphNode(
                    id=node_id,
                    labels=labels,
                    properties=properties,
                    source_pmids=[pmid] if pmid else [],
                )

    # Second pass: resolve edges via registry definitions
    if registry is not None:
        _resolve_edges(results, nodes_by_id, edges, registry)

    return Graph(nodes=list(nodes_by_id.values()), edges=edges)


def _resolve_edges(
    results: list[ArticleExtractionResult],
    nodes_by_id: dict[str, GraphNode],
    edges: list[GraphEdge],
    registry: Any,
) -> None:
    """Resolve edges from inline relations and reference fields."""
    for article_result in results:
        pmid = article_result.pmid

        for ext in article_result.extractions:
            entity_type = ext.extraction_class
            primary_text = ext.extraction_text
            if not primary_text:
                continue

            try:
                entity_def = registry.entity_def(entity_type)
            except (KeyError, AttributeError):
                continue

            source_id = entity_global_id(entity_type, primary_text, pmid)

            # --- Resolve reference fields ---
            for ref in entity_def.references:
                ref_value = _get_attr(ext.attributes, ref.name)
                if ref_value is None:
                    continue

                target_id = entity_global_id(ref.target_entity, str(ref_value), pmid)
                _add_edge(edges, source_id, target_id, ref.edge_type, pmid, ext.attributes)

            # --- Resolve inline relations ---
            for ir in entity_def.inline_relations:
                ir_values = _get_attr(ext.attributes, ir.via_field)
                if ir_values is None:
                    continue

                if not isinstance(ir_values, list):
                    ir_values = [ir_values]

                for ir_val in ir_values:
                    if not ir_val:
                        continue
                    for target_type in ir.target:
                        target_id = entity_global_id(target_type, str(ir_val), pmid)
                        _add_edge(edges, source_id, target_id, ir.name, pmid, ext.attributes)


def _get_attr(attributes: dict[str, Any] | None, key: str) -> Any:
    """Safely get an attribute value."""
    if attributes is None:
        return None
    return attributes.get(key)


def _add_edge(
    edges: list[GraphEdge],
    source_id: str,
    target_id: str,
    edge_type: str,
    pmid: str,
    attrs: dict[str, Any] | None,
) -> None:
    """Add an edge if source and target are different, deduplicating by (source, target, type)."""
    if source_id == target_id:
        return

    # Check for existing edge
    for e in edges:
        if e.source_id == source_id and e.target_id == target_id and e.type == edge_type:
            if pmid and pmid not in e.source_pmids:
                e.source_pmids.append(pmid)
            return

    evidence_text = None
    if attrs:
        evidence_text = attrs.get("evidence_text")

    props: dict[str, Any] = {}
    if evidence_text:
        props["evidence_text"] = evidence_text

    edges.append(GraphEdge(
        source_id=source_id,
        target_id=target_id,
        type=edge_type,
        properties=props,
        source_pmids=[pmid] if pmid else [],
    ))


def export_neo4j_csv(graph: Graph, output_dir: str | Path) -> None:
    """Export a Graph to Neo4j-compatible CSV files.

    Writes ``nodes.csv`` and ``edges.csv`` into *output_dir*, creating the
    directory if it does not exist.

    Parameters
    ----------
    graph : Graph
        The graph to export.
    output_dir : str or Path
        Directory to write CSV files into.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Nodes CSV ---
    _write_nodes_csv(graph, output_dir / "nodes.csv")

    # --- Edges CSV ---
    _write_edges_csv(graph, output_dir / "edges.csv")


def _write_nodes_csv(graph: Graph, path: Path) -> None:
    """Write nodes.csv with Neo4j-compatible headers."""
    # Collect all property names across all nodes
    all_props: set[str] = set()
    for node in graph.nodes:
        all_props.update(node.properties.keys())

    fieldnames = ["entity_id:ID", "entity_type:LABEL"] + list(all_props) + ["source_pmids"]

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for node in graph.nodes:
            row: dict[str, Any] = {
                "entity_id:ID": node.id,
                "entity_type:LABEL": ";".join(node.labels),
                "source_pmids": "|".join(node.source_pmids),
            }
            row.update(node.properties)
            writer.writerow(row)


def _write_edges_csv(graph: Graph, path: Path) -> None:
    """Write edges.csv with Neo4j-compatible headers."""
    # Collect all property names across all edges
    all_props: set[str] = set()
    for edge in graph.edges:
        all_props.update(edge.properties.keys())

    fieldnames = ["source_id:START_ID", "target_id:END_ID", "relation_type:TYPE"] + list(all_props) + ["source_pmids"]

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for edge in graph.edges:
            row: dict[str, Any] = {
                "source_id:START_ID": edge.source_id,
                "target_id:END_ID": edge.target_id,
                "relation_type:TYPE": edge.type,
                "source_pmids": "|".join(edge.source_pmids),
            }
            row.update(edge.properties)
            writer.writerow(row)
