"""Graph builder — deduplicates entities across articles, resolves edges, and
exports Neo4j-compatible CSV files.

Generic — no entity-type hardcoding.  The distinction between article-scoped
and global entity types is controlled by the *global_types* parameter.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.extraction import DocumentExtractionResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GraphNode:
    id: str
    labels: list[str]           # Neo4j labels e.g. ["Chemical", "Entity"]
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
# Helpers
# ---------------------------------------------------------------------------


def _normalize_name(name: str | None) -> str:
    """Aggressively normalise entity names for ID generation.

    Lowercase, replace hyphens and underscores with spaces, collapse all
    whitespace.  This ensures ``"Microbe-derived antioxidants"`` and
    ``"microbe derived antioxidants"`` hash to the same global ID.
    """
    if not name:
        return ""
    normalized = name.strip().lower()
    normalized = re.sub(r"[-_]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def entity_global_id(
    entity_type: str,
    primary_text: str,
    article_pmid: str | None = None,
    *,
    global_types: frozenset[str] | set[str] | None = None,
) -> str:
    """Generate a deterministic global ID for an entity.

    When *entity_type* is in *global_types*, the ID is based solely on the
    entity type and name (shared across articles).  Otherwise the PMID is
    included, making the ID article-specific.

    Parameters
    ----------
    entity_type:
        The entity class name (e.g. ``"Chemical"``, ``"Measurement"``).
    primary_text:
        The primary identifying text (name) of the entity.
    article_pmid:
        PMID of the article.  Required for non-global types, ignored for
        global types.
    global_types:
        Set of entity type names that are shared across articles (deduplicated
        by name).  When ``None``, all types are article-scoped (include PMID).

    Returns
    -------
    str
        Global ID like ``"alte_a1b2c3d4e5f6"``.
    """
    norm_name = _normalize_name(primary_text)

    if global_types and entity_type in global_types:
        key = f"{entity_type}:{norm_name}"
    else:
        pmid = article_pmid or ""
        key = f"{entity_type}:{norm_name}:{pmid}"

    h = hashlib.sha256(key.encode()).hexdigest()[:12]
    prefix = entity_type.lower()[:4]
    return f"{prefix}_{h}"


def build_graph(
    results: list[DocumentExtractionResult],
    registry: Any = None,
    *,
    global_types: frozenset[str] | set[str] | None = None,
) -> Graph:
    """Build a deduplicated Graph from a list of article extraction results.

    Parameters
    ----------
    results:
        Extraction results from one or more articles.
    registry:
        SchemaRegistry or None.  When ``None`` edge resolution is skipped.
    global_types:
        Set of entity type names that should be deduplicated across articles
        (i.e. the same name always maps to the same node).  Types not in this
        set are scoped to their source article.

    Returns
    -------
    Graph
        Deduplicated graph with nodes and edges.
    """
    nodes_by_id: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []

    # First pass: create nodes from all extractions
    for article_result in results:
        pmid = article_result.document_id

        for ext in article_result.extractions:
            entity_type = ext.extraction_class
            if not ext.extraction_text:
                continue

            # Use the entity definition's primary_text field as the identity
            # key, falling back to extraction_text.  This ensures e.g. that an
            # entity extracted with an abbreviation gets the same
            # global ID as one extracted with the canonical name.
            identity_key = ext.extraction_text
            if registry is not None:
                try:
                    ed = registry.entity_def(entity_type)
                    if ed.primary_text and ext.attributes:
                        identity_key = ext.attributes.get(
                            ed.primary_text, ext.extraction_text
                        )
                except (KeyError, AttributeError):
                    pass

            node_id = entity_global_id(
                entity_type, identity_key, pmid, global_types=global_types
            )

            # Build properties from extraction attributes
            properties: dict[str, Any] = {
                "name": ext.extraction_text,
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

    # Post-processing: create classification/category nodes from post-process
    # enum fields, with belongs_to edges. Generic — driven by registry metadata.
    if registry is not None:
        _create_classification_nodes(nodes_by_id, edges, registry, global_types)

    # Post-dedup: for global entity types, merge nodes whose extraction_text
    # matches another node's abbreviation or standard_name (handles the case
    # where the LLM uses the abbreviation as primary text in one chunk and the
    # full name in another).  Returns a remap dict so edges can be updated.
    node_remap: dict[str, str] = {}
    if global_types:
        node_remap = _deduplicate_global_nodes(nodes_by_id, global_types)

    # Second pass: resolve edges via registry definitions
    if registry is not None:
        _resolve_edges(results, nodes_by_id, edges, registry, global_types, node_remap)

    # After edge resolution, remap edge source/target IDs for merged nodes
    # and remove duplicate edges that result from node merges.
    if node_remap:
        _remap_edges_after_dedup(edges, node_remap)

    return Graph(nodes=list(nodes_by_id.values()), edges=edges)


def _create_classification_nodes(
    nodes_by_id: dict[str, GraphNode],
    edges: list[GraphEdge],
    registry: Any,
    global_types: frozenset[str] | set[str] | None = None,
) -> None:
    """Create classification/category nodes from post-process enum fields.

    Generic — reads all entity definitions from the registry.  For each entity
    type that has a ``source: post, type: enum`` attribute, creates one
    ``<EntityType>_Class`` node per unique enum value found on that entity's
    nodes, connected via ``belongs_to`` edges.
    """
    # Build a map: entity_type → classification_field_name
    try:
        all_entity_names = registry.all_entity_names()
    except AttributeError:
        return

    for ename in all_entity_names:
        try:
            ed = registry.entity_def(ename)
        except (KeyError, AttributeError):
            continue

        # Find the first post-process enum field — this is the classification field
        class_field: str | None = None
        for a in ed.attributes:
            if a.source == "post" and a.type == "enum":
                class_field = a.name
                break
        if class_field is None:
            continue

        class_node_type = f"{ename}_Class"
        class_set: dict[str, str] = {}  # classification value -> class_node_id

        for node_id, node in list(nodes_by_id.items()):
            if node.properties.get("entity_type") != ename:
                continue
            classification = node.properties.get(class_field)
            if not classification or not isinstance(classification, str) or not classification.strip():
                continue
            class_name = classification.strip()

            if class_name not in class_set:
                class_node_id = entity_global_id(
                    class_node_type, class_name, None, global_types=global_types
                )
                class_set[class_name] = class_node_id

                class_node = GraphNode(
                    id=class_node_id,
                    labels=[class_node_type, "Entity"],
                    properties={
                        "name": class_name,
                        "entity_type": class_node_type,
                        "class_name": class_name,
                    },
                    source_pmids=[],
                )
                if class_node_id in nodes_by_id:
                    existing = nodes_by_id[class_node_id]
                    for pmid in node.source_pmids:
                        if pmid not in existing.source_pmids:
                            existing.source_pmids.append(pmid)
                    if "class_name" not in existing.properties:
                        existing.properties["class_name"] = class_name
                else:
                    nodes_by_id[class_node_id] = class_node

            # Collect PMIDs for the class node
            class_node_id = class_set[class_name]
            if class_node_id in nodes_by_id:
                for pmid in node.source_pmids:
                    if pmid not in nodes_by_id[class_node_id].source_pmids:
                        nodes_by_id[class_node_id].source_pmids.append(pmid)

            # Add belongs_to edge from entity -> class node
            _add_edge(edges, node_id, class_node_id, "belongs_to", "", None)


def _deduplicate_global_nodes(
    nodes_by_id: dict[str, GraphNode],
    global_types: frozenset[str] | set[str],
) -> dict[str, str]:
    """Merge global-scope nodes that refer to the same real-world entity.

    Handles the case where the LLM uses an abbreviation (e.g. ``"MA"``) as the
    primary text in one chunk and the full name (``"microbe-derived
    antioxidants"``) in another.  When node A's ``extraction_text`` matches
    node B's ``abbreviation`` or ``standard_name``, node A is merged into
    node B.

    Returns a ``{child_id: parent_id}`` map for downstream edge remapping.
    """
    merges: dict[str, str] = {}  # child_node_id → parent_node_id

    for node_id, node in list(nodes_by_id.items()):
        entity_type = node.properties.get("entity_type", "")
        if entity_type not in global_types:
            continue

        node_name = _normalize_name(node.properties.get("name", ""))
        if not node_name:
            continue

        for other_id, other in nodes_by_id.items():
            if other_id == node_id:
                continue
            other_abbrev = _normalize_name(other.properties.get("abbreviation") or "")
            other_std = _normalize_name(other.properties.get("standard_name") or "")

            if (other_abbrev and other_abbrev == node_name) or \
               (other_std and other_std == node_name):
                merges[node_id] = other_id
                break

    # Apply merges
    for child_id, parent_id in merges.items():
        if child_id not in nodes_by_id or parent_id not in nodes_by_id:
            continue
        child = nodes_by_id[child_id]
        parent = nodes_by_id[parent_id]

        for pmid in child.source_pmids:
            if pmid not in parent.source_pmids:
                parent.source_pmids.append(pmid)
        for k, v in child.properties.items():
            if k not in parent.properties:
                parent.properties[k] = v
        del nodes_by_id[child_id]

    return merges


def _remap_edges_after_dedup(
    edges: list[GraphEdge],
    node_remap: dict[str, str],
) -> None:
    """Apply node-ID remapping after dedup and remove duplicate edges.

    After merging nodes, edges may reference old (deleted) node IDs.
    This remaps them to the surviving parent IDs and deduplicates edges
    that now share the same (source, target, type) triple.
    """
    # Remap source/target IDs
    for e in edges:
        if e.source_id in node_remap:
            e.source_id = node_remap[e.source_id]
        if e.target_id in node_remap:
            e.target_id = node_remap[e.target_id]
        # Drop self-edges that may result from merging
        if e.source_id == e.target_id:
            e.source_id = ""  # mark for removal

    # Remove self-edges (marked with empty source_id)
    edges[:] = [e for e in edges if e.source_id]

    # Deduplicate edges with the same (source, target, type) triple
    seen: dict[tuple[str, str, str], GraphEdge] = {}
    deduped: list[GraphEdge] = []
    for e in edges:
        key = (e.source_id, e.target_id, e.type)
        if key in seen:
            existing = seen[key]
            for pmid in e.source_pmids:
                if pmid not in existing.source_pmids:
                    existing.source_pmids.append(pmid)
        else:
            seen[key] = e
            deduped.append(e)
    edges[:] = deduped


def _resolve_edges(
    results: list[DocumentExtractionResult],
    nodes_by_id: dict[str, GraphNode],
    edges: list[GraphEdge],
    registry: Any,
    global_types: frozenset[str] | set[str] | None = None,
    node_remap: dict[str, str] | None = None,
) -> None:
    """Resolve edges from inline relations and reference fields.

    Accepts *node_remap* to correct edge source/target IDs when nodes have
    been merged by the post-dedup step.
    """
    remap = node_remap or {}

    for article_result in results:
        pmid = article_result.document_id

        for ext in article_result.extractions:
            entity_type = ext.extraction_class
            if not ext.extraction_text:
                continue

            try:
                entity_def = registry.entity_def(entity_type)
            except (KeyError, AttributeError):
                continue

            # Use primary_text field from entity definition as identity key
            identity_key = ext.extraction_text
            if entity_def.primary_text and ext.attributes:
                identity_key = ext.attributes.get(
                    entity_def.primary_text, ext.extraction_text
                )

            source_id = entity_global_id(
                entity_type, identity_key, pmid, global_types=global_types
            )
            # Apply node remap if this ID was merged away
            source_id = remap.get(source_id, source_id)

            # --- Resolve reference fields ---
            for ref in entity_def.references:
                ref_value = _get_attr(ext.attributes, ref.name)
                if ref_value is None:
                    # Auto-detect: scan all attribute values for a match
                    # against target entity names.  LLMs often output
                    # semantically meaningful values (e.g. "ADG" for
                    # indicator) without using the exact reference field name.
                    ref_value = _auto_detect_ref(
                        ext, ref, nodes_by_id, registry, global_types
                    )
                if ref_value is None:
                    continue

                target_id = entity_global_id(
                    ref.target_entity, str(ref_value), pmid, global_types=global_types
                )
                target_id = remap.get(target_id, target_id)
                _add_edge(edges, source_id, target_id, ref.edge_type, pmid, ext)

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
                        target_id = entity_global_id(
                            target_type, str(ir_val), pmid, global_types=global_types
                        )
                        target_id = remap.get(target_id, target_id)
                        _add_edge(edges, source_id, target_id, ir.name, pmid, ext)


def _get_attr(attributes: dict[str, Any] | None, key: str) -> Any:
    """Safely get an attribute value."""
    if attributes is None:
        return None
    return attributes.get(key)


def _auto_detect_ref(
    ext: Any,
    ref: Any,
    nodes_by_id: dict[str, GraphNode],
    registry: Any,
    global_types: frozenset[str] | set[str] | None = None,
) -> str | None:
    """Auto-detect a cross-entity reference when the exact reference field
    name was not populated by the LLM.

    Scans the extraction's attributes for values that match a target entity's
    name (by exact or normalized comparison).  Used as a fallback in
    ``_resolve_edges``.

    Returns the **identity key** of the matched target node, so that
    ``entity_global_id()`` produces the same ID the target node already has.
    When the target entity has a ``primary_text`` field, the identity key is
    that attribute's value; otherwise it is the extraction_text.
    """
    attrs = ext.attributes or {}
    target_type = ref.target_entity

    # Determine the identity-key attribute for the target entity type
    pt_attr: str | None = None
    if registry is not None:
        try:
            tgt_ed = registry.entity_def(target_type)
            pt_attr = tgt_ed.primary_text
        except (KeyError, AttributeError):
            pass

    def _identity_key(node: GraphNode) -> str:
        """Get the identity key for *node* consistent with build_graph."""
        if pt_attr and node.properties.get(pt_attr):
            return str(node.properties[pt_attr])
        return node.properties.get("name", "")

    # Strategy 1: scan all attribute string values for a match against any
    # existing node of the target entity type
    for attr_val in attrs.values():
        if not isinstance(attr_val, str) or not attr_val.strip():
            continue
        v = attr_val.strip()
        for node in nodes_by_id.values():
            if node.properties.get("entity_type") != target_type:
                continue
            node_name = node.properties.get("name", "")
            if (node_name.lower() == v.lower() or
                _normalize_name(node_name) == _normalize_name(v)):
                return _identity_key(node)

    # Strategy 2: try matching source extraction_text against target node names
    # by word overlap (e.g., "thymol" in Intervention → "thymol" in Alternative)
    src_text = ext.extraction_text.strip()
    src_norm = _normalize_name(src_text)
    for node in nodes_by_id.values():
        if node.properties.get("entity_type") != target_type:
            continue
        node_name = node.properties.get("name", "")
        node_norm = _normalize_name(node_name)
        if node_norm and src_norm and (node_norm == src_norm or
                                       node_norm in src_norm or
                                       src_norm in node_norm):
            return _identity_key(node)

    return None


def _add_edge(
    edges: list[GraphEdge],
    source_id: str,
    target_id: str,
    edge_type: str,
    pmid: str,
    extraction: Any,
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

    props: dict[str, Any] = {}
    # evidence_text is now on the Extraction directly (not in attributes)
    ev = getattr(extraction, "evidence_text", "") if extraction else ""
    if ev:
        props["evidence_text"] = ev

    edges.append(GraphEdge(
        source_id=source_id,
        target_id=target_id,
        type=edge_type,
        properties=props,
        source_pmids=[pmid] if pmid else [],
    ))


# ---------------------------------------------------------------------------
# Neo4j CSV export
# ---------------------------------------------------------------------------


def export_neo4j_csv(graph: Graph, output_dir: str | Path) -> None:
    """Export a Graph to Neo4j-compatible CSV files.

    Writes ``nodes.csv`` and ``edges.csv`` into *output_dir*, creating the
    directory if it does not exist.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_nodes_csv(graph, output_dir / "nodes.csv")
    _write_edges_csv(graph, output_dir / "edges.csv")


def _write_nodes_csv(graph: Graph, path: Path) -> None:
    """Write nodes.csv with Neo4j-compatible headers."""
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
