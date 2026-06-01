"""Stage 4: Export knowledge graph as nodes.tsv + edges.tsv + evidence.tsv.

Uses global IDs for shared entities (Alternative, Tissue_Site, Indicator, etc.)
and local IDs for paper-specific entities (Intervention, Result, Control_Group).
Deduplicates evidence_text via hash-based evidence registry.
"""
import csv
import logging
from pathlib import Path
from dataclasses import asdict
from typing import Optional

from src.config import settings
from src.entity_id import global_id, local_id, EvidenceRegistry, evidence_id
from src.glossary import GlossaryIndex
from src.utils import normalize_dose_unit

logger = logging.getLogger(__name__)

# Node types and their TSV columns
NODE_SCHEMA = {
    "Alternative":       ["node_id", "standard_name", "alternative_class", "subclass", "match_source", "evidence_id"],
    "Alternative_Class": ["node_id", "class_name", "description"],
    "Composite_Product": ["node_id", "product_name", "manufacturer", "is_commercial", "evidence_id"],
    "Literature":        ["node_id", "doi", "pmid", "title", "journal", "publication_year", "abstract_conclusion"],
    "Experiment":        ["node_id", "pmid", "description", "evidence_id"],
    "Swine":             ["node_id", "breed", "sex", "age", "physiological_stage", "initial_body_weight", "sample_size", "evidence_id"],
    "Intervention":      ["node_id", "pmid", "intervention_target", "dose_value", "dose_unit_original", "dose_unit_standard", "administration_route", "duration", "basal_diet", "positive_control", "evidence_id"],
    "Control_Group":     ["node_id", "pmid", "group_name", "group_type", "description", "evidence_id"],
    "Tissue_Site":       ["node_id", "site_name", "site_category", "evidence_id"],
    "Indicator":         ["node_id", "standard_name", "abbreviation", "unit", "indicator_category", "measurement_method", "measured_in", "evidence_id"],
    "Result":            ["node_id", "pmid", "direction", "p_value", "p_value_original_text", "significance_level", "effect_size", "time_point", "subgroup", "compared_to_group", "relation_type", "evidence_id"],
    "Method":            ["node_id", "method_name", "description", "evidence_id"],
}

EDGE_COLUMNS = ["head_id", "head_type", "rel_type", "tail_id", "tail_type"]


def _source_location_to_section(article: dict, source_text: str) -> str:
    """Map source_location text to the first-level section title.

    If source_location contains a subsection reference like "Methods 2.3",
    map it up to "Materials and Methods". Similarly for Results/Discussion.
    """
    if not source_text:
        return ""
    s = source_text.lower()
    if any(w in s for w in ("methods", "materials", "m&m", "2.", "experimental")):
        return "Materials and Methods"
    if any(w in s for w in ("results", "3.", "findings")):
        return "Results"
    if any(w in s for w in ("discussion", "4.", "conclusions")):
        return "Discussion"
    if any(w in s for w in ("introduction", "1.", "background")):
        return "Introduction"
    return source_text


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_graph(
    all_entities: list[dict],
    all_results: list[dict],
    articles: Optional[list[dict]] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """Export knowledge graph as nodes.tsv + edges.tsv + evidence.tsv.

    Returns summary dict with node/edge counts.
    """
    output_dir = output_dir or settings.output_tsv_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence = EvidenceRegistry()

    # Known classifications for global nodes
    gi = GlossaryIndex()
    gi.load(str(settings.alternative_tsv))

    # --- Collectors ---
    nodes: dict[str, list[dict]] = {t: [] for t in NODE_SCHEMA}
    edges: list[dict] = []
    seen_nodes: set[str] = set()  # dedup by node_id

    def add_node(node_type: str, node_id: str, **kwargs):
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        # Register evidence
        ev_text = kwargs.pop("evidence_text", "")
        ev_id = evidence.register(ev_text)
        row = {"node_id": node_id}
        row.update(kwargs)
        row["evidence_id"] = ev_id
        nodes[node_type].append(row)

    def add_edge(head_id: str, head_type: str, rel_type: str, tail_id: str, tail_type: str):
        edges.append({
            "head_id": head_id, "head_type": head_type,
            "rel_type": rel_type,
            "tail_id": tail_id, "tail_type": tail_type,
        })

    # --- Global nodes (name-based) ---
    # Alternative_Class (static)
    for _, info in gi._exact.items():
        cls = info["class"]
        cls_id = global_id("Alternative_Class", cls)
        add_node("Alternative_Class", cls_id, class_name=cls, description="")

    # --- Process each article ---
    int_seq = 0
    res_seq = 0
    ctl_seq_global = 0
    sw_seq = 0

    for idx, ent in enumerate(all_entities):
        doi = ent.get("doi", "")
        pmid = ent.get("pmid", "") or doi.replace("/", "_").replace(":", "_")

        # ---- Alternative (global) ----
        for a in ent.get("alternatives", []):
            name = a.get("standard_name", "")
            if not name:
                continue
            alt_id = global_id("Alternative", name)
            cls = a.get("alternative_class", "Other")
            add_node("Alternative", alt_id,
                     standard_name=name, alternative_class=cls,
                     subclass=a.get("subclass", ""),
                     match_source=a.get("match_source", ""),
                     evidence_text=a.get("evidence_text", ""))
            # belongs_to: Alternative → Alternative_Class
            if cls != "Other":
                add_edge(alt_id, "Alternative", "belongs_to",
                         global_id("Alternative_Class", cls), "Alternative_Class")

        # ---- Composite_Product (global) ----
        for cp in ent.get("composite_products", []):
            pname = cp.get("product_name") or cp.get("standard_name", "")
            if not pname:
                continue
            cp_id = global_id("Alternative", pname)  # composite stored as Alternative
            add_node("Composite_Product", cp_id,
                     product_name=pname,
                     manufacturer=cp.get("manufacturer"),
                     is_commercial=cp.get("is_commercial", False),
                     evidence_text=cp.get("evidence_text", ""))
            for comp in cp.get("components", []):
                cname = comp.get("standard_name", "")
                if cname:
                    add_edge(cp_id, "Composite_Product", "has_component",
                             global_id("Alternative", cname), "Alternative")

        # ---- Swine (global by breed) ----
        for s in ent.get("swine", []):
            breed = s.get("breed", "")
            if not breed:
                continue
            sw_id = global_id("Swine", breed)
            add_node("Swine", sw_id,
                     breed=breed,
                     sex=s.get("sex", ""),
                     age=s.get("age", ""),
                     physiological_stage=s.get("physiological_stage", ""),
                     initial_body_weight=s.get("initial_body_weight", ""),
                     sample_size=s.get("sample_size"),
                     evidence_text=s.get("evidence_text", ""))

        # ---- Tissue_Site (global) ----
        for t in ent.get("tissue_sites", []):
            sname = t.get("site_name", "")
            if not sname:
                continue
            ts_id = global_id("Tissue_Site", sname)
            add_node("Tissue_Site", ts_id,
                     site_name=sname,
                     site_category=t.get("site_category", ""),
                     evidence_text=t.get("evidence_text", ""))

        # ---- Indicator (global) ----
        for ind in ent.get("indicators", []):
            iname = ind.get("standard_name", "")
            if not iname:
                continue
            ind_id = global_id("Indicator", iname)
            add_node("Indicator", ind_id,
                     standard_name=iname,
                     abbreviation=ind.get("abbreviation", ""),
                     unit=ind.get("unit", ""),
                     indicator_category=ind.get("indicator_category", ""),
                     measurement_method=ind.get("measurement_method", ""),
                     measured_in=ind.get("measured_in", ""),
                     evidence_text=ind.get("evidence_text", ""))
            # measured_in: Indicator → Tissue_Site
            msite = ind.get("measured_in", "")
            if msite:
                add_edge(ind_id, "Indicator", "measured_in",
                         global_id("Tissue_Site", msite), "Tissue_Site")

        # ---- Method (global) ----
        for m in ent.get("methods", []):
            mname = m.get("method_name", "")
            if not mname:
                continue
            met_id = global_id("Method", mname)
            add_node("Method", met_id,
                     method_name=mname,
                     description=m.get("description", ""),
                     evidence_text=m.get("evidence_text", ""))

        # ---- Control_Group (local) ----
        for c in ent.get("control_groups", []):
            ctl_id = local_id(pmid, "Control_Group", ctl_seq_global)
            ctl_seq_global += 1
            add_node("Control_Group", ctl_id,
                     pmid=pmid,
                     group_name=c.get("group_name", ""),
                     group_type=c.get("group_type", ""),
                     description=c.get("description", ""),
                     evidence_text=c.get("evidence_text", ""))

        # ---- Intervention (local) ----
        for inter in ent.get("interventions", []):
            target = inter.get("intervention_target")
            if not target:
                continue
            int_id = local_id(pmid, "Intervention", int_seq)
            int_seq += 1

            dose_val = inter.get("dose_value")
            dose_unit = inter.get("dose_unit_original", "")
            std_unit = ""
            if dose_val is not None and dose_unit:
                dose_val, std_unit = normalize_dose_unit(float(dose_val), dose_unit)

            add_node("Intervention", int_id,
                     pmid=pmid,
                     intervention_target=target,
                     dose_value=dose_val,
                     dose_unit_original=dose_unit,
                     dose_unit_standard=std_unit,
                     administration_route=inter.get("administration_route", ""),
                     duration=inter.get("duration", ""),
                     basal_diet=inter.get("basal_diet", ""),
                     positive_control=inter.get("positive_control"),
                     evidence_text=inter.get("evidence_text", ""))
            # uses: Intervention → Alternative
            add_edge(int_id, "Intervention", "uses",
                     global_id("Alternative", target), "Alternative")

    # ---- Results (local) ----
    for r in all_results:
        doi = r.get("doi", "")
        pmid = r.get("pmid", "") or doi.replace("/", "_").replace(":", "_")
        res_id = local_id(pmid, "Result", res_seq)
        res_seq += 1

        # Map source_location to section heading
        src = r.get("source_location", "")
        section = _source_location_to_section({}, src) if src else ""

        add_node("Result", res_id,
                 pmid=pmid,
                 direction=r.get("direction", ""),
                 p_value=r.get("p_value"),
                 p_value_original_text=r.get("p_value_original_text", ""),
                 significance_level=r.get("significance_level", ""),
                 effect_size=r.get("effect_size"),
                 time_point=r.get("time_point"),
                 subgroup=r.get("subgroup"),
                 compared_to_group=r.get("compared_to_group", ""),
                 relation_type=r.get("relation_type", ""),
                 evidence_text=r.get("evidence_text", ""))

        ind_abbr = r.get("indicator_abbreviation", "")
        site_name = r.get("tissue_site", "")
        ctrl_name = r.get("compared_to_group", "")
        rel_type = r.get("relation_type", "")

        # corresponds_to: Result → Indicator (find by abbreviation)
        if ind_abbr:
            # Look up the indicator by abbreviation (global ID uses standard_name, not abbreviation)
            # We need to search through nodes to find the matching Indicator
            ind_id = None
            for ind_node in nodes.get("Indicator", []):
                if ind_node.get("abbreviation", "").lower() == ind_abbr.lower():
                    ind_id = ind_node["node_id"]
                    break
            if ind_id:
                add_edge(res_id, "Result", "corresponds_to", ind_id, "Indicator")

        # occurs_in: Result → Tissue_Site
        if site_name:
            add_edge(res_id, "Result", "occurs_in",
                     global_id("Tissue_Site", site_name), "Tissue_Site")

        # compared_to: Result → Control_Group (find by group_name)
        if ctrl_name:
            for ctl_node in nodes.get("Control_Group", []):
                if ctl_node.get("group_name", "").lower() == ctrl_name.lower():
                    add_edge(res_id, "Result", "compared_to",
                             ctl_node["node_id"], "Control_Group")
                    break

        # Effect relation: find Intervention by pmid, connect Result
        for int_node in nodes.get("Intervention", []):
            if int_node.get("pmid", "") == pmid and rel_type:
                add_edge(int_node["node_id"], "Intervention", rel_type,
                         res_id, "Result")
                break

    # ---- Write nodes.tsv ----
    nodes_path = output_dir / "nodes.tsv"
    with open(nodes_path, "w", newline="", encoding="utf-8-sig") as f:
        # Collect all unique columns across node types
        all_node_types = [t for t in NODE_SCHEMA if nodes.get(t)]
        # Write header with node_type column
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["node_id", "node_type"] + sorted(set(
            col for t in all_node_types for col in NODE_SCHEMA[t] if col not in ("node_id",)
        )))
        for ntype, nlist in nodes.items():
            if not nlist:
                continue
            cols = NODE_SCHEMA.get(ntype, [])
            for row in nlist:
                writer.writerow([row.get("node_id", ""), ntype] + [
                    row.get(c, "") for c in sorted(set(
                        col for t in all_node_types for col in NODE_SCHEMA[t] if col not in ("node_id",)
                    ))
                ])

    # ---- Write edges.tsv ----
    edges_path = output_dir / "edges.tsv"
    with open(edges_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=EDGE_COLUMNS, delimiter="\t")
        writer.writeheader()
        for e in edges:
            writer.writerow(e)

    # ---- Write evidence.tsv ----
    ev_path = output_dir / "evidence.tsv"
    with open(ev_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["evidence_id", "evidence_text"], delimiter="\t")
        writer.writeheader()
        for ev in evidence.to_rows():
            writer.writerow(ev)

    # Count
    total_nodes = sum(len(nl) for nl in nodes.values())
    total_edges = len(edges)
    logger.info("Exported: %d nodes, %d edges, %d evidence texts (%s, %s, %s)",
                total_nodes, total_edges, len(evidence),
                nodes_path.name, edges_path.name, ev_path.name)

    return {
        "nodes": total_nodes,
        "edges": total_edges,
        "evidence_texts": len(evidence),
        "nodes_path": str(nodes_path),
        "edges_path": str(edges_path),
        "evidence_path": str(ev_path),
    }
