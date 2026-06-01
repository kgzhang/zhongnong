"""Stage 4 (Merge): Align entities across articles and export nodes.tsv + edges.tsv.

Reads per-article entity data from data/entities/{pmid}/ directories,
aligns shared entities by name, deduplicates, and produces the final graph.

This runs AFTER all articles have been processed — not per-article.
"""
import csv
import json
import logging
from pathlib import Path
from typing import Optional

from src.config import settings
from src.entity_id import global_id, local_id, EvidenceRegistry, evidence_id
from src.utils import normalize_dose_unit

logger = logging.getLogger(__name__)

EDGE_COLUMNS = ["head_id", "head_type", "rel_type", "tail_id", "tail_type"]

# Node types and their TSV columns (no Alternative_Class — that's metadata, not a node)
NODE_SCHEMA = {
    "Alternative":       ["node_id", "standard_name", "alternative_class", "subclass", "match_source", "evidence_id"],
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


def _load_entity_dir(pmid: str) -> dict:
    """Load all entity JSON files from a single article's directory."""
    d = settings.entities_dir / pmid
    result = {}
    for fname in ["alternatives", "composite_products", "swine", "interventions",
                   "control_groups", "tissue_sites", "indicators", "methods", "results"]:
        p = d / f"{fname}.json"
        if p.exists():
            data = json.loads(p.read_text())
            result[fname] = data.get(fname, [])
        else:
            result[fname] = []
    return result


def _source_location_to_section(source_text: str) -> str:
    """Map source_location text to first-level section title."""
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


def export_graph(output_dir: Optional[Path] = None) -> dict:
    """Merge all per-article entity data and export nodes.tsv + edges.tsv + evidence.tsv.

    Reads from data/entities/{pmid}/ for all processed articles,
    aligns shared entities by name, and produces the final merged graph.

    Returns summary dict with node/edge counts.
    """
    output_dir = output_dir or settings.output_tsv_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence = EvidenceRegistry()
    nodes: dict[str, list[dict]] = {t: [] for t in NODE_SCHEMA}
    edges: list[dict] = []
    seen_nodes: set[str] = set()

    def add_node(node_type: str, node_id: str, **kwargs):
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        ev_text = kwargs.pop("evidence_text", "")
        ev_id = evidence.register(ev_text)
        row = {"node_id": node_id}
        row.update(kwargs)
        row["evidence_id"] = ev_id
        nodes[node_type].append(row)

    def add_edge(head_id, head_type, rel_type, tail_id, tail_type):
        edges.append({
            "head_id": head_id, "head_type": head_type,
            "rel_type": rel_type,
            "tail_id": tail_id, "tail_type": tail_type,
        })

    # Scan all entity directories
    entity_dirs = sorted(
        d for d in settings.entities_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )

    if not entity_dirs:
        logger.warning("No entity directories found in %s", settings.entities_dir)
        return {"nodes": 0, "edges": 0, "evidence_texts": 0}

    logger.info("Merging %d articles...", len(entity_dirs))

    int_seq = 0
    res_seq = 0
    ctl_seq = 0

    for ent_dir in entity_dirs:
        pmid = ent_dir.name
        ent = _load_entity_dir(pmid)

        # ---- Alternative (global by name) ----
        for a in ent.get("alternatives", []):
            name = a.get("standard_name", "")
            if not name:
                continue
            alt_id = global_id("Alternative", name)
            add_node("Alternative", alt_id,
                     standard_name=name,
                     alternative_class=a.get("alternative_class", "Other"),
                     subclass=a.get("subclass", ""),
                     match_source=a.get("match_source", ""),
                     evidence_text=a.get("evidence_text", ""))

        # ---- Composite_Product (global by name) ----
        for cp in ent.get("composite_products", []):
            pname = cp.get("product_name") or cp.get("standard_name", "")
            if not pname:
                continue
            cp_id = global_id("Alternative", pname)
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
            add_node("Swine", global_id("Swine", breed),
                     breed=breed,
                     sex=s.get("sex", ""), age=s.get("age", ""),
                     physiological_stage=s.get("physiological_stage", ""),
                     initial_body_weight=s.get("initial_body_weight", ""),
                     sample_size=s.get("sample_size"),
                     evidence_text=s.get("evidence_text", ""))

        # ---- Tissue_Site (global by name) ----
        for t in ent.get("tissue_sites", []):
            sname = t.get("site_name", "")
            if not sname:
                continue
            ts_id = global_id("Tissue_Site", sname)
            add_node("Tissue_Site", ts_id,
                     site_name=sname,
                     site_category=t.get("site_category", ""),
                     evidence_text=t.get("evidence_text", ""))

        # ---- Indicator (global by name) ----
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
            msite = ind.get("measured_in", "")
            if msite:
                add_edge(ind_id, "Indicator", "measured_in",
                         global_id("Tissue_Site", msite), "Tissue_Site")

        # ---- Method (global by name) ----
        for m in ent.get("methods", []):
            mname = m.get("method_name", "")
            if not mname:
                continue
            add_node("Method", global_id("Method", mname),
                     method_name=mname,
                     description=m.get("description", ""),
                     evidence_text=m.get("evidence_text", ""))

        # ---- Control_Group (local, PMID-based) ----
        for c in ent.get("control_groups", []):
            ctl_id = local_id(pmid, "Control_Group", ctl_seq)
            ctl_seq += 1
            add_node("Control_Group", ctl_id,
                     pmid=pmid,
                     group_name=c.get("group_name", ""),
                     group_type=c.get("group_type", ""),
                     description=c.get("description", ""),
                     evidence_text=c.get("evidence_text", ""))

        # ---- Intervention (local, PMID-based) ----
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
                     pmid=pmid, intervention_target=target,
                     dose_value=dose_val,
                     dose_unit_original=dose_unit,
                     dose_unit_standard=std_unit,
                     administration_route=inter.get("administration_route", ""),
                     duration=inter.get("duration", ""),
                     basal_diet=inter.get("basal_diet", ""),
                     positive_control=inter.get("positive_control"),
                     evidence_text=inter.get("evidence_text", ""))
            add_edge(int_id, "Intervention", "uses",
                     global_id("Alternative", target), "Alternative")

        # ---- Result (local, PMID-based) ----
        for r in ent.get("results", []):
            res_id = local_id(pmid, "Result", res_seq)
            res_seq += 1

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

            # corresponds_to: Result → Indicator
            if ind_abbr:
                for ind_node in nodes.get("Indicator", []):
                    if ind_node.get("abbreviation", "").lower() == ind_abbr.lower():
                        add_edge(res_id, "Result", "corresponds_to", ind_node["node_id"], "Indicator")
                        break

            # occurs_in: Result → Tissue_Site
            if site_name:
                add_edge(res_id, "Result", "occurs_in",
                         global_id("Tissue_Site", site_name), "Tissue_Site")

            # compared_to: Result → Control_Group
            if ctrl_name:
                for ctl_node in nodes.get("Control_Group", []):
                    if ctl_node.get("group_name", "").lower() == ctrl_name.lower():
                        add_edge(res_id, "Result", "compared_to", ctl_node["node_id"], "Control_Group")
                        break

            # Effect relation: Intervention → Result (same PMID)
            if rel_type:
                for int_node in nodes.get("Intervention", []):
                    if int_node.get("pmid", "") == pmid:
                        add_edge(int_node["node_id"], "Intervention", rel_type, res_id, "Result")
                        break

    # ---- Write nodes.tsv ----
    active_types = [t for t in NODE_SCHEMA if nodes.get(t)]
    all_cols = sorted(set(col for t in active_types for col in NODE_SCHEMA[t] if col != "node_id"))

    nodes_path = output_dir / "nodes.tsv"
    with open(nodes_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["node_id", "node_type"] + all_cols)
        for ntype, nlist in nodes.items():
            if not nlist:
                continue
            for row in nlist:
                writer.writerow([row.get("node_id", ""), ntype] + [row.get(c, "") for c in all_cols])

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

    total_nodes = sum(len(nl) for nl in nodes.values())
    total_edges = len(edges)
    logger.info("Merged %d articles → %d nodes, %d edges, %d evidence texts",
                len(entity_dirs), total_nodes, total_edges, len(evidence))

    return {
        "articles": len(entity_dirs),
        "nodes": total_nodes,
        "edges": total_edges,
        "evidence_texts": len(evidence),
    }
