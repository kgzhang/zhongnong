"""End-to-end pipeline using DSPy for structured extraction + TSV/Neo4j export.

Stages:
  1. XML → structured sections (existing code)
  2. DSPy entity extraction (alternatives, experiment design, indicators)
  3. DSPy result extraction
  4. Validation + TSV + Neo4j export
"""
import json
import logging
from pathlib import Path
from typing import Optional

from src.config import settings, ensure_dirs
from src.stage1_xml_parser import parse_xml_to_sections
from src.glossary import GlossaryIndex


def _eq_ic(a, b):
    """Case-insensitive string equality, safe for None values."""
    return (a or "").strip().lower() == (b or "").strip().lower()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1: XML parsing (unchanged, wraps existing code)
# ---------------------------------------------------------------------------

def stage1_parse_xml(xml_dir: Optional[Path] = None) -> list[dict]:
    """Parse all XML files and save structured sections. Returns article list."""
    xml_dir = xml_dir or settings.xml_dir
    out_dir = settings.sections_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    articles = []
    for xml_path in sorted(xml_dir.glob("*.xml")):
        logger.info("Parsing %s", xml_path.name)
        try:
            art = parse_xml_to_sections(str(xml_path))
            doi_safe = art["doi"].replace("/", "_").replace(":", "_")
            out_path = out_dir / f"{doi_safe}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(art, f, ensure_ascii=False, indent=2)
            articles.append(art)
            logger.info("  -> %s (M&M=%dch, Results=%dch)",
                        art["doi"],
                        len(art["sections"].get("materials_and_methods", {}).get("full_text", "")),
                        len(art["sections"].get("results", {}).get("full_text", "")))
        except Exception as e:
            logger.error("Failed to parse %s: %s", xml_path.name, e)

    return articles


# ---------------------------------------------------------------------------
# Stage 2: DSPy entity extraction
# ---------------------------------------------------------------------------

def stage2_extract_entities(article: dict) -> Optional[dict]:
    """Extract entities from one article. Returns None if article should be skipped.

    Gate: If Module C finds no known Alternative (only "Other"), skip the article.
    """
    from src.dspy_extract import extract_alternatives, extract_experiment_design, extract_indicators

    doi = article["doi"]
    mm_text = article["sections"].get("materials_and_methods", {}).get("full_text", "")

    if not mm_text:
        logger.warning("No M&M text for %s — skipping", doi)
        return None

    logger.info("[Stage2:%s] Extracting entities...", doi[:30])

    # Module C: Alternatives — THE GATE
    alt_result = extract_alternatives(mm_text)
    n_alt = len(alt_result.get("alternatives", []))
    n_comp = len(alt_result.get("composite_products", []))
    has_known = alt_result.get("has_known_alternative", False)

    logger.info("  [C] %d alternatives, %d composites (has_known=%s)", n_alt, n_comp, has_known)

    # GATE CHECK: skip if no known Alternative
    if not has_known and n_comp == 0:
        logger.info("  -> GATE: No known alternative found. Skipping article.")
        return None

    # Module A: Experiment design (no Swine_Model)
    exp_result = extract_experiment_design(mm_text)
    sw = exp_result.get("swine", [])
    if isinstance(sw, dict):
        sw = [sw] if sw else []
        exp_result["swine"] = sw
    logger.info("  [A] %d interventions, %d controls, %d swine groups",
                len(exp_result.get("interventions", [])),
                len(exp_result.get("control_groups", [])),
                len(sw))

    # Module B: Indicators
    ind_result = extract_indicators(mm_text)
    logger.info("  [B] %d indicators, %d tissue sites, %d methods",
                len(ind_result.get("indicators", [])),
                len(ind_result.get("tissue_sites", [])),
                len(ind_result.get("methods", [])))

    return {
        "doi": doi,
        **alt_result,
        **exp_result,
        **ind_result,
    }


# ---------------------------------------------------------------------------
# Stage 3: DSPy result extraction
# ---------------------------------------------------------------------------

def stage3_extract_results(article: dict, entities: dict) -> list[dict]:
    """Extract all statistical results from one article using DSPy."""
    from src.dspy_extract import extract_results

    results_text = article["sections"].get("results", {}).get("full_text", "")
    discussion_text = article["sections"].get("discussion", {}).get("full_text", "")

    if not results_text:
        logger.warning("No Results text for %s", article["doi"])
        return []

    results = extract_results(
        results_text=results_text,
        discussion_text=discussion_text,
        indicators=entities.get("indicators", []),
        control_groups=entities.get("control_groups", []),
        tissue_sites=entities.get("tissue_sites", []),
    )

    # Normalize each result
    from src.stage3_result_extract import _normalize_result_values
    for r in results:
        _normalize_result_values(r)
        r.setdefault("indicator_abbreviation", "")
        r.setdefault("tissue_site", "")
        r.setdefault("direction", "no_significant_change")
        r.setdefault("relation_type", "affects")
        r.setdefault("significance_level", "not_significant")
        r.setdefault("evidence_text", "")
        r.setdefault("source_location", "")
        r.setdefault("compared_to_group", "")
        r.setdefault("p_value", None)

    logger.info("[Stage3:%s] %d results extracted", article["doi"][:30], len(results))
    return results


# ---------------------------------------------------------------------------
# Stage 4: Validation + TSV + Neo4j export
# ---------------------------------------------------------------------------

def _clean_entity_dir(doi_safe: str) -> None:
    """Remove entity directory for a skipped/gated article."""
    import shutil
    d = settings.entities_dir / doi_safe
    if d.exists():
        shutil.rmtree(d)
        logger.debug("Cleaned stale entity dir: %s", doi_safe)


def stage4_export(all_entities: list[dict], all_results: list[dict]):
    """Validate entities/results and export TSV + Neo4j Cypher files."""
    from dataclasses import asdict
    from src.models import (
        Alternative, AlternativeClass, CompositeProduct, Literature, Experiment,
        Swine, Intervention, ControlGroup, TissueSite, Indicator, Result, Method, Relationship,
    )
    from src.stage2_entity_extract import build_entity_id
    from src.stage4_export_tsv import write_entities_tsv
    from src.stage4_export_neo4j import generate_cypher, generate_indexes

    def _write_neo4j(entity_groups, relationships, output_dir=None):
        import os
        output_dir = output_dir or settings.output_neo4j_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "import.cypher"
        lines = ["// ===== INDEXES =====", generate_indexes(), ""]
        for etype, elist in entity_groups.items():
            if elist:
                lines.append(f"// ===== {etype} NODES ({len(elist)} rows) =====")
                lines.append(generate_cypher(elist, etype))
                lines.append("")
        # Relationships
        if relationships:
            lines.append(f"// ===== RELATIONSHIPS ({len(relationships)} rows) =====")
            for r in relationships:
                from dataclasses import asdict
                d = asdict(r) if hasattr(r, '__dataclass_fields__') else r
                hl = d.get("head_entity_type", "")
                tl = d.get("tail_entity_type", "")
                hid = d.get("head_entity_id", "")
                tid = d.get("tail_entity_id", "")
                rt = d.get("rel_type", "")
                lines.append(f"MATCH (a:{hl} {{entity_id: \"{hid}\"}})")
                lines.append(f"MATCH (b:{tl} {{entity_id: \"{tid}\"}})")
                lines.append(f"MERGE (a)-[:{rt}]->(b)")
                lines.append("")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return out_path

    ensure_dirs()

    # Collect all entities
    alternatives: list[Alternative] = []
    composites: list[CompositeProduct] = []
    alt_classes: list[AlternativeClass] = []
    swines: list[Swine] = []
    interventions: list[Intervention] = []
    controls: list[ControlGroup] = []
    tissues: list[TissueSite] = []
    indicators: list[Indicator] = []
    methods: list[Method] = []
    results_list: list[Result] = []
    relationships: list[Relationship] = []
    literatures: list[Literature] = []

    # Known classification list from glossary
    gi = GlossaryIndex()
    gi.load(str(settings.alternative_tsv))
    seen_classes = set()
    for _, info in gi._exact.items():
        cls = info["class"]
        if cls not in seen_classes:
            seen_classes.add(cls)
            alt_classes.append(AlternativeClass(class_name=cls, level=1))

    alt_seq = 0
    cp_seq = 0
    sw_seq = 0
    int_seq = 0
    ctl_seq = 0
    tis_seq = 0
    ind_seq = 0
    met_seq = 0
    res_seq = 0

    for ent in all_entities:
        doi = ent.get("doi", "")
        doi_safe = doi.replace("/", "_").replace(":", "_")
        exp_id = f"{doi_safe}_Exp_01"

        # Alternatives
        for a in ent.get("alternatives", []):
            eid = build_entity_id(doi, "ALT", alt_seq)
            alt_seq += 1
            alternatives.append(Alternative(
                entity_id=eid,
                standard_name=a.get("standard_name", ""),
                abbreviation=a.get("abbreviation"),
                alternative_class=a.get("alternative_class", "Other"),
                subclass=a.get("subclass"),
                match_source=a.get("match_source", "Other_未匹配"),
                evidence_text=a.get("evidence_text", ""),
                source_location=a.get("source_location", ""),
            ))

        # Composites
        for cp in ent.get("composite_products", []):
            eid = build_entity_id(doi, "CPD", cp_seq)
            cp_seq += 1
            composites.append(CompositeProduct(
                entity_id=eid,
                product_name=cp.get("product_name", cp.get("standard_name", "")),
                manufacturer=cp.get("manufacturer"),
                is_commercial=cp.get("is_commercial", False),
                components=cp.get("components", []),
                evidence_text=cp.get("evidence_text", ""),
                source_location=cp.get("source_location", ""),
            ))
            # has_component relationships
            for comp in cp.get("components", []):
                # Find the alternative entity_id
                for a in alternatives:
                    if _eq_ic(a.standard_name, comp.get("standard_name", "")):
                        relationships.append(Relationship(
                            rel_type="has_component",
                            head_entity_type="Composite_Product",
                            head_entity_id=eid,
                            tail_entity_type="Alternative",
                            tail_entity_id=a.entity_id,
                        ))
                        break

        # Swine
        for s in ent.get("swine", []):
            eid = build_entity_id(doi, "SWN", sw_seq)
            sw_seq += 1
            swines.append(Swine(
                entity_id=eid, experiment_id=exp_id,
                breed=s.get("breed", ""),
                sex=s.get("sex", ""),
                age=str(s.get("age", "")),
                physiological_stage=s.get("physiological_stage", ""),
                initial_body_weight=s.get("initial_body_weight", ""),
                sample_size=s.get("sample_size") if isinstance(s.get("sample_size"), int) else None,
                evidence_text=s.get("evidence_text", ""),
                source_location=s.get("source_location", ""),
            ))

        # Interventions (skip empty ones from LLM)
        for inter in ent.get("interventions", []):
            if not inter.get("intervention_target"):
                continue
            eid = build_entity_id(doi, "INT", int_seq)
            int_seq += 1
            # Normalize dose
            from src.stage2_entity_extract import normalize_dose_unit
            dose_val = inter.get("dose_value")
            dose_unit = inter.get("dose_unit_original", "")
            std_unit = inter.get("dose_unit_standard", "")
            if dose_val is not None and dose_unit and not std_unit:
                dose_val, std_unit = normalize_dose_unit(float(dose_val), dose_unit)

            interventions.append(Intervention(
                entity_id=eid, experiment_id=exp_id,
                intervention_target=inter.get("intervention_target", ""),
                dose_value=dose_val,
                dose_unit_original=dose_unit,
                dose_unit_standard=std_unit,
                administration_route=inter.get("administration_route", ""),
                duration=inter.get("duration", ""),
                basal_diet=inter.get("basal_diet", ""),
                positive_control=inter.get("positive_control"),
                evidence_text=inter.get("evidence_text", ""),
                source_location=inter.get("source_location", ""),
            ))
            # uses relationship: Intervention -> Alternative or Composite_Product
            target = inter.get("intervention_target") or ""
            for a in alternatives:
                if target and _eq_ic(a.standard_name, target):
                    relationships.append(Relationship(
                        rel_type="uses",
                        head_entity_type="Intervention", head_entity_id=eid,
                        tail_entity_type="Alternative", tail_entity_id=a.entity_id,
                    ))
                    break
            else:
                for cp in composites:
                    if target and _eq_ic(cp.product_name, target):
                        relationships.append(Relationship(
                            rel_type="uses",
                            head_entity_type="Intervention", head_entity_id=eid,
                            tail_entity_type="Composite_Product", tail_entity_id=cp.entity_id,
                        ))
                        break

        # Controls
        for c in ent.get("control_groups", []):
            eid = build_entity_id(doi, "CTL", ctl_seq)
            ctl_seq += 1
            controls.append(ControlGroup(
                entity_id=eid, experiment_id=exp_id,
                group_name=c.get("group_name", ""),
                group_type=c.get("group_type", ""),
                description=c.get("description", ""),
                evidence_text=c.get("evidence_text", ""),
                source_location=c.get("source_location", ""),
            ))

        # Tissues
        for t in ent.get("tissue_sites", []):
            eid = build_entity_id(doi, "TIS", tis_seq)
            tis_seq += 1
            tissues.append(TissueSite(
                entity_id=eid, doi=doi,
                site_name=t.get("site_name", ""),
                site_category=t.get("site_category", ""),
                evidence_text=t.get("evidence_text", ""),
                source_location=t.get("source_location", ""),
            ))

        # Indicators
        for ind in ent.get("indicators", []):
            eid = build_entity_id(doi, "IND", ind_seq)
            ind_seq += 1
            indicators.append(Indicator(
                entity_id=eid, doi=doi,
                standard_name=ind.get("standard_name", ""),
                abbreviation=ind.get("abbreviation", ""),
                unit=ind.get("unit", ""),
                indicator_category=ind.get("indicator_category", ""),
                measurement_method=ind.get("measurement_method", ""),
                measured_in=ind.get("measured_in", ""),
                evidence_text=ind.get("evidence_text", ""),
                source_location=ind.get("source_location", ""),
            ))

        # Methods
        for m in ent.get("methods", []):
            eid = build_entity_id(doi, "MET", met_seq)
            met_seq += 1
            methods.append(Method(
                entity_id=eid, doi=doi,
                method_name=m.get("method_name", ""),
                description=m.get("description", ""),
                evidence_text=m.get("evidence_text", ""),
                source_location=m.get("source_location", ""),
            ))

    # ---- Structural relationships (entity-to-entity) ----
    # belongs_to: Alternative → Alternative_Class
    for a in alternatives:
        cls_name = a.alternative_class
        if cls_name and cls_name != "Other":
            for ac in alt_classes:
                if _eq_ic(ac.class_name, cls_name):
                    relationships.append(Relationship(
                        rel_type="belongs_to",
                        head_entity_type="Alternative", head_entity_id=a.entity_id,
                        tail_entity_type="Alternative_Class", tail_entity_id=ac.class_name,
                    ))
                    break

    # measured_in: Indicator → Tissue_Site
    for ind in indicators:
        if ind.measured_in:
            for t in tissues:
                if _eq_ic(t.site_name, ind.measured_in):
                    relationships.append(Relationship(
                        rel_type="measured_in",
                        head_entity_type="Indicator", head_entity_id=ind.entity_id,
                        tail_entity_type="Tissue_Site", tail_entity_id=t.entity_id,
                    ))
                    break

    # uses_method: Indicator → Method
    for ind in indicators:
        if ind.measurement_method:
            for m in methods:
                if _eq_ic(m.method_name, ind.measurement_method):
                    relationships.append(Relationship(
                        rel_type="uses_method",
                        head_entity_type="Indicator", head_entity_id=ind.entity_id,
                        tail_entity_type="Method", tail_entity_id=m.entity_id,
                    ))
                    break

    # Results (with alignment)
    for r in all_results:
        doi = r.get("doi", "")
        doi_safe = doi.replace("/", "_").replace(":", "_")
        exp_id = f"{doi_safe}_Exp_01"
        eid = build_entity_id(doi, "RES", res_seq)
        res_seq += 1

        # Find matching indicator
        ind_abbr = r.get("indicator_abbreviation", "").strip().lower()
        matched_ind = ""
        for ind in indicators:
            if _eq_ic(ind.abbreviation, ind_abbr):
                matched_ind = ind.entity_id
                break
        if not matched_ind:
            for ind in indicators:
                if _eq_ic(ind.standard_name, ind_abbr):
                    matched_ind = ind.entity_id
                    break

        # Find matching tissue
        site_name = r.get("tissue_site", "").strip().lower()
        matched_tis = ""
        for t in tissues:
            if _eq_ic(t.site_name, site_name):
                matched_tis = t.entity_id
                break

        # Find matching control
        ctrl_name = r.get("compared_to_group", "").strip()
        matched_ctl = ""
        for c in controls:
            if _eq_ic(c.group_name, ctrl_name):
                matched_ctl = c.entity_id
                break

        # Find intervention
        int_id = ""
        for inter in interventions:
            if inter.experiment_id == exp_id:
                int_id = inter.entity_id
                break

        result = Result(
            entity_id=eid, doi=doi, experiment_id=exp_id,
            direction=r.get("direction", ""),
            p_value=r.get("p_value"),
            p_value_original_text=r.get("p_value_original_text", ""),
            significance_level=r.get("significance_level", ""),
            effect_size=r.get("effect_size"),
            time_point=r.get("time_point"),
            subgroup=r.get("subgroup"),
            evidence_text=r.get("evidence_text", ""),
            source_location=r.get("source_location", ""),
            matched_indicator=matched_ind,
            matched_tissue=matched_tis,
            compared_to_group=r.get("compared_to_group", ""),
            relation_type=r.get("relation_type", ""),
        )
        results_list.append(result)

        # Build relationships
        if int_id and result.relation_type:
            relationships.append(Relationship(
                rel_type=result.relation_type,
                head_entity_type="Intervention", head_entity_id=int_id,
                tail_entity_type="Result", tail_entity_id=eid,
                evidence_text=result.evidence_text, source_location=result.source_location,
            ))
        if matched_ind:
            relationships.append(Relationship(
                rel_type="corresponds_to",
                head_entity_type="Result", head_entity_id=eid,
                tail_entity_type="Indicator", tail_entity_id=matched_ind,
            ))
        if matched_tis:
            relationships.append(Relationship(
                rel_type="occurs_in",
                head_entity_type="Result", head_entity_id=eid,
                tail_entity_type="Tissue_Site", tail_entity_id=matched_tis,
            ))
        if matched_ctl:
            relationships.append(Relationship(
                rel_type="compared_to",
                head_entity_type="Result", head_entity_id=eid,
                tail_entity_type="Control_Group", tail_entity_id=matched_ctl,
            ))

    # ---- Export ----
    entity_groups = {
        "Alternative": alternatives,
        "Alternative_Class": alt_classes,
        "Composite_Product": composites,
        "Experiment": [],
        "Swine": swines,
        "Intervention": interventions,
        "Control_Group": controls,
        "Tissue_Site": tissues,
        "Indicator": indicators,
        "Result": results_list,
        "Method": methods,
    }

    # TSV export
    total_entities = 0
    for etype, elist in entity_groups.items():
        if elist:
            write_entities_tsv(elist, etype)
            total_entities += len(elist)
    logger.info("Stage 4: Wrote %d entities to TSV files", total_entities)

    # Neo4j export
    neo4j_path = _write_neo4j(entity_groups, relationships)
    logger.info("Stage 4: Generated %s with %d nodes, %d relationships",
                neo4j_path, total_entities, len(relationships))

    # Summary
    summary = {
        "alternatives": len(alternatives),
        "composite_products": len(composites),
        "swine_groups": len(swines),
        "interventions": len(interventions),
        "control_groups": len(controls),
        "tissue_sites": len(tissues),
        "indicators": len(indicators),
        "results": len(results_list),
        "methods": len(methods),
        "relationships": len(relationships),
    }
    return summary


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_full_pipeline(xml_dir: Optional[str] = None, skip_stage1: bool = False) -> dict:
    """Run the full end-to-end pipeline.

    Args:
        xml_dir: Path to XML files. Uses settings.xml_dir if None.
        skip_stage1: If True, re-use existing structured_sections JSON files.

    Returns:
        Summary dict with entity/relationship counts.
    """
    ensure_dirs()

    # Pre-clean: remove any entity dirs without valid alternatives (from previous runs)
    import shutil
    for d in settings.entities_dir.iterdir():
        if not d.is_dir() or d.name.startswith("_"):
            continue
        alt_file = d / "alternatives.json"
        if not alt_file.exists():
            shutil.rmtree(d)
            continue
        try:
            data = json.loads(alt_file.read_text())
            if len(data.get("alternatives", [])) == 0:
                shutil.rmtree(d)
        except Exception:
            shutil.rmtree(d)

    # Stage 1: XML parsing
    articles = []
    if not skip_stage1:
        articles = stage1_parse_xml(Path(xml_dir) if xml_dir else None)
    else:
        for f in sorted(settings.sections_dir.glob("*.json")):
            if f.name.startswith("_"):
                continue
            with open(f, "r", encoding="utf-8") as fh:
                articles.append(json.load(fh))

    if not articles:
        logger.warning("No articles to process")
        return {}

    logger.info("Processing %d articles", len(articles))

    all_entities = []
    all_results = []

    for article in articles:
        doi = article["doi"]
        doi_safe = doi.replace("/", "_").replace(":", "_")
        logger.info("=" * 60)
        logger.info("Article: %s", doi)

        # Stage 2: Entity extraction (returns None if gated)
        entities = stage2_extract_entities(article)
        if entities is None:
            logger.info("  SKIPPED by gate check")
            _clean_entity_dir(doi_safe)
            continue
        entities["doi"] = doi

        # Save entity outputs
        ent_dir = settings.entities_dir / doi_safe
        ent_dir.mkdir(parents=True, exist_ok=True)

        for key in ["alternatives", "composite_products", "swine",
                     "interventions", "control_groups", "tissue_sites", "indicators", "methods"]:
            data = {key: entities.get(key, []), "doi": doi}
            with open(ent_dir / f"{key}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        # Stage 3: Result extraction
        results = stage3_extract_results(article, entities)
        with open(ent_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump({"doi": doi, "results": results}, f, ensure_ascii=False, indent=2)

        # Annotate results with DOI
        for r in results:
            r["doi"] = doi

        all_entities.append(entities)
        all_results.extend(results)

        logger.info("  Entities: %d alts, %d intv, %d inds, %d sites",
                    len(entities.get("alternatives", [])),
                    len(entities.get("interventions", [])),
                    len(entities.get("indicators", [])),
                    len(entities.get("tissue_sites", [])))
        logger.info("  Results: %d", len(results))

    # Stage 4: Export
    logger.info("=" * 60)
    logger.info("Stage 4: Validation + Export")
    summary = stage4_export(all_entities, all_results)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    for k, v in summary.items():
        logger.info("  %s: %d", k, v)

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_full_pipeline(skip_stage1=True)
