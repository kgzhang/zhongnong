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
    from src.utils import normalize_result_values
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
# Stage 4: Export graph (nodes.tsv + edges.tsv + evidence.tsv)
# ---------------------------------------------------------------------------

def _clean_entity_dir(doi_safe: str) -> None:
    """Remove entity directory for a skipped/gated article."""
    import shutil
    d = settings.entities_dir / doi_safe
    if d.exists():
        shutil.rmtree(d)
        logger.debug("Cleaned stale entity dir: %s", doi_safe)


def stage4_export(all_entities: list[dict], all_results: list[dict]):
    """Export knowledge graph as nodes.tsv + edges.tsv + evidence.tsv."""
    from src.stage4_export_graph import export_graph
    return export_graph(all_entities, all_results)
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
    run_full_pipeline(skip_stage1=False)
