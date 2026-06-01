"""Stage 3: Result extraction — Pass 1 (LLM) + Pass 2 (alignment) + orchestrator."""
import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from src.config import settings
from src.llm_client import LLMClient
from src.models import Result, Relationship, Indicator, TissueSite, ControlGroup
from src.stage2_entity_extract import load_section_json, checkpoint_exists, save_entity_output, build_entity_id
from src.stage3_prompts import build_result_prompt, SYSTEM_PROMPT_RESULT

logger = logging.getLogger(__name__)


def align_result_to_indicator(result: dict, indicators: list[Indicator]) -> Optional[str]:
    """Match result's indicator_abbreviation to an Indicator entity. Returns entity_id or None."""
    abbr = result.get("indicator_abbreviation", "").strip().lower()
    if not abbr:
        return None

    # 1. Exact match on abbreviation
    for ind in indicators:
        if ind.abbreviation.lower().strip() == abbr:
            return ind.entity_id

    # 2. Exact match on standard_name
    for ind in indicators:
        if ind.standard_name.lower().strip() == abbr:
            return ind.entity_id

    # 3. Containment match (fuzzy)
    for ind in indicators:
        ind_abbr = ind.abbreviation.lower().strip()
        if ind_abbr and (ind_abbr in abbr or abbr in ind_abbr):
            return ind.entity_id

    return None


def align_result_to_tissue(result: dict, tissues: list[TissueSite]) -> Optional[str]:
    """Match result's tissue_site to a TissueSite entity."""
    site = result.get("tissue_site", "").strip().lower()
    if not site:
        return None
    for ts in tissues:
        if ts.site_name.lower().strip() == site:
            return ts.entity_id
    for ts in tissues:
        if ts.site_name.lower().strip() in site or site in ts.site_name.lower().strip():
            return ts.entity_id
    return None


# Relation type -> expected direction mapping
DIRECTION_CONSISTENCY_MAP = {
    "increases": "increased",
    "decreases": "decreased",
    "upregulates": "increased",
    "downregulates": "decreased",
    "enriches": "increased",
    "depletes": "decreased",
    "affects": None,  # always consistent
}


def check_direction_consistency(rel_type: str, direction: str) -> bool:
    """Check that relation_type and direction are consistent."""
    expected = DIRECTION_CONSISTENCY_MAP.get(rel_type)
    if expected is None:
        return True  # "affects" matches anything
    return expected == direction


def check_compared_to_baseline(
    result: dict,
    control_groups: list[ControlGroup],
    is_challenge_model: bool,
) -> tuple[bool, str]:
    """Validate compared_to_group for challenge models (BACKGROUND 4.1)."""
    if not is_challenge_model:
        return True, ""

    compared_to = result.get("compared_to_group", "")
    for cg in control_groups:
        if cg.group_name == compared_to and cg.group_type in ("negative_control", "sham"):
            return True, ""

    correct = [cg.group_name for cg in control_groups if cg.group_type in ("negative_control", "sham")]
    return False, f"Challenge model should compare vs challenged control ({correct}), got '{compared_to}'"


def build_relation_records(
    result_entity: Result,
    intervention_entity_id: str,
    indicator_entity_id: str,
    tissue_entity_id: str,
    control_group_entity_id: Optional[str],
) -> list[Relationship]:
    """Build all Relationship records for a Result entity."""
    rels = []

    if result_entity.relation_type and intervention_entity_id:
        rels.append(Relationship(
            rel_type=result_entity.relation_type,
            head_entity_type="Intervention",
            head_entity_id=intervention_entity_id,
            tail_entity_type="Result",
            tail_entity_id=result_entity.entity_id,
            evidence_text=result_entity.evidence_text,
            source_location=result_entity.source_location,
        ))

    if indicator_entity_id:
        rels.append(Relationship(
            rel_type="corresponds_to",
            head_entity_type="Result",
            head_entity_id=result_entity.entity_id,
            tail_entity_type="Indicator",
            tail_entity_id=indicator_entity_id,
        ))

    if tissue_entity_id:
        rels.append(Relationship(
            rel_type="occurs_in",
            head_entity_type="Result",
            head_entity_id=result_entity.entity_id,
            tail_entity_type="Tissue_Site",
            tail_entity_id=tissue_entity_id,
        ))

    if control_group_entity_id and result_entity.compared_to_group:
        rels.append(Relationship(
            rel_type="compared_to",
            head_entity_type="Result",
            head_entity_id=result_entity.entity_id,
            tail_entity_type="Control_Group",
            tail_entity_id=control_group_entity_id,
        ))

    return rels


# ---------------------------------------------------------------------------
# Stage 3 orchestrator helpers
# ---------------------------------------------------------------------------

def _load_stage2_entities(doi_safe: str) -> dict:
    """Load Stage 2 entity outputs for a given article.

    Returns dict with keys: indicators, controls, tissues, interventions,
    each a list of raw dicts from the stage2 output.
    """
    entities_dir = settings.entities_dir
    result: dict[str, list] = {
        "indicators": [], "controls": [], "tissues": [], "interventions": [],
    }

    # Load indicators (Module B)
    ind_path = entities_dir / doi_safe / "indicators.json"
    if ind_path.exists():
        with open(ind_path, "r") as f:
            data = json.load(f)
            result["indicators"] = data.get("indicators", [])
            result["tissues"] = data.get("tissue_sites", [])

    # Load experiment design (Module A) for controls and interventions
    exp_path = entities_dir / doi_safe / "experiment_design.json"
    if exp_path.exists():
        with open(exp_path, "r") as f:
            data = json.load(f)
            result["controls"] = data.get("control_groups", [])
            result["interventions"] = data.get("interventions", [])

    return result


def _is_challenge_model(doi_safe: str) -> bool:
    """Check if the article uses a challenge model."""
    exp_path = settings.entities_dir / doi_safe / "experiment_design.json"
    if not exp_path.exists():
        return False
    with open(exp_path, "r") as f:
        data = json.load(f)
    return data.get("swine_model", {}).get("model_type") == "challenge"


# ---------------------------------------------------------------------------
# Async orchestrator
# ---------------------------------------------------------------------------

async def _extract_results(
    llm: LLMClient, article: dict, schema: dict,
    indicators: list[dict], controls: list[dict], tissues: list[dict],
) -> dict:
    """Run Pass 1: LLM result extraction for one article."""
    doi = article.get("doi", "unknown")
    results_text = article.get("sections", {}).get("results", {}).get("full_text", "")
    discussion_text = article.get("sections", {}).get("discussion", {}).get("full_text", "")

    if not results_text and not discussion_text:
        logger.warning("No Results/Discussion text for %s", doi)
        return {"doi": doi, "results": []}

    prompt = build_result_prompt(results_text, discussion_text, indicators, controls, tissues)
    result = llm.extract_json(prompt, schema, system_prompt=SYSTEM_PROMPT_RESULT)
    logger.info("Pass 1 done: %s — %d results", doi, len(result.get("results", [])))
    return result


def _run_pass2_alignment(
    raw_results: dict[str, dict],
    entities_dir: Optional[Path] = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Pass 2: Align results to indicators/tissues/controls.

    Returns: (result_dicts, relationship_dicts, warnings)
    """
    entities_dir = entities_dir or settings.entities_dir
    all_results: list[dict] = []
    all_relations: list[dict] = []
    all_warnings: list[dict] = []

    result_seq: dict[str, int] = {}

    for doi_safe, data in raw_results.items():
        if isinstance(data, dict) and "error" in data:
            all_warnings.append({"doi_safe": doi_safe, "error": data["error"]})
            continue

        results = data.get("results", [])
        stage2 = _load_stage2_entities(doi_safe)
        indicators = stage2["indicators"]
        controls = stage2["controls"]
        tissues = stage2["tissues"]
        interventions = stage2["interventions"]
        is_challenge = _is_challenge_model(doi_safe)

        seq = result_seq.get(doi_safe, 0)

        for r in results:
            # Create Indicator/TissueSite objects for alignment
            ind_objs = [Indicator(
                entity_id=ind.get("entity_id", f"{doi_safe}_IND_{i:06d}"),
                doi=data.get("doi", ""),
                standard_name=ind.get("standard_name", ""),
                abbreviation=ind.get("abbreviation", ""),
                indicator_category=ind.get("indicator_category", ""),
                unit=ind.get("unit", ""),
                measurement_method=ind.get("measurement_method", ""),
                measured_in=ind.get("measured_in", ""),
                evidence_text=ind.get("evidence_text", ""),
                source_location=ind.get("source_location", ""),
            ) for i, ind in enumerate(indicators)]

            tis_objs = [TissueSite(
                entity_id=ts.get("entity_id", f"{doi_safe}_TIS_{i:06d}"),
                doi=data.get("doi", ""),
                site_name=ts.get("site_name", ""),
                site_category=ts.get("site_category", ""),
                evidence_text=ts.get("evidence_text", ""),
                source_location=ts.get("source_location", ""),
            ) for i, ts in enumerate(tissues)]

            ctl_objs = [ControlGroup(
                entity_id=cg.get("entity_id", f"{doi_safe}_CTL_{i:06d}"),
                experiment_id=cg.get("experiment_id", f"{doi_safe}_Exp_01"),
                group_name=cg.get("group_name", ""),
                group_type=cg.get("group_type", ""),
                description=cg.get("description", ""),
                evidence_text=cg.get("evidence_text", ""),
                source_location=cg.get("source_location", ""),
            ) for i, cg in enumerate(controls)]

            # Alignment
            ind_id = align_result_to_indicator(r, ind_objs)
            tis_id = align_result_to_tissue(r, tis_objs)

            # Consistency checks
            if not check_direction_consistency(r.get("relation_type", ""), r.get("direction", "")):
                all_warnings.append({"doi_safe": doi_safe, "issue": "direction mismatch", "result": r})

            if not check_compared_to_baseline(r, ctl_objs, is_challenge)[0]:
                all_warnings.append({"doi_safe": doi_safe, "issue": "baseline mismatch", "result": r})

            # Build Result dict
            entity_id = build_entity_id(data.get("doi", doi_safe), "RES", seq)
            seq += 1

            result_dict = {
                "entity_id": entity_id,
                "doi": data.get("doi", ""),
                "experiment_id": f"{data.get('doi', doi_safe).replace('/', '_')}_Exp_01",
                "direction": r.get("direction", ""),
                "p_value": r.get("p_value"),
                "p_value_original_text": r.get("p_value_original_text", ""),
                "corrected_significance": r.get("corrected_significance"),
                "significance_level": r.get("significance_level", ""),
                "effect_size": r.get("effect_size"),
                "time_point": r.get("time_point"),
                "subgroup": r.get("subgroup"),
                "evidence_text": r.get("evidence_text", ""),
                "source_location": r.get("source_location", ""),
                "matched_indicator": ind_id or "",
                "matched_tissue": tis_id or "",
                "compared_to_group": r.get("compared_to_group", ""),
                "relation_type": r.get("relation_type", ""),
            }
            all_results.append(result_dict)

            # Build relationships
            if interventions:
                int_id = interventions[0].get("entity_id", "")
                ctl_id = None
                for cg in controls:
                    if cg.get("group_name") == r.get("compared_to_group"):
                        ctl_id = cg.get("entity_id")
                        break

                # Effect relation
                if result_dict["relation_type"] and int_id:
                    all_relations.append({
                        "rel_type": result_dict["relation_type"],
                        "head_entity_type": "Intervention",
                        "head_entity_id": int_id,
                        "tail_entity_type": "Result",
                        "tail_entity_id": entity_id,
                        "evidence_text": result_dict["evidence_text"],
                        "source_location": result_dict["source_location"],
                    })
                if ind_id:
                    all_relations.append({
                        "rel_type": "corresponds_to",
                        "head_entity_type": "Result",
                        "head_entity_id": entity_id,
                        "tail_entity_type": "Indicator",
                        "tail_entity_id": ind_id,
                    })
                if tis_id:
                    all_relations.append({
                        "rel_type": "occurs_in",
                        "head_entity_type": "Result",
                        "head_entity_id": entity_id,
                        "tail_entity_type": "Tissue_Site",
                        "tail_entity_id": tis_id,
                    })
                if ctl_id:
                    all_relations.append({
                        "rel_type": "compared_to",
                        "head_entity_type": "Result",
                        "head_entity_id": entity_id,
                        "tail_entity_type": "Control_Group",
                        "tail_entity_id": ctl_id,
                    })

        result_seq[doi_safe] = seq

    return all_results, all_relations, all_warnings


async def run_stage3(
    section_files: Optional[list[Path]] = None,
    resume: bool = True,
) -> dict:
    """Run Stage 3 result extraction on all articles that have Stage 2 outputs.

    Args:
        section_files: Section JSON files. If None, globs SECTIONS_DIR.
        resume: If True, skip articles with existing result output.

    Returns:
        Dict with: {results_path, relationships_path, warnings_path, counts}.
    """
    from src.stage2_entity_extract import list_section_files

    if section_files is None:
        section_files = list_section_files()

    schema = LLMClient.load_schema(str(settings.schemas_dir / "results.json"))
    llm = LLMClient()
    semaphore = asyncio.Semaphore(settings.max_concurrent)

    raw_results: dict[str, dict] = {}
    processed = 0
    skipped = 0
    failed = 0

    logger.info("Stage 3: processing articles (max %d concurrent)", settings.max_concurrent)

    async def _limited(coro):
        async with semaphore:
            return await coro

    for sf in section_files:
        doi_safe = sf.stem
        if doi_safe.startswith("_"):
            continue

        if resume and checkpoint_exists(doi_safe, "results"):
            skipped += 1
            continue

        article = load_section_json(doi_safe)
        if article is None:
            continue

        stage2_entities = _load_stage2_entities(doi_safe)

        try:
            result = await _limited(_extract_results(
                llm, article, schema,
                stage2_entities["indicators"],
                stage2_entities["controls"],
                stage2_entities["tissues"],
            ))
            raw_results[doi_safe] = result
            save_entity_output(doi_safe, "results", result)
            processed += 1
        except Exception as e:
            logger.error("Stage 3 failed for %s: %s", doi_safe, e)
            raw_results[doi_safe] = {"error": str(e)}
            failed += 1

    # Pass 2: Alignment
    logger.info("Stage 3 Pass 2: aligning %d articles...", len(raw_results))
    all_results, all_relations, all_warnings = _run_pass2_alignment(raw_results)

    # Persist
    results_path = settings.entities_dir / "_stage3_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    rels_path = settings.relations_dir
    rels_path.mkdir(parents=True, exist_ok=True)
    rels_file = rels_path / "_stage3_relationships.json"
    with open(rels_file, "w", encoding="utf-8") as f:
        json.dump(all_relations, f, ensure_ascii=False, indent=2)

    warn_file = rels_path / "_stage3_warnings.json"
    with open(warn_file, "w", encoding="utf-8") as f:
        json.dump(all_warnings, f, ensure_ascii=False, indent=2)

    logger.info("Stage 3 complete: processed=%d skipped=%d failed=%d results=%d relations=%d warnings=%d",
                processed, skipped, failed, len(all_results), len(all_relations), len(all_warnings))

    return {
        "results_path": str(results_path),
        "relationships_path": str(rels_file),
        "warnings_path": str(warn_file),
        "counts": {"processed": processed, "skipped": skipped, "failed": failed,
                   "results": len(all_results), "relations": len(all_relations)},
    }
