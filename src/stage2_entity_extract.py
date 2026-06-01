"""Stage 2: Entity extraction — helpers and async orchestrator for Modules C/A/B."""
import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from src.config import settings
from src.llm_client import LLMClient
from src.glossary import GlossaryIndex
from src.stage2_prompts import (
    build_module_c_prompt, build_module_a_prompt, build_module_b_prompt,
    SYSTEM_PROMPT_ENTITY,
)

logger = logging.getLogger(__name__)


def build_entity_id(doi: str, entity_type_abbr: str, seq: int) -> str:
    """Build unique entity ID: DOI_safe + type abbreviation + zero-padded sequence (6 digits)."""
    doi_safe = doi.replace(":", "_")
    return f"{doi_safe}_{entity_type_abbr}_{seq:06d}"


def normalize_dose_unit(value: float, unit: str) -> tuple[float, str]:
    """Normalize dose units per BACKGROUND.md 3.4.

    Rules:
    - ppm -> mg/kg_feed (1:1)
    - % -> mg/kg_feed (1% = 10000 mg/kg)
    - mg/kg BW -> mg/kg_BW
    - mg/kg (without BW) -> mg/kg_feed
    - g/kg -> g/kg (pass through)
    - Unknown units pass through unchanged.
    """
    unit_lower = unit.lower().strip()
    if "ppm" in unit_lower:
        return value, "mg/kg_feed"
    if "%" in unit_lower:
        return value * 10000, "mg/kg_feed"
    if "mg/kg bw" in unit_lower or "mg/kg_bw" in unit_lower:
        return value, "mg/kg_BW"
    if "mg/kg" in unit_lower:
        return value, "mg/kg_feed"
    if "g/kg" in unit_lower:
        return value, "g/kg"
    return value, unit


def entities_to_tsv_rows(entities: list) -> list[dict]:
    """Convert a list of entity dataclass instances to list of dicts for TSV writing."""
    return [asdict(e) for e in entities]


def postprocess_alternatives(data: dict) -> dict:
    """Deduplicate alternatives within same DOI by standard_name (case-insensitive)."""
    seen = set()
    deduped = []
    for alt in data.get("alternatives", []):
        key = alt.get("standard_name", "").lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(alt)
    data["alternatives"] = deduped
    return data


def postprocess_experiment_design(data: dict) -> dict:
    """Normalize dose units in intervention data."""
    for inter in data.get("interventions", []):
        if inter.get("dose_value") is not None and inter.get("dose_unit_original"):
            new_val, new_unit = normalize_dose_unit(
                float(inter.get("dose_value", 0)), inter.get("dose_unit_original", "")
            )
            inter["dose_unit_standard"] = new_unit
            inter["dose_value"] = new_val
    return data


# ---------------------------------------------------------------------------
# Orchestrator helpers
# ---------------------------------------------------------------------------

def list_section_files(sections_dir: Optional[Path] = None) -> list[Path]:
    """List all JSON section files in the sections directory, excluding _failures."""
    d = sections_dir or settings.sections_dir
    if not d.exists():
        return []
    return sorted(
        p for p in d.glob("*.json")
        if not p.name.startswith("_")
    )


def load_section_json(doi_safe: str, sections_dir: Optional[Path] = None) -> Optional[dict]:
    """Load a single article's structured section JSON. Returns None if missing."""
    d = sections_dir or settings.sections_dir
    path = d / f"{doi_safe}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def checkpoint_exists(doi_safe: str, module: str, entities_dir: Optional[Path] = None) -> bool:
    """Check if entity extraction output already exists for this article+module."""
    d = entities_dir or settings.entities_dir
    return (d / doi_safe / f"{module}.json").exists()


def save_entity_output(
    doi_safe: str, module: str, data: dict, entities_dir: Optional[Path] = None
) -> Path:
    """Save LLM extraction output for one article+module. Returns the saved path."""
    d = entities_dir or settings.entities_dir
    out_dir = d / doi_safe
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{module}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Async orchestrator
# ---------------------------------------------------------------------------

async def _extract_module_c(
    llm: LLMClient, article: dict, schema: dict, glossary: GlossaryIndex,
) -> dict:
    """Run Module C (alternatives) on a single article."""
    doi = article.get("doi", "unknown")
    mm_text = article.get("sections", {}).get("materials_and_methods", {}).get("full_text", "")
    if not mm_text:
        logger.warning("No M&M text for %s, skipping Module C", doi)
        return {"doi": doi, "alternatives": [], "composite_products": [], "warnings": ["No M&M text"]}

    prompt = build_module_c_prompt(mm_text)
    result = llm.extract_json(prompt, schema, system_prompt=SYSTEM_PROMPT_ENTITY)
    result = postprocess_alternatives(result)
    logger.info("Module C done: %s — %d alternatives, %d composites",
                doi, len(result.get("alternatives", [])), len(result.get("composite_products", [])))
    return result


async def _extract_module_a(
    llm: LLMClient, article: dict, schema: dict, glossary: GlossaryIndex,
) -> dict:
    """Run Module A (experiment design) on a single article."""
    doi = article.get("doi", "unknown")
    mm_text = article.get("sections", {}).get("materials_and_methods", {}).get("full_text", "")
    if not mm_text:
        return {"doi": doi, "experiment_id": f"{doi}_Exp_01", "swine_model": {}, "swine": {}, "interventions": [], "control_groups": []}

    prompt = build_module_a_prompt(mm_text)
    result = llm.extract_json(prompt, schema, system_prompt=SYSTEM_PROMPT_ENTITY)
    result = postprocess_experiment_design(result)
    logger.info("Module A done: %s — %d interventions", doi, len(result.get("interventions", [])))
    return result


async def _extract_module_b(
    llm: LLMClient, article: dict, schema: dict,
) -> dict:
    """Run Module B (indicators) on a single article."""
    doi = article.get("doi", "unknown")
    mm_text = article.get("sections", {}).get("materials_and_methods", {}).get("full_text", "")
    if not mm_text:
        return {"doi": doi, "tissue_sites": [], "indicators": [], "methods": []}

    prompt = build_module_b_prompt(mm_text)
    result = llm.extract_json(prompt, schema, system_prompt=SYSTEM_PROMPT_ENTITY)
    logger.info("Module B done: %s — %d indicators, %d tissue sites",
                doi, len(result.get("indicators", [])), len(result.get("tissue_sites", [])))
    return result


async def run_stage2(
    section_files: Optional[list[Path]] = None,
    resume: bool = True,
) -> dict[str, int]:
    """Run Stage 2 entity extraction on all section files.

    Args:
        section_files: List of JSON files to process. If None, globs SECTIONS_DIR.
        resume: If True, skip articles with existing output (checkpoint).

    Returns:
        Dict with counts: {processed, skipped, failed_c, failed_a, failed_b}.
    """
    if section_files is None:
        section_files = list_section_files()

    if not section_files:
        logger.warning("No section files found in %s", settings.sections_dir)
        return {"processed": 0, "skipped": 0, "failed_c": 0, "failed_a": 0, "failed_b": 0}

    # Load schemas
    schemas = {
        "C": LLMClient.load_schema(str(settings.schemas_dir / "alternatives.json")),
        "A": LLMClient.load_schema(str(settings.schemas_dir / "experiment_design.json")),
        "B": LLMClient.load_schema(str(settings.schemas_dir / "indicators.json")),
    }

    glossary = GlossaryIndex()
    glossary.load(str(settings.alternative_tsv))

    llm = LLMClient()
    semaphore = asyncio.Semaphore(settings.max_concurrent)

    counts = {"processed": 0, "skipped": 0, "failed_c": 0, "failed_a": 0, "failed_b": 0}

    async def _limited(coro):
        async with semaphore:
            return await coro

    articles: list[dict] = []
    for sf in section_files:
        article = load_section_json(sf.stem)
        if article:
            articles.append(article)

    logger.info("Stage 2: processing %d articles (max %d concurrent, batch %d)",
                len(articles), settings.max_concurrent, settings.batch_size)

    for i in range(0, len(articles), settings.batch_size):
        batch = articles[i:i + settings.batch_size]
        logger.info("Batch %d/%d (%d articles)", i // settings.batch_size + 1,
                    (len(articles) + settings.batch_size - 1) // settings.batch_size, len(batch))

        for article in batch:
            doi = article.get("doi", "unknown")
            doi_safe = doi.replace("/", "_").replace(":", "_")

            # Module C first (dependency for A)
            if resume and checkpoint_exists(doi_safe, "alternatives"):
                counts["skipped"] += 1
                logger.debug("Skipping %s (already processed)", doi)
                continue

            try:
                result_c = await _limited(_extract_module_c(llm, article, schemas["C"], glossary))
                save_entity_output(doi_safe, "alternatives", result_c)
            except Exception as e:
                logger.error("Module C failed for %s: %s", doi, e)
                counts["failed_c"] += 1
                continue

            # Modules A and B in parallel (both depend on C completion)
            try:
                result_a, result_b = await asyncio.gather(
                    _limited(_extract_module_a(llm, article, schemas["A"], glossary)),
                    _limited(_extract_module_b(llm, article, schemas["B"])),
                )
                save_entity_output(doi_safe, "experiment_design", result_a)
                save_entity_output(doi_safe, "indicators", result_b)
            except Exception as e:
                logger.error("Module A/B failed for %s: %s", doi, e)
                if isinstance(e, RuntimeError):
                    counts["failed_a"] += 1
                counts["failed_b"] += 1
                continue

            counts["processed"] += 1

    logger.info("Stage 2 complete: processed=%d skipped=%d failed(C=%d A=%d B=%d)",
                counts["processed"], counts["skipped"],
                counts["failed_c"], counts["failed_a"], counts["failed_b"])
    return counts
