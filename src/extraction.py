"""Main extraction API — coordinates 4-phase extraction per article."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.data import Document, Extraction

try:
    from lxml import etree
except ImportError:
    etree = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class ArticleExtractionResult:
    doi: str
    pmid: str
    extractions: list[Extraction] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""
    warnings: list[str] = field(default_factory=list)


def extract(
    article_xml,
    *,
    model=None,
    model_id: str | None = None,
    api_key: str | None = None,
    registry=None,
    max_char_buffer: int | None = None,
    skip_if_no_known_alternative: bool | None = None,
    **kwargs,
) -> ArticleExtractionResult:
    """Extract entities from one PMC XML article using 4-phase pipeline.

    All defaults come from ``src.config.settings`` unless explicitly overridden.
    """
    from src.config import settings

    if model_id is None:
        model_id = settings.llm_model
    if max_char_buffer is None:
        max_char_buffer = settings.max_char_buffer
    if skip_if_no_known_alternative is None:
        skip_if_no_known_alternative = settings.skip_if_no_known_alternative

    # 1. Load registry if not provided
    if registry is None:
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry()

    # 2. Create model if not provided (uses global settings)
    if model is None:
        from src.factory import create_model

        model = create_model()

    t0 = time.time()

    # 3. Parse article sections
    article_path = Path(article_xml)
    sections = _parse_article_sections(article_path)

    doi = sections.get("doi", "")
    pmid = sections.get("pmid", "")
    result = ArticleExtractionResult(doi=doi, pmid=pmid)

    if not any(sections.get(k) for k in ("abstract", "methods", "results", "discussion", "body")):
        result.skipped = True
        result.skip_reason = "No extractable sections found"
        return result

    # 4-7: 4-phase extraction
    from src.format_handler import FormatHandler
    from src.prompting import PromptTemplateStructured
    from src.annotation import Annotator

    fh = FormatHandler(use_fences=model.requires_fence_output)
    all_extractions: list[Extraction] = []
    phase_results: dict[str, list[Extraction]] = {}

    # Determine section names for source_location tracking
    section_name_map = {
        0: "Materials and Methods",
        1: "Materials and Methods",
        2: "Materials and Methods",
        3: "Results and Discussion",
    }

    for phase_idx, phase in enumerate(registry.phase_defs()):
        # Build prompt from entity metadata
        prompt_text = registry.build_extraction_prompt(phase.extracts)

        # Determine which section text to use
        if phase_idx == 0:
            section_text = sections.get("methods") or sections.get("body") or sections.get("abstract") or ""
        elif phase_idx <= 2:
            section_text = sections.get("methods") or sections.get("body") or ""
        else:
            section_text = sections.get("results") or sections.get("discussion") or sections.get("body") or ""

        if not section_text.strip():
            logger.warning("Phase %d (%s): no source text available, skipping", phase_idx + 1, phase.name)
            result.warnings.append(f"Phase {phase_idx+1} ({phase.name}): no source text available")
            continue

        section_label = section_name_map.get(phase_idx, "Full Text")
        source_doc = Document(text=section_text, document_id=f"{pmid}_{section_label.replace(' ', '_')}")

        # Progress
        phase_label = f"Phase {phase_idx+1}/4: {phase.name}"
        text_len = len(section_text)
        logger.info("%s — extracting from %s (%d chars, section=%s)",
                     phase_label, ", ".join(phase.extracts), text_len, section_label)
        print(f"  {phase_label} ({', '.join(phase.extracts)}) — {text_len} chars", flush=True)

        # Inject context from prior phases into Phase 4
        additional_context = None
        if phase.context_from:
            ctx_parts = []
            for ctx_phase_name in phase.context_from:
                ctx_exts = phase_results.get(ctx_phase_name, [])
                if ctx_exts:
                    ctx_parts.append(_build_context_from_extractions(ctx_exts))
            if ctx_parts:
                additional_context = "\n\n".join(ctx_parts)

        template = PromptTemplateStructured(description=prompt_text)
        annotator = Annotator(model, template, fh)
        phase_start = time.time()
        try:
            doc_result = annotator.annotate_text(
                source_doc.text,
                max_char_buffer=max_char_buffer,
                additional_context=additional_context,
                document_id=source_doc.document_id,
                **kwargs,
            )
            phase_exts = doc_result.extractions or []
        except Exception as e:
            logger.warning("Phase %d (%s) failed: %s", phase_idx + 1, phase.name, e)
            result.warnings.append(f"Phase {phase_idx+1} failed: {e}")
            continue

        phase_elapsed = time.time() - phase_start
        all_extractions.extend(phase_exts)
        phase_results[phase.name] = phase_exts

        # Count entity types for feedback
        type_counts: dict[str, int] = {}
        for ext in phase_exts:
            type_counts[ext.extraction_class] = type_counts.get(ext.extraction_class, 0) + 1
        type_summary = ", ".join(f"{t}={c}" for t, c in sorted(type_counts.items()))
        logger.info("%s done — %d entities in %.1fs: %s",
                     phase_label, len(phase_exts), phase_elapsed, type_summary)
        print(f"    → {len(phase_exts)} entities in {phase_elapsed:.1f}s: {type_summary}", flush=True)

        # Gate check after Phase 1
        if phase_idx == 0 and skip_if_no_known_alternative and phase.gate:
            known_classes = {
                "Plant_Extract", "Trace_Element", "Organic_Acid", "Probiotic",
                "Polysaccharides_and_Oligosaccharides", "Enzyme", "Bioactive_Peptides",
            }
            alt_exts = [e for e in phase_exts if e.extraction_class == "Alternative"]
            has_known = any(
                (e.attributes or {}).get("alternative_class") in known_classes
                for e in alt_exts
            )
            if not has_known:
                result.skipped = True
                result.skip_reason = "No known Alternative found (gate check)"
                result.extractions = all_extractions
                # Save intermediate checkpoint for manual inspection
                _save_checkpoint(
                    result, all_extractions, pmid, phase_exts,
                )
                logger.info(
                    "Gate check failed — %d alternatives found, none in known classes. "
                    "Checkpoint saved to data/intermediates/",
                    len(alt_exts),
                )
                return result

    # Post-process all extractions
    for ext in all_extractions:
        try:
            registry.post_process(ext)
        except Exception:
            pass

    total_time = time.time() - t0
    type_counts_final: dict[str, int] = {}
    for ext in all_extractions:
        type_counts_final[ext.extraction_class] = type_counts_final.get(ext.extraction_class, 0) + 1
    logger.info("Extraction complete: %d entities in %.1fs — %s",
                 len(all_extractions), total_time,
                 ", ".join(f"{t}={c}" for t, c in sorted(type_counts_final.items())))
    print(f"  Total: {len(all_extractions)} entities in {total_time:.1f}s", flush=True)

    # Save checkpoint for successful extraction too
    _save_checkpoint(result, all_extractions, pmid, all_extractions)

    result.extractions = all_extractions
    return result


# ---------------------------------------------------------------------------
# Checkpoint persistence
# ---------------------------------------------------------------------------

def _save_checkpoint(
    result: ArticleExtractionResult,
    all_extractions: list[Extraction],
    pmid: str,
    phase1_exts: list[Extraction],
) -> None:
    """Save intermediate extraction results to disk for manual inspection.

    Writes ``data/intermediates/{pmid}.json`` with a human-readable summary
    of extracted entities, organized by phase.
    """
    import json as _json

    out_dir = Path("data/intermediates")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _ext_to_dict(ext: Extraction) -> dict:
        return {
            "class": ext.extraction_class,
            "text": ext.extraction_text,
            "attributes": ext.attributes or {},
        }

    payload = {
        "doi": result.doi,
        "pmid": pmid,
        "skipped": result.skipped,
        "skip_reason": result.skip_reason,
        "total_entities": len(all_extractions),
        "phase1_entities": [_ext_to_dict(e) for e in phase1_exts],
        "all_entities": [_ext_to_dict(e) for e in all_extractions],
        "warnings": result.warnings,
    }

    safe_pmid = pmid or result.doi.replace("/", "_").replace(":", "_") or "unknown"
    out_path = out_dir / f"{safe_pmid}.json"
    out_path.write_text(_json.dumps(payload, ensure_ascii=False, indent=2))
    if result.skipped:
        print(f"    ⚠ Gate failed — Phase 1 results saved to {out_path}", flush=True)
    else:
        logger.debug("Checkpoint saved to %s", out_path)


def _parse_article_sections(xml_path: Path) -> dict[str, str]:
    """Parse PMC XML into {doi, pmid, abstract, methods, results, discussion}.

    Uses lxml.  Returns empty strings for missing sections.
    """
    if etree is None:
        logger.warning("lxml not installed — cannot parse PMC XML")
        return {
            "doi": "",
            "pmid": "",
            "abstract": "",
            "methods": "",
            "results": "",
            "discussion": "",
        }

    result: dict[str, str] = {
        "doi": "",
        "pmid": "",
        "abstract": "",
        "methods": "",
        "results": "",
        "discussion": "",
        "body": "",
    }

    try:
        tree = etree.parse(str(xml_path))
        root = tree.getroot()
        nsmap = _get_namespace(root)

        # --- DOI / PMID from front matter ---
        front = root.find(".//front", nsmap)
        if front is not None:
            for aid in front.findall(".//article-id", nsmap):
                aid_type = aid.get("pub-id-type", "")
                if aid_type == "doi":
                    result["doi"] = (aid.text or "").strip()
                elif aid_type == "pmid":
                    result["pmid"] = (aid.text or "").strip()

        # --- Body sections ---
        body = root.find(".//body", nsmap)
        if body is None:
            # Fallback: try to get text from the entire article
            result["body"] = _element_text(root, nsmap)
            return result

        # Store full body text as fallback
        result["body"] = _element_text(body, nsmap)

        sections = body.findall(".//sec", nsmap)
        for sec in sections:
            title_el = sec.find("./title", nsmap)
            title_text = " ".join(title_el.itertext()).strip().lower() if title_el is not None else ""
            body_text = _element_text(sec, nsmap)

            if "abstract" in title_text or "abstract" == _get_section_type(sec):
                result["abstract"] = body_text
            elif "method" in title_text or "materials and methods" in title_text or "materials" in title_text:
                if result["methods"]:
                    result["methods"] += "\n\n" + body_text
                else:
                    result["methods"] = body_text
            elif "result" in title_text:
                if result["results"]:
                    result["results"] += "\n\n" + body_text
                else:
                    result["results"] = body_text
            elif "discussion" in title_text or "conclusion" in title_text:
                if result["discussion"]:
                    result["discussion"] += "\n\n" + body_text
                else:
                    result["discussion"] = body_text

        # Fallback: if abstract not found, try front matter
        if not result["abstract"]:
            abstract_sec = root.find(".//front//abstract", nsmap)
            if abstract_sec is not None:
                result["abstract"] = _element_text(abstract_sec, nsmap)

        return result

    except Exception:
        logger.exception("Failed to parse PMC XML: %s", xml_path)
        return result


def _get_namespace(root) -> dict[str, str]:
    """Extract namespace map from PMC XML article.

    PMC article sets have: <pmc-articleset><article xmlns:...="">...
    The article element carries the namespace declarations for JATS XML.
    """
    nsmap: dict[str, str] = {}
    # Try to get namespaces from the <article> child (not the root <pmc-articleset>)
    article = root.find("article") if root.tag == "pmc-articleset" else root
    if article is None:
        article = root
    # Use lxml's nsmap on the article element
    if hasattr(article, 'nsmap') and article.nsmap:
        for prefix, uri in article.nsmap.items():
            if prefix is None:
                nsmap[""] = uri  # default namespace
            else:
                nsmap[prefix] = uri
    # Fallback: look at tag
    if not nsmap:
        tag = article.tag
        if "}" in tag:
            nsmap[""] = tag.split("}")[0].lstrip("{")
    return nsmap


def _get_section_type(sec) -> str:
    """Get the sec-type attribute if present."""
    return (sec.get("sec-type") or "").lower()


def _element_text(elem, nsmap) -> str:
    """Extract full text content from an element, joining all text nodes."""
    if elem is None:
        return ""
    parts: list[str] = []
    try:
        text = "".join(elem.itertext())
        parts.append(text.strip())
    except Exception:
        pass
    return " ".join(p for p in parts if p)


def _build_context_from_extractions(extractions: list[Extraction]) -> str:
    """Build a compact context listing from a list of Extractions.

    Used to pass Phase 2+3 results into Phase 4 prompts.
    """
    lines: list[str] = []
    # Group by entity type
    by_type: dict[str, list[Extraction]] = {}
    for ext in extractions:
        by_type.setdefault(ext.extraction_class, []).append(ext)

    for etype, exts in sorted(by_type.items()):
        lines.append(f"Available {etype}s:")
        for ext in exts[:30]:  # limit to 30 per type
            attrs = ext.attributes or {}
            name = attrs.get("abbreviation") or attrs.get("group_name") or ext.extraction_text
            extra = ""
            if etype == "Indicator":
                extra = f" ({attrs.get('standard_name', '')})"
            elif etype == "Control_Group":
                extra = f" ({attrs.get('group_type', '')})"
            lines.append(f"  - {name}{extra}")
    return "\n".join(lines)


def _build_results_context(design_result, indicator_result) -> str:
    """Build context string listing available indicators and control groups.

    Parameters
    ----------
    design_result : ArticleExtractionResult
        Result from Phase 2 (Experiment Design).
    indicator_result : ArticleExtractionResult
        Result from Phase 3 (Indicators).

    Returns
    -------
    str
        Context string to inject into Phase 4 prompts.
    """
    lines: list[str] = []

    # Collect indicators
    indicators: set[str] = set()
    for ext in indicator_result.extractions:
        if ext.extraction_class == "Indicator":
            indicators.add(ext.extraction_text)

    # Collect groups
    groups: set[str] = set()
    for ext in design_result.extractions:
        if ext.extraction_class in ("ExperimentalGroup", "ControlGroup"):
            groups.add(ext.extraction_text)

    if indicators:
        lines.append("Available indicators: " + ", ".join(sorted(indicators)))
    if groups:
        lines.append("Available groups: " + ", ".join(sorted(groups)))

    return "\n".join(lines) if lines else ""
