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
class DocumentExtractionResult:
    """Extraction result for one document — format-agnostic."""
    document_id: str
    metadata: dict = field(default_factory=dict)   # format-specific metadata
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
    gate_enabled: bool | None = None,
    **kwargs,
) -> DocumentExtractionResult:
    """Extract entities from one PMC XML article using 4-phase pipeline.

    All defaults come from ``src.config.settings`` unless explicitly overridden.
    """
    from src.config import settings

    if model_id is None:
        model_id = settings.llm_model
    if max_char_buffer is None:
        max_char_buffer = settings.max_char_buffer
    if gate_enabled is None:
        gate_enabled = settings.gate_enabled

    # 1. Load registry if not provided
    if registry is None:
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry()

    # 2. Create model if not provided (uses global settings)
    if model is None:
        from src.factory import create_model

        model = create_model()

    t0 = time.time()

    # 3. Parse document — derive ID from filename
    article_path = Path(article_xml)
    doc_id = article_path.stem  # e.g. "PMC12183824"
    sections = _parse_article_sections(article_path)

    # Metadata from parser (format-specific: doi, pmid, title, etc.)
    metadata = {k: v for k, v in sections.items()
                if k in ("doi", "pmid", "title", "journal")}
    result = DocumentExtractionResult(document_id=doc_id, metadata=metadata)

    # Populate Literature entity from metadata (not LLM extraction)
    lit_ext = _make_literature_entity(doc_id, sections)
    all_extractions: list[Extraction] = [lit_ext] if lit_ext else []

    # Sections are resolved per-phase via extraction_phases.yaml.
    # If a phase's required sections are empty, only that phase is skipped.

    # 4-7: 4-phase extraction
    from src.format_handler import FormatHandler
    from src.prompting import PromptTemplateStructured
    from src.annotation import Annotator

    fh = FormatHandler(use_fences=model.requires_fence_output)
    total_phases = len(registry.phase_defs())
    phase_results: dict[str, list[Extraction]] = {}

    for phase_idx, phase in enumerate(registry.phase_defs()):
        prompt_text = registry.build_extraction_prompt(phase.extracts)

        # Resolve section text from phase config (fallback to "body")
        preferred = getattr(phase, "sections", None) or ["body"]
        section_text = ""
        for sec_name in preferred:
            section_text = sections.get(sec_name, "")
            if section_text.strip():
                break

        if not section_text.strip():
            logger.warning("Phase %d (%s): no text for sections %s",
                           phase_idx + 1, phase.name, preferred)
            result.warnings.append(
                f"Phase {phase_idx+1} ({phase.name}): no source text available"
            )
            continue

        section_label = preferred[0].replace("_", " ").title()
        source_doc = Document(text=section_text, document_id=f"{doc_id}_{section_label.replace(' ', '_')}")

        # Progress
        phase_label = f"Phase {phase_idx+1}/{total_phases}: {phase.name}"
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
        # Gate check — driven entirely by extraction_phases.yaml
        if phase_idx == 0 and gate_enabled and phase.gate:
            passed = registry.evaluate_gate(
                phase_exts, phase.gate.entity, phase.gate.condition,
            )
            if not passed:
                result.skipped = True
                result.skip_reason = (
                    f"Gate check failed: {phase.gate.entity} {phase.gate.condition}"
                )
                result.extractions = all_extractions
                _save_checkpoint(result, all_extractions, doc_id, phase_exts)
                logger.info("Gate check failed — checkpoint saved")
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
    _save_checkpoint(result, all_extractions, doc_id, all_extractions)

    result.extractions = all_extractions
    return result


# ---------------------------------------------------------------------------
# Checkpoint persistence
# ---------------------------------------------------------------------------

def _save_checkpoint(
    result: DocumentExtractionResult,
    all_extractions: list[Extraction],
    doc_id: str,
    phase1_exts: list[Extraction],
) -> None:
    """Save intermediate extraction results to disk for manual inspection.

    Writes ``data/intermediates/{doc_id}.json``.
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
        "document_id": doc_id,
        "metadata": result.metadata,
        "skipped": result.skipped,
        "skip_reason": result.skip_reason,
        "total_entities": len(all_extractions),
        "phase1_entities": [_ext_to_dict(e) for e in phase1_exts],
        "all_entities": [_ext_to_dict(e) for e in all_extractions],
        "warnings": result.warnings,
    }

    safe_id = doc_id.replace("/", "_").replace(":", "_") or "unknown"
    out_path = out_dir / f"{safe_id}.json"
    out_path.write_text(_json.dumps(payload, ensure_ascii=False, indent=2))
    if result.skipped:
        print(f"    ⚠ Gate failed — Phase 1 results saved to {out_path}", flush=True)
    else:
        logger.debug("Checkpoint saved to %s", out_path)


def _make_literature_entity(
    doc_id: str, sections: dict[str, str],
) -> Extraction | None:
    """Create a Literature Extraction from document metadata.

    Literature metadata comes from the parser (doi, pmid, title) and is
    NOT extracted by the LLM.  This keeps the LLM focused on domain entities.
    """
    from src.data import Extraction

    attrs = {
        "doi": sections.get("doi", ""),
        "pmid": sections.get("pmid", ""),
        "title": sections.get("title", ""),
        "journal": sections.get("journal", ""),
        "evidence_text": "",
        "source_location": "front-matter",
    }
    # Only create if we have at least a title or DOI
    if not attrs["title"] and not attrs["doi"]:
        return None
    return Extraction(
        extraction_class="Literature",
        extraction_text=attrs["title"] or doc_id,
        attributes=attrs,
    )


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

            # FIXME pmc 的所有情况不一定能够完整覆盖，需要考虑更大的兼容性，该代码适合放到单独文件中
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
    design_result : DocumentExtractionResult
        Result from Phase 2 (Experiment Design).
    indicator_result : DocumentExtractionResult
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
