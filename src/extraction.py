"""Main extraction API — coordinates 4-phase extraction per article."""
from __future__ import annotations

import logging
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
    model_id: str = "deepseek-chat",
    api_key: str | None = None,
    registry=None,
    max_char_buffer: int = 8000,
    skip_if_no_known_alternative: bool = True,
    **kwargs,
) -> ArticleExtractionResult:
    """Extract entities from one PMC XML article using 4-phase pipeline.

    Parameters
    ----------
    article_xml : str or Path
        Path to a PMC XML article file.
    model : BaseLanguageModel or None
        Pre-configured language model.  Created via :func:`create_model` when
        not supplied.
    model_id : str
        Model identifier used when *model* is ``None``.  Default ``"deepseek-chat"``.
    api_key : str or None
        API key used when *model* is ``None``.
    registry : SchemaRegistry or None
        Schema registry for post-processing.  Loaded from default config when
        not supplied.
    max_char_buffer : int
        Maximum characters per chunk.  Default 8000.
    skip_if_no_known_alternative : bool
        When ``True``, articles with no recognized alternatives are skipped.
    **kwargs
        Passed through to :meth:`Annotator.annotate_documents`.

    Returns
    -------
    ArticleExtractionResult
    """
    # 1. Load registry if not provided
    if registry is None:
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry()

    # 2. Create model if not provided
    if model is None:
        from src.factory import ModelConfig, create_model

        config = ModelConfig(
            model_id=model_id,
            provider_kwargs={"api_key": api_key or ""},
        )
        model = create_model(config)

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
            result.warnings.append(f"Phase {phase_idx+1} ({phase.name}): no source text available")
            continue

        section_label = section_name_map.get(phase_idx, "Full Text")
        source_doc = Document(text=section_text, document_id=f"{pmid}_{section_label.replace(' ', '_')}")

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

        all_extractions.extend(phase_exts)
        phase_results[phase.name] = phase_exts

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
                return result

    # Post-process all extractions
    for ext in all_extractions:
        try:
            registry.post_process(ext)
        except Exception:
            pass

    result.extractions = all_extractions
    return result


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
