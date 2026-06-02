"""Main extraction API — coordinates 4-phase extraction per article."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.data import Extraction

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

    if not any(sections.get(k) for k in ("abstract", "methods", "results", "discussion")):
        result.skipped = True
        result.skip_reason = "No extractable sections found"
        return result

    # 4-7: 4-phase extraction
    # This is a scaffold — the full 4-phase orchestration uses the Annotator
    # with phase-specific templates driven by SchemaRegistry extraction_phases.
    # For now, return an empty result (extraction is done via the Annotator
    # directly or through a higher-level pipeline driver).
    result.warnings.append("4-phase pipeline not yet wired — use Annotator directly")
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
            return result

        sections = body.findall(".//sec", nsmap)
        for sec in sections:
            title_el = sec.find("./title", nsmap)
            title_text = " ".join(title_el.itertext()).strip().lower() if title_el is not None else ""
            body_text = _element_text(sec, nsmap)

            if "abstract" in title_text or "abstract" == _get_section_type(sec):
                result["abstract"] = body_text
            elif "method" in title_text or "materials and methods" in title_text:
                result["methods"] = body_text
            elif "result" in title_text:
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
    """Extract namespace map from root element."""
    nsmap: dict[str, str] = {}
    # Iterate through nsmap from the root tag
    tag = root.tag
    if "}" in tag:
        nsmap[""] = tag.split("}")[0].lstrip("{")
    # Collect all namespaces
    for elem in root.iter():
        if "}" in elem.tag:
            ns = elem.tag.split("}")[0].lstrip("{")
            if "" not in nsmap:
                nsmap[""] = ns
            break
    return nsmap or {}


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
