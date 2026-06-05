"""PMC Section-based Extraction Adapter.

Routes entity types to specific article sections as specified in BACKGROUND.md:

    Abstract        → Literature (abstract_conclusion)
    Materials & Methods → Alternative, Composite_Product, Swine,
                           Intervention, Control_Group, Tissue_Site, Indicator, Method
    Results & Discussion → Result

The generic ``src.extraction.extract()`` is called once per section with the
appropriate entity-type filter.  Results are then merged and deduplicated.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.adapters.pmc import (
    PmcArticleMeta,
    parse_article,
    build_document,
    literature_pre_extractor,
    export_entity_review_csv,
    _extract_conclusion,
)
from src.aligner import align_and_evidence_batch
from src.coreference import resolve_coreferences, clean_evidence_batch
from src.data import Document, Extraction
from src.extraction import DocumentExtractionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section → Entity type routing (from BACKGROUND.md)
# ---------------------------------------------------------------------------

# Entities extracted from Materials & Methods — default fallback.
# Actual routing is derived from extraction_phases.yaml via
# registry.build_section_routing() when a registry is available.
_METHODS_ENTITIES = [
    "Alternative",
    "Composite_Product",
    "Swine",
    "Swine_Model",
    "Intervention",
    "Control_Group",
    "Tissue_Site",
    "Indicator",
    "Method",
    "Experiment",
]

# Entities extracted from Results & Discussion — default fallback.
_RESULTS_ENTITIES = [
    "Result",
]

# Literature is pre-extracted (never sent to LLM)


def _get_section_entities(registry, section_name: str) -> list[str]:
    """Get entity types to extract for a given section from phase config.

    Falls back to the hardcoded lists if no registry or phase config is available.

    When multiple phases target the same section (e.g., gate + bulk both use
    ``sections: [body]`` for methods), their entity types are **combined**
    so that all required entity types are extracted from that section.
    """
    if registry is not None:
        try:
            routing = registry.build_section_routing()
            combined: list[str] = []
            # Collect entity names from all phases whose section matches
            for entry in routing:
                if entry["section"] == section_name:
                    combined.extend(entry["entity_names"])
            if combined:
                return list(dict.fromkeys(combined))  # dedup, preserve order
            # If exact section match fails, try matching "body" for methods
            # (both gate and bulk phases use sections: [body])
            if section_name == "methods":
                for entry in routing:
                    if entry["section"] == "body":
                        combined.extend(entry["entity_names"])
                if combined:
                    return list(dict.fromkeys(combined))
        except Exception:
            pass
    # Fallback
    return _METHODS_ENTITIES if section_name == "methods" else _RESULTS_ENTITIES


def _build_indicator_context(extractions: list) -> str:
    """Build a reference list of Indicators for the Results prompt.

    CRITICAL: The Results LLM MUST use the EXACT abbreviation from this list
    as the indicator_abbreviation field value.
    """
    lines = ["CRITICAL: Use only these abbreviations for Result.indicator_abbreviation:"]
    for ext in extractions:
        if ext.extraction_class != "Indicator":
            continue
        abbr = (ext.attributes or {}).get("abbreviation", "") or ext.extraction_text
        std = (ext.attributes or {}).get("standard_name", "")
        site = (ext.attributes or {}).get("measured_in", "")
        txt = ext.extraction_text[:60]
        parts = [f'abbreviation="{abbr}"']
        if std and std != abbr:
            parts.append(f'full_name="{std}"')
        if site:
            parts.append(f'measured_in="{site}"')
        parts.append(f'| extracted_as="{txt}"')
        lines.append("  " + ", ".join(parts))
    return "\n".join(lines) if lines else ""


def _build_control_context(extractions: list) -> str:
    """Build a compact reference list of Control_Groups for the Results prompt."""
    lines = []
    for ext in extractions:
        if ext.extraction_class != "Control_Group":
            continue
        name = (ext.attributes or {}).get("group_name", "") or ext.extraction_text
        gtype = (ext.attributes or {}).get("group_type", "")
        desc = (ext.attributes or {}).get("description", "")
        parts = [name]
        if gtype:
            parts.append(f"({gtype})")
        if desc:
            parts.append(f"- {desc[:60]}")
        lines.append("  " + " ".join(parts))
    return "\n".join(lines) if lines else ""


def _build_tissue_context(extractions: list) -> str:
    """Build a compact reference list of Tissue_Sites for the Results prompt."""
    lines = []
    for ext in extractions:
        if ext.extraction_class != "Tissue_Site":
            continue
        name = (ext.attributes or {}).get("site_name", "") or ext.extraction_text
        cat = (ext.attributes or {}).get("site_category", "")
        parts = [name]
        if cat:
            parts.append(f"({cat})")
        lines.append("  " + " ".join(parts))
    return "\n".join(lines) if lines else ""


# ---------------------------------------------------------------------------
# Section text builder
# ---------------------------------------------------------------------------


def _build_section_text(meta: PmcArticleMeta, section_key: str) -> str:
    """Concatenate relevant sections for *section_key*.

    For 'methods': combine all sections whose title contains method/material keywords.
    For 'results': combine results + discussion sections.
    """
    if section_key == "methods":
        parts = []
        for name, text in meta.sections.items():
            if any(kw in name.lower() for kw in (
                "method", "material", "experiment", "animal", "sample",
                "design", "diet", "management", "collection", "analysis",
                "chemical", "statistical", "data",
            )):
                parts.append(text)
        if not parts:
            # Fallback: use all sections except intro/discussion/conclusion
            for name, text in meta.sections.items():
                if not any(kw in name.lower() for kw in (
                    "intro", "discussion", "conclusion", "result",
                )):
                    parts.append(text)
        return "\n\n".join(parts) if parts else meta.body_text

    elif section_key == "results":
        parts = []
        for name, text in meta.sections.items():
            if any(kw in name.lower() for kw in (
                "result", "discussion", "conclusion",
            )):
                parts.append(text)
        if not parts:
            # Fallback: use body text
            return meta.body_text
        return "\n\n".join(parts)

    else:
        return meta.full_text


# ---------------------------------------------------------------------------
# Section-based extract
# ---------------------------------------------------------------------------


def extract_sectioned(
    xml_path: str | Path,
    *,
    registry=None,
    model=None,
    max_char_buffer: int | None = None,
    **kwargs,
) -> DocumentExtractionResult:
    """Extract entities from a PMC XML using section-based routing.

    Parameters
    ----------
    xml_path:
        Path to a PMC XML file.
    registry:
        SchemaRegistry instance.
    model:
        Pre-configured LLM.
    max_char_buffer:
        Max chars per chunk.
    **kwargs:
        Passed to :func:`extract`.

    Returns
    -------
    DocumentExtractionResult
        Merged result with deduplicated extractions.
    """
    from src.extraction import extract

    # 1. Parse XML
    meta = parse_article(xml_path)
    doc_id = meta.document_id
    logger.info("Sectioned extraction for %s", doc_id)

    all_extractions: list[Extraction] = []
    warnings: list[str] = meta.parse_warnings.copy()

    # 2. Pre-extraction: Literature from front-matter metadata
    pre_ext = literature_pre_extractor(meta)
    lit_doc = build_document(meta)
    lit_exts = pre_ext(lit_doc)
    # Also populate abstract_conclusion from the abstract text
    if meta.abstract_text:
        for lit in lit_exts:
            if lit.attributes:
                lit.attributes["abstract_conclusion"] = _extract_conclusion(meta.abstract_text)
    all_extractions.extend(lit_exts)
    logger.debug("Pre-extracted %d Literature entities", len(lit_exts))

    # 3. Methods section extraction — entity types from phase config
    methods_text = _build_section_text(meta, "methods")
    if methods_text.strip():
        methods_entities = _get_section_entities(registry, "methods")
        logger.info("Methods section: %d chars → %d entity types",
                     len(methods_text), len(methods_entities))
        methods_doc = Document(
            text=methods_text,
            document_id=f"{doc_id}_methods",
        )
        try:
            methods_result = extract(
                methods_doc,
                registry=registry,
                model=model,
                max_char_buffer=max_char_buffer,
                entity_names=methods_entities,
                **kwargs,
            )
            methods_exts = list(methods_result.extractions)
            all_extractions.extend(methods_exts)
            warnings.extend(methods_result.warnings)
            logger.info("Methods: %d entities extracted", len(methods_exts))
        except Exception as exc:
            logger.warning("Methods extraction failed: %s", exc)
            warnings.append(f"Methods extraction failed: {exc}")

    # 4. Gate check: evaluate whether the article passes the gate before
    #    continuing to the bulk (results) extraction phase.
    do_results = True
    if registry is not None:
        phases = registry.phase_defs()
        gate_phase = next((p for p in phases if p.gate is not None), None)
        if gate_phase and gate_phase.gate is not None:
            do_results = registry.evaluate_gate(
                all_extractions,
                gate_phase.gate.entity,
                gate_phase.gate.condition,
            )
            if not do_results:
                logger.info(
                    "Gate FAILED for %s — %s: %s. on_fail=%s",
                    doc_id,
                    gate_phase.gate.entity,
                    gate_phase.gate.condition,
                    gate_phase.gate.on_fail,
                )
                warnings.append(
                    f"Gate check failed: {gate_phase.gate.entity} "
                    f"condition '{gate_phase.gate.condition}' not met"
                )

    # 5. Collect Methods entities as context for Results extraction.
    #    Context is wrapped in XML-style tags to structurally separate
    #    reference data from the source text, preventing the LLM from
    #    confusing context descriptions with actual article content.
    indicator_context = _build_indicator_context(all_extractions)
    control_context = _build_control_context(all_extractions)
    tissue_context = _build_tissue_context(all_extractions)
    results_context = ""
    if indicator_context or control_context or tissue_context:
        ctx_parts = ["<reference_context>"]
        ctx_parts.append("<!-- The following entities were extracted from the Methods section. -->")
        ctx_parts.append("<!-- Use EXACT abbreviations and names from this list as reference field values. -->")
        ctx_parts.append("<!-- This is REFERENCE DATA, NOT part of the article text to extract from. -->")
        if indicator_context:
            ctx_parts.append(indicator_context)
        if control_context:
            ctx_parts.append(control_context)
        if tissue_context:
            ctx_parts.append(tissue_context)
        ctx_parts.append("</reference_context>")
        results_context = "\n\n".join(ctx_parts)

    # 6. Results section extraction (with Methods context) — gated by phase config
    if do_results:
        results_text = _build_section_text(meta, "results")
        if results_text.strip():
            results_entities = _get_section_entities(registry, "results")
            logger.info("Results section: %d chars → %d entity types (context: %d chars)",
                         len(results_text), len(results_entities), len(results_context))
            results_doc = Document(
                text=results_text,
                document_id=f"{doc_id}_results",
                additional_context=results_context if results_context else None,
            )
            try:
                results_result = extract(
                    results_doc,
                    registry=registry,
                    model=model,
                    max_char_buffer=max_char_buffer,
                    entity_names=results_entities,
                    additional_context=results_context if results_context else None,
                    **kwargs,
                )
                results_exts = list(results_result.extractions)
                all_extractions.extend(results_exts)
                warnings.extend(results_result.warnings)
                logger.info("Results: %d entities extracted", len(results_exts))
            except Exception as exc:
                logger.warning("Results extraction failed: %s", exc)
                warnings.append(f"Results extraction failed: {exc}")
    else:
        logger.info("Results extraction SKIPPED for %s (gate failed)", doc_id)

    # 5. Per-extraction alignment against full text — replaces the old
    #    derive_evidence + ensure_evidence fallback chain.  Aligns every
    #    extraction independently against the complete article text.
    full_text = meta.full_text
    align_and_evidence_batch(all_extractions, full_text, context_chars=400)

    # 5.5 Resolve coreferences (abbreviation→full name, dedup)
    all_extractions = resolve_coreferences(all_extractions, registry)
    logger.info("After coref: %d total entities", len(all_extractions))

    # 6. Verify extraction_text in evidence BEFORE trimming (uses full evidence)
    from src.validation import (
        validate_mutual_exclusivity,
        validate_cross_references,
        clean_synthetic_names,
        filter_incomplete_entities,
        verify_extraction_text_in_evidence,
    )
    all_extractions = validate_mutual_exclusivity(all_extractions, registry)
    all_extractions = clean_synthetic_names(all_extractions)
    xref_warnings = validate_cross_references(all_extractions, registry)
    warnings.extend(xref_warnings)
    all_extractions = filter_incomplete_entities(all_extractions, registry)
    all_extractions, verify_warnings = verify_extraction_text_in_evidence(all_extractions)
    warnings.extend(verify_warnings)

    # 7. Clean + trim evidence text (AFTER verification, so verify uses full text)
    clean_evidence_batch(all_extractions)

    logger.info("After validation: %d total entities (warnings: %d)", len(all_extractions), len(warnings))

    # 8. Post-process
    if registry is not None:
        for ext in all_extractions:
            try:
                registry.post_process(ext)
            except Exception:
                pass

    return DocumentExtractionResult(
        document_id=doc_id,
        metadata={
            "doi": meta.doi,
            "pmid": meta.pmid,
            "title": meta.title,
            "journal": meta.journal,
        },
        extractions=all_extractions,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Strict Coreference Deduplication
# ---------------------------------------------------------------------------


def deduplicate_extractions(
    extractions: list[Extraction],
    registry: Any = None,
) -> list[Extraction]:
    """Deduplicate extractions by entity type using canonical identity keys.

    For each entity type, extractions with the same canonical identity are
    merged into ONE entity.  Merged fields:

    - ``evidence_text``: concatenated with `` <|> `` separator
    - ``source_location``: concatenated with `` <|> `` separator
    - ``attributes``: first non-empty value wins for each key

    After deduplication, each canonical entity appears exactly once.

    Parameters
    ----------
    extractions:
        List of Extraction objects to deduplicate.
    registry:
        SchemaRegistry for primary_text lookup.  When None, uses
        extraction_text as the identity key.

    Returns
    -------
    list[Extraction]
        Deduplicated list.
    """
    if not extractions:
        return []

    # Group by entity type
    by_type: dict[str, list[Extraction]] = {}
    for ext in extractions:
        by_type.setdefault(ext.extraction_class, []).append(ext)

    result: list[Extraction] = []

    for etype, exts in by_type.items():
        # Determine identity key field from registry
        identity_field: str | None = None
        if registry is not None:
            try:
                ed = registry.entity_def(etype)
                identity_field = ed.primary_text
            except (KeyError, AttributeError):
                pass

        # Group by canonical identity
        groups: dict[str, list[Extraction]] = {}
        for ext in exts:
            key = _canonical_identity(ext, identity_field)
            groups.setdefault(key, []).append(ext)

        # Merge each group into one entity
        for key, group in groups.items():
            merged = _merge_group(group)
            result.append(merged)

    return result


def _canonical_identity(ext: Extraction, identity_field: str | None) -> str:
    """Compute a canonical identity key for *ext*.

    Uses *identity_field* from attributes if available, otherwise falls back
    to extraction_text.  Normalises via lowercase + whitespace collapse.
    """
    import re

    if identity_field and ext.attributes:
        val = ext.attributes.get(identity_field)
        if val and isinstance(val, str) and val.strip():
            key = val
        else:
            key = ext.extraction_text
    else:
        key = ext.extraction_text

    # Normalize: lowercase, replace hyphens/underscores, collapse whitespace
    # (consistent with _normalize_name in src/graph.py)
    norm = key.strip().lower()
    norm = re.sub(r"[-_]", " ", norm)
    norm = re.sub(r"\s+", " ", norm)
    return norm


def _merge_group(group: list[Extraction]) -> Extraction:
    """Merge a group of extractions into one.

    - First extraction's class, text, and attributes form the base.
    - evidence_text and source_location are concatenated with `` <|> ``.
    - Attributes: first non-empty, non-None value wins.
    """
    if len(group) == 1:
        return group[0]

    base = group[0]

    # Merge evidence_text
    evidence_parts: list[str] = []
    for ext in group:
        ev = getattr(ext, "evidence_text", "")
        if ev and ev.strip():
            evidence_parts.append(ev.strip())
    base.evidence_text = " <|> ".join(evidence_parts)

    # Merge source_location
    location_parts: list[str] = []
    for ext in group:
        sl = getattr(ext, "source_location", "")
        if sl and sl.strip():
            location_parts.append(sl.strip())
    base.source_location = " <|> ".join(location_parts)

    # Merge attributes: first non-empty wins
    if base.attributes is None:
        base.attributes = {}
    merged_attrs = dict(base.attributes)
    for ext in group[1:]:
        if ext.attributes:
            for k, v in ext.attributes.items():
                if k not in merged_attrs or not merged_attrs[k]:
                    if v is not None and v != "":
                        merged_attrs[k] = v

    base.attributes = merged_attrs
    return base


# ---------------------------------------------------------------------------
# Convenience: full pipeline with section-based extraction + review export
# ---------------------------------------------------------------------------


def run_sectioned_pipeline(
    xml_path: str | Path,
    output_dir: str | Path = "output",
    *,
    registry=None,
    model=None,
    global_types: frozenset[str] | None = None,
    **kwargs,
) -> dict:
    """Run the full section-based pipeline and export all outputs.

    Returns a dict with keys: ``result``, ``graph``, ``output_dir``.
    """
    from src.graph import build_graph, export_neo4j_csv
    from src.schema_registry import SchemaRegistry

    if registry is None:
        registry = SchemaRegistry()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Section-based extraction
    result = extract_sectioned(xml_path, registry=registry, model=model, **kwargs)

    # 2. Export review CSVs
    review_dir = out / "review"
    export_entity_review_csv([result], review_dir, registry)

    # 3. Build graph
    if global_types is None:
        global_types = frozenset({
            "Alternative", "Composite_Product", "Tissue_Site",
            "Indicator", "Method",
        })
    graph = build_graph([result], registry, global_types=global_types)

    # 4. Export Neo4j CSV
    graph_dir = out / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    export_neo4j_csv(graph, graph_dir)

    return {
        "result": result,
        "graph": graph,
        "output_dir": out,
    }
