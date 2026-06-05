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

    Matches phases by their ``sections`` key against *section_name*.
    When multiple phases target the same section (e.g., gate and bulk
    both use ``sections: [methods]``), their entity types are **not**
    combined — each phase is extracted separately to keep prompts
    focused and token-efficient.

    Falls back to the hardcoded lists if no registry or phase config is available.
    """
    if registry is not None:
        try:
            routing = registry.build_section_routing()
            combined: list[str] = []
            for entry in routing:
                if entry["section"] == section_name:
                    combined.extend(entry["entity_names"])
            if combined:
                return list(dict.fromkeys(combined))  # dedup, preserve order
        except Exception:
            pass
    # Fallback
    return _METHODS_ENTITIES if section_name == "methods" else _RESULTS_ENTITIES


def _get_phase_entities_for_section(registry, section_name: str) -> list[dict]:
    """Return a list of {phase_name, entity_names} for a given section.

    Each entry represents one independent extraction pass for the section.
    This enables multi-pass extraction where each pass has a focused,
    smaller prompt — improving accuracy and reducing token waste.

    Gate phase (gate_alternative) extracts ONLY Alternative — fast check
    before committing to full extraction.
    """
    if registry is None:
        if section_name == "methods":
            return [
                {"phase": "gate_alternative", "entity_names": ["Alternative", "Swine", "Swine_Model"]},
                {"phase": "methods_core", "entity_names": [
                    "Swine", "Swine_Model", "Intervention",
                    "Control_Group", "Tissue_Site",
                ]},
                {"phase": "bulk_methods", "entity_names": [
                    "Experiment", "Indicator", "Method", "Composite_Product",
                ]},
            ]
        else:
            return [{"phase": "results", "entity_names": ["Result", "Tissue_Site", "Method", "Indicator"]}]

    try:
        routing = registry.build_section_routing()
        phases_for_section: list[dict] = []
        for entry in routing:
            if entry["section"] == section_name:
                phases_for_section.append({
                    "phase": entry["phase"],
                    "entity_names": entry["entity_names"],
                })
        if phases_for_section:
            return phases_for_section
    except Exception:
        pass

    # Fallback
    if section_name == "methods":
        return [
            {"phase": "gate_alternative", "entity_names": ["Alternative", "Swine", "Swine_Model"]},
            {"phase": "methods_core", "entity_names": [
                "Intervention", "Control_Group", "Tissue_Site", "Alternative",
            ]},
            {"phase": "bulk_methods", "entity_names": [
                "Experiment", "Indicator", "Method", "Composite_Product",
            ]},
        ]
    else:
        return [{"phase": "results", "entity_names": ["Result", "Tissue_Site", "Method", "Indicator"]}]


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


def _build_gate_text(meta: PmcArticleMeta) -> str:
    """Build focused text for gate relevance check.

    Uses the abstract only — the paper's core claim — which is the
    most concise signal of whether the paper is about pig antibiotic
    alternatives.  The LLM reads this and decides: relevant or not.
    """
    if meta.abstract_text:
        return meta.abstract_text
    # Fallback: first 2000 chars of body
    return meta.body_text[:2000] if meta.body_text else ""


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
    """Extract entities from a PMC XML using section-based multi-pass routing.

    Each phase in the extraction config is processed independently from its
    designated section, keeping prompts focused and token-efficient:

    1. Methods → Gate phase: [Alternative, Swine, Swine_Model, Intervention,
       Control_Group, Tissue_Site]
    2. Gate check → skip remaining phases on failure
    3. Methods → Bulk_methods phase: [Experiment, Indicator, Method, Composite_Product]
    4. Results → Bulk_results phase: [Result] (with context from 1 and 3)

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

    # 3. Methods section — multi-pass extraction.
    #    Gate phase runs first (must validate before spending tokens).
    #    Then independent phases (methods_core, bulk_methods) run in PARALLEL
    #    since they extract from the same text but different entity types.
    methods_text = _build_section_text(meta, "methods")
    methods_phases = _get_phase_entities_for_section(registry, "methods")
    gate_passed = True

    if methods_text.strip():
        # Separate gate phase from independent methods phases
        gate_phase_info = None
        independent_phases: list[dict] = []
        for pi in methods_phases:
            is_gate = False
            if registry is not None:
                for p in registry.phase_defs():
                    if p.name == pi["phase"] and p.gate is not None:
                        is_gate = True
                        break
            if is_gate:
                gate_phase_info = pi
            else:
                independent_phases.append(pi)

        # --- Step 3a: Gate phase (sequential — must complete first) ---
        if gate_phase_info:
            phase_name = gate_phase_info["phase"]
            phase_entities = gate_phase_info["entity_names"]
            gate_text = _build_gate_text(meta)
            logger.info("Gate phase '%s': %d chars → %d entity types",
                         phase_name, len(gate_text), len(phase_entities))
            gate_doc = Document(text=gate_text, document_id=f"{doc_id}_gate")
            try:
                gate_result = extract(
                    gate_doc, registry=registry, model=model,
                    max_char_buffer=max_char_buffer, entity_names=phase_entities,
                    **kwargs,
                )
                gate_exts = list(gate_result.extractions)
                for ge in gate_exts:
                    if ge.attributes is None:
                        ge.attributes = {}
                    ge.attributes["_gate_preserved"] = True
                    if not ge.attributes.get("original_text"):
                        ge.attributes["original_text"] = ge.attributes.get(
                            "standard_name", ge.extraction_text
                        )
                all_extractions.extend(gate_exts)
                warnings.extend(gate_result.warnings)
                logger.info("  Gate phase: %d entities extracted", len(gate_exts))

                # Gate check 1: must have Alternative or Composite_Product
                alt_count = sum(1 for e in gate_exts
                                if e.extraction_class == "Alternative")
                swine_count = sum(1 for e in gate_exts
                                  if e.extraction_class in ("Swine", "Swine_Model"))

                if alt_count == 0 and swine_count == 0:
                    gate_passed = False
                    logger.info("Gate FAILED for %s: 0 entities found", doc_id)
                    warnings.append("Gate check failed: no relevant entities found")
                elif alt_count == 0:
                    gate_passed = False
                    logger.info("Gate FAILED for %s: 0 Alternative, has Swine but no antibiotic alternative", doc_id)
                    warnings.append("Gate check failed: has swine info but no antibiotic alternative substances")
                elif swine_count == 0:
                    gate_passed = False
                    logger.info("Gate FAILED for %s: 0 Swine/Swine_Model, has Alternative but no pig context", doc_id)
                    warnings.append("Gate check failed: has Alternative but no pig/swine information")
                elif registry is not None:
                    gate_phase_def = next(
                        (p for p in registry.phase_defs() if p.name == phase_name), None
                    )
                    if gate_phase_def and gate_phase_def.gate is not None:
                        gate_passed = registry.evaluate_gate(
                            all_extractions, gate_phase_def.gate.entity,
                            gate_phase_def.gate.condition,
                        )
                        if not gate_passed:
                            logger.info("Gate FAILED for %s — no valid classification", doc_id)
                            warnings.append("Gate check failed: no Alternative with valid classification")
            except Exception as exc:
                logger.warning("Gate phase extraction failed: %s", exc)
                warnings.append(f"Gate phase extraction failed: {exc}")

        # --- Step 3b: Sequential methods phases ---
        if gate_passed and independent_phases:
            for pi in independent_phases:
                pname = pi["phase"]
                pentities = pi["entity_names"]
                pdoc = Document(
                    text=methods_text,
                    document_id=f"{doc_id}_methods_{pname}",
                )
                try:
                    presult = extract(
                        pdoc, registry=registry, model=model,
                        max_char_buffer=max_char_buffer, entity_names=pentities,
                        **kwargs,
                    )
                    all_extractions.extend(presult.extractions)
                    warnings.extend(presult.warnings)
                    logger.info("  %s: %d entities", pname, len(presult.extractions))
                except Exception as exc:
                    logger.warning("%s extraction failed: %s", pname, exc)
                    warnings.append(f"{pname} extraction failed: {exc}")

    # 4. Build combined context from all methods extractions for Results phase.
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

    # 5. Results section extraction (with Methods context) — only if gate passed
    if gate_passed:
        results_text = _build_section_text(meta, "results")
        if results_text.strip():
            results_phases = _get_phase_entities_for_section(registry, "results")
            for phase_info in results_phases:
                phase_name = phase_info["phase"]
                phase_entities = phase_info["entity_names"]
                logger.info("Results phase '%s': %d chars → %d entity types (context: %d chars)",
                             phase_name, len(results_text), len(phase_entities), len(results_context))
                results_doc = Document(
                    text=results_text,
                    document_id=f"{doc_id}_results",
                )
                try:
                    results_result = extract(
                        results_doc,
                        registry=registry,
                        model=model,
                        max_char_buffer=max_char_buffer,
                        entity_names=phase_entities,
                        additional_context=results_context if results_context else None,
                        **kwargs,
                    )
                    results_exts = list(results_result.extractions)
                    all_extractions.extend(results_exts)
                    warnings.extend(results_result.warnings)
                    logger.info("  %s: %d entities extracted", phase_name, len(results_exts))
                except Exception as exc:
                    logger.warning("Results extraction failed: %s", exc)
                    warnings.append(f"Results extraction failed: {exc}")
    else:
        logger.info("Results extraction SKIPPED for %s (gate failed)", doc_id)

    # 6. Per-extraction alignment against full text — replaces the old
    #    derive_evidence + ensure_evidence fallback chain.  Aligns every
    #    extraction independently against the complete article text.
    full_text = meta.full_text
    align_and_evidence_batch(all_extractions, full_text, context_chars=400)

    # 6.5 Decompose Composite_Product components into Alternative entities.
    # 6.5.3 Decompose Composite_Product components into Alternative entities
    _decompose_composites_to_alternatives(all_extractions)

    # 6.5.4 Remove orphan Indicators and Methods — only keep those
    #       that are referenced by at least one Result entity.
    _remove_orphan_indicators_and_methods(all_extractions)

    # 6.5.4.5 Remove orphan entities across all types — any entity
    #         not referenced by another entity is removed (e.g. an
    #         Alternative not linked to any Intervention or Composite).
    _remove_general_orphans(all_extractions, registry)

    # 6.5.5 Auto-create missing Indicator entities from Result references.
    #       This is the ONLY auto-create step — Indicators like microbial
    #       taxa are naturally enumerated in Results, not Methods.
    #       Tissue_Site and Method are NOT auto-created — those entities
    #       MUST be properly extracted by the LLM from the Methods text.
    #       Auto-creating them from Result references propagates LLM errors
    #       (e.g., "LB medium", "RAW264.7 macrophages" are NOT tissue sites).
    _auto_create_missing_indicators(all_extractions)

    # 6.6 Resolve coreferences (abbreviation→full name, dedup)
    all_extractions = resolve_coreferences(all_extractions, registry)
    logger.info("After coref: %d total entities", len(all_extractions))

    # 7. Verify extraction_text in evidence BEFORE trimming (uses full evidence)
    from src.validation import (
        validate_mutual_exclusivity,
        validate_cross_references,
        clean_synthetic_names,
        filter_incomplete_entities,
        verify_extraction_text_in_evidence,
        validate_composite_product_components,
        filter_empty_evidence,
    )
    all_extractions = validate_mutual_exclusivity(all_extractions, registry)
    all_extractions = clean_synthetic_names(all_extractions)
    xref_warnings = validate_cross_references(all_extractions, registry)
    warnings.extend(xref_warnings)
    all_extractions = filter_incomplete_entities(all_extractions, registry)
    # Remove false Composite_Product entities (single substances misclassified)
    all_extractions, cp_warnings = validate_composite_product_components(all_extractions)
    warnings.extend(cp_warnings)
    # Drop entities with empty evidence_text
    all_extractions, empty_ev_warnings = filter_empty_evidence(all_extractions)
    warnings.extend(empty_ev_warnings)
    # Verify extraction_text in evidence — pass full_text as fallback
    all_extractions, verify_warnings = verify_extraction_text_in_evidence(
        all_extractions, full_text=full_text,
    )
    warnings.extend(verify_warnings)

    # 8. Clean + trim evidence text (AFTER verification, so verify uses full text)
    clean_evidence_batch(all_extractions)

    logger.info("After validation: %d total entities (warnings: %d)", len(all_extractions), len(warnings))

    # 9. Post-process
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
# General orphan removal
# ---------------------------------------------------------------------------


def _remove_general_orphans(
    extractions: list[Extraction],
    registry: Any = None,
) -> None:
    """Remove entities not referenced by any other entity.

    Scans all cross-entity references (reference fields + inline relations)
    and drops entities that have no incoming references.  This keeps the
    graph clean — no disconnected nodes.
    """
    import logging
    _log = logging.getLogger(__name__)

    if registry is None or not extractions:
        return

    # Collect all referenced entity identities by type
    # {(target_type, normalized_name)} — entities that ARE referenced
    referenced: set[tuple[str, str]] = set()

    for ext in extractions:
        etype = ext.extraction_class
        attrs = ext.attributes or {}
        try:
            ed = registry.entity_def(etype)
        except (KeyError, AttributeError):
            continue

        # From reference fields
        for ref in ed.references:
            val = attrs.get(ref.name)
            if val and isinstance(val, str) and val.strip():
                referenced.add((ref.target_entity, val.strip().lower()))

        # From inline relations
        for ir in ed.inline_relations:
            vals = attrs.get(ir.via_field)
            if not vals:
                continue
            if not isinstance(vals, list):
                vals = [vals]
            for v in vals:
                if isinstance(v, str) and v.strip():
                    for target_type in ir.target:
                        referenced.add((target_type, v.strip().lower()))
                elif isinstance(v, dict):
                    for target_type in ir.target:
                        name = v.get("standard_name", str(v))
                        if name:
                            referenced.add((target_type, str(name).strip().lower()))

    removed = 0
    keep = []
    for ext in extractions:
        etype = ext.extraction_class
        eattrs = ext.attributes or {}

        # Only check specific types for orphan status
        if etype == "Alternative":
            name = ext.extraction_text.strip().lower()
            std_name = (eattrs.get("standard_name", "") or "").strip().lower()
            is_referenced = (
                (etype, name) in referenced
                or (std_name and (etype, std_name) in referenced)
            )
            if not is_referenced:
                removed += 1
                _log.info("Removed orphan Alternative '%s'", ext.extraction_text)
                continue
        keep.append(ext)

    if removed:
        _log.info("Removed %d orphan entities across all types", removed)
        extractions[:] = keep


# ---------------------------------------------------------------------------
# Composite → Alternative decomposition
# ---------------------------------------------------------------------------


def _decompose_composites_to_alternatives(
    extractions: list[Extraction],
) -> None:
    """Decompose Composite_Product components into Alternative entities in-place.

    For each Composite_Product that has a ``components`` list, create a
    corresponding ``Alternative`` entity for every component that doesn't
    already exist.  This ensures every gated paper contributes individual
    Alternative entities to the knowledge graph.
    """
    import logging
    _log = logging.getLogger(__name__)

    # Collect existing Alternative names (by standard_name and extraction_text)
    existing_alts: set[str] = set()
    for ext in extractions:
        if ext.extraction_class != "Alternative":
            continue
        existing_alts.add(ext.extraction_text.strip().lower())
        name = (ext.attributes or {}).get("standard_name", "")
        if name:
            existing_alts.add(name.strip().lower())

    new_alts: list[Extraction] = []
    for ext in extractions:
        if ext.extraction_class != "Composite_Product":
            continue
        attrs = ext.attributes or {}
        components = attrs.get("components")
        if not components or not isinstance(components, list):
            continue

        for comp in components:
            if not isinstance(comp, str) or not comp.strip():
                continue
            name = comp.strip()
            if name.lower() in existing_alts:
                continue
            # Only create if the name looks like a real substance
            if len(name) < 3 or name.lower() in ("none", "not reported", "n/a", "na"):
                continue

            alt = Extraction(
                extraction_class="Alternative",
                extraction_text=name,
                attributes={
                    "standard_name": name,
                    "original_text": name,
                    "_gate_preserved": True,
                    "_decomposed_from": attrs.get("product_name", ext.extraction_text),
                },
            )
            alt.evidence_text = getattr(ext, "evidence_text", "") or ""
            alt.source_location = getattr(ext, "source_location", "") or ""
            new_alts.append(alt)
            existing_alts.add(name.lower())

    if new_alts:
        _log.info(
            "Decomposed %d Composite_Product component(s) into Alternative entities",
            len(new_alts),
        )
        extractions.extend(new_alts)


# ---------------------------------------------------------------------------
# Remove orphan Indicators and Methods
# ---------------------------------------------------------------------------


def _remove_orphan_indicators_and_methods(
    extractions: list[Extraction],
) -> None:
    """Remove Indicator and Method entities not referenced by any Result.

    Indicator: removed unless referenced by a Result via corresponds_to.
    Method: removed unless referenced by an Indicator via uses_method
            AND that Indicator is referenced by a Result.

    This keeps the graph clean — no orphan measurement definitions.
    """
    import logging
    _log = logging.getLogger(__name__)

    # Collect all indicator_abbreviation values from Results
    referenced_indicators: set[str] = set()
    for ext in extractions:
        if ext.extraction_class != "Result":
            continue
        ref = (ext.attributes or {}).get("indicator_abbreviation", "")
        if ref and isinstance(ref, str):
            referenced_indicators.add(ref.strip().lower())

    # Build a set of Indicator names that are referenced by Results
    kept_indicators: set[int] = set()
    for i, ext in enumerate(extractions):
        if ext.extraction_class != "Indicator":
            continue
        # Check if this Indicator is referenced by any Result
        name = ext.extraction_text.strip().lower()
        abbr = ((ext.attributes or {}).get("abbreviation", "") or "").strip().lower()
        std = ((ext.attributes or {}).get("standard_name", "") or "").strip().lower()
        if (name in referenced_indicators or
            abbr in referenced_indicators or
            std in referenced_indicators):
            kept_indicators.add(i)

    # Collect Methods referenced by kept Indicators
    referenced_methods: set[str] = set()
    for i in kept_indicators:
        ext = extractions[i]
        ref = (ext.attributes or {}).get("measurement_method", "")
        if ref and isinstance(ref, str):
            referenced_methods.add(ref.strip().lower())

    # Remove unreferenced Indicators and Methods
    removed_inds = 0
    removed_meths = 0
    keep = []
    for i, ext in enumerate(extractions):
        if ext.extraction_class == "Indicator":
            if i not in kept_indicators:
                removed_inds += 1
                continue
        elif ext.extraction_class == "Method":
            name = ext.extraction_text.strip().lower()
            mname = ((ext.attributes or {}).get("method_name", "") or "").strip().lower()
            if name not in referenced_methods and mname not in referenced_methods:
                removed_meths += 1
                continue
        keep.append(ext)

    if removed_inds or removed_meths:
        _log.info(
            "Removed orphans: %d Indicators, %d Methods (not referenced by any Result)",
            removed_inds, removed_meths,
        )
        extractions[:] = keep


# ---------------------------------------------------------------------------
# Auto-create missing Indicators from Result references
# ---------------------------------------------------------------------------


def _auto_create_missing_indicators(
    extractions: list[Extraction],
) -> None:
    """Create Indicator entities for Result references with no matching Indicator.

    Some indicators (especially microbial taxa like ``g_Alloprevotella``,
    ``c_Bacteroidia``) are only enumerated in the Results section, not in
    Methods.  This function ensures every Result's ``indicator_abbreviation``
    has a corresponding Indicator entity.
    """
    import logging
    _log = logging.getLogger(__name__)

    # Collect existing Indicator identities
    existing_inds: set[str] = set()
    for ext in extractions:
        if ext.extraction_class != "Indicator":
            continue
        et = ext.extraction_text.strip().lower()
        if et:
            existing_inds.add(et)
        abbr = (ext.attributes or {}).get("abbreviation", "")
        if abbr:
            existing_inds.add(abbr.strip().lower())
        std = (ext.attributes or {}).get("standard_name", "")
        if std:
            existing_inds.add(std.strip().lower())

    # Collect all unique indicator_abbreviation values from Results
    needed: dict[str, list[Extraction]] = {}
    for ext in extractions:
        if ext.extraction_class != "Result":
            continue
        ref = (ext.attributes or {}).get("indicator_abbreviation", "")
        if not ref or not isinstance(ref, str):
            continue
        ref_norm = ref.strip().lower()
        if ref_norm in existing_inds:
            continue
        needed.setdefault(ref_norm, []).append(ext)

    if not needed:
        return

    # Determine if any Tissue_Site references exist for auto-linking
    tissue_sites: dict[str, str] = {}
    for ext in extractions:
        if ext.extraction_class != "Tissue_Site":
            continue
        name = (ext.attributes or {}).get("site_name", "") or ext.extraction_text
        if name:
            tissue_sites[name.strip().lower()] = name

    new_inds: list[Extraction] = []
    for ref_norm, ref_results in needed.items():
        # Use the original casing from the first Result
        ref_original = ref_results[0].attributes.get("indicator_abbreviation", ref_norm)

        # Determine indicator category from the Result context
        category = "microbiome"  # default for taxa-like names
        # Infer from reference name patterns
        if any(ref_norm.startswith(p) for p in ("g_", "f_", "o_", "c_", "p_", "s_")):
            category = "microbiome"
        elif any(kw in ref_norm for kw in ("shannon", "chao1", "simpson", "alpha", "beta", "pcoa")):
            category = "microbiome"
        elif any(kw in ref_norm for kw in ("adg", "adfi", "fbw", "weight", "gain", "growth")):
            category = "macro_phenotype"
        elif any(kw in ref_norm for kw in ("acetate", "butyrate", "propionate", "scfa", " acid")):
            category = "metabolome"
        elif any(kw in ref_norm for kw in ("il-", "tnf", "zo-1", "claudin", "expres", "mrna", "gene")):
            category = "molecular"

        # Inherit evidence and tissue_site from the first referencing Result
        ev = getattr(ref_results[0], "evidence_text", "") or ""
        sl = getattr(ref_results[0], "source_location", "") or ""
        tissue_ref = (ref_results[0].attributes or {}).get("tissue_site", "")

        ind = Extraction(
            extraction_class="Indicator",
            extraction_text=ref_original,
            attributes={
                "standard_name": ref_original,
                "abbreviation": ref_original,
                "indicator_category": category,
                "measured_in": tissue_ref if tissue_ref else "",
                "_auto_created": True,
            },
        )
        ind.evidence_text = ev
        ind.source_location = sl
        new_inds.append(ind)

    if new_inds:
        _log.info(
            "Auto-created %d missing Indicator(s) from Result references",
            len(new_inds),
        )
        extractions.extend(new_inds)


# ---------------------------------------------------------------------------
# Auto-create missing Tissue_Site entities
# ---------------------------------------------------------------------------


def _auto_create_missing_tissue_sites(
    extractions: list[Extraction],
) -> None:
    """Create Tissue_Site entities for references that don't match any existing site.

    Both Indicator (measured_in) and Result (tissue_site / occurs_in) can
    reference tissue sites that weren't extracted from Methods.
    """
    import logging
    _log = logging.getLogger(__name__)

    # Collect existing Tissue_Site identities
    existing: set[str] = set()
    for ext in extractions:
        if ext.extraction_class != "Tissue_Site":
            continue
        name = (ext.attributes or {}).get("site_name", "") or ext.extraction_text
        if name:
            existing.add(name.strip().lower())
        existing.add(ext.extraction_text.strip().lower())

    # Collect all unique tissue references from Indicators and Results
    needed: dict[str, Extraction] = {}
    for ext in extractions:
        ref = ""
        if ext.extraction_class == "Indicator":
            ref = (ext.attributes or {}).get("measured_in", "")
        elif ext.extraction_class == "Result":
            ref = (ext.attributes or {}).get("tissue_site", "")
        if not ref or not isinstance(ref, str) or not ref.strip():
            continue
        ref_norm = ref.strip().lower()
        if ref_norm in existing:
            continue
        if ref_norm not in needed:
            needed[ref_norm] = ext

    if not needed:
        return

    # Infer site_category from name patterns
    def _infer_category(name: str) -> str:
        n = name.lower()
        if any(kw in n for kw in ("content", "digesta", "chyme", "feces", "manure")):
            return "content"
        if any(kw in n for kw in ("mucosa", "epithelium", "mucus")):
            return "mucosa"
        if any(kw in n for kw in ("serum", "plasma", "blood")):
            return "serum"
        if any(kw in n for kw in ("tissue", "muscle", "liver", "spleen", "kidney", "heart", "lung", "intestine", "colon", "cecum", "jejunum", "ileum", "duodenum")):
            return "tissue"
        if any(kw in n for kw in ("feces", "fecal", "stool", "manure")):
            return "feces"
        if any(kw in n for kw in ("whole body", "whole", "body")):
            return "whole_organism"
        return "tissue"

    new_sites: list[Extraction] = []
    for ref_norm, ref_ext in needed.items():
        ref_original = ref_ext.attributes.get(
            "measured_in" if ref_ext.extraction_class == "Indicator" else "tissue_site",
            ref_norm
        )
        category = _infer_category(ref_original)

        site = Extraction(
            extraction_class="Tissue_Site",
            extraction_text=ref_original,
            attributes={
                "site_name": ref_original,
                "site_category": category,
                "_auto_created": True,
            },
        )
        site.evidence_text = getattr(ref_ext, "evidence_text", "") or ""
        site.source_location = getattr(ref_ext, "source_location", "") or ""
        new_sites.append(site)

    if new_sites:
        _log.info(
            "Auto-created %d missing Tissue_Site(s) from references",
            len(new_sites),
        )
        extractions.extend(new_sites)


# ---------------------------------------------------------------------------
# Auto-create missing Method entities
# ---------------------------------------------------------------------------


def _auto_create_missing_methods(
    extractions: list[Extraction],
) -> None:
    """Create Method entities for Indicator measurement_method references
    that don't match any existing Method.

    Some indicators reference methods (like "weighing", "feed recording")
    that were blacklisted or not extracted from Methods text.  Auto-create
    minimal Method entities so that uses_method edges resolve correctly.
    """
    import logging
    _log = logging.getLogger(__name__)

    # Collect existing Method identities
    existing: set[str] = set()
    for ext in extractions:
        if ext.extraction_class != "Method":
            continue
        name = (ext.attributes or {}).get("method_name", "") or ext.extraction_text
        if name:
            existing.add(name.strip().lower())
        existing.add(ext.extraction_text.strip().lower())

    # Collect unique method references from Indicators
    needed: dict[str, Extraction] = {}
    for ext in extractions:
        if ext.extraction_class != "Indicator":
            continue
        ref = (ext.attributes or {}).get("measurement_method", "")
        if not ref or not isinstance(ref, str) or not ref.strip():
            continue
        ref_norm = ref.strip().lower()
        if ref_norm in existing:
            continue
        if ref_norm not in needed:
            needed[ref_norm] = ext

    if not needed:
        return

    new_methods: list[Extraction] = []
    for ref_norm, ref_ext in needed.items():
        ref_original = ref_ext.attributes.get("measurement_method", ref_norm)

        m = Extraction(
            extraction_class="Method",
            extraction_text=ref_original,
            attributes={
                "method_name": ref_original,
                "_auto_created": True,
            },
        )
        m.evidence_text = getattr(ref_ext, "evidence_text", "") or ""
        m.source_location = getattr(ref_ext, "source_location", "") or ""
        new_methods.append(m)

    if new_methods:
        _log.info(
            "Auto-created %d missing Method(s) from Indicator references",
            len(new_methods),
        )
        extractions.extend(new_methods)


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
