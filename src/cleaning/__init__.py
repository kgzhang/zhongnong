"""Data cleaning pipeline — post-LLM extraction normalization and validation.

Operates on Extraction objects after the LLM extraction phase and before CSV
export.  All entities and relationships are cleaned, normalized, validated,
deduplicated, and optionally repaired.

**Scoping model (driven by entity ``dedup_mode`` in schema YAML):**

- **Global entities** (``dedup_mode: fuzzy|exact``): Alternative, Composite_Product,
  Indicator, Method, Swine, Swine_Model, Tissue_Site.  These receive **canonical
  name normalization** so the same entity extracted from different papers maps to
  the same graph node.  Within-article dedup only; cross-article merging is
  handled by the graph builder via ``registry.get_global_types()``.
- **Article-scoped entities** (``dedup_mode: article``): Control_Group, Experiment,
  Intervention, Literature, Result.  Normalized and deduplicated **within a single
  article only** — each article keeps its own control groups, results, etc.

Usage::

    from src.cleaning import clean_extractions, clean_from_results

    # Per-article cleaning
    cleaned = clean_extractions(extractions, registry=registry)

    # Batch: per-article then cross-article consistency check
    all_results = clean_from_results(all_results, registry=registry)

The pipeline is organized in 4 phases:
1. **Normalize** — text standardization (case, whitespace, Unicode, parens)
2. **Validate** — cross-table reference integrity, value domain constraints
3. **Dedup** — merge duplicates within same source_doc (same dedup key regardless of scope)
4. **Repair** — auto-fix known error patterns (p_value, direction, etc.)
"""

from __future__ import annotations

import logging
from typing import Any

from src.data import Extraction

from .normalizer import normalize_extractions
from .validator import validate_extractions
from .dedup import deduplicate_extractions
from .repair import repair_extractions
from .report import generate_cleaning_report, CleaningReport

logger = logging.getLogger(__name__)


def clean_extractions(
    extractions: list[Extraction],
    *,
    registry: Any = None,
    generate_report: bool = True,
    repair: bool = True,
    strict: bool = False,
) -> list[Extraction]:
    """Run the full cleaning pipeline on a list of Extraction objects.

    All entities are normalized and validated.  Entities are deduplicated within
    the same source_doc — article-scoped entities stay per-article, and global
    entities also stay per-article (cross-article merging is handled later by the
    graph builder).  Names are canonically normalized so that the graph-level
    global dedup sees matching keys.

    Parameters
    ----------
    extractions:
        The extractions to clean (mutated in-place).
    registry:
        SchemaRegistry instance for cross-entity validation and dedup scoping.
    generate_report:
        If True, produce a CleaningReport summarizing changes.
    repair:
        If True, run Phase 4 auto-repair.
    strict:
        If True, drop entities that fail validation rather than just warning.

    Returns
    -------
    list[Extraction]
        The cleaned (potentially expanded) extraction list.
    """
    if not extractions:
        return []

    n_input = len(extractions)
    report = CleaningReport()

    # ------------------------------------------------------------------
    # Phase 1: Normalization — text standardization
    #   All entities get canonical case, whitespace, Unicode normalization.
    #   Global-type entities use the full domain taxonomy so names match
    #   across articles (required for graph-level global dedup).
    # ------------------------------------------------------------------
    logger.info("Phase 1: Normalization (%d entities)", len(extractions))
    extractions, norm_changes = normalize_extractions(extractions, registry=registry)
    report.add_phase("normalize", changes=norm_changes)
    logger.info("  → %d normalization changes applied", norm_changes)

    # ------------------------------------------------------------------
    # Phase 2: Validation — reference integrity & domain constraints
    # ------------------------------------------------------------------
    logger.info("Phase 2: Validation (%d entities)", len(extractions))
    extractions, validation_issues = validate_extractions(
        extractions, registry=registry, strict=strict
    )
    report.add_phase("validate", warnings=validation_issues)
    logger.info("  → %d validation issues found (%d dropped)",
                 len(validation_issues), n_input - len(extractions))

    # ------------------------------------------------------------------
    # Phase 3: Deduplication — within source_doc only
    #   All entity types use article-scoped dedup keys.
    #   Cross-article global-dedup is the graph builder's job.
    # ------------------------------------------------------------------
    logger.info("Phase 3: Deduplication (%d entities)", len(extractions))
    extractions, dup_merged = deduplicate_extractions(extractions, registry=registry)
    report.add_phase("dedup", changes=dup_merged)
    logger.info("  → %d duplicates merged", dup_merged)

    # ------------------------------------------------------------------
    # Phase 4: Repair — auto-fix mapped values
    # ------------------------------------------------------------------
    if repair:
        logger.info("Phase 4: Repair (%d entities)", len(extractions))
        extractions, repair_changes = repair_extractions(extractions)
        report.add_phase("repair", changes=repair_changes)
        logger.info("  → %d repairs applied", repair_changes)

    # ------------------------------------------------------------------
    # Generate report
    # ------------------------------------------------------------------
    if generate_report:
        report.summary = {
            "entities_before": n_input,
            "entities_after": len(extractions),
            "phases": {
                "normalize": report.phase_changes.get("normalize", 0),
                "validate_warnings": len(report.phase_warnings.get("validate", [])),
                "dedup_merged": report.phase_changes.get("dedup", 0),
                "repair": report.phase_changes.get("repair", 0),
            },
        }
        report.log_summary()

    return extractions


def clean_from_results(
    all_results: list[Any],
    *,
    registry: Any = None,
    generate_report: bool = True,
) -> list[Any]:
    """Clean extractions from a list of DocumentExtractionResult objects.

    Two-pass strategy:
    1. **Per-article pass**: For each result, run the full cleaning pipeline
       (normalize, validate, dedup within article, repair).  Global entities
       get canonical names so they match across articles.
    2. **Cross-article consistency check**: Verify that global entities share
       the same canonical form across articles.  Log discrepancies.

    The actual merging of global entities into single graph nodes is done by
    the graph builder (``src/graph.py``) via ``registry.get_global_types()``.
    This cleaning pipeline ensures names are consistent so that merge succeeds.

    Parameters
    ----------
    all_results:
        List of result objects, each with ``extractions`` and ``document_id``.
    registry:
        SchemaRegistry instance (required).
    generate_report:
        If True, produce a CleaningReport.

    Returns
    -------
    list
        The same result objects with cleaned extractions.
    """
    if not all_results:
        return all_results

    # ------------------------------------------------------------------
    # Pass 1: Per-article cleaning
    # ------------------------------------------------------------------
    logger.info("Pass 1: Per-article cleaning (%d articles)", len(all_results))
    for result in all_results:
        original = list(result.extractions)
        cleaned = clean_extractions(
            original,
            registry=registry,
            generate_report=False,
            repair=True,
        )
        result.extractions = cleaned

    # ------------------------------------------------------------------
    # Pass 2: Cross-article name consistency check
    #   Verify same-base-name global entities use consistent canonical forms.
    #   This is a safety net — the normalizer should have already applied
    #   canonical names via the taxonomy, but we verify here.
    # ------------------------------------------------------------------
    logger.info("Pass 2: Cross-article consistency check")
    cross_article_issues = _check_global_name_consistency(all_results, registry)
    if cross_article_issues:
        for issue in cross_article_issues[:20]:
            logger.warning("  %s", issue)

    logger.info("Cleaning complete: %d articles processed", len(all_results))

    if generate_report:
        generate_cleaning_report(all_results)

    return all_results


def _check_global_name_consistency(
    all_results: list[Any],
    registry: Any,
) -> list[str]:
    """Verify global entity names are consistent across articles.

    For each global entity type, check that names matching the same base
    text use the same canonical form across all articles.

    Returns list of issue descriptions.
    """
    from .dedup import _get_entity_dedup_mode, _GLOBAL_MODES

    issues: list[str] = []

    # Collect global entity names per type, grouped by normalized base
    from collections import defaultdict
    by_type: dict[str, dict[str, set[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    # entity_type -> normalized_base -> {(actual_text, article_id)}

    for result in all_results:
        doc_id = getattr(result, "document_id", "?")
        for ext in result.extractions:
            etype = ext.extraction_class
            mode = _get_entity_dedup_mode(etype, registry)
            if mode not in _GLOBAL_MODES:
                continue
            name = ext.extraction_text.strip()
            base = " ".join(name.lower().split())
            by_type[etype][base].add((name, doc_id))

    for etype, bases in by_type.items():
        for base, variants in bases.items():
            unique_names = {v[0] for v in variants}
            if len(unique_names) > 1:
                doc_list = ", ".join(sorted({v[1] for v in variants}))
                issues.append(
                    f"{etype} '{base}' has inconsistent case across articles: "
                    f"{unique_names} (docs: {doc_list})"
                )

    return issues
