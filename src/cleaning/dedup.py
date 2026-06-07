"""Phase 3: Entity deduplication — merge duplicate entities.

**Scoping model (registry-driven):**

Deduplication is always **within the same source_doc**.  The graph builder
(``src/graph.py``) handles cross-article merging of global entities via
``registry.get_global_types()``.  The cleaning pipeline's job is to ensure
names are canonically normalized so that cross-article merge succeeds.

- **Global entities** (``dedup_mode: fuzzy|exact``): Alternative, Composite_Product,
  Indicator, Method, Swine, Swine_Model, Tissue_Site.  Names are normalized to
  canonical form; within-article duplicates are merged.
- **Article-scoped** (``dedup_mode: article``): Control_Group, Experiment,
  Intervention, Literature, Result.  Normalized and deduplicated within one
  source_doc only.

Merge rules:
- extraction_text: longer variant preferred
- evidence_text: concatenated (deduplicated chunks)
- attributes: deep-merged (non-empty fields preferred)
- ``_merged_from`` attribute tracks the origin for audit.
"""

from __future__ import annotations

import logging
from typing import Any

from src.data import Extraction

logger = logging.getLogger(__name__)

# Dedup modes that indicate a GLOBAL-type entity
_GLOBAL_MODES: frozenset[str] = frozenset({"fuzzy", "exact"})

# Dedup modes that indicate article-scoped
_ARTICLE_MODES: frozenset[str] = frozenset({"article"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _get_entity_dedup_mode(entity_type: str, registry: Any = None) -> str:
    """Return the dedup_mode for an entity type from the registry.

    Falls back to ``"article"`` (safest) when the registry is unavailable.
    """
    if registry is None:
        return "article"
    try:
        ed = registry.entity_def(entity_type)
        return ed.dedup_mode
    except (KeyError, AttributeError):
        return "article"


def deduplicate_extractions(
    extractions: list[Extraction],
    *,
    registry: Any = None,
) -> tuple[list[Extraction], int]:
    """Deduplicate entities within the same source_doc.

    All entity types use the same dedup key: (source_doc, entity_type,
    normalized_name).  This means article-scoped entities stay per-article,
    and global entities also stay per-article for now — cross-article
    merging of global entities is done by the graph builder.

    Returns (deduplicated_list, merge_count).
    """
    if not extractions:
        return [], 0

    # Group all entities by (source_doc, entity_type, normalized_name)
    groups: dict[tuple[str, str, str], list[Extraction]] = {}
    for ext in extractions:
        key = _dedup_key(ext, registry)
        groups.setdefault(key, []).append(ext)

    # Merge groups with >1 member
    merged: int = 0
    result: list[Extraction] = []

    for key, group in groups.items():
        if len(group) == 1:
            result.append(group[0])
        else:
            kept = _merge_group(group)
            result.append(kept)
            merged += len(group) - 1
            logger.debug(
                "Merged %d duplicates: %s '%s'",
                len(group), key[1], key[2],
            )

    return result, merged


# ---------------------------------------------------------------------------
# Dedup key
# ---------------------------------------------------------------------------


def _dedup_key(ext: Extraction, registry: Any = None) -> tuple[str, str, str]:
    """Build a deduplication key: (source_doc, entity_type, normalized_name).

    Uses the entity's **primary_text** attribute value when available from the
    registry, falling back to extraction_text.  This ensures that after
    normalization, two entities with different extraction_text but the same
    canonical primary attribute (e.g. two Control_Groups with group_name
    'Control group' but extraction_text 'CON' and 'Control') are properly
    merged.
    """
    attrs = ext.attributes or {}
    source_doc = _extract_source_doc(attrs)
    etype = ext.extraction_class

    # Prefer the primary_text attribute (e.g. group_name, standard_name,
    # site_name) since it is the canonical identifier after normalization.
    name = _get_primary_text_value(ext, registry) or ext.extraction_text
    name = _normalize_name(name)
    return (source_doc, etype, name)


def _get_primary_text_value(ext: Extraction, registry: Any = None) -> str | None:
    """Return the primary_text attribute value for this entity if defined."""
    if registry is None:
        return None
    try:
        ed = registry.entity_def(ext.extraction_class)
    except (KeyError, AttributeError):
        return None
    if not ed.primary_text:
        return None
    attrs = ext.attributes or {}
    val = attrs.get(ed.primary_text)
    if val and isinstance(val, str) and val.strip():
        return val.strip()
    return None


def _extract_source_doc(attrs: dict[str, Any]) -> str:
    """Extract document identifier from attributes."""
    return (
        str(attrs.get("_source_doc", ""))
        or str(attrs.get("source_doc", ""))
        or str(attrs.get("pmid", ""))
        or str(attrs.get("doi", ""))
    )


def _normalize_name(name: str) -> str:
    """Normalize a name for dedup key comparison."""
    return " ".join(name.strip().lower().split())


# ---------------------------------------------------------------------------
# Group merging
# ---------------------------------------------------------------------------


def _merge_group(group: list[Extraction]) -> Extraction:
    """Merge a group of duplicate extractions into one best-version entity."""
    base = group[0]
    merged_from: list[str] = []

    for ext in group[1:]:
        merged_from.append(ext.extraction_text)

        # Prefer longer extraction_text
        if len(ext.extraction_text.strip()) > len(base.extraction_text.strip()):
            base.extraction_text = ext.extraction_text.strip()

        # Merge evidence_text
        base.evidence_text = _merge_evidence(base.evidence_text, ext.evidence_text)

        # Deep merge attributes
        base.attributes = _merge_attributes(base.attributes, ext.attributes)

        # Keep source_location if base is empty
        if not base.source_location and ext.source_location:
            base.source_location = ext.source_location

    # Track merge history
    if base.attributes is None:
        base.attributes = {}
    deduped = merged_from if len(merged_from) > 1 else (merged_from[0] if merged_from else None)
    if deduped:
        base.attributes["_merged_from"] = deduped

    return base


def _merge_evidence(ev1: str, ev2: str) -> str:
    """Concatenate evidence texts, avoiding exact overlaps."""
    e1 = ev1.strip() if ev1 else ""
    e2 = ev2.strip() if ev2 else ""
    if not e1:
        return e2
    if not e2:
        return e1
    if e2 in e1:
        return e1
    if e1 in e2:
        return e2
    return f"{e1}\n---\n{e2}"


def _merge_attributes(
    a1: dict[str, Any] | None,
    a2: dict[str, Any] | None,
) -> dict[str, Any]:
    """Deep-merge two attribute dicts, preferring non-empty values from a2."""
    result = dict(a1) if a1 else {}
    if not a2:
        return result

    for key, val2 in a2.items():
        if key.startswith("_"):
            continue
        val1 = result.get(key)
        if _is_empty(val1) and not _is_empty(val2):
            result[key] = val2
        elif isinstance(val1, list) and isinstance(val2, list):
            merged = list(val1)
            for item in val2:
                if item not in merged:
                    merged.append(item)
            result[key] = merged
        elif isinstance(val1, str) and isinstance(val2, str):
            if len(val2) > len(val1):
                result[key] = val2

    return result


def _is_empty(val: Any) -> bool:
    """Check if a value is effectively empty."""
    if val is None:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    if isinstance(val, (list, dict)) and len(val) == 0:
        return True
    return False
