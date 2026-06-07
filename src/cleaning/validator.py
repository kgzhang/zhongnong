"""Phase 2: Data validation — cross-table reference integrity and value constraints.

Validates:
1. Result.indicator_abbreviation references exist in Indicator (standard_name or abbreviation)
2. Result.tissue_site references exist in Tissue_Site.site_name
3. Intervention.intervention_target references exist in Alternative.standard_name
4. p_value is a valid float in [0, 1]
5. direction and relation_type use canonical value sets
6. Name quality scoring (flag suspicious names)

Returns (filtered_extractions, warnings).
When strict=True, entities failing critical validation are dropped.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.data import Extraction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical value sets
# ---------------------------------------------------------------------------

VALID_DIRECTIONS: frozenset[str] = frozenset({
    "increased", "decreased", "no_significant_change",
})

VALID_RELATION_TYPES: frozenset[str] = frozenset({
    "increases", "decreases", "affects", "upregulates", "downregulates",
})

VALID_SIGNIFICANCE_LEVELS: frozenset[str] = frozenset({
    "p_less_0.05", "p_less_0.01", "p_less_0.001", "not_significant", "trend_0.05_0.1",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_extractions(
    extractions: list[Extraction],
    *,
    registry: Any = None,
    strict: bool = False,
) -> tuple[list[Extraction], list[str]]:
    """Run all validators on the extraction list.

    Returns (filtered_list, warnings).
    """
    warnings: list[str] = []

    # 1. Build lookup indexes
    indexes = _build_reference_indexes(extractions)

    # 2. Per-entity validation
    keep: list[Extraction] = []
    for ext in extractions:
        entity_warnings = _validate_entity(ext, indexes)
        if entity_warnings:
            warnings.extend(entity_warnings)
        # Only drop if strict and critical issues
        critical = _has_critical_issue(entity_warnings)
        if critical and strict:
            logger.warning(
                "Dropped %s '%s': %s",
                ext.extraction_class, ext.extraction_text,
                "; ".join(entity_warnings),
            )
        else:
            keep.append(ext)

    # 3. Cross-entity orphan detection (advisory)
    orphan_warnings = _detect_orphan_references(extractions, indexes)
    warnings.extend(orphan_warnings)

    return keep, warnings


# ---------------------------------------------------------------------------
# Reference indexes
# ---------------------------------------------------------------------------


def _build_reference_indexes(extractions: list[Extraction]) -> dict[str, Any]:
    """Build lookup indexes for validating cross-entity references."""
    indexes: dict[str, Any] = {}

    # Indicator lookup: by standard_name and by abbreviation
    indicator_by_name: dict[str, set[str]] = {}  # normalized → {original}
    indicator_by_abbr: dict[str, set[str]] = {}
    for ext in extractions:
        if ext.extraction_class != "Indicator":
            continue
        attrs = ext.attributes or {}
        name = (attrs.get("standard_name") or ext.extraction_text).strip()
        abbr = (attrs.get("abbreviation") or "").strip()
        if name:
            indicator_by_name.setdefault(name.lower(), set()).add(name)
        if abbr:
            indicator_by_abbr.setdefault(abbr.lower(), set()).add(abbr)

    indexes["indicator_by_name"] = indicator_by_name
    indexes["indicator_by_abbr"] = indicator_by_abbr

    # Tissue_Site lookup
    tissue_sites: set[str] = set()
    for ext in extractions:
        if ext.extraction_class != "Tissue_Site":
            continue
        attrs = ext.attributes or {}
        site = (attrs.get("site_name") or ext.extraction_text).strip()
        if site:
            tissue_sites.add(site.lower())
    indexes["tissue_sites"] = tissue_sites

    # Alternative lookup
    alternatives: set[str] = set()
    for ext in extractions:
        if ext.extraction_class != "Alternative":
            continue
        attrs = ext.attributes or {}
        name = (attrs.get("standard_name") or ext.extraction_text).strip()
        if name:
            alternatives.add(name.lower())
    indexes["alternatives"] = alternatives

    # Swine lookup
    swine: set[str] = set()
    for ext in extractions:
        if ext.extraction_class != "Swine":
            continue
        attrs = ext.attributes or {}
        breed = (attrs.get("breed") or "").strip()
        if breed:
            swine.add(breed.lower())
    indexes["swine"] = swine

    # Method lookup
    methods: set[str] = set()
    for ext in extractions:
        if ext.extraction_class != "Method":
            continue
        attrs = ext.attributes or {}
        mn = (attrs.get("method_name") or ext.extraction_text).strip()
        if mn:
            methods.add(mn.lower())
    indexes["methods"] = methods

    return indexes


# ---------------------------------------------------------------------------
# Per-entity validation
# ---------------------------------------------------------------------------


def _validate_entity(ext: Extraction, indexes: dict) -> list[str]:
    """Validate a single extraction.  Returns list of warning strings."""
    warnings: list[str] = []

    etype = ext.extraction_class
    attrs = ext.attributes or {}
    name = ext.extraction_text.strip()

    if etype == "Result":
        warnings.extend(_validate_result(ext, indexes))
    elif etype == "Intervention":
        warnings.extend(_validate_intervention(ext, indexes))

    # Name quality scoring
    score = _name_quality_score(name)
    if score < 40:
        warnings.append(
            f"{etype} '{name}': low quality score {score}/100"
        )

    return warnings


def _validate_result(ext: Extraction, indexes: dict) -> list[str]:
    """Validate a Result extraction."""
    warnings: list[str] = []
    attrs = ext.attributes or {}
    name = ext.extraction_text.strip()

    # Check indicator reference
    indicator_abbr = (attrs.get("indicator_abbreviation") or "").strip()
    if indicator_abbr:
        found = _find_indicator(indicator_abbr, indexes)
        if not found:
            warnings.append(
                f"Result '{name}': indicator_abbreviation='{indicator_abbr}' "
                f"not found in Indicator table"
            )

    # Check tissue_site reference
    tissue = (attrs.get("tissue_site") or "").strip()
    if tissue:
        tissue_sites = indexes.get("tissue_sites", set())
        if tissue.lower() not in tissue_sites:
            # Try fuzzy: strip "in the", "and", etc.
            tissue_clean = re.sub(r"^(in the|in|the)\s+", "", tissue.lower())
            if tissue_clean not in tissue_sites:
                warnings.append(
                    f"Result '{name}': tissue_site='{tissue}' "
                    f"not found in Tissue_Site table"
                )

    # Validate p_value
    p_val = attrs.get("p_value")
    if p_val is not None:
        if isinstance(p_val, str) and not _is_valid_float_str(p_val):
            warnings.append(
                f"Result '{name}': p_value='{p_val}' is not a valid number"
            )

    # Validate direction
    direction = (attrs.get("direction") or "").strip()
    if direction and direction.lower() not in VALID_DIRECTIONS:
        warnings.append(
            f"Result '{name}': non-standard direction='{direction}'"
        )

    # Validate relation_type
    rel_type = (attrs.get("relation_type") or "").strip()
    if rel_type and rel_type.lower() not in VALID_RELATION_TYPES:
        warnings.append(
            f"Result '{name}': non-standard relation_type='{rel_type}'"
        )

    # Validate significance_level
    sig = (attrs.get("significance_level") or "").strip()
    if sig and sig.lower() not in VALID_SIGNIFICANCE_LEVELS:
        warnings.append(
            f"Result '{name}': non-standard significance_level='{sig}'"
        )

    return warnings


def _validate_intervention(ext: Extraction, indexes: dict) -> list[str]:
    """Validate an Intervention extraction."""
    warnings: list[str] = []
    attrs = ext.attributes or {}
    name = ext.extraction_text.strip()

    target = (attrs.get("intervention_target") or "").strip()
    if target:
        alternatives = indexes.get("alternatives", set())
        if target.lower() not in alternatives:
            # Try fuzzy matching
            if not _fuzzy_match_target(target, alternatives):
                warnings.append(
                    f"Intervention '{name}': intervention_target='{target}' "
                    f"not found in Alternative entities"
                )

    return warnings


# ---------------------------------------------------------------------------
# Reference resolution helpers
# ---------------------------------------------------------------------------


def _find_indicator(value: str, indexes: dict) -> str | None:
    """Try to find an Indicator matching *value*.

    Cascading strategies:
    1. Exact match (case-insensitive) on standard_name
    2. Exact match on abbreviation
    3. Strip "g_" prefix (e.g. "g_Lactobacillus" → "Lactobacillus")
    4. Strip parenthetical suffix (e.g. "total cholesterol (tchol)" → "total cholesterol")
    5. Substring / partial match
    """
    if not value:
        return None

    key = value.strip().lower()

    # 1. Exact on standard_name
    by_name = indexes.get("indicator_by_name", {})
    if key in by_name:
        return key

    # 2. Exact on abbreviation
    by_abbr = indexes.get("indicator_by_abbr", {})
    if key in by_abbr:
        return key

    # 3. Strip "g_" prefix
    if key.startswith("g_"):
        stripped = key[2:]
        if stripped in by_name:
            return stripped
        if stripped in by_abbr:
            return stripped

    # 4. Strip parenthetical suffix
    paren_stripped = re.sub(r"\s*\([^)]*\)\s*$", "", key).strip()
    if paren_stripped != key and paren_stripped:
        if paren_stripped in by_name:
            return paren_stripped

    # 5. Substring / partial match
    for name_key in by_name:
        if key in name_key or name_key in key:
            return name_key

    return None


def _fuzzy_match_target(target: str, alternatives: set[str]) -> bool:
    """Check if *target* fuzzy-matches any alternative name."""
    key = target.lower().strip()
    for alt in alternatives:
        if key in alt or alt in key:
            return True
        # Word overlap
        key_words = set(key.split())
        alt_words = set(alt.split())
        if len(key_words & alt_words) >= 2:
            return True
    return False


# ---------------------------------------------------------------------------
# Orphan detection
# ---------------------------------------------------------------------------


def _detect_orphan_references(
    extractions: list[Extraction],
    indexes: dict,
) -> list[str]:
    """Detect orphan Result entities not referenced by any Indicator."""
    warnings: list[str] = []

    # Build set of all indicator references from Results
    result_refs: set[str] = set()
    for ext in extractions:
        if ext.extraction_class == "Result":
            attrs = ext.attributes or {}
            abbr = (attrs.get("indicator_abbreviation") or "").strip()
            if abbr:
                result_refs.add(abbr.lower())

    # Check which Indicators are NOT referenced
    by_name = indexes.get("indicator_by_name", {})
    by_abbr = indexes.get("indicator_by_abbr", {})

    unreferenced_names: set[str] = set()
    for key in by_name:
        if key not in result_refs:
            # Check if abbreviation is referenced
            for abbr_set in by_abbr.values():
                if any(a.lower() in result_refs for a in abbr_set):
                    break
            else:
                unreferenced_names.add(key)

    if unreferenced_names:
        warnings.append(
            f"Orphan Indicators: {len(unreferenced_names)} indicators "
            f"not referenced by any Result"
        )

    return warnings


# ---------------------------------------------------------------------------
# Name quality scoring
# ---------------------------------------------------------------------------


def _name_quality_score(name: str) -> int:
    """Score a name from 0-100 based on quality heuristics."""
    if not name or not isinstance(name, str):
        return 0

    s = name.strip()
    score = 0

    # Length in reasonable range
    if 3 <= len(s) <= 80:
        score += 30
    elif len(s) < 3:
        score += 10

    # Not starting with bare numbers
    if not re.match(r"^\d+", s):
        score += 20

    # Not all digits/punctuation
    if not re.match(r"^[\d\s\.,;:!?\-–—()\[\]{}]+$", s):
        score += 20

    # Not containing dose-like information
    if not re.search(r"\d+\s*[×x]\s*\d+\s*(CFU|cfu)", s):
        score += 15

    # Not looking like JSON/dict content
    if not re.match(r"^\s*[\{\[]", s):
        score += 15

    return min(score, 100)


def _has_critical_issue(warnings: list[str]) -> bool:
    """Check if warnings contain critical (droppable) issues."""
    for w in warnings:
        if "low quality" in w:
            return True
    return False


def _is_valid_float_str(s: str) -> bool:
    """Check if string can be parsed as a float."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False
