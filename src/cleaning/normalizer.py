"""Phase 1: Name normalization for extraction entities.

Applies a sequence of text standardization rules to entity names (extraction_text
and attribute values).  All rules are idempotent — running them twice produces
the same result.

Rules (in order):
1.  Trim leading/trailing whitespace
2.  Remove control characters (\\x00–\\x1f except \\t, \\n)
3.  Collapse multiple spaces into a single space
4.  Unicode NFC normalization
5.  Strip empty parenthetical suffixes e.g. ``"Name ()"`` → ``"Name"``
6.  Strip bracketed prefixes e.g. ``"[Prevotella]"`` → ``"Prevotella"``
7.  Case normalization using the domain taxonomy dictionary
8.  Multi-value decomposition (split "A and B" names)

All mutations are tracked and returned as change counts for the cleaning report.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

from src.data import Extraction
from .taxonomy import (
    get_canonical_name,
    CONTROL_GROUP_NORMALIZATION,
    METHOD_NAME_NORMALIZATION,
)
import src.cleaning.taxonomy as _taxonomy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_extractions(
    extractions: list[Extraction],
    *,
    registry: Any = None,
) -> tuple[list[Extraction], int]:
    """Apply all normalization rules to a list of Extraction objects.

    Case normalization uses the domain taxonomy for all entity types, but for
    article-scoped entities (Control_Group, Experiment, Intervention, Literature,
    Result), the scope is per-article — the normalization is applied individually
    per extraction without cross-article canonical lookup.

    Returns (cleaned_list, change_count).
    """
    changes = 0
    expanded: list[Extraction] = []

    for ext in extractions:
        before = ext.extraction_text
        attrs_before = dict(ext.attributes) if ext.attributes else {}

        # 1. Normalize extraction_text
        ext.extraction_text = _normalize_text(ext.extraction_text, ext.extraction_class)

        # 2. Normalize attribute values (name-like fields only)
        if ext.attributes:
            ext.attributes = _normalize_attributes(ext.attributes, ext.extraction_class)
            _normalize_entity_specific(ext)

        # 3. Multi-value decomposition (may expand list)
        decomposed = _decompose_multi_value(ext)
        expanded.extend(decomposed)

        if ext.extraction_text != before or ext.attributes != attrs_before:
            changes += 1
        if len(decomposed) > 1:
            changes += len(decomposed) - 1

    return expanded, changes


# ---------------------------------------------------------------------------
# Core text normalization (applied to every name-like field)
# ---------------------------------------------------------------------------


def _normalize_text(text: str, entity_type: str = "") -> str:
    """Apply all generic text normalization rules to a single string."""
    if not text or not isinstance(text, str):
        return text

    s = text

    # Rule 1: Trim whitespace
    s = s.strip()

    if not s:
        return s

    # Rule 2: Remove control characters (keep tab and newline for now)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)

    # Rule 3: Collapse multiple spaces (but not tabs/newlines)
    s = re.sub(r" {2,}", " ", s)

    # Rule 4: Unicode NFC normalization
    s = unicodedata.normalize("NFC", s)

    # Rule 5: Strip empty parenthetical suffixes: "Name ()" → "Name"
    s = re.sub(r"\s*\(\s*\)\s*$", "", s).strip()

    # Rule 6: Strip bracketed prefixes: "[Prevotella]" → "Prevotella"
    s = _strip_brackets(s)

    # Rule 7: Case normalization via taxonomy
    s = _apply_case_normalization(s, entity_type)

    return s


def _strip_brackets(text: str) -> str:
    """Strip leading/trailing single brackets that wrap the entire name."""
    s = text.strip()
    # Only remove brackets if they wrap the ENTIRE string (not partial brackets)
    if s.startswith("[") and s.endswith("]") and s.count("[") == 1 and s.count("]") == 1:
        inner = s[1:-1].strip()
        if inner:
            return inner
    return s


def _apply_case_normalization(text: str, entity_type: str = "") -> str:
    """Apply domain-aware case normalization via semantic rules.

    Uses taxonomy.py's pattern-based classifier instead of a hardcoded
    dictionary.  The classifier determines the semantic category (amino acid,
    organic acid, mineral, microorganism, anatomical site, enzyme, cytokine,
    etc.) and applies the appropriate casing convention.

    Falls back to Title Case for single-word names and Title Case with stop
    words for multi-word names when no specific rule matches.
    """
    if not text or not isinstance(text, str):
        return text

    s = text.strip()
    if not s:
        return s

    # Delegate to the pattern-based taxonomy
    canonical = get_canonical_name(s)
    if canonical != s:
        logger.debug("Case normalized (rule): %r → %r", s, canonical)
    return canonical


# ---------------------------------------------------------------------------
# Attribute normalization
# ---------------------------------------------------------------------------

# Fields that should be treated as "name-like" (subject to normalization)
_NAME_LIKE_FIELDS: set[str] = {
    "standard_name", "original_text", "abbreviation", "product_name",
    "group_name", "method_name", "site_name", "intervention_target",
    "breed", "model_type", "stressor_name", "indicator_abbreviation",
    "tissue_site", "experiment_id", "description",
}


def _normalize_attributes(attrs: dict[str, Any], entity_type: str) -> dict[str, Any]:
    """Normalize name-like attribute values."""
    result = {}
    for key, value in attrs.items():
        if isinstance(value, str) and key in _NAME_LIKE_FIELDS:
            result[key] = _normalize_text(value, entity_type)
        elif isinstance(value, list):
            # Normalize each string element in lists
            result[key] = [
                _normalize_text(v, entity_type) if isinstance(v, str) else v
                for v in value
            ]
        else:
            result[key] = value
    return result


def _normalize_entity_specific(ext: Extraction) -> None:
    """Apply entity-type-specific normalization rules."""
    attrs = ext.attributes
    if attrs is None:
        return

    etype = ext.extraction_class

    # Control_Group: normalize group names
    if etype == "Control_Group":
        for field in ("group_name",):
            val = attrs.get(field)
            if isinstance(val, str):
                key = val.strip().lower()
                if key in CONTROL_GROUP_NORMALIZATION:
                    attrs[field] = CONTROL_GROUP_NORMALIZATION[key]

    # Method: normalize method names
    elif etype == "Method":
        for field in ("method_name",):
            val = attrs.get(field)
            if isinstance(val, str):
                key = val.strip().lower()
                if key in METHOD_NAME_NORMALIZATION:
                    attrs[field] = METHOD_NAME_NORMALIZATION[key]


# ---------------------------------------------------------------------------
# Multi-value decomposition
# ---------------------------------------------------------------------------

# Patterns that indicate composite/multi-value names that should be split
_MULTI_SUBSTANCE_AND_RE = re.compile(
    r"^(.+?)\s+and\s+(.+)$", re.IGNORECASE
)

# Names with "and" that should NOT be split
_NO_SPLIT_PATTERNS: list[str] = [
    r"mono.*and\s+di.*and\s+tri",     # "mono-, di-, and triglycerides..."
    r"fatty\s+acid",                  # "...fatty acid" context
    r"\d+,\s*\d+.*and",              # "1,3- and 1,6-..." chemical numbering
    r"β-.*and\s+β-",       # "β-1,3 and β-1,6..."
    r"villi\s+and\s+crypt",          # "villi and crypts"
    r"mortality\s+and\s+",           # "mortality and morbidity"
    r"health\s+and\s+\w+",           # "health and welfare"
    r"growth\s+and\s+\w+",           # "growth and development"
    r"quality\s+and\s+\w+",          # "quality and safety"
    # Don't split when the right side contains these compound-indicator words
    r"combination",
    r"complex",
    r"blend",
    r"mixture",
    r"extract.*and.*extract",         # "grape seed and grape marc extract"
    r"powder",                        # "A and B powder"
    r"fiber",                         # "A and B fiber"
    r"oil.*and.*oil",                # "soybean oil and palm oil"
    r"protein.*and",                  # "soy and yeast protein"
    r"and.*protein",                 # "...and yeast protein"
    r"acid.*and.*acid",              # "capric acid and lauric acid"
    r"enzyme",                        # "xylanase and glucanase enzyme combination"
    r"cells",                         # "medium and bacterial cells"
    r"leaves.*and.*branch",          # "mulberry leaves and branches"
    r"fruit.*and.*vegetable",        # "fruits and vegetables"
    r"oil",                           # "essential oil" etc.
]


def _should_split_and_name(name: str) -> bool:
    """Determine if a name with 'and' represents two distinct substances."""
    for pat in _NO_SPLIT_PATTERNS:
        if re.search(pat, name, re.IGNORECASE):
            return False
    # Don't split if the name is very long (likely a descriptive phrase)
    if len(name) > 60:
        return False
    # Don't split if it's 4 words or less total
    if name.count(" ") <= 4:
        return False
    # Don't split if either part is a single word (too aggressive)
    parts = name.split(" and ", 1)
    if len(parts) == 2:
        if parts[0].strip().count(" ") == 0 or parts[1].strip().count(" ") == 0:
            return False
    return True


def _decompose_multi_value(ext: Extraction) -> list[Extraction]:
    """Split multi-substance names into separate entity rows.

    Detects patterns like "A and B" where A and B are distinct substances
    and creates separate extraction rows for each.

    Returns the list of extractions (may be >1 if decomposition occurred).
    """
    name = ext.extraction_text.strip()

    # Check for "A and B" pattern
    m = _MULTI_SUBSTANCE_AND_RE.match(name)
    if not m:
        return [ext]

    if not _should_split_and_name(name):
        return [ext]

    part_a = m.group(1).strip()
    part_b = m.group(2).strip()

    # Only split if both parts look like valid substance names
    if len(part_a) < 2 or len(part_b) < 2:
        return [ext]
    # Don't split if the parts are too similar (likely a phrase)
    if part_a.lower() in part_b.lower() or part_b.lower() in part_a.lower():
        return [ext]

    # Create decomposed copies
    results = [ext]  # keep original
    attrs = dict(ext.attributes) if ext.attributes else {}
    decomposed_from = attrs.get("_decomposed_from", name)

    for part in (part_a, part_b):
        new_attrs = dict(attrs)
        new_attrs["_decomposed_from"] = decomposed_from
        new_ext = Extraction(
            extraction_class=ext.extraction_class,
            extraction_text=part,
            evidence_text=ext.evidence_text,
            source_location=ext.source_location,
            attributes=new_attrs,
        )
        results.append(new_ext)

    logger.debug(
        "Decomposed multi-value %r → [%r, %r]",
        name, part_a, part_b,
    )
    return results
