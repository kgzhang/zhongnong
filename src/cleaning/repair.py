"""Phase 4: Auto-repair — fix known error patterns in extracted data.

Repairs:
1. p_value: convert string to float (strip "p =", "P <", trailing "*" etc.)
2. direction: map non-standard values to canonical (see taxonomy.py)
3. relation_type: map non-standard values to canonical
4. Control_Group names: normalize variant forms
5. Empty/null values in name fields → set empty string

All repairs are tracked for the cleaning report.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.data import Extraction
from .taxonomy import (
    DIRECTION_MAPPING,
    RELATION_TYPE_MAPPING,
    METHOD_NAME_NORMALIZATION,
    get_canonical_direction,
    get_canonical_relation_type,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def repair_extractions(
    extractions: list[Extraction],
) -> tuple[list[Extraction], int]:
    """Apply auto-repair rules to a list of extractions.

    Returns (repaired_list, change_count).
    """
    changes = 0

    for ext in extractions:
        if ext.attributes is None:
            ext.attributes = {}

        etype = ext.extraction_class
        attrs = ext.attributes

        if etype == "Result":
            changes += _repair_result(ext)
        elif etype == "Control_Group":
            changes += _repair_control_group(ext)
        elif etype == "Method":
            changes += _repair_method(ext)

        # Universal: repair null extraction_text
        if not ext.extraction_text or not ext.extraction_text.strip():
            ext.extraction_text = attrs.get("standard_name", "") or attrs.get("name", "") or ""

    return extractions, changes


# ---------------------------------------------------------------------------
# Result repairs
# ---------------------------------------------------------------------------


def _repair_result(ext: Extraction) -> int:
    """Repair a Result extraction.  Returns change count."""
    changes = 0
    attrs = ext.attributes or {}

    # 1. p_value normalization
    p_val = attrs.get("p_value")
    if p_val is not None and isinstance(p_val, str) and p_val.strip():
        cleaned = _parse_p_value(str(p_val))
        if cleaned is not None and cleaned != p_val:
            attrs["p_value"] = cleaned
            changes += 1
            logger.debug("  Repaired p_value: %r → %r", p_val, cleaned)
    # Handle p_value that is the string "None" or empty → set to None
    if p_val is not None and isinstance(p_val, str) and not p_val.strip():
        attrs["p_value"] = None
        changes += 1

    # 2. direction normalization
    direction = (attrs.get("direction") or "").strip()
    if direction:
        canonical = get_canonical_direction(direction)
        if canonical != direction:
            attrs["direction"] = canonical
            changes += 1
            logger.debug("  Repaired direction: %r → %r", direction, canonical)

    # 3. relation_type normalization
    rel_type = (attrs.get("relation_type") or "").strip()
    if rel_type:
        canonical = get_canonical_relation_type(rel_type)
        if canonical != rel_type:
            attrs["relation_type"] = canonical
            changes += 1
            logger.debug("  Repaired relation_type: %r → %r", rel_type, canonical)

    return changes


# ---------------------------------------------------------------------------
# p_value parsing
# ---------------------------------------------------------------------------

_P_VALUE_PATTERNS: list[tuple[str, str]] = [
    # (regex, replacement)
    (r"^[pP]\s*[<>=≈]+\s*", ""),       # "p < 0.05" → "0.05"
    (r"^[pP]\s*", ""),                   # "p = 0.03" → "0.03"
    (r"\*+$", ""),                       # "0.04*" or "0.01**" → "0.04"
    (r"^<\s*", ""),                      # "< 0.001" → "0.001"
    (r"^>\s*", ""),                      # "> 0.05" → "0.05"
    (r"^≈\s*", ""),                      # "≈ 0.05" → "0.05"
    (r"\s*\(.*\)$", ""),                 # "0.04 (adjusted)" → "0.04"
]


def _parse_p_value(raw: str) -> float | str | None:
    """Parse a p_value string into a float.

    Handles formats like:
    - "0.045"
    - "p = 0.045"
    - "P < 0.001"
    - "p < 0.05"
    - "0.04*"
    - "> 0.05"
    - "P=0.03"

    Returns a float, or the original string if unparseable.
    """
    if not raw or not isinstance(raw, str):
        return raw

    s = raw.strip()

    # Already a number?
    try:
        return float(s)
    except (ValueError, TypeError):
        pass

    # Apply patterns to strip prefixes/suffixes
    for pat, repl in _P_VALUE_PATTERNS:
        s = re.sub(pat, repl, s).strip()

    # Try parsing the cleaned string
    try:
        return float(s)
    except (ValueError, TypeError):
        pass

    # Try extracting the first float-like substring
    m = re.search(r"(\d+\.?\d*)", s)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass

    return raw  # Return original if unparseable


# ---------------------------------------------------------------------------
# Control_Group repair
# ---------------------------------------------------------------------------


def _repair_control_group(ext: Extraction) -> int:
    """Repair Control_Group names.  Returns change count."""
    changes = 0
    attrs = ext.attributes or {}

    group_name = (attrs.get("group_name") or "").strip()
    if group_name:
        # Short single-letter names (A, B, C, D, etc.) are treatment group codes, keep them
        if len(group_name) == 1 and group_name.isalpha():
            return 0

        # Numeric-only names that aren't treatment codes — flag but keep
        if group_name.isdigit():
            return 0

    return changes


# ---------------------------------------------------------------------------
# Method repair
# ---------------------------------------------------------------------------


def _repair_method(ext: Extraction) -> int:
    """Repair Method names.  Returns change count."""
    changes = 0
    attrs = ext.attributes or {}

    method_name = (attrs.get("method_name") or "").strip()
    if method_name:
        key = method_name.strip().lower()
        if key in METHOD_NAME_NORMALIZATION:
            attrs["method_name"] = METHOD_NAME_NORMALIZATION[key]
            changes += 1

    return changes
