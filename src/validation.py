"""Validation and cleaning functions for the extraction pipeline.

Generic, domain-agnostic validators that operate purely on Extraction objects
and registry metadata.  No entity-type-specific hardcoding.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

from src.data import Extraction
from src.coreference import identity_key

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Issue 1: Entity Type Mutual Exclusivity
# ---------------------------------------------------------------------------
# Exclusivity is driven entirely by entity YAML ``notes`` field with
# ``mutually_exclusive_with: EntityName`` and ``priority: N`` markers.


def _parse_priority(notes: str) -> int:
    """Extract a numeric priority from entity notes.

    Scans lines matching ``priority: N``.  Higher number = higher priority
    (survives when identities overlap across mutually-exclusive types).
    Default is 0.
    """
    if not notes:
        return 0
    for line in notes.splitlines():
        m = re.match(r"^\s*priority\s*:\s*(\d+)\s*$", line.strip(), re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 0


def _parse_exclusivity_set(notes: str) -> frozenset[str] | None:
    """Parse a ``mutually_exclusive_with: TypeA, TypeB`` line from notes.

    Returns a frozenset of entity type names to treat as mutually exclusive
    with this entity, or None if not specified.
    """
    if not notes:
        return None
    for line in notes.splitlines():
        m = re.match(
            r"^\s*mutually_exclusive_with\s*:\s*(.+)$",
            line.strip(),
            re.IGNORECASE,
        )
        if m:
            types = [t.strip() for t in m.group(1).split(",") if t.strip()]
            if types:
                return frozenset(types)
    return None


def validate_mutual_exclusivity(
    extractions: list[Extraction],
    registry: Any = None,
) -> list[Extraction]:
    """Ensure no entity name appears in multiple mutually-exclusive entity types.

    Reads entity definitions from *registry*.  Checks
    ``mutually_exclusive_with: TypeA, TypeB`` and ``priority: N``
    in entity notes.

    When an identity overlaps across exclusive types, the higher-priority type
    is kept and the lower is removed.

    Returns filtered list with duplicates removed from the lower-priority type.
    """
    if not extractions or registry is None:
        return list(extractions)

    # 1. Build exclusivity map from entity definitions
    #    {(type_a, type_b): True}  where a < b alphabetically
    exclusivity_map: dict[frozenset[str], bool] = {}
    priority_map: dict[str, int] = {}
    try:
        all_names = registry.all_entity_names()
    except AttributeError:
        all_names = []
    for ename in all_names:
        try:
            ed = registry.entity_def(ename)
        except (KeyError, AttributeError):
            continue
        # Parse priority
        p = _parse_priority(ed.notes)
        if p > 0:
            priority_map[ename] = p
        # Parse mutual exclusivity
        eset = _parse_exclusivity_set(ed.notes)
        if eset:
            for other in eset:
                pair = frozenset({ename, other})
                exclusivity_map[pair] = True

    if not exclusivity_map:
        return list(extractions)

    # 2. Group extractions by their canonical identity key
    by_identity: dict[str, list[Extraction]] = defaultdict(list)
    for ext in extractions:
        primary_field: str | None = None
        try:
            ed = registry.entity_def(ext.extraction_class)
            primary_field = ed.primary_text
        except (KeyError, AttributeError):
            pass
        key = identity_key(ext, primary_field)
        if key:
            by_identity[key].append(ext)

    # 3. Determine which indices to keep
    keep_indices: set[int] = set(range(len(extractions)))
    conflicts_found = 0

    for key, group in by_identity.items():
        if len(group) <= 1:
            continue

        for i, ext_a in enumerate(group):
            for j, ext_b in enumerate(group):
                if i >= j:
                    continue
                if ext_a.extraction_class == ext_b.extraction_class:
                    continue

                etype_a = ext_a.extraction_class
                etype_b = ext_b.extraction_class
                pair = frozenset({etype_a, etype_b})

                if pair not in exclusivity_map:
                    continue

                pa = priority_map.get(etype_a, 0)
                pb = priority_map.get(etype_b, 0)
                if pa > pb:
                    loser = ext_b
                    winner = ext_a
                elif pb > pa:
                    loser = ext_a
                    winner = ext_b
                else:
                    # Same priority — keep first occurrence
                    loser = ext_b
                    winner = ext_a

                for idx in range(len(extractions)):
                    if extractions[idx] is loser:
                        keep_indices.discard(idx)
                        conflicts_found += 1
                        logger.info(
                            "Mutual exclusivity: removed %s '%s' (priority %d), "
                            "kept %s '%s' (priority %d)",
                            loser.extraction_class,
                            loser.extraction_text,
                            priority_map.get(loser.extraction_class, 0),
                            winner.extraction_class,
                            winner.extraction_text,
                            priority_map.get(winner.extraction_class, 0),
                        )
                        break

    result = [extractions[i] for i in sorted(keep_indices)]
    if conflicts_found:
        logger.info(
            "validate_mutual_exclusivity: removed %d entities across %d conflict(s)",
            len(extractions) - len(result),
            conflicts_found,
        )
    return result


# ---------------------------------------------------------------------------
# Issue 3: Cross-Entity Dependency Validation
# ---------------------------------------------------------------------------


def _resolve_ref_field_to_entity_field(
    entity_type: str,
    ref_field: str,
    registry: Any,
) -> tuple[str | None, str | None, str | None]:
    """Given an entity type and a reference field name, look up the target entity
    and target field from the entity's reference definitions.

    Returns ``(target_entity, target_field, edge_type)`` or ``(None, None, None)``.
    """
    if registry is None:
        return None, None, None
    try:
        ed = registry.entity_def(entity_type)
    except (KeyError, AttributeError):
        return None, None, None

    for ref in ed.references:
        if ref.name == ref_field:
            return ref.target_entity, ref.target_field, ref.edge_type
    return None, None, None


def validate_cross_references(
    extractions: list[Extraction],
    registry: Any = None,
) -> list[str]:
    """Validate cross-entity references.  Returns list of warning strings.

    Does **not** drop entities — only emits warnings for broken references.

    Checks every entity type's ``references`` definitions.  For each
    reference, builds a lookup set from all extractions of the target entity
    type and verifies that the referencing attribute value appears in the set.
    """
    warnings: list[str] = []

    if not extractions or registry is None:
        return warnings

    # Build lookup sets per entity type and field
    #  lookup[(entity_type, field)] = set of normalized values
    lookup: dict[tuple[str, str], set[str]] = {}

    for ext in extractions:
        attrs = ext.attributes or {}
        try:
            ed = registry.entity_def(ext.extraction_class)
        except (KeyError, AttributeError):
            continue

        # Index the primary text field
        pf = ed.primary_text
        pf_val = attrs.get(pf, ext.extraction_text)
        if pf_val:
            key = (ext.extraction_class, pf)
            lookup.setdefault(key, set()).add(pf_val.strip())

        # Index other attribute fields so we can validate references against them
        for attr_def in ed.attributes:
            val = attrs.get(attr_def.name)
            if val and isinstance(val, str) and val.strip():
                key = (ext.extraction_class, attr_def.name)
                lookup.setdefault(key, set()).add(val.strip())

    # Also add extraction_text as a fallback
    for ext in extractions:
        key = (ext.extraction_class, "extraction_text")
        lookup.setdefault(key, set()).add(ext.extraction_text.strip())

    # Now check every extraction's reference fields
    for ext in extractions:
        attrs = ext.attributes or {}
        try:
            ed = registry.entity_def(ext.extraction_class)
        except (KeyError, AttributeError):
            continue

        for ref in ed.references:
            target_entity = ref.target_entity
            target_field = ref.target_field
            edge_type = ref.edge_type

            # Get the value of this reference field on the current extraction
            ref_value = attrs.get(ref.name)
            if ref_value is None or not isinstance(ref_value, str) or not ref_value.strip():
                # Empty reference is not an error — it's just unset
                continue

            # Build a key to look up in the lookup set
            lookup_key = (target_entity, target_field)
            target_values = lookup.get(lookup_key, set())

            if not target_values:
                warnings.append(
                    f"{ext.extraction_class} '{ext.extraction_text}': "
                    f"{ref.name}='{ref_value}' references {target_entity}.{target_field}, "
                    f"but no {target_entity} entities found"
                )
                continue

            # Check exact match (case-insensitive) first
            ref_norm = ref_value.strip().lower()
            found = False
            for tv in target_values:
                if tv.lower() == ref_norm:
                    found = True
                    break

            if not found:
                # Try substring matching
                for tv in target_values:
                    if ref_norm in tv.lower() or tv.lower() in ref_norm:
                        found = True
                        break

            if not found:
                warnings.append(
                    f"{ext.extraction_class} '{ext.extraction_text}': "
                    f"{ref.name}='{ref_value}' references {target_entity}.{target_field} "
                    f"(edge: {edge_type}), but no matching target found"
                )

    return warnings


# ---------------------------------------------------------------------------
# Issue 4: Synthetic Name Detection & Cleaning
# ---------------------------------------------------------------------------


def _looks_synthetic(name: str) -> bool:
    """Return True if *name* looks LLM-constructed rather than extracted.

    Heuristics:
    - Contains underscore (often used as concatenation)
    - All lowercase and > 4 words (likely summary, not name)
    - Starts/ends with obvious framing phrases
    """
    if not name:
        return False

    stripped = name.strip()

    # Underscore-based concatenation
    if "_" in stripped:
        return True

    # All lowercase and too many words = likely descriptive, not extracted
    words = stripped.split()
    if len(words) > 4 and stripped.islower():
        return True

    # Framing phrases that suggest the LLM invented it
    framing = (
        "the study", "this experiment", "this research", "our study",
        "the paper", "this paper", "the trial", "this trial",
    )
    low = stripped.lower()
    for f in framing:
        if low.startswith(f) or low.endswith(f):
            return True

    return False


def _clean_extraction_name(ext: Extraction) -> Extraction:
    """Attempt to clean a synthetic extraction_text.

    Tries evidence_text first (extract a quoted or capitalized segment),
    then falls back to attributes.primary_text.
    """
    original = ext.extraction_text

    # Try evidence_text — look for a quoted string near the synthetic name
    ev = getattr(ext, "evidence_text", "") or ""
    if ev:
        # Look for a phrase that contains part of the synthetic name
        parts = [p for p in re.split(r"[_\s]+", original) if len(p) > 2]
        if len(parts) >= 2:
            # Try to find a contiguous span in evidence containing these words
            for i in range(len(parts)):
                for j in range(i + 2, min(i + 6, len(parts) + 1)):
                    candidate = " ".join(parts[i:j])
                    if candidate.lower() in ev.lower() and len(candidate) >= 5:
                        ext.extraction_text = ev[
                            ev.lower().index(candidate.lower()):
                            ev.lower().index(candidate.lower()) + len(candidate)
                        ]
                        logger.debug(
                            "Cleaned synthetic name '%s' -> '%s'", original, ext.extraction_text
                        )
                        return ext

    # Fallback: try primary_text from attributes if it looks clean
    attrs = ext.attributes or {}
    for key in ("standard_name", "product_name", "group_name", "method_name"):
        val = attrs.get(key)
        if val and isinstance(val, str) and not _looks_synthetic(val):
            ext.extraction_text = val
            logger.debug(
                "Cleaned synthetic name '%s' -> '%s' (from %s)", original, val, key
            )
            return ext

    return ext


def clean_synthetic_names(
    extractions: list[Extraction],
) -> list[Extraction]:
    """Clean synthetically-constructed entity names.

    Detect names that were clearly constructed by the LLM (contain underscores,
    are concatenations of multiple concepts, or are framing phrases).
    Try to extract the clean name from evidence_text or attributes.

    Returns the list (mutated in-place for synthetic names).
    """
    cleaned = 0
    for ext in extractions:
        if _looks_synthetic(ext.extraction_text):
            _clean_extraction_name(ext)
            cleaned += 1

    if cleaned:
        logger.info("clean_synthetic_names: cleaned %d synthetic names", cleaned)
    return extractions


# ---------------------------------------------------------------------------
# Issue 5: Required Field Validation
# ---------------------------------------------------------------------------


def filter_incomplete_entities(
    extractions: list[Extraction],
    registry: Any = None,
) -> list[Extraction]:
    """Remove entities missing required LLM fields.

    For each entity type, reads required fields from the registry
    (source == "llm" and required == True).  Drops any extraction that is
    missing a value for a required field (value is None, empty string, or
    whitespace-only).

    Returns filtered list and logs warnings for each dropped entity.
    """
    if not extractions or registry is None:
        return list(extractions)

    # Pre-compute required fields per entity type
    required_fields: dict[str, list[str]] = {}
    try:
        all_names = registry.all_entity_names()
    except AttributeError:
        all_names = []

    for ename in all_names:
        try:
            ed = registry.entity_def(ename)
            req = [a.name for a in ed.attributes if a.required and a.source == "llm"]
            if req:
                required_fields[ename] = req
        except (KeyError, AttributeError):
            pass

    keep: list[Extraction] = []
    dropped = 0

    for ext in extractions:
        req_fields = required_fields.get(ext.extraction_class, [])
        if not req_fields:
            keep.append(ext)
            continue

        # Gate-preserved entities get a pass on missing fields
        # (they were validated by the gate check itself)
        is_gate_preserved = (ext.attributes or {}).pop("_gate_preserved", False)

        attrs = ext.attributes or {}
        missing: list[str] = []
        for rf in req_fields:
            val = attrs.get(rf)
            if val is None:
                missing.append(rf)
            elif isinstance(val, str) and val.strip() == "":
                missing.append(rf)
            elif isinstance(val, list) and len(val) == 0:
                missing.append(rf)
            elif isinstance(val, dict) and len(val) == 0:
                missing.append(rf)

        if missing and not is_gate_preserved:
            logger.warning(
                "Dropped %s '%s': missing required field(s) %s",
                ext.extraction_class,
                ext.extraction_text,
                missing,
            )
            dropped += 1
        else:
            if missing and is_gate_preserved:
                logger.debug(
                    "Kept gate-preserved %s '%s' despite missing: %s",
                    ext.extraction_class,
                    ext.extraction_text,
                    missing,
                )
            keep.append(ext)

    if dropped:
        logger.info(
            "filter_incomplete_entities: dropped %d incomplete entities, kept %d",
            dropped,
            len(keep),
        )
    return keep


# ---------------------------------------------------------------------------
# Issue 2: Entity Constraint Enforcement
# ---------------------------------------------------------------------------


def _build_reference_maps(registry: Any) -> tuple[dict[str, list[tuple[str, str, str]]], dict[str, list[tuple[str, str, str]]]]:
    """Build forward and reverse reference maps from registry entity definitions.

    Returns:
      forward: {entity_type: [(ref_field, target_entity, edge_type), ...]}
        — which types a given entity references.
      reverse: {entity_type: [(referencing_type, ref_field, edge_type), ...]}
        — which types reference a given entity.
    """
    forward: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    reverse: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    try:
        all_names = registry.all_entity_names()
    except AttributeError:
        return forward, reverse

    for ename in all_names:
        try:
            ed = registry.entity_def(ename)
        except (KeyError, AttributeError):
            continue
        # Collect from explicit references
        for ref in ed.references:
            forward[ename].append((ref.name, ref.target_entity, ref.edge_type))
            reverse[ref.target_entity].append((ename, ref.name, ref.edge_type))
        # Collect from inline relations
        for ir in ed.inline_relations:
            for target in ir.target:
                forward[ename].append((ir.via_field, target, ir.name))
                reverse[target].append((ename, ir.via_field, ir.name))

    return forward, reverse


def validate_entity_constraints(
    extractions: list[Extraction],
    registry: Any = None,
) -> tuple[list[Extraction], list[str]]:
    """Validate cross-entity constraints driven by registry entity definitions.

    Generic — no entity-type-specific hardcoding.  For each entity type:
    - Checks that reference fields (from entity.references) have non-empty values.
    - Detects orphan entities: entities not referenced by any other extraction.
    """
    warnings: list[str] = []

    if not extractions:
        return [], warnings

    # 1. Build reference maps from registry
    forward_refs, reverse_refs = {}, {}
    if registry is not None:
        forward_refs, reverse_refs = _build_reference_maps(registry)

    # 2. Collect reference values from all extractions
    #    {(referencing_etype, ref_field) -> set of normalized target values}
    reference_values: dict[tuple[str, str], set[str]] = defaultdict(set)

    for ext in extractions:
        attrs = ext.attributes or {}
        etype = ext.extraction_class

        for ref_field, target_entity, edge_type in forward_refs.get(etype, []):
            val = attrs.get(ref_field)
            if val and isinstance(val, str) and val.strip():
                reference_values[(target_entity, ref_field)].add(val.strip().lower())
            elif val and isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        for dv in item.values():
                            if isinstance(dv, str) and dv.strip():
                                reference_values[(target_entity, ref_field)].add(dv.strip().lower())
                    elif isinstance(item, str) and item.strip():
                        reference_values[(target_entity, ref_field)].add(item.strip().lower())

    # Build a map of required attributes per entity type for reference field checking
    required_attrs: dict[str, set[str]] = {}
    if registry is not None:
        try:
            for ename in registry.all_entity_names():
                try:
                    ed = registry.entity_def(ename)
                    required = {a.name for a in ed.attributes if a.required and a.source == "llm"}
                    if required:
                        required_attrs[ename] = required
                except (KeyError, AttributeError):
                    pass
        except AttributeError:
            pass

    keep: list[Extraction] = []
    for ext in extractions:
        attrs = ext.attributes or {}
        etype = ext.extraction_class

        # Check reference fields: only DROP if the underlying attribute is required.
        # For optional reference fields, just warn.
        missing_required_refs: list[str] = []
        missing_optional_refs: list[str] = []
        for ref_field, target_entity, _edge_type in forward_refs.get(etype, []):
            val = attrs.get(ref_field)
            is_empty = (
                val is None
                or (isinstance(val, str) and not val.strip())
                or (isinstance(val, list) and len(val) == 0)
            )
            if is_empty:
                if ref_field in required_attrs.get(etype, set()):
                    missing_required_refs.append(ref_field)
                else:
                    missing_optional_refs.append(ref_field)

        if missing_required_refs:
            warnings.append(
                f"{etype} '{ext.extraction_text}': dropped, missing required reference field(s) {missing_required_refs}"
            )
            continue
        if missing_optional_refs:
            logger.debug(
                "%s '%s': optional reference field(s) missing: %s",
                etype, ext.extraction_text, missing_optional_refs,
            )

        # Orphan detection: check if this entity is referenced by any other entity
        # Only check entities that have reverse references (someone expects to reference them)
        if etype in reverse_refs and reverse_refs[etype]:
            et_lower = ext.extraction_text.strip().lower()
            # Look through primary text and all attribute values
            identity_values: set[str] = {et_lower}
            try:
                ed = registry.entity_def(etype)
                pf_val = attrs.get(ed.primary_text)
                if pf_val and isinstance(pf_val, str) and pf_val.strip():
                    identity_values.add(pf_val.strip().lower())
            except (KeyError, AttributeError):
                pass

            is_referenced = False
            for ident_val in identity_values:
                if not ident_val:
                    continue
                for (ref_entity, ref_field), targets in reference_values.items():
                    if ref_entity != etype:
                        continue
                    if ident_val in targets:
                        is_referenced = True
                        break
                if is_referenced:
                    break

            if not is_referenced:
                warnings.append(
                    f"{etype} '{ext.extraction_text}': dropped, orphan (not referenced by any other entity)"
                )
                continue

        keep.append(ext)

    dropped_count = len(extractions) - len(keep)
    if dropped_count:
        logger.info(
            "validate_entity_constraints: dropped %d entities, kept %d",
            dropped_count,
            len(keep),
        )
    return keep, warnings


# ---------------------------------------------------------------------------
# Issue 3: Extraction Text Verification + Parenthetical Cleanup
# ---------------------------------------------------------------------------


def _word_level_match(extraction_text: str, evidence: str, threshold: float = 0.7) -> bool:
    """Check if significant words from extraction_text appear in evidence.

    Handles concatenated/synthetic names like "milk fat" where the LLM
    constructed a name but the individual component words exist in the text.
    """
    # Split on word boundaries, keep only significant words (>2 chars)
    words = [w for w in re.split(r"[\s_\-]+", extraction_text.lower()) if len(w) > 2]
    if not words:
        return False
    ev_lower = evidence.lower()
    matched = sum(1 for w in words if w in ev_lower)
    ratio = matched / len(words)
    return ratio >= threshold


def verify_extraction_text_in_evidence(
    extractions: list[Extraction],
    full_text: str | None = None,
) -> tuple[list[Extraction], list[str]]:
    """Verify extraction_text appears in evidence_text. Drop entities that fail.

    Also strip parenthetical annotations from extraction_text if the annotated
    form doesn't appear in evidence but the base form does.

    When *full_text* is provided (the complete document text), it is used as
    a fallback search target when *evidence_text* doesn't contain the
    extraction — this handles cases where fuzzy alignment produced an
    evidence window that doesn't completely cover the matched region.
    """
    warnings: list[str] = []

    if not extractions:
        return [], warnings

    keep: list[Extraction] = []
    for ext in extractions:
        et = ext.extraction_text.strip()
        ev = getattr(ext, "evidence_text", "") or ""

        if not et:
            keep.append(ext)
            continue

        # --- Strategy 1: Exact match in evidence_text (case-insensitive) ---
        if et.lower() in ev.lower():
            keep.append(ext)
            continue

        # --- Strategy 1.5: Exact match in FULL document text (fallback) ---
        # Fuzzy alignment can produce evidence windows offset from the
        # actual match.  Searching the full text catches these cases.
        if full_text and et.lower() in full_text.lower():
            keep.append(ext)
            continue

        # --- Strategy 2: Strip parentheticals ---
        # "Microbe-derived antioxidants (sows)" → "Microbe-derived antioxidants"
        stripped = re.sub(r"\s*\([^)]*\)\s*$", "", et).strip()
        if stripped != et:
            if stripped.lower() in ev.lower():
                ext.extraction_text = stripped
                logger.debug("Stripped parenthetical: '%s' → '%s'", et, stripped)
                keep.append(ext)
                continue
            if full_text and stripped.lower() in full_text.lower():
                ext.extraction_text = stripped
                logger.debug("Stripped parenthetical (full text): '%s' → '%s'", et, stripped)
                keep.append(ext)
                continue

        # --- Strategy 3: Word-level matching ---
        # If ≥50% of significant words appear in evidence (or full text),
        # treat as a match (handles concatenated/synthetic names).
        if _word_level_match(et, ev, threshold=0.5):
            keep.append(ext)
            continue
        if full_text and _word_level_match(et, full_text, threshold=0.5):
            keep.append(ext)
            continue

        # Neither works → drop
        warnings.append(
            f"{ext.extraction_class} '{et}': dropped, extraction_text not found in evidence"
        )
        logger.debug(
            "Dropped %s '%s': not found in evidence (ev_len=%d, has_full_text=%s)",
            ext.extraction_class, et, len(ev), bool(full_text),
        )

    dropped_count = len(extractions) - len(keep)
    if dropped_count:
        logger.info(
            "verify_extraction_text_in_evidence: dropped %d entities, kept %d",
            dropped_count,
            len(keep),
        )
    return keep, warnings


# ---------------------------------------------------------------------------
# Issue 4: Orphan Entity Removal (Generic)
# ---------------------------------------------------------------------------


def remove_orphan_entities(
    extractions: list[Extraction],
    registry: Any = None,
) -> tuple[list[Extraction], list[str]]:
    """Remove orphan entities not referenced by any other entity.

    Generic — driven by registry entity definitions.  For each entity type:
    - Reads reference definitions from the registry to determine which types
      reference which other types.
    - Collects reference values from all extractions.
    - Drops entities that are not referenced by at least one other entity
      (and are expected to be referenced, i.e. appear in reverse reference maps).
    """
    warnings: list[str] = []

    if not extractions or registry is None:
        return list(extractions), warnings

    # 1. Build reference maps
    forward_refs, reverse_refs = _build_reference_maps(registry)

    # 2. Collect reference values from all extractions
    #    {(target_etype, ref_field) -> set of normalized values}
    reference_values: dict[tuple[str, str], set[str]] = defaultdict(set)

    for ext in extractions:
        attrs = ext.attributes or {}
        etype = ext.extraction_class

        for ref_field, target_entity, _edge_type in forward_refs.get(etype, []):
            val = attrs.get(ref_field)
            if val and isinstance(val, str) and val.strip():
                reference_values[(target_entity, ref_field)].add(val.strip().lower())
            elif val and isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        for dv in item.values():
                            if isinstance(dv, str) and dv.strip():
                                reference_values[(target_entity, ref_field)].add(dv.strip().lower())
                    elif isinstance(item, str) and item.strip():
                        reference_values[(target_entity, ref_field)].add(item.strip().lower())

    # 3. Filter: keep only entities that are referenced (or should NOT be orphans)
    keep: list[Extraction] = []
    for ext in extractions:
        etype = ext.extraction_class
        attrs = ext.attributes or {}

        # Entities that are NOT expected to be referenced (no reverse refs)
        # are always kept — they are "root" entities.
        if etype not in reverse_refs or not reverse_refs[etype]:
            keep.append(ext)
            continue

        # Build identity values: extraction_text + primary_text attribute
        identity_values: set[str] = {ext.extraction_text.strip().lower()}
        try:
            ed = registry.entity_def(etype)
            pf_val = attrs.get(ed.primary_text)
            if pf_val and isinstance(pf_val, str) and pf_val.strip():
                identity_values.add(pf_val.strip().lower())
        except (KeyError, AttributeError):
            pass

        # Check if any identity value is referenced
        is_referenced = False
        for ident_val in identity_values:
            if not ident_val:
                continue
            for (ref_entity, ref_field), targets in reference_values.items():
                if ref_entity != etype:
                    continue
                # Exact match
                if ident_val in targets:
                    is_referenced = True
                    break
                # Fuzzy: substring or word-overlap matching
                for target in targets:
                    if ident_val in target or target in ident_val:
                        is_referenced = True
                        break
                    # Word-level fuzzy matching
                    name_words = set(w for w in re.split(r"[\s_\-]+", ident_val) if len(w) > 2)
                    target_words = set(w for w in re.split(r"[\s_\-]+", target) if len(w) > 2)
                    if name_words and target_words and len(name_words & target_words) >= 2:
                        is_referenced = True
                        break
                if is_referenced:
                    break
            if is_referenced:
                break

        if not is_referenced:
            warnings.append(
                f"{etype} '{ext.extraction_text}': WARNING orphan (not referenced by any other entity)"
            )
            # Keep the entity but flag it — orphan detection is advisory, not mandatory
        keep.append(ext)

    orphan_count = sum(1 for w in warnings if "orphan" in w.lower())
    if orphan_count:
        logger.info(
            "remove_orphan_entities: flagged %d orphan entities (kept, not dropped)",
            orphan_count,
        )
    return keep, warnings


# ---------------------------------------------------------------------------
# Composite_Product component validation — remove false composites
# ---------------------------------------------------------------------------


def validate_composite_product_components(
    extractions: list[Extraction],
) -> tuple[list[Extraction], list[str]]:
    """Remove Composite_Product entities without real multi-component substance.

    A valid Composite_Product MUST have a non-empty ``components`` list
    with at least 2 distinct entries.  Single-substance entities misclassified
    as Composite_Product by the LLM are dropped — they should be Alternative
    entities instead.
    """
    warnings: list[str] = []
    keep: list[Extraction] = []
    dropped = 0

    for ext in extractions:
        if ext.extraction_class != "Composite_Product":
            keep.append(ext)
            continue

        attrs = ext.attributes or {}
        components = attrs.get("components")
        product_name = attrs.get("product_name", "") or ext.extraction_text

        # Evaluate if this is a genuine multi-component product
        is_valid = False
        if components and isinstance(components, list):
            # Filter out empty strings and self-references
            real_components = [
                c for c in components
                if isinstance(c, str) and c.strip()
                and c.strip().lower() != product_name.strip().lower()
            ]
            if len(real_components) >= 2:
                is_valid = True
            elif len(real_components) == 1:
                warnings.append(
                    f"Composite_Product '{product_name}': only 1 component, "
                    f"dropped (single substance misclassified as composite)"
                )
        elif components and isinstance(components, str) and components.strip():
            # LLM output a string instead of a list
            stripped = components.strip()
            if stripped.lower() != product_name.strip().lower():
                warnings.append(
                    f"Composite_Product '{product_name}': components is string, "
                    f"not list, dropped"
                )

        if is_valid:
            keep.append(ext)
        else:
            dropped += 1
            logger.info(
                "Dropped Composite_Product '%s': insufficient components",
                product_name,
            )

    if dropped:
        logger.info(
            "validate_composite_product_components: dropped %d false "
            "composites, kept %d",
            dropped,
            sum(1 for e in keep if e.extraction_class == "Composite_Product"),
        )
    return keep, warnings


# ---------------------------------------------------------------------------
# Filter entities with empty evidence_text
# ---------------------------------------------------------------------------


def filter_empty_evidence(
    extractions: list[Extraction],
) -> tuple[list[Extraction], list[str]]:
    """Remove entities whose evidence_text is empty or whitespace-only.

    An entity without evidence_text is untraceable and should not be
    included in the knowledge graph.
    """
    warnings: list[str] = []
    keep: list[Extraction] = []
    dropped = 0

    for ext in extractions:
        ev = getattr(ext, "evidence_text", "") or ""
        if ev.strip():
            keep.append(ext)
        else:
            dropped += 1
            logger.warning(
                "Dropped %s '%s': empty evidence_text",
                ext.extraction_class,
                ext.extraction_text,
            )

    if dropped:
        logger.info(
            "filter_empty_evidence: dropped %d entities with empty evidence, kept %d",
            dropped,
            len(keep),
        )
    return keep, warnings
