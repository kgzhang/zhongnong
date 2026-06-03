"""Coreference Resolution Service.

Multi-strategy entity resolution:
  1. Exact normalization (case, hyphens, whitespace)
  2. Fuzzy matching (Levenshtein distance ≤ 2)
  3. Abbreviation ↔ full-name resolution
  4. Vocabulary-based alias resolution (via TSV synonym lists)

Used by the extraction pipeline (dedup before CSV export) and the graph
builder (dedup nodes before edge resolution).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from src.data import Extraction


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------


def normalize_name(name: str | None) -> str:
    """Canonical name normalization.

    - Lowercase
    - Replace hyphens, underscores, slashes, dots with spaces
    - Remove parenthetical annotations like ``(THY)`` or ``(purity >= 99%)``
    - Collapse all whitespace

    >>> normalize_name("Microbe-derived_Antioxidants (MA)")
    'microbe derived antioxidants'
    """
    if not name:
        return ""
    n = name.strip().lower()
    # Strip parentheticals
    n = re.sub(r"\s*\([^)]*\)", "", n)
    # Replace separators with space
    n = re.sub(r"[-_/.,;:]", " ", n)
    # Collapse whitespace
    n = re.sub(r"\s+", " ", n)
    return n.strip()


# ---------------------------------------------------------------------------
# Levenshtein distance
# ---------------------------------------------------------------------------


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,      # deletion
                curr[j] + 1,           # insertion
                prev[j] + (0 if ca == cb else 1),  # substitution
            ))
        prev = curr
    return prev[-1]


def is_fuzzy_match(a: str, b: str, max_dist: int = 2) -> bool:
    """Check if two normalized names match within Levenshtein distance."""
    if not a or not b:
        return False
    # Quick length check: if lengths differ too much, can't match
    if abs(len(a) - len(b)) > max_dist * 2:
        return False
    return _levenshtein(a, b) <= max_dist


# ---------------------------------------------------------------------------
# Identity key computation
# ---------------------------------------------------------------------------


def identity_key(
    extraction: Extraction,
    primary_text_field: str | None = None,
) -> str:
    """Compute the canonical identity key for an extraction.

    Uses *primary_text_field* from attributes if available, falls back to
    extraction_text.  Both are normalized via :func:`normalize_name`.
    """
    if primary_text_field and extraction.attributes:
        val = extraction.attributes.get(primary_text_field)
        if val and isinstance(val, str) and val.strip():
            return normalize_name(val)
    return normalize_name(extraction.extraction_text)


# ---------------------------------------------------------------------------
# Abbreviation map + alias expansion
# ---------------------------------------------------------------------------


def build_abbreviation_map(
    extractions: list[Extraction],
    primary_text_field: str | None = None,
) -> dict[str, str]:
    """Build a mapping from normalized abbreviations/aliases to canonical identities.

    Sources of aliases:
    - ``abbreviation`` field in entity attributes
    - ``components`` field (each component's standard_name is an alias)
    - Short names (≤4 chars) that match the start of a longer name

    Returns ``{alias_norm: canonical_norm}``.
    """
    abbr_map: dict[str, str] = {}

    for ext in extractions:
        attrs = ext.attributes or {}
        canonical = identity_key(ext, primary_text_field)
        if not canonical:
            continue

        # Source 1: explicit abbreviation field
        abbr = attrs.get("abbreviation")
        if abbr and isinstance(abbr, str) and abbr.strip():
            abbr_norm = normalize_name(abbr)
            if abbr_norm and abbr_norm != canonical and len(abbr_norm) >= 2:
                abbr_map[abbr_norm] = canonical

        # Source 2: short extraction_text (likely abbreviation)
        ext_text_norm = normalize_name(ext.extraction_text)
        if 2 <= len(ext_text_norm) <= 5 and ext_text_norm != canonical:
            # Only map if it looks like an abbreviation (all caps in original)
            if ext.extraction_text.isupper() or ext.extraction_text.istitle():
                abbr_map[ext_text_norm] = canonical

    return abbr_map


# ---------------------------------------------------------------------------
# Synonym expansion from vocabulary
# ---------------------------------------------------------------------------


def build_vocabulary_synonyms(
    extractions: list[Extraction],
    registry: Any = None,
) -> dict[str, str]:
    """Build synonym map from ALTERNATIVE.tsv vocabulary.

    For each Alternative extraction, look up the vocabulary to find:
    - Other names in the same TSV cell (comma-separated synonyms)
    - Standard name → all alternative spellings

    Returns ``{synonym_norm: canonical_norm}``.
    """
    synonyms: dict[str, str] = {}

    for ext in extractions:
        if ext.extraction_class != "Alternative":
            continue
        canonical = normalize_name(ext.extraction_text)
        if not canonical:
            continue

        # Get vocabulary entry
        if registry is None:
            continue
        vocab = None
        try:
            vocab = registry.vocabulary("Alternative")
        except (KeyError, AttributeError):
            pass
        if vocab is None:
            continue

        # Look up this extraction in the vocabulary
        entry = vocab.lookup(ext.extraction_text) or vocab.fuzzy_match(ext.extraction_text)
        if entry is None:
            continue

        # Get the key field value (may contain comma-separated synonyms)
        binding = vocab.binding
        raw = entry.get(binding.key_field, "")
        if binding.value_delimiter and binding.value_delimiter in raw:
            parts = [p.strip() for p in raw.split(binding.value_delimiter)]
        else:
            parts = [raw.strip()]

        # Map each synonym to the canonical name
        for part in parts:
            # Strip parenthetical
            name = re.sub(r"\s*\([^)]*\)", "", part).strip()
            if name:
                syn_norm = normalize_name(name)
                if syn_norm and syn_norm != canonical and len(syn_norm) >= 2:
                    synonyms[syn_norm] = canonical

    return synonyms


# ---------------------------------------------------------------------------
# Coreference resolution
# ---------------------------------------------------------------------------


def resolve_coreferences(
    extractions: list[Extraction],
    registry: Any = None,
) -> list[Extraction]:
    """Multi-strategy coreference resolution.

    Strategies applied in order:
    1. Exact normalization match
    2. Abbreviation/alias resolution
    3. Vocabulary synonym expansion
    4. Fuzzy Levenshtein matching (≤ 2 edits)

    Extractions within each merged group are combined via
    :func:`_merge_extraction_group`.
    """
    if not extractions:
        return []

    # Group by entity type
    by_type: dict[str, list[Extraction]] = defaultdict(list)
    for ext in extractions:
        by_type[ext.extraction_class].append(ext)

    result: list[Extraction] = []

    for etype, exts in by_type.items():
        # Determine primary_text field from registry
        primary_field: str | None = None
        if registry is not None:
            try:
                ed = registry.entity_def(etype)
                primary_field = ed.primary_text
            except (KeyError, AttributeError):
                pass

        # Build all alias maps
        abbr_map = build_abbreviation_map(exts, primary_field)
        vocab_syns = build_vocabulary_synonyms(exts, registry)

        # Merge abbreviation and vocabulary maps
        all_aliases: dict[str, str] = {}
        all_aliases.update(vocab_syns)  # vocab synonyms take priority
        all_aliases.update(abbr_map)

        # Group by canonical identity
        groups: dict[str, list[Extraction]] = defaultdict(list)
        ungrouped: list[Extraction] = []

        for ext in exts:
            key = identity_key(ext, primary_field)
            if not key:
                ungrouped.append(ext)
                continue

            # Resolve aliases
            resolved_key = all_aliases.get(key, key)
            groups[resolved_key].append(ext)

        # Fuzzy merge: check if any group keys are fuzzy matches
        merged_keys: set[str] = set()
        key_list = list(groups.keys())
        for i in range(len(key_list)):
            if key_list[i] in merged_keys:
                continue
            for j in range(i + 1, len(key_list)):
                if key_list[j] in merged_keys:
                    continue
                if is_fuzzy_match(key_list[i], key_list[j], max_dist=2):
                    # Merge group j into group i
                    groups[key_list[i]].extend(groups[key_list[j]])
                    merged_keys.add(key_list[j])

        # Remove merged-away groups
        for k in merged_keys:
            del groups[k]

        # Merge each group
        for key, group in groups.items():
            merged = _merge_extraction_group(group)
            result.append(merged)

        # Add ungrouped as-is
        result.extend(ungrouped)

    return result


# ---------------------------------------------------------------------------
# Group merging
# ---------------------------------------------------------------------------


def _merge_extraction_group(group: list[Extraction]) -> Extraction:
    """Merge a group of coreferent extractions into one.

    - evidence_text: concatenated with `` <|> ``, duplicates removed.
    - source_location: concatenated with `` <|> ``, duplicates removed.
    - Attributes: first non-empty value wins for each key.
    """
    if len(group) == 1:
        return group[0]

    base = group[0]

    # Merge evidence_text
    evidence_parts = _collect_unique_parts(
        [getattr(e, "evidence_text", "") for e in group]
    )
    base.evidence_text = " <|> ".join(evidence_parts)

    # Merge source_location
    location_parts = _collect_unique_parts(
        [getattr(e, "source_location", "") for e in group]
    )
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


def _collect_unique_parts(texts: list[str]) -> list[str]:
    """Collect unique non-empty text parts, removing substring duplicates."""
    seen: list[str] = []
    for t in texts:
        t = t.strip()
        if not t:
            continue
        is_dup = False
        for i, s in enumerate(seen):
            if t in s:
                is_dup = True
                break
            if s in t:
                seen[i] = t
                is_dup = True
                break
        if not is_dup:
            seen.append(t)
    return seen


# ---------------------------------------------------------------------------
# Evidence text guarantee
# ---------------------------------------------------------------------------


def ensure_evidence(
    extraction: Extraction,
    document_text: str,
    context_chars: int = 200,
) -> Extraction:
    """Guarantee that *extraction* has evidence_text.

    Strategy (in order):
    1. If evidence exists and contains extraction_text → keep as-is.
    2. If evidence exists but doesn't contain extraction_text → keep it
       (alignment found something, even if LLM's synthetic name differs).
    3. If evidence is empty → fall back to ``str.find()`` on document_text.
    4. If str.find also fails → try finding individual words from extraction_text
       to locate a nearby passage.

    Returns the same extraction (mutated in-place).
    """
    txt = extraction.extraction_text.strip()
    if not txt or not document_text:
        return extraction

    existing_ev = getattr(extraction, "evidence_text", "") or ""

    # Case 1: evidence already contains extraction_text → perfect
    if txt.lower() in existing_ev.lower():
        return extraction

    # Case 2: evidence exists from alignment, keep it even if synthetic name
    if existing_ev.strip():
        return extraction

    # Case 3: no evidence at all → try str.find on full name
    pos = document_text.lower().find(txt.lower())
    if pos >= 0:
        start = max(0, pos - context_chars)
        end = min(len(document_text), pos + len(txt) + context_chars)
        extraction.evidence_text = document_text[start:end].strip()
        if not getattr(extraction, "source_location", ""):
            extraction.source_location = f"doc@{pos}"
        return extraction

    # Case 4: even full name not found → try individual significant words
    # Split on spaces, underscores, hyphens to extract component words
    word_candidates = re.split(r"[\s_\-]+", txt.lower())
    words = [w for w in word_candidates if len(w) > 3 and w not in
             ("with", "from", "that", "this", "than", "then", "were", "been")]
    if words:
        # Find the word that appears earliest in the document
        best_pos = len(document_text)
        for w in words[:3]:  # try first 3 significant words
            p = document_text.lower().find(w)
            if 0 <= p < best_pos:
                best_pos = p
        if best_pos < len(document_text):
            start = max(0, best_pos - context_chars)
            end = min(len(document_text), best_pos + context_chars)
            extraction.evidence_text = document_text[start:end].strip()
            if not getattr(extraction, "source_location", ""):
                extraction.source_location = f"doc@{best_pos}"

    return extraction


def ensure_evidence_batch(
    extractions: list[Extraction],
    document_text: str,
    context_chars: int = 200,
) -> list[Extraction]:
    """Apply :func:`ensure_evidence` to all extractions."""
    for ext in extractions:
        ensure_evidence(ext, document_text, context_chars)
    return extractions


# ---------------------------------------------------------------------------
# Evidence text cleanup
# ---------------------------------------------------------------------------


def clean_evidence_text(text: str) -> str:
    """Clean evidence text for CSV output.

    - Strip leading partial words (token-boundary fragments)
    - Remove newlines and tabs
    - Collapse multiple spaces
    """
    if not text:
        return ""

    t = text.strip()

    # Leading fragment: starts mid-word (lowercase, short, not a common word)
    first_space = t.find(" ")
    if first_space > 0:
        first_token = t[:first_space]
        _COMMON_STARTS = frozenset({
            "the", "a", "an", "in", "on", "at", "to", "of", "for", "with",
            "from", "by", "as", "is", "was", "were", "are", "this", "that",
            "these", "those", "it", "we", "they", "no", "not", "our", "their",
            "and", "or", "but", "if", "so", "be", "has", "had", "been", "can",
            "may", "will", "would", "could", "should", "also", "then", "than",
        })
        if first_token.islower() and len(first_token) <= 6 and first_token not in _COMMON_STARTS:
            t = t[first_space + 1:]

    t = re.sub(r"\s+", " ", t)
    return t.strip()


def clean_evidence_batch(extractions: list[Extraction]) -> None:
    """Apply :func:`clean_evidence_text` to all extractions in-place."""
    for ext in extractions:
        ev = getattr(ext, "evidence_text", "")
        if ev:
            ext.evidence_text = clean_evidence_text(ev)
