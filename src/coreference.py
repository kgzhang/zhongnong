"""Coreference Resolution Service.

Multi-strategy entity resolution:
  1. Exact normalization (case, hyphens, whitespace)
  2. Fuzzy matching (Levenshtein distance ≤ 2)
  3. Abbreviation ↔ full-name resolution
  4. Vocabulary-based alias resolution (via TSV synonym lists)
  5. Within-type abbreviation/substring matching (e.g. "CON" → "Control")

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
    """Build synonym map from entity vocabulary bindings.

    For each extraction whose entity type has a vocabulary binding, look up
    the vocabulary to find:
    - Other names in the same TSV cell (comma-separated synonyms)
    - Standard name -> all alternative spellings

    Returns ``{synonym_norm: canonical_norm}``.
    """
    synonyms: dict[str, str] = {}

    for ext in extractions:
        # Only run for entity types that have a vocabulary binding
        if registry is None:
            continue
        try:
            vocab = registry.vocabulary(ext.extraction_class)
        except (KeyError, AttributeError):
            vocab = None
        if vocab is None:
            continue

        canonical = normalize_name(ext.extraction_text)
        if not canonical:
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
# Within-type abbreviation/substring resolution (Strategy 5)
# ---------------------------------------------------------------------------


def _resolve_within_type_coref(
    extractions: list[Extraction],
    group_key: str,
    groups: dict[str, list[Extraction]],
) -> dict[str, list[Extraction]]:
    """Within-type abbreviation/substring matching.

    For entity types where one extraction's text is a substring or
    abbreviation of another's, merge them (keeping the longer/fuller name).

    Examples:
    - "CON" vs "Control" → "CON" is ≤5 uppercase chars, matches "Control"
    - "ADG" vs "Average Daily Gain" → abbreviation resolution for Indicator

    Returns updated groups dict with merged groups.
    """
    if len(groups) <= 1:
        return groups

    merged_keys: set[str] = set()
    key_list = list(groups.keys())

    for i in range(len(key_list)):
        if key_list[i] in merged_keys:
            continue
        ki = key_list[i]
        group_i = groups[ki]
        for j in range(i + 1, len(key_list)):
            if key_list[j] in merged_keys:
                continue
            kj = key_list[j]
            group_j = groups[kj]

            if _same_entity_via_substring(ki, kj, group_i, group_j):
                # Merge shorter one into longer one
                if len(ki) >= len(kj):
                    groups[ki].extend(group_j)
                    merged_keys.add(kj)
                else:
                    groups[kj].extend(group_i)
                    merged_keys.add(ki)
                    # Break inner loop if we just merged ki away
                    break

    # Remove merged-away groups
    for k in merged_keys:
        del groups[k]

    return groups


def _same_entity_via_substring(
    key_a: str,
    key_b: str,
    group_a: list[Extraction],
    group_b: list[Extraction],
) -> bool:
    """Determine if two group keys represent the same entity via substring/abbreviation.

    1. If one key is a substring of the other (≥50% overlap) → match.
    2. If one extraction_text is short (≤5 uppercase chars) and matches the
       first letters of a longer extraction_text → abbreviation match.
    3. If both keys look like abbreviations (≤5 chars, uppercase) → no match
       (different abbreviations).
    """
    # Rule 1: Substring containment (one is substring of other)
    if len(key_a) <= len(key_b):
        shorter, longer = key_a, key_b
    else:
        shorter, longer = key_b, key_a

    if longer and shorter:
        # Short is fully contained in long
        if shorter in longer:
            # Check overlap ratio to avoid false matches
            ratio = len(shorter) / len(longer)
            if ratio >= 0.6:  # at least 60% overlap
                return True

    # Rule 2: Abbreviation detection
    # Collect original extraction_text values to check for abbreviation patterns
    texts_a = [e.extraction_text for e in group_a]
    texts_b = [e.extraction_text for e in group_b]

    for ta in texts_a:
        for tb in texts_b:
            if _is_abbreviation_pair(ta, tb):
                return True

    # Rule 3: Both look like abbreviations but different → no match
    # (handled by returning False below)

    return False


def _is_abbreviation_pair(a: str, b: str) -> bool:
    """Check if *a* and *b* form an abbreviation ↔ full-name pair.

    One must be short (≤5 chars, uppercase or title case) and the other
    longer.  The short one's letters must match the first letters of the
    longer one's words.
    """
    a = a.strip()
    b = b.strip()
    if not a or not b:
        return False

    len_a, len_b = len(a), len(b)

    # Determine which is the candidate abbreviation
    if len_a <= 5 and len_b > 5 and (a.isupper() or a.istitle()):
        abbr, full = a, b
    elif len_b <= 5 and len_a > 5 and (b.isupper() or b.istitle()):
        abbr, full = b, a
    else:
        # Both short or both long → check substring more carefully
        return False

    # Try first-letter matching: "ADG" should match "Average Daily Gain"
    full_words = full.split()
    if len(full_words) >= len(abbr):
        first_letters = "".join(w[0].upper() for w in full_words if w)
        if first_letters.startswith(abbr.upper()):
            return True
        # Also try: abbr letters match the start of words in order
        abbr_idx = 0
        for w in full_words:
            if abbr_idx < len(abbr) and w.upper().startswith(abbr[abbr_idx].upper()):
                abbr_idx += 1
            if abbr_idx >= len(abbr):
                return True

    # Also check: the short name appears at the start of the long name
    if abbr.lower() in full.lower():
        return True

    return False


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
    4. Fuzzy Levenshtein matching (≤ 2 edits) — skipped for ``dedup_mode: exact`` entities
    5. Within-type abbreviation/substring matching — skipped for ``dedup_mode: exact`` entities

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
        dedup_exact_only: bool = False
        if registry is not None:
            try:
                ed = registry.entity_def(etype)
                primary_field = ed.primary_text
                # Parse dedup_mode from entity notes
                dedup_exact_only = _parse_dedup_mode(ed.notes) == "exact"
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

        if not dedup_exact_only:
            # Fuzzy merge: check if any group keys are fuzzy matches
            merged_keys: set[str] = set()
            key_list = list(groups.keys())
            for i in range(len(key_list)):
                if key_list[i] in merged_keys:
                    continue
                for j in range(i + 1, len(key_list)):
                    if key_list[j] in merged_keys:
                        continue
                    if is_fuzzy_match(key_list[i], key_list[j], max_dist=1):
                        # Merge group j into group i
                        groups[key_list[i]].extend(groups[key_list[j]])
                        merged_keys.add(key_list[j])

            # Remove merged-away groups
            for k in merged_keys:
                del groups[k]

            # Strategy 5: Within-type abbreviation/substring matching
            groups = _resolve_within_type_coref(exts, etype, groups)

        # Merge each group
        for key, group in groups.items():
            merged = _merge_extraction_group(group)
            result.append(merged)

        # Add ungrouped as-is
        result.extend(ungrouped)

    return result


def _parse_dedup_mode(notes: str) -> str:
    """Extract dedup_mode from entity notes.

    Scans for ``dedup_mode: exact`` or ``dedup_mode: fuzzy``.
    Default is ``fuzzy`` (backward compatible).
    """
    if not notes:
        return "fuzzy"
    import re
    for line in notes.splitlines():
        m = re.match(r"^\s*dedup_mode\s*:\s*(exact|fuzzy)\s*$", line.strip(), re.IGNORECASE)
        if m:
            return m.group(1).lower()
    return "fuzzy"


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
    2. If evidence exists but doesn't contain extraction_text → verify
       component words appear in evidence; if not, expand context window
       progressively (400, 800, 1600 chars).
    3. If still not matching → mark for review with a synthetic-name flag.
    4. If evidence is empty → fall back to ``str.find()`` on document_text
       with progressive context expansion.
    5. If str.find also fails → try finding individual words from
       extraction_text to locate a nearby passage.

    Returns the same extraction (mutated in-place).
    """
    txt = extraction.extraction_text.strip()
    if not txt or not document_text:
        return extraction

    existing_ev = getattr(extraction, "evidence_text", "") or ""

    # Case 1: evidence already contains extraction_text → perfect
    if txt.lower() in existing_ev.lower():
        return extraction

    # Case 2: evidence exists from alignment, verify word-level containment
    if existing_ev.strip():
        if _text_components_in_evidence(txt, existing_ev):
            return extraction
        # Evidence doesn't contain meaningful parts of extraction_text.
        # Try progressive context expansion around the alignment position.
        pos = _infer_char_position(extraction)
        if pos is not None and pos >= 0:
            for factor in (2, 4, 8):  # 400, 800, 1600 chars
                expanded = _extract_context(document_text, pos, context_chars * factor)
                if _text_components_in_evidence(txt, expanded):
                    extraction.evidence_text = expanded
                    return extraction
        # Still not found — mark as for-review but keep best-effort evidence
        _flag_synthetic_if_needed(extraction, txt)
        return extraction

    # Case 3: no evidence at all → try str.find on full name with
    # progressive context expansion
    pos = document_text.lower().find(txt.lower())
    if pos >= 0:
        extraction.evidence_text = _extract_context(document_text, pos, context_chars)
        if not getattr(extraction, "source_location", ""):
            extraction.source_location = f"doc@{pos}"
        return extraction

    # Case 4: full name not found → try individual significant words
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
            for factor in (1, 2, 4):  # 200, 400, 800 chars
                expanded = _extract_context(document_text, best_pos, context_chars * factor)
                if _text_components_in_evidence(txt, expanded):
                    extraction.evidence_text = expanded
                    if not getattr(extraction, "source_location", ""):
                        extraction.source_location = f"doc@{best_pos}"
                    return extraction
            # Best-effort fallback with widest window
            extraction.evidence_text = _extract_context(
                document_text, best_pos, context_chars * 8
            )
            if not getattr(extraction, "source_location", ""):
                extraction.source_location = f"doc@{best_pos}"

    # Case 5: nothing found — flag as synthetic
    _flag_synthetic_if_needed(extraction, txt)
    return extraction


def _text_components_in_evidence(name: str, evidence: str) -> bool:
    """Check if significant words from *name* appear in *evidence*.

    Returns True if all significant component words of *name* (words > 2
    chars, not stopwords) appear somewhere in *evidence* (case-insensitive).
    """
    if not evidence:
        return False
    ev_lower = evidence.lower()
    name_lower = name.lower()

    # Full name present → good
    if name_lower in ev_lower:
        return True

    # Check component words
    word_candidates = re.split(r"[\s_\-]+", name_lower)
    significant = [
        w for w in word_candidates
        if len(w) > 2 and w not in ("the", "and", "for", "with", "from",
                                     "that", "this", "than", "then", "were",
                                     "been", "was", "are", "has", "had")
    ]
    if not significant:
        return len(name_lower) > 0 and name_lower in ev_lower

    # At least 60% of significant words must appear
    found = sum(1 for w in significant if w in ev_lower)
    ratio = found / len(significant)
    return ratio >= 0.6


def _infer_char_position(extraction: Extraction) -> int | None:
    """Try to infer a character position from extraction metadata."""
    ci = getattr(extraction, "char_interval", None)
    if ci is not None and ci.start_pos is not None:
        return ci.start_pos
    # Try parsing from source_location
    sl = getattr(extraction, "source_location", "")
    m = re.search(r"doc@(\d+)", sl)
    if m:
        return int(m.group(1))
    return None


def _extract_context(
    document_text: str,
    pos: int,
    context_chars: int,
) -> str:
    """Extract a safe window of text around *pos* from *document_text*."""
    start = max(0, pos - context_chars)
    end = min(len(document_text), pos + context_chars)
    return document_text[start:end].strip()


def _flag_synthetic_if_needed(extraction: Extraction, name: str) -> None:
    """Detect synthetically-constructed names and flag them.

    A name is flagged as synthetic if:
    - It contains underscores (LLM concatenation)
    - It is all lowercase and > 4 words (descriptive, not extracted)
    - It starts with framing phrases
    """
    if "_" in name:
        extraction.source_location = (
            f"{extraction.source_location or ''} [FLAG: synthetic name, contains underscores]"
        ).strip()
        return

    words = name.split()
    if len(words) > 4 and name.islower():
        extraction.source_location = (
            f"{extraction.source_location or ''} [FLAG: possible synthetic name, >4 lowercase words]"
        ).strip()
        return

    framing = (
        "the study", "this experiment", "this research", "our study",
        "the paper", "this paper", "the trial", "this trial",
    )
    low = name.lower()
    for f in framing:
        if low.startswith(f) or low.endswith(f):
            extraction.source_location = (
                f"{extraction.source_location or ''} [FLAG: framing phrase, possible synthetic]"
            ).strip()
            return


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
# Evidence trimming
# ---------------------------------------------------------------------------


def _trim_evidence(evidence: str, extraction_text: str, context_words: int = 10) -> str:
    """Trim evidence: first try full sentence, fall back to ±context_words."""
    if not evidence or not extraction_text:
        return evidence
    pos = evidence.lower().find(extraction_text.lower())
    if pos < 0:
        return evidence

    # Strategy 1: Try sentence boundaries — find the sentence containing the extraction
    sentences = re.split(r'(?<=[.!?])\s+', evidence)
    for sent in sentences:
        if extraction_text.lower() in sent.lower():
            if len(sent) <= 600:
                return sent.strip()
            break  # sentence found but too long, fall through to Strategy 2

    # Strategy 2: Word-based trimming (fallback)
    words = evidence.split()
    before = evidence[:pos]
    word_pos = len(before.split())
    start_word = max(0, word_pos - context_words)
    end_word = min(len(words), word_pos + len(extraction_text.split()) + context_words)
    return ' '.join(words[start_word:end_word])


def _apply_evidence_trim(extraction: Extraction, txt: str) -> None:
    """Trim evidence_text: 30-word max, ±10 words around extraction, add ... if truncated."""
    ev = getattr(extraction, "evidence_text", "") or ""
    if not ev or not txt:
        return

    words = ev.split()
    txt_words = txt.split()
    txt_len = len(txt_words)

    # Strategy 1: if total ≤ 30 words, keep as-is
    if len(words) <= 30:
        extraction.evidence_text = ev.strip()
        return

    # Strategy 2: try sentence containing extraction (cap at 30 words)
    sentences = re.split(r'(?<=[.!?])\s+', ev)
    for sent in sentences:
        if txt.lower() in sent.lower():
            sent_words = sent.split()
            if len(sent_words) <= 30:
                extraction.evidence_text = sent.strip()
            else:
                # Sentence is too long — trim to ±10 around extraction
                extraction.evidence_text = _trim_evidence(sent, txt, context_words=10)
                if len(extraction.evidence_text.split()) > 30:
                    extraction.evidence_text = '...' + ' '.join(sent_words[-28:]) + '...'
            return

    # Strategy 3: trim to ±10 words, add ... if truncated
    pos = ev.lower().find(txt.lower())
    if pos < 0:
        # Can't find text — trim to first 30 words
        extraction.evidence_text = ' '.join(words[:30]) + '...'
        return

    before = ev[:pos]
    word_pos = len(before.split())
    start_word = max(0, word_pos - 10)
    end_word = min(len(words), word_pos + txt_len + 10)

    prefix = '...' if start_word > 0 else ''
    suffix = '...' if end_word < len(words) else ''
    trimmed = prefix + ' '.join(words[start_word:end_word]) + suffix
    extraction.evidence_text = trimmed


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
    """Clean + trim evidence text for all extractions in-place."""
    for ext in extractions:
        ev = getattr(ext, "evidence_text", "")
        if not ev:
            continue
        ev = clean_evidence_text(ev)
        # Apply 30-word trim after cleanup
        _apply_evidence_trim(ext, ext.extraction_text)

