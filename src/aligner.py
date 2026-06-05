"""Evidence alignment engine — per-extraction alignment against full document text.

Replaces the three-layer pipeline (WordAligner grouped-join alignment →
derive_evidence → ensure_evidence fallback) with a single per-extraction
function that:

1. **Primary**: ``str.find()`` on full document text (handles whitespace
   normalization automatically since Python's ``str.find`` is case-sensitive
   but we try both exact and whitespace-normalized forms).

2. **Token-level fallback**: When ``str.find`` fails (LLM hallucination,
   spelling variants, abbreviation expansion), tokenizes both the extraction
   text and document text and uses difflib ``SequenceMatcher`` LOCAL to a
   sliding window around candidate positions.

3. **Evidence extraction**: Once the best character interval is found,
   directly slices the source text for evidence (no separate post-pass
   needed). Returns both ``char_interval`` and ``evidence_text`` in one
   call.
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import NamedTuple

from src.data import AlignmentStatus, CharInterval, Extraction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class AlignResult(NamedTuple):
    """Result of aligning one extraction against document text."""

    char_interval: CharInterval | None
    evidence_text: str
    alignment_status: AlignmentStatus | None
    source_location: str  # e.g. "pos@1234"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def align_and_evidence(
    extraction: Extraction,
    document_text: str,
    *,
    context_chars: int = 300,
) -> AlignResult:
    """Align one extraction against *document_text* and derive its evidence.

    This is the **single call** that replaces the old three-step pipeline
    (WordAligner → derive_evidence → ensure_evidence).  Each extraction is
    aligned independently against the full document, avoiding the
    delimiter-join multi-extraction mismatch problem.

    Strategy (in order):
    1. ``str.find`` exact match (case-insensitive) → MATCH_EXACT
    2. ``str.find`` whitespace-normalized match → MATCH_EXACT_WS_NORM
    3. Normalize comparison operators (``<`` ↔ ``less than`` etc.) → MATCH
    4. Token-level sliding-window alignment (local LCS) → MATCH_FUZZY
    5. Word-level word-by-word alignment → MATCH_FUZZY
    6. Nothing found → unaligned (None, "", None)

    Returns an ``AlignResult`` with ``char_interval``, ``evidence_text``,
    ``alignment_status``, and ``source_location``.
    """
    return _align_and_evidence_impl(extraction, document_text, context_chars)


def align_and_evidence_batch(
    extractions: list[Extraction],
    document_text: str,
    *,
    context_chars: int = 300,
) -> list[Extraction]:
    """Apply :func:`align_and_evidence` to every extraction in *extractions*.

    Mutates each extraction in-place (sets ``char_interval``,
    ``evidence_text``, ``alignment_status``, ``source_location``) and
    returns the same list.
    """
    for ext in extractions:
        result = _align_and_evidence_impl(ext, document_text, context_chars)
        ext.char_interval = result.char_interval
        ext.evidence_text = result.evidence_text
        ext.alignment_status = result.alignment_status
        ext.source_location = result.source_location
    return extractions


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _align_and_evidence_impl(
    extraction: Extraction,
    document_text: str,
    context_chars: int,
) -> AlignResult:
    txt = extraction.extraction_text.strip()
    if not txt or not document_text:
        return AlignResult(None, "", None, "")

    doc_lower = document_text.lower()
    txt_lower = txt.lower()

    # ---- Strategy 1: str.find exact (case-insensitive) ----
    pos = doc_lower.find(txt_lower)
    if pos >= 0:
        # Only accept if the match is at a WORD BOUNDARY (not mid-word).
        # Check that characters immediately before/after the match are not
        # alphabetic, preventing e.g. "flavone" matching inside "isoflavones"
        # or "ma" matching inside "inflammatory".
        before_ok = pos == 0 or not document_text[pos - 1].isalpha()
        after_ok = (pos + len(txt) >= len(document_text)
                    or not document_text[pos + len(txt)].isalpha())
        if not before_ok or not after_ok:
            # Mid-word match — try the NEXT occurrence
            next_pos = doc_lower.find(txt_lower, pos + 1)
            while next_pos >= 0:
                n_before_ok = next_pos == 0 or not document_text[next_pos - 1].isalpha()
                n_after_ok = (next_pos + len(txt) >= len(document_text)
                              or not document_text[next_pos + len(txt)].isalpha())
                if n_before_ok and n_after_ok:
                    pos = next_pos
                    before_ok = True
                    after_ok = True
                    break
                next_pos = doc_lower.find(txt_lower, next_pos + 1)
        if before_ok and after_ok:
            return _build_result(document_text, pos, pos + len(txt), context_chars,
                                 AlignmentStatus.MATCH_EXACT)

    # ---- Strategy 2: whitespace-normalized find ----
    txt_compact = re.sub(r"\s+", "", txt_lower)
    # Also strip parenthetical trailing content for matching:
    # "thymol (THY)" should match "thymol (THY, purity >= 99%)"
    # Normalize: collapse all parenthetical content to just the first word
    txt_compact_paren_norm = re.sub(r"\((\w+)[^)]*\)", r"(\1)", txt_compact)
    doc_compact = re.sub(r"\s+", "", doc_lower)
    doc_compact_paren_norm = re.sub(r"\((\w+)[^)]*\)", r"(\1)", doc_compact)

    # Try multiple normalization levels
    pos_found: int = -1
    txt_matched_len: int = 0
    for txt_norm, doc_norm in [
        (txt_compact, doc_compact),
        (txt_compact_paren_norm, doc_compact_paren_norm),
        # Also try stripping parens entirely for substring matching
        (re.sub(r"\([^)]*\)", "", txt_compact), re.sub(r"\([^)]*\)", "", doc_compact)),
    ]:
        if not txt_norm or len(txt_norm) < 3:
            continue
        pos_compact = doc_norm.find(txt_norm)
        if pos_compact >= 0:
            txt_matched_len = len(txt_norm)
            pos_found = pos_compact
            break

    if pos_found >= 0:
        # Map compact position back to original document position
        orig_pos = _compact_to_original(doc_lower, doc_compact, pos_found)
        # End position: count forward txt_matched_len non-whitespace chars
        # from orig_pos to get the end of the matched span
        compact_count = 0
        end_pos = orig_pos
        for i in range(orig_pos, len(doc_lower)):
            if not doc_lower[i].isspace():
                compact_count += 1
                if compact_count == txt_matched_len:
                    end_pos = i + 1
                    break

        # Extend end_pos to include a closing paren if it follows immediately
        # (handles "thymol (THY)" → "thymol (THY, purity >= 99%)")
        if end_pos < len(document_text) and document_text[end_pos - 1] == " ":
            # Check if a closing paren follows shortly after
            scan = document_text[end_pos:end_pos + 30]
            paren_close = scan.find(")")
            if 0 < paren_close < 15:
                # Check that there's no opening paren between here and the close
                if "(" not in scan[:paren_close]:
                    end_pos = end_pos + paren_close + 1

        if orig_pos >= 0 and end_pos > orig_pos:
            return _build_result(document_text, orig_pos, end_pos, context_chars,
                                 AlignmentStatus.MATCH_EXACT)

    # ---- Strategy 3: normalize comparison operators ----
    pos_op = _find_with_operator_normalization(txt_lower, doc_lower)
    if pos_op >= 0:
        return _build_result(document_text, pos_op, pos_op + len(txt), context_chars,
                             AlignmentStatus.MATCH_FUZZY)

    # ---- Strategy 4: token-level sliding-window alignment ----
    result = _token_window_align(txt, document_text, context_chars)
    if result is not None:
        return result

    # ---- Strategy 5: word-level alignment ----
    result = _word_level_align(txt, document_text, context_chars)
    if result is not None:
        return result

    # ---- Nothing found ----
    return AlignResult(None, "", None, "")


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------


def _compact_to_original(original: str, compact: str, compact_pos: int) -> int:
    """Map a position in whitespace-stripped *compact* back to *original*."""
    count = 0
    for i, ch in enumerate(original):
        if not ch.isspace():
            if count == compact_pos:
                return i
            count += 1
    return -1


def _find_with_operator_normalization(txt_lower: str, doc_lower: str) -> int:
    """Try to find extraction text after normalising comparison operators.

    Handles cases like ``"P<0.05"`` vs ``"P less than 0.05"``, or
    ``"p = 0.03"`` vs ``"p=0.03"``.
    """
    # Pattern: letter + optional space + operator + optional space + number
    m = re.match(r"([a-z]+)\s*([<>=≤≥]|less than|greater than)\s*(.+)", txt_lower)
    if not m:
        return -1

    prefix = m.group(1)
    op = m.group(2).strip()
    suffix = m.group(3).strip()

    # Build alternative spellings to search for
    alternatives = []

    # Standard operator forms
    op_map = {"<": ["<", "less than"], ">": [">", "greater than"],
              "=": ["="], "≤": ["≤", "<=", "less than or equal to"],
              "≥": ["≥", ">=", "greater than or equal to"]}

    for alt_op in op_map.get(op, [op]):
        # With and without spaces around operator
        alternatives.append(f"{prefix}{alt_op}{suffix}")
        alternatives.append(f"{prefix} {alt_op} {suffix}")
        alternatives.append(f"{prefix} {alt_op} {suffix}")
        if alt_op in ("<", ">", "=", "≤", "≥"):
            alternatives.append(f"{prefix} {alt_op}{suffix}")
            alternatives.append(f"{prefix}{alt_op} {suffix}")

    for alt in alternatives:
        pos = doc_lower.find(alt)
        if pos >= 0:
            # Found matching P-value — verify by checking surrounding context
            # Make sure we found near a statistical context
            return pos

    return -1


def _token_window_align(
    txt: str, document_text: str, context_chars: int
) -> AlignResult | None:
    """Token-level sliding-window alignment using local SequenceMatcher.

    Tokenizes the extraction text, then scans the document for candidate
    windows where enough extraction tokens appear.  For each candidate,
    runs SequenceMatcher locally to find the best span.
    """
    from src.tokenizer import RegexTokenizer

    tok = RegexTokenizer()
    doc_lower = document_text.lower()
    doc_tt = tok.tokenize(document_text)

    # Tokenize extraction text
    ext_tt = tok.tokenize(txt)
    ext_tokens = [
        document_text[et.char_interval.start_pos:et.char_interval.end_pos].lower()
        for et in ext_tt.tokens
    ]
    if not ext_tokens:
        return None

    # Build a set of significant extraction tokens for fast candidate finding
    significant = [t for t in ext_tokens if len(t) > 1 and t not in
                   ("<", ">", "=", ".", ",", ";", ":", "(", ")", "[", "]", "-")]
    if not significant:
        return None

    # Find candidate positions: where any significant extraction token
    # appears in the document
    candidates: set[int] = set()
    for stok in significant:
        pos = 0
        while True:
            pos = doc_lower.find(stok, pos)
            if pos < 0:
                break
            candidates.add(pos)
            pos += 1

    if not candidates:
        return None

    # For each candidate, extract a window and run SequenceMatcher locally
    best_ratio = 0.0
    best_pos = -1
    best_end = -1

    window_size = max(len(txt) * 6, 200)
    ext_tokens_set = set(ext_tokens)

    for cand_pos in sorted(candidates):
        win_start = max(0, cand_pos - window_size // 2)
        win_end = min(len(document_text), cand_pos + window_size // 2)
        window = document_text[win_start:win_end]

        win_tt = tok.tokenize(window)
        win_tokens = [
            window[wt.char_interval.start_pos:wt.char_interval.end_pos].lower()
            for wt in win_tt.tokens
        ]

        # Quick pre-filter: need overlap of significant tokens
        if not ext_tokens_set & set(win_tokens):
            continue

        sm = difflib.SequenceMatcher(a=win_tokens, b=ext_tokens)
        ratio = sm.ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            # Find the actual matching block
            for block in sm.get_matching_blocks():
                if block.size > 0:
                    # Map window-relative token positions to absolute positions
                    block_start_token = win_tt.tokens[block.a]
                    block_end_token = win_tt.tokens[block.a + block.size - 1]
                    best_pos = win_start + block_start_token.char_interval.start_pos
                    best_end = win_start + block_end_token.char_interval.end_pos
                    break

    if best_ratio >= 0.6 and best_pos >= 0:
        return _build_result(document_text, best_pos, best_end, context_chars,
                             AlignmentStatus.MATCH_FUZZY)

    return None


def _word_level_align(
    txt: str, document_text: str, context_chars: int
) -> AlignResult | None:
    """Last-resort: word-by-word alignment.

    Splits the extraction text into component words, finds their locations
    in the document, and picks the span that covers the most words.
    """
    words = re.split(r"[\s_\-]+", txt.lower())
    words = [w for w in words if len(w) > 2]
    if len(words) < 2:
        return None

    # Find all occurrences of each word
    word_positions: dict[str, list[int]] = {}
    doc_lower = document_text.lower()
    for w in words:
        positions = []
        pos = 0
        while True:
            pos = doc_lower.find(w, pos)
            if pos < 0:
                break
            positions.append(pos)
            pos += 1
        word_positions[w] = positions

    # Find the best cluster: shortest span covering the most words
    best_start = -1
    best_end = -1
    best_score = 0

    for first_word, first_positions in word_positions.items():
        if not first_positions:
            continue
        for fp in first_positions[:5]:  # try first 5 occurrences
            span_start = fp
            span_end = fp + len(first_word)
            covered = 1
            for w in words[1:]:
                if w not in word_positions or not word_positions[w]:
                    continue
                # Find the closest occurrence to span_end
                w_positions = word_positions[w]
                best_w_pos = min(w_positions, key=lambda p: abs(p - span_end))
                if abs(best_w_pos - span_end) < len(txt) * 3:
                    span_end = max(span_end, best_w_pos + len(w))
                    covered += 1

            if covered > best_score:
                best_score = covered
                best_start = span_start
                best_end = span_end

    if best_score >= max(2, len(words) * 0.5) and best_start >= 0:
        return _build_result(document_text, best_start, best_end, context_chars,
                             AlignmentStatus.MATCH_FUZZY)

    return None


def _build_result(
    document_text: str,
    start_pos: int,
    end_pos: int,
    context_chars: int,
    status: AlignmentStatus,
) -> AlignResult:
    """Build an AlignResult from an aligned character interval.

    evidence_text:
        Context window (±*context_chars*) expanded to sentence boundaries.
        May span multiple sentences — provides full context for the match.

    source_location:
        ONLY the single sentence that contains the matched text range.
        Always a strict substring of evidence_text, shorter, used solely
        for locating the entity in the source document.
    """
    ci = CharInterval(start_pos=start_pos, end_pos=end_pos)

    # -- evidence_text: multi-sentence context window --
    raw_start = max(0, start_pos - context_chars)
    raw_end = min(len(document_text), end_pos + context_chars)

    ev_start = _sentence_start(document_text, raw_start)
    ev_end = _sentence_end(document_text, raw_end)
    ev_start = _ltrim(document_text, ev_start, ev_end)
    ev_end = _rtrim(document_text, ev_start, ev_end)
    evidence = document_text[ev_start:ev_end]

    # -- source_location: ONLY the sentence containing the match --
    # Find the sentence that contains start_pos by looking for the
    # nearest sentence boundaries around the match itself.
    sl_start = _sentence_start(document_text, start_pos)
    sl_end = _sentence_end(document_text, end_pos)
    # Constrain to evidence_text bounds (should always be within)
    sl_start = max(ev_start, _ltrim(document_text, sl_start, sl_end))
    sl_end = min(ev_end, _rtrim(document_text, sl_start, sl_end))
    source_loc = document_text[sl_start:sl_end]

    return AlignResult(
        char_interval=ci,
        evidence_text=evidence,
        alignment_status=status,
        source_location=source_loc,
    )


def _sentence_start(text: str, pos: int) -> int:
    """Move *pos* backward to the nearest sentence-start boundary.

    A sentence boundary is defined as: start of text, or a position
    following a sentence-ending punctuation mark (``. ! ?``) plus
    whitespace, or a newline.
    """
    if pos <= 0:
        return 0
    # Search backward for sentence boundary within 500 chars
    search_start = max(0, pos - 500)
    chunk = text[search_start:pos]
    # Sentence-ending punctuation followed by space/newline + capital letter
    # or just a newline
    for i in range(len(chunk) - 1, -1, -1):
        ch = chunk[i]
        if ch == '\n':
            return search_start + i + 1
        if ch in '.!?':
            # Check if followed by space and capital letter or newline
            after = text[search_start + i + 1:search_start + i + 10]
            if after and (after[0] in ' \n\t' or (len(after) >= 2 and after[0].isspace())):
                return search_start + i + 1
    # Fallback: move pos back 200 chars, then forward past whitespace
    fallback = max(0, pos - 200)
    while fallback < pos and text[fallback] in ' \t\n\r':
        fallback += 1
    return fallback


def _ltrim(text: str, start: int, end: int) -> int:
    """Move *start* forward past leading whitespace."""
    while start < end and text[start] in ' \t\n\r':
        start += 1
    return start


def _rtrim(text: str, start: int, end: int) -> int:
    """Move *end* backward past trailing whitespace."""
    while end > start and text[end - 1] in ' \t\n\r':
        end -= 1
    return end


def _sentence_end(text: str, pos: int) -> int:
    """Move *pos* forward to the nearest sentence-end boundary.

    A sentence boundary is defined as: end of text, or a sentence-ending
    punctuation mark (``. ! ?``) followed by space+capital or newline.
    """
    if pos >= len(text):
        return len(text)
    # Search forward for sentence boundary within 500 chars
    search_end = min(len(text), pos + 500)
    chunk = text[pos:search_end]
    for i, ch in enumerate(chunk):
        if ch == '\n':
            return pos + i
        if ch in '.!?':
            after_start = pos + i + 1
            after = text[after_start:after_start + 5]
            if after and (after[0] in ' \n\t' or after[0].isspace()):
                return min(len(text), after_start)
            # Also end at period-space-number pattern (e.g., "P < 0.05")
            if after and after[0].isdigit():
                continue
    return min(len(text), pos + 200)  # fallback: 200 chars forward
