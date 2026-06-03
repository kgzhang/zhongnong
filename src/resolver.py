"""Resolver — parses LLM output into Extraction objects and aligns them to source text.

Port of the original langextract resolver with full difflib-based exact matching
and LCS-based fuzzy alignment with coverage/density gates.
"""

from __future__ import annotations

import collections
import difflib
import functools
import itertools
import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, NamedTuple

from src.data import AlignmentStatus, CharInterval, Extraction
from src.format_handler import FormatHandler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FUZZY_ALIGNMENT_MIN_THRESHOLD = 0.75
_FUZZY_ALIGNMENT_MIN_DENSITY = 1.0 / 3.0
DEFAULT_INDEX_SUFFIX = "_index"

# Unicode unit separator used as delimiter between extraction texts during alignment.
_DELIM = "\x1f"


# ---------------------------------------------------------------------------
# Fuzzy alignment helpers
# ---------------------------------------------------------------------------


class LcsSpan(NamedTuple):
    """Result of _best_lcs_span: matched token count and source span."""

    matches: int
    start: int
    end: int

    @property
    def span_len(self) -> int:
        return 0 if self.start < 0 else self.end - self.start + 1


_NO_MATCH = LcsSpan(matches=0, start=-1, end=-1)


def _tokenize_with_lowercase(text: str, tokenizer_inst=None):
    """Yield lowercase token text for each token produced by *tokenizer_inst*."""
    from src.tokenizer import RegexTokenizer

    if tokenizer_inst is None:
        tokenizer_inst = RegexTokenizer()
    tt = tokenizer_inst.tokenize(text)
    for token in tt.tokens:
        yield tt.text[token.char_interval.start_pos : token.char_interval.end_pos].lower()


@functools.lru_cache(maxsize=10000)
def _normalize_token(token: str) -> str:
    """Lowercases and applies light pluralisation stemming."""
    token = token.lower()
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token


def _best_lcs_spans(
    source: Sequence[str],
    extraction: Sequence[str],
) -> dict[int, LcsSpan]:
    """Finds the tightest source span for each achievable match count.

    Runs an O(n * m^2) time / O(m^2) memory DP with rolling rows on the
    i dimension. For each (i, j, k), tracks the latest source index s
    such that source[s..i) contains at least k tokens of extraction[0..j)
    as a subsequence.

    Args:
        source: Normalized source tokens.
        extraction: Normalized extraction tokens.

    Returns:
        Dict mapping each achievable match count k (1..m) to its tightest
        LcsSpan. Empty dict if no matches found.
    """
    n = len(source)
    m = len(extraction)
    if n == 0 or m == 0:
        return {}

    prev_row = [[-1] * (m + 1) for _ in range(m + 1)]
    curr_row = [[-1] * (m + 1) for _ in range(m + 1)]
    for j in range(m + 1):
        prev_row[j][0] = 0

    best_per_k: dict[int, LcsSpan] = {}

    for i in range(1, n + 1):
        src_tok = source[i - 1]
        curr_row[0][0] = i
        for k in range(1, m + 1):
            curr_row[0][k] = -1

        for j in range(1, m + 1):
            curr_row[j][0] = i
            matches_here = src_tok == extraction[j - 1]
            for k in range(1, m + 1):
                skip_source = prev_row[j][k]
                skip_extraction = curr_row[j - 1][k]
                best = skip_source if skip_source > skip_extraction else skip_extraction
                if matches_here:
                    if k == 1:
                        match_cand = i - 1
                    else:
                        match_cand = prev_row[j - 1][k - 1]
                    best = max(best, match_cand)
                curr_row[j][k] = best

        end = i - 1
        for k in range(1, m + 1):
            s = curr_row[m][k]
            if s < 0:
                continue
            existing = best_per_k.get(k)
            if existing is None:
                best_per_k[k] = LcsSpan(matches=k, start=s, end=end)
                continue
            new_len = end - s + 1
            cur_len = existing.span_len
            if new_len < cur_len or (new_len == cur_len and s < existing.start):
                best_per_k[k] = LcsSpan(matches=k, start=s, end=end)

        prev_row, curr_row = curr_row, prev_row

    return best_per_k


def _accept_lcs_match(
    span: LcsSpan,
    extraction_len: int,
    threshold: float = _FUZZY_ALIGNMENT_MIN_THRESHOLD,
    min_density: float = _FUZZY_ALIGNMENT_MIN_DENSITY,
) -> bool:
    """Applies coverage and density gates to an LCS result.

    Coverage gate (threshold): did we find enough of the extraction?
    Density gate (min_density): is the match tight enough?
    """
    if span.matches == 0 or extraction_len == 0:
        return False
    needed = math.ceil(extraction_len * threshold)
    if span.span_len <= 0:
        return False
    density = span.matches / span.span_len
    return span.matches >= needed and density >= min_density


# ---------------------------------------------------------------------------
# Abstract resolver
# ---------------------------------------------------------------------------


class AbstractResolver(ABC):
    """Abstract base for resolvers that produce and align Extraction objects."""

    @abstractmethod
    def resolve(self, input_text, **kwargs) -> Sequence[Extraction]:
        """Parse *input_text* and return a sequence of Extraction objects."""

    @abstractmethod
    def align(
        self, extractions, source_text, token_offset, char_offset, **kwargs
    ) -> Iterator[Extraction]:
        """Align *extractions* to *source_text* and yield aligned Extractions."""


# ---------------------------------------------------------------------------
# Concrete resolver
# ---------------------------------------------------------------------------


class Resolver(AbstractResolver):
    """Resolves LLM output into Extraction objects using a FormatHandler.

    Parameters
    ----------
    format_handler:
        The FormatHandler used for parsing.  Defaults to a JSON handler.
    extraction_index_suffix:
        Suffix used to identify index keys inside extraction groups.
        Defaults to ``"_index"``.
    """

    def __init__(
        self,
        format_handler: FormatHandler | None = None,
        extraction_index_suffix: str | None = None,
    ) -> None:
        self.format_handler = format_handler or FormatHandler(use_fences=False)
        self.extraction_index_suffix = (
            extraction_index_suffix or DEFAULT_INDEX_SUFFIX
        )

    # ------------------------------------------------------------------
    # resolve
    # ------------------------------------------------------------------

    def resolve(
        self, input_text: str, suppress_parse_errors: bool = False, **kwargs
    ) -> Sequence[Extraction]:
        """Parse *input_text* and return a list of Extraction objects."""
        try:
            extraction_data = self.format_handler.parse_output(input_text)
        except (ValueError, Exception) as exc:
            if suppress_parse_errors:
                return []
            raise ValueError(f"Failed to resolve input text: {exc}") from exc

        return self._extract_ordered_extractions(extraction_data)

    # ------------------------------------------------------------------
    # _extract_ordered_extractions
    # ------------------------------------------------------------------

    def _extract_ordered_extractions(
        self, extraction_data: Sequence[Mapping[str, Any]]
    ) -> Sequence[Extraction]:
        """Build Extraction objects from parsed extraction data."""
        attr_suffix = self.format_handler.attribute_suffix
        idx_suffix = self.extraction_index_suffix

        result: list[Extraction] = []
        for group_idx, group in enumerate(extraction_data):
            items = list(group.items())
            if not items:
                continue

            extraction_key: str | None = None
            extraction_value = None
            attr_value: dict[str, Any] | None = None

            for key, value in items:
                if key.endswith(attr_suffix):
                    attr_value = value if isinstance(value, dict) else None
                elif key.endswith(idx_suffix):
                    continue
                else:
                    if extraction_key is None:
                        extraction_key = key
                        extraction_value = value

            if extraction_key is None:
                continue

            if not isinstance(extraction_value, (str, int, float)):
                continue

            ext = Extraction(
                extraction_class=extraction_key,
                extraction_text=str(extraction_value),
                group_index=group_idx,
                attributes=attr_value,
            )
            result.append(ext)

        for i, ext in enumerate(result):
            ext.extraction_index = i

        return result

    # ------------------------------------------------------------------
    # align
    # ------------------------------------------------------------------

    def align(
        self,
        extractions: Sequence[Extraction],
        source_text: str,
        token_offset: int = 0,
        char_offset: int = 0,
        enable_fuzzy_alignment: bool = True,
        **kwargs,
    ) -> Iterator[Extraction]:
        """Align *extractions* to *source_text* via WordAligner."""
        tokenizer_impl = kwargs.get("tokenizer_impl", None)
        aligner = WordAligner()
        grouped = [list(extractions)]
        aligned_groups = aligner.align_extractions(
            grouped,
            source_text,
            token_offset=token_offset,
            char_offset=char_offset,
            enable_fuzzy_alignment=enable_fuzzy_alignment,
            tokenizer_impl=tokenizer_impl,
        )
        for group in aligned_groups:
            yield from group


# ---------------------------------------------------------------------------
# WordAligner — port of the original langextract alignment engine
# ---------------------------------------------------------------------------


class WordAligner:
    """Aligns extraction texts to a source text using difflib.SequenceMatcher.

    Uses exact matching first (via difflib), then falls back to LCS-based
    fuzzy alignment with coverage and density gates.  Properly handles
    multi-extraction alignment via delimiter-based token joining, matching
    the original langextract behaviour.
    """

    def align_extractions(
        self,
        extraction_groups: list[list[Extraction]],
        source_text: str,
        token_offset: int = 0,
        char_offset: int = 0,
        delim: str = _DELIM,
        enable_fuzzy_alignment: bool = True,
        tokenizer_impl=None,
    ) -> list[list[Extraction]]:
        """Align extractions in *extraction_groups* to *source_text*.

        Parameters
        ----------
        extraction_groups:
            List of lists of Extraction objects. Each inner list is aligned
            as a group.
        source_text:
            The original source text to align against.
        token_offset:
            Offset added to token indices in resulting intervals.
        char_offset:
            Offset added to character positions in resulting intervals.
        delim:
            Delimiter used to join extraction texts for tokenization.
        enable_fuzzy_alignment:
            Whether to attempt fuzzy matching after exact matching.
        tokenizer_impl:
            Tokenizer instance. Defaults to RegexTokenizer if not provided.

        Returns
        -------
        list[list[Extraction]]
            Same structure as *extraction_groups* with alignment metadata set.
        """
        from src.tokenizer import RegexTokenizer

        if tokenizer_impl is None:
            tokenizer_impl = RegexTokenizer()

        if not extraction_groups:
            return []

        # ---- 1. Tokenize source (lowercased) ----
        source_tokens = list(_tokenize_with_lowercase(source_text, tokenizer_impl))

        # ---- 2. Build extraction token list (joined with delimiter) ----
        all_extractions = list(itertools.chain(*extraction_groups))
        extraction_texts = [ext.extraction_text for ext in all_extractions]

        # Validate delimiter doesn't appear in extraction texts
        for ext in all_extractions:
            if delim in ext.extraction_text:
                raise ValueError(
                    f"Delimiter {delim!r} appears inside extraction text "
                    f"{ext.extraction_text!r}. This would corrupt alignment."
                )

        joined = f" {delim} ".join(extraction_texts)
        extraction_tokens = list(_tokenize_with_lowercase(joined, tokenizer_impl))

        if not source_tokens or not extraction_tokens:
            return extraction_groups

        # ---- 3. Use SequenceMatcher for exact matching ----
        sm = difflib.SequenceMatcher(autojunk=False, a=source_tokens, b=extraction_tokens)

        # Map extraction token positions → (extraction, group_index)
        index_to_extraction_group: dict[int, tuple[Extraction, int]] = {}
        extraction_idx = 0
        for group_idx, group in enumerate(extraction_groups):
            for extraction in group:
                index_to_extraction_group[extraction_idx] = (extraction, group_idx)
                ext_text_tokens = list(
                    _tokenize_with_lowercase(extraction.extraction_text, tokenizer_impl)
                )
                extraction_idx += len(ext_text_tokens) + 1  # +1 for delimiter token

        # ---- 4. Source tokenized text for char-interval computation ----
        source_tt = tokenizer_impl.tokenize(source_text)

        # Track matched extractions
        aligned_extractions: list[Extraction] = []
        exact_matches = 0

        for block in sm.get_matching_blocks():
            i, j, n = block.a, block.b, block.size
            if n == 0:
                continue  # dummy block

            extraction, group_idx = index_to_extraction_group.get(j, (None, None))
            if extraction is None:
                continue

            # Compute token interval
            extraction.token_interval = TokenInterval(
                start_index=i + token_offset,
                end_index=i + n + token_offset,
            )

            # Compute character interval from source tokenized text
            try:
                start_token = source_tt.tokens[i]
                end_token = source_tt.tokens[i + n - 1]
                extraction.char_interval = CharInterval(
                    start_pos=char_offset + start_token.char_interval.start_pos,
                    end_pos=char_offset + end_token.char_interval.end_pos,
                )
            except IndexError:
                extraction.token_interval = None
                extraction.char_interval = None
                extraction.alignment_status = None
                continue

            # Determine alignment status
            ext_text_tokens = list(
                _tokenize_with_lowercase(extraction.extraction_text, tokenizer_impl)
            )
            if len(ext_text_tokens) == n:
                extraction.alignment_status = AlignmentStatus.MATCH_EXACT
                exact_matches += 1
            else:
                # Partial match (extraction longer than matched text)
                extraction.alignment_status = AlignmentStatus.MATCH_LESSER

            aligned_extractions.append(extraction)

        # ---- 5. Fuzzy alignment for unmatched extractions ----
        unaligned = [e for e in all_extractions if e not in aligned_extractions]

        if enable_fuzzy_alignment and unaligned:
            source_tokens_norm = [_normalize_token(t) for t in source_tokens]
            for extraction in unaligned:
                result = self._lcs_fuzzy_align_extraction(
                    extraction,
                    source_tokens_norm,
                    source_tt,
                    token_offset,
                    char_offset,
                    tokenizer_impl=tokenizer_impl,
                )
                if result:
                    aligned_extractions.append(result)

        # ---- 6. Rebuild result groups ----
        result_groups: list[list[Extraction]] = [[] for _ in extraction_groups]
        for extraction, group_idx in index_to_extraction_group.values():
            result_groups[group_idx].append(extraction)

        return result_groups

    # ------------------------------------------------------------------
    # LCS fuzzy alignment
    # ------------------------------------------------------------------

    def _lcs_fuzzy_align_extraction(
        self,
        extraction: Extraction,
        source_tokens_norm: list[str],
        tokenized_text,
        token_offset: int,
        char_offset: int,
        tokenizer_impl=None,
    ) -> Extraction | None:
        """Fuzzy-align an extraction using LCS DP with coverage/density gates.

        Args:
            extraction: The extraction to align.
            source_tokens_norm: Pre-normalized source tokens.
            tokenized_text: The tokenized source text.
            token_offset: Token offset of the current chunk.
            char_offset: Character offset of the current chunk.
            tokenizer_impl: Optional tokenizer instance.

        Returns:
            The aligned extraction on success, None otherwise.
        """
        extraction_tokens = list(
            _tokenize_with_lowercase(extraction.extraction_text, tokenizer_impl)
        )
        if not extraction_tokens:
            return None

        extraction_tokens_norm = [_normalize_token(t) for t in extraction_tokens]

        # Try spans by decreasing match count: a sparse max-match span may fail
        # the density gate while a denser sub-match span still passes coverage.
        spans = _best_lcs_spans(source_tokens_norm, extraction_tokens_norm)
        accepted: LcsSpan | None = None
        for k in sorted(spans.keys(), reverse=True):
            candidate = spans[k]
            if _accept_lcs_match(
                candidate,
                len(extraction_tokens_norm),
                threshold=_FUZZY_ALIGNMENT_MIN_THRESHOLD,
                min_density=_FUZZY_ALIGNMENT_MIN_DENSITY,
            ):
                accepted = candidate
                break

        if accepted is None:
            return None

        extraction.token_interval = TokenInterval(
            start_index=accepted.start + token_offset,
            end_index=accepted.end + 1 + token_offset,
        )
        start_token = tokenized_text.tokens[accepted.start]
        end_token = tokenized_text.tokens[accepted.end]
        extraction.char_interval = CharInterval(
            start_pos=char_offset + start_token.char_interval.start_pos,
            end_pos=char_offset + end_token.char_interval.end_pos,
        )
        extraction.alignment_status = AlignmentStatus.MATCH_FUZZY
        return extraction


# Import TokenInterval for type annotations in WordAligner
from src.tokenizer import TokenInterval
