"""Resolver — parses LLM output into Extraction objects and aligns them to source text."""

from __future__ import annotations

import difflib
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

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
# Helpers
# ---------------------------------------------------------------------------


def _tokenize_with_lowercase(text: str, tokenizer_inst):
    """Yield lowercase token text for each token produced by *tokenizer_inst*."""
    tt = tokenizer_inst.tokenize(text)
    for token in tt.tokens:
        yield tt.text[token.char_interval.start_pos : token.char_interval.end_pos].lower()


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
        """Parse *input_text* and return a list of Extraction objects.

        Parameters
        ----------
        input_text:
            Raw LLM output text.
        suppress_parse_errors:
            When True, returns an empty list on parse failure instead of raising.
        """
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
        """Build Extraction objects from parsed extraction data.

        Each item in *extraction_data* is a mapping.  Keys that end with
        *extraction_index_suffix* or the format handler's *attribute_suffix*
        are treated as metadata and skipped as extraction classes.  The first
        (remaining) key becomes the extraction class and its value (which must
        be str, int, or float) becomes the extraction text.  Any key ending in
        ``_attributes`` provides a dict of attributes.
        """
        attr_suffix = self.format_handler.attribute_suffix
        idx_suffix = self.extraction_index_suffix

        result: list[Extraction] = []
        for group_idx, group in enumerate(extraction_data):
            items = list(group.items())
            if not items:
                continue

            # Find the extraction class key and attribute key
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

            # Validate extraction value type
            if not isinstance(extraction_value, (str, int, float)):
                continue

            ext = Extraction(
                extraction_class=extraction_key,
                extraction_text=str(extraction_value),
                group_index=group_idx,
                attributes=attr_value,
            )
            result.append(ext)

        # Assign extraction indices
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
# WordAligner
# ---------------------------------------------------------------------------


class WordAligner:
    """Aligns extraction texts to a source text using difflib.SequenceMatcher."""

    def align_extractions(
        self,
        extraction_groups: list[list[Extraction]],
        source_text: str,
        token_offset: int = 0,
        char_offset: int = 0,
        delim: str = _DELIM,
        enable_fuzzy_alignment: bool = True,
        tokenizer_impl=None,
    ):
        """Align extractions in *extraction_groups* to *source_text*.

        Parameters
        ----------
        extraction_groups:
            List of lists of Extraction objects.  Each inner list is aligned
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
            Tokenizer instance.  Defaults to a RegexTokenizer if not provided.

        Returns
        -------
        list[list[Extraction]]
            Same structure as *extraction_groups* with alignment metadata set.
        """
        from src.tokenizer import RegexTokenizer

        if tokenizer_impl is None:
            tokenizer_impl = RegexTokenizer()

        # Tokenize source (lowercased)
        source_tokens = list(_tokenize_with_lowercase(source_text, tokenizer_impl))

        result_groups: list[list[Extraction]] = []
        for group in extraction_groups:
            if not group:
                result_groups.append(group)
                continue

            # Build extraction token list
            extraction_texts = [ext.extraction_text for ext in group]
            joined = delim.join(extraction_texts)
            ext_tokens = list(_tokenize_with_lowercase(joined, tokenizer_impl))
            # We also need the original (non-lowercased) tokenized text for
            # character interval computation.
            joined_tt = tokenizer_impl.tokenize(joined)
            ext_tokens_original = joined_tt.tokens

            if not source_tokens or not ext_tokens:
                result_groups.append(group)
                continue

            # Use SequenceMatcher with autojunk=False
            sm = difflib.SequenceMatcher(
                autojunk=False, a=source_tokens, b=ext_tokens
            )

            for block in sm.get_matching_blocks():
                a_start, b_start, length = block.a, block.b, block.size
                if length == 0:
                    continue

                # Find which extraction this token range falls into
                ext_idx = self._find_extraction_at_index(
                    group, ext_tokens, b_start, delim, ext_tokens_original
                )
                if ext_idx is None:
                    continue

                extraction = group[ext_idx]

                # Compute token interval in source
                token_start = a_start + token_offset
                token_end = a_start + length + token_offset

                # Compute character interval in source
                # We need the source tokenized text for char intervals
                source_tt = tokenizer_impl.tokenize(source_text)
                source_token_objs = source_tt.tokens
                if a_start < len(source_token_objs) and (a_start + length) <= len(source_token_objs):
                    c_start = source_token_objs[a_start].char_interval.start_pos + char_offset
                    c_end = source_token_objs[a_start + length - 1].char_interval.end_pos + char_offset
                else:
                    c_start = None
                    c_end = None

                if c_start is not None and c_end is not None:
                    extraction.char_interval = CharInterval(
                        start_pos=c_start, end_pos=c_end
                    )

                # Determine alignment status
                extraction.alignment_status = AlignmentStatus.MATCH_EXACT

            # Fuzzy alignment: if no exact match was found, try fuzzy
            if enable_fuzzy_alignment:
                for ext in group:
                    if ext.alignment_status is not None:
                        continue
                    # Simple fuzzy: use difflib get_close_matches
                    ext_lower = ext.extraction_text.lower()
                    matches = difflib.get_close_matches(
                        ext_lower, [source_text.lower()], n=1, cutoff=_FUZZY_ALIGNMENT_MIN_THRESHOLD
                    )
                    if matches:
                        # Try to find the matched substring position
                        pos = source_text.lower().find(ext_lower)
                        if pos >= 0:
                            ext.char_interval = CharInterval(
                                start_pos=pos + char_offset,
                                end_pos=pos + len(ext.extraction_text) + char_offset,
                            )
                            ext.alignment_status = AlignmentStatus.MATCH_FUZZY

            result_groups.append(group)

        return result_groups

    # ------------------------------------------------------------------
    # _find_extraction_at_index
    # ------------------------------------------------------------------

    def _find_extraction_at_index(
        self,
        groups: list[Extraction],
        ext_tokens: list[str],
        j: int,
        delim: str,
        tok: list,
    ) -> int | None:
        """Map token index *j* back to the extraction in *groups*.

        Walks through extraction texts (joined with *delim*) and determines
        which extraction the token at position *j* belongs to.
        """
        texts = [ext.extraction_text for ext in groups]
        cumulative = 0

        for ext_idx, text in enumerate(texts):
            # Tokenize this individual extraction text to count its tokens
            text_lower_tokens = text.lower().split()
            # Actually we should use the pre-tokenized ext_tokens with delimiter
            # Count delimiter tokens as well.
            # Simpler approach: walk the joined text positions
            token_count = len(text_lower_tokens)
            if cumulative <= j < cumulative + token_count:
                return ext_idx
            cumulative += token_count
            # Add delimiter (if not last)
            if ext_idx < len(texts) - 1:
                delim_tokens = delim.lower().split()
                if delim.strip():
                    cumulative += len(delim_tokens)

        return None
