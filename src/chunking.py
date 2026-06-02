"""Chunking system for breaking tokenized text into sentence-aligned chunks.

Faithful to the langextract chunking design.  Provides chunk iterators,
sentence iterators, and batching utilities.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from src.data import CharInterval, Document
from src.tokenizer import (
    RegexTokenizer,
    TokenInterval,
    TokenizedText,
    Tokenizer,
    find_sentence_range,
    tokens_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_token_interval(start_index: int, end_index: int) -> TokenInterval:
    """Create a validated TokenInterval.

    Raises
    ------
    ValueError
        If *start_index* is negative or if *start_index* >= *end_index*.
    """
    if start_index < 0:
        raise ValueError(f"start_index must be >= 0, got {start_index}")
    if start_index >= end_index:
        raise ValueError(
            f"start_index ({start_index}) must be < end_index ({end_index})"
        )
    return TokenInterval(start_index=start_index, end_index=end_index)


def get_token_interval_text(
    tokenized_text: TokenizedText, token_interval: TokenInterval
) -> str:
    """Extract the substring spanned by *token_interval*.

    Delegates directly to :func:`src.tokenizer.tokens_text`.
    """
    return tokens_text(tokenized_text, token_interval)


def get_char_interval(
    tokenized_text: TokenizedText, token_interval: TokenInterval
) -> CharInterval:
    """Map a *token_interval* to a :class:`CharInterval`.

    The returned interval covers the full character span from the first token's
    start to the last token's end.
    """
    start = token_interval.start_index
    end = token_interval.end_index

    if start < 0 or end > len(tokenized_text.tokens) or start > end:
        raise ValueError(
            f"Invalid token interval {token_interval} for tokenized text "
            f"with {len(tokenized_text.tokens)} tokens"
        )

    if start == end:
        return CharInterval(start_pos=0, end_pos=0)

    first_token = tokenized_text.tokens[start]
    last_token = tokenized_text.tokens[end - 1]
    return CharInterval(
        start_pos=first_token.char_interval.start_pos,
        end_pos=last_token.char_interval.end_pos,
    )


def make_batches_of_textchunk(
    chunk_iter: Iterable[TextChunk], batch_length: int
) -> Iterable[Sequence[TextChunk]]:
    """Group *chunk_iter* into batches of *batch_length*.

    The last batch may be shorter than *batch_length*.
    """
    batch: list[TextChunk] = []
    for chunk in chunk_iter:
        batch.append(chunk)
        if len(batch) == batch_length:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# TextChunk
# ---------------------------------------------------------------------------


@dataclass
class TextChunk:
    """A chunk of text spanning a token interval, optionally tied to a Document.

    Lazy properties
    ---------------
    chunk_text
        The raw text of this chunk (from the document's tokenized text).
    sanitized_chunk_text
        Stripped/cleaned version of *chunk_text*.
    char_interval
        The character interval for this chunk.
    document_id
        The document's ID (or ``"unknown"`` if no document).
    document_text
        The full document text.
    additional_context
        The document's additional context (or ``""``).
    """

    token_interval: TokenInterval
    document: Document | None = None

    # ------------------------------------------------------------------
    # Lazy properties
    # ------------------------------------------------------------------

    @cached_property
    def chunk_text(self) -> str:
        """Raw text spanned by *token_interval* on the document."""
        if self.document is None or not hasattr(self.document, "tokenized_text"):
            return ""
        return get_token_interval_text(self.document.tokenized_text, self.token_interval)  # type: ignore[arg-type]

    @cached_property
    def sanitized_chunk_text(self) -> str:
        """Stripped/cleaned version of :attr:`chunk_text`."""
        return self.chunk_text.strip()

    @cached_property
    def char_interval(self) -> CharInterval:
        """Character interval for this chunk."""
        if self.document is None or not hasattr(self.document, "tokenized_text"):
            return CharInterval()
        return get_char_interval(self.document.tokenized_text, self.token_interval)  # type: ignore[arg-type]

    @cached_property
    def document_id(self) -> str:
        """Document ID or ``"unknown"``."""
        if self.document is not None:
            return self.document.document_id
        return "unknown"

    @cached_property
    def document_text(self) -> str:
        """Full document text."""
        if self.document is not None:
            return self.document.text
        return ""

    @cached_property
    def additional_context(self) -> str:
        """Document additional context."""
        if self.document is not None and self.document.additional_context is not None:
            return self.document.additional_context
        return ""


# ---------------------------------------------------------------------------
# SentenceIterator
# ---------------------------------------------------------------------------


class SentenceIterator:
    """Iterate over sentences in a :class:`TokenizedText`.

    Yields :class:`TokenInterval` for each sentence, using
    :func:`find_sentence_range` to detect sentence boundaries.
    """

    def __init__(
        self, tokenized_text: TokenizedText, curr_token_pos: int = 0
    ) -> None:
        self.tokenized_text = tokenized_text
        self.curr_token_pos = curr_token_pos

    def __iter__(self) -> SentenceIterator:
        return self

    def __next__(self) -> TokenInterval:
        if self.curr_token_pos >= len(self.tokenized_text.tokens):
            raise StopIteration

        interval = find_sentence_range(
            self.tokenized_text.text,
            self.tokenized_text.tokens,
            self.curr_token_pos,
        )
        # Guard against infinite loop: if no progress, advance by at least 1 token
        if interval.end_index <= self.curr_token_pos:
            self.curr_token_pos = min(
                self.curr_token_pos + 1,
                len(self.tokenized_text.tokens),
            )
        else:
            self.curr_token_pos = interval.end_index
        return interval


# ---------------------------------------------------------------------------
# ChunkIterator
# ---------------------------------------------------------------------------


class ChunkIterator:
    """Iterate over sentence-aligned chunks that fit within *max_char_buffer*.

    Three cases handled in ``__next__``:

    1. A single token exceeds the buffer → that token is yielded as its own chunk.
    2. A sentence (as a whole) exceeds the buffer → split at newline boundaries.
    3. Multiple sentences fit within the buffer → combined into one chunk.

    Parameters
    ----------
    text_or_tokenized
        Raw text string or already-tokenized :class:`TokenizedText`.
    max_char_buffer
        Maximum number of characters per chunk.
    tokenizer_impl
        Tokenizer implementation (used if *text_or_tokenized* is a string).
    document
        Optional :class:`Document` to attach to produced chunks.
    """

    def __init__(
        self,
        text_or_tokenized: str | TokenizedText | None = None,
        max_char_buffer: int = 8000,
        tokenizer_impl: Tokenizer | None = None,
        document: Document | None = None,
        text: str | None = None,
    ) -> None:
        # Accept both `text_or_tokenized` and `text` keyword argument
        source: str | TokenizedText
        if text is not None:
            source = text
        elif text_or_tokenized is not None:
            source = text_or_tokenized
        else:
            raise TypeError("ChunkIterator requires text or TokenizedText input")
        if tokenizer_impl is None:
            tokenizer_impl = RegexTokenizer()
        if isinstance(source, TokenizedText):
            self.tokenized_text = source
        else:
            self.tokenized_text = tokenizer_impl.tokenize(source)

        self.max_char_buffer = max_char_buffer
        self.document = document or Document(text=self.tokenized_text.text)
        # Monkey-patch tokenized_text onto the Document so TextChunk can
        # lazily compute chunk_text.
        self.document.tokenized_text = self.tokenized_text  # type: ignore[attr-defined]

        self._sentence_iter = SentenceIterator(self.tokenized_text)
        # Intervals that still need to be processed (from splits, etc.)
        self._pending: list[TokenInterval] = []
        self._exhausted: bool = False

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> ChunkIterator:
        return self

    def __next__(self) -> TextChunk:
        return self._build_next_chunk()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokens_exceed_buffer(self, token_interval: TokenInterval) -> bool:
        """Return ``True`` when the text of *token_interval* exceeds the buffer."""
        text = tokens_text(self.tokenized_text, token_interval)
        return len(text) > self.max_char_buffer

    def _split_at_newlines(self, interval: TokenInterval) -> list[TokenInterval]:
        """Split *interval* at newline boundaries into sub-intervals.

        Returns a list where each sub-interval starts at a token that
        follows a newline.  If no newlines are found, returns a single
        sub-interval identical to the input.
        """
        tokens = self.tokenized_text.tokens
        subintervals: list[TokenInterval] = []
        chunk_start = interval.start_index

        for i in range(interval.start_index, interval.end_index):
            token = tokens[i]
            if token.first_token_after_newline and i > chunk_start:
                subintervals.append(
                    TokenInterval(start_index=chunk_start, end_index=i)
                )
                chunk_start = i

        if chunk_start < interval.end_index:
            subintervals.append(
                TokenInterval(start_index=chunk_start, end_index=interval.end_index)
            )

        return subintervals

    def _get_next_interval(self) -> TokenInterval | None:
        """Return the next interval to process, or ``None`` if exhausted."""
        if self._pending:
            return self._pending.pop(0)

        if self._exhausted:
            return None

        try:
            return next(self._sentence_iter)
        except StopIteration:
            self._exhausted = True
            return None

    def _build_next_chunk(self) -> TextChunk:
        """Assemble the next chunk from sentences / pending intervals."""
        chunk_start: int | None = None
        chunk_end: int | None = None

        while True:
            interval = self._get_next_interval()
            if interval is None:
                if chunk_start is not None and chunk_end is not None:
                    return TextChunk(
                        token_interval=TokenInterval(
                            start_index=chunk_start, end_index=chunk_end
                        ),
                        document=self.document,
                    )
                raise StopIteration

            interval_text = tokens_text(self.tokenized_text, interval)

            if chunk_start is None:
                # --- Empty chunk: try adding this interval ---
                if len(interval_text) <= self.max_char_buffer:
                    chunk_start = interval.start_index
                    chunk_end = interval.end_index
                else:
                    # Interval alone exceeds buffer — handle special cases.

                    # Case 1: Single token exceeds buffer
                    single = TokenInterval(
                        start_index=interval.start_index,
                        end_index=interval.start_index + 1,
                    )
                    if self._tokens_exceed_buffer(single):
                        # Yield the huge token; save the rest for later.
                        remaining = TokenInterval(
                            start_index=interval.start_index + 1,
                            end_index=interval.end_index,
                        )
                        if remaining.start_index < remaining.end_index:
                            self._pending.insert(0, remaining)
                        return TextChunk(
                            token_interval=single, document=self.document
                        )

                    # Case 2: Sentence (no huge token) exceeds buffer
                    sub = self._split_at_newlines(interval)
                    if len(sub) == 1 and sub[0].start_index == interval.start_index and sub[0].end_index == interval.end_index:
                        # No newlines to split on — yield the whole interval
                        return TextChunk(
                            token_interval=interval, document=self.document
                        )
                    self._pending = sub + self._pending
                    # Loop to pick up the first sub-interval
                    continue
            else:
                # --- Chunk has content — try adding this interval ---
                candidate_text = self._make_combined_text(
                    chunk_start, chunk_end, interval
                )
                if len(candidate_text) <= self.max_char_buffer:
                    chunk_end = interval.end_index
                else:
                    # Doesn't fit — push interval back and yield current chunk.
                    self._pending.insert(0, interval)
                    return TextChunk(
                        token_interval=TokenInterval(
                            start_index=chunk_start, end_index=chunk_end
                        ),
                        document=self.document,
                    )

    def _make_combined_text(
        self, start: int, current_end: int, new_interval: TokenInterval
    ) -> str:
        """Return the combined text of the current chunk plus *new_interval*."""
        combined = TokenInterval(
            start_index=start,
            end_index=new_interval.end_index,
        )
        return tokens_text(self.tokenized_text, combined)
