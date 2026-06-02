"""Tokenization utilities for alignment and sentence boundary detection.

Provides regex-based and Unicode-aware tokenizers, along with convenience
functions for reconstructing text from token intervals and finding sentence
ranges within token sequences.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence, Set
from dataclasses import dataclass, field
from enum import IntEnum

import regex


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CharInterval:
    """Character interval within the original text (half-open)."""

    start_pos: int
    end_pos: int


@dataclass(slots=True)
class TokenInterval:
    """Token interval (half-open: start_index inclusive, end_index exclusive)."""

    start_index: int = 0
    end_index: int = 0


class TokenType(IntEnum):
    """Classification of a token."""

    WORD = 0
    NUMBER = 1
    PUNCTUATION = 2


@dataclass(slots=True)
class Token:
    """A single token with its type, character span, and metadata."""

    index: int
    token_type: TokenType
    char_interval: CharInterval
    first_token_after_newline: bool = False


@dataclass
class TokenizedText:
    """Result of tokenizing a text string."""

    text: str
    tokens: list[Token]


# ---------------------------------------------------------------------------
# Abstract tokenizer
# ---------------------------------------------------------------------------


class Tokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def tokenize(self, text: str) -> TokenizedText:
        """Tokenize *text* and return a TokenizedText result."""

    def __call__(self, text: str) -> TokenizedText:
        return self.tokenize(text)


# ---------------------------------------------------------------------------
# Regex-based tokenizer
# ---------------------------------------------------------------------------


class RegexTokenizer(Tokenizer):
    """Tokenizes text using regular-expression patterns.

    * Letters (Unicode-aware word characters excluding digits/underscore)
    * Digits
    * Punctuation / symbols (runs of non-word, non-whitespace characters,
      including underscore runs)
    """

    _LETTERS_PATTERN = r"[^\W\d_]+"
    _DIGITS_PATTERN = r"\d+"
    _SYMBOLS_PATTERN = r"([^\w\s]|_)\1*"

    _TOKEN_PATTERN = regex.compile(
        rf"{_LETTERS_PATTERN}|{_DIGITS_PATTERN}|{_SYMBOLS_PATTERN}",
        regex.UNICODE,
    )

    _WORD_PATTERN = regex.compile(
        rf"(?:{_LETTERS_PATTERN}|{_DIGITS_PATTERN})\Z",
        regex.UNICODE,
    )

    def tokenize(self, text: str) -> TokenizedText:
        tokens: list[Token] = []
        prev_end = 0
        for idx, match in enumerate(self._TOKEN_PATTERN.finditer(text)):
            start, end = match.start(), match.end()
            # Determine if there was a newline between the previous token
            # (or the start of text) and this token's start.
            gap = text[prev_end:start]
            first_token_after_newline = bool("\n" in gap or "\r" in gap)
            prev_end = end

            token_text = match.group(0)
            if self._WORD_PATTERN.match(token_text):
                token_type = TokenType.WORD if regex.match(
                    self._LETTERS_PATTERN, token_text
                ) else TokenType.NUMBER
            else:
                token_type = TokenType.PUNCTUATION

            token = Token(
                index=idx,
                token_type=token_type,
                char_interval=CharInterval(start_pos=start, end_pos=end),
                first_token_after_newline=first_token_after_newline,
            )
            tokens.append(token)

        return TokenizedText(text=text, tokens=tokens)


# ---------------------------------------------------------------------------
# Unicode-aware (grapheme-cluster) tokenizer
# ---------------------------------------------------------------------------


class UnicodeTokenizer(Tokenizer):
    """Tokenizes text by grapheme clusters (``\\X``) — one token per cluster.

    Classification:
    * ``str.isalpha()`` → WORD
    * ``str.isdigit()`` → NUMBER
    * otherwise        → PUNCTUATION
    """

    _GRAPHEME_PATTERN = regex.compile(r"\X")

    def tokenize(self, text: str) -> TokenizedText:
        tokens: list[Token] = []
        prev_end = 0
        for idx, match in enumerate(self._GRAPHEME_PATTERN.finditer(text)):
            start, end = match.start(), match.end()
            gap = text[prev_end:start]
            first_token_after_newline = bool("\n" in gap or "\r" in gap)
            prev_end = end

            cluster_text = match.group(0)
            if cluster_text.isalpha():
                token_type = TokenType.WORD
            elif cluster_text.isdigit():
                token_type = TokenType.NUMBER
            else:
                token_type = TokenType.PUNCTUATION

            token = Token(
                index=idx,
                token_type=token_type,
                char_interval=CharInterval(start_pos=start, end_pos=end),
                first_token_after_newline=first_token_after_newline,
            )
            tokens.append(token)

        return TokenizedText(text=text, tokens=tokens)


# ---------------------------------------------------------------------------
# Convenience / helpers
# ---------------------------------------------------------------------------


def tokenize(text: str, tokenizer: Tokenizer | None = None) -> TokenizedText:
    """Convenience function — tokenize *text* with *tokenizer*.

    Defaults to :class:`RegexTokenizer`.
    """
    if tokenizer is None:
        tokenizer = RegexTokenizer()
    return tokenizer.tokenize(text)


def tokens_text(tokenized_text: TokenizedText, token_interval: TokenInterval) -> str:
    """Reconstruct the substring spanned by *token_interval*.

    Returns ``""`` for an empty interval.  Raises :class:`ValueError` when
    *token_interval* refers to indices outside the token list.
    """
    start = token_interval.start_index
    end = token_interval.end_index

    if start < 0 or end > len(tokenized_text.tokens) or start > end:
        raise ValueError(
            f"Invalid token interval {token_interval} for tokenized text "
            f"with {len(tokenized_text.tokens)} tokens"
        )

    if start == end:
        return ""

    first = tokenized_text.tokens[start].char_interval.start_pos
    last = tokenized_text.tokens[end - 1].char_interval.end_pos
    return tokenized_text.text[first:last]


def _is_end_of_sentence_token(
    text: str,
    tokens: Sequence[Token],
    idx: int,
    known_abbreviations: frozenset[str],
) -> bool:
    """Return True when the token at *idx* is end-of-sentence punctuation.

    The check handles:
    * Abbreviations — a period following a known abbreviation is *not*
      end-of-sentence.
    * Ellipsis-like sequences are not treated as sentence ends.
    """
    token = tokens[idx]
    if token.token_type != TokenType.PUNCTUATION:
        return False

    punct_text = text[token.char_interval.start_pos : token.char_interval.end_pos]

    # Only consider single sentence-ending characters.
    if punct_text not in {".", "?", "!", "。", "！", "？", "।"}:
        return False

    # If the period follows a known abbreviation, it is not end-of-sentence.
    # The abbreviation set stores the form *with* the period (e.g. "Dr."),
    # but the tokenizer splits "Dr." into WORD "Dr" + PUNCTUATION ".",
    # so we reconstruct the full token span to match.
    if punct_text == "." and idx > 0:
        prev = tokens[idx - 1]
        prev_text = text[prev.char_interval.start_pos : prev.char_interval.end_pos]
        if prev_text in known_abbreviations or (prev_text + ".") in known_abbreviations:
            return False

    return True


def _is_sentence_break_after_newline(text: str, tokens: Sequence[Token], idx: int) -> bool:
    """Return True when a newline + uppercase letter suggests a sentence break."""
    token = tokens[idx]
    if not token.first_token_after_newline:
        return False

    # Check if the next token starts with an uppercase letter.
    if idx + 1 < len(tokens):
        nxt = tokens[idx + 1]
        nxt_text = text[nxt.char_interval.start_pos : nxt.char_interval.end_pos]
        if nxt_text and nxt_text[0].isupper():
            return True

    return False


_DEFAULT_ABBREVIATIONS = frozenset({
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.",
})


def find_sentence_range(
    text: str,
    tokens: Sequence[Token],
    start_token_index: int,
    known_abbreviations: frozenset[str] = _DEFAULT_ABBREVIATIONS,
) -> TokenInterval:
    """Find the sentence range beginning at *start_token_index*.

    The function scans forward from *start_token_index*, looking for
    end-of-sentence punctuation (``.?!。！？।``) that is *not* a known
    abbreviation, or for a newline + uppercase-letter break.

    Returns a :class:`TokenInterval` whose ``end_index`` points one past
    the last token of the sentence (i.e. it is the exclusive end).
    """
    if not tokens:
        return TokenInterval(start_index=0, end_index=0)

    start = start_token_index

    for idx in range(start_token_index, len(tokens)):
        if _is_end_of_sentence_token(text, tokens, idx, known_abbreviations):
            # Include the punctuation token in the sentence.
            return TokenInterval(start_index=start, end_index=idx + 1)

        if _is_sentence_break_after_newline(text, tokens, idx):
            # The break occurs *before* this token — the sentence ends
            # at the previous token.
            return TokenInterval(start_index=start, end_index=idx)

    # No explicit end found — sentence runs to the end of the token list.
    return TokenInterval(start_index=start, end_index=len(tokens))
