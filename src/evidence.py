"""EvidenceExtractor — derive verbatim evidence from aligned char_interval."""
from src.data import Extraction
from src.tokenizer import TokenizedText, find_sentence_range, tokens_text, TokenInterval


class EvidenceExtractor:
    """Extract verbatim evidence text surrounding an aligned extraction."""

    def extract_evidence(
        self,
        extraction: Extraction,
        document_text: str,
        tokenized_text: TokenizedText,
        context_sentences: int = 2,
    ) -> str:
        """Return the sentence-range evidence text for *extraction*.

        Uses *extraction.char_interval* to locate the token position in
        *tokenized_text*, then expands outward by *context_sentences*.
        Returns ``""`` when the extraction has no char_interval or when
        an error occurs during lookup.
        """
        ci = extraction.char_interval
        if ci is None or ci.start_pos is None or ci.end_pos is None:
            return ""
        try:
            start_idx = self._char_to_token_index(tokenized_text, ci.start_pos)
            sent_range = find_sentence_range(
                document_text, tokenized_text.tokens, start_idx
            )
            expanded_start = sent_range.start_index
            expanded_end = sent_range.end_index
            for _ in range(context_sentences):
                if expanded_start > 0:
                    prev = find_sentence_range(
                        document_text,
                        tokenized_text.tokens,
                        max(0, expanded_start - 1),
                    )
                    expanded_start = prev.start_index
                if expanded_end < len(tokenized_text.tokens):
                    fwd = find_sentence_range(
                        document_text, tokenized_text.tokens, expanded_end
                    )
                    expanded_end = fwd.end_index
            return tokens_text(
                tokenized_text,
                TokenInterval(start_index=expanded_start, end_index=expanded_end),
            )
        except (ValueError, IndexError):
            return ""

    def extract_batch(
        self,
        extractions,
        document_text: str,
        tokenized_text: TokenizedText,
        context_sentences: int = 2,
    ) -> None:
        """Set ``evidence_text`` on every extraction in *extractions*."""
        for ext in extractions:
            ext.attributes = dict(ext.attributes or {})
            ext.attributes["evidence_text"] = self.extract_evidence(
                ext, document_text, tokenized_text, context_sentences
            )

    def _char_to_token_index(
        self, tokenized_text: TokenizedText, char_pos: int
    ) -> int:
        """Return the token index within *tokenized_text* for *char_pos*."""
        for i, token in enumerate(tokenized_text.tokens):
            if token.char_interval.start_pos <= char_pos < token.char_interval.end_pos:
                return i
        return 0
