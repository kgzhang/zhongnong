"""Tests for chunking."""
import pytest
from src.data import Document
from src.tokenizer import RegexTokenizer, TokenizedText, TokenInterval
from src.chunking import (
    TextChunk, SentenceIterator, ChunkIterator,
    make_batches_of_textchunk, get_token_interval_text,
    get_char_interval, create_token_interval,
)


def _make_tokenized(text: str) -> TokenizedText:
    t = RegexTokenizer()
    return t.tokenize(text)


class TestCreateTokenInterval:
    def test_valid(self):
        ti = create_token_interval(0, 5)
        assert ti.start_index == 0
        assert ti.end_index == 5

    def test_invalid_negative(self):
        with pytest.raises(ValueError):
            create_token_interval(-1, 5)

    def test_invalid_order(self):
        with pytest.raises(ValueError):
            create_token_interval(5, 0)


class TestGetTokenIntervalText:
    def test_simple(self):
        tt = _make_tokenized("Hello beautiful world")
        ti = TokenInterval(start_index=0, end_index=2)
        result = get_token_interval_text(tt, ti)
        assert result == "Hello beautiful"


class TestGetCharInterval:
    def test_simple(self):
        tt = _make_tokenized("Hello world")
        ti = TokenInterval(start_index=0, end_index=2)
        ci = get_char_interval(tt, ti)
        assert ci.start_pos == 0
        assert ci.end_pos == len("Hello world")


class TestTextChunk:
    def test_basic(self):
        doc = Document(text="Hello world. This is a test.")
        tt = _make_tokenized(doc.text)
        # Monkey-patch tokenized_text onto doc
        doc.tokenized_text = tt
        ti = TokenInterval(start_index=0, end_index=len(tt.tokens))
        chunk = TextChunk(token_interval=ti, document=doc)
        assert chunk.document_id == doc.document_id
        assert "Hello world" in chunk.chunk_text


class TestSentenceIterator:
    def test_two_sentences(self):
        tt = _make_tokenized("Roses are red. Violets are blue.")
        si = SentenceIterator(tt)
        sentences = list(si)
        assert len(sentences) >= 2


class TestChunkIterator:
    def test_short_text_one_chunk(self):
        text = "Short text. Still short."
        tokenizer = RegexTokenizer()
        ci = ChunkIterator(text=text, max_char_buffer=1000, tokenizer_impl=tokenizer)
        chunks = list(ci)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)

    def test_long_text_multiple_chunks(self):
        text = "A. " * 500
        tokenizer = RegexTokenizer()
        ci = ChunkIterator(text=text, max_char_buffer=100, tokenizer_impl=tokenizer)
        chunks = list(ci)
        assert len(chunks) > 1


class TestMakeBatches:
    def test_batch_of_two(self):
        doc = Document(text="A. B. C. D.")
        tt = _make_tokenized(doc.text)
        doc.tokenized_text = tt
        ti = TokenInterval(start_index=0, end_index=len(tt.tokens))
        chunks = [TextChunk(token_interval=ti, document=doc) for _ in range(5)]
        batches = list(make_batches_of_textchunk(iter(chunks), batch_length=2))
        assert len(batches) == 3  # 2 + 2 + 1
        assert len(batches[0]) == 2
        assert len(batches[2]) == 1


class TestChunkIteratorEdgeCases:
    def test_empty_text_produces_no_chunks(self):
        """ChunkIterator with empty text should produce no chunks."""
        tokenizer = RegexTokenizer()
        doc = Document(text="")
        ci = ChunkIterator(text="", max_char_buffer=100, tokenizer_impl=tokenizer, document=doc)
        chunks = list(ci)
        assert isinstance(chunks, list)
        assert len(chunks) == 0

    def test_very_long_text_produces_multiple_chunks(self):
        """ChunkIterator should split long text into multiple chunks."""
        text = "A. B. C. D. E. F. G. H. "
        tokenizer = RegexTokenizer()
        ci = ChunkIterator(text=text, max_char_buffer=10, tokenizer_impl=tokenizer)
        chunks = list(ci)
        assert len(chunks) > 1
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)


class TestSentenceIteratorEdgeCases:
    def test_resume_from_mid_sentence_position(self):
        """SentenceIterator should resume from a given token position."""
        tt = _make_tokenized("Roses are red. Violets are blue. Flowers are nice.")
        si = SentenceIterator(tt, curr_token_pos=0)
        first = next(si)
        assert first.start_index == 0
        assert first.end_index > first.start_index

    def test_position_at_end_raises_stop(self):
        """SentenceIterator at end of tokens should raise StopIteration."""
        tt = _make_tokenized("Roses are red.")
        si = SentenceIterator(tt, curr_token_pos=len(tt.tokens))
        with pytest.raises(StopIteration):
            next(si)


class TestTextChunkProperties:
    def test_sanitized_chunk_text_strips_whitespace(self):
        """sanitized_chunk_text should strip leading/trailing whitespace."""
        doc = Document(text="  Hello world.  \n")
        tt = _make_tokenized(doc.text)
        doc.tokenized_text = tt
        ti = TokenInterval(start_index=0, end_index=len(tt.tokens))
        chunk = TextChunk(token_interval=ti, document=doc)
        sanitized = chunk.sanitized_chunk_text
        assert not sanitized.startswith(" ")
        assert not sanitized.endswith("\n")

    def test_additional_context_forwarded(self):
        """TextChunk should expose document's additional_context."""
        doc = Document(text="Test.", additional_context="Extra info")
        tt = _make_tokenized(doc.text)
        doc.tokenized_text = tt
        ti = TokenInterval(start_index=0, end_index=len(tt.tokens))
        chunk = TextChunk(token_interval=ti, document=doc)
        assert chunk.additional_context == "Extra info"

    def test_additional_context_none_returns_empty(self):
        """TextChunk with no additional_context should return empty string."""
        doc = Document(text="Test.")
        tt = _make_tokenized(doc.text)
        doc.tokenized_text = tt
        ti = TokenInterval(start_index=0, end_index=len(tt.tokens))
        chunk = TextChunk(token_interval=ti, document=doc)
        assert chunk.additional_context == ""

    def test_document_id_property(self):
        """TextChunk should expose document_id."""
        doc = Document(text="Test.", document_id="my-doc-123")
        tt = _make_tokenized(doc.text)
        doc.tokenized_text = tt
        ti = TokenInterval(start_index=0, end_index=len(tt.tokens))
        chunk = TextChunk(token_interval=ti, document=doc)
        assert chunk.document_id == "my-doc-123"

    def test_document_text_property(self):
        """TextChunk should expose full document_text."""
        doc = Document(text="Full document text.")
        tt = _make_tokenized(doc.text)
        doc.tokenized_text = tt
        ti = TokenInterval(start_index=0, end_index=len(tt.tokens))
        chunk = TextChunk(token_interval=ti, document=doc)
        assert chunk.document_text == "Full document text."

    def test_char_interval_property(self):
        """TextChunk should expose char_interval."""
        doc = Document(text="Hello world")
        tt = _make_tokenized(doc.text)
        doc.tokenized_text = tt
        ti = TokenInterval(start_index=0, end_index=len(tt.tokens))
        chunk = TextChunk(token_interval=ti, document=doc)
        assert chunk.char_interval.start_pos == 0


class TestGetCharIntervalEdgeCases:
    def test_get_char_interval_invalid_start_gt_end(self):
        """get_char_interval should raise on invalid token interval (start > end)."""
        tt = _make_tokenized("Hello world")
        with pytest.raises(ValueError):
            get_char_interval(tt, TokenInterval(start_index=5, end_index=0))

    def test_get_char_interval_start_equals_end(self):
        """get_char_interval with start == end should return empty interval."""
        tt = _make_tokenized("Hello world")
        ci = get_char_interval(tt, TokenInterval(start_index=0, end_index=0))
        assert ci.start_pos == 0
        assert ci.end_pos == 0

    def test_get_char_interval_start_out_of_range(self):
        """get_char_interval should raise when start >= len(tokens)."""
        tt = _make_tokenized("Hello world")
        with pytest.raises(ValueError):
            get_char_interval(tt, TokenInterval(start_index=999, end_index=1000))
