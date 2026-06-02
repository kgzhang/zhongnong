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
