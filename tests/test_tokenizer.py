"""Tests for tokenization utilities."""
import pytest
from src.tokenizer import (
    CharInterval, TokenInterval, TokenType, Token, TokenizedText,
    RegexTokenizer, tokenize, tokens_text, find_sentence_range,
)


class TestTokenInterval:
    def test_create(self):
        ti = TokenInterval(start_index=0, end_index=5)
        assert ti.start_index == 0
        assert ti.end_index == 5


class TestRegexTokenizer:
    def setup_method(self):
        self.tokenizer = RegexTokenizer()

    def test_tokenize_simple(self):
        result = self.tokenizer.tokenize("Hello world.")
        assert len(result.tokens) >= 3
        assert result.tokens[0].token_type == TokenType.WORD
        assert result.text == "Hello world."

    def test_tokenize_digits(self):
        result = self.tokenizer.tokenize("123 pigs")
        assert result.tokens[0].token_type == TokenType.NUMBER

    def test_tokenize_punctuation(self):
        result = self.tokenizer.tokenize("Test.")
        punct = [t for t in result.tokens if t.token_type == TokenType.PUNCTUATION]
        assert len(punct) >= 1

    def test_tokenize_newline_tracking(self):
        result = self.tokenizer.tokenize("Line one.\nLine two.")
        newline_tokens = [t for t in result.tokens if t.first_token_after_newline]
        assert len(newline_tokens) >= 1

    def test_tokenize_empty(self):
        result = self.tokenizer.tokenize("")
        assert len(result.tokens) == 0
        assert result.text == ""

    def test_char_intervals(self):
        result = self.tokenizer.tokenize("ab cd ef")
        assert result.tokens[0].char_interval.start_pos == 0
        assert result.tokens[0].char_interval.end_pos == 2


class TestTokensText:
    def setup_method(self):
        self.tokenizer = RegexTokenizer()

    def test_reconstruct_text(self):
        tt = self.tokenizer.tokenize("Hello world today")
        interval = TokenInterval(start_index=0, end_index=2)
        text = tokens_text(tt, interval)
        assert text == "Hello world"

    def test_empty_interval(self):
        tt = self.tokenizer.tokenize("Hello world")
        interval = TokenInterval(start_index=0, end_index=0)
        assert tokens_text(tt, interval) == ""

    def test_invalid_interval(self):
        tt = self.tokenizer.tokenize("Hello world")
        with pytest.raises(ValueError):
            tokens_text(tt, TokenInterval(start_index=0, end_index=999))


class TestFindSentenceRange:
    def setup_method(self):
        self.tokenizer = RegexTokenizer()

    def test_simple_sentence(self):
        text = "Roses are red. Violets are blue."
        tt = self.tokenizer.tokenize(text)
        rng = find_sentence_range(text, tt.tokens, 0)
        assert rng.end_index > rng.start_index

    def test_empty_tokens(self):
        rng = find_sentence_range("", [], 0)
        assert rng.start_index == 0
        assert rng.end_index == 0

    def test_known_abbreviation_not_sentence_end(self):
        text = "Dr. Smith conducted the study."
        tt = self.tokenizer.tokenize(text)
        rng = find_sentence_range(text, tt.tokens, 0)
        sentence_text = tokens_text(tt, rng)
        assert "Smith" in sentence_text or len(sentence_text.split()) > 1


class TestConvenienceTokenize:
    def test_tokenize_function(self):
        tt = tokenize("Simple test.")
        assert len(tt.tokens) >= 2
        assert isinstance(tt, TokenizedText)
