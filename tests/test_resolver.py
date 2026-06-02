"""Tests for resolver."""
import pytest
from src.data import FormatType, Extraction, AlignmentStatus, CharInterval
from src.format_handler import FormatHandler
from src.tokenizer import RegexTokenizer
from src.resolver import Resolver, WordAligner


@pytest.fixture
def fh():
    return FormatHandler(format_type=FormatType.JSON, use_fences=False)


class TestResolver:
    def test_resolve_simple(self, fh):
        r = Resolver(format_handler=fh)
        result = r.resolve(
            '{"extractions": [{"Alternative": "thymol", "Alternative_attributes": {"abbreviation": "THY"}}]}'
        )
        assert len(result) == 1
        assert result[0].extraction_class == "Alternative"
        assert result[0].extraction_text == "thymol"
        assert result[0].attributes["abbreviation"] == "THY"

    def test_resolve_multiple(self, fh):
        r = Resolver(format_handler=fh)
        result = r.resolve(
            '{"extractions": [{"Alternative": "thymol"}, {"Alternative": "curcumin"}]}'
        )
        assert len(result) == 2

    def test_resolve_empty(self, fh):
        r = Resolver(format_handler=fh)
        result = r.resolve('{"extractions": []}')
        assert len(result) == 0

    def test_resolve_suppress_errors(self, fh):
        r = Resolver(format_handler=fh)
        result = r.resolve("not valid json", suppress_parse_errors=True)
        assert result == []

    def test_resolve_raises_without_suppress(self, fh):
        r = Resolver(format_handler=fh)
        with pytest.raises(ValueError):
            r.resolve("not valid json", suppress_parse_errors=False)


class TestWordAligner:
    def setup_method(self):
        self.aligner = WordAligner()
        self.tokenizer = RegexTokenizer()

    def test_align_exact_match(self):
        source = "thymol was added to the diet"
        exts = [Extraction(extraction_class="Alt", extraction_text="thymol")]
        result = self.aligner.align_extractions(
            [exts],
            source,
            token_offset=0,
            char_offset=0,
            enable_fuzzy_alignment=False,
            tokenizer_impl=self.tokenizer,
        )
        aligned = result[0][0]
        assert aligned.alignment_status == AlignmentStatus.MATCH_EXACT
        assert aligned.char_interval is not None
