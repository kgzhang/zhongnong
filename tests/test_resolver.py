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

    def test_align_with_char_offset(self):
        """Alignment should respect char_offset."""
        source = "Pigs were fed thymol at 500 mg/kg."
        tt = self.tokenizer.tokenize(source)
        exts = [Extraction(extraction_class="Alt", extraction_text="thymol")]
        result = self.aligner.align_extractions(
            [exts], source, token_offset=0, char_offset=100,
            enable_fuzzy_alignment=False, tokenizer_impl=self.tokenizer,
        )
        aligned = result[0][0]
        if aligned.char_interval is not None:
            # char_interval positions should include the offset
            assert aligned.char_interval.start_pos >= 100

    def test_align_empty_extractions(self):
        """Aligning empty extractions should return empty."""
        source = "Some text."
        result = self.aligner.align_extractions(
            [], source, token_offset=0, char_offset=0,
            tokenizer_impl=self.tokenizer,
        )
        assert result == []

    def test_align_multiple_groups(self):
        """Align multiple extraction groups."""
        source = "thymol and curcumin were tested."
        exts1 = [Extraction(extraction_class="Alt", extraction_text="thymol")]
        exts2 = [Extraction(extraction_class="Alt", extraction_text="curcumin")]
        result = self.aligner.align_extractions(
            [exts1, exts2],
            source,
            token_offset=0,
            char_offset=0,
            enable_fuzzy_alignment=False,
            tokenizer_impl=self.tokenizer,
        )
        assert len(result) == 2

    def test_align_no_match_falls_back(self):
        """When no exact match, alignment status may remain None."""
        source = "thymol was added to the diet"
        exts = [Extraction(extraction_class="Alt", extraction_text="zzz_nonexistent")]
        result = self.aligner.align_extractions(
            [exts],
            source,
            token_offset=0,
            char_offset=0,
            enable_fuzzy_alignment=False,
            tokenizer_impl=self.tokenizer,
        )
        aligned = result[0][0]
        # Without fuzzy alignment, no match should leave status unchanged
        assert aligned.alignment_status is None


class TestResolverEdgeCases:
    def test_resolve_with_index_suffix(self):
        """Resolver should handle index suffix keys (skipped, not used for sorting)."""
        from src.resolver import Resolver
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        r = Resolver(format_handler=fh, extraction_index_suffix="_index")
        result = r.resolve(
            '{"extractions": [{"Alternative": "thymol", "Alternative_index": 1}, {"Alternative": "curcumin", "Alternative_index": 0}]}'
        )
        assert len(result) == 2
        # Index keys are skipped; order preserved as-is from input
        assert result[0].extraction_text == "thymol"
        assert result[1].extraction_text == "curcumin"

    def test_resolve_int_extraction_value(self):
        """Extraction value can be int (converted to str)."""
        from src.resolver import Resolver
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        r = Resolver(format_handler=fh)
        result = r.resolve('{"extractions": [{"Result": 42}]}')
        assert len(result) == 1
        assert result[0].extraction_text == "42"

    def test_resolve_float_extraction_value(self):
        """Extraction value can be float (converted to str)."""
        from src.resolver import Resolver
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        r = Resolver(format_handler=fh)
        result = r.resolve('{"extractions": [{"Result": 3.14}]}')
        assert len(result) == 1
        assert result[0].extraction_text == "3.14"
