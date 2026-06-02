"""Tests for evidence extractor."""
import pytest
from src.data import Extraction, CharInterval
from src.tokenizer import RegexTokenizer
from src.evidence import EvidenceExtractor


class TestEvidenceExtractor:
    def setup_method(self):
        self.extractor = EvidenceExtractor()
        self.tokenizer = RegexTokenizer()

    def test_extract_evidence_aligned(self):
        text = (
            "Pigs were fed a basal diet supplemented with 500 mg/kg thymol. "
            "The trial lasted 28 days."
        )
        tt = self.tokenizer.tokenize(text)
        thymol_start = text.index("thymol")
        thymol_end = thymol_start + len("thymol")
        ext = Extraction(
            extraction_class="Alternative",
            extraction_text="thymol",
            char_interval=CharInterval(start_pos=thymol_start, end_pos=thymol_end),
        )
        evidence = self.extractor.extract_evidence(ext, text, tt)
        assert "thymol" in evidence
        assert len(evidence) > 0

    def test_extract_evidence_unaligned(self):
        ext = Extraction(extraction_class="Alternative", extraction_text="thymol")
        tt = self.tokenizer.tokenize("Some text.")
        assert self.extractor.extract_evidence(ext, "Some text.", tt) == ""

    def test_extract_batch(self):
        text = "Thymol was used. It improved growth."
        tt = self.tokenizer.tokenize(text)
        t_start = text.index("Thymol")
        ext1 = Extraction(
            extraction_class="Alt",
            extraction_text="Thymol",
            char_interval=CharInterval(t_start, t_start + 6),
        )
        ext2 = Extraction(
            extraction_class="Alt", extraction_text="zinc", char_interval=None
        )
        self.extractor.extract_batch([ext1, ext2], text, tt)
        assert ext1.attributes["evidence_text"] != ""
        assert ext2.attributes["evidence_text"] == ""


class TestSourceLocation:
    def test_basic(self):
        from src.source_location import SourceLocationResolver

        r = SourceLocationResolver()
        ext = Extraction(extraction_class="Alt", extraction_text="x")
        loc = r.resolve_location(ext, "Methods", "Some text.", 0)
        assert "Methods" in loc

    def test_table_detection(self):
        from src.source_location import SourceLocationResolver

        r = SourceLocationResolver()
        ext = Extraction(
            extraction_class="Alt",
            extraction_text="x",
            char_interval=CharInterval(start_pos=50, end_pos=55),
        )
        text = "The results are shown in Table 2. " + "x " * 50 + "thymol here."
        loc = r.resolve_location(ext, "Results", text, 0)
        assert "Table 2" in loc
