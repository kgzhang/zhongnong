"""Tests for evidence derivation."""
import pytest
from src.data import Extraction, CharInterval
from src.evidence import derive_evidence, derive_evidence_batch, EvidenceExtractor


class TestEvidenceExtractor:
    def setup_method(self):
        self.extractor = EvidenceExtractor()

    def test_extract_evidence_aligned(self):
        text = (
            "Pigs were fed a basal diet supplemented with 500 mg/kg thymol. "
            "The trial lasted 28 days."
        )
        thymol_start = text.index("thymol")
        thymol_end = thymol_start + len("thymol")
        ext = Extraction(
            extraction_class="Alternative",
            extraction_text="thymol",
            char_interval=CharInterval(start_pos=thymol_start, end_pos=thymol_end),
        )
        evidence = self.extractor.extract_evidence(ext, text)
        assert "thymol" in evidence
        assert len(evidence) > 0

    def test_extract_evidence_unaligned(self):
        ext = Extraction(extraction_class="Alternative", extraction_text="thymol")
        assert self.extractor.extract_evidence(ext, "Some text.") == ""

    def test_extract_batch(self):
        text = "Thymol was used. It improved growth."
        t_start = text.index("Thymol")
        ext1 = Extraction(
            extraction_class="Alt",
            extraction_text="Thymol",
            char_interval=CharInterval(t_start, t_start + 6),
        )
        ext2 = Extraction(
            extraction_class="Alt", extraction_text="zinc", char_interval=None
        )
        self.extractor.extract_batch([ext1, ext2], text)
        # evidence_text is now a top-level field on Extraction, not in attributes
        assert ext1.evidence_text != ""
        assert ext2.evidence_text == ""

    def test_derive_evidence_function(self):
        """Module-level derive_evidence function works."""
        text = "The study used thymol as a feed additive at 500 mg/kg."
        start = text.index("thymol")
        ext = Extraction(
            extraction_class="Alt",
            extraction_text="thymol",
            char_interval=CharInterval(start, start + 6),
        )
        evidence = derive_evidence(ext, text, context_chars=20)
        assert "thymol" in evidence
        assert "feed additive" in evidence  # context included

    def test_derive_evidence_batch_function(self):
        """Module-level derive_evidence_batch sets evidence_text on all extractions."""
        text = "Thymol and curcumin were tested."
        t_start = text.index("Thymol")
        c_start = text.index("curcumin")
        ext1 = Extraction(
            extraction_class="Alt", extraction_text="Thymol",
            char_interval=CharInterval(t_start, t_start + 6),
        )
        ext2 = Extraction(
            extraction_class="Alt", extraction_text="curcumin",
            char_interval=CharInterval(c_start, c_start + 8),
        )
        derive_evidence_batch([ext1, ext2], text)
        assert "Thymol" in ext1.evidence_text
        assert "curcumin" in ext2.evidence_text


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
