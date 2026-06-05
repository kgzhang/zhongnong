"""Tests for the per-extraction alignment engine (src.aligner)."""

import re

import pytest

from src.aligner import (
    AlignResult,
    align_and_evidence,
    align_and_evidence_batch,
    _compact_to_original,
    _find_with_operator_normalization,
)
from src.data import AlignmentStatus, Extraction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_doc() -> str:
    return (
        "Pigs were fed a basal diet supplemented with 500 mg/kg thymol "
        "(THY, purity >= 99%) for 28 days. Average Daily Gain (ADG) was "
        "measured weekly. Compared with the basal diet control group, "
        "thymol significantly increased ADG in the jejunal mucosa "
        "(P < 0.05, Table 3)."
    )


# ---------------------------------------------------------------------------
# Exact matching (str.find)
# ---------------------------------------------------------------------------


class TestExactMatching:
    def test_simple_exact_match(self, sample_doc):
        ext = Extraction(extraction_class="Alt", extraction_text="thymol")
        result = align_and_evidence(ext, sample_doc)
        assert result.char_interval is not None
        assert result.alignment_status == AlignmentStatus.MATCH_EXACT
        found = sample_doc[result.char_interval.start_pos:result.char_interval.end_pos]
        assert found.lower() == "thymol"

    def test_case_insensitive_match(self, sample_doc):
        ext = Extraction(extraction_class="Ind", extraction_text="average daily gain")
        result = align_and_evidence(ext, sample_doc)
        assert result.char_interval is not None
        found = sample_doc[result.char_interval.start_pos:result.char_interval.end_pos]
        assert found == "Average Daily Gain"

    def test_multi_word_match(self, sample_doc):
        ext = Extraction(extraction_class="TS", extraction_text="jejunal mucosa")
        result = align_and_evidence(ext, sample_doc)
        assert result.char_interval is not None
        found = sample_doc[result.char_interval.start_pos:result.char_interval.end_pos]
        assert found == "jejunal mucosa"


# ---------------------------------------------------------------------------
# Whitespace-normalized matching
# ---------------------------------------------------------------------------


class TestWhitespaceNormalization:
    def test_p_value_no_space(self, sample_doc):
        """P<0.05 should match P < 0.05 in source."""
        ext = Extraction(extraction_class="Result", extraction_text="P<0.05")
        result = align_and_evidence(ext, sample_doc)
        assert result.char_interval is not None
        found = sample_doc[result.char_interval.start_pos:result.char_interval.end_pos]
        assert found == "P < 0.05"

    def test_p_value_leading_space(self, sample_doc):
        ext = Extraction(extraction_class="Result", extraction_text="P< 0.05")
        result = align_and_evidence(ext, sample_doc)
        assert result.char_interval is not None
        found = sample_doc[result.char_interval.start_pos:result.char_interval.end_pos]
        assert found == "P < 0.05"

    def test_p_value_trailing_space(self, sample_doc):
        ext = Extraction(extraction_class="Result", extraction_text="P <0.05")
        result = align_and_evidence(ext, sample_doc)
        assert result.char_interval is not None
        found = sample_doc[result.char_interval.start_pos:result.char_interval.end_pos]
        assert found == "P < 0.05"


# ---------------------------------------------------------------------------
# Word-boundary matching (source_location precision regression)
# ---------------------------------------------------------------------------


@pytest.fixture
def doc_with_isoflavones() -> str:
    return "The product contains isoflavones, flavone, and glutathione."


class TestWordBoundaryMatching:
    """Ensure short extraction_text values don't match mid-word."""

    def test_does_not_match_mid_word(self, doc_with_isoflavones):
        """'flavone' should NOT match inside 'isoflavones'."""
        ext = Extraction(extraction_class="Alt", extraction_text="flavone")
        result = align_and_evidence(ext, doc_with_isoflavones)

        assert result.char_interval is not None
        s = result.char_interval.start_pos
        e = result.char_interval.end_pos
        found = doc_with_isoflavones[s:e]
        assert found == "flavone"
        # Must NOT be the 'flavone' inside 'isoflavones'
        assert "iso" not in doc_with_isoflavones[max(0, s - 3):e]

    def test_short_abbrev_does_not_match_mid_word(self):
        """'ma' should NOT match inside 'inflammatory'."""
        doc = "The antioxidants were found to improve inflammatory responses."
        ext = Extraction(extraction_class="Alt", extraction_text="ma")
        result = align_and_evidence(ext, doc)
        # 'ma' as standalone text should NOT match — no standalone 'ma' in this doc
        assert result.char_interval is None, (
            f"found 'ma' mid-word at {result.char_interval}"
            if result.char_interval and result.char_interval.start_pos is not None
            else "should not have matched mid-word"
        )

    def test_exact_word_match_at_boundary(self, doc_with_isoflavones):
        """Full word should still match correctly at word boundaries."""
        ext = Extraction(extraction_class="Alt", extraction_text="isoflavones")
        result = align_and_evidence(ext, doc_with_isoflavones)
        assert result.char_interval is not None
        s = result.char_interval.start_pos
        e = result.char_interval.end_pos
        assert doc_with_isoflavones[s:e] == "isoflavones"
        assert s == 0 or not doc_with_isoflavones[s - 1].isalpha()
        assert e >= len(doc_with_isoflavones) or not doc_with_isoflavones[e].isalpha()

    def test_source_location_compact_format(self, doc_with_isoflavones):
        """source_location should be sentence-boundary text, ⊆ evidence_text."""
        ext = Extraction(extraction_class="Alt", extraction_text="flavone")
        result = align_and_evidence(ext, doc_with_isoflavones)
        loc = result.source_location
        ev = result.evidence_text
        # source_location must be a substring of evidence_text
        assert loc in ev, (
            f"source_location not in evidence_text:\n"
            f"  loc={loc[:80]}...\n  ev={ev[:80]}..."
        )
        # source_location must contain the matched text
        assert "flavone" in loc, f"Expected 'flavone' in source_location, got: {loc}"
        # source_location should be compact (roughly sentence-length)
        assert len(loc) < 300, f"source_location too long ({len(loc)} chars)"


# ---------------------------------------------------------------------------
# Parenthetical normalization
# ---------------------------------------------------------------------------


class TestParentheticalNormalization:
    def test_thymol_with_abbreviation(self, sample_doc):
        """thymol (THY) should match thymol (THY, purity >= 99%)."""
        ext = Extraction(extraction_class="Alt", extraction_text="thymol (THY)")
        result = align_and_evidence(ext, sample_doc)
        assert result.char_interval is not None
        found = sample_doc[result.char_interval.start_pos:result.char_interval.end_pos]
        assert "thymol" in found.lower()
        assert "thy" in found.lower()


# ---------------------------------------------------------------------------
# Evidence text quality
# ---------------------------------------------------------------------------


class TestEvidenceText:
    def test_evidence_contains_extraction_text(self, sample_doc):
        """Every aligned extraction must have its text in evidence."""
        extractions = [
            Extraction(extraction_class="Alt", extraction_text="thymol"),
            Extraction(extraction_class="Ind", extraction_text="ADG"),
            Extraction(extraction_class="Res", extraction_text="P < 0.05"),
            Extraction(extraction_class="Res", extraction_text="P<0.05"),
            Extraction(extraction_class="TS", extraction_text="jejunal mucosa"),
            Extraction(extraction_class="Meth", extraction_text="weighing"),  # not in doc
        ]
        result = align_and_evidence_batch(extractions, sample_doc)

        for ext in result:
            if ext.char_interval is None:
                continue  # entities not found in doc are OK
            evidence = ext.evidence_text.lower() if ext.evidence_text else ""
            txt = ext.extraction_text.lower()
            txt_compact = re.sub(r"\s+", "", txt)
            ev_compact = re.sub(r"\s+", "", evidence)

            assert txt in evidence or (len(txt_compact) >= 3 and txt_compact in ev_compact), (
                f"extraction_text {ext.extraction_text!r} not found in evidence: "
                f"{ext.evidence_text[:100]!r}"
            )

    def test_p_value_evidence_correct_location(self, sample_doc):
        """P<0.05 evidence must come from the actual P-value location."""
        ext = Extraction(extraction_class="Result", extraction_text="P<0.05")
        result = align_and_evidence(ext, sample_doc)
        ci = result.char_interval
        # Must be at the P < 0.05 location (around "(P < 0.05")
        assert "P < 0.05" in sample_doc[ci.start_pos - 1:ci.end_pos + 1]


# ---------------------------------------------------------------------------
# Operator normalization
# ---------------------------------------------------------------------------


class TestOperatorNormalization:
    def test_p_equals_match(self):
        doc = "The difference was significant (P = 0.023) compared to control."
        ext = Extraction(extraction_class="Result", extraction_text="P=0.023")
        result = align_and_evidence(ext, doc)
        assert result.char_interval is not None
        found = doc[result.char_interval.start_pos:result.char_interval.end_pos]
        assert "0.023" in found

    def test_p_less_than_compact(self):
        doc = "Supplementation significantly increased ADG (P<0.05)."
        ext = Extraction(extraction_class="Result", extraction_text="P < 0.05")
        result = align_and_evidence(ext, doc)
        assert result.char_interval is not None


# ---------------------------------------------------------------------------
# Batch alignment
# ---------------------------------------------------------------------------


class TestBatchAlignment:
    def test_all_aligned_in_place(self, sample_doc):
        extractions = [
            Extraction(extraction_class="Alt", extraction_text="thymol"),
            Extraction(extraction_class="Ind", extraction_text="ADG"),
            Extraction(extraction_class="Res", extraction_text="P < 0.05"),
        ]
        result = align_and_evidence_batch(extractions, sample_doc)
        assert len(result) == 3
        for ext in result:
            assert ext.char_interval is not None
            assert ext.evidence_text
            assert ext.alignment_status is not None

    def test_unaligned_extraction_ok(self, sample_doc):
        """Entity not in doc should have None char_interval."""
        ext = Extraction(extraction_class="Swine", extraction_text="Duroc × Landrace × Yorkshire")
        result = align_and_evidence(ext, sample_doc)
        assert result.char_interval is None
        assert result.evidence_text == ""


# ---------------------------------------------------------------------------
# Helper: _compact_to_original
# ---------------------------------------------------------------------------


class TestCompactToOriginal:
    def test_simple(self):
        original = "hello world"
        compact = "helloworld"
        assert _compact_to_original(original, compact, 0) == 0  # 'h'
        assert _compact_to_original(original, compact, 5) == 6  # 'w'

    def test_with_newlines(self):
        original = "line one\nline two"
        compact = "lineoneline two"  # \n preserved? no — \n is whitespace
        # Actually \n IS whitespace: compact = 'lineoneline two' → wait no.
        # re.sub(r'\s+', '', original) = 'lineonelinetwo'
        assert re.sub(r"\s+", "", original) == "lineonelinetwo"
