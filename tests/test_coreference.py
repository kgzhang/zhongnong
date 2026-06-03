"""Tests for coreference resolution service."""
import pytest
from src.data import Extraction, CharInterval
from src.coreference import (
    normalize_name,
    identity_key,
    build_abbreviation_map,
    resolve_coreferences,
    clean_evidence_text,
    _collect_unique_parts,
)


class TestNormalizeName:
    def test_lowercase(self):
        assert normalize_name("Thymol") == "thymol"

    def test_hyphen_to_space(self):
        assert normalize_name("microbe-derived") == "microbe derived"

    def test_underscore_to_space(self):
        assert normalize_name("Control_Group") == "control group"

    def test_collapse_whitespace(self):
        assert normalize_name("  thymol  500  ") == "thymol 500"

    def test_empty(self):
        assert normalize_name("") == ""
        assert normalize_name(None) == ""


class TestIdentityKey:
    def test_uses_primary_field(self):
        ext = Extraction(
            "Alternative", "MA",
            attributes={"standard_name": "microbe-derived antioxidants", "abbreviation": "MA"},
        )
        key = identity_key(ext, "standard_name")
        assert key == "microbe derived antioxidants"

    def test_falls_back_to_extraction_text(self):
        ext = Extraction("Alternative", "thymol")
        key = identity_key(ext, "standard_name")
        assert key == "thymol"


class TestAbbreviationMap:
    def test_builds_map(self):
        exts = [
            Extraction("Alt", "microbe-derived antioxidants",
                      attributes={"standard_name": "microbe-derived antioxidants", "abbreviation": "MA"}),
            Extraction("Alt", "superoxide dismutase",
                      attributes={"standard_name": "superoxide dismutase", "abbreviation": "SOD"}),
        ]
        abbr_map = build_abbreviation_map(exts, "standard_name")
        assert normalize_name("MA") in abbr_map
        assert abbr_map[normalize_name("MA")] == normalize_name("microbe-derived antioxidants")


class TestCoreferenceResolution:
    def test_merges_abbreviation_variants(self):
        exts = [
            Extraction("Alternative", "microbe-derived antioxidants",
                      attributes={"standard_name": "microbe-derived antioxidants", "abbreviation": "MA"}),
            Extraction("Alternative", "MA",
                      attributes={"standard_name": "microbe-derived antioxidants", "abbreviation": "MA"}),
            Extraction("Alternative", "Microbe-derived antioxidants",
                      attributes={"standard_name": "microbe-derived antioxidants"}),
        ]
        for e in exts:
            e.evidence_text = f"evidence: {e.extraction_text}"
            e.source_location = "Methods"
        result = resolve_coreferences(exts)
        assert len(result) == 1
        assert " <|> " in result[0].evidence_text

    def test_keeps_different_entities_separate(self):
        exts = [
            Extraction("Alternative", "thymol", attributes={"standard_name": "thymol"}),
            Extraction("Alternative", "curcumin", attributes={"standard_name": "curcumin"}),
        ]
        for e in exts:
            e.evidence_text = "test"
        result = resolve_coreferences(exts)
        assert len(result) == 2


class TestCleanEvidence:
    def test_removes_leading_partial_word(self):
        raw = "bited a fermented taste and was well accepted"
        cleaned = clean_evidence_text(raw)
        assert cleaned == "a fermented taste and was well accepted"

    def test_collapses_whitespace(self):
        raw = "  thymol   improved\n  growth  "
        cleaned = clean_evidence_text(raw)
        assert "  " not in cleaned
        assert "\n" not in cleaned


class TestCollectUniqueParts:
    def test_removes_duplicates(self):
        parts = ["thymol improved growth", "thymol improved growth"]
        result = _collect_unique_parts(parts)
        assert len(result) == 1

    def test_removes_substring_duplicates(self):
        parts = ["thymol improved growth significantly", "improved growth"]
        result = _collect_unique_parts(parts)
        assert len(result) == 1
        assert "significantly" in result[0]

    def test_keeps_distinct_parts(self):
        parts = ["thymol improved ADG", "curcumin reduced F:G"]
        result = _collect_unique_parts(parts)
        assert len(result) == 2
