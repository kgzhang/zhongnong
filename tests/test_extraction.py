"""Tests for extraction API."""
import pytest
from src.extraction import ArticleExtractionResult


class TestArticleExtractionResult:
    def test_create(self):
        r = ArticleExtractionResult(doi="10.1234/x", pmid="12345")
        assert r.doi == "10.1234/x"
        assert r.skipped is False

    def test_skipped(self):
        r = ArticleExtractionResult(doi="10.1234/x", pmid="12345", skipped=True, skip_reason="No alternatives")
        assert r.skipped is True
        assert r.skip_reason == "No alternatives"

    def test_warnings(self):
        r = ArticleExtractionResult(doi="10.1234/x", pmid="12345", warnings=["Parse error in section"])
        assert len(r.warnings) == 1

    def test_extractions(self):
        from src.data import Extraction
        ext = Extraction(extraction_class="Alternative", extraction_text="thymol")
        r = ArticleExtractionResult(doi="10.1234/x", pmid="12345", extractions=[ext])
        assert len(r.extractions) == 1
        assert r.extractions[0].extraction_text == "thymol"
