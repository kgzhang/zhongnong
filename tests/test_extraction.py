"""Tests for extraction API."""
import pytest
from src.extraction import DocumentExtractionResult


class TestDocumentExtractionResult:
    def test_create(self):
        r = DocumentExtractionResult(document_id="12345", metadata={"doi": "10.1234/x", "pmid": "12345"})
        assert r.metadata.get("doi") == "10.1234/x"
        assert r.document_id == "12345"
        assert r.extractions == []

    def test_warnings(self):
        r = DocumentExtractionResult(
            document_id="12345",
            metadata={"doi": "10.1234/x", "pmid": "12345"},
            warnings=["Parse error in section"],
        )
        assert len(r.warnings) == 1

    def test_extractions(self):
        from src.data import Extraction
        ext = Extraction(extraction_class="Alternative", extraction_text="thymol")
        r = DocumentExtractionResult(
            document_id="12345",
            metadata={"doi": "10.1234/x", "pmid": "12345"},
            extractions=[ext],
        )
        assert len(r.extractions) == 1
        assert r.extractions[0].extraction_text == "thymol"
