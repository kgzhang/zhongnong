"""Tests for core data types."""
import pytest
from src.data import (
    FormatType, CharInterval, AlignmentStatus,
    Extraction, Document, ExampleData, AnnotatedDocument,
)


class TestFormatType:
    def test_json(self):
        assert FormatType.JSON.value == "json"
    def test_yaml(self):
        assert FormatType.YAML.value == "yaml"


class TestCharInterval:
    def test_create(self):
        ci = CharInterval(start_pos=0, end_pos=10)
        assert ci.start_pos == 0
        assert ci.end_pos == 10
    def test_none_defaults(self):
        ci = CharInterval()
        assert ci.start_pos is None
        assert ci.end_pos is None


class TestAlignmentStatus:
    def test_values(self):
        assert AlignmentStatus.MATCH_EXACT.value == "match_exact"
        assert AlignmentStatus.MATCH_LESSER.value == "match_lesser"
        assert AlignmentStatus.MATCH_FUZZY.value == "match_fuzzy"
        assert AlignmentStatus.MATCH_GREATER.value == "match_greater"


class TestExtraction:
    def test_create_minimal(self):
        ext = Extraction(extraction_class="Alternative", extraction_text="thymol")
        assert ext.extraction_class == "Alternative"
        assert ext.extraction_text == "thymol"
        assert ext.char_interval is None
        assert ext.alignment_status is None
        assert ext.attributes is None

    def test_create_with_attributes(self):
        ext = Extraction(extraction_class="Alternative", extraction_text="thymol",
                        attributes={"standard_name": "thymol", "abbreviation": "THY"})
        assert ext.attributes["standard_name"] == "thymol"

    def test_create_with_position(self):
        ci = CharInterval(start_pos=5, end_pos=11)
        ext = Extraction(extraction_class="Alternative", extraction_text="thymol",
                        char_interval=ci, alignment_status=AlignmentStatus.MATCH_EXACT,
                        extraction_index=0, group_index=0)
        assert ext.char_interval.start_pos == 5
        assert ext.alignment_status == AlignmentStatus.MATCH_EXACT


class TestDocument:
    def test_create_minimal(self):
        doc = Document(text="Test text")
        assert doc.text == "Test text"
        assert doc.document_id is not None
        assert doc.document_id.startswith("doc_")

    def test_create_with_id(self):
        doc = Document(text="Test text", document_id="PMC123")
        assert doc.document_id == "PMC123"

    def test_additional_context(self):
        doc = Document(text="Text", additional_context="Extra info")
        assert doc.additional_context == "Extra info"

    def test_with_additional_context(self):
        doc = Document(text="Text", document_id="id1")
        doc2 = doc.with_additional_context("New context")
        assert doc2.document_id == "id1"
        assert doc2.text == "Text"
        assert doc2.additional_context == "New context"
        assert doc.additional_context is None


class TestExampleData:
    def test_create(self):
        ext = Extraction(extraction_class="Alt", extraction_text="thymol")
        ed = ExampleData(text="Some text", extractions=[ext])
        assert ed.text == "Some text"
        assert len(ed.extractions) == 1


class TestAnnotatedDocument:
    def test_create(self):
        ext = Extraction(extraction_class="Alt", extraction_text="thymol")
        ad = AnnotatedDocument(document_id="PMC123", extractions=[ext], text="Source text")
        assert ad.document_id == "PMC123"
        assert len(ad.extractions) == 1

    def test_auto_generated_id(self):
        ad = AnnotatedDocument(extractions=[], text="text")
        assert ad.document_id is not None
        assert ad.document_id.startswith("doc_")
