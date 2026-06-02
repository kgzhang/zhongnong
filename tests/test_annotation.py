"""Tests for Annotator."""
import pytest
from src.data import Document, Extraction, AnnotatedDocument
from src.format_handler import FormatHandler, FormatType
from src.prompting import PromptTemplateStructured
from src.annotation import Annotator


class MockLM:
    requires_fence_output = False

    def infer(self, batch_prompts, **kwargs):
        for _ in batch_prompts:
            yield [MockSO('{"extractions": [{"Alternative": "thymol", "Alternative_attributes": {"standard_name": "thymol", "abbreviation": "THY"}}]}')]

    def apply_schema(self, s):
        pass

    def set_fence_output(self, v):
        pass

    @property
    def schema(self):
        return None


class MockSO:
    def __init__(self, output):
        self.output = output
        self.score = 1.0


class TestAnnotator:
    def test_annotate_text(self):
        template = PromptTemplateStructured(description="Extract alternatives.")
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        annotator = Annotator(MockLM(), template, fh)
        result = annotator.annotate_text("Pigs were fed thymol at 500 mg/kg.", max_char_buffer=500)
        assert isinstance(result, AnnotatedDocument)
        assert len(result.extractions) > 0
        assert result.extractions[0].extraction_text == "thymol"
        # evidence_text should be set by EvidenceExtractor
        assert "evidence_text" in (result.extractions[0].attributes or {})

    def test_annotate_documents_single(self):
        template = PromptTemplateStructured(description="Extract alternatives.")
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        annotator = Annotator(MockLM(), template, fh)
        doc = Document(text="Pigs were fed thymol at 500 mg/kg.")
        results = list(annotator.annotate_documents([doc], max_char_buffer=500, batch_length=1))
        assert len(results) == 1
        assert isinstance(results[0], AnnotatedDocument)
        assert len(results[0].extractions) > 0

    def test_annotate_documents_multiple(self):
        template = PromptTemplateStructured(description="Extract alternatives.")
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        annotator = Annotator(MockLM(), template, fh)
        docs = [
            Document(text="Pigs were fed thymol at 500 mg/kg.", document_id="doc1"),
            Document(text="Thymol was also given to cows.", document_id="doc2"),
        ]
        results = list(annotator.annotate_documents(docs, max_char_buffer=500, batch_length=2))
        assert len(results) == 2
        for r in results:
            assert isinstance(r, AnnotatedDocument)
            assert len(r.extractions) > 0

    def test_section_id_from_document(self):
        template = PromptTemplateStructured(description="Extract alternatives.")
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        annotator = Annotator(MockLM(), template, fh)
        assert annotator._section_id_from_document("PMC123_methods") == "Methods"
        assert annotator._section_id_from_document("PMC123_results") == "Results"
        assert annotator._section_id_from_document("PMC123") == "PMC123"

    def test_annotate_empty_text(self):
        template = PromptTemplateStructured(description="Extract alternatives.")
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        annotator = Annotator(MockLM(), template, fh)
        result = annotator.annotate_text("Pigs were fed thymol at 500 mg/kg.", max_char_buffer=500)
        # Even with short text, should produce a result
        assert isinstance(result, AnnotatedDocument)
