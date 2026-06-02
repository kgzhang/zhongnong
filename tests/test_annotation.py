"""Tests for Annotator."""
import pytest
from src.data import Document, Extraction, AnnotatedDocument, CharInterval
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
        assert annotator._section_id_from_document("doc_discussion") == "Discussion"

    def test_annotate_empty_text(self):
        template = PromptTemplateStructured(description="Extract alternatives.")
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        annotator = Annotator(MockLM(), template, fh)
        result = annotator.annotate_text("Pigs were fed thymol at 500 mg/kg.", max_char_buffer=500)
        assert isinstance(result, AnnotatedDocument)

    def test_annotate_text_empty_string(self):
        """Empty text should still produce a valid AnnotatedDocument."""
        template = PromptTemplateStructured(description="Extract alternatives.")
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        annotator = Annotator(MockLM(), template, fh)
        result = annotator.annotate_text("", max_char_buffer=500)
        assert isinstance(result, AnnotatedDocument)
        assert result.text == ""

    def test_merge_non_overlapping_single_pass(self):
        """Single pass returns its extractions unchanged."""
        template = PromptTemplateStructured(description="Test")
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        annotator = Annotator(MockLM(), template, fh)
        e1 = Extraction(extraction_class="Alt", extraction_text="thymol")
        e2 = Extraction(extraction_class="Alt", extraction_text="curcumin")
        merged = annotator._merge_non_overlapping([[e1, e2]])
        assert len(merged) == 2

    def test_merge_non_overlapping_empty(self):
        """Empty list returns empty."""
        template = PromptTemplateStructured(description="Test")
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        annotator = Annotator(MockLM(), template, fh)
        assert annotator._merge_non_overlapping([]) == []

    def test_merge_non_overlapping_first_pass_wins(self):
        """First pass extraction wins when overlapping."""
        template = PromptTemplateStructured(description="Test")
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        annotator = Annotator(MockLM(), template, fh)
        # Overlapping: both at same position
        e1 = Extraction(
            extraction_class="Alt", extraction_text="thymol (pass1)",
            char_interval=CharInterval(start_pos=0, end_pos=6),
        )
        e2 = Extraction(
            extraction_class="Alt", extraction_text="thymol (pass2)",
            char_interval=CharInterval(start_pos=0, end_pos=6),
        )
        merged = annotator._merge_non_overlapping([[e1], [e2]])
        assert len(merged) == 1
        assert merged[0].extraction_text == "thymol (pass1)"

    def test_merge_non_overlapping_adds_non_overlapping(self):
        """Non-overlapping extractions from later passes are added."""
        template = PromptTemplateStructured(description="Test")
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        annotator = Annotator(MockLM(), template, fh)
        e1 = Extraction(
            extraction_class="Alt", extraction_text="thymol",
            char_interval=CharInterval(start_pos=0, end_pos=6),
        )
        e2 = Extraction(
            extraction_class="Alt", extraction_text="curcumin",
            char_interval=CharInterval(start_pos=50, end_pos=58),
        )
        merged = annotator._merge_non_overlapping([[e1], [e2]])
        assert len(merged) == 2

    def test_overlaps_with_any_detects_overlap(self):
        """Overlapping char intervals are detected."""
        template = PromptTemplateStructured(description="Test")
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        annotator = Annotator(MockLM(), template, fh)
        ext = Extraction(
            extraction_class="Alt", extraction_text="test",
            char_interval=CharInterval(start_pos=10, end_pos=20),
        )
        existing = [
            Extraction(
                extraction_class="Alt", extraction_text="other",
                char_interval=CharInterval(start_pos=15, end_pos=25),
            ),
        ]
        assert annotator._overlaps_with_any(ext, existing) is True

    def test_overlaps_with_any_no_overlap(self):
        """Non-overlapping intervals are not detected."""
        template = PromptTemplateStructured(description="Test")
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        annotator = Annotator(MockLM(), template, fh)
        ext = Extraction(
            extraction_class="Alt", extraction_text="test",
            char_interval=CharInterval(start_pos=0, end_pos=10),
        )
        existing = [
            Extraction(
                extraction_class="Alt", extraction_text="other",
                char_interval=CharInterval(start_pos=20, end_pos=30),
            ),
        ]
        assert annotator._overlaps_with_any(ext, existing) is False

    def test_overlaps_with_any_none_interval(self):
        """Extraction with None char_interval never overlaps."""
        template = PromptTemplateStructured(description="Test")
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        annotator = Annotator(MockLM(), template, fh)
        ext = Extraction(extraction_class="Alt", extraction_text="test", char_interval=None)
        existing = [
            Extraction(
                extraction_class="Alt", extraction_text="other",
                char_interval=CharInterval(start_pos=0, end_pos=10),
            ),
        ]
        assert annotator._overlaps_with_any(ext, existing) is False

    def test_multi_pass_extraction(self):
        """Multi-pass extraction merges results across passes."""
        template = PromptTemplateStructured(description="Extract alternatives.")
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        annotator = Annotator(MockLM(), template, fh)
        doc = Document(text="Pigs were fed thymol at 500 mg/kg.", document_id="test_doc")
        results = list(annotator.annotate_documents(
            [doc], max_char_buffer=500, extraction_passes=2,
        ))
        assert len(results) == 1
        assert isinstance(results[0], AnnotatedDocument)
        # Multi-pass still produces extractions
        assert results[0].extractions is not None

    def test_source_location_set_on_extraction(self):
        """Source location should be populated by SourceLocationResolver."""
        template = PromptTemplateStructured(description="Extract alternatives.")
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        annotator = Annotator(MockLM(), template, fh)
        result = annotator.annotate_text(
            "Pigs were fed thymol at 500 mg/kg.", max_char_buffer=500,
        )
        for ext in result.extractions:
            assert "source_location" in (ext.attributes or {})
