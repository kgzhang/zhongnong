"""Tests for prompting layer."""
import pytest
from src.data import FormatType, Extraction, ExampleData
from src.format_handler import FormatHandler
from src.prompting import (
    PromptTemplateStructured, QAPromptGenerator,
    PromptBuilder, ContextAwarePromptBuilder,
)


@pytest.fixture
def fh():
    return FormatHandler(format_type=FormatType.JSON, use_fences=False)


@pytest.fixture
def template():
    return PromptTemplateStructured(
        description="Extract entities from text.",
        examples=[
            ExampleData(
                text="Sample input text.",
                extractions=[Extraction(extraction_class="Alt", extraction_text="thymol")],
            )
        ],
    )


class TestPromptTemplateStructured:
    def test_create(self):
        pt = PromptTemplateStructured(description="Test")
        assert pt.description == "Test"
        assert pt.examples == []

    def test_with_examples(self):
        ex = ExampleData(text="t", extractions=[])
        pt = PromptTemplateStructured(description="D", examples=[ex])
        assert len(pt.examples) == 1


class TestQAPromptGenerator:
    def test_render_basic(self, template, fh):
        gen = QAPromptGenerator(template=template, format_handler=fh)
        result = gen.render("What is this?")
        assert "Extract entities" in result
        assert "Q: What is this?" in result
        assert "A:" in result

    def test_render_with_examples(self, template, fh):
        gen = QAPromptGenerator(template=template, format_handler=fh)
        result = gen.render("What is this?")
        assert "Sample input text" in result

    def test_render_with_context(self, template, fh):
        gen = QAPromptGenerator(template=template, format_handler=fh)
        result = gen.render("Q text", additional_context="Extra info")
        assert "Extra info" in result

    def test_format_example_as_text(self, template, fh):
        gen = QAPromptGenerator(template=template, format_handler=fh)
        ex = template.examples[0]
        result = gen.format_example_as_text(ex)
        assert "Q: Sample input text." in result
        assert "A:" in result
        assert "thymol" in result


class TestPromptBuilder:
    def test_build_prompt(self, template, fh):
        gen = QAPromptGenerator(template=template, format_handler=fh)
        builder = PromptBuilder(gen)
        result = builder.build_prompt("Chunk text", "doc1")
        assert "Chunk text" in result


class TestContextAwarePromptBuilder:
    def test_build_prompt_with_context(self, template, fh):
        gen = QAPromptGenerator(template=template, format_handler=fh)
        builder = ContextAwarePromptBuilder(gen, context_window_chars=50)
        result1 = builder.build_prompt("First chunk.", "doc1")
        result2 = builder.build_prompt("Second chunk.", "doc1")
        assert "[Previous text]" in result2

    def test_no_context_bleeding(self, template, fh):
        gen = QAPromptGenerator(template=template, format_handler=fh)
        builder = ContextAwarePromptBuilder(gen, context_window_chars=50)
        builder.build_prompt("Doc1 text.", "doc1")
        result2 = builder.build_prompt("Doc2 text.", "doc2")
        assert "Doc1" not in result2
