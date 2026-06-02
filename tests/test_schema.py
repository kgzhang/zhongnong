"""Tests for schema layer."""
from src.data import FormatType, ExampleData, Extraction
from src.schema import FormatModeSchema


class TestFormatModeSchema:
    def test_default_json(self):
        s = FormatModeSchema()
        assert s.requires_raw_output is True
        assert s.to_provider_config()["format"] == "json"

    def test_yaml(self):
        s = FormatModeSchema(format_type=FormatType.YAML)
        assert s.requires_raw_output is False

    def test_from_examples(self):
        ex = ExampleData(
            text="Test",
            extractions=[Extraction(extraction_class="A", extraction_text="t")],
        )
        s = FormatModeSchema.from_examples([ex])
        assert isinstance(s, FormatModeSchema)
