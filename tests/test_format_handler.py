"""Tests for format handler."""
import pytest
from src.data import FormatType, Extraction
from src.format_handler import FormatHandler


class TestFormatHandlerInit:
    def test_defaults(self):
        fh = FormatHandler()
        assert fh.format_type == FormatType.JSON
        assert fh.use_wrapper is True
        assert fh.wrapper_key == "extractions"
        assert fh.use_fences is True

    def test_json_no_fences(self):
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        assert fh.use_fences is False

    def test_yaml(self):
        fh = FormatHandler(format_type=FormatType.YAML)
        assert fh.format_type == FormatType.YAML

    def test_custom_wrapper_key(self):
        fh = FormatHandler(wrapper_key="items")
        assert fh.wrapper_key == "items"

    def test_no_wrapper(self):
        fh = FormatHandler(use_wrapper=False)
        assert fh.wrapper_key is None


class TestFormatExtractionExample:
    def test_json_format(self):
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        ext = Extraction(extraction_class="Alternative", extraction_text="thymol",
                        attributes={"abbreviation": "THY"})
        result = fh.format_extraction_example([ext])
        assert "Alternative" in result
        assert "thymol" in result

    def test_json_with_fences(self):
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=True)
        ext = Extraction(extraction_class="Alternative", extraction_text="thymol")
        result = fh.format_extraction_example([ext])
        assert result.startswith("```json")
        assert result.endswith("```")

    def test_yaml_format(self):
        fh = FormatHandler(format_type=FormatType.YAML, use_fences=False)
        ext = Extraction(extraction_class="Alternative", extraction_text="thymol")
        result = fh.format_extraction_example([ext])
        assert "Alternative" in result
        assert "thymol" in result


class TestParseOutput:
    def test_parse_json_no_fences(self):
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        result = fh.parse_output('{"extractions": [{"Alternative": "thymol"}]}')
        assert len(result) == 1
        assert result[0]["Alternative"] == "thymol"

    def test_parse_json_with_fences(self):
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=True)
        result = fh.parse_output('```json\n{"extractions": [{"Alternative": "thymol"}]}\n```')
        assert len(result) == 1

    def test_parse_empty_raises(self):
        fh = FormatHandler()
        with pytest.raises(ValueError):
            fh.parse_output("")

    def test_parse_think_tag_stripping(self):
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        result = fh.parse_output('<think>reasoning</think>\n{"extractions": [{"Alternative": "thymol"}]}')
        assert len(result) == 1
        assert result[0]["Alternative"] == "thymol"

    def test_parse_with_attributes(self):
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        result = fh.parse_output(
            '{"extractions": [{"Alternative": "thymol", "Alternative_attributes": {"abbreviation": "THY"}}]}'
        )
        assert len(result) == 1
        assert result[0]["Alternative_attributes"]["abbreviation"] == "THY"

    def test_parse_top_level_list_no_wrapper(self):
        fh = FormatHandler(use_wrapper=False, use_fences=False)
        result = fh.parse_output('[{"Alternative": "thymol"}, {"Alternative": "curcumin"}]')
        assert len(result) == 2

    def test_yaml_parse(self):
        """FormatHandler should parse YAML output."""
        fh = FormatHandler(format_type=FormatType.YAML, use_fences=False)
        result = fh.parse_output("extractions:\n- Alternative: thymol\n")
        assert len(result) == 1
        assert result[0]["Alternative"] == "thymol"

    def test_strict_fences_invalid_language_raises(self):
        """Strict mode raises when fence language tag doesn't match format type."""
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=True, strict_fences=True)
        with pytest.raises(ValueError, match="Invalid fence language tag"):
            fh.parse_output('```yaml\n{"extractions": []}\n```')

    def test_parse_invalid_json_raises(self):
        """Invalid JSON should raise ValueError."""
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        with pytest.raises(ValueError):
            fh.parse_output('{invalid json}}}')

    def test_parse_non_dict_item_raises(self):
        """Each item in extractions list must be a dict."""
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False)
        with pytest.raises(ValueError, match="dict"):
            fh.parse_output('{"extractions": ["not a dict"]}')

    def test_yaml_format_with_fences(self):
        """YAML output with fences should work."""
        fh = FormatHandler(format_type=FormatType.YAML, use_fences=True)
        ext = Extraction(extraction_class="Alternative", extraction_text="thymol")
        result = fh.format_extraction_example([ext])
        assert result.startswith("```yaml")

    def test_parse_unwrapped_dict(self):
        """Dict without wrapper key should be treated as single extraction."""
        fh = FormatHandler(format_type=FormatType.JSON, use_fences=False, use_wrapper=False)
        result = fh.parse_output('{"Alternative": "thymol"}')
        assert len(result) == 1
        assert result[0]["Alternative"] == "thymol"

    def test_non_string_key_raises(self):
        """Non-string keys in extraction dicts should raise in YAML mode."""
        fh = FormatHandler(format_type=FormatType.YAML, use_fences=False, use_wrapper=False)
        with pytest.raises(ValueError, match="string"):
            fh.parse_output('{1: value}')
