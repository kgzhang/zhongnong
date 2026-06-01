import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from stage1_xml_parser import parse_xml_to_sections, extract_conclusion

FIXTURE = Path(__file__).parent / "fixtures" / "sample.xml"

def test_parse_metadata():
    result = parse_xml_to_sections(str(FIXTURE))
    assert result["doi"] == "10.1016/test.2024.001"
    assert result["pmid"] == "12345678"
    assert result["journal"] == "Journal of Animal Science"
    assert result["publication_year"] == 2024

def test_section_mapping():
    result = parse_xml_to_sections(str(FIXTURE))
    assert "materials_and_methods" in result["sections"]
    assert "results" in result["sections"]
    assert "discussion" in result["sections"]
    mm = result["sections"]["materials_and_methods"]
    assert len(mm["subsections"]) >= 2
    assert any("Animals" in s["title"] for s in mm["subsections"])

def test_conclusion_extraction():
    result = parse_xml_to_sections(str(FIXTURE))
    conclusion = result["sections"]["abstract"]["conclusion_sentence"]
    assert "In conclusion" in conclusion
    assert "thymol" in conclusion

def test_extract_conclusion_function():
    paragraphs = [
        "Background: Antibiotics are widely used.",
        "Results show that thymol improved ADG.",
        "In conclusion, thymol significantly improved growth.",
    ]
    result = extract_conclusion(paragraphs)
    assert "In conclusion" in result["sentence"]
    assert result["marker"] == "In conclusion"
    assert result["no_marker"] is False

def test_extract_conclusion_no_marker():
    paragraphs = ["Some data.", "More data.", "Final sentence without marker."]
    result = extract_conclusion(paragraphs)
    assert result["sentence"] == "Final sentence without marker."
    assert result["no_marker"] is True

def test_extract_conclusion_empty():
    result = extract_conclusion([])
    assert result["sentence"] == ""
    assert result["no_marker"] is True
