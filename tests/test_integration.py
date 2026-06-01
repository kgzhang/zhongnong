"""Integration test: end-to-end pipeline modules."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

FIXTURE = Path(__file__).parent / "fixtures" / "sample.xml"


def test_full_pipeline_on_sample_xml():
    """Verify: XML parse + DSPy extraction prompts build without errors."""
    from stage1_xml_parser import parse_xml_to_sections

    # Stage 1: Parse XML
    sections = parse_xml_to_sections(str(FIXTURE))
    assert sections["doi"] == "10.1016/test.2024.001"
    assert sections["publication_year"] == 2024

    conclusion = sections["sections"]["abstract"]["conclusion_sentence"]
    assert "In conclusion" in conclusion

    mm_text = sections["sections"]["materials_and_methods"]["full_text"]
    assert len(mm_text) > 0
    results_text = sections["sections"]["results"]["full_text"]
    assert len(results_text) > 0


def test_glossary_loaded_correctly():
    """Verify glossary can be loaded and returns expected classifications."""
    from glossary import GlossaryIndex
    gi = GlossaryIndex()
    gi.load("ALTERNATIVE.tsv")
    assert gi.lookup("thymol")["class"] == "Plant_Extract"
    assert gi.lookup("butyric acid")["class"] == "Organic_Acid"
    assert gi.lookup("made_up_substance_xyz")["class"] == "Other"


def test_entity_id_builder():
    """Verify entity ID format is consistent."""
    from utils import build_entity_id
    eid = build_entity_id("10.1016/j.animal.2024", "ALT", 1)
    assert "ALT" in eid
    assert "000001" in eid


def test_all_modules_importable():
    """Verify all active pipeline modules can be imported."""
    modules = [
        "config", "utils", "entity_id", "glossary",
        "stage0_search", "stage1_xml_parser",
        "stage4_validate", "stage4_export_graph",
        "dspy_extract",
    ]
    for mod_name in modules:
        __import__(mod_name)
