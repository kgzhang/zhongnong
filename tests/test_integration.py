"""Integration test: end-to-end prompt generation from sample XML."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

FIXTURE = Path(__file__).parent / "fixtures" / "sample.xml"


def test_full_pipeline_on_sample_xml():
    """Verify: XML -> sections -> prompts without errors."""
    from stage1_xml_parser import parse_xml_to_sections
    from stage2_prompts import build_module_c_prompt, build_module_a_prompt, build_module_b_prompt
    from stage3_prompts import build_result_prompt

    # Stage 1: Parse XML
    sections = parse_xml_to_sections(str(FIXTURE))

    # Verify metadata
    assert sections["doi"] == "10.1016/test.2024.001"
    assert sections["publication_year"] == 2024

    # Verify conclusion extraction
    conclusion = sections["sections"]["abstract"]["conclusion_sentence"]
    assert "In conclusion" in conclusion

    # Verify section mapping
    mm_text = sections["sections"]["materials_and_methods"]["full_text"]
    assert len(mm_text) > 0
    assert "barrows" in mm_text.lower() or "pig" in mm_text.lower()

    results_text = sections["sections"]["results"]["full_text"]
    assert len(results_text) > 0
    assert "ADG" in results_text or "thymol" in results_text.lower()

    # Stage 2: Verify prompts can be built (don't call LLM)
    c_prompt = build_module_c_prompt(mm_text)
    assert len(c_prompt) > 0
    assert "Composite_Product" in c_prompt or "composite" in c_prompt.lower()

    a_prompt = build_module_a_prompt(mm_text)
    assert len(a_prompt) > 0
    assert "challenge" in a_prompt.lower() or "dose" in a_prompt.lower()

    b_prompt = build_module_b_prompt(mm_text)
    assert len(b_prompt) > 0
    assert "ADG" in b_prompt or "indicator" in b_prompt.lower()

    # Stage 3: Verify result prompt can be built
    r_prompt = build_result_prompt(results_text, "", [], [], [])
    assert len(r_prompt) > 0
    assert "ADG" in r_prompt or "results" in r_prompt.lower() or "Results" in r_prompt


def test_glossary_loaded_correctly():
    """Verify glossary can be loaded and returns expected classifications."""
    from glossary import GlossaryIndex

    gi = GlossaryIndex()
    gi.load("ALTERNATIVE.tsv")

    # Check a few known substances
    thymol = gi.lookup("thymol")
    assert thymol["class"] == "Plant_Extract"

    butyric = gi.lookup("butyric acid")
    assert butyric["class"] == "Organic_Acid"

    # Unknown substance
    unknown = gi.lookup("made_up_substance_xyz")
    assert unknown["class"] == "Other"


def test_entity_id_builder():
    """Verify entity ID format is consistent."""
    from stage2_entity_extract import build_entity_id
    eid = build_entity_id("10.1016/j.animal.2024", "ALT", 1)
    assert "ALT" in eid
    assert "000001" in eid


def test_all_modules_importable():
    """Verify all pipeline modules can be imported."""
    modules = [
        "config", "models", "glossary",
        "stage0_search", "stage1_xml_parser",
        "stage2_entity_extract", "stage2_prompts",
        "stage3_result_extract", "stage3_prompts",
        "stage4_validate", "stage4_export_tsv", "stage4_export_neo4j",
    ]
    for mod_name in modules:
        __import__(mod_name)
