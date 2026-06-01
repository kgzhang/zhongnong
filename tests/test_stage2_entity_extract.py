from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from stage2_entity_extract import (
    normalize_dose_unit, build_entity_id, entities_to_tsv_rows,
    postprocess_alternatives,
)
from models import Alternative


def test_normalize_dose_ppm():
    val, unit = normalize_dose_unit(500, "ppm")
    assert val == 500
    assert unit == "mg/kg_feed"


def test_normalize_dose_percent():
    val, unit = normalize_dose_unit(0.05, "%")
    assert val == 500
    assert unit == "mg/kg_feed"


def test_normalize_dose_mg_per_kg_BW():
    val, unit = normalize_dose_unit(10, "mg/kg BW")
    assert val == 10
    assert unit == "mg/kg_BW"


def test_normalize_dose_mg_per_kg_no_BW():
    val, unit = normalize_dose_unit(100, "mg/kg")
    assert val == 100
    assert unit == "mg/kg_feed"


def test_normalize_dose_g_per_kg():
    val, unit = normalize_dose_unit(2, "g/kg")
    assert val == 2
    assert unit == "g/kg"


def test_normalize_dose_unknown_unit():
    val, unit = normalize_dose_unit(50, "CFU/mL")
    assert val == 50
    assert unit == "CFU/mL"


def test_build_entity_id():
    eid = build_entity_id("10.1016/x", "ALT", 3)
    assert eid == "10.1016/x_ALT_000003"


def test_build_entity_id_with_special_chars():
    eid = build_entity_id("10.1016/j.animal.2023", "RES", 42)
    assert "10.1016" in eid
    assert "_RES_" in eid
    assert "000042" in eid


def test_entities_to_tsv_rows():
    alts = [
        Alternative(entity_id="1", standard_name="thymol", alternative_class="Plant_Extract", evidence_text="..."),
        Alternative(entity_id="2", standard_name="butyric acid", alternative_class="Organic_Acid", evidence_text="..."),
    ]
    rows = entities_to_tsv_rows(alts)
    assert len(rows) == 2
    assert rows[0]["standard_name"] == "thymol"
    assert rows[0]["alternative_class"] == "Plant_Extract"
    assert rows[1]["standard_name"] == "butyric acid"


def test_postprocess_alternatives_dedup():
    data = {
        "doi": "10.1016/test",
        "alternatives": [
            {"standard_name": "thymol", "alternative_class": "Plant_Extract", "evidence_text": "a", "source_location": "M2.3"},
            {"standard_name": "thymol", "alternative_class": "Plant_Extract", "evidence_text": "b", "source_location": "M2.3"},
        ],
        "composite_products": [],
        "warnings": [],
    }
    result = postprocess_alternatives(data)
    assert len(result["alternatives"]) == 1
