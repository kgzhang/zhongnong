import json
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from stage2_entity_extract import (
    normalize_dose_unit, build_entity_id, entities_to_tsv_rows,
    postprocess_alternatives, postprocess_experiment_design,
    load_section_json, checkpoint_exists, save_entity_output,
    list_section_files,
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


# --- Orchestrator helper tests ---

def test_load_section_json():
    """load_section_json reads and parses a section JSON file."""
    with tempfile.TemporaryDirectory() as tmp:
        sections_dir = Path(tmp) / "sections"
        sections_dir.mkdir()
        article = {"doi": "10.1016/test", "title": "Test Article", "sections": {}}
        (sections_dir / "10.1016_test.json").write_text(json.dumps(article))

        result = load_section_json("10.1016_test", sections_dir)
        assert result is not None
        assert result["doi"] == "10.1016/test"


def test_load_section_json_missing():
    """Returns None for non-existent file."""
    result = load_section_json("nonexistent_doi", Path("/tmp/nonexistent"))
    assert result is None


def test_checkpoint_exists(tmp_path):
    """checkpoint_exists detects existing output files."""
    entities_dir = tmp_path / "entities"
    entities_dir.mkdir()
    # No file yet
    assert checkpoint_exists("doi1", "alternatives", entities_dir) is False
    # Create output
    (entities_dir / "doi1").mkdir(parents=True)
    (entities_dir / "doi1" / "alternatives.json").write_text("{}")
    assert checkpoint_exists("doi1", "alternatives", entities_dir) is True


def test_save_and_load_entity_output(tmp_path):
    """save_entity_output writes JSON that can be read back."""
    entities_dir = tmp_path / "entities"
    data = {"doi": "10.1016/x", "alternatives": [{"standard_name": "thymol"}]}
    path = save_entity_output("doi1", "alternatives", data, entities_dir)
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["alternatives"][0]["standard_name"] == "thymol"


def test_list_section_files(tmp_path):
    """list_section_files finds all JSON files in directory."""
    sections_dir = tmp_path / "sections"
    sections_dir.mkdir()
    (sections_dir / "a.json").write_text("{}")
    (sections_dir / "b.json").write_text("{}")
    (sections_dir / "not_json.txt").write_text("")
    files = list_section_files(sections_dir)
    json_files = [f for f in files if f.suffix == ".json"]
    assert len(json_files) == 2  # a.json, b.json


def test_postprocess_experiment_design():
    """Dose units are normalized during postprocessing."""
    data = {
        "interventions": [
            {"dose_value": 500, "dose_unit_original": "ppm", "dose_unit_standard": ""},
            {"dose_value": 0.05, "dose_unit_original": "%", "dose_unit_standard": ""},
        ]
    }
    result = postprocess_experiment_design(data)
    assert result["interventions"][0]["dose_unit_standard"] == "mg/kg_feed"
    assert result["interventions"][1]["dose_value"] == 500  # 0.05% = 500 ppm
    assert result["interventions"][1]["dose_unit_standard"] == "mg/kg_feed"
