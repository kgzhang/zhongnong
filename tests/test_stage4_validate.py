from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from stage4_validate import (
    ValidationReport, validate_required_fields, validate_enum_values,
    validate_foreign_keys, validate_business_rules,
)
from models import Alternative, SwineModel, Result, ControlGroup, CompositeProduct


def test_validation_report_add_and_summary():
    report = ValidationReport()
    report.add("FATAL", "E1", "Required field missing")
    report.add("WARNING", "E2", "Suspicious value")
    report.add("INFO", None, "Processed 100 entities")
    summary = report.summary()
    assert summary["FATAL"] == 1
    assert summary["WARNING"] == 1
    assert summary["INFO"] == 1


def test_validation_report_empty():
    report = ValidationReport()
    assert report.summary() == {"FATAL": 0, "WARNING": 0, "INFO": 0}


def test_validate_required_fields_passes():
    entities = [Alternative(entity_id="1", standard_name="thymol", alternative_class="Plant_Extract", evidence_text="...")]
    report = ValidationReport()
    validate_required_fields(entities, "Alternative", ["entity_id", "standard_name", "alternative_class", "evidence_text"], report)
    assert len(report.fatals) == 0


def test_validate_required_fields_fails():
    entities = [Alternative(entity_id="", standard_name="", alternative_class="", evidence_text="")]
    report = ValidationReport()
    validate_required_fields(entities, "Alternative", ["entity_id", "standard_name", "alternative_class", "evidence_text"], report)
    assert len(report.fatals) >= 1


def test_validate_enum_values_ok():
    entities = [SwineModel(entity_id="1", experiment_id="E1", model_type="challenge")]
    report = ValidationReport()
    validate_enum_values(entities, "Swine_Model", {"model_type": ["challenge", "normal"]}, report)
    assert len(report.fatals) == 0


def test_validate_enum_values_bad():
    entities = [SwineModel(entity_id="1", experiment_id="E1", model_type="challenged")]
    report = ValidationReport()
    validate_enum_values(entities, "Swine_Model", {"model_type": ["challenge", "normal"]}, report)
    assert len(report.fatals) >= 1


def test_validate_enum_values_skip_none():
    """None values should be skipped (optional fields)."""
    entities = [SwineModel(entity_id="1", experiment_id="E1", model_type="")]
    report = ValidationReport()
    validate_enum_values(entities, "Swine_Model", {"model_type": ["challenge", "normal"]}, report)
    assert len(report.fatals) == 0  # empty string is not validated


def test_validate_foreign_keys():
    results = [
        Result(entity_id="R1", doi="x", experiment_id="E1", matched_indicator="I1", matched_tissue="T1"),
        Result(entity_id="R2", doi="x", experiment_id="E1", matched_indicator="I99", matched_tissue="T1"),
    ]
    indicators = {"I1": "ind_1"}
    tissues = {"T1": "tis_1"}
    report = ValidationReport()
    validate_foreign_keys(results, indicators, tissues, {}, {}, report)
    assert len(report.warnings) >= 1  # I99 not found


def test_validation_report_to_dict():
    report = ValidationReport()
    report.add("FATAL", "E1", "msg")
    d = report.to_dict()
    assert "summary" in d
    assert "fatals" in d
    assert len(d["fatals"]) == 1
