from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from stage3_result_extract import (
    align_result_to_indicator, build_relation_records,
    check_direction_consistency, check_compared_to_baseline,
)
from models import Indicator, TissueSite, ControlGroup, Result, Relationship


def test_align_result_exact_match():
    indicators = [
        Indicator(entity_id="I1", doi="x", standard_name="Average Daily Gain", abbreviation="ADG"),
        Indicator(entity_id="I2", doi="x", standard_name="ZO-1 expression", abbreviation="ZO-1"),
    ]
    match = align_result_to_indicator({"indicator_abbreviation": "ADG"}, indicators)
    assert match == "I1"


def test_align_result_no_match():
    indicators = [Indicator(entity_id="I1", doi="x", abbreviation="ADG")]
    match = align_result_to_indicator({"indicator_abbreviation": "unknown"}, indicators)
    assert match is None


def test_align_result_empty_abbr():
    indicators = [Indicator(entity_id="I1", doi="x", abbreviation="ADG")]
    match = align_result_to_indicator({"indicator_abbreviation": ""}, indicators)
    assert match is None


def test_direction_consistency():
    assert check_direction_consistency("increases", "increased") is True
    assert check_direction_consistency("upregulates", "increased") is True
    assert check_direction_consistency("depletes", "decreased") is True
    assert check_direction_consistency("increases", "decreased") is False
    assert check_direction_consistency("affects", "increased") is True
    assert check_direction_consistency("affects", "decreased") is True


def test_compared_to_baseline_valid():
    controls = [
        ControlGroup(entity_id="C1", experiment_id="E1", group_name="Challenged Control", group_type="negative_control"),
        ControlGroup(entity_id="C2", experiment_id="E1", group_name="Normal Control", group_type="basal_control"),
    ]
    result = {"compared_to_group": "Challenged Control"}
    ok, msg = check_compared_to_baseline(result, controls, is_challenge_model=True)
    assert ok is True


def test_compared_to_baseline_invalid():
    controls = [
        ControlGroup(entity_id="C1", experiment_id="E1", group_name="Challenged Control", group_type="negative_control"),
        ControlGroup(entity_id="C2", experiment_id="E1", group_name="Normal Control", group_type="basal_control"),
    ]
    result = {"compared_to_group": "Normal Control"}
    ok, msg = check_compared_to_baseline(result, controls, is_challenge_model=True)
    assert ok is False


def test_compared_to_baseline_non_challenge():
    controls = []
    ok, msg = check_compared_to_baseline({}, controls, is_challenge_model=False)
    assert ok is True


def test_build_relation_records():
    result = Result(
        entity_id="R1", doi="x", experiment_id="E1",
        direction="increased", relation_type="increases",
        compared_to_group="Challenged Control",
        evidence_text="...", source_location="Results 3.1"
    )
    rels = build_relation_records(result, "INT_1", "IND_1", "TIS_1", "CTL_1")
    assert len(rels) == 4
    rel_types = {r.rel_type for r in rels}
    assert "increases" in rel_types
    assert "corresponds_to" in rel_types
    assert "occurs_in" in rel_types
    assert "compared_to" in rel_types


def test_build_relation_records_partial():
    """Relations should still work with missing IDs."""
    result = Result(entity_id="R1", doi="x", experiment_id="E1", relation_type="increases")
    rels = build_relation_records(result, "", "", "", None)
    assert len(rels) == 0  # No intervention, no matches -> no relations
