import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models import Alternative, Literature, Result, Relationship


def test_alternative_defaults():
    a = Alternative(entity_id="10.1016/xxx_ALT_01", standard_name="thymol")
    assert a.alternative_class == ""
    assert a.is_synthetic is False
    assert a.abbreviation is None


def test_alternative_full():
    a = Alternative(
        entity_id="10.1016/xxx_ALT_01",
        standard_name="thymol",
        alternative_class="Plant_Extract",
        evidence_text="thymol was supplemented...",
        source_location="M2.3"
    )
    assert a.alternative_class == "Plant_Extract"
    assert a.evidence_text == "thymol was supplemented..."


def test_literature_defaults():
    l = Literature(doi="10.1016/test.2024.001")
    assert l.title == ""
    assert l.publication_year is None


def test_result_direction():
    r = Result(entity_id="1", doi="x", experiment_id="E1", direction="increased")
    assert r.direction == "increased"


def test_relationship():
    rel = Relationship(
        rel_type="belongs_to",
        head_entity_type="Alternative",
        head_entity_id="ALT_1",
        tail_entity_type="Alternative_Class",
        tail_entity_id="Plant_Extract"
    )
    assert rel.rel_type == "belongs_to"
