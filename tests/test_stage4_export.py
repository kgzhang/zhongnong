import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from stage4_export_tsv import write_entities_tsv, ENTITY_COLUMNS
from stage4_export_neo4j import generate_cypher, generate_indexes, NEO4J_LABEL_MAP
from models import Alternative, Relationship


def test_write_entities_tsv():
    entities = [
        Alternative(entity_id="1", standard_name="thymol", alternative_class="Plant_Extract", evidence_text="...", source_location="M2.3"),
        Alternative(entity_id="2", standard_name="butyric acid", alternative_class="Organic_Acid", evidence_text="...", source_location="M2.3"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = write_entities_tsv(entities, "Alternative", Path(tmp))
        assert path.exists()
        import pandas as pd
        df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
        assert len(df) == 2
        assert df.iloc[0]["standard_name"] == "thymol"


def test_entity_columns_has_all_13_types():
    types = ["Alternative", "Alternative_Class", "Composite_Product", "Literature", "Experiment",
             "Swine_Model", "Swine", "Intervention", "Control_Group", "Tissue_Site",
             "Indicator", "Result", "Method"]
    for t in types:
        assert t in ENTITY_COLUMNS, f"Missing columns for {t}"


def test_generate_cypher_creates_node():
    entities = [Alternative(entity_id="1", standard_name="thymol", alternative_class="Plant_Extract", evidence_text="...", source_location="M2.3")]
    cypher = generate_cypher(entities, "Alternative")
    assert "MERGE" in cypher
    assert ":Alternative" in cypher or "Alternative" in cypher
    assert "standard_name" in cypher
    assert "thymol" in cypher


def test_generate_cypher_escapes_quotes():
    entities = [Alternative(entity_id="1", standard_name='test with "quotes"', alternative_class="Other", evidence_text="...")]
    cypher = generate_cypher(entities, "Alternative")
    assert '\\"' in cypher or '"test with' in cypher


def test_generate_cypher_null_skipped():
    """None values should be skipped in SET clauses."""
    entities = [Alternative(entity_id="1", standard_name="thymol", alternative_class="Plant_Extract", abbreviation=None, cas_number=None, evidence_text="...")]
    cypher = generate_cypher(entities, "Alternative")
    # Should not contain 'null' in SET clause
    lines = [l for l in cypher.split("\n") if l.startswith("SET")]
    assert not any("null" in l for l in lines)


def test_generate_indexes():
    indexes = generate_indexes()
    assert "CREATE INDEX" in indexes
    assert ":Alternative" in indexes
    assert ":Literature" in indexes


def test_neo4j_label_map_complete():
    types = ["Alternative", "Alternative_Class", "Composite_Product", "Literature", "Experiment",
             "Swine_Model", "Swine", "Intervention", "Control_Group", "Tissue_Site",
             "Indicator", "Result", "Method"]
    for t in types:
        assert t in NEO4J_LABEL_MAP, f"Missing label for {t}"
