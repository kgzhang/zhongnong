from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from glossary import GlossaryIndex


def test_parse_and_lookup():
    gi = GlossaryIndex()
    gi.load("ALTERNATIVE.tsv")
    result = gi.lookup("thymol")
    assert result is not None
    assert result["class"] == "Plant_Extract"
    assert result["subclass"] == "Volatile oils and Terpenoids"
    assert result["standard_name"] == "thymol"


def test_synonym_mapping():
    gi = GlossaryIndex()
    gi.load("ALTERNATIVE.tsv")
    result = gi.lookup("Lactobacillus plantarum")
    assert result is not None
    assert result["standard_name"] == "Lactiplantibacillus plantarum"
    assert result["class"] == "Probiotic"


def test_fuzzy_match():
    gi = GlossaryIndex()
    gi.load("ALTERNATIVE.tsv")
    result = gi.lookup("zinc oxide")
    assert result is not None
    assert result["standard_name"] == "zinc oxide (ZnO)"
    assert result["class"] == "Trace_Element"


def test_not_found_returns_other():
    gi = GlossaryIndex()
    gi.load("ALTERNATIVE.tsv")
    result = gi.lookup("novel_compound_xyz")
    assert result is not None
    assert result["class"] == "Other"
    assert result["match_source"] == "Other_未匹配"
