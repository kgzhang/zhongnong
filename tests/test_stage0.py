from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from stage0_search import build_query, dedup_by_doi


def test_build_query():
    query = build_query("thymol")
    assert "thymol" in query
    pig_keywords = ["pig", "swine", "piglet"]
    assert any(kw in query.lower() for kw in pig_keywords)
    assert query != ""


def test_build_query_another_substance():
    query = build_query("zinc oxide")
    assert "zinc oxide" in query
    assert any(kw in query.lower() for kw in ["pig", "swine", "piglet"])


def test_dedup_by_doi():
    new_results = [
        {"doi": "10.1016/a.2023.1", "pmid": "111", "title": "A"},
        {"doi": "10.1016/b.2023.2", "pmid": "222", "title": "B"},
    ]
    existing_dois = {"10.1016/a.2023.1", "10.1016/c.2023.3"}
    merged, novel = dedup_by_doi(new_results, existing_dois)
    assert len(merged) == 3     # B + 2 existing DOIs counted
    assert len(novel) == 1      # only B is new
    assert novel[0]["doi"] == "10.1016/b.2023.2"


def test_dedup_no_existing():
    new_results = [{"doi": "10.1016/x.1", "pmid": "1", "title": "X"}]
    merged, novel = dedup_by_doi(new_results, set())
    assert len(merged) == 1
    assert len(novel) == 1


def test_dedup_empty_new():
    merged, novel = dedup_by_doi([], {"10.1016/x.1"})
    assert len(merged) == 0
    assert len(novel) == 0


def test_dedup_missing_doi_skipped():
    new_results = [
        {"doi": "", "pmid": "1", "title": "No DOI"},
        {"doi": "10.1016/valid.1", "pmid": "2", "title": "Has DOI"},
    ]
    merged, novel = dedup_by_doi(new_results, set())
    assert len(merged) == 1   # only the one with a DOI
    assert len(novel) == 1
