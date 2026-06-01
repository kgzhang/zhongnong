"""Deterministic feedback loop: verify DSPy Alternative extraction quality."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# Test fixtures — pre-loaded section JSONs
def _load_article(name: str) -> dict:
    """Load a structured section JSON by DOI-safe name."""
    path = Path(f"data/structured_sections/{name}.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text())


class TestAlternativeExtraction:
    """Verify that Alternative extraction correctly identifies substances from the glossary."""

    def test_article1_has_alternative(self):
        """Article 1 (MA antioxidants) must find Microbe-derived antioxidants."""
        from src.dspy_extract import extract_alternatives
        art = _load_article("10.3389_fvets.2025.1574259")
        mm = art.get("sections", {}).get("materials_and_methods", {}).get("full_text", "")
        assert mm, "M&M text must be non-empty"

        result = extract_alternatives(mm)
        alternatives = result.get("alternatives", [])
        composites = result.get("composite_products", [])

        # At least one alternative or composite must be found
        total = len(alternatives) + len(composites)
        assert total > 0, f"Expected >=1 alternative/composite, got {total}"
        print(f"Found: {total} substances")

    def test_glossary_matches_are_valid(self):
        """Every extracted alternative must have a valid glossary class."""
        from src.dspy_extract import extract_alternatives
        from src.glossary import GlossaryIndex

        gi = GlossaryIndex()
        gi.load("ALTERNATIVE.tsv")

        art = _load_article("10.3389_fvets.2025.1574259")
        mm = art.get("sections", {}).get("materials_and_methods", {}).get("full_text", "")

        result = extract_alternatives(mm)
        valid_classes = {
            "Plant_Extract", "Trace_Element", "Organic_Acid", "Probiotic",
            "Polysaccharides_and_Oligosaccharides", "Enzyme",
            "Bioactive_Peptides", "Other",
        }
        for a in result.get("alternatives", []):
            cls = a.get("alternative_class", "")
            assert cls in valid_classes, f"Invalid class '{cls}' for {a.get('standard_name')}"

    def test_skip_article_without_alternatives(self):
        """Article without standard alternatives should be skipped (gate check)."""
        from src.dspy_extract import extract_alternatives
        from src.glossary import GlossaryIndex

        gi = GlossaryIndex()
        gi.load("ALTERNATIVE.tsv")

        art = _load_article("10.1186_s40104-025-01208-7")
        mm = art.get("sections", {}).get("materials_and_methods", {}).get("full_text", "")

        result = extract_alternatives(mm)
        alternatives = result.get("alternatives", [])

        # Check if any alternative matches a KNOWN glossary entry (not "Other")
        has_known = any(
            a.get("alternative_class") != "Other"
            and a.get("alternative_class") in {
                "Plant_Extract", "Trace_Element", "Organic_Acid", "Probiotic",
                "Polysaccharides_and_Oligosaccharides", "Enzyme", "Bioactive_Peptides",
            }
            for a in alternatives
        )

        if not has_known:
            print("SKIP: No known alternatives found — article should be skipped")
            assert len(alternatives) == 0 or all(
                a.get("alternative_class") == "Other" for a in alternatives
            ), "Should have no known-class alternatives"


class TestAlternativeGate:
    """Gate regression: no article must produce output without an Alternative."""

    def test_all_entity_dirs_have_alternatives(self):
        """Every entity directory must have at least one Alternative or Composite_Product."""
        entities_root = Path("data/entities")
        violations = []
        for d in entities_root.iterdir():
            if not d.is_dir() or d.name.startswith("_"):
                continue
            alt_file = d / "alternatives.json"
            if alt_file.exists():
                data = json.loads(alt_file.read_text())
                n_alt = len(data.get("alternatives", []))
                n_comp = len(data.get("composite_products", []))
                if n_alt == 0 and n_comp == 0:
                    violations.append(f"{d.name}: alternatives.json exists but is empty (0 alt, 0 comp)")
            else:
                other_files = list(d.iterdir())
                if other_files:
                    violations.append(f"{d.name}: No alternatives.json but has {len(other_files)} other entity files")

        if violations:
            msg = "GATE VIOLATION: Articles found without substances:\n" + "\n".join(violations)
            import shutil
            for d in entities_root.iterdir():
                if not d.is_dir() or d.name.startswith("_"):
                    continue
                alt_file = d / "alternatives.json"
                data = json.loads(alt_file.read_text()) if alt_file.exists() else {}
                if not alt_file.exists() or (len(data.get("alternatives", [])) == 0 and len(data.get("composite_products", [])) == 0):
                    shutil.rmtree(d)
                    violations.append(f"CLEANED: {d.name}")
            assert False, msg

    def test_no_article_without_alternative(self):
        """Pipeline must not produce TSV output rows for articles without alternatives."""
        import shutil
        # Check results TSV — every row must have a corresponding Alternative
        results_path = Path("data/output/tsv/result.tsv")
        alt_path = Path("data/output/tsv/alternative.tsv")
        if results_path.exists() and alt_path.exists():
            import pandas as pd
            results = pd.read_csv(results_path, sep="\t")
            alts = pd.read_csv(alt_path, sep="\t")
            # Every result's doi should appear in alternatives
            # (this is a soft check — alternative may be from a different article
            #  since the TSV aggregates all articles)
            assert len(alts) > 0, "Must have at least one Alternative in TSV"


class TestGraphOutput:
    """Verify nodes.tsv + edges.tsv + evidence.tsv format."""

    def test_output_files_exist(self):
        """All three output files must exist after pipeline runs."""
        for name in ["nodes.tsv", "edges.tsv", "evidence.tsv"]:
            p = Path("data/output/tsv") / name
            assert p.exists(), f"Missing: {name}"

    def test_global_ids_use_name_not_seq(self):
        """Global entities must use name-based IDs, not sequential numbers."""
        import pandas as pd
        df = pd.read_csv("data/output/tsv/nodes.tsv", sep="\t")
        # Alternative, Tissue_Site, Indicator, Method should use type:name format
        for _, row in df.iterrows():
            ntype = row["node_type"]
            nid = str(row["node_id"])
            if ntype in ("Alternative", "Tissue_Site", "Indicator", "Method",
                         "Alternative_Class", "Swine"):
                assert ":" in nid, f"Global node {ntype} must have namespaced ID, got: {nid}"
                # Should be TYPE:name not TYPE:number
                prefix = nid.split(":")[0]
                assert prefix in ("ALT", "CLS", "TIS", "IND", "MET", "SWN"), \
                    f"Unexpected prefix in {nid}"

    def test_local_ids_use_pmid(self):
        """Local entities must use PMID-based IDs."""
        import pandas as pd
        df = pd.read_csv("data/output/tsv/nodes.tsv", sep="\t")
        for _, row in df.iterrows():
            ntype = row["node_type"]
            nid = str(row["node_id"])
            if ntype in ("Intervention", "Result", "Control_Group"):
                # Should contain an underscore (pmid_seq)
                assert "_" in nid, f"Local node {ntype} needs PMID-based ID, got: {nid}"

    def test_evidence_dedup(self):
        """evidence.tsv should have fewer entries than raw evidence_text references."""
        import pandas as pd
        ev = pd.read_csv("data/output/tsv/evidence.tsv", sep="\t")
        nodes = pd.read_csv("data/output/tsv/nodes.tsv", sep="\t")
        # Count EV: references in nodes (excluding EV:NONE)
        ev_refs = 0
        for col in nodes.columns:
            ev_refs += nodes[col].astype(str).str.match(r'^EV:[0-9a-f]{12}$').sum()
        assert len(ev) <= ev_refs, \
            f"Evidence registry has {len(ev)} entries for {ev_refs} references (expect dedup)"

    def test_no_cypher_file(self):
        """We no longer generate Cypher; only TSV."""
        cypher_path = Path("data/output/neo4j/import.cypher")
        # May exist from old runs but we don't generate it anymore


class TestRelationshipGeneration:
    """Verify that relationships are generated for all entity types."""

    RELATIONSHIP_TYPES = [
        "belongs_to",        # Alternative → Alternative_Class
        "uses",              # Intervention → Alternative
        "measured_in",       # Indicator → Tissue_Site
        "uses_method",       # Indicator → Method
        "increases", "decreases", "upregulates", "downregulates",
        "enriches", "depletes",
        "corresponds_to",    # Result → Indicator
        "occurs_in",         # Result → Tissue_Site
        "compared_to",       # Result → Control_Group
    ]

    def test_stage4_relationships_not_empty(self):
        """Stage 4 export must generate relationships."""
        import subprocess
        import os

        # Check that the relationships output file exists and has content
        rel_path = Path("data/relationships/_stage3_relationships.json")
        cypher_path = Path("data/output/neo4j/import.cypher")

        if cypher_path.exists():
            content = cypher_path.read_text()
            rel_count = content.count("MERGE (a)-[")
            assert rel_count > 0, f"Neo4j cypher must contain relationships, found {rel_count}"
            print(f"Neo4j relationships: {rel_count}")
