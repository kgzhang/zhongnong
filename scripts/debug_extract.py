"""Direct extract() call — convenient for debugging with breakpoints.

Edit the XML path below, then launch "Direct: extract() call" from VS Code.
"""
from pathlib import Path

from src.extraction import extract
from src.graph import build_graph, export_neo4j_csv
from src.schema_registry import SchemaRegistry

# --- EDIT THESE ---
XML_FILE = Path("data/xml/PMC12183824.xml")
SKIP_GATE = False
OUTPUT_DIR = Path("output")
# -----------------

if __name__ == "__main__":
    result = extract(str(XML_FILE), skip_if_no_known_alternative=not SKIP_GATE)
    print(f"\n=== Result ===")
    print(f"Entities: {len(result.extractions)}")
    print(f"Skipped: {result.skipped} ({result.skip_reason})")

    if result.extractions:
        type_counts: dict[str, int] = {}
        for ext in result.extractions:
            type_counts[ext.extraction_class] = type_counts.get(ext.extraction_class, 0) + 1
        for t, c in sorted(type_counts.items()):
            print(f"  {t}: {c}")

        registry = SchemaRegistry()
        graph = build_graph([result], registry)
        print(f"\nGraph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        export_neo4j_csv(graph, OUTPUT_DIR)
        print(f"Exported to {OUTPUT_DIR}/")
