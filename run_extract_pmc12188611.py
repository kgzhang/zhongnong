#!/usr/bin/env python3
"""Extract entities from PMC12188611.xml — thin wrapper around batch_extract_pmc.

For full control use the CLI or scripts/batch_extract.py instead::

    uv run python -m src.cli batch --input data/xml/PMC12188611.xml --output output/my_run
    python scripts/batch_extract.py --input data/xml/PMC12188611.xml --output output/my_run
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.extraction import batch_extract_pmc

if __name__ == "__main__":
    input_xml = "data/xml/PMC12188611.xml"
    output_dir = "output/test_e2e_v3"

    result = batch_extract_pmc(input_xml, output_dir=output_dir)

    total = sum(len(r.extractions) for r in result["results"])
    graph = result["graph"]
    print(f"\nDone! {total} total entities, "
          f"{len(graph.nodes)} graph nodes, {len(graph.edges)} graph edges")

    type_counts: dict[str, int] = {}
    for r in result["results"]:
        for ext in r.extractions:
            type_counts[ext.extraction_class] = (
                type_counts.get(ext.extraction_class, 0) + 1
            )
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    print(f"\nOutput: {output_dir}/")
