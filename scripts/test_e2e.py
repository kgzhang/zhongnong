"""End-to-end extraction test using section-based adapter + coreference dedup.

Uses src.adapters.pmc_sectioned for BACKGROUND.md-compliant section routing
and strict entity deduplication.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adapters.pmc import parse_article
from src.adapters.pmc_sectioned import (
    extract_sectioned,
    run_sectioned_pipeline,
    deduplicate_extractions,
    _METHODS_ENTITIES,
    _RESULTS_ENTITIES,
)
from src.schema_registry import SchemaRegistry

XML_FILE = Path("data/xml/PMC12188611.xml")
OUTPUT_DIR = Path("output/test_e2e_v3")

if __name__ == "__main__":
    if not XML_FILE.exists():
        print(f"File not found: {XML_FILE}")
        sys.exit(1)

    # Parse XML metadata
    print("=" * 60)
    print("PMC Section-based Extraction Test")
    print("=" * 60)
    meta = parse_article(XML_FILE)
    print(f"  Title: {meta.title[:80]}...")
    print(f"  PMID: {meta.pmid}  DOI: {meta.doi}")
    print(f"  Sections: {list(meta.sections.keys())}")
    print(f"  Body: {len(meta.body_text)} chars")
    print(f"  Methods text: {len(meta.sections.get('methods', ''))} chars")
    print(f"  Results text: ~{sum(len(v) for k,v in meta.sections.items() if 'result' in k.lower() or 'discussion' in k.lower())} chars")

    # Run section-based pipeline
    registry = SchemaRegistry()
    print(f"\nEntity routing:")
    print(f"  Methods section → {_METHODS_ENTITIES}")
    print(f"  Results section → {_RESULTS_ENTITIES}")
    print(f"  Pre-extracted   → Literature (from front-matter)")

    print(f"\nStarting extraction...")
    t0 = time.time()

    result = extract_sectioned(
        XML_FILE,
        registry=registry,
    )

    elapsed = time.time() - t0

    # Results
    print(f"\n{'=' * 60}")
    print(f"RESULTS ({elapsed:.1f}s)")
    print(f"{'=' * 60}")

    type_counts: dict[str, int] = {}
    for ext in result.extractions:
        type_counts[ext.extraction_class] = type_counts.get(ext.extraction_class, 0) + 1
    print(f"Total (pre-dedup via extract_sectioned): {len(result.extractions)} entities")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    if result.warnings:
        print(f"\nWarnings:")
        for w in result.warnings:
            print(f"  ⚠ {w}")

    # Export deduplicated CSVs
    print(f"\nExporting review CSVs with strict coreference dedup...")
    from src.adapters.pmc import export_entity_review_csv
    export_entity_review_csv([result], OUTPUT_DIR, registry, deduplicate=True)

    # Show dedup stats
    from src.adapters.pmc_sectioned import deduplicate_extractions
    deduped = deduplicate_extractions(result.extractions, registry)
    dedup_counts: dict[str, int] = {}
    for ext in deduped:
        dedup_counts[ext.extraction_class] = dedup_counts.get(ext.extraction_class, 0) + 1
    print(f"\nAfter strict dedup:")
    for t, c in sorted(dedup_counts.items()):
        orig = type_counts.get(t, 0)
        if orig != c:
            print(f"  {t}: {orig} → {c} (merged {orig - c})")
        else:
            print(f"  {t}: {c}")

    # Show sample deduplicated entities with merged evidence
    print(f"\nSample deduplicated entities (first 2 per type):")
    shown: dict[str, int] = {}
    for ext in deduped:
        etype = ext.extraction_class
        shown.setdefault(etype, 0)
        if shown[etype] >= 2:
            continue
        shown[etype] += 1

        ev = getattr(ext, "evidence_text", "")
        src = getattr(ext, "source_location", "")
        merged_count = ev.count(" <|> ") + 1 if ev else 0
        print(f"  [{etype}] {ext.extraction_text}")
        print(f"         evidence sources: {merged_count}")
        if src:
            print(f"         locations: {src}")
        if merged_count > 1:
            print(f"         (merged from {merged_count} occurrences)")

    print(f"\nDone! Review CSVs at {OUTPUT_DIR}/review/")
