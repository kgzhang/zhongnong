#!/usr/bin/env python3
"""Extract entities from PMC12188611.xml using the new schema."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.adapters.pmc_sectioned import run_sectioned_pipeline
from src.schema_registry import SchemaRegistry
from src.config import setup_logging

setup_logging("INFO")

registry = SchemaRegistry("schemas")
print(f"Loaded {len(registry.all_entity_names())} entity types, "
      f"{len(registry.all_relation_names())} relations")

output_dir = Path("output/test_e2e_v3")
output_dir.mkdir(parents=True, exist_ok=True)

result = run_sectioned_pipeline(
    "data/xml/PMC12188611.xml",
    output_dir=output_dir,
    registry=registry,
)

print(f"\nDone! {len(result['result'].extractions)} total entities extracted")
print(f"Warnings: {len(result['result'].warnings)}")

# Print entity type counts
type_counts: dict[str, int] = {}
for ext in result['result'].extractions:
    type_counts[ext.extraction_class] = type_counts.get(ext.extraction_class, 0) + 1
for t, c in sorted(type_counts.items()):
    print(f"  {t}: {c}")

print(f"\nOutput: {output_dir}/")
