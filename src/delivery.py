"""Delivery layer — produce final cleaned outputs ready for downstream consumption.

Three outputs are produced, all written to ``output/delivery/``:

1. **Literature manifest** (``literature.csv``)
   - All articles with their metadata (title, DOI, PMID, journal, year, etc.)
   - Articles that FAILED the gate check are excluded (no valid Alternative entities).

2. **Entity review CSVs** (``entities/``)
   - One CSV per entity type, after the full cleaning pipeline.
   - Plus ``relationships.csv`` with all resolved edges.

3. **Neo4j import CSVs** (``neo4j/``)
   - ``nodes.csv`` and ``edges.csv`` ready for ``neo4j-admin import``.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# Public API
# ============================================================================


def export_delivery(
    all_results: list[Any],
    output_dir: str | Path = "output/delivery",
    *,
    registry: Any = None,
) -> dict[str, Path]:
    """Produce the three delivery outputs from cleaned extraction results.

    Parameters
    ----------
    all_results:
        List of DocumentExtractionResult objects (already cleaned).
    output_dir:
        Root directory for delivery outputs.
    registry:
        SchemaRegistry instance for gate checks and graph building.

    Returns
    -------
    dict
        Paths to each output: ``{"literature": ..., "entities": ..., "neo4j": ...}``
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Gate filter: determine which articles passed ----
    passed_results, failed_results = _filter_by_gate(all_results, registry)
    logger.info(
        "Gate filter: %d passed, %d failed (of %d total)",
        len(passed_results), len(failed_results), len(all_results),
    )

    # ---- 1. Literature manifest (passed articles only) ----
    lit_path = _export_literature_manifest(passed_results, out / "literature.csv")
    logger.info("Literature manifest: %s (%d articles)", lit_path, len(passed_results))

    # ---- 2. Entity review CSVs (passed articles, cleaned data) ----
    entities_dir = out / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)
    _export_entity_review(passed_results, entities_dir, registry)
    logger.info("Entity review CSVs: %s/", entities_dir)

    # ---- 3. Neo4j import CSVs ----
    neo4j_dir = out / "neo4j"
    _export_neo4j_delivery(passed_results, neo4j_dir, registry)
    logger.info("Neo4j CSVs: %s/", neo4j_dir)

    # ---- Summary ----
    _write_delivery_summary(out, passed_results, failed_results, all_results)

    return {
        "literature": lit_path,
        "entities": entities_dir,
        "neo4j": neo4j_dir,
    }


# ============================================================================
# Gate filtering
# ============================================================================


def _filter_by_gate(
    all_results: list[Any],
    registry: Any = None,
) -> tuple[list[Any], list[Any]]:
    """Split results into passed and failed based on gate evaluation.

    The gate checks that at least one Alternative entity has a valid
    classification (not 'Other', '', or 'unmatched').  Articles with no
    valid Alternative are excluded from delivery.
    """
    if registry is None:
        return list(all_results), []

    # Find the gate definition from the extraction phases
    gate_def = None
    try:
        for phase in registry.phase_defs():
            if phase.gate and phase.gate.entity == "Alternative":
                gate_def = phase.gate
                break
    except AttributeError:
        pass

    passed: list[Any] = []
    failed: list[Any] = []

    for result in all_results:
        exts = getattr(result, "extractions", []) or []
        doc_id = getattr(result, "document_id", "?")

        if gate_def is None:
            passed.append(result)
            continue

        # Check if any Alternative passes the gate condition
        alt_passes = any(
            ext.extraction_class == "Alternative"
            and _check_gate_condition(ext, gate_def)
            for ext in exts
        )

        if alt_passes:
            passed.append(result)
        else:
            failed.append(result)
            logger.debug("Gate failed: %s (no valid Alternative)", doc_id)

    return passed, failed


def _check_gate_condition(ext: Any, gate_def: Any) -> bool:
    """Check if a single Alternative extraction passes the gate condition."""
    condition = getattr(gate_def, "condition", "")
    if not condition:
        return True

    # Parse "classification not in ['Other', '', 'unmatched']"
    import re
    m = re.match(r"(\w+)\s+(not\s+in|in)\s+\[(.+)\]", condition.strip())
    if not m:
        return True  # unparseable → pass

    field = m.group(1)
    operator = m.group(2).replace(" ", "_")
    values_str = m.group(3)
    # Parse the list: strip quotes from each entry
    excluded = {
        v.strip().strip("'").strip('"')
        for v in values_str.split(",")
        if v.strip().strip("'").strip('"')
    }

    attrs = ext.attributes or {}
    val = attrs.get(field, "")
    if isinstance(val, str):
        val = val.strip()

    if operator == "not_in":
        return bool(val) and val.lower() not in {e.lower() for e in excluded}
    elif operator == "in":
        return val.lower() in {e.lower() for e in excluded}
    return True


# ============================================================================
# 1. Literature manifest
# ============================================================================


def _export_literature_manifest(
    passed_results: list[Any],
    path: Path,
) -> Path:
    """Write literature.csv with one row per passed article."""
    # Collect literature metadata from Literature entities
    rows: list[dict[str, str]] = []

    for result in passed_results:
        doc_id = getattr(result, "document_id", "")
        exts = getattr(result, "extractions", []) or []

        # Find the Literature entity for this article
        lit_ext = None
        for ext in exts:
            if ext.extraction_class == "Literature":
                lit_ext = ext
                break

        row = {"source_doc": doc_id}
        if lit_ext and lit_ext.attributes:
            for key, val in lit_ext.attributes.items():
                if not key.startswith("_") and isinstance(val, (str, int, float)):
                    row[key] = str(val)
        rows.append(row)

    # Collect all column names
    all_cols: list[str] = ["source_doc"]
    seen_cols = {"source_doc"}
    for row in rows:
        for col in row:
            if col not in seen_cols:
                all_cols.append(col)
                seen_cols.add(col)

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return path


# ============================================================================
# 2. Entity review CSVs
# ============================================================================


def _export_entity_review(
    passed_results: list[Any],
    entities_dir: Path,
    registry: Any = None,
) -> None:
    """Export one CSV per entity type + relationships.csv.

    Reuses the existing PMC adapter's CSV export logic.
    """
    from src.adapters.pmc import export_entity_review_csv
    export_entity_review_csv(passed_results, entities_dir, registry)


# ============================================================================
# 3. Neo4j import CSVs
# ============================================================================


def _export_neo4j_delivery(
    passed_results: list[Any],
    neo4j_dir: Path,
    registry: Any = None,
) -> None:
    """Build graph from passed results and export Neo4j CSVs."""
    from src.graph import build_graph, export_neo4j_csv

    global_types: frozenset[str] = frozenset()
    if registry is not None:
        try:
            global_types = registry.get_global_types()
        except Exception:
            pass

    graph = build_graph(passed_results, registry, global_types=global_types)
    logger.info(
        "Delivery graph: %d nodes, %d edges",
        len(graph.nodes), len(graph.edges),
    )
    export_neo4j_csv(graph, neo4j_dir)


# ============================================================================
# Summary
# ============================================================================


def _write_delivery_summary(
    out: Path,
    passed: list[Any],
    failed: list[Any],
    all_results: list[Any],
) -> None:
    """Write a delivery summary report."""
    # Count entities by type
    from collections import Counter
    type_counts: Counter[str] = Counter()
    total_entities = 0
    for result in passed:
        for ext in (result.extractions or []):
            type_counts[ext.extraction_class] += 1
            total_entities += 1

    lines = [
        "# Delivery Summary",
        "",
        f"Articles: {len(passed)} passed / {len(failed)} failed / {len(all_results)} total",
        f"Entities: {total_entities}",
        "",
        "## Entity counts (passed articles only)",
    ]
    for t, c in sorted(type_counts.items()):
        lines.append(f"  {t}: {c}")

    if failed:
        lines.append("")
        lines.append("## Failed articles (no valid Alternative)")
        for r in failed[:50]:  # max 50
            doc_id = getattr(r, "document_id", "?")
            lines.append(f"  - {doc_id}")
        if len(failed) > 50:
            lines.append(f"  ... and {len(failed) - 50} more")

    summary_path = out / "delivery_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Delivery summary: %s", summary_path)
