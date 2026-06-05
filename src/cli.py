"""CLI entry point for llm-extract — Thin wrapper over the extraction API.

Usage::

    # Single file
    llm-extract batch --input data/xml/PMC12188611.xml --output output/my_run

    # Directory (all *.xml files, combined into one graph)
    llm-extract batch --input data/xml/ --output output/batch_run

    # Directory with custom glob
    llm-extract batch --input data/xml/ --glob "PMC*.xml" --output output/run

    # Disable gate (always run full extraction)
    llm-extract batch --input data/xml/ --no-skip-gate --output output/run

    # Debug: inspect a single file
    llm-extract debug --input data/xml/PMC12188611.xml
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """llm-extract — Knowledge Graph Extraction Pipeline."""


# ---------------------------------------------------------------------------
# batch — single file or directory → combined graph
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--input", "-i", "input_path", required=True,
    help="Path to a PMC XML file or a directory of XML files.",
)
@click.option(
    "--output", "-o", "output_dir", default="output",
    help="Output directory (review/ and graph/ subdirectories created here).",
)
@click.option(
    "--glob", "glob_pattern", default="*.xml",
    help="File-matching pattern when INPUT is a directory (default: *.xml).",
)
@click.option(
    "--no-skip-gate", is_flag=True, default=False,
    help="Disable gate — always run the full extraction even when the gate fails.",
)
@click.option(
    "--model", "model_name", default=None,
    help="Model name override (provider-specific).",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False,
    help="Enable DEBUG-level logging.",
)
@click.option(
    "--workers", "-w", "max_workers", type=int, default=None,
    help="Maximum parallel workers (default: from config, currently 4).",
)
def batch(input_path, output_dir, glob_pattern, no_skip_gate, model_name, verbose, max_workers):
    """Extract entities from one or more PMC XML files and export a combined graph.

    INPUT can be a single XML file or a directory.  When a directory is
    provided, all files matching --glob (default: ``*.xml``) are processed
    and their results are merged into one graph.
    """
    from src.extraction import batch_extract_pmc
    from src.config import setup_logging

    setup_logging("DEBUG" if verbose else "INFO")

    click.echo(f"Input:    {input_path}")
    click.echo(f"Output:   {output_dir}")
    click.echo(f"Pattern:  {glob_pattern}")
    click.echo(f"Gate:     {'disabled' if no_skip_gate else 'enabled'}")
    click.echo(f"Workers:  {max_workers or 'auto'}")

    try:
        result = batch_extract_pmc(
            input_path,
            output_dir=output_dir,
            glob_pattern=glob_pattern,
            skip_gate=no_skip_gate,
            max_workers=max_workers,
        )
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    total = sum(len(r.extractions) for r in result["results"])
    graph = result["graph"]
    click.echo()
    click.echo(f"Done. {total} entities, {len(graph.nodes)} graph nodes, "
               f"{len(graph.edges)} graph edges")
    click.echo(f"Review CSVs:  {result['output_dir']}/review/")
    click.echo(f"Neo4j CSVs:   {result['output_dir']}/graph/")

    # Per-type summary
    type_counts: dict[str, int] = {}
    for r in result["results"]:
        for ext in r.extractions:
            type_counts[ext.extraction_class] = (
                type_counts.get(ext.extraction_class, 0) + 1
            )
    if type_counts:
        click.echo()
        click.echo("Entity type counts:")
        for t, c in sorted(type_counts.items()):
            click.echo(f"  {t}: {c}")


# ---------------------------------------------------------------------------
# debug — inspect a single file
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--input", "-i", "input_path", required=True,
    help="Path to a PMC XML file.",
)
@click.option(
    "--max", "-n", "max_entities", default=20,
    help="Maximum entities to display per type (default: 20).",
)
def debug(input_path, max_entities):
    """Debug: extract entities from a single PMC XML file and print them.

    Shows extraction_text, evidence_text, and source_location for each
    entity, grouped by entity type.
    """
    from src.adapters.pmc_sectioned import extract_sectioned
    from src.schema_registry import SchemaRegistry
    from src.config import setup_logging

    setup_logging("INFO")

    registry = SchemaRegistry()
    click.echo(f"Extracting from: {input_path}")

    result = extract_sectioned(input_path, registry=registry)
    total = len(result.extractions)

    click.echo(f"\n{total} entities extracted")
    if result.warnings:
        for w in result.warnings[:10]:
            click.echo(f"  ⚠ {w}")

    # Group by entity type
    by_type: dict[str, list] = {}
    for ext in result.extractions:
        by_type.setdefault(ext.extraction_class, []).append(ext)

    for etype, exts in sorted(by_type.items()):
        click.echo(f"\n─── {etype} ({len(exts)}) ───")
        for ext in exts[:max_entities]:
            evidence = (getattr(ext, "evidence_text", "") or "")[:100].strip()
            src_loc = (getattr(ext, "source_location", "") or "")[:80]

            click.echo(f"  [{ext.extraction_text}]")
            if evidence:
                click.echo(f"    evidence: {evidence}...")
            if src_loc:
                click.echo(f"    location: {src_loc}")
            # Show key attributes
            if ext.attributes:
                key_attrs = {k: v for k, v in ext.attributes.items()
                             if v and v != "Not reported" and k not in
                             ("evidence_text", "source_location")}
                if key_attrs:
                    attr_str = ", ".join(
                        f"{k}={v!r}" for k, v in list(key_attrs.items())[:5]
                    )
                    click.echo(f"    attrs: {attr_str}")


if __name__ == "__main__":
    cli()
