"""CLI entry point for llm-extract."""
import sys
from pathlib import Path

import click

from src.config import settings


@click.group()
def cli():
    """llm-extract — Knowledge Graph Extraction Pipeline v2."""


@cli.command()
@click.option("--xml", required=True)
@click.option("--output", "-o", "output_dir", default="output")
@click.option("--skip-gate/--no-skip-gate", default=False,
              help="Skip the known-alternative gate check")
def extract(xml, output_dir, skip_gate):
    """Extract entities from one PMC XML article and export graph."""
    from src.extraction import extract as extract_article
    from src.graph import build_graph, export_neo4j_csv
    from src.schema_registry import SchemaRegistry

    registry = SchemaRegistry()
    result = extract_article(
        xml, skip_if_no_known_alternative=not skip_gate,
    )
    click.echo(f"Entities: {len(result.extractions)}")
    if result.skipped:
        click.echo(f"SKIPPED: {result.skip_reason}")
        click.echo(f"Checkpoint saved to data/intermediates/ for inspection")
        # Show what was extracted anyway
        type_counts: dict[str, int] = {}
        for ext in result.extractions:
            type_counts[ext.extraction_class] = type_counts.get(ext.extraction_class, 0) + 1
        if type_counts:
            click.echo("Phase 1 extractions:")
            for t, c in sorted(type_counts.items()):
                click.echo(f"  {t}: {c}")
        return

    # Show entity distribution
    type_counts: dict[str, int] = {}
    for ext in result.extractions:
        type_counts[ext.extraction_class] = type_counts.get(ext.extraction_class, 0) + 1
    for t, c in sorted(type_counts.items()):
        click.echo(f"  {t}: {c}")

    graph = build_graph([result], registry)
    click.echo(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    export_neo4j_csv(graph, out)
    click.echo(f"Exported nodes.csv + edges.csv to {out}/")


@cli.command()
@click.option("--xml", required=True)
def debug(xml):
    """Debug: extract from a single article and show first 10 entities."""
    from src.extraction import extract

    click.echo(f"Debug extraction from: {xml}")
    result = extract(xml)
    click.echo(f"Extracted {len(result.extractions)} entities")
    if result.skipped:
        click.echo(f"SKIPPED: {result.skip_reason}")
    for ext in result.extractions[:10]:
        attrs = ext.attributes or {}
        src_loc = attrs.get("source_location", "?")
        click.echo(f"  [{ext.extraction_class}] {ext.extraction_text}  ({src_loc})")


@cli.command()
@click.option("--dir", "-d", "xml_dir", default="data/xml",
              help="Directory containing PMC XML files")
@click.option("--output", "-o", "output_dir", default="output")
@click.option("--skip-gate/--no-skip-gate", default=False,
              help="Skip the known-alternative gate check")
def run(xml_dir, output_dir, skip_gate):
    """Run the full pipeline on all articles in a directory."""
    from src.extraction import extract
    from src.graph import build_graph, export_neo4j_csv
    from src.schema_registry import SchemaRegistry

    xml_path = Path(xml_dir)
    xml_files = sorted(xml_path.glob("*.xml"))
    if not xml_files:
        click.echo(f"No XML files found in {xml_dir}")
        sys.exit(1)

    registry = SchemaRegistry()
    all_results = []
    total_entities = 0

    for f in xml_files:
        click.echo(f"=== {f.name} ===")
        result = extract(str(f), skip_if_no_known_alternative=not skip_gate)
        all_results.append(result)
        total_entities += len(result.extractions)
        click.echo(f"  Entities: {len(result.extractions)}"
                   f"{' SKIPPED: ' + result.skip_reason if result.skipped else ''}")

    graph = build_graph(all_results, registry)
    click.echo(f"\nTotal: {total_entities} entities, "
               f"{len(graph.nodes)} nodes, {len(graph.edges)} edges "
               f"across {len(all_results)} articles")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    export_neo4j_csv(graph, out)
    click.echo(f"Exported to {out}/nodes.csv + {out}/edges.csv")


@cli.command()
@click.option("--checkpoints", required=True, help="Path to intermediates directory")
@click.option("--output", "-o", "output_dir", default="output")
def export(checkpoints, output_dir):
    """Export graph from existing checkpoints (skip extraction)."""
    click.echo(f"Exporting from checkpoints: {checkpoints}")
    # TODO: load checkpoints, build graph, export CSV
    click.echo("Not yet implemented — use 'run' command instead.")


if __name__ == "__main__":
    cli()
