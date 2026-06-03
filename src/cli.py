"""CLI entry point for llm-extract.  Thin wrapper over the generic extraction API."""
import sys
from pathlib import Path

import click


@click.group()
def cli():
    """llm-extract — Generic Knowledge Graph Extraction Pipeline."""


@cli.command()
@click.option("--file", "-f", "file_path", required=True,
              help="Path to a text file to extract from")
@click.option("--output", "-o", "output_dir", default="output")
def extract(file_path, output_dir):
    """Extract entities from one file and export graph."""
    from src.extraction import extract_from_file
    from src.graph import build_graph, export_neo4j_csv
    from src.schema_registry import SchemaRegistry

    registry = SchemaRegistry()
    result = extract_from_file(file_path)

    click.echo(f"Entities: {len(result.extractions)}")

    # Show entity distribution
    type_counts: dict[str, int] = {}
    for ext in result.extractions:
        type_counts[ext.extraction_class] = type_counts.get(ext.extraction_class, 0) + 1
    for t, c in sorted(type_counts.items()):
        click.echo(f"  {t}: {c}")

    # Show warnings
    for w in result.warnings:
        click.echo(f"  ⚠ {w}")

    graph = build_graph([result], registry)
    click.echo(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    export_neo4j_csv(graph, out)
    click.echo(f"Exported nodes.csv + edges.csv to {out}/")


@cli.command()
@click.option("--file", "-f", "file_path", required=True,
              help="Path to a file to extract from")
def debug(file_path):
    """Debug: extract from a single file and show first 10 entities."""
    from src.extraction import extract_from_file

    click.echo(f"Debug extraction from: {file_path}")
    result = extract_from_file(file_path)
    click.echo(f"Extracted {len(result.extractions)} entities")
    for w in result.warnings:
        click.echo(f"  ⚠ {w}")
    for ext in result.extractions[:10]:
        src_loc = getattr(ext, "source_location", "?")
        evidence = getattr(ext, "evidence_text", "")[:60]
        click.echo(f"  [{ext.extraction_class}] {ext.extraction_text}  ({src_loc})")
        if evidence:
            click.echo(f"    evidence: {evidence}...")


@cli.command()
@click.option("--dir", "-d", "input_dir", default="data",
              help="Directory containing text files to process")
@click.option("--output", "-o", "output_dir", default="output")
@click.option("--glob", "glob_pattern", default="*.txt",
              help="File glob pattern (default: *.txt)")
def run(input_dir, output_dir, glob_pattern):
    """Run the full pipeline on all files in a directory."""
    from src.extraction import extract_from_file
    from src.graph import build_graph, export_neo4j_csv
    from src.schema_registry import SchemaRegistry

    dir_path = Path(input_dir)
    files = sorted(dir_path.glob(glob_pattern))
    if not files:
        click.echo(f"No files matching '{glob_pattern}' found in {input_dir}")
        sys.exit(1)

    registry = SchemaRegistry()
    all_results = []
    total_entities = 0

    for f in files:
        click.echo(f"=== {f.name} ===")
        result = extract_from_file(str(f))
        all_results.append(result)
        total_entities += len(result.extractions)
        click.echo(f"  Entities: {len(result.extractions)}")
        for w in result.warnings:
            click.echo(f"  ⚠ {w}")

    graph = build_graph(all_results, registry)
    click.echo(f"\nTotal: {total_entities} entities, "
               f"{len(graph.nodes)} nodes, {len(graph.edges)} edges "
               f"across {len(all_results)} files")

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
