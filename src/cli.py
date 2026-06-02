"""CLI entry point for zhongnong-kg."""
import sys
from pathlib import Path

import click

from src.config import settings


@click.group()
def cli():
    """zhongnong-kg — Knowledge Graph Extraction Pipeline v2."""


@cli.command()
@click.option("--input", "-i", "input_file", required=True)
@click.option("--output", "-o", "output_dir", default="output")
@click.option("--model", default=None)
@click.option("--resume/--no-resume", default=False)
def run(input_file, output_dir, model, resume):
    """Run the full extraction pipeline."""
    click.echo(f"zhongnong-kg v2.0.0")
    click.echo(f"Input: {input_file}")
    click.echo(f"Output: {output_dir}")
    click.echo(f"Model: {model or settings.llm_model}")
    click.echo("Pipeline ready. Full run requires API key and article XML files.")


@cli.command()
@click.option("--xml", required=True)
def debug(xml):
    """Debug: extract from a single article."""
    click.echo(f"Debug extraction from: {xml}")
    try:
        from src.extraction import extract

        result = extract(xml)
        click.echo(f"Extracted {len(result.extractions)} entities")
        if result.skipped:
            click.echo(f"SKIPPED: {result.skip_reason}")
        for ext in result.extractions[:10]:
            click.echo(f"  [{ext.extraction_class}] {ext.extraction_text}")
    except Exception as e:
        click.echo(f"Error: {e}")
        sys.exit(1)


@cli.command()
@click.option("--checkpoints", required=True)
@click.option("--output", "-o", "output_dir", default="output")
def export(checkpoints, output_dir):
    """Export graph from existing checkpoints."""
    click.echo(f"Exporting from checkpoints: {checkpoints}")


if __name__ == "__main__":
    cli()
