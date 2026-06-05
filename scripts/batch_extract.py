#!/usr/bin/env python3
"""Batch extraction script — single PMC XML file or directory → combined graph.

Run this directly without installing the package::

    python scripts/batch_extract.py --input data/xml/PMC12188611.xml --output output/my_run
    python scripts/batch_extract.py --input data/xml/ --output output/batch_run
    python scripts/batch_extract.py --input data/xml/ --glob "PMC*.xml" --no-skip-gate

Requirements: ``uv run`` or an activated virtual environment with dependencies
installed (see ``make install``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch PMC XML extraction — combine results into one graph.",
    )
    parser.add_argument(
        "--input", "-i", dest="input_path", required=True,
        help="Path to a PMC XML file or a directory of XML files.",
    )
    parser.add_argument(
        "--output", "-o", dest="output_dir", default="output",
        help="Output directory (review/ and graph/ subdirectories created here).",
    )
    parser.add_argument(
        "--glob", dest="glob_pattern", default="*.xml",
        help="File-matching pattern when INPUT is a directory (default: *.xml).",
    )
    parser.add_argument(
        "--no-skip-gate", dest="no_skip_gate", action="store_true",
        help="Disable gate — always run the full extraction even when the gate fails.",
    )
    parser.add_argument(
        "--model", dest="model_name", default=None,
        help="Model name override (provider-specific).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only list the files that would be processed, then exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from src.extraction import batch_extract_pmc
    from src.config import setup_logging

    setup_logging("DEBUG" if args.verbose else "INFO")

    input_p = Path(args.input_path).resolve()

    # Dry-run: just list files
    if args.dry_run:
        if input_p.is_file():
            print(f"Would process: {input_p}")
        elif input_p.is_dir():
            files = sorted(input_p.glob(args.glob_pattern))
            if not files:
                print(f"No files matching '{args.glob_pattern}' in {input_p}")
                return 1
            print(f"Would process {len(files)} file(s):")
            for f in files:
                print(f"  {f}")
        else:
            print(f"Input path not found: {input_p}")
            return 1
        print(f"Output directory: {args.output_dir}")
        print(f"Gate: {'disabled' if args.no_skip_gate else 'enabled'}")
        return 0

    print(f"Input:    {args.input_path}")
    print(f"Output:   {args.output_dir}")
    print(f"Pattern:  {args.glob_pattern}")
    print(f"Gate:     {'disabled' if args.no_skip_gate else 'enabled'}")
    print()

    try:
        result = batch_extract_pmc(
            args.input_path,
            output_dir=args.output_dir,
            glob_pattern=args.glob_pattern,
            skip_gate=args.no_skip_gate,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    total = sum(len(r.extractions) for r in result["results"])
    graph = result["graph"]
    print(f"Done. {total} entities, {len(graph.nodes)} graph nodes, "
          f"{len(graph.edges)} graph edges")
    print(f"Review CSVs:  {result['output_dir']}/review/")
    print(f"Neo4j CSVs:   {result['output_dir']}/graph/")

    # Per-type summary
    type_counts: dict[str, int] = {}
    for r in result["results"]:
        for ext in r.extractions:
            type_counts[ext.extraction_class] = (
                type_counts.get(ext.extraction_class, 0) + 1
            )
    if type_counts:
        print()
        print("Entity type counts:")
        for t, c in sorted(type_counts.items()):
            print(f"  {t}: {c}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
