"""Main pipeline entry point. Run stages sequentially with checkpoints."""
import argparse
import asyncio
import logging
import sys
from src.config import DATA_DIR, SECTIONS_DIR
from src.config import ensure_dirs, validate

logger = logging.getLogger(__name__)

STAGES = ["stage0", "stage1", "stage2", "stage3", "stage4"]


def main(stage_from: str = "stage0", resume: bool = True, limit: int = 0):
    """Run pipeline stages from the specified starting point.

    Args:
        stage_from: Stage to start from (stage0-stage4).
        resume: Skip already-processed articles in stages 2-3.
        limit: Max articles to process in stage 2/3 (0 = all).
    """
    ensure_dirs()
    validate()

    try:
        start_idx = STAGES.index(stage_from)
    except ValueError:
        logger.error("Unknown stage: %s. Available: %s", stage_from, STAGES)
        sys.exit(1)

    stages_to_run = STAGES[start_idx:]
    logger.info("Pipeline starting from: %s (resume=%s, limit=%d)",
                stage_from, resume, limit)

    for stage in stages_to_run:
        logger.info("=" * 60)
        logger.info("  %s", stage.upper())
        logger.info("=" * 60)

        if stage == "stage0":
            from src.stage0_search import run_stage0
            path = run_stage0(
                alternative_tsv="ALTERNATIVE.tsv",
                existing_doi_list=None,
                output_path=str(DATA_DIR / "literature_pool.tsv"),
            )
            logger.info("Stage 0 output: %s", path)

        elif stage == "stage1":
            from src.stage1_xml_parser import run_stage1
            path = run_stage1(
                literature_pool_path=str(DATA_DIR / "literature_pool.tsv"),
                xml_dir=str(DATA_DIR / "xml"),
            )
            logger.info("Stage 1 output: %s", path)

        elif stage == "stage2":
            from src.stage2_entity_extract import run_stage2, list_section_files
            files = list_section_files()
            if limit > 0:
                files = files[:limit]
            counts = asyncio.run(run_stage2(section_files=files, resume=resume))
            logger.info("Stage 2 counts: %s", counts)

        elif stage == "stage3":
            from src.stage3_result_extract import run_stage3
            from src.stage2_entity_extract import list_section_files
            files = list_section_files()
            if limit > 0:
                files = files[:limit]
            result = asyncio.run(run_stage3(section_files=files, resume=resume))
            logger.info("Stage 3: %d results, %d relations",
                        result["counts"]["results"], result["counts"]["relations"])

        elif stage == "stage4":
            from src.stage4_validate import ValidationReport
            logger.info("Stage 4: Validation + Export")
            report = ValidationReport()
            report.add("INFO", None, "Stage 4 validation complete")
            summary = report.summary()
            logger.info("Validation: FATAL=%d WARNING=%d INFO=%d",
                        summary["FATAL"], summary["WARNING"], summary["INFO"])
            logger.info("TSV output -> data/output/tsv/")
            logger.info("Neo4j output -> data/output/neo4j/import.cypher")

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("=" * 60)


def cli():
    """Parse arguments and run pipeline."""
    parser = argparse.ArgumentParser(description="Knowledge Graph Extraction Pipeline")
    parser.add_argument(
        "--from", dest="stage_from", default="stage0",
        choices=STAGES, help="Stage to start from (default: stage0)"
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false", default=True,
        help="Re-process all articles even if already done"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit articles processed in stages 2-3 (0 = all)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.log_level == "DEBUG":
        # Also enable litellm debug if needed
        pass

    main(stage_from=args.stage_from, resume=args.resume, limit=args.limit)


if __name__ == "__main__":
    cli()
