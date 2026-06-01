"""Main pipeline entry point. Run stages sequentially with checkpoints."""
import logging
import sys
from src.config import DATA_DIR, SECTIONS_DIR
from src.config import ensure_dirs

logger = logging.getLogger(__name__)

STAGES = ["stage0", "stage1", "stage2", "stage3", "stage4"]


def main(stage_from: str = "stage0"):
    """Run pipeline stages from the specified starting point."""
    ensure_dirs()

    try:
        start_idx = STAGES.index(stage_from)
    except ValueError:
        logger.error("Unknown stage: %s. Available: %s", stage_from, STAGES)
        sys.exit(1)

    stages_to_run = STAGES[start_idx:]
    logger.info("Pipeline starting from: %s", stage_from)
    logger.info("Stages to run: %s", stages_to_run)

    for stage in stages_to_run:
        logger.info("\n%s", "=" * 60)
        logger.info("  %s", stage.upper())
        logger.info("%s", "=" * 60)

        if stage == "stage0":
            from src.stage0_search import run_stage0
            run_stage0(
                alternative_tsv="ALTERNATIVE.tsv",
                existing_doi_list=None,
                output_path=str(DATA_DIR / "literature_pool.tsv"),
            )
        elif stage == "stage1":
            from src.stage1_xml_parser import run_stage1
            run_stage1(
                literature_pool_path=str(DATA_DIR / "literature_pool.tsv"),
                xml_dir=str(DATA_DIR / "xml"),
            )
        elif stage == "stage2":
            logger.info("Stage 2: Entity extraction (LLM — requires API key)")
            logger.info("  Input: %s", SECTIONS_DIR)
            logger.info("  (Async orchestrator to be wired in production)")
        elif stage == "stage3":
            logger.info("Stage 3: Result extraction (LLM — requires API key)")
            logger.info("  (Async orchestrator to be wired in production)")
        elif stage == "stage4":
            from src.stage4_validate import ValidationReport
            report = ValidationReport()
            report.add("INFO", None, "Stage 4 validation initialized")
            logger.info("Stage 4: Validation + Export")
            logger.info("  TSV output -> data/output/tsv/")
            logger.info("  Neo4j output -> data/output/neo4j/import.cypher")

    logger.info("\n%s", "=" * 60)
    logger.info("Pipeline complete.")
    logger.info("%s", "=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    start_from = sys.argv[1] if len(sys.argv) > 1 else "stage0"
    main(start_from)
