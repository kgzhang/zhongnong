"""Main pipeline entry point. Run stages sequentially with checkpoints."""
import sys
from src.config import DATA_DIR, SECTIONS_DIR
from src.config import ensure_dirs

STAGES = ["stage0", "stage1", "stage2", "stage3", "stage4"]


def main(stage_from: str = "stage0"):
    """Run pipeline stages from the specified starting point."""
    ensure_dirs()

    try:
        start_idx = STAGES.index(stage_from)
    except ValueError:
        print(f"Unknown stage: {stage_from}. Available: {STAGES}")
        sys.exit(1)

    stages_to_run = STAGES[start_idx:]
    print(f"Pipeline starting from: {stage_from}")
    print(f"Stages to run: {stages_to_run}")

    for stage in stages_to_run:
        print(f"\n{'='*60}")
        print(f"  {stage.upper()}")
        print(f"{'='*60}")

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
            print("Stage 2: Entity extraction (LLM — requires API key)")
            print(f"  Input: {SECTIONS_DIR}")
            print("  (Async orchestrator to be wired in production)")
        elif stage == "stage3":
            print("Stage 3: Result extraction (LLM — requires API key)")
            print("  (Async orchestrator to be wired in production)")
        elif stage == "stage4":
            from src.stage4_validate import ValidationReport
            report = ValidationReport()
            report.add("INFO", None, "Stage 4 validation initialized")
            print(f"Stage 4: Validation + Export")
            print(f"  TSV output -> data/output/tsv/")
            print(f"  Neo4j output -> data/output/neo4j/import.cypher")

    print(f"\n{'='*60}")
    print("Pipeline complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    start_from = sys.argv[1] if len(sys.argv) > 1 else "stage0"
    main(start_from)
