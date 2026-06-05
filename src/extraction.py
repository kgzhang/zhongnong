"""Generic extraction API — pre-extraction + single-pass LLM extraction.

Design
------
1. **Pre-extraction**: entities derived from document metadata or structure
   (e.g. Literature from front-matter).  Configurable via a *pre_extractor*
   callable — the framework does not hardcode any entity type.

2. **Full extraction**: single LLM pass over the document text using the
   Annotator pipeline (chunk → prompt → infer → resolve → align → evidence).

3. **Post-processing**: vocabulary matching, normalisation, etc. driven by
   the SchemaRegistry (fully generic).

This replaces the earlier 4-phase pipeline.  Multi-pass extraction is
supported via ``extraction_passes`` (each pass is an independent LLM call;
non-overlapping extractions from later passes are merged).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.data import Document, Extraction

logger = logging.getLogger(__name__)


# Type alias for the pre-extraction hook.
# Takes a Document and returns a list of Extractions derived from metadata.
PreExtractor = Callable[[Document], list[Extraction]]


def _has_any_llm_field(registry, entity_name: str) -> bool:
    """Return True if *entity_name* has at least one attribute with source='llm'.

    Entities that have ONLY structure/align/post source attributes (e.g. entities
    derived entirely from pre-extractors or post-processing) are excluded from
    LLM extraction.
    """
    try:
        ed = registry.entity_def(entity_name)
        return any(a.source == "llm" for a in ed.attributes)
    except (KeyError, AttributeError):
        return True  # If unknown, include it (fail open).


@dataclass
class DocumentExtractionResult:
    """Extraction result for one document — format-agnostic."""

    document_id: str
    metadata: dict = field(default_factory=dict)
    extractions: list[Extraction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def extract(
    document: Document,
    *,
    model=None,
    model_id: str | None = None,
    api_key: str | None = None,
    registry=None,
    max_char_buffer: int | None = None,
    pre_extractor: PreExtractor | None = None,
    additional_context: str | None = None,
    entity_names: list[str] | None = None,
    **kwargs,
) -> DocumentExtractionResult:
    """Extract entities from one Document using pre-extraction + single LLM pass.

    All defaults come from ``src.config.settings`` unless explicitly overridden.

    Parameters
    ----------
    document:
        The Document to extract from.
    model:
        Pre-configured language model.  Created from *model_id* / *api_key*
        if not provided.
    model_id:
        Model identifier (e.g. ``"gpt-4o"``).  Defaults to ``settings.llm_model``.
    api_key:
        API key for the LLM provider.
    registry:
        SchemaRegistry instance.  Created with defaults if not provided.
    max_char_buffer:
        Maximum characters per chunk.  Defaults to ``settings.max_char_buffer``.
    pre_extractor:
        Optional callable ``(Document) -> list[Extraction]`` for metadata-derived
        entities (e.g. Literature from front-matter).  Called before the LLM pass.
    additional_context:
        Additional context injected into every prompt.
    entity_names:
        Optional explicit list of entity type names to extract via LLM.
        When None, defaults to all entity names minus 'Literature'.
        Use this for section-based extraction (e.g. only Methods entities).
    **kwargs:
        Passed through to :meth:`Annotator.annotate_text`.

    Returns
    -------
    DocumentExtractionResult
    """
    from src.config import settings

    if model_id is None:
        model_id = settings.llm_model
    if max_char_buffer is None:
        max_char_buffer = settings.max_char_buffer

    # 1. Load registry if not provided
    if registry is None:
        from src.schema_registry import SchemaRegistry

        registry = SchemaRegistry()

    # 2. Create model if not provided
    if model is None:
        from src.factory import create_model

        model = create_model()

    t0 = time.time()
    result = DocumentExtractionResult(document_id=document.document_id)

    # ------------------------------------------------------------------
    # Step A: Pre-extraction (metadata-derived entities)
    # ------------------------------------------------------------------
    all_extractions: list[Extraction] = []
    if pre_extractor is not None:
        try:
            pre_exts = pre_extractor(document)
            all_extractions.extend(pre_exts)
            logger.debug("Pre-extraction yielded %d entities", len(pre_exts))
        except Exception:
            logger.warning("Pre-extraction failed", exc_info=True)
            result.warnings.append("Pre-extraction failed")

    # ------------------------------------------------------------------
    # Step B: Full LLM extraction (single pass)
    # ------------------------------------------------------------------
    from src.format_handler import FormatHandler
    from src.prompting import PromptTemplateStructured
    from src.annotation import Annotator

    fh = FormatHandler(use_fences=model.requires_fence_output)

    # Build prompt from entity definitions.
    # Exclude entities that have ONLY structure/align/post source attributes
    # (no LLM-source attributes) — they are derived from pre-extractors or
    # post-processing and should never be sent to the LLM.
    # When *entity_names* is provided explicitly (e.g. section-based extraction),
    # use that list instead.
    if entity_names is not None:
        llm_entity_names = [n for n in entity_names if _has_any_llm_field(registry, n)]
    else:
        llm_entity_names = [
            n for n in registry.all_entity_names() if _has_any_llm_field(registry, n)
        ]
    prompt_text = registry.build_extraction_prompt(llm_entity_names)

    template = PromptTemplateStructured(description=prompt_text)
    annotator = Annotator(model, template, fh)

    phase_label = "Extraction"
    logger.info(
        "%s — extracting %d entity types from %d chars",
        phase_label,
        len(llm_entity_names),
        len(document.text),
    )
    print(
        f"  {phase_label} ({len(llm_entity_names)} entity types) — {len(document.text)} chars",
        flush=True,
    )

    phase_start = time.time()
    try:
        doc_result = annotator.annotate_text(
            document.text,
            max_char_buffer=max_char_buffer,
            additional_context=additional_context,
            document_id=document.document_id,
            **kwargs,
        )
        llm_extractions = doc_result.extractions or []
    except Exception as e:
        logger.warning("Extraction failed: %s", e)
        result.warnings.append(f"Extraction failed: {e}")
        result.extractions = all_extractions
        return result

    phase_elapsed = time.time() - phase_start
    all_extractions.extend(llm_extractions)

    # Count entity types for feedback
    type_counts: dict[str, int] = {}
    for ext in llm_extractions:
        type_counts[ext.extraction_class] = type_counts.get(ext.extraction_class, 0) + 1
    type_summary = ", ".join(f"{t}={c}" for t, c in sorted(type_counts.items()))
    logger.info(
        "%s done — %d entities in %.1fs: %s",
        phase_label,
        len(llm_extractions),
        phase_elapsed,
        type_summary,
    )
    print(f"    → {len(llm_extractions)} entities in {phase_elapsed:.1f}s: {type_summary}", flush=True)

    # ------------------------------------------------------------------
    # Step C: Post-processing (vocabulary matching, normalisation, etc.)
    # ------------------------------------------------------------------
    for ext in all_extractions:
        try:
            registry.post_process(ext)
        except Exception:
            pass

    total_time = time.time() - t0
    type_counts_final: dict[str, int] = {}
    for ext in all_extractions:
        type_counts_final[ext.extraction_class] = type_counts_final.get(ext.extraction_class, 0) + 1
    logger.info(
        "Extraction complete: %d entities in %.1fs — %s",
        len(all_extractions),
        total_time,
        ", ".join(f"{t}={c}" for t, c in sorted(type_counts_final.items())),
    )
    print(f"  Total: {len(all_extractions)} entities in {total_time:.1f}s", flush=True)

    result.extractions = all_extractions
    return result


def extract_from_file(
    file_path: str | Path,
    *,
    text_loader: Callable[[str | Path], str] | None = None,
    document_id: str | None = None,
    pre_extractor: PreExtractor | None = None,
    **kwargs,
) -> DocumentExtractionResult:
    """Extract entities from a file.

    Parameters
    ----------
    file_path:
        Path to the source file.
    text_loader:
        Callable that reads *file_path* and returns text.  Defaults to
        ``Path.read_text()``.  Use this to inject format-specific parsing
        (e.g. PMC XML → plain text).
    document_id:
        Override the document ID (default: stem of *file_path*).
    pre_extractor:
        Pre-extraction hook for metadata-derived entities.
    **kwargs:
        Passed to :func:`extract`.

    Returns
    -------
    DocumentExtractionResult
    """
    path = Path(file_path)
    doc_id = document_id or path.stem

    if text_loader is not None:
        text = text_loader(path)
    else:
        text = path.read_text(encoding="utf-8")

    doc = Document(text=text, document_id=doc_id)
    return extract(document, pre_extractor=pre_extractor, **kwargs)


# ---------------------------------------------------------------------------
# Batch processing — single file or directory → combined graph
# ---------------------------------------------------------------------------


def batch_extract_pmc(
    input_path: str | Path,
    output_dir: str | Path = "output",
    *,
    registry=None,
    model=None,
    model_factory: Callable[[], Any] | None = None,
    glob_pattern: str = "*.xml",
    skip_gate: bool = False,
    max_workers: int | None = None,
    **kwargs,
) -> dict:
    """Process one PMC XML file or a directory of XML files, combining all
    results into a single graph.

    When *max_workers* > 1, processes files in parallel using a
    ``ThreadPoolExecutor``.  Each worker thread creates its own model
    instance for thread safety.

    Parameters
    ----------
    input_path:
        Path to a PMC XML file or a directory containing ``.xml`` files.
    output_dir:
        Directory where ``review/`` and ``graph/`` subdirectories will be
        written.
    registry:
        SchemaRegistry instance.  Created with defaults when ``None``.
    model:
        Pre-configured LLM provider.  Only used in single-worker mode
        (``max_workers=1``).  For parallel mode, use *model_factory* instead
        so each thread gets its own instance.
    model_factory:
        Zero-argument callable that returns a fresh model instance.
        Required for thread safety when ``max_workers > 1``.  Defaults to
        ``create_model`` from ``src.factory``.
    glob_pattern:
        File-matching pattern when *input_path* is a directory
        (default: ``"*.xml"``).
    skip_gate:
        When ``True``, skip the gate check and always run the bulk (results)
        extraction phase.  Default ``False``.
    max_workers:
        Maximum number of parallel worker threads.  Defaults to
        ``settings.max_workers`` (currently 4).  Set to 1 for sequential
        processing.

    Returns
    -------
    dict
        ``{"results": [...], "graph": Graph, "output_dir": Path}``
    """
    del model  # replaced by model_factory for thread safety
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path
    import threading

    from src.adapters.pmc_sectioned import extract_sectioned
    from src.adapters.pmc import export_entity_review_csv
    from src.checkpoint import save_checkpoint, load_checkpoint
    from src.graph import build_graph, export_neo4j_csv
    from src.schema_registry import SchemaRegistry
    from src.config import setup_logging, settings

    setup_logging("INFO")

    if registry is None:
        registry = SchemaRegistry()
    if model_factory is None:
        from src.factory import create_model as model_factory
    if max_workers is None:
        max_workers = settings.max_workers

    input_p = Path(input_path).resolve()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve input files
    if input_p.is_file():
        xml_files = [input_p]
    elif input_p.is_dir():
        xml_files = sorted(input_p.glob(glob_pattern))
        if not xml_files:
            raise FileNotFoundError(
                f"No files matching '{glob_pattern}' found in {input_p}"
            )
    else:
        raise FileNotFoundError(f"Input path not found: {input_p}")

    logger.info("Batch extraction: %d file(s) → %s (workers=%d)",
                 len(xml_files), out, max_workers)

    checkpoint_dir = settings.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resume from checkpoints
    all_results: list = []
    unprocessed: list[Path] = []
    for xml_file in xml_files:
        ckpt_path = checkpoint_dir / f"{xml_file.stem}.checkpoint"
        loaded = load_checkpoint(ckpt_path)
        if loaded is not None:
            all_results.append(loaded)
            logger.info("Resumed checkpoint for %s (%d entities)",
                         xml_file.stem, len(loaded.extractions))
        else:
            unprocessed.append(xml_file)

    if not unprocessed:
        logger.info("All %d file(s) already processed (checkpoints found)", len(xml_files))
        return _build_graph_and_export(all_results, out, registry)

    total_extractions = sum(len(r.extractions) for r in all_results)
    skipped_articles = 0
    stats_lock = threading.Lock()

    # 2. Process files (parallel or sequential)
    if max_workers > 1 and len(unprocessed) > 1:
        logger.info("Parallel processing %d file(s) with %d workers",
                     len(unprocessed), max_workers)

        def process_one(xml_file: Path):
            try:
                model_inst = model_factory()
                result = extract_sectioned(
                    xml_file, registry=registry, model=model_inst,
                    **kwargs,
                )
                ckpt_path = checkpoint_dir / f"{xml_file.stem}.checkpoint"
                save_checkpoint(result, ckpt_path)
                return (xml_file, result, None, None)
            except Exception as exc:
                logger.error("Failed to extract %s: %s", xml_file.name, exc)
                return (xml_file, None, str(exc), None)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_one, xf): xf
                for xf in unprocessed
            }
            for future in as_completed(future_to_file):
                xml_file = future_to_file[future]
                try:
                    f, result, error, _ = future.result()
                    if result is not None:
                        with stats_lock:
                            all_results.append(result)
                            n = len(result.extractions)
                            total_extractions += n
                    if error:
                        with stats_lock:
                            skipped_articles += 1
                    logger.info("  %s: %d entities", xml_file.stem,
                                 len(result.extractions) if result else 0)
                except Exception as exc:
                    logger.error("Unexpected error processing %s: %s",
                                 xml_file.name, exc)
    else:
        # Sequential fallback
        for xml_file in unprocessed:
            logger.info("=== %s ===", xml_file.name)
            try:
                model_inst = model_factory()
                result = extract_sectioned(
                    xml_file, registry=registry, model=model_inst,
                    **kwargs,
                )
            except Exception as exc:
                logger.error("Failed to extract %s: %s", xml_file.name, exc)
                continue

            ckpt_path = checkpoint_dir / f"{xml_file.stem}.checkpoint"
            save_checkpoint(result, ckpt_path)

            n = len(result.extractions)
            total_extractions += n
            all_results.append(result)

            type_counts: dict[str, int] = {}
            for ext in result.extractions:
                type_counts[ext.extraction_class] = (
                    type_counts.get(ext.extraction_class, 0) + 1
                )
            type_summary = ", ".join(
                f"{t}={c}" for t, c in sorted(type_counts.items())
            )
            logger.info("  %d entities: %s", n, type_summary)
            if result.warnings:
                gate_fails = [w for w in result.warnings if "Gate" in w]
                if gate_fails:
                    skipped_articles += 1
                    logger.info("  ⚠ gate failed, results phase skipped")

    logger.info(
        "Total: %d entities across %d file(s) (%d gate-skipped)",
        total_extractions, len(xml_files), skipped_articles,
    )

    return _build_graph_and_export(all_results, out, registry)


def _build_graph_and_export(
    all_results: list,
    output_dir: Path,
    registry: Any,
) -> dict:
    """Build graph and export CSVs from collected results."""
    from src.adapters.pmc import export_entity_review_csv
    from src.graph import build_graph, export_neo4j_csv

    out = output_dir

    # Export review CSVs (combined across all files)
    review_dir = out / "review"
    export_entity_review_csv(all_results, review_dir, registry)
    logger.info("Review CSVs exported to %s/", review_dir)

    # Build combined graph
    global_types: frozenset[str] = frozenset()
    if registry is not None:
        try:
            global_types = registry.get_global_types()
        except Exception:
            pass
    graph = build_graph(all_results, registry, global_types=global_types)
    logger.info(
        "Graph: %d nodes, %d edges", len(graph.nodes), len(graph.edges)
    )

    # Export Neo4j CSV
    graph_dir = out / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    export_neo4j_csv(graph, graph_dir)
    logger.info("Neo4j CSV exported to %s/", graph_dir)

    return {
        "results": all_results,
        "graph": graph,
        "output_dir": out,
    }
