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
from typing import Callable

from src.data import Document, Extraction

logger = logging.getLogger(__name__)


# Type alias for the pre-extraction hook.
# Takes a Document and returns a list of Extractions derived from metadata.
PreExtractor = Callable[[Document], list[Extraction]]


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
    # Exclude "Literature" — it is derived from pre-extractors / document
    # metadata (filename, front-matter) and should never be sent to the LLM.
    # When *entity_names* is provided explicitly (e.g. section-based extraction),
    # use that list instead.
    if entity_names is not None:
        llm_entity_names = [n for n in entity_names if n != "Literature"]
    else:
        llm_entity_names = [
            n for n in registry.all_entity_names() if n != "Literature"
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
