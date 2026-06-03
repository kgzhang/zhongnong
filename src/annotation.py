"""Annotator — orchestrates the full extraction pipeline.

Ties together chunking, prompting, inference, resolution, alignment,
and evidence derivation.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Any

from src.chunking import ChunkIterator, TextChunk, make_batches_of_textchunk
from src.data import (
    AnnotatedDocument,
    Document,
    Extraction,
)
from src.evidence import derive_evidence_batch
from src.format_handler import FormatHandler
from src.prompting import (
    ContextAwarePromptBuilder,
    PromptTemplateStructured,
    QAPromptGenerator,
)
from src.resolver import Resolver
from src.tokenizer import RegexTokenizer, TokenizedText, Tokenizer

logger = logging.getLogger(__name__)


class Annotator:
    """Orchestrates the full extraction pipeline from documents to AnnotatedDocuments.

    Parameters
    ----------
    language_model:
        The LLM provider (e.g. an OpenAICompatProvider instance).
    prompt_template:
        Structured prompt with description and optional few-shot examples.
    format_handler:
        Format handler for parsing model output.  Created with JSON defaults
        when not supplied.
    """

    def __init__(
        self,
        language_model,
        prompt_template: PromptTemplateStructured,
        format_handler: FormatHandler | None = None,
    ) -> None:
        self.language_model = language_model
        self.prompt_template = prompt_template
        self.format_handler = format_handler or FormatHandler(use_fences=False)

        self._qa_generator = QAPromptGenerator(
            template=prompt_template,
            format_handler=self.format_handler,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_documents(
        self,
        documents: Sequence[Document],
        max_char_buffer: int = 200,
        batch_length: int = 1,
        extraction_passes: int = 1,
        context_window_chars: int | None = None,
        show_progress: bool = True,
        tokenizer: Tokenizer | None = None,
        **kwargs: Any,
    ) -> Iterator[AnnotatedDocument]:
        """Extract entities from *documents*.

        Parameters
        ----------
        documents:
            Sequence of Documents to annotate.
        max_char_buffer:
            Maximum characters per chunk (passed to ChunkIterator).
        batch_length:
            Number of chunks per inference batch.
        extraction_passes:
            Number of extraction passes.  1 = single streaming pass;
            >1 = multi-pass with merge of non-overlapping extractions.
        context_window_chars:
            Number of characters from the previous chunk to include as context.
        show_progress:
            When True, log progress information.
        tokenizer:
            Tokenizer for chunking and alignment.  Defaults to RegexTokenizer.
        **kwargs:
            Passed through to ``model.infer()``.

        Yields
        ------
        AnnotatedDocument
        """
        if tokenizer is None:
            tokenizer = RegexTokenizer()

        resolver = Resolver(format_handler=self.format_handler)

        if extraction_passes == 1:
            yield from self._annotate_single_pass(
                documents=documents,
                resolver=resolver,
                max_char_buffer=max_char_buffer,
                batch_length=batch_length,
                context_window_chars=context_window_chars,
                tok=tokenizer,
                **kwargs,
            )
        else:
            # Multi-pass: collect all extractions per document across passes,
            # then merge non-overlapping.
            all_pass_extractions: dict[str, list[list[Extraction]]] = {}

            for pass_idx in range(extraction_passes):
                if show_progress:
                    logger.info("Extraction pass %d/%d", pass_idx + 1, extraction_passes)
                for annotated in self._annotate_single_pass(
                    documents=documents,
                    resolver=resolver,
                    max_char_buffer=max_char_buffer,
                    batch_length=batch_length,
                    context_window_chars=context_window_chars,
                    tok=tokenizer,
                    **kwargs,
                ):
                    doc_id = annotated.document_id
                    all_pass_extractions.setdefault(doc_id, []).append(
                        annotated.extractions or []
                    )

            # Merge and yield
            for doc in documents:
                doc_id = doc.document_id
                pass_lists = all_pass_extractions.get(doc_id, [[]])
                merged = self._merge_non_overlapping(pass_lists)
                yield AnnotatedDocument(
                    extractions=merged,
                    text=doc.text,
                    document_id=doc_id,
                )

    def annotate_text(
        self,
        text: str,
        resolver: Resolver | None = None,
        max_char_buffer: int = 200,
        **kwargs: Any,
    ) -> AnnotatedDocument:
        """Convenience: wrap *text* in a Document and return the first result."""
        document_id = kwargs.pop("document_id", None)
        additional_context = kwargs.pop("additional_context", None)
        doc = Document(
            text=text,
            document_id=document_id,
            additional_context=additional_context,
        )
        results = list(
            self.annotate_documents([doc], max_char_buffer=max_char_buffer, **kwargs)
        )
        if results:
            return results[0]
        return AnnotatedDocument(extractions=[], text=text, document_id=doc.document_id)

    # ------------------------------------------------------------------
    # Internal: single-pass streaming
    # ------------------------------------------------------------------

    def _annotate_single_pass(
        self,
        documents: Sequence[Document],
        resolver: Resolver,
        max_char_buffer: int,
        batch_length: int,
        context_window_chars: int | None,
        tok: Tokenizer,
        **kwargs: Any,
    ) -> Iterator[AnnotatedDocument]:
        """Streaming single pass over *documents*.

        Yield AnnotatedDocuments as soon as all chunks for a document have been
        processed (enabled by the ordered nature of the chunk iterator).
        """
        # 1. Capture doc metadata
        doc_ids_ordered = [d.document_id for d in documents]
        doc_map = {d.document_id: d for d in documents}

        # Internal state
        per_doc: dict[str, list[Extraction]] = {}
        seen_doc_ids: set[str] = set()

        # 2. Build chunk iterator across all documents
        chunk_iter = self._document_chunk_iterator(documents, max_char_buffer, tok)

        # 3. Create prompt builder
        prompt_builder = ContextAwarePromptBuilder(
            generator=self._qa_generator,
            context_window_chars=context_window_chars,
        )

        # 4. Batch and process
        batch_count = 0
        for batch in make_batches_of_textchunk(chunk_iter, batch_length):
            batch_count += 1
            # Build prompts for the batch
            batch_prompts = [
                prompt_builder.build_prompt(
                    chunk.sanitized_chunk_text,
                    chunk.document_id,
                )
                for chunk in batch
            ]

            prompt_lens = [len(p) for p in batch_prompts]
            logger.debug(
                "Batch %d: %d chunks, prompt sizes %s",
                batch_count,
                len(batch),
                prompt_lens,
            )

            # Run inference on the batch
            try:
                infer_results = list(
                    self.language_model.infer(batch_prompts, **kwargs)
                )
            except Exception:
                logger.exception("Inference failed for batch of %d chunks", len(batch))
                infer_results = []

            # Process each result, pairing with its chunk
            if len(infer_results) != len(batch):
                logger.warning(
                    "Inference returned %d results for %d prompts — "
                    "truncating or results may be misaligned",
                    len(infer_results),
                    len(batch),
                )

            batch_doc_ids: set[str] = set()
            for chunk, scored_list in zip(batch, infer_results):
                doc_id = chunk.document_id
                batch_doc_ids.add(doc_id)
                self._process_chunk(
                    chunk=chunk,
                    scored_list=scored_list,
                    resolver=resolver,
                    per_doc=per_doc,
                    doc_map=doc_map,
                    tok=tok,
                )

            seen_doc_ids.update(batch_doc_ids)

            # Emit completed documents
            yield from self._emit_docs_iter(
                per_doc=per_doc,
                doc_ids_ordered=doc_ids_ordered,
                seen_doc_ids=seen_doc_ids,
                keep_last_doc=True,
                doc_map=doc_map,
            )

        # 5. Emit any remaining documents
        yield from self._emit_docs_iter(
            per_doc=per_doc,
            doc_ids_ordered=doc_ids_ordered,
            seen_doc_ids=seen_doc_ids,
            keep_last_doc=False,
            doc_map=doc_map,
        )

    # ------------------------------------------------------------------
    # Internal: chunk processing
    # ------------------------------------------------------------------

    def _process_chunk(
        self,
        chunk: TextChunk,
        scored_list: Sequence[Any],
        resolver: Resolver,
        per_doc: dict[str, list[Extraction]],
        doc_map: dict[str, Document],
        tok: Tokenizer,
    ) -> None:
        """Resolve, align, and enrich extractions for one chunk."""
        if not scored_list:
            return

        doc_id = chunk.document_id
        doc = doc_map.get(doc_id)
        if doc is None:
            return

        output_text = scored_list[0].output if scored_list else ""
        if not output_text:
            return

        # 1. Resolve LLM output → Extraction objects
        try:
            extractions = list(resolver.resolve(output_text))
        except Exception:
            logger.debug("Resolver failed on chunk output", exc_info=True)
            return

        if not extractions:
            return

        # 2. Compute offsets: map chunk-level positions to document-level
        token_offset = chunk.token_interval.start_index
        char_offset = chunk.char_interval.start_pos or 0

        # 3. Align extractions to the CHUNK text (not the full document).
        #    Offsets map chunk positions back to document coordinates.
        #    This matches how the original langextract does it.
        chunk_text = chunk.chunk_text
        chunk_text_stripped = chunk.sanitized_chunk_text

        # Alignment works against the chunk text for precision.
        # The prompt uses sanitized_chunk_text, so alignment should too.
        # Use the non-stripped chunk_text for alignment since extractions
        # refer to positions in the original text.
        try:
            aligned = list(
                resolver.align(
                    extractions,
                    chunk_text,
                    token_offset=token_offset,
                    char_offset=char_offset,
                    tokenizer_impl=tok,
                )
            )
        except Exception:
            logger.debug("Alignment failed for chunk", exc_info=True)
            # Fall back: use extractions without alignment
            aligned = list(extractions)

        # 4. Derive evidence text from the full document text at aligned positions.
        #    This is pure engineering — extract verbatim source text at char_interval.
        derive_evidence_batch(aligned, doc.text)

        # 5. Guarantee every extraction has evidence containing its text.
        #    Falls back to str.find() on the document if alignment didn't set char_interval.
        from src.coreference import ensure_evidence_batch
        ensure_evidence_batch(aligned, doc.text)

        # 6. Derive source location from the document context.
        _derive_source_locations(aligned, doc_id, doc.text, char_offset)

        # 7. Accumulate
        per_doc.setdefault(doc_id, []).extend(aligned)

    # ------------------------------------------------------------------
    # Internal: document chunk iterator
    # ------------------------------------------------------------------

    def _document_chunk_iterator(
        self,
        documents: Sequence[Document],
        max_char_buffer: int,
        tokenizer: Tokenizer,
    ) -> Iterator[TextChunk]:
        """Yield TextChunks from ChunkIterator for each document in order."""
        for doc in documents:
            ci = ChunkIterator(
                text_or_tokenized=doc.text,
                max_char_buffer=max_char_buffer,
                tokenizer_impl=tokenizer,
                document=doc,
            )
            yield from ci

    # ------------------------------------------------------------------
    # Internal: streaming emit
    # ------------------------------------------------------------------

    def _emit_docs_iter(
        self,
        per_doc: dict[str, list[Extraction]],
        doc_ids_ordered: list[str],
        seen_doc_ids: set[str],
        keep_last_doc: bool,
        doc_map: dict[str, Document],
    ) -> Iterator[AnnotatedDocument]:
        """Yield AnnotatedDocuments for documents whose chunks are all processed.

        A document is "complete" when we have seen at least one chunk from a
        later document in *doc_ids_ordered* (since chunks come in document
        order).  When *keep_last_doc* is ``True`` the furthest-seen document
        is held back because it may still have pending chunks.
        """
        if not seen_doc_ids:
            return

        max_seen_idx = -1
        for i, doc_id in enumerate(doc_ids_ordered):
            if doc_id in seen_doc_ids:
                max_seen_idx = i

        if max_seen_idx < 0:
            return

        end_idx = max_seen_idx if keep_last_doc else len(doc_ids_ordered)

        to_emit: list[str] = []
        for i in range(end_idx):
            doc_id = doc_ids_ordered[i]
            if doc_id in per_doc:
                to_emit.append(doc_id)

        for doc_id in to_emit:
            doc = doc_map.get(doc_id)
            text = doc.text if doc else ""
            extractions = per_doc.pop(doc_id, [])
            yield AnnotatedDocument(
                extractions=extractions,
                text=text,
                document_id=doc_id,
            )

    # ------------------------------------------------------------------
    # Internal: multi-pass merge
    # ------------------------------------------------------------------

    def _merge_non_overlapping(
        self, all_pass_extractions: list[list[Extraction]]
    ) -> list[Extraction]:
        """Multi-pass merge: first-pass extractions win for overlapping intervals."""
        if not all_pass_extractions:
            return []

        if len(all_pass_extractions) == 1:
            return all_pass_extractions[0]

        result = list(all_pass_extractions[0])

        for pass_extractions in all_pass_extractions[1:]:
            for ext in pass_extractions:
                if not self._overlaps_with_any(ext, result):
                    result.append(ext)

        return result

    @staticmethod
    def _overlaps_with_any(extraction: Extraction, existing: list[Extraction]) -> bool:
        """Return True when *extraction* overlaps with any in *existing*."""
        ci = extraction.char_interval
        if ci is None or ci.start_pos is None or ci.end_pos is None:
            return False

        for other in existing:
            oci = other.char_interval
            if oci is None or oci.start_pos is None or oci.end_pos is None:
                continue
            if ci.start_pos < oci.end_pos and oci.start_pos < ci.end_pos:
                return True

        return False


# ---------------------------------------------------------------------------
# Internal: source location derivation
# ---------------------------------------------------------------------------

import re as _re

_TABLE_PATTERN = _re.compile(r"(Table|Tab\.)\s*\d+", _re.IGNORECASE)
_FIGURE_PATTERN = _re.compile(r"(Figure|Fig\.)\s*\d+", _re.IGNORECASE)


def _derive_source_locations(
    extractions: list[Extraction],
    doc_id: str,
    document_text: str,
    char_offset: int,
) -> None:
    """Set ``source_location`` on each extraction.

    Derives a human-readable location string from the document ID and
    nearby table/figure references in the surrounding text.
    """
    section_id = _section_id_from_document_id(doc_id)
    for ext in extractions:
        ext.source_location = _resolve_location(
            ext, section_id, document_text, char_offset
        )


def _section_id_from_document_id(document_id: str) -> str:
    """Extract a human-readable section name from a document ID.

    Converts ``"PMC123_methods"`` → ``"Methods"``.
    """
    parts = document_id.rsplit("_", 1)
    if len(parts) == 1:
        return parts[0]
    return parts[1].capitalize()


def _resolve_location(
    extraction: Extraction,
    section_id: str,
    section_text: str,
    char_offset: int,
) -> str:
    """Return a source location string for *extraction*."""
    parts = [section_id]
    ci = extraction.char_interval
    if ci is not None and ci.start_pos is not None:
        nearby_start = max(0, ci.start_pos - 200)
        nearby_end = min(len(section_text), (ci.end_pos or 0) + 200)
        nearby_text = section_text[nearby_start:nearby_end]
        details = []
        tm = _TABLE_PATTERN.search(nearby_text)
        fm = _FIGURE_PATTERN.search(nearby_text)
        if tm:
            details.append(tm.group(0))
        if fm:
            details.append(fm.group(0))
        if details:
            parts.append(", ".join(details))
    return ", ".join(parts)
