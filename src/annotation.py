"""Annotator — orchestrates the full extraction pipeline.

Ties together chunking, prompting, inference, resolution, alignment,
evidence extraction, and source location.
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
from src.evidence import EvidenceExtractor
from src.format_handler import FormatHandler
from src.prompting import (
    ContextAwarePromptBuilder,
    PromptTemplateStructured,
    QAPromptGenerator,
)
from src.resolver import Resolver
from src.source_location import SourceLocationResolver
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
    evidence_extractor:
        Extracts verbatim evidence around aligned extractions.  Default
        :class:`EvidenceExtractor` when not supplied.
    source_location_resolver:
        Derives section-level source locations.  Default
        :class:`SourceLocationResolver` when not supplied.
    """

    def __init__(
        self,
        language_model,
        prompt_template: PromptTemplateStructured,
        format_handler: FormatHandler | None = None,
        evidence_extractor: EvidenceExtractor | None = None,
        source_location_resolver: SourceLocationResolver | None = None,
    ) -> None:
        self.language_model = language_model
        self.prompt_template = prompt_template
        self.format_handler = format_handler or FormatHandler(use_fences=False)
        self.evidence_extractor = evidence_extractor or EvidenceExtractor()
        self.source_location_resolver = (
            source_location_resolver or SourceLocationResolver()
        )

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
            Passed to ContextAwarePromptBuilder.
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
            all_pass_extractions: dict[
                str, list[list[Extraction]]
            ] = {}  # doc_id → [pass1_extractions, pass2_extractions, ...]

            for pass_idx in range(extraction_passes):
                if show_progress:
                    logger.info(
                        "Extraction pass %d/%d", pass_idx + 1, extraction_passes
                    )
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
        """Convenience: wrap *text* in a Document and return the first result.

        Parameters
        ----------
        text:
            Raw text to annotate.
        resolver:
            Ignored (accepted for API compatibility).  The Annotator creates
            its own Resolver from the internal FormatHandler.
        max_char_buffer:
            Passed to :meth:`annotate_documents`.
        **kwargs:
            Passed through to :meth:`annotate_documents`.

        Returns
        -------
        AnnotatedDocument
        """
        doc = Document(text=text)
        results = list(
            self.annotate_documents(
                [doc],
                max_char_buffer=max_char_buffer,
                **kwargs,
            )
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
        for batch in make_batches_of_textchunk(chunk_iter, batch_length):
            # Build prompts for the batch
            batch_prompts = [
                prompt_builder.build_prompt(
                    chunk.sanitized_chunk_text,
                    chunk.document_id,
                )
                for chunk in batch
            ]

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

            # Track which document IDs we have seen so far
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

        # Resolve LLM output → Extraction objects
        try:
            extractions = list(resolver.resolve(output_text))
        except Exception:
            logger.debug("Resolver failed on chunk output", exc_info=True)
            return

        if not extractions:
            return

        # Align extractions to the full document text with chunk offsets
        tt: TokenizedText
        tt = getattr(doc, "tokenized_text", None)  # type: ignore[assignment]
        if tt is None:
            tt = tok.tokenize(doc.text)

        token_offset = chunk.token_interval.start_index
        char_offset = chunk.char_interval.start_pos or 0

        try:
            aligned = list(
                resolver.align(
                    extractions,
                    doc.text,
                    token_offset=token_offset,
                    char_offset=char_offset,
                )
            )
        except Exception:
            logger.debug("Alignment failed for chunk", exc_info=True)
            return

        # Enrich with evidence text
        self.evidence_extractor.extract_batch(aligned, doc.text, tt)

        # Enrich with source location
        section_id = self._section_id_from_document(doc_id)
        self.source_location_resolver.resolve_batch(
            aligned, section_id, doc.text, char_offset
        )

        # Accumulate
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
        """Yield TextChunks from ChunkIterator for each document in order.

        The ChunkIterator attaches ``tokenized_text`` to the Document object
        so that downstream processing can reuse the same tokenization.
        """
        for doc in documents:
            ci = ChunkIterator(
                text=doc.text,
                max_char_buffer=max_char_buffer,
                tokenizer_impl=tokenizer,
                document=doc,
            )
            for chunk in ci:
                yield chunk

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

        Parameters
        ----------
        per_doc:
            Accumulated extractions keyed by document ID.  Entries for yielded
            documents are popped.
        doc_ids_ordered:
            Document IDs in the order they were submitted.
        seen_doc_ids:
            Set of document IDs that have appeared in processed batches so far.
        keep_last_doc:
            When True, the last document whose chunks have been seen is held
            back (it may still have more chunks coming).
        doc_map:
            Original Document objects keyed by ID.
        """
        if not seen_doc_ids:
            return

        # Find the index of the furthest-seen document in the ordered list
        max_seen_idx = -1
        for i, doc_id in enumerate(doc_ids_ordered):
            if doc_id in seen_doc_ids:
                max_seen_idx = i

        if max_seen_idx < 0:
            return

        # Determine the range of documents to emit
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
        """Multi-pass merge: first-pass extractions win for overlapping
        character intervals.

        Parameters
        ----------
        all_pass_extractions:
            List of extraction lists, one per pass.

        Returns
        -------
        list[Extraction]
            Merged list of non-overlapping extractions.
        """
        if not all_pass_extractions:
            return []

        if len(all_pass_extractions) == 1:
            return all_pass_extractions[0]

        result = list(all_pass_extractions[0])  # First pass wins

        for pass_extractions in all_pass_extractions[1:]:
            for ext in pass_extractions:
                if not self._overlaps_with_any(ext, result):
                    result.append(ext)

        return result

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _section_id_from_document(document_id: str) -> str:
        """Extract a human-readable section name from a document ID.

        Converts ``"PMC123_methods"`` → ``"Methods"``.

        Parameters
        ----------
        document_id:
            A document identifier that may contain a section suffix.

        Returns
        -------
        str
            Section name with the first character capitalised.
        """
        parts = document_id.rsplit("_", 1)
        if len(parts) == 1:
            return parts[0]
        return parts[1].capitalize()

    @staticmethod
    def _overlaps_with_any(
        extraction: Extraction, existing: list[Extraction]
    ) -> bool:
        """Return True when *extraction* overlaps with any in *existing*.

        Overlap is determined by character intervals.
        """
        ci = extraction.char_interval
        if ci is None or ci.start_pos is None or ci.end_pos is None:
            return False

        for other in existing:
            oci = other.char_interval
            if oci is None or oci.start_pos is None or oci.end_pos is None:
                continue
            # Overlap: intervals [a,b) and [c,d) overlap if a < d and c < b
            if ci.start_pos < oci.end_pos and oci.start_pos < ci.end_pos:
                return True

        return False
