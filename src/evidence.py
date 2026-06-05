"""Evidence — derive verbatim evidence text from aligned char_interval.

Evidence is strictly an engineering concern: after alignment has set
``char_interval`` on each extraction, we extract the exact source text
at that position.  This is NOT model output — it is the original text
that the model identified, recovered from the source document.

The extraction_text (model output) tells you WHAT the model found;
the evidence_text (verbatim source at char_interval) confirms WHERE
and gives the surrounding context for human verification.
"""

from src.data import Extraction


def derive_evidence(
    extraction: Extraction,
    document_text: str,
    context_chars: int = 300,
) -> str:
    """Return the verbatim source text surrounding *extraction*'s character interval.

    When *context_chars* > 0 the returned text extends *context_chars* characters
    on each side of the interval to provide reading context.

    Returns ``""`` when the extraction has no ``char_interval`` (alignment failed).
    """
    ci = extraction.char_interval
    if ci is None or ci.start_pos is None or ci.end_pos is None:
        return ""

    start = max(0, ci.start_pos - context_chars)
    end = min(len(document_text), ci.end_pos + context_chars)

    return document_text[start:end]


def derive_evidence_batch(
    extractions: list[Extraction],
    document_text: str,
    context_chars: int = 300,
) -> None:
    """Set ``evidence_text`` on every extraction in *extractions*.

    The evidence is extracted from *document_text* at the position given by
    each extraction's ``char_interval`` (set during alignment).  When an
    extraction has no interval, ``evidence_text`` is set to ``""``.

    Note: ``evidence_text`` is stored as a top-level attribute on the
    Extraction, separate from the LLM-output ``attributes`` dict, to make
    it clear that it is engineering-derived rather than model output.
    """
    for ext in extractions:
        evidence = derive_evidence(ext, document_text, context_chars)
        # Store evidence_text directly on the Extraction object, not in
        # the LLM-output attributes dict.  This separates engineering-derived
        # fields from model-output fields.
        ext.evidence_text = evidence


class EvidenceExtractor:
    """Extract verbatim evidence text surrounding an aligned extraction.

    Deprecated in favour of the module-level :func:`derive_evidence` and
    :func:`derive_evidence_batch` functions.  Kept for backward compatibility.
    """

    def extract_evidence(
        self,
        extraction: Extraction,
        document_text: str,
        context_chars: int = 300,
    ) -> str:
        """Return verbatim evidence for a single extraction."""
        return derive_evidence(extraction, document_text, context_chars)

    def extract_batch(
        self,
        extractions: list[Extraction],
        document_text: str,
        context_chars: int = 300,
    ) -> None:
        """Set evidence_text on a batch of extractions."""
        derive_evidence_batch(extractions, document_text, context_chars)
