"""PMC XML adapter — parse PMC articles into the generic extraction pipeline.

Usage::

    from src.adapters.pmc import parse_article, literature_pre_extractor, build_document

    meta = parse_article("data/xml/PMC12188611.xml")
    doc = build_document(meta)
    from src.extraction import extract
    result = extract(doc, pre_extractor=literature_pre_extractor(meta))

This adapter is **format-specific** — it handles PMC's JATS XML namespace
quirks, section parsing, and front-matter extraction.  It lives in
``src/adapters/``, NOT in the generic ``src/`` framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data import Document, Extraction

# ---------------------------------------------------------------------------
# Data class for parsed PMC metadata
# ---------------------------------------------------------------------------


@dataclass
class PmcArticleMeta:
    """Parsed metadata from a PMC XML article."""

    file_path: Path
    pmcid: str = ""
    pmid: str = ""
    doi: str = ""
    title: str = ""
    journal: str = ""
    abstract_text: str = ""
    body_text: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    parse_warnings: list[str] = field(default_factory=list)

    @property
    def document_id(self) -> str:
        """Derive document ID: PMID if available, otherwise PMCID, otherwise filename stem."""
        return self.pmid or self.pmcid or self.file_path.stem

    @property
    def full_text(self) -> str:
        """Combined abstract + body text for extraction."""
        parts = []
        if self.abstract_text:
            parts.append(self.abstract_text)
        if self.body_text:
            parts.append(self.body_text)
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# XML Parser
# ---------------------------------------------------------------------------


def parse_article(xml_path: str | Path) -> PmcArticleMeta:
    """Parse a PMC XML article into a :class:`PmcArticleMeta`.

    Handles JATS namespace resolution, front-matter extraction (DOI, PMID,
    title, journal), abstract, and body text.  Section-level parsing groups
    text by semantic section type (introduction, methods, results, discussion).

    Parameters
    ----------
    xml_path:
        Path to a PMC XML file (``.xml``, ``.nxml``).

    Returns
    -------
    PmcArticleMeta
    """
    from lxml import etree

    path = Path(xml_path)
    meta = PmcArticleMeta(file_path=path)

    try:
        tree = etree.parse(str(path))
        root = tree.getroot()
        nsmap = _resolve_namespace(root)

        # ---- Front matter ----
        front = root.find(".//front", nsmap)
        if front is not None:
            _parse_article_ids(front, nsmap, meta)
            _parse_title(front, nsmap, meta)
            _parse_journal(front, nsmap, meta)
            _parse_abstract(front, nsmap, meta)

        # ---- Body ----
        body = root.find(".//body", nsmap)
        if body is not None:
            meta.body_text = "".join(body.itertext()).strip()
            _parse_sections(body, nsmap, meta)

    except Exception as exc:
        meta.parse_warnings.append(f"XML parse error: {exc}")

    return meta


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------


def build_document(meta: PmcArticleMeta) -> Document:
    """Build a :class:`Document` from parsed PMC metadata.

    The document's ``additional_context`` carries a serialised copy of the
    metadata (DOI, PMID, title) so that pre-extractors can access it.
    """
    return Document(
        text=meta.full_text,
        document_id=meta.document_id,
        additional_context=_serialize_meta(meta),
    )


# ---------------------------------------------------------------------------
# Literature pre-extractor
# ---------------------------------------------------------------------------


def literature_pre_extractor(meta: PmcArticleMeta):
    """Return a pre-extractor callable for Literature entities.

    The returned callable creates a Literature Extraction from the PMC
    front-matter (DOI, PMID, title, journal) — NOT from the LLM.

    Usage::

        pre_ext = literature_pre_extractor(meta)
        result = extract(doc, pre_extractor=pre_ext)
    """

    def _pre_extractor(document: Document) -> list[Extraction]:
        attrs = {
            "doi": meta.doi,
            "pmid": meta.pmid,
            "title": meta.title,
            "journal": meta.journal,
        }
        if not attrs["title"] and not attrs["doi"]:
            return []

        ext = Extraction(
            extraction_class="Literature",
            extraction_text=attrs["title"] or meta.document_id,
            attributes=attrs,
        )
        ext.evidence_text = attrs["title"]
        ext.source_location = f"front-matter:{meta.file_path.name}"
        return [ext]

    return _pre_extractor


# ---------------------------------------------------------------------------
# CSV export — entity-type review sheets
# ---------------------------------------------------------------------------


def export_entity_review_csv(
    results: list[Any],  # list[DocumentExtractionResult]
    output_dir: str | Path,
    registry: Any = None,
    *,
    deduplicate: bool = True,
) -> None:
    """Export extractions as CSV files organised by entity type.

    Produces one CSV per entity type (``Alternative.csv``, ``Indicator.csv``,
    etc.) plus a combined ``all_entities.csv``.  Each row contains the entity
    text, attributes, evidence text, and source location — suitable for
    manual review in Excel / Google Sheets.

    When *deduplicate* is True (default), extractions are first deduplicated
    by canonical identity so each entity appears exactly once per CSV.

    Parameters
    ----------
    results:
        List of :class:`DocumentExtractionResult` objects (from :func:`extract`).
    output_dir:
        Directory to write CSV files into.
    registry:
        Optional :class:`SchemaRegistry` for field ordering.
    deduplicate:
        When True, deduplicate extractions before exporting.
    """
    import csv

    # Deduplicate extractions so each canonical entity appears only once
    if deduplicate:
        from src.coreference import resolve_coreferences, clean_evidence_batch

        for result in results:
            clean_evidence_batch(result.extractions)
            result.extractions = resolve_coreferences(
                result.extractions, registry
            )

    out = Path(output_dir) / "review"
    out.mkdir(parents=True, exist_ok=True)

    # Group extractions by entity type
    by_type: dict[str, list[Extraction]] = {}
    all_exts: list[tuple[str, Extraction]] = []

    for result in results:
        doc_id = result.document_id
        for ext in result.extractions:
            etype = ext.extraction_class
            # Tag with source document
            if ext.attributes is None:
                ext.attributes = {}
            ext.attributes.setdefault("_source_doc", doc_id)
            by_type.setdefault(etype, []).append(ext)
            all_exts.append((doc_id, ext))

    # Determine columns from the entity schema (if registry available)
    def _field_names(etype: str) -> list[str]:
        """Get ordered field names for *etype* from the registry."""
        if registry is None:
            return []
        try:
            ed = registry.entity_def(etype)
            return [a.name for a in ed.attributes]
        except (KeyError, AttributeError):
            return []

    # Write per-type CSVs
    for etype, exts in sorted(by_type.items()):
        path = out / f"{etype}.csv"
        field_order = _field_names(etype)

        # Build header
        static_cols = ["source_doc", "extraction_text", "evidence_text", "source_location"]
        attr_cols = list(field_order)  # ordered schema fields
        # Add any extra attributes not in schema
        all_attr_keys: set[str] = set()
        for ext in exts:
            if ext.attributes:
                all_attr_keys.update(ext.attributes.keys())
        extra_cols = sorted(all_attr_keys - set(attr_cols) - {"_source_doc"})
        header = static_cols + attr_cols + extra_cols

        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            for ext in exts:
                ev = getattr(ext, "evidence_text", "")
                sl = getattr(ext, "source_location", "")
                # Sanitize for CSV: collapse newlines, strip
                ev_safe = " ".join(ev.split()) if ev else ""
                sl_safe = " ".join(sl.split()) if sl else ""
                row: dict[str, Any] = {
                    "source_doc": (ext.attributes or {}).get("_source_doc", ""),
                    "extraction_text": ext.extraction_text,
                    "evidence_text": ev_safe,
                    "source_location": sl_safe,
                }
                # Add attribute fields
                if ext.attributes:
                    for k, v in ext.attributes.items():
                        if k == "_source_doc":
                            continue
                        if isinstance(v, (list, dict)):
                            import json as _json
                            row[k] = _json.dumps(v, ensure_ascii=False)
                        else:
                            row[k] = v
                writer.writerow(row)

    # Write combined CSV
    combined_path = out / "all_entities.csv"
    with open(combined_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source_doc", "entity_type", "extraction_text",
                          "evidence_text", "source_location", "attributes"])
        for doc_id, ext in all_exts:
            attrs = ext.attributes or {}
            attrs.pop("_source_doc", None)
            writer.writerow([
                doc_id,
                ext.extraction_class,
                ext.extraction_text,
                getattr(ext, "evidence_text", ""),
                getattr(ext, "source_location", ""),
                str({k: v for k, v in attrs.items()
                     if not isinstance(v, (list, dict))}),
            ])

    print(f"Review CSVs exported to {out}/")
    for etype in sorted(by_type.keys()):
        count = len(by_type[etype])
        print(f"  {etype}.csv: {count} entities")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_namespace(root) -> dict[str, str]:
    """Extract namespace map from a PMC article element."""
    nsmap: dict[str, str] = {}
    article = root.find("article")
    if article is None:
        article = root
    if hasattr(article, "nsmap") and article.nsmap:
        for prefix, uri in article.nsmap.items():
            nsmap[prefix if prefix else ""] = uri
    if not nsmap:
        tag = article.tag
        if "}" in tag:
            nsmap[""] = tag.split("}")[0].lstrip("{")
    return nsmap


def _parse_article_ids(front, nsmap, meta: PmcArticleMeta) -> None:
    """Extract DOI, PMID, PMCID from front matter."""
    for aid in front.findall(".//article-id", nsmap):
        pub_type = aid.get("pub-id-type", "")
        text = (aid.text or "").strip()
        if pub_type == "doi":
            meta.doi = text
        elif pub_type == "pmid":
            meta.pmid = text
        elif pub_type == "pmcid":
            meta.pmcid = text


def _parse_title(front, nsmap, meta: PmcArticleMeta) -> None:
    """Extract article title."""
    el = front.find(".//article-title", nsmap)
    if el is not None:
        meta.title = "".join(el.itertext()).strip()


def _parse_journal(front, nsmap, meta: PmcArticleMeta) -> None:
    """Extract journal name."""
    el = front.find(".//journal-title", nsmap)
    if el is not None:
        meta.journal = "".join(el.itertext()).strip()


def _parse_abstract(front, nsmap, meta: PmcArticleMeta) -> None:
    """Extract abstract text."""
    el = front.find(".//abstract", nsmap)
    if el is not None:
        meta.abstract_text = "".join(el.itertext()).strip()


def _parse_sections(body, nsmap, meta: PmcArticleMeta) -> None:
    """Parse body sections by semantic type (introduction, methods, etc.)."""
    sections = body.findall(".//sec", nsmap)
    for sec in sections:
        title_el = sec.find("./title", nsmap)
        title_text = "".join(title_el.itertext()).strip().lower() if title_el is not None else ""
        sec_text = "".join(sec.itertext()).strip()

        # Map section title to semantic category
        if "introduction" in title_text or "background" in title_text:
            key = "introduction"
        elif "method" in title_text or "materials" in title_text:
            key = "methods"
        elif "result" in title_text:
            key = "results"
        elif "discussion" in title_text or "conclusion" in title_text:
            key = "discussion"
        else:
            key = title_text.replace(" ", "_")[:30] if title_text else "other"

        if key in meta.sections:
            meta.sections[key] += "\n\n" + sec_text
        else:
            meta.sections[key] = sec_text


def _serialize_meta(meta: PmcArticleMeta) -> str:
    """Serialize key metadata fields as a compact string."""
    import json as _json

    return _json.dumps({
        "doi": meta.doi,
        "pmid": meta.pmid,
        "title": meta.title,
        "journal": meta.journal,
    }, ensure_ascii=False)
