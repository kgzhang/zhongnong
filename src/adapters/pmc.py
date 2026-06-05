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

import re
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
    publication_year: str = ""
    publication_date: str = ""
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
            _parse_pub_date(front, nsmap, meta)

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
    front-matter (DOI, PMID, title, journal, publication dates,
    abstract conclusion) — NOT from the LLM.

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
            "publication_year": int(meta.publication_year) if meta.publication_year else "",
            "publication_date": meta.publication_date,
            "abstract_conclusion": _extract_conclusion(meta.abstract_text) if meta.abstract_text else "",
            "study_design": "Not reported",
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

    # Write relationships CSV for manual validation
    _write_relationships_csv(out, all_exts, registry)

    print(f"Review CSVs exported to {out}/")
    for etype in sorted(by_type.keys()):
        count = len(by_type[etype])
        print(f"  {etype}.csv: {count} entities")


def _normalize_key(text: str) -> str:
    """Normalize for fuzzy lookup: lowercase, strip parens/punctuation, collapse spaces."""
    t = text.strip().lower()
    t = re.sub(r'\s*\([^)]*\)', '', t)  # strip parentheticals
    t = re.sub(r'[^\w\s]', '', t)        # strip punctuation
    t = re.sub(r'\s+', ' ', t)            # collapse spaces
    return t.strip()


def _write_relationships_csv(
    out_dir: Path,
    all_exts: list[tuple[str, Any]],
    registry: Any = None,
) -> None:
    """Write relationships.csv with all cross-entity edges for manual validation.

    Derives edges from:

    - Entity definition ``references`` (e.g., Result → Indicator).  Reference
      fields use the naming convention ``<role>_<target>`` (e.g.
      ``result_indicator``).  The field is looked up first in ``attributes``
      (where it would be if the LLM output it directly), then falls back to
      ``primary_text`` / ``extraction_text`` for auto-linking.
    - Entity definition ``inline_relations`` (e.g., Composite_Product →
      Alternative via ``components``).
    - Attribute-based references: any attribute whose value matches a
      ``primary_text`` value of a known entity type.
    """
    import csv as _csv

    path = out_dir / "relationships.csv"
    rows: list[dict] = []

    # Build lookup: (entity_type, field_value) → canonical extraction
    # Index by extraction_text, primary_text, AND all attribute values
    # so that references using abbreviations, standard names, etc. all match.
    lookup: dict[tuple[str, str], Any] = {}
    normalized_lookup: dict[tuple[str, str], Any] = {}  # stripped punctuation
    for doc_id, ext in all_exts:
        attrs = ext.attributes or {}
        etype = ext.extraction_class

        # Index by extraction_text
        et = ext.extraction_text.strip()
        if et:
            lookup[(etype, et.lower())] = ext
            normalized_lookup[(etype, _normalize_key(et))] = ext

        # Index by all attribute values (standard_name, abbreviation, etc.)
        for attr_val in attrs.values():
            if isinstance(attr_val, str) and attr_val.strip():
                v = attr_val.strip()
                lookup[(etype, v.lower())] = ext
                normalized_lookup[(etype, _normalize_key(v))] = ext

        # Index by primary_text attribute (highest priority for matching)
        if registry:
            try:
                ed = registry.entity_def(etype)
                pt_val = attrs.get(ed.primary_text, "")
                if pt_val and isinstance(pt_val, str) and pt_val.strip():
                    v = pt_val.strip()
                    lookup[(etype, v.lower())] = ext
                    normalized_lookup[(etype, _normalize_key(v))] = ext
            except (KeyError, AttributeError):
                pass

    def _fuzzy_find(etype: str, target_text: str) -> bool:
        """Try exact, normalized, substring, abbreviation, and word-overlap matching."""
        t = target_text.strip()
        if not t:
            return False
        t_lower = t.lower()
        tn = _normalize_key(t)

        # 1. Exact match (case-insensitive)
        if (etype, t_lower) in lookup:
            return True

        # 2. Normalized match (no punctuation, collapsed spaces)
        if (etype, tn) in normalized_lookup and tn:
            return True

        # 3. Substring match (bidirectional)
        for (et, k), _ in lookup.items():
            if et != etype:
                continue
            if t_lower in k or k in t_lower:
                return True

        # 4. Word overlap ≥ 2 (handles partial name matches)
        t_words = set(w for w in re.split(r'[\s_\-]+', tn) if len(w) > 2)
        if len(t_words) >= 2:
            for (et, k), _ in lookup.items():
                if et != etype:
                    continue
                k_words = set(w for w in re.split(r'[\s_\-]+', k) if len(w) > 2)
                if len(t_words & k_words) >= 2:
                    return True

        # 5. Single-word match for short abbreviations (≥3 chars)
        if len(t_lower) >= 3 and len(t_words) <= 1:
            for (et, k), _ in lookup.items():
                if et != etype:
                    continue
                k_norm = _normalize_key(k)
                # Exact abbreviation match
                if t_lower == k_norm:
                    return True
                # Target is an abbreviation of a longer name
                # (e.g., "ADG" matches "Average Daily Gain")
                if len(t_lower) <= 10 and len(k_norm) > len(t_lower):
                    k_words = [w for w in re.split(r'[\s_\-]+', k_norm) if len(w) > 1]
                    k_initials = ''.join(w[0] for w in k_words if w)
                    if t_lower == k_initials.lower():
                        return True

        return False

    for doc_id, ext in all_exts:
        attrs = ext.attributes or {}
        etype = ext.extraction_class

        # Resolve edges from entity definition
        if registry:
            try:
                ed = registry.entity_def(etype)

                # From references — try the reference field name first,
                # then auto-detect from primary_text / extraction_text
                for ref in ed.references:
                    val = attrs.get(ref.name)
                    if not val or not isinstance(val, str) or not val.strip():
                        # Fallback: auto-link via primary_text of target entity
                        # e.g. if Intervention has extraction_text = "OEO 500 mg/kg"
                        # and there's an Alternative with standard_name = "OEO",
                        # auto-detect the uses relationship
                        val = _auto_detect_reference(
                            ext, ref, lookup, normalized_lookup
                        )
                    if val and isinstance(val, str) and val.strip():
                        found = _fuzzy_find(ref.target_entity, val.strip())
                        rows.append({
                            "source_doc": doc_id,
                            "source_type": etype,
                            "source_text": ext.extraction_text,
                            "relation": ref.edge_type,
                            "target_type": ref.target_entity,
                            "target_text": val.strip(),
                            "target_found": "yes" if found else "no",
                        })

                # From inline_relations
                for ir in ed.inline_relations:
                    vals = attrs.get(ir.via_field)
                    if not vals:
                        continue
                    if not isinstance(vals, list):
                        vals = [vals]
                    for v in vals:
                        if not v:
                            continue
                        if isinstance(v, dict):
                            target_name = v.get("standard_name", str(v))
                        else:
                            target_name = str(v)
                        # Try ALL target types; use the first one that matches
                        found = False
                        matched_type = ""
                        for target_type in ir.target:
                            if _fuzzy_find(target_type, target_name.strip()):
                                found = True
                                matched_type = target_type
                                break
                        # Fallback: use the first target type if none match
                        if not matched_type and ir.target:
                            matched_type = ir.target[0]
                        rows.append({
                            "source_doc": doc_id,
                            "source_type": etype,
                            "source_text": ext.extraction_text,
                            "relation": ir.name,
                            "target_type": matched_type,
                            "target_text": target_name.strip(),
                            "target_found": "yes" if found else "no",
                        })

            except (KeyError, AttributeError):
                pass

    # Filter: only keep relationships where the target entity actually exists.
    # Unmatched edges (target_found="no") are dropped — it's better to have
    # a clean graph with only verified edges than broken references.
    matched_rows = [r for r in rows if r.get("target_found") == "yes"]
    skipped = len(rows) - len(matched_rows)
    if skipped:
        print(f"  relationships.csv: {len(matched_rows)} matched edges "
              f"({skipped} unmatched skipped)")

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=[
            "source_doc", "source_type", "source_text",
            "relation", "target_type", "target_text", "target_found",
        ])
        writer.writeheader()
        for row in matched_rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _auto_detect_reference(
    ext: Any,
    ref: Any,
    lookup: dict,
    normalized_lookup: dict,
) -> str | None:
    """Auto-detect cross-entity references when the LLM didn't output a
    dedicated reference field.

    Uses the entity's own extraction_text and attributes (abbreviation,
    standard_name, primary_text) to find matching target entities.

    Strategies:
    1. Check if the source entity's primary_text matches a target entity name
    2. Check all attribute values for matches against target entity names
    3. For Intervention→Alternative: check extraction_text for substance names
    """
    etype = ext.extraction_class
    attrs = ext.attributes or {}
    target_type = ref.target_entity

    # Strategy 1: primary_text of source matches a target entity's primary_text
    # (e.g., Result's extraction_text matches Indicator's abbreviation)
    src_text = ext.extraction_text.strip()
    if (target_type, src_text.lower()) in lookup:
        return src_text

    # Strategy 2: check each attribute value — if it matches a target entity
    # name, use that as the reference value
    for attr_val in attrs.values():
        if not isinstance(attr_val, str) or not attr_val.strip():
            continue
        v = attr_val.strip()
        if (target_type, v.lower()) in lookup:
            return v
        vn = _normalize_key(v)
        if (target_type, vn) in normalized_lookup:
            return v

    # Strategy 3: word-overlap between source extraction_text and target names
    src_words = set(w for w in re.split(r'[\s_\-]+', src_text.lower())
                    if len(w) > 2)
    if len(src_words) >= 1:
        for (et, k), _ in lookup.items():
            if et != target_type:
                continue
            k_words = set(w for w in re.split(r'[\s_\-]+', k) if len(w) > 2)
            if src_words & k_words:
                # Found a word overlap — use the matched target name
                return k

    return None


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


def _parse_pub_date(front, nsmap, meta: PmcArticleMeta) -> None:
    """Extract publication date from front matter.

    Prefers epub date over collection date.  Stores both a structured
    ``publication_date`` (YYYY-MM-DD when available) and a
    ``publication_year``.
    """
    pub_dates = front.findall(".//pub-date", nsmap)
    if not pub_dates:
        return

    # Prefer epub, then pmc-release, then collection, then first available
    priority = {"epub": 0, "ppub": 0, "pmc-release": 1, "collection": 2}
    best: Any = None
    best_prio = 999
    for pd_el in pub_dates:
        ptype = (pd_el.get("pub-type") or "").strip()
        prio = priority.get(ptype, 3)
        if prio < best_prio:
            best_prio = prio
            best = pd_el

    if best is None:
        return

    year_el = best.find("./year", nsmap)
    month_el = best.find("./month", nsmap)
    day_el = best.find("./day", nsmap)

    year = (year_el.text or "").strip() if year_el is not None else ""
    month = (month_el.text or "").strip() if month_el is not None else ""
    day = (day_el.text or "").strip() if day_el is not None else ""

    if year:
        meta.publication_year = year
    if year and month and day:
        meta.publication_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    elif year and month:
        meta.publication_date = f"{year}-{month.zfill(2)}"
    elif year:
        meta.publication_date = year


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


def _extract_conclusion(abstract_text: str) -> str:
    """Extract the conclusion sentence(s) from an abstract.

    Looks for sentences starting with conclusion-signalling phrases in the
    last 1-3 sentences.  Falls back to the last sentence.
    """
    import re

    conclusion_markers = [
        "in conclusion", "these results suggest", "these results indicate",
        "these findings suggest", "overall", "therefore", "in summary",
        "collectively", "taken together", "our results demonstrate",
        "our findings demonstrate", "this study demonstrates",
        "the present study demonstrates",
    ]
    sentences = re.split(r"(?<=[.!?])\s+", abstract_text.strip())
    if not sentences:
        return ""

    # Search last 3 sentences for conclusion markers
    for sent in reversed(sentences[-3:]):
        sent_lower = sent.strip().lower()
        for marker in conclusion_markers:
            if sent_lower.startswith(marker):
                return sent.strip()

    # Fallback: last sentence
    return sentences[-1].strip()
