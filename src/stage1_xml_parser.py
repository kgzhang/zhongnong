"""Stage 1: JATS/NLM XML Parser.

Extracts structured sections from PMC full-text XML files.
"""

from __future__ import annotations

import re
from typing import Any

from lxml import etree


# ---------------------------------------------------------------------------
# Conclusion markers (ordered by priority)
# ---------------------------------------------------------------------------
_CONCLUSION_MARKERS = [
    "In conclusion",
    "These results suggest",
    "These results indicate",
    "Overall",
    "Therefore",
    "In summary",
    "Collectively",
    "Taken together",
    "Our results demonstrate",
    "Our findings suggest",
]

# ---------------------------------------------------------------------------
# Section-title -> canonical key mapping
# ---------------------------------------------------------------------------
_SECTION_KEYWORDS: dict[str, list[str]] = {
    "materials_and_methods": [
        "materials and methods",
        "methods",
        "experimental procedures",
        "experimental",
        "methodology",
        "materials & methods",
        "animals and methods",
    ],
    "results": [
        "results",
        "findings",
        "results and discussion",
    ],
    "discussion": [
        "discussion",
        "discussion and conclusions",
        "conclusions",
        "general discussion",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_text(text: str | None) -> str:
    """Collapse whitespace and trim."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _get_text_from_children(parent, tag: str) -> list[str]:
    """Return a list of normalised text from *all* <tag> children of *parent*."""
    return [_normalise_text(p.text) for p in parent.iterfind(tag) if _normalise_text(p.text)]


def _recursive_paragraphs(sec_elem) -> list[str]:
    """Gather *all* <p> text recursively inside a <sec> element (depth-first)."""
    paragraphs: list[str] = []
    for child in sec_elem.iter():
        if child.tag == "p":
            text = _normalise_text(child.text)
            if text:
                paragraphs.append(text)
    return paragraphs


def _resolve_section_key(title: str) -> str | None:
    """Return the canonical key for *title*, or None if no match."""
    title_lower = title.lower().strip()
    for key, keywords in _SECTION_KEYWORDS.items():
        for kw in keywords:
            if kw in title_lower:
                return key
    return None


def _map_section_titles(section_title_map: dict[str, list[str]]) -> dict[str, str]:
    """Build a title->canonical lookup from the raw title-map dict."""
    mapping: dict[str, str] = {}
    for sec_key, variations in section_title_map.items():
        for v in variations:
            mapping[v] = sec_key
    return mapping


# Reverse builder
_SECTION_TITLE_MAP = _map_section_titles(_SECTION_KEYWORDS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_conclusion(paragraphs: list[str]) -> dict[str, Any]:
    """Scan the last 3 paragraphs for a conclusion marker.

    Returns
    -------
    dict with keys:
        sentence   – the paragraph that matched (or last paragraph)
        marker     – the matched marker string, or None
        no_marker  – True when no marker was found
    """
    if not paragraphs:
        return {"sentence": "", "marker": None, "no_marker": True}

    # Scan the last 3 paragraphs in order
    candidates = paragraphs[-3:]
    for para in candidates:
        for marker in _CONCLUSION_MARKERS:
            if marker.lower() in para.lower():
                return {"sentence": para, "marker": marker, "no_marker": False}

    # Fallback: last non-empty paragraph
    return {"sentence": paragraphs[-1], "marker": None, "no_marker": True}


def parse_xml_to_sections(xml_path: str) -> dict[str, Any]:
    """Parse a JATS/NLM XML file and extract structured sections.

    Returns
    -------
    dict with top-level keys:
        doi, pmid, title, journal, publication_year, publication_date,
        sections (dict), tables (list), warnings (list)
    """
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(xml_path, parser)
    root = tree.getroot()

    # -- Strip namespace if present so plain tag names work ----------
    # JATS files often carry a namespace; we remove it for simplicity.
    _strip_ns(root)

    result: dict[str, Any] = {}
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Metadata
    # ------------------------------------------------------------------
    article_meta = root.find(".//article-meta")
    meta: dict[str, Any] = {}
    if article_meta is not None:
        meta["doi"] = _get_article_id(article_meta, "doi")
        meta["pmid"] = _get_article_id(article_meta, "pmid")
        title_el = article_meta.find(".//article-title")
        meta["title"] = _normalise_text(title_el.text) if title_el is not None else ""

        journal_meta = root.find(".//journal-meta")
        journal_el = journal_meta.find("journal-title") if journal_meta is not None else None
        meta["journal"] = _normalise_text(journal_el.text) if journal_el is not None else ""

        pub_date = _get_pub_date(article_meta)
        meta["publication_year"] = pub_date.get("year")
        meta["publication_date"] = pub_date.get("iso")
    else:
        meta = {"doi": "", "pmid": "", "title": "", "journal": "",
                "publication_year": None, "publication_date": None}

    result.update(meta)

    # ------------------------------------------------------------------
    # 2. Abstract
    # ------------------------------------------------------------------
    abstract_el = root.find(".//abstract")
    abstract: dict[str, Any] = {"paragraphs": [], "conclusion_sentence": "",
                                 "conclusion_marker": None, "no_conclusion_marker": True}
    if abstract_el is not None:
        # Collect all <p> inside abstract, including those in nested <sec> elements
        paragraphs = _recursive_paragraphs(abstract_el)
        abstract["paragraphs"] = paragraphs
        conclusion = extract_conclusion(paragraphs)
        abstract["conclusion_sentence"] = conclusion["sentence"]
        abstract["conclusion_marker"] = conclusion["marker"]
        abstract["no_conclusion_marker"] = conclusion["no_marker"]

    # ------------------------------------------------------------------
    # 3. Body sections
    # ------------------------------------------------------------------
    body_el = root.find("body")
    sections: dict[str, Any] = {"abstract": abstract}

    if body_el is not None:
        for sec_elem in body_el.iterfind("sec"):
            title_el = sec_elem.find("title")
            if title_el is None or not title_el.text:
                continue
            section_title = _normalise_text(title_el.text)
            canonical = _resolve_section_key(section_title)
            if canonical is None:
                continue

            # Build subsections
            subsections: list[dict[str, Any]] = []
            for sub_sec in sec_elem.iterfind("sec"):
                sub_title_el = sub_sec.find("title")
                sub_title = _normalise_text(sub_title_el.text) if sub_title_el is not None else ""
                sub_paras = _recursive_paragraphs(sub_sec)
                subsections.append({
                    "title": sub_title,
                    "paragraphs": sub_paras,
                    "full_text": " ".join(sub_paras),
                })

            # Full text from all sub-paragraphs
            all_paras = _recursive_paragraphs(sec_elem)
            # But exclude paragraphs that belong to subsections (already captured)
            # Actually, _recursive_paragraphs gathers ALL <p>; for full_text we want
            # the top-level <p> + all sub-<p> together. We'll use all of them.
            full_text = " ".join(all_paras)

            sections[canonical] = {
                "subsections": subsections,
                "full_text": full_text,
            }

    # Ensure keys exist even when section not found
    for key in ("materials_and_methods", "results", "discussion"):
        if key not in sections:
            sections[key] = {"subsections": [], "full_text": ""}
            warnings.append(f"No '{key}' section found by title matching")

    result["sections"] = sections

    # ------------------------------------------------------------------
    # 4. Tables
    # ------------------------------------------------------------------
    tables: list[dict[str, Any]] = []
    for tbl in root.iterfind(".//table-wrap"):
        tbl_id = tbl.get("id", "")
        label_el = tbl.find("label")
        title_el = tbl.find("caption//title") or tbl.find("caption//p")
        tbl_title = _normalise_text(title_el.text) if title_el is not None else ""
        # Collect text content of the table
        tbl_text_parts: list[str] = []
        for p in tbl.iterfind(".//p"):
            txt = _normalise_text(p.text)
            if txt:
                tbl_text_parts.append(txt)
        for td in tbl.iterfind(".//td"):
            txt = _normalise_text(td.text)
            if txt:
                tbl_text_parts.append(txt)
        tables.append({
            "table_id": tbl_id,
            "title": tbl_title,
            "content": " ".join(tbl_text_parts),
        })
    result["tables"] = tables

    # ------------------------------------------------------------------
    # 5. Warnings
    # ------------------------------------------------------------------
    result["warnings"] = warnings

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_article_id(article_meta, id_type: str) -> str:
    """Get an article-id by pub-id-type attribute."""
    el = article_meta.find(f'article-id[@pub-id-type="{id_type}"]')
    if el is not None and el.text:
        return el.text.strip()
    return ""


def _get_pub_date(article_meta) -> dict:
    """Extract publication year and ISO date."""
    pub_date_el = article_meta.find('pub-date')
    if pub_date_el is None:
        return {"year": None, "iso": None}

    year_el = pub_date_el.find("year")
    month_el = pub_date_el.find("month")
    day_el = pub_date_el.find("day")
    year = int(year_el.text.strip()) if year_el is not None and year_el.text else None
    month = month_el.text.strip() if month_el is not None and month_el.text else None
    day = day_el.text.strip() if day_el is not None and day_el.text else None

    iso: str | None = None
    if year is not None:
        mm = month or "01"
        dd = day or "01"
        iso = f"{year}-{mm.zfill(2)}-{dd.zfill(2)}"

    return {"year": year, "iso": iso}


def _strip_ns(el):
    """Recursively strip XML namespaces from element and its children.

    Modifies the element *in place* by removing the ``xmlns`` attribute
    and rewriting the tag's ``{uri}local`` form to just ``local``.
    """
    if el.tag.startswith("{"):
        el.tag = el.tag.split("}", 1)[1]
    for attr in list(el.attrib.keys()):
        if attr.startswith("{"):
            new_attr = attr.split("}", 1)[1]
            el.attrib[new_attr] = el.attrib.pop(attr)
    for child in el:
        _strip_ns(child)


def run_stage1(literature_pool_path: str = "data/literature_pool.tsv",
               xml_dir: str = "data/xml") -> str:
    """Batch process all articles in the literature pool.

    Reads the literature pool TSV, finds corresponding XML files,
    parses each one, and writes structured_sections/{doi_safe}.json.

    Returns path to the output directory.
    """
    import csv
    import json
    from pathlib import Path
    from src.config import SECTIONS_DIR

    pool = []
    try:
        with open(literature_pool_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                pool.append(row)
    except FileNotFoundError:
        print(f"Literature pool not found: {literature_pool_path}")
        return ""

    xml_dir_path = Path(xml_dir)
    output_dir = Path(SECTIONS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = []

    for article in pool:
        doi = article.get("doi", "")
        doi_safe = doi.replace("/", "_").replace(":", "_") if doi else article.get("pmid", "unknown")
        out_path = output_dir / f"{doi_safe}.json"

        if out_path.exists():
            success += 1
            continue

        xml_path = xml_dir_path / f"{doi_safe}.xml"
        if not xml_path.exists():
            candidates = list(xml_dir_path.glob(f"*{doi_safe[:30]}*"))
            if candidates:
                xml_path = candidates[0]
            else:
                failed.append({"doi": doi, "reason": "XML file not found"})
                continue

        try:
            result = parse_xml_to_sections(str(xml_path))
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            success += 1
        except Exception as e:
            failed.append({"doi": doi, "reason": str(e)})

    fail_path = output_dir / "_failures.json"
    with open(fail_path, "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)

    print(f"Stage 1 complete: {success} parsed, {len(failed)} failed")
    return str(output_dir)
