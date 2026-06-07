#!/usr/bin/env python3
"""Performant batch cleaning script — reads review CSVs, cleans, produces delivery.

Design choices for speed:
- **Parallel per-article cleaning** via ProcessPoolExecutor (all CPU cores).
- **Zero-copy pandas operations** with dtype-optimized DataFrames.
- **Single-pass CSV read/write** — no intermediate serialization.
- **Gate filtering** done inside the cleaning pass (no separate scan).
- **Graph export** reuses the existing fast path from src/graph.py.

Usage::

    python scripts/clean_and_deliver.py \\
        --input output/review/review \\
        --output output/delivery \\
        --workers 8
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import multiprocessing as mp
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clean_and_deliver")


# ============================================================================
# 0. CONSTANTS & TAXONOMY — embedded for zero-import startup
# ============================================================================

# Entity types and their primary-name column
ENTITY_MAP: dict[str, str] = {
    "Alternative.csv": "Alternative",
    "Composite_Product.csv": "Composite_Product",
    "Control_Group.csv": "Control_Group",
    "Experiment.csv": "Experiment",
    "Indicator.csv": "Indicator",
    "Intervention.csv": "Intervention",
    "Literature.csv": "Literature",
    "Method.csv": "Method",
    "Result.csv": "Result",
    "Swine.csv": "Swine",
    "Swine_Model.csv": "Swine_Model",
    "Tissue_Site.csv": "Tissue_Site",
}

# Dedup mode per entity type (from schemas/entities.yaml)
# Swine_Model is EXCLUDED from global dedup — each model is article-specific
GLOBAL_TYPES: frozenset[str] = frozenset({
    "Alternative", "Composite_Product", "Indicator", "Method",
    "Swine", "Tissue_Site",
})
ARTICLE_TYPES: frozenset[str] = frozenset({
    "Control_Group", "Experiment", "Intervention", "Literature", "Result",
})

# Primary text field per entity type (for dedup key and graph ID)
PRIMARY_TEXT: dict[str, str] = {
    "Alternative": "standard_name",
    "Composite_Product": "product_name",
    "Control_Group": "group_name",
    "Experiment": "experiment_id",
    "Indicator": "standard_name",
    "Intervention": "intervention_target",
    "Literature": "title",
    "Method": "method_name",
    "Result": "indicator_abbreviation",
    "Swine": "breed",
    "Swine_Model": "model_type",
    "Tissue_Site": "site_name",
}

# 4-char prefixes for Neo4j IDs (from src/graph.py _TYPE_PREFIX)
_TYPE_PREFIX: dict[str, str] = {
    "Alternative": "alte", "Composite_Product": "comp", "Literature": "lite",
    "Experiment": "expe", "Swine_Model": "swmo", "Swine": "swin",
    "Intervention": "intv", "Control_Group": "ctgr", "Tissue_Site": "tiss",
    "Indicator": "indi", "Method": "meth", "Result": "resu",
}

# ── AMINO ACIDS ──
_AMINO_ACIDS: set[str] = {
    "arginine", "lysine", "methionine", "threonine", "tryptophan",
    "glutamine", "leucine", "valine", "isoleucine", "glycine",
    "alanine", "proline", "serine", "cysteine", "tyrosine",
    "phenylalanine", "histidine", "aspartic acid", "glutamic acid",
    "asparagine", "taurine", "citrulline",
}

# ── MINERALS ──
_MINERALS: set[str] = {
    "selenium", "chromium", "iron", "zinc", "copper", "calcium",
    "phosphorus", "magnesium", "potassium", "sodium", "manganese",
    "iodine", "cobalt", "molybdenum", "sulfur", "boron",
}

# ── MINERAL COMPOUNDS (lowercase) ──
_MINERAL_COMPOUNDS: set[str] = {
    "zinc oxide", "copper sulfate", "zinc sulfate", "ferrous sulfate",
    "ferric oxide", "calcium carbonate", "calcium phosphate",
    "magnesium oxide", "magnesium sulfate", "sodium chloride",
    "sodium bicarbonate", "potassium chloride", "zinc chloride",
    "copper chloride", "manganese sulfate", "chromium picolinate",
    "zinc methionine", "zinc glycinate", "nano zinc oxide", "nano-zno",
}

# ── ANATOMICAL SITES ──
_ANATOMICAL: set[str] = {
    "liver", "spleen", "kidney", "heart", "lung", "duodenum",
    "jejunum", "ileum", "colon", "cecum", "stomach", "pancreas",
    "hypothalamus", "pituitary", "serum", "plasma", "blood", "feces",
    "urine", "saliva", "muscle", "adipose", "bone marrow", "lymph node",
}

# ── PHYSIOLOGICAL MARKERS (lowercase) ──
_PHYSIOLOGICAL: set[str] = {
    "body weight", "average daily gain", "average daily feed intake",
    "feed conversion ratio", "diarrhea rate", "diarrhea score",
    "survival rate", "total protein", "crude protein",
}

# ── KNOWN ACRONYMS ──
_KNOWN_ACRONYMS: set[str] = {
    "ADG", "ADFI", "FCR", "LPS", "BW", "FBW", "NBW", "IUGR",
    "ROS", "MDA", "TNF", "IFN", "IGA", "IGG", "IGM",
    "ALP", "ALT", "AST", "CK", "LDH", "BUN", "TC", "TG",
    "HDL", "LDL", "VLDL", "NEFA", "SOD", "CAT", "GPX",
    "T-AOC", "GSH", "PCV", "RBC", "WBC", "HB", "CFU", "MIC",
    "PBS", "PCR", "ELISA", "HPLC", "NMR",
}

# ── CONTROL GROUP NORMALIZATION ──
_CONTROL_NORM: dict[str, str] = {
    "con": "Control group", "con.": "Control group",
    "ctr": "Control group", "ctrl": "Control group",
    "control": "Control group", "control group": "Control group",
    "control diet": "Basal diet", "nc": "Negative control",
    "pc": "Positive control", "ab": "Antibiotic group",
    "antibiotic": "Antibiotic group", "antibiotic group": "Antibiotic group",
    "pbs": "PBS", "saline": "Saline",
}

# ── METHOD NAME NORMALIZATION ──
_METHOD_NORM: dict[str, str] = {
    "elisa": "ELISA", "elisa assay": "ELISA assay", "elisa kit": "ELISA kit",
    "pcr": "PCR", "rt-pcr": "RT-PCR", "qpcr": "qPCR", "rt-qpcr": "RT-qPCR",
    "western blot": "Western blot", "western blotting": "Western blotting",
    "sds-page": "SDS-PAGE", "nmr": "NMR", "gc-ms": "GC-MS", "lc-ms": "LC-MS",
    "hplc": "HPLC", "icp-oes": "ICP-OES", "icp-ms": "ICP-MS",
}

# ── DIRECTION / RELATION MAPPINGS ──
_DIRECTION_MAP: dict[str, str] = {
    "altered": "no_significant_change", "improved": "increased",
    "downregulated": "decreased", "changed": "no_significant_change",
    "affected": "no_significant_change", "regulated": "no_significant_change",
    "promoted": "increased", "reduced": "decreased",
    "alleviated": "decreased", "modified": "no_significant_change",
    "enhanced": "increased", "elevated": "increased",
    "up-regulated": "increased", "down-regulated": "decreased",
    "lowered": "decreased",
}

_RELATION_MAP: dict[str, str] = {
    "enriches": "increases", "depletes": "decreases",
    "enhances": "upregulates", "suppresses": "downregulates",
    "reduces": "decreases", "raises": "increases", "lowers": "decreases",
    "boosts": "increases", "inhibits": "decreases",
    "stimulates": "upregulates", "induces": "upregulates",
    "represses": "downregulates", "activates": "upregulates",
    "blocks": "decreases", "attenuates": "decreases",
    "modulates": "affects", "regulates": "affects", "alters": "affects",
    "impairs": "decreases", "improves": "increases",
    "ameliorates": "increases", "augments": "increases",
    "diminishes": "decreases",
}

_PRESERVE_RE: re.Pattern = re.compile(
    r"^[A-Z][A-Z0-9]{1,7}\d*$|"           # gene symbols: TLR4, SOD1
    r"^[a-z]{1,3}\d{1,3}[A-Za-z]?$|"      # p38, p65
    r"^\d+\([A-Za-z]+\)[A-Za-z0-9]*$|"     # 25(OH)D3
    r"^[α-ωΑ-Ω][\w\-].*$|"                 # Greek-lead: β-glucan
    r"^[A-Za-z]\. [a-z]"                    # binomial abbr: E. coli
)

_STOP_WORDS: frozenset[str] = frozenset({
    "of", "in", "and", "the", "to", "for", "with", "from",
    "by", "or", "at", "on", "as", "per", "via",
})


# ============================================================================
# 1. CORE NORMALIZATION FUNCTIONS (module-level for pickle)
# ============================================================================

def _norm_key(s: str) -> str:
    """Fast normalize for dedup/comparison."""
    return " ".join(s.strip().lower().split())


def _apply_case(text: str) -> str:
    """Pattern-driven case normalization.  Returns canonical form."""
    s = text.strip()
    if not s:
        return s

    # Preserve
    if _PRESERVE_RE.match(s):
        return s
    if s.upper() in _KNOWN_ACRONYMS and len(s) <= 6:
        return s

    lo = s.lower()

    # Acid → lowercase
    if lo.endswith(" acid") and lo not in _AMINO_ACIDS and "fatty" not in lo and "nucleic" not in lo:
        return lo

    # Enzyme → lowercase
    if (lo.endswith("ase") or lo.endswith("zyme")) and len(lo) > 5:
        return lo

    # Mineral compound → lowercase
    if lo in _MINERAL_COMPOUNDS:
        return lo

    # Physiological → lowercase
    if lo in _PHYSIOLOGICAL:
        return lo

    # Cytokine → lowercase
    if lo.startswith(("interleukin", "interferon", "immunoglobulin", "tumor necrosis factor")):
        return lo

    # Amino acid → Title
    if lo in _AMINO_ACIDS:
        return _title_case(s)

    # Mineral → Title
    if lo in _MINERALS:
        return _title_case(s)

    # Anatomical → Title
    if lo in _ANATOMICAL:
        return _title_case(s)

    # Microorganism → binomial
    if _is_microorganism(lo):
        return _binomial_case(s)

    # Vitamin → "Vitamin X"
    if lo.startswith("vitamin "):
        return f"Vitamin {lo[8:].strip().upper()}"
    if lo == "alpha-tocopherol":
        return "α-tocopherol"

    # Title Case fallback
    if " " in lo and s.islower():
        return _title_case(s)
    if " " not in lo and s.islower() and len(s) > 2:
        return _title_case(s)

    return s


def _title_case(text: str) -> str:
    words = text.split()
    out = []
    for i, w in enumerate(words):
        if i > 0 and w.lower() in _STOP_WORDS:
            out.append(w.lower())
        else:
            out.append(w[0].upper() + w[1:].lower() if len(w) > 1 else w.upper())
    return " ".join(out)


def _binomial_case(text: str) -> str:
    words = text.split()
    if len(words) >= 2:
        return words[0][0].upper() + words[0][1:].lower() + " " + words[1].lower() + (
            " " + " ".join(words[2:]) if len(words) > 2 else ""
        )
    return text


def _is_microorganism(name_lower: str) -> bool:
    common = {
        "bacillus", "lactobacillus", "clostridium", "escherichia",
        "salmonella", "staphylococcus", "streptococcus", "enterococcus",
        "bifidobacterium", "saccharomyces", "pediococcus", "prevotella",
        "bacteroides", "faecalibacterium", "campylobacter", "lawsonia",
        "mycobacterium", "mycoplasma", "pseudomonas", "klebsiella",
        "aspergillus", "penicillium", "candida", "porcine", "avian",
    }
    words = name_lower.split()
    if len(words) != 2:
        return False
    return words[0] in common or (words[0][0].isupper() and words[1].islower()) or (words[0].islower() and words[1].islower())


def _parse_p_value(raw: Any) -> float | str | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    for pat, repl in [
        (r"^[pP]\s*[<>=≈]+\s*", ""), (r"^[pP]\s*", ""),
        (r"\*+$", ""), (r"^<\s*", ""), (r"^>\s*", ""), (r"\s*\(.*\)$", ""),
    ]:
        s = re.sub(pat, repl, s).strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        pass
    m = re.search(r"(\d+\.?\d*)", s)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass
    return raw


# ============================================================================
# 2. SINGLE-ARTICLE CLEANING (runs in parallel workers)
# ============================================================================

def clean_one_article(payload: dict) -> dict:
    """Clean entities for a single source_doc.  Returns cleaned dict ready to merge.

    This is the work unit dispatched to every worker process.
    """
    source_doc = payload["source_doc"]
    entities: list[dict] = payload["entities"]

    # ── Phase 1: Normalize text ──
    # Build index of existing names for within-article dedup
    seen: dict[tuple[str, str], int] = {}  # (entity_type, norm_key) -> idx

    cleaned: list[dict] = []
    for row in entities:
        etype = row["entity_type"]
        et_raw = row.get("extraction_text", "")

        # Normalize extraction_text
        et_clean = _clean_text(str(et_raw))
        # Preserve the raw original text before any normalization
        original_text = str(et_raw).strip()

        # Normalize name-like attributes
        attrs = dict(row)
        # Remove metadata keys
        for k in list(attrs.keys()):
            if k in ("entity_type",):
                del attrs[k]

        # Normalize primary_text attribute
        pt = PRIMARY_TEXT.get(etype)
        if pt and pt in attrs:
            # Save the pre-normalized value as original_text if not already set
            if "original_text" not in attrs:
                attrs["original_text"] = str(attrs[pt]).strip()
            attrs[pt] = _clean_text(str(attrs[pt]))

        # Entity-specific normalization
        if etype == "Control_Group":
            gn = str(attrs.get("group_name", "")).strip().lower()
            if gn in _CONTROL_NORM:
                attrs["group_name"] = _CONTROL_NORM[gn]
        elif etype == "Method":
            mn = str(attrs.get("method_name", "")).strip().lower()
            if mn in _METHOD_NORM:
                attrs["method_name"] = _METHOD_NORM[mn]
        elif etype == "Result":
            # p_value repair
            if "p_value" in attrs and isinstance(attrs["p_value"], str):
                attrs["p_value"] = _parse_p_value(attrs["p_value"])
            # direction repair
            d = str(attrs.get("direction", "")).strip().lower()
            if d in _DIRECTION_MAP:
                attrs["direction"] = _DIRECTION_MAP[d]
            # relation_type repair
            r = str(attrs.get("relation_type", "")).strip().lower()
            if r in _RELATION_MAP:
                attrs["relation_type"] = _RELATION_MAP[r]

        row["extraction_text"] = et_clean
        row["original_text"] = original_text
        for k, v in attrs.items():
            row[k] = v

        # ── Phase 2: Within-article dedup ──
        dedup_name = _norm_key(str(attrs.get(pt, et_clean))) if pt else _norm_key(et_clean)
        dedup_key = (etype, dedup_name)
        if dedup_key in seen:
            # Merge into existing
            idx = seen[dedup_key]
            _merge_rows(cleaned[idx], row)
        else:
            seen[dedup_key] = len(cleaned)
            cleaned.append(row)

    # ── Phase 3: Gate check ──
    gate_passed = any(
        r["entity_type"] == "Alternative"
        and str(r.get("classification", "")).strip().lower() not in ("", "other", "unmatched")
        for r in cleaned
    )

    return {
        "source_doc": source_doc,
        "entities": cleaned,
        "gate_passed": gate_passed,
    }


def _clean_text(text: str) -> str:
    """Fast text normalization."""
    s = text.strip()
    if not s:
        return s
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
    s = re.sub(r" {2,}", " ", s)
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s*\(\s*\)\s*$", "", s).strip()
    if s.startswith("[") and s.endswith("]") and s.count("[") == 1:
        inner = s[1:-1].strip()
        if inner:
            s = inner
    return _apply_case(s)


def _merge_rows(base: dict, new: dict) -> None:
    """Merge new row into base (non-empty from new wins)."""
    for k, v in new.items():
        if k in ("entity_type",):
            continue
        bv = base.get(k)
        if bv is None or (isinstance(bv, str) and not bv.strip()):
            if v is not None and (not isinstance(v, str) or v.strip()):
                base[k] = v
        elif isinstance(bv, str) and isinstance(v, str) and len(v) > len(bv):
            base[k] = v


# ============================================================================
# 3. CROSS-ARTICLE GLOBAL DEDUP
# ============================================================================

def global_dedup(articles: list[dict]) -> list[dict]:
    """Dedup global entities across articles.  Only for GLOBAL_TYPES.

    Merged rows record ``_merged_from_pmids`` so surviving Neo4j nodes carry
    full provenance — every PMID that contributed to each global node is stored
    in the ``source_pmids`` column.
    """
    # Collect global entities with their article ownership
    by_type: dict[str, dict[str, list[tuple[int, int, dict]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    # entity_type -> norm_key -> [(article_idx, row_idx, row)]

    for ai, art in enumerate(articles):
        for ri, row in enumerate(art["entities"]):
            etype = row["entity_type"]
            if etype not in GLOBAL_TYPES:
                continue
            pt = PRIMARY_TEXT.get(etype)
            key = _norm_key(str(row.get(pt, row.get("extraction_text", ""))))
            by_type[etype][key].append((ai, ri, row))

    merged = 0
    for etype, groups in by_type.items():
        for key, entries in groups.items():
            if len(entries) <= 1:
                continue
            # Merge all into the first — accumulate source provenance
            base_ai, base_ri, base = entries[0]
            base_pmids: list[str] = [articles[base_ai]["source_doc"]]
            for ai, ri, row in entries[1:]:
                _merge_rows(base, row)
                base_pmids.append(articles[ai]["source_doc"])
                # Mark for removal
                articles[ai]["entities"][ri] = None
                merged += 1
            # Store provenance on the canonical row
            base["source_doc"] = articles[base_ai]["source_doc"]  # canonical PMID
            base["_merged_from_pmids"] = "|".join(sorted(set(base_pmids)))
            base["_source_pmids"] = "|".join(sorted(set(base_pmids)))

    # Remove marked entities
    for art in articles:
        art["entities"] = [r for r in art["entities"] if r is not None]

    logger.info("Global dedup: %d entities merged across articles", merged)
    return articles


# ============================================================================
# 4. GRAPH BUILDER (fast Neo4j CSV export)
# ============================================================================

def _graph_id(etype: str, name: str, pmid: str = "") -> str:
    """Generate deterministic graph node ID."""
    norm = _norm_key(name)
    if etype in GLOBAL_TYPES:
        key = f"{etype}:{norm}"
    else:
        key = f"{etype}:{norm}:{pmid}"
    h = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f"{_TYPE_PREFIX.get(etype, etype.lower()[:4])}_{h}"


def build_and_export_graph(articles: list[dict], output_dir: Path) -> tuple[int, int]:
    """Build unified graph and export Neo4j-ready CSVs with full provenance.

    Nodes
    -----
    - ``entity_id:ID`` — deterministic graph ID
    - ``entity_type:LABEL`` — ``EntityType;Entity``
    - ``name`` — canonical (normalised) display name
    - ``original_text`` — raw extraction text from the LLM (for audit)
    - ``source_pmids`` — pipe-separated PMIDs that contributed to this node
      - For **global** entities this can be many PMIDs (cross-article dedup)
      - For **article** entities this is a single PMID
    - ``_merged_from_pmids`` — for global entities, pipe-separated list of
      PMIDs that were merged into the canonical first-appearing PMID
    - All other entity attributes from the extraction

    Edges
    -----
    - ``source_id:START_ID``, ``target_id:END_ID``, ``relation_type:TYPE``
    - ``source_pmids`` — pipe-separated PMIDs for edge provenance
    """
    nodes: dict[str, dict] = {}            # node_id -> {id, entity_type, props, pmids}
    edges: dict[tuple[str, str, str], list[str]] = {}  # (src, tgt, rel) -> [pmids]

    # ── Build nodes with provenance ──
    for art in articles:
        pmid = art["source_doc"]
        for row in art["entities"]:
            etype = row["entity_type"]
            pt = PRIMARY_TEXT.get(etype, "")
            name = str(row.get(pt, row.get("extraction_text", "")))
            if not name.strip():
                continue
            nid = _graph_id(etype, name, pmid)

            # Properties: use canonical name as "name" and preserve raw text as "original_text"
            canonical_name = str(row.get("extraction_text", ""))
            raw_name = str(row.get("original_text", "")) or canonical_name

            props = {
                "name": canonical_name,
                "original_text": raw_name,
                "entity_type": etype,
            }
            for k, v in row.items():
                if k not in ("entity_type", "original_text",) and v is not None and (not isinstance(v, str) or v.strip()):
                    props[k] = v

            if nid in nodes:
                existing = nodes[nid]
                # Fill empty properties
                for k, v in props.items():
                    ev = existing["props"].get(k)
                    if ev is None or (isinstance(ev, str) and not ev.strip()):
                        if v is not None:
                            existing["props"][k] = v
                if pmid not in existing["pmids"]:
                    existing["pmids"].append(pmid)
            else:
                # Base PMIDs: the row may already carry _merged_from_pmids from
                # global_dedup — use that; otherwise start with the current article.
                inherited = str(row.get("_merged_from_pmids", ""))
                all_pmids = [p for p in inherited.split("|") if p] if inherited else [pmid]
                nodes[nid] = {"id": nid, "entity_type": etype, "props": props, "pmids": all_pmids}

    # ── Build edges with provenance and evidence_text ──
    # Also collect for the relationships CSV export
    relationships_rows: list[dict] = []

    rel_path = Path("output/review/review/relationships.csv")
    if rel_path.exists():
        logger.info("Loading relationships from %s", rel_path)
        for chunk in pd.read_csv(rel_path, chunksize=100_000, dtype=str):
            for _, r in chunk.iterrows():
                src_doc = str(r.get("source_doc", ""))
                src_type = str(r.get("source_type", ""))
                src_text = str(r.get("source_text", ""))
                rel = str(r.get("relation", ""))
                tgt_type = str(r.get("target_type", ""))
                tgt_text = str(r.get("target_text", ""))

                src_id = _graph_id(src_type, src_text, src_doc)
                tgt_id = _graph_id(tgt_type, tgt_text, src_doc)

                key = (src_id, tgt_id, rel)
                if key not in edges:
                    edges[key] = []
                if src_doc not in edges[key]:
                    edges[key].append(src_doc)

                # Collect for relationships CSV
                relationships_rows.append({
                    "source_doc": src_doc,
                    "source_type": src_type,
                    "source_text": src_text,
                    "source_id": src_id,
                    "relation": rel,
                    "target_type": tgt_type,
                    "target_text": tgt_text,
                    "target_id": tgt_id,
                })

    # ── Lookup evidence_text from nodes for each edge ──
    # For every relationship, pull evidence_text from the source entity node
    for rel_row in relationships_rows:
        src_id = rel_row["source_id"]
        if src_id in nodes:
            nd = nodes[src_id]
            ev = nd["props"].get("evidence_text", "") or nd["props"].get("evidence_text.1", "")
            if ev:
                rel_row["evidence_text"] = ev[:500]  # truncate for CSV

    # ── Write nodes.csv ──
    # Strip internal helper columns that should not appear in downstream data
    _INTERNAL_COLS: set[str] = {
        "_auto_created", "_decomposed_from", "_gate_preserved",
        "_merged_from_pmids", "_source_pmids",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    all_props: set[str] = set()
    for nd in nodes.values():
        all_props.update(k for k in nd["props"].keys() if k not in _INTERNAL_COLS)

    with open(output_dir / "nodes.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["entity_id:ID", "entity_type:LABEL", "source_pmids"] + sorted(all_props)
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for nd in nodes.values():
            row = {
                "entity_id:ID": nd["id"],
                "entity_type:LABEL": f"{nd['entity_type']};Entity",
                "source_pmids": "|".join(sorted(nd["pmids"])),
            }
            # Only write non-internal properties
            clean_props = {k: v for k, v in nd["props"].items() if k not in _INTERNAL_COLS}
            row.update(clean_props)
            w.writerow(row)

    # ── Write edges.csv (Neo4j import — no evidence_text, clean format) ──
    with open(output_dir / "edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_id:START_ID", "target_id:END_ID", "relation_type:TYPE", "source_pmids"])
        for (src, tgt, rel), pmids in sorted(edges.items()):
            w.writerow([src, tgt, rel, "|".join(sorted(pmids))])

    # ── Write relationships.csv to entities/ (human-readable, with evidence) ──
    entities_dir = output_dir.parent / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing relationships CSV (%d rows) to entities/ ...", len(relationships_rows))
    rel_cols = ["source_doc", "source_type", "source_text", "source_id",
                "relation", "target_type", "target_text", "target_id",
                "evidence_text"]
    rel_cols = ["source_doc", "source_type", "source_text", "source_id",
                "relation", "target_type", "target_text", "target_id",
                "evidence_text"]
    with open(entities_dir / "relationships.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rel_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(relationships_rows)

    return len(nodes), len(edges)


# ============================================================================
# 5. ENTITY REVIEW CSV EXPORT
# ============================================================================

def export_entity_csvs(articles: list[dict], output_dir: Path) -> None:
    """Export one CSV per entity type, plus all_entities (cleaned data)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all entities
    by_type: dict[str, list[dict]] = defaultdict(list)
    for art in articles:
        src = art["source_doc"]
        for row in art["entities"]:
            row["source_doc"] = src
            by_type[row["entity_type"]].append(row)

    # Per-type CSVs
    for fname, etype in sorted(ENTITY_MAP.items()):
        rows = by_type.get(etype, [])
        if not rows:
            continue
        all_cols = list(dict.fromkeys(
            ["source_doc", "extraction_text", "evidence_text", "source_location"]
            + [k for row in rows for k in row if k not in ("entity_type", "source_doc", "extraction_text", "evidence_text", "source_location")]
        ))
        _write_csv(output_dir / fname, rows, all_cols)

    # all_entities.csv
    all_rows = []
    for etype, rows in by_type.items():
        for row in rows:
            r = dict(row)
            r["entity_type"] = etype
            all_rows.append(r)
    all_cols = ["source_doc", "entity_type", "extraction_text", "evidence_text", "source_location"]
    extra = [k for row in all_rows for k in row if k not in all_cols]
    all_cols += list(dict.fromkeys(extra))
    _write_csv(output_dir / "all_entities.csv", all_rows, all_cols)

    logger.info("Cleaned entity CSVs exported: %d types, %d entities", len(by_type), len(all_rows))


def export_raw_entity_csvs(all_articles: list[dict], output_dir: Path) -> None:
    """Export one CSV per entity type from raw (pre-cleaning) gate-passed articles.

    These are the uncleaned extraction data, saved for audit before any
    normalization, dedup, or repair is applied.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    by_type: dict[str, list[dict]] = defaultdict(list)
    for art in all_articles:
        src = art["source_doc"]
        for row in art["entities"]:
            row["source_doc"] = src
            by_type[row["entity_type"]].append(row)

    for fname, etype in sorted(ENTITY_MAP.items()):
        rows = by_type.get(etype, [])
        if not rows:
            continue
        all_cols = list(dict.fromkeys(
            ["source_doc", "extraction_text", "evidence_text", "source_location"]
            + [k for row in rows for k in row if k not in ("entity_type", "source_doc", "extraction_text", "evidence_text", "source_location")]
        ))
        _write_csv(output_dir / fname, rows, all_cols)

    # all_entities.csv
    all_rows = []
    for etype, rows in by_type.items():
        for row in rows:
            r = dict(row)
            r["entity_type"] = etype
            all_rows.append(r)
    all_cols = ["source_doc", "entity_type", "extraction_text", "evidence_text", "source_location"]
    extra = [k for row in all_rows for k in row if k not in all_cols]
    all_cols += list(dict.fromkeys(extra))
    _write_csv(output_dir / "all_entities.csv", all_rows, all_cols)

    logger.info("Raw entity CSVs exported: %d types, %d entities", len(by_type), len(all_rows))


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Fast CSV write with pre-computed columns."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ============================================================================
# 6. LITERATURE MANIFEST
# ============================================================================

def export_literature(articles: list[dict], path: Path) -> int:
    """Export literature.csv for gate-passed articles."""
    rows = []
    for art in articles:
        row = {"source_doc": art["source_doc"]}
        for e in art["entities"]:
            if e["entity_type"] == "Literature":
                row.update({k: v for k, v in e.items() if k != "entity_type"})
                break
        rows.append(row)

    all_cols = list(dict.fromkeys(
        ["source_doc"] + [k for row in rows for k in row if k != "source_doc"]
    ))
    _write_csv(path, rows, all_cols)
    return len(rows)


# ============================================================================
# 7. RELATIONSHIP CSV GENERATORS (source + cleaned)
# ============================================================================

def _source_entity_id(etype: str, name: str, pmid: str) -> str:
    """Generate entity ID for raw (pre-cleaning) data — always article-scoped,
    since no cross-article dedup has happened yet.
    """
    return _graph_id(etype, name, pmid)


def _write_source_relationships(articles: list[dict], path: Path) -> None:
    """Export raw relationships.csv for source data (pre-cleaning).

    Uses the original extraction_text from the raw review relationships.csv
    to build (source_type, source_text, relation, target_type, target_text)
    rows.  Evidence_text is left empty since the raw review data has not
    had its evidence aligned by the cleaning pipeline.
    """
    rel_path = Path("output/review/review/relationships.csv")
    if not rel_path.exists():
        _write_csv(path, [], ["source_doc", "source_type", "source_text", "source_id",
                               "relation", "target_type", "target_text", "target_id",
                               "evidence_text"])
        return

    rows = []
    for chunk in pd.read_csv(rel_path, chunksize=100_000, dtype=str):
        for _, r in chunk.iterrows():
            src_doc = str(r.get("source_doc", ""))
            src_type = str(r.get("source_type", ""))
            src_text = str(r.get("source_text", ""))
            rel = str(r.get("relation", ""))
            tgt_type = str(r.get("target_type", ""))
            tgt_text = str(r.get("target_text", ""))
            rows.append({
                "source_doc": src_doc,
                "source_type": src_type,
                "source_text": src_text,
                "source_id": _source_entity_id(src_type, src_text, src_doc),
                "relation": rel,
                "target_type": tgt_type,
                "target_text": tgt_text,
                "target_id": _source_entity_id(tgt_type, tgt_text, src_doc),
                "evidence_text": "",
            })

    logger.info("Writing source relationships (%d rows) ...", len(rows))
    rel_cols = ["source_doc", "source_type", "source_text", "source_id",
                "relation", "target_type", "target_text", "target_id",
                "evidence_text"]
    _write_csv(path, rows, rel_cols)


def _write_cleaned_relationships(articles: list[dict], path: Path) -> None:
    """Export cleaned relationships.csv with evidence_text linked to cleaned entities.

    Reads the raw review relationships.csv and resolves source/target IDs
    using the CLEANED entity graph IDs.  Evidence_text is looked up from
    the cleaned node attributes.
    """
    rel_path = Path("output/review/review/relationships.csv")
    if not rel_path.exists():
        _write_csv(path, [], ["source_doc", "source_type", "source_text", "source_id",
                               "relation", "target_type", "target_text", "target_id",
                               "evidence_text"])
        return

    # Build node lookup from cleaned articles: (etype, original_name, pmid) -> cleaned entity
    node_attrs: dict[tuple[str, str, str], dict] = {}  # (etype, orig_text, pmid) -> props
    for art in articles:
        pmid = art["source_doc"]
        for row in art["entities"]:
            etype = row["entity_type"]
            orig = str(row.get("original_text", row.get("extraction_text", "")))
            pt = PRIMARY_TEXT.get(etype, "")
            name = str(row.get(pt, row.get("extraction_text", "")))
            key = (etype, orig, pmid)
            node_attrs[key] = {
                "graph_id": _graph_id(etype, name, pmid),
                "evidence_text": row.get("evidence_text", "") or row.get("evidence_text.1", ""),
                "cleaned_name": name,
            }

    rows = []
    for chunk in pd.read_csv(rel_path, chunksize=100_000, dtype=str):
        for _, r in chunk.iterrows():
            src_doc = str(r.get("source_doc", ""))
            src_type = str(r.get("source_type", ""))
            src_text = str(r.get("source_text", ""))
            rel = str(r.get("relation", ""))
            tgt_type = str(r.get("target_type", ""))
            tgt_text = str(r.get("target_text", ""))

            src_key = (src_type, src_text, src_doc)
            tgt_key = (tgt_type, tgt_text, src_doc)
            src_info = node_attrs.get(src_key, {})
            tgt_info = node_attrs.get(tgt_key, {})

            src_id = src_info.get("graph_id", _graph_id(src_type, src_text, src_doc))
            tgt_id = tgt_info.get("graph_id", _graph_id(tgt_type, tgt_text, src_doc))
            ev = src_info.get("evidence_text", "")[:500]

            rows.append({
                "source_doc": src_doc,
                "source_type": src_type,
                "source_text": src_text,
                "source_id": src_id,
                "relation": rel,
                "target_type": tgt_type,
                "target_text": tgt_text,
                "target_id": tgt_id,
                "evidence_text": ev,
            })

    logger.info("Writing cleaned relationships (%d rows) ...", len(rows))
    rel_cols = ["source_doc", "source_type", "source_text", "source_id",
                "relation", "target_type", "target_text", "target_id",
                "evidence_text"]
    _write_csv(path, rows, rel_cols)


# ============================================================================
# 8. LOAD DATA (parallel-read, single-pass)
# ============================================================================

def load_review_data(input_dir: str) -> list[dict]:
    """Load all review CSVs and group entities by source_doc.  Returns list of article dicts."""
    ip = Path(input_dir)
    articles: dict[str, list[dict]] = defaultdict(list)

    for fname, etype in ENTITY_MAP.items():
        path = ip / fname
        if not path.exists():
            continue
        logger.info("Loading %s ...", fname)
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        # Pre-compute entity_type
        rows = df.to_dict("records")
        for r in rows:
            r["entity_type"] = etype
            src = r.get("source_doc", "") or r.get("pmid", "") or ""
            articles[str(src)].append(r)
        logger.info("  -> %d rows for %d source_docs", len(rows), len(articles))

    # Convert to list of dicts for pickle dispatch
    result: list[dict] = [
        {"source_doc": src, "entities": ents}
        for src, ents in articles.items()
    ]
    logger.info("Loaded %d articles, %d total entities", len(result), sum(len(a["entities"]) for a in result))
    return result


# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Clean and deliver extraction data")
    parser.add_argument("--input", "-i", default="output/review/review", help="Review CSV directory")
    parser.add_argument("--output", "-o", default="output/delivery", help="Delivery output directory")
    parser.add_argument("--workers", "-w", type=int, default=min(8, os.cpu_count() or 4),
                        help="Parallel workers (default: min(8, cpu_count))")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ── Load ──
    articles = load_review_data(args.input)
    logger.info("Loaded in %.1fs", time.time() - t0)

    # ── Parallel per-article cleaning ──
    t1 = time.time()
    logger.info("Cleaning with %d workers ...", args.workers)
    cleaned: list[dict] = []
    gate_passed: list[dict] = []
    raw_gate_passed: list[dict] = []   # snapshot of raw entities before cleaning
    gate_failed: int = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(clean_one_article, a): a for a in articles}
        for fut in as_completed(futures):
            result = fut.result()
            cleaned.append(result)
            if result["gate_passed"]:
                gate_passed.append(result)

                # Snapshot raw entities: deep-copy before they are mutated by dedup
                import copy as _copy
                raw_entities = _copy.deepcopy(result["entities"])
                raw_gate_passed.append({
                    "source_doc": result["source_doc"],
                    "entities": raw_entities,
                })
            else:
                gate_failed += 1

    logger.info(
        "Per-article cleaning done in %.1fs: %d passed, %d failed",
        time.time() - t1, len(gate_passed), gate_failed,
    )

    # ── Cross-article global dedup (single-threaded, fast) ──
    t2 = time.time()
    gate_passed = global_dedup(gate_passed)
    logger.info("Global dedup done in %.1fs", time.time() - t2)

    # ── Literature manifest ──
    t3 = time.time()
    n_lit = export_literature(gate_passed, out / "literature.csv")
    logger.info("Literature: %d articles in %.1fs", n_lit, time.time() - t3)

    # ── SOURCE: raw entities + raw relationships (pre-cleaning, gate-passed) ──
    t4 = time.time()
    export_raw_entity_csvs(raw_gate_passed, out / "source")
    _write_source_relationships(raw_gate_passed, out / "source" / "relationships.csv")
    logger.info("Source CSVs in %.1fs", time.time() - t4)

    # ── CLEANED: normalized + deduped entities + relationships ──
    t5 = time.time()
    export_entity_csvs(gate_passed, out / "cleaned")
    _write_cleaned_relationships(gate_passed, out / "cleaned" / "relationships.csv")
    logger.info("Cleaned CSVs in %.1fs", time.time() - t5)

    # ── Neo4j graph ──
    t6 = time.time()
    n_nodes, n_edges = build_and_export_graph(gate_passed, out / "neo4j")
    logger.info("Neo4j: %d nodes, %d edges in %.1fs", n_nodes, n_edges, time.time() - t6)

    # ── Summary ──
    total_time = time.time() - t0
    total_ents = sum(len(a["entities"]) for a in cleaned)
    delivered_ents = sum(len(a["entities"]) for a in gate_passed)

    summary = (
        f"# Delivery Summary\n"
        f"Articles: {len(gate_passed)} passed / {gate_failed} failed / {len(articles)} total\n"
        f"Entities: {delivered_ents} delivered / {total_ents} cleaned\n"
        f"Graph: {n_nodes} nodes, {n_edges} edges\n"
        f"Time: {total_time:.1f}s\n"
        f"Workers: {args.workers}\n"
    )
    (out / "delivery_summary.txt").write_text(summary)
    logger.info("\n%s", summary)

    return 0


if __name__ == "__main__":
    # Must freeze before fork for macOS compatibility
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
