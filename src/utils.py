"""Shared utilities used across pipeline stages.

Extracted from stage2_entity_extract.py and stage3_result_extract.py to
eliminate cross-stage imports of private functions.
"""
import json
import logging
import re
from pathlib import Path
from typing import Optional

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dose normalization (from stage2_entity_extract)
# ---------------------------------------------------------------------------

def normalize_dose_unit(value: float, unit: str) -> tuple[float, str]:
    """Normalize dose units per BACKGROUND.md 3.4.

    Rules:
    - ppm -> mg/kg_feed (1:1)
    - % -> mg/kg_feed (1% = 10000 mg/kg)
    - mg/kg BW -> mg/kg_BW
    - mg/kg (without BW) -> mg/kg_feed
    - g/kg -> g/kg (pass through)
    - Unknown units pass through unchanged.
    """
    unit_lower = unit.lower().strip()
    if "ppm" in unit_lower:
        return value, "mg/kg_feed"
    if "%" in unit_lower:
        return value * 10000, "mg/kg_feed"
    if "mg/kg bw" in unit_lower or "mg/kg_bw" in unit_lower:
        return value, "mg/kg_BW"
    if "mg/kg" in unit_lower:
        return value, "mg/kg_feed"
    if "g/kg" in unit_lower:
        return value, "g/kg"
    return value, unit


# ---------------------------------------------------------------------------
# Direction / significance / relation type normalization
# ---------------------------------------------------------------------------

DIRECTION_CONSISTENCY_MAP: dict[str, Optional[str]] = {
    "increases": "increased",
    "decreases": "decreased",
    "upregulates": "increased",
    "downregulates": "decreased",
    "enriches": "increased",
    "depletes": "decreased",
    "affects": None,  # always consistent
}


def check_direction_consistency(rel_type: str, direction: str) -> bool:
    """Check that relation_type and direction are consistent."""
    expected = DIRECTION_CONSISTENCY_MAP.get(rel_type)
    if expected is None:
        return True
    return expected == direction


def normalize_result_values(r: dict) -> None:
    """Normalize LLM value variations to match our enum/type expectations (in-place)."""
    # Direction: "decreases" -> "decreased", "increases" -> "increased"
    d = str(r.get("direction", "")).strip().lower()
    if d in ("decreases", "decrease", "reduces", "reduce", "lowered", "lower"):
        r["direction"] = "decreased"
    elif d in ("increases", "increase", "raises", "raise", "elevated", "elevate"):
        r["direction"] = "increased"
    elif d in ("no_change", "no change", "unchanged", "ns"):
        r["direction"] = "no_significant_change"

    # Significance: "significant" -> "p_less_0.05"
    s = str(r.get("significance_level", "")).strip().lower()
    if s in ("p<0.01", "p < 0.01", "highly significant", "highly_significant"):
        r["significance_level"] = "p_less_0.01"
    elif s in ("p<0.05", "p < 0.05", "significant", "sig"):
        r["significance_level"] = "p_less_0.05"
    elif s in ("trend", "tendency", "0.05<p<0.10", "0.05 < p < 0.10"):
        r["significance_level"] = "trend_0.05_0.1"
    elif s in ("ns", "not significant", "non-significant", "n.s."):
        r["significance_level"] = "not_significant"

    # p_value: parse "P=0.023" -> 0.023, "0.05<P<0.10" -> None
    p = r.get("p_value")
    if isinstance(p, str):
        m = re.search(r'=\s*([0-9.]+)', str(p))
        r["p_value"] = float(m.group(1)) if m else None

    # Relation type
    rel = str(r.get("relation_type", "")).strip().lower()
    if rel in ("decreases", "decrease", "reduced"):
        r["relation_type"] = "decreases"
    elif rel in ("increases", "increase", "elevated"):
        r["relation_type"] = "increases"
    elif rel in ("up", "upregulated", "up-regulates"):
        r["relation_type"] = "upregulates"
    elif rel in ("down", "downregulated", "down-regulates"):
        r["relation_type"] = "downregulates"
    elif rel in ("enriched", "enrichment"):
        r["relation_type"] = "enriches"
    elif rel in ("depleted", "depletion"):
        r["relation_type"] = "depletes"


# ---------------------------------------------------------------------------
# Entity ID building
# ---------------------------------------------------------------------------

def build_entity_id(doi: str, entity_type_abbr: str, seq: int) -> str:
    """Legacy ID builder — prefer entity_id.py:global_id/local_id instead.

    Builds: {doi_safe}_{abbr}_{seq:06d}
    """
    doi_safe = doi.replace(":", "_")
    return f"{doi_safe}_{entity_type_abbr}_{seq:06d}"


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def load_section_json(doi_safe: str, sections_dir: Optional[Path] = None) -> Optional[dict]:
    """Load a single article's structured section JSON."""
    d = sections_dir or settings.sections_dir
    path = d / f"{doi_safe}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def checkpoint_exists(doi_safe: str, module: str, entities_dir: Optional[Path] = None) -> bool:
    """Check if entity extraction output already exists for this article+module."""
    d = entities_dir or settings.entities_dir
    return (d / doi_safe / f"{module}.json").exists()


def save_entity_output(
    doi_safe: str, module: str, data: dict, entities_dir: Optional[Path] = None
) -> Path:
    """Save LLM extraction output for one article+module."""
    d = entities_dir or settings.entities_dir
    out_dir = d / doi_safe
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{module}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


def list_section_files(sections_dir: Optional[Path] = None) -> list[Path]:
    """List all JSON section files excluding _prefixed files."""
    d = sections_dir or settings.sections_dir
    if not d.exists():
        return []
    return sorted(p for p in d.glob("*.json") if not p.name.startswith("_"))
