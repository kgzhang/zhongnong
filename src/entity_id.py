"""Entity ID system — global IDs for shared entities, local IDs for paper-specific ones.

Global entities (name-based, stable across papers):
  Alternative:       ALT:{standard_name}
  Tissue_Site:       TIS:{site_name}
  Indicator:         IND:{standard_name}
  Method:            MET:{method_name}
  Swine breed:       SWN:{breed}

Local entities (paper-specific, PMID+seq):
  Experiment:        {pmid}_EXP_{seq}
  Intervention:      {pmid}_INT_{seq}
  Control_Group:     {pmid}_CTL_{seq}
  Result:            {pmid}_RES_{seq}
  Literature:        {pmid}

Note: Alternative_Class nodes are NOT generated — they are classification metadata,
not extracted entities. The class is stored as an attribute on Alternative nodes.
"""
import hashlib
import re


def global_id(entity_type: str, name: str) -> str:
    """Build a stable global ID from entity type + name (lowercase, dedup-safe)."""
    key = re.sub(r'\s+', ' ', name.strip().lower())
    prefix = {
        "Alternative": "ALT",
        "Tissue_Site": "TIS",
        "Indicator": "IND",
        "Method": "MET",
        "Swine": "SWN",
    }
    p = prefix.get(entity_type, entity_type[:3].upper())
    return f"{p}:{key}"


def local_id(pmid: str, entity_type: str, seq: int) -> str:
    """Build a paper-scoped local ID from PMID + type + sequence."""
    abbr = {
        "Experiment": "EXP",
        "Intervention": "INT",
        "Control_Group": "CTL",
        "Result": "RES",
        "Literature": "LIT",
    }
    a = abbr.get(entity_type, entity_type[:3].upper())
    return f"{pmid}_{a}_{seq:04d}"


def evidence_id(text: str) -> str:
    """Build a short evidence ID from SHA256 hash of the text."""
    if not text:
        return "EV:NONE"
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"EV:{h}"


class EvidenceRegistry:
    """Global registry mapping evidence_id → evidence_text. Deduplicates automatically."""

    def __init__(self):
        self._store: dict[str, str] = {}

    def register(self, text: str) -> str:
        if not text:
            return "EV:NONE"
        eid = evidence_id(text)
        if eid not in self._store:
            self._store[eid] = text
        return eid

    def get(self, eid: str) -> str:
        return self._store.get(eid, "")

    def to_rows(self) -> list[dict]:
        return [{"evidence_id": k, "evidence_text": v} for k, v in self._store.items()]

    def __len__(self) -> int:
        return len(self._store)
