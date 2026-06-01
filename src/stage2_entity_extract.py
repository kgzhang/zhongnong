"""Stage 2: Entity extraction orchestrator — dispatches LLM calls for Modules C/A/B."""
from dataclasses import asdict


def build_entity_id(doi: str, entity_type_abbr: str, seq: int) -> str:
    """Build unique entity ID: DOI_safe + type abbreviation + zero-padded sequence (6 digits)."""
    doi_safe = doi.replace(":", "_")
    return f"{doi_safe}_{entity_type_abbr}_{seq:06d}"


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


def entities_to_tsv_rows(entities: list) -> list[dict]:
    """Convert a list of entity dataclass instances to list of dicts for TSV writing."""
    return [asdict(e) for e in entities]


def postprocess_alternatives(data: dict) -> dict:
    """Deduplicate alternatives within same DOI by standard_name (case-insensitive)."""
    seen = set()
    deduped = []
    for alt in data.get("alternatives", []):
        key = alt.get("standard_name", "").lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(alt)
    data["alternatives"] = deduped
    return data


def postprocess_experiment_design(data: dict) -> dict:
    """Normalize dose units in intervention data."""
    for inter in data.get("interventions", []):
        if inter.get("dose_value") is not None and inter.get("dose_unit_original"):
            new_val, new_unit = normalize_dose_unit(
                float(inter.get("dose_value", 0)), inter.get("dose_unit_original", "")
            )
            inter["dose_unit_standard"] = new_unit
            inter["dose_value"] = new_val
    return data
