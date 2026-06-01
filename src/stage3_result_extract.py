"""Stage 3: Result and relationship extraction with Pass 1 (LLM) and Pass 2 (alignment)."""
from typing import Optional
from src.models import Result, Relationship, Indicator, TissueSite, ControlGroup


def align_result_to_indicator(result: dict, indicators: list[Indicator]) -> Optional[str]:
    """Match result's indicator_abbreviation to an Indicator entity. Returns entity_id or None."""
    abbr = result.get("indicator_abbreviation", "").strip().lower()
    if not abbr:
        return None

    # 1. Exact match on abbreviation
    for ind in indicators:
        if ind.abbreviation.lower().strip() == abbr:
            return ind.entity_id

    # 2. Exact match on standard_name
    for ind in indicators:
        if ind.standard_name.lower().strip() == abbr:
            return ind.entity_id

    # 3. Containment match (fuzzy)
    for ind in indicators:
        ind_abbr = ind.abbreviation.lower().strip()
        if ind_abbr and (ind_abbr in abbr or abbr in ind_abbr):
            return ind.entity_id

    return None


def align_result_to_tissue(result: dict, tissues: list[TissueSite]) -> Optional[str]:
    """Match result's tissue_site to a TissueSite entity."""
    site = result.get("tissue_site", "").strip().lower()
    if not site:
        return None
    for ts in tissues:
        if ts.site_name.lower().strip() == site:
            return ts.entity_id
    for ts in tissues:
        if ts.site_name.lower().strip() in site or site in ts.site_name.lower().strip():
            return ts.entity_id
    return None


# Relation type -> expected direction mapping
DIRECTION_CONSISTENCY_MAP = {
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
        return True  # "affects" matches anything
    return expected == direction


def check_compared_to_baseline(
    result: dict,
    control_groups: list[ControlGroup],
    is_challenge_model: bool,
) -> tuple[bool, str]:
    """Validate compared_to_group for challenge models (BACKGROUND 4.1)."""
    if not is_challenge_model:
        return True, ""

    compared_to = result.get("compared_to_group", "")
    for cg in control_groups:
        if cg.group_name == compared_to and cg.group_type in ("negative_control", "sham"):
            return True, ""

    correct = [cg.group_name for cg in control_groups if cg.group_type in ("negative_control", "sham")]
    return False, f"Challenge model should compare vs challenged control ({correct}), got '{compared_to}'"


def build_relation_records(
    result_entity: Result,
    intervention_entity_id: str,
    indicator_entity_id: str,
    tissue_entity_id: str,
    control_group_entity_id: Optional[str],
) -> list[Relationship]:
    """Build all Relationship records for a Result entity."""
    rels = []

    if result_entity.relation_type and intervention_entity_id:
        rels.append(Relationship(
            rel_type=result_entity.relation_type,
            head_entity_type="Intervention",
            head_entity_id=intervention_entity_id,
            tail_entity_type="Result",
            tail_entity_id=result_entity.entity_id,
            evidence_text=result_entity.evidence_text,
            source_location=result_entity.source_location,
        ))

    if indicator_entity_id:
        rels.append(Relationship(
            rel_type="corresponds_to",
            head_entity_type="Result",
            head_entity_id=result_entity.entity_id,
            tail_entity_type="Indicator",
            tail_entity_id=indicator_entity_id,
        ))

    if tissue_entity_id:
        rels.append(Relationship(
            rel_type="occurs_in",
            head_entity_type="Result",
            head_entity_id=result_entity.entity_id,
            tail_entity_type="Tissue_Site",
            tail_entity_id=tissue_entity_id,
        ))

    if control_group_entity_id and result_entity.compared_to_group:
        rels.append(Relationship(
            rel_type="compared_to",
            head_entity_type="Result",
            head_entity_id=result_entity.entity_id,
            tail_entity_type="Control_Group",
            tail_entity_id=control_group_entity_id,
        ))

    return rels
