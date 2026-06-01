"""Stage 4: Validation engine with schema-level and business-rule checks."""
from typing import Optional


class ValidationReport:
    """Collects and reports validation findings at three severity levels."""

    def __init__(self):
        self.fatals: list[dict] = []
        self.warnings: list[dict] = []
        self.infos: list[dict] = []

    def add(self, severity: str, entity_id: Optional[str], message: str, details: Optional[dict] = None):
        entry = {"severity": severity, "entity_id": entity_id, "message": message}
        if details:
            entry["details"] = details
        if severity == "FATAL":
            self.fatals.append(entry)
        elif severity == "WARNING":
            self.warnings.append(entry)
        else:
            self.infos.append(entry)

    def summary(self) -> dict:
        return {"FATAL": len(self.fatals), "WARNING": len(self.warnings), "INFO": len(self.infos)}

    def to_dict(self) -> dict:
        return {"summary": self.summary(), "fatals": self.fatals, "warnings": self.warnings, "infos": self.infos}


def validate_required_fields(
    entities: list, entity_type: str, required_fields: list[str], report: ValidationReport
):
    """Check that all required fields are non-empty for each entity."""
    for e in entities:
        for field in required_fields:
            value = getattr(e, field, None)
            if value is None or (isinstance(value, str) and not str(value).strip()):
                report.add(
                    "FATAL",
                    getattr(e, "entity_id", str(e)),
                    f"{entity_type}: required field '{field}' is empty or None",
                )


def validate_enum_values(
    entities: list, entity_type: str, enum_fields: dict[str, list[str]], report: ValidationReport
):
    """Check that enum fields contain only allowed values."""
    for e in entities:
        for field, allowed in enum_fields.items():
            value = getattr(e, field, None)
            if value is None or (isinstance(value, str) and not str(value).strip()):
                continue  # Skip None/empty (handled by required_fields check)
            if value not in allowed:
                report.add(
                    "FATAL",
                    getattr(e, "entity_id", str(e)),
                    f"{entity_type}.{field}: '{value}' not in allowed values {allowed}",
                )


def validate_foreign_keys(
    results: list,
    indicators_keyed: dict[str, str],
    tissues_keyed: dict[str, str],
    controls_keyed: dict[str, str],
    interventions_keyed: dict[str, str],
    report: ValidationReport,
):
    """Validate that Result foreign keys reference existing entities."""
    for r in results:
        if r.matched_indicator and r.matched_indicator not in indicators_keyed:
            report.add(
                "WARNING", r.entity_id,
                f"Result.matched_indicator '{r.matched_indicator}' not found in Indicator entities"
            )
        if r.matched_tissue and r.matched_tissue not in tissues_keyed:
            report.add(
                "WARNING", r.entity_id,
                f"Result.matched_tissue '{r.matched_tissue}' not found in Tissue_Site entities"
            )


def validate_business_rules(
    results: list,
    controls: list,
    swine_models: list,
    composites: list,
    others: list,
    report: ValidationReport,
):
    """Validate BACKGROUND.md business rules across entities."""
    # Build experiment_id -> is_challenge map
    challenge_map = {}
    for sm in swine_models:
        challenge_map[sm.experiment_id] = sm.model_type == "challenge"

    # Rule 4.1: Challenge model -> compared_to must be challenged control
    for r in results:
        is_challenge = challenge_map.get(r.experiment_id, False)
        if is_challenge and r.compared_to_group:
            matched = [
                c for c in controls
                if c.group_name == r.compared_to_group and c.experiment_id == r.experiment_id
            ]
            if matched and matched[0].group_type == "basal_control":
                report.add(
                    "WARNING", r.entity_id,
                    f"Challenge model result compared to basal_control '{r.compared_to_group}' instead of challenged control"
                )

    # Rule 4.2: Direction consistency
    direction_map = {
        "increases": "increased", "decreases": "decreased",
        "upregulates": "increased", "downregulates": "decreased",
        "enriches": "increased", "depletes": "decreased",
    }
    for r in results:
        expected = direction_map.get(r.relation_type)
        if expected and expected != r.direction:
            report.add(
                "WARNING", r.entity_id,
                f"relation_type={r.relation_type} vs direction={r.direction} mismatch (expected {expected})"
            )

    # Composite_Product must have components
    for cp in composites:
        if not getattr(cp, 'components', None) or len(cp.components) == 0:
            report.add("WARNING", cp.entity_id, "Composite_Product has no has_component relationships")

    # Other must NOT have components
    for o in others:
        if getattr(o, 'components', None) and o.components:
            report.add("FATAL", o.entity_id, "Other entity has components (should be Composite_Product)")
