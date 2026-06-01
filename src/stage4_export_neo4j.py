"""Stage 4: Neo4j Cypher script generation."""
from pathlib import Path
from dataclasses import asdict
from typing import Optional, List, Any
from src.config import OUTPUT_NEO4J_DIR

NEO4J_LABEL_MAP = {
    "Alternative": "Alternative", "Alternative_Class": "Alternative_Class",
    "Composite_Product": "Composite_Product", "Literature": "Literature",
    "Experiment": "Experiment", "Swine_Model": "Swine_Model",
    "Swine": "Swine", "Intervention": "Intervention",
    "Control_Group": "Control_Group", "Tissue_Site": "Tissue_Site",
    "Indicator": "Indicator", "Result": "Result", "Method": "Method",
}

SKIP_PROPERTIES = {"entity_id", "components", "matched_indicator", "matched_tissue"}


def _escape_cypher(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    s = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def generate_cypher(entities: List, entity_type: str) -> str:
    """Generate Cypher MERGE statements for a list of entities."""
    label = NEO4J_LABEL_MAP.get(entity_type, entity_type)
    lines = []
    for e in entities:
        d = asdict(e) if hasattr(e, '__dataclass_fields__') else e
        eid = d.get("entity_id", d.get("doi", d.get("experiment_id", d.get("class_name", ""))))
        lines.append(f"MERGE (n:{label} {{entity_id: {_escape_cypher(eid)}}})")
        for key, value in d.items():
            if key in SKIP_PROPERTIES or key == "entity_id":
                continue
            if value is not None and value != "":
                lines.append(f"  SET n.{key} = {_escape_cypher(value)}")
        lines.append("")
    return "\n".join(lines)


def generate_indexes() -> str:
    lines = []
    for label in NEO4J_LABEL_MAP.values():
        lines.append(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.entity_id);")
    lines.append("CREATE INDEX IF NOT EXISTS FOR (n:Alternative) ON (n.standard_name);")
    lines.append("CREATE INDEX IF NOT EXISTS FOR (n:Literature) ON (n.doi);")
    return "\n".join(lines)
