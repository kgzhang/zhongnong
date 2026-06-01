"""Stage 4: TSV export for all entity types."""
import csv
from pathlib import Path
from dataclasses import asdict
from typing import Optional
from src.config import OUTPUT_TSV_DIR

ENTITY_COLUMNS = {
    "Alternative": ["entity_id","standard_name","abbreviation","cas_number","source_organism","is_synthetic","alternative_class","subclass","match_source","original_text","evidence_text","source_location"],
    "Alternative_Class": ["class_name","level","description"],
    "Composite_Product": ["entity_id","product_name","manufacturer","is_commercial","original_text","evidence_text","source_location"],
    "Literature": ["doi","pmid","title","journal","abstract_conclusion","publication_year","publication_date","study_design"],
    "Experiment": ["experiment_id","doi","description","evidence_text","source_location"],
    "Swine_Model": ["entity_id","experiment_id","model_type","stressor_name","challenge_method","challenge_dose","challenge_timing","evidence_text","source_location"],
    "Swine": ["entity_id","experiment_id","breed","sex","age","physiological_stage","initial_body_weight","sample_size","evidence_text","source_location"],
    "Intervention": ["entity_id","experiment_id","intervention_target","dose_value","dose_unit_original","dose_unit_standard","administration_route","duration","basal_diet","positive_control","evidence_text","source_location"],
    "Control_Group": ["entity_id","experiment_id","group_name","group_type","description","evidence_text","source_location"],
    "Tissue_Site": ["entity_id","doi","site_name","site_category","evidence_text","source_location"],
    "Indicator": ["entity_id","doi","standard_name","abbreviation","unit","indicator_category","measurement_method","measured_in","evidence_text","source_location"],
    "Result": ["entity_id","doi","experiment_id","direction","p_value","p_value_original_text","corrected_significance","significance_level","effect_size","time_point","subgroup","evidence_text","source_location","matched_indicator","matched_tissue","compared_to_group","relation_type"],
    "Method": ["entity_id","doi","method_name","description","evidence_text","source_location"],
}

RELATIONSHIP_COLUMNS = ["rel_type","head_entity_type","head_entity_id","tail_entity_type","tail_entity_id","evidence_text","source_location"]


def write_entities_tsv(entities, entity_type: str, output_dir: Optional[Path] = None) -> Path:
    """Write entity list to a TSV file."""
    output_dir = output_dir or OUTPUT_TSV_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{entity_type.lower()}.tsv"
    columns = ENTITY_COLUMNS.get(entity_type, [])
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for e in entities:
            d = asdict(e) if hasattr(e, '__dataclass_fields__') else e
            writer.writerow({k: d.get(k, "") for k in columns})
    return out_path


def write_relationships_tsv(relationships, output_dir: Optional[Path] = None) -> Path:
    """Write relationships to a TSV."""
    output_dir = output_dir or OUTPUT_TSV_DIR
    out_path = output_dir / "relationships.tsv"
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RELATIONSHIP_COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for r in relationships:
            d = asdict(r) if hasattr(r, '__dataclass_fields__') else r
            writer.writerow({k: d.get(k, "") for k in RELATIONSHIP_COLUMNS})
    return out_path
