from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Alternative:
    entity_id: str
    standard_name: str
    abbreviation: Optional[str] = None
    cas_number: Optional[str] = None
    source_organism: Optional[str] = None
    is_synthetic: bool = False
    alternative_class: str = ""
    subclass: Optional[str] = None
    match_source: str = ""
    original_text: str = ""
    evidence_text: str = ""
    source_location: str = ""


@dataclass
class AlternativeClass:
    class_name: str
    level: int = 1
    description: str = ""


@dataclass
class CompositeProduct:
    entity_id: str
    product_name: str
    manufacturer: Optional[str] = None
    is_commercial: bool = False
    components: list = field(default_factory=list)
    original_text: str = ""
    evidence_text: str = ""
    source_location: str = ""


@dataclass
class Literature:
    doi: str
    pmid: Optional[str] = None
    title: str = ""
    journal: str = ""
    abstract_conclusion: str = ""
    publication_year: Optional[int] = None
    publication_date: str = ""
    study_design: str = ""


@dataclass
class Experiment:
    experiment_id: str
    doi: str
    description: str = ""
    evidence_text: str = ""
    source_location: str = ""


@dataclass
class SwineModel:
    entity_id: str
    experiment_id: str
    model_type: str = ""
    stressor_name: Optional[str] = None
    challenge_method: Optional[str] = None
    challenge_dose: Optional[str] = None
    challenge_timing: Optional[str] = None
    evidence_text: str = ""
    source_location: str = ""


@dataclass
class Swine:
    entity_id: str
    experiment_id: str
    breed: str = ""
    sex: str = ""
    age: str = ""
    physiological_stage: str = ""
    initial_body_weight: str = ""
    sample_size: Optional[int] = None
    evidence_text: str = ""
    source_location: str = ""


@dataclass
class Intervention:
    entity_id: str
    experiment_id: str
    intervention_target: str = ""
    dose_value: Optional[float] = None
    dose_unit_original: str = ""
    dose_unit_standard: str = ""
    administration_route: str = ""
    duration: str = ""
    basal_diet: str = ""
    positive_control: Optional[str] = None
    evidence_text: str = ""
    source_location: str = ""


@dataclass
class ControlGroup:
    entity_id: str
    experiment_id: str
    group_name: str = ""
    group_type: str = ""
    description: str = ""
    evidence_text: str = ""
    source_location: str = ""


@dataclass
class TissueSite:
    entity_id: str
    doi: str
    site_name: str = ""
    site_category: str = ""
    evidence_text: str = ""
    source_location: str = ""


@dataclass
class Indicator:
    entity_id: str
    doi: str
    standard_name: str = ""
    abbreviation: str = ""
    unit: str = ""
    indicator_category: str = ""
    measurement_method: str = ""
    measured_in: str = ""
    evidence_text: str = ""
    source_location: str = ""


@dataclass
class Result:
    entity_id: str
    doi: str
    experiment_id: str
    direction: str = ""
    p_value: Optional[float] = None
    p_value_original_text: str = ""
    corrected_significance: Optional[str] = None
    significance_level: str = ""
    effect_size: Optional[str] = None
    time_point: Optional[str] = None
    subgroup: Optional[str] = None
    evidence_text: str = ""
    source_location: str = ""
    matched_indicator: str = ""
    matched_tissue: str = ""
    compared_to_group: str = ""
    relation_type: str = ""


@dataclass
class Method:
    entity_id: str
    doi: str
    method_name: str = ""
    description: str = ""
    evidence_text: str = ""
    source_location: str = ""


@dataclass
class Relationship:
    rel_type: str
    head_entity_type: str
    head_entity_id: str
    tail_entity_type: str
    tail_entity_id: str
    evidence_text: str = ""
    source_location: str = ""
