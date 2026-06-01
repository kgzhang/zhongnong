"""DSPy-based extraction modules for Stages 2-3.

Each module uses a dspy.Signature with strictly typed output fields,
eliminating the need for post-hoc field normalization and schema repair.
"""
import json
import logging
from typing import Optional

import dspy

from src.config import settings
from src.glossary import GlossaryIndex

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LM configuration — DSPy wraps litellm
# ---------------------------------------------------------------------------
_glossary: Optional[GlossaryIndex] = None


def _get_glossary() -> GlossaryIndex:
    global _glossary
    if _glossary is None:
        _glossary = GlossaryIndex()
        _glossary.load(str(settings.alternative_tsv))
    return _glossary


def _glossary_summary() -> str:
    """Compact glossary summary for prompts."""
    gi = _get_glossary()
    by_class: dict[str, list[str]] = {}
    for name, info in gi._exact.items():
        cls = info["class"]
        by_class.setdefault(cls, []).append(info["standard_name"])
    lines = []
    for cls, names in sorted(by_class.items()):
        lines.append(f"  {cls}: {', '.join(names[:25])}")
    return "\n".join(lines)


def _configure_lm():
    """Configure DSPy LM from settings. Call once at module load."""
    model = settings.llm_model
    if not model.startswith("deepseek/") and "/" not in model:
        model = f"deepseek/{model}"
    lm = dspy.LM(
        model,
        api_key=settings.deepseek_api_key,
        max_tokens=16384,
        temperature=0.1,
    )
    dspy.configure(lm=lm)
    return lm


# ---------------------------------------------------------------------------
# Module C: Alternative extraction
# ---------------------------------------------------------------------------

class ExtractAlternatives(dspy.Signature):
    """Extract all antibiotic alternative substances and composite products from the Materials and Methods section of a swine nutrition paper.

For each substance:
- Match against the provided glossary of standard classifications.
- If found, use the glossary's class and subclass.
- If NOT found AND it's a single substance: mark as 'Other'.
- If it's a mixture of multiple substances: create a Composite_Product.

CRITICAL: NEVER concatenate multiple names with '+' as a single entity.
For composites: decompose into individual components, then create a separate Composite_Product with its own product_name.

Always include evidence_text: 1-3 exact sentences from the source.
"""
    methods_text: str = dspy.InputField(desc="Full text of the Materials and Methods section")
    glossary: str = dspy.InputField(desc="Standard substance classification glossary")

    alternatives_json: str = dspy.OutputField(
        desc="JSON array of alternative objects. Each object has keys: standard_name (string), "
             "alternative_class (one of: Plant_Extract, Trace_Element, Organic_Acid, Probiotic, "
             "Polysaccharides_and_Oligosaccharides, Enzyme, Bioactive_Peptides, Other), "
             "subclass (string or null), match_source (string: 词表精确匹配/词表同义映射/Other_未匹配), "
             "evidence_text (1-3 verbatim sentences), source_location (e.g. Methods 2.3)"
    )
    composites_json: str = dspy.OutputField(
        desc="JSON array of composite product objects. Each object has keys: product_name (string), "
             "manufacturer (string or null), is_commercial (boolean), "
             "components (JSON array of {standard_name, entity_type}), "
             "evidence_text, source_location. "
             "Return [] if no composites found."
    )


# ---------------------------------------------------------------------------
# Module A: Experiment design extraction
# ---------------------------------------------------------------------------

class ExtractExperimentDesign(dspy.Signature):
    """Extract experiment design entities from the Materials and Methods section.

Swine_Model: Determine if challenge/stress model was used. If yes, extract stressor details.
If no challenge, model_type='normal'.

Swine: Extract breed, sex, age, physiological stage, initial body weight, sample size for EACH
group of animals in the study. Return as JSON array (one object per animal group).

Intervention: For each treatment group, extract the substance name, dose (exact numeric value
and original unit), administration route, duration, basal diet, and positive control if any.
Standardize dose units: 1 ppm = 1 mg/kg (feed-based), 1% = 10000 mg/kg.

Control_Group: Identify each control group with name, type (negative_control/positive_control/
basal_control/sham), and description.
"""
    methods_text: str = dspy.InputField(desc="Full text of the Materials and Methods section")

    swine_model_json: str = dspy.OutputField(
        desc="JSON object with keys: model_type ('challenge' or 'normal'), "
             "stressor_name (string or null), challenge_method (string or null), "
             "challenge_dose (string or null), challenge_timing (string or null), "
             "evidence_text, source_location"
    )
    swine_json: str = dspy.OutputField(
        desc="JSON array of animal group objects. Each has: breed (string), sex (string), "
             "age (string), physiological_stage (string), initial_body_weight (string), "
             "sample_size (integer or string), evidence_text, source_location"
    )
    interventions_json: str = dspy.OutputField(
        desc="JSON array of intervention objects. Each has: intervention_target (string-alternative name), "
             "dose_value (number), dose_unit_original (string), dose_unit_standard (string), "
             "administration_route (string: diet/drinking_water/oral_gavage/injection/topical/other), "
             "duration (string), basal_diet (string), positive_control (string or null), "
             "evidence_text, source_location"
    )
    control_groups_json: str = dspy.OutputField(
        desc="JSON array of control group objects. Each has: group_name (string), "
             "group_type (string: negative_control/positive_control/basal_control/sham), "
             "description (string), evidence_text, source_location"
    )


# ---------------------------------------------------------------------------
# Module B: Indicator extraction
# ---------------------------------------------------------------------------

class ExtractIndicators(dspy.Signature):
    """Extract all indicators, tissue sites, and methods from Materials and Methods.

Indicators (by category):
3.5.1 Growth: ADG (g/d), ADFI (g/d), F:G, FBW (kg)
3.5.2 Digestibility: ATTD/AID/SID × nutrient (DM, CP, GE, NDF, ADF)
3.5.3 Gut morphology: VH (um), CD (um), VH/CD per intestinal segment
3.5.4 Omics: microbiome (alpha/beta diversity, taxa abundance), metabolome (SCFAs), gene expression
3.5.5 Serum biochemistry: IgA, IgG, IgM, SOD, MDA, T-AOC, BUN, Glucose etc.
3.5.6 Other: diarrhea rate, mortality, organ indices

For EACH indicator, record: standard_name, abbreviation, unit, indicator_category
(macro_phenotype/microbiome/metabolome/molecular), measurement_method, measured_in (tissue site name).

Also extract all Tissue_Site entities (site_name, site_category: content/mucosa/serum/tissue/feces)
and Method entities (method_name, description).
"""
    methods_text: str = dspy.InputField(desc="Full text of the Materials and Methods section")

    tissue_sites_json: str = dspy.OutputField(
        desc="JSON array of tissue site objects. Each has: site_name, site_category, evidence_text, source_location"
    )
    indicators_json: str = dspy.OutputField(
        desc="JSON array of indicator objects. Each has: standard_name, abbreviation, unit, "
             "indicator_category (macro_phenotype/microbiome/metabolome/molecular), "
             "measurement_method, measured_in (tissue site name), evidence_text, source_location"
    )
    methods_json: str = dspy.OutputField(
        desc="JSON array of method objects. Each has: method_name, description, evidence_text, source_location"
    )


# ---------------------------------------------------------------------------
# Stage 3: Result extraction
# ---------------------------------------------------------------------------

class ExtractResults(dspy.Signature):
    """Extract ALL statistically evaluated results from Results and Discussion.

INCLUDE every indicator change that reports a statistical comparison, regardless of P-value.
Include: P<0.01, P<0.05, trends (0.05<P<0.10), AND non-significant (P>0.10).

For each result, record:
- indicator_abbreviation: the abbreviation from the Available Indicators list
- tissue_site: the site name from Available Tissue Sites
- direction: 'increased' or 'decreased' or 'no_significant_change'
- relation_type: 'increases'/'decreases' for most, 'upregulates'/'downregulates' for genes,
  'enriches'/'depletes' for microbes
- significance_level: 'p_less_0.01'/'p_less_0.05'/'trend_0.05_0.1'/'not_significant'
- p_value: numeric value or null
- p_value_original_text: as written in paper
- compared_to_group: which control group was compared against
- evidence_text: 1-2 EXACT sentences from the paper
- source_location: where in the paper (e.g. 'Results 3.1', 'Figure 2')
"""
    results_text: str = dspy.InputField(desc="Full text of the Results section")
    discussion_text: str = dspy.InputField(desc="Full text of the Discussion section")
    indicators_list: str = dspy.InputField(desc="List of available indicators with abbreviations")
    control_groups_list: str = dspy.InputField(desc="List of available control groups")
    tissue_sites_list: str = dspy.InputField(desc="List of available tissue sites")

    results_json: str = dspy.OutputField(
        desc="JSON array of result objects. Each has: indicator_abbreviation, tissue_site, "
             "direction, relation_type, significance_level, p_value, p_value_original_text, "
             "effect_size, time_point, compared_to_group, evidence_text, source_location"
    )


# ---------------------------------------------------------------------------
# Extraction functions (public API)
# ---------------------------------------------------------------------------

_predictor_cache: dict = {}


def _get_predictor(signature_cls):
    """Get or create a DSPy predictor (cached)."""
    cls_name = signature_cls.__name__
    if cls_name not in _predictor_cache:
        _predictor_cache[cls_name] = dspy.Predict(signature_cls)
    return _predictor_cache[cls_name]


def extract_alternatives(methods_text: str) -> dict:
    """Run Module C: extract alternatives and composites."""
    if not methods_text:
        return {"alternatives": [], "composite_products": []}

    # Truncate for context limits
    if len(methods_text) > 8000:
        cut = methods_text[:8000].rfind(". ")
        methods_text = methods_text[:cut + 1]

    predictor = _get_predictor(ExtractAlternatives)
    result = predictor(
        methods_text=methods_text,
        glossary=_glossary_summary(),
    )

    try:
        alternatives = json.loads(result.alternatives_json)
    except (json.JSONDecodeError, AttributeError):
        alternatives = []

    try:
        composites = json.loads(result.composites_json)
    except (json.JSONDecodeError, AttributeError):
        composites = []

    if not isinstance(alternatives, list):
        alternatives = []
    if not isinstance(composites, list):
        composites = []

    return {"alternatives": alternatives, "composite_products": composites}


def extract_experiment_design(methods_text: str) -> dict:
    """Run Module A: extract experiment design entities."""
    if not methods_text:
        return {"swine_model": {}, "swine": [], "interventions": [], "control_groups": []}

    if len(methods_text) > 8000:
        cut = methods_text[:8000].rfind(". ")
        methods_text = methods_text[:cut + 1]

    predictor = _get_predictor(ExtractExperimentDesign)
    result = predictor(methods_text=methods_text)

    def safe_json(val, default):
        if not val:
            return default
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return default

    return {
        "swine_model": safe_json(result.swine_model_json, {}),
        "swine": safe_json(result.swine_json, []),
        "interventions": safe_json(result.interventions_json, []),
        "control_groups": safe_json(result.control_groups_json, []),
    }


def extract_indicators(methods_text: str) -> dict:
    """Run Module B: extract indicators, tissue sites, and methods."""
    if not methods_text:
        return {"tissue_sites": [], "indicators": [], "methods": []}

    if len(methods_text) > 8000:
        cut = methods_text[:8000].rfind(". ")
        methods_text = methods_text[:cut + 1]

    predictor = _get_predictor(ExtractIndicators)
    result = predictor(methods_text=methods_text)

    def safe_json(val, default):
        if not val:
            return default
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return default

    return {
        "tissue_sites": safe_json(result.tissue_sites_json, []),
        "indicators": safe_json(result.indicators_json, []),
        "methods": safe_json(result.methods_json, []),
    }


def extract_results(
    results_text: str,
    discussion_text: str,
    indicators: list[dict],
    control_groups: list[dict],
    tissue_sites: list[dict],
) -> list[dict]:
    """Run Stage 3: extract all statistical results."""
    if not results_text:
        return []

    if len(results_text) > 6000:
        cut = results_text[:6000].rfind(". ")
        results_text = results_text[:cut + 1]
    if len(discussion_text) > 3000:
        cut = discussion_text[:3000].rfind(". ")
        discussion_text = discussion_text[:cut + 1]

    # Build compact context lists
    ind_list = "\n".join(
        f"  {i.get('abbreviation','?')} = {i.get('standard_name','?')} [{i.get('indicator_category','?')}]"
        for i in indicators[:40]
    )
    ctrl_list = "\n".join(
        f"  {c.get('group_name','?')} ({c.get('group_type','?')})"
        for c in control_groups
    )
    tissue_list = "\n".join(
        f"  {t.get('site_name','?')} [{t.get('site_category','?')}]"
        for t in tissue_sites
    )

    predictor = _get_predictor(ExtractResults)
    result = predictor(
        results_text=results_text,
        discussion_text=discussion_text or "",
        indicators_list=ind_list or "(none)",
        control_groups_list=ctrl_list or "(none)",
        tissue_sites_list=tissue_list or "(none)",
    )

    try:
        results = json.loads(result.results_json)
        if isinstance(results, list):
            return results
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return []


# Initialize LM on module load
_configure_lm()
