"""DSPy-based extraction modules for Stages 2-3.

Key design:
- Module C (Alternatives) is the GATE: if no standard Alternative is found in an
  article, the article is skipped entirely.
- Full glossary passed to LLM for accurate matching.
- Swine_Model removed (not needed).
"""
import json
import logging
import re
from typing import Optional

import dspy

from src.config import settings
from src.glossary import GlossaryIndex

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LM configuration
# ---------------------------------------------------------------------------
_glossary: Optional[GlossaryIndex] = None


def _get_glossary() -> GlossaryIndex:
    global _glossary
    if _glossary is None:
        _glossary = GlossaryIndex()
        _glossary.load(str(settings.alternative_tsv))
    return _glossary


def _full_glossary_text() -> str:
    """Build a COMPLETE glossary listing for the Alternative extraction prompt.

    Returns the full substance list organized by class, NOT truncated.
    This is critical for accurate matching.
    """
    gi = _get_glossary()
    by_class: dict[str, list[dict]] = {}
    for name, info in gi._exact.items():
        cls = info["class"]
        by_class.setdefault(cls, []).append({
            "name": info["standard_name"],
            "subclass": info.get("subclass", ""),
        })

    lines = []
    class_descriptions = {
        "Plant_Extract": "植物提取物 — plant extracts (essential oils, phenols, flavonoids, alkaloids, saponins)",
        "Trace_Element": "微量元素 — trace elements (Zn, Cu, Fe, Mn, Se, Cr, Co, I, Mo — inorganic and organic forms)",
        "Organic_Acid": "有机酸 — organic acids (SCFA, MCFA, carboxylic acids used as feed acidifiers)",
        "Probiotic": "益生菌 — probiotics (Lactobacillus, Bifidobacterium, Enterococcus, Pediococcus, Streptococcus, Bacillus, Saccharomyces, Clostridium, Propionibacteria)",
        "Polysaccharides_and_Oligosaccharides": "多糖与低聚糖 — polysaccharides & oligosaccharides (FOS, GOS, XOS, MOS, COS, beta-glucan, chitosan, plant/microbial/animal polysaccharides)",
        "Enzyme": "酶制剂 — enzymes (digestive: protease, amylase, lipase, cellulase; non-digestive: phytase, xylanase; functional: glucose oxidase, lysozyme, SOD, catalase)",
        "Bioactive_Peptides": "生物活性肽 — bioactive peptides (antimicrobial peptides, defensins, cathelicidins, regulatory peptides, glutathione)",
    }

    for cls_key in [
        "Plant_Extract", "Trace_Element", "Organic_Acid", "Probiotic",
        "Polysaccharides_and_Oligosaccharides", "Enzyme", "Bioactive_Peptides",
    ]:
        substances = by_class.get(cls_key, [])
        desc = class_descriptions.get(cls_key, cls_key)
        lines.append(f"\n## {cls_key} — {desc}")
        # Group by subclass
        by_sub = {}
        for s in substances:
            sub = s.get("subclass", "") or "(general)"
            by_sub.setdefault(sub, []).append(s["name"])
        for sub, names in sorted(by_sub.items()):
            names_str = ", ".join(names)
            lines.append(f"  [{sub}] {names_str}")

    return "\n".join(lines)


def _configure_lm():
    """Configure DSPy LM from settings."""
    model = settings.llm_model
    if "/" not in model:
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
# Module C: Alternative extraction (THE GATE)
# ---------------------------------------------------------------------------

class ExtractAlternatives(dspy.Signature):
    """You are an expert in swine nutrition and antibiotic alternatives. Your task is to scan the Materials and Methods section of a scientific paper and identify ALL substances that serve as antibiotic alternatives — substances being tested or used to replace antibiotics in pig production.

IMPORTANT: Look carefully at the ENTIRE Methods text. Substances may be described as:
- Feed additives, supplements, dietary treatments
- Substances added to the basal diet at specific doses
- Commercial products being tested
- Natural compounds, extracts, or fermentation products
- Probiotics, prebiotics, enzymes, organic acids, plant extracts
- Any substance administered to pigs with the intent to improve health/growth

CRITICAL RULES:
1. For EACH substance found, check against the FULL GLOSSARY below.
2. If the substance name or a close variant appears in the glossary → record the glossary classification.
3. If the substance is a SINGLE compound not in the glossary → mark as 'Other'.
4. If the substance is a MIXTURE of multiple compounds → create a Composite_Product AND list individual components.
5. NEVER concatenate names with '+'. Each component is a separate entity.
6. Include the exact 1-3 sentences from the source as evidence_text.
7. Include source_location (section/paragraph where the substance is described).
8. If NOTHING in the article matches any glossary category AND no substance is being tested as an antibiotic alternative → return empty arrays.

FULL GLOSSARY OF STANDARD SUBSTANCES:
"""
    methods_text: str = dspy.InputField(desc="Full Materials and Methods text")
    alternatives_json: str = dspy.OutputField(
        desc="JSON array of alternative substances found. Each object: "
             "{standard_name, alternative_class (Plant_Extract/Trace_Element/Organic_Acid/"
             "Probiotic/Polysaccharides_and_Oligosaccharides/Enzyme/Bioactive_Peptides/Other), "
             "subclass (string|null), match_source (词表精确匹配/词表模糊匹配/Other_未匹配), "
             "evidence_text, source_location}. Return [] if none found."
    )
    composites_json: str = dspy.OutputField(
        desc="JSON array of composite products. Each: {product_name, manufacturer (string|null), "
             "is_commercial (bool), components: [{standard_name, entity_type}], "
             "evidence_text, source_location}. Return [] if none."
    )


# ---------------------------------------------------------------------------
# Module A: Experiment design (no Swine_Model)
# ---------------------------------------------------------------------------

class ExtractExperimentDesign(dspy.Signature):
    """Extract experiment design from Materials and Methods. Focus on animal parameters,
intervention details, and control groups. Do NOT extract disease models.

Swine: Extract breed, sex, age, physiological stage, initial body weight, sample size
for EACH group of animals. Return as JSON array.

Intervention: For each treatment group, extract substance name, exact dose value and
original unit, administration route (diet/drinking_water/oral_gavage/injection/topical/other),
duration, basal diet type, and positive control (antibiotic used, if any).
Standardize units: 1 ppm = 1 mg/kg (feed), 1% = 10000 mg/kg (feed), mg/kg BW stays as-is.

Control_Group: Each control group with name, type (negative_control/positive_control/
basal_control/sham), and description.
"""
    methods_text: str = dspy.InputField(desc="Full Materials and Methods text")
    swine_json: str = dspy.OutputField(
        desc="JSON array of animal groups. Each: {breed, sex, age, physiological_stage, "
             "initial_body_weight, sample_size (int|string), evidence_text, source_location}"
    )
    interventions_json: str = dspy.OutputField(
        desc="JSON array of interventions. Each: {intervention_target, dose_value, "
             "dose_unit_original, dose_unit_standard, administration_route, duration, "
             "basal_diet, positive_control, evidence_text, source_location}"
    )
    control_groups_json: str = dspy.OutputField(
        desc="JSON array of control groups. Each: {group_name, group_type, description, "
             "evidence_text, source_location}"
    )


# ---------------------------------------------------------------------------
# Module B: Indicator extraction
# ---------------------------------------------------------------------------

class ExtractIndicators(dspy.Signature):
    """Extract ALL indicators, tissue sites, and methods from Materials and Methods.

For each indicator, record: standard_name (full name), abbreviation, unit,
indicator_category (macro_phenotype/microbiome/metabolome/molecular),
measurement_method description, measured_in (which tissue site).

Indicator categories:
- macro_phenotype: ADG, ADFI, F:G, FBW, digestibility, VH, CD, VH/CD, diarrhea, mortality, organ indices
- microbiome: alpha diversity (Shannon/Simpson/Chao1/ACE), beta diversity (PCoA/NMDS), taxa abundance (phylum/genus/species)
- metabolome: SCFAs (acetate/propionate/butyrate etc.), bile acids, serum biochemistry (IgA/IgG/IgM, SOD, MDA, T-AOC, BUN, Glucose)
- molecular: gene/protein expression (ZO-1, Occludin, Claudin-1, TNF-a, IL-1b, IL-6, IL-10, MUC2, GLUT2, PEPT1)

Also extract Tissue_Site entities (site_name, site_category: content/mucosa/serum/tissue/feces)
and Method entities (method_name, description).
"""
    methods_text: str = dspy.InputField(desc="Full Materials and Methods text")
    tissue_sites_json: str = dspy.OutputField(
        desc="JSON array of tissue sites: {site_name, site_category, evidence_text, source_location}"
    )
    indicators_json: str = dspy.OutputField(
        desc="JSON array of indicators: {standard_name, abbreviation, unit, "
             "indicator_category, measurement_method, measured_in, evidence_text, source_location}"
    )
    methods_json: str = dspy.OutputField(
        desc="JSON array of methods: {method_name, description, evidence_text, source_location}"
    )


# ---------------------------------------------------------------------------
# Stage 3: Result extraction
# ---------------------------------------------------------------------------

class ExtractResults(dspy.Signature):
    """Extract ALL statistically evaluated results from Results and Discussion.

INCLUDE every result that reports a comparison — regardless of P-value (include P>0.10 too).
For non-significant results, direction='no_significant_change', significance_level='not_significant'.

For each result record:
- indicator_abbreviation: abbreviation from the available indicators list
- tissue_site: where measured
- direction: increased/decreased/no_significant_change
- relation_type: increases/decreases (numerical), upregulates/downregulates (genes), enriches/depletes (microbes), affects (unclear)
- significance_level: p_less_0.01/p_less_0.05/trend_0.05_0.1/not_significant
- p_value: numeric or null
- p_value_original_text: as written in paper
- compared_to_group: control group name
- evidence_text: 1-2 EXACT sentences from source
- source_location: where reported (e.g. 'Results 3.1', 'Table 2')
"""
    results_text: str = dspy.InputField(desc="Results section text")
    discussion_text: str = dspy.InputField(desc="Discussion section text")
    indicators_list: str = dspy.InputField(desc="Available indicators with abbreviations")
    control_groups_list: str = dspy.InputField(desc="Available control groups")
    tissue_sites_list: str = dspy.InputField(desc="Available tissue sites")

    results_json: str = dspy.OutputField(
        desc="JSON array of results: {indicator_abbreviation, tissue_site, direction, "
             "relation_type, significance_level, p_value, p_value_original_text, "
             "effect_size, time_point, compared_to_group, evidence_text, source_location}"
    )


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

_predictor_cache: dict = {}


def _get_predictor(signature_cls):
    cls_name = signature_cls.__name__
    if cls_name not in _predictor_cache:
        _predictor_cache[cls_name] = dspy.Predict(signature_cls)
    return _predictor_cache[cls_name]


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rfind(". ")
    return text[:cut + 1] if cut > max_chars // 2 else text[:max_chars]


def _safe_json(val, default):
    if not val:
        return default
    try:
        parsed = json.loads(val)
        return parsed if isinstance(parsed, type(default)) else default
    except (json.JSONDecodeError, TypeError):
        return default


def extract_alternatives(methods_text: str) -> dict:
    """Module C: Extract alternatives. Returns {alternatives, composite_products, has_known_alternative}."""
    if not methods_text.strip():
        return {"alternatives": [], "composite_products": [], "has_known_alternative": False}

    methods_text = _truncate(methods_text, 8000)
    glossary = _full_glossary_text()

    predictor = _get_predictor(ExtractAlternatives)
    result = predictor(methods_text=methods_text + "\n\n" + glossary)

    alternatives = _safe_json(result.alternatives_json, [])
    composites = _safe_json(result.composites_json, [])

    # Validate and enrich each alternative with glossary lookup
    gi = _get_glossary()
    for i, a in enumerate(alternatives):
        name = a.get("standard_name", "")
        if not name:
            continue
        # Try glossary match
        match = gi.lookup(name)
        if match and match.get("class") != "Other":
            a["alternative_class"] = match["class"]
            a["subclass"] = match.get("subclass")
            a["match_source"] = match.get("match_source", "词表精确匹配")
        elif match and match.get("class") == "Other":
            # Check fuzzy match
            fuzzy = _fuzzy_match_glossary(name, gi)
            if fuzzy:
                a["standard_name"] = fuzzy["standard_name"]
                a["alternative_class"] = fuzzy["class"]
                a["subclass"] = fuzzy.get("subclass")
                a["match_source"] = "词表模糊匹配"
            else:
                a["alternative_class"] = "Other"
                a["match_source"] = "Other_未匹配"
        else:
            a["alternative_class"] = "Other"
            a["match_source"] = "Other_未匹配"

        a.setdefault("evidence_text", "")
        a.setdefault("source_location", "")

    # Determine if article has at least one KNOWN alternative
    known_classes = {
        "Plant_Extract", "Trace_Element", "Organic_Acid", "Probiotic",
        "Polysaccharides_and_Oligosaccharides", "Enzyme", "Bioactive_Peptides",
    }
    has_known = any(
        a.get("alternative_class") in known_classes
        for a in alternatives
    )

    return {
        "alternatives": alternatives,
        "composite_products": composites,
        "has_known_alternative": has_known,
    }


def _fuzzy_match_glossary(name: str, gi: GlossaryIndex) -> Optional[dict]:
    """Try harder fuzzy matching — check substrings and normalized forms."""
    # Normalize: lowercase, remove special chars
    norm = re.sub(r'[^a-z0-9\s]', '', name.lower().strip())

    # Check if name CONTAINS a glossary entry or vice versa
    for gname, info in gi._exact.items():
        gnorm = re.sub(r'[^a-z0-9\s]', '', gname.lower().strip())
        if not gnorm or not norm:
            continue
        if gnorm in norm or norm in gnorm:
            return {
                "standard_name": info["standard_name"],
                "class": info["class"],
                "subclass": info.get("subclass"),
                "match_source": "词表模糊匹配",
            }
        # Check Levenshtein distance
        if gi.levenshtein(norm, gnorm) <= 3:
            return {
                "standard_name": info["standard_name"],
                "class": info["class"],
                "subclass": info.get("subclass"),
                "match_source": "词表模糊匹配",
            }

    return None


def extract_experiment_design(methods_text: str) -> dict:
    """Module A: Extract experiment design (no Swine_Model)."""
    if not methods_text.strip():
        return {"swine": [], "interventions": [], "control_groups": []}

    methods_text = _truncate(methods_text, 8000)
    predictor = _get_predictor(ExtractExperimentDesign)
    result = predictor(methods_text=methods_text)

    return {
        "swine": _safe_json(result.swine_json, []),
        "interventions": _safe_json(result.interventions_json, []),
        "control_groups": _safe_json(result.control_groups_json, []),
    }


def extract_indicators(methods_text: str) -> dict:
    """Module B: Extract indicators, tissue sites, methods."""
    if not methods_text.strip():
        return {"tissue_sites": [], "indicators": [], "methods": []}

    methods_text = _truncate(methods_text, 8000)
    predictor = _get_predictor(ExtractIndicators)
    result = predictor(methods_text=methods_text)

    return {
        "tissue_sites": _safe_json(result.tissue_sites_json, []),
        "indicators": _safe_json(result.indicators_json, []),
        "methods": _safe_json(result.methods_json, []),
    }


def extract_results(
    results_text: str,
    discussion_text: str,
    indicators: list[dict],
    control_groups: list[dict],
    tissue_sites: list[dict],
) -> list[dict]:
    """Stage 3: Extract all statistical results."""
    if not results_text.strip():
        return []

    results_text = _truncate(results_text, 6000)
    discussion_text = _truncate(discussion_text, 3000)

    ind_list = "\n".join(
        f"  {i.get('abbreviation','?')} = {i.get('standard_name','?')} [{i.get('indicator_category','?')}]"
        for i in indicators[:50]
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

    parsed = _safe_json(result.results_json, [])
    # Normalize values
    for r in parsed:
        r.setdefault("indicator_abbreviation", "")
        r.setdefault("tissue_site", "")
        r.setdefault("direction", "no_significant_change")
        r.setdefault("relation_type", "affects")
        r.setdefault("significance_level", "not_significant")
        r.setdefault("evidence_text", "")
        r.setdefault("source_location", "")
        r.setdefault("compared_to_group", "")
        r.setdefault("p_value", None)
    return parsed


# Initialize LM on module load
_configure_lm()
