"""Stage 2 prompt templates for LLM entity extraction (Modules C, A, B)."""
from src.glossary import GlossaryIndex
from src.config import ALTERNATIVE_TSV


def _load_glossary_summary() -> str:
    """Build a compact summary of the glossary for inclusion in prompts."""
    gi = GlossaryIndex()
    gi.load(str(ALTERNATIVE_TSV))
    lines = []
    by_class = {}
    for name, info in gi._exact.items():
        cls = info["class"]
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(info["standard_name"])
    for cls, names in sorted(by_class.items()):
        sample = ", ".join(names[:30])
        lines.append(f"  {cls}: {sample}")
    return "\n".join(lines)


SYSTEM_PROMPT_ENTITY = """You are an expert in swine nutrition and antibiotic alternatives research literature.
You extract structured entities from the Materials and Methods sections of scientific papers with extreme precision.
All numeric values must be copied exactly as they appear. Never round or approximate.
Always copy evidence sentences verbatim from the source text. Never paraphrase evidence.
Output only valid JSON matching the schema."""


def build_module_c_prompt(methods_text: str) -> str:
    """Build prompt for Module C: Alternative identification and classification."""
    glossary_summary = _load_glossary_summary()

    return f"""## Task: Extract all antibiotic alternative substances from the Materials and Methods section.

### Standard Substance Classification (glossary):
{glossary_summary}

### CRITICAL RULES (BACKGROUND.md Section 3.1):

1. **Entity Mapping**: For each substance mentioned, first try to match against the glossary above.
   - If found: use the glossary's classification (Alternative_Class).
   - If NOT found (single substance): set alternative_class = "Other", entity_type = "Alternative".
   - "Other" is ONLY for single substances not in the standard classification.

2. **Composite Products (BACKGROUND 3.1, three-step rule)**:
   - Step 1: Decompose the composite into individual component entities. Align each component against the glossary.
   - Step 2: Create a Composite_Product entity for the mixture itself. NEVER concatenate component names with "+" as the node name.
   - Step 3: List components with has_component relationships.
   - Other is NOT Composite_Product: they are mutually exclusive.

3. **Forbidden**: DO NOT concatenate multiple substance names into a single string entity.
   - If the product has a commercial name, use it. If not, use a descriptive identifier.

4. **Evidence**: For each entity, copy the exact 1-3 sentences from the source text describing the substance and its use. Include the section/paragraph location.

### Source Text (Materials and Methods):
{methods_text}

Output valid JSON matching the schema exactly."""


def build_module_a_prompt(methods_text: str) -> str:
    """Build prompt for Module A: Experiment design entities (BACKGROUND 3.2-3.4)."""
    return f"""## Task: Extract experiment design entities from the Materials and Methods section.

### Extract the following:

#### Swine_Model (BACKGROUND 3.2):
- Determine if a challenge/stress model was used (pathogen, toxin, heat stress, etc.)
- If yes: extract stressor name, challenge method (oral_gavage/injection/other), dose, timing.
- If no challenge: model_type = "normal", stressor_name = null.
- Distinguish between blank control and challenged control groups.

#### Swine (BACKGROUND 3.3):
- Extract: breed, sex (boar/sow/barrow/male/female/mixed), age in days, physiological stage, initial body weight (with +/- value), sample size.

#### Intervention (BACKGROUND 3.4):
- Extract dose with HIGHEST PRECISION: exact numeric value and original unit.
- Standardize units: 1 ppm = 1 mg/kg, 1% = 10000 mg/kg.
  - Feed-based -> mg/kg_feed
  - Body-weight-based -> mg/kg_BW
- Multiple dose gradients -> one Intervention per dose level.
- Administration route: diet/drinking_water/oral_gavage/injection/topical/other.
- Duration: "XX days". Basal diet type. Positive control (if any).
- If dose is NOT reported -> mark as "Not reported".

#### Control_Group:
- Identify each control group: name, type (negative_control/positive_control/basal_control/sham), description.

### CRITICAL: For every field, copy the exact evidence sentence from the source text into evidence_text.
### If a field cannot be found in the text, mark it as "Not reported" for text fields or null for optional fields.
### Do NOT invent or infer values.

### Source Text (Materials and Methods):
{methods_text}

Output valid JSON matching the schema exactly."""


def build_module_b_prompt(methods_text: str) -> str:
    """Build prompt for Module B: Indicator and tissue site entities (BACKGROUND 3.5)."""
    return f"""## Task: Extract all indicators, tissue sites, and methods from the Materials and Methods section.

### Follow BACKGROUND.md Section 3.5 — extract by indicator category:

#### 3.5.1 Growth Performance:
ADG (g/d), ADFI (g/d), F:G (ratio), FBW (kg). Note measurement period.

#### 3.5.2 Digestibility:
Distinguish: ATTD (total tract apparent), AID (apparent ileal), SID (standardized ileal).
Link to nutrient: DM, CP, GE, EE, CF, NDF, ADF, amino acids.

#### 3.5.3 Gut Morphology:
Specify intestinal segment: duodenum/jejunum/ileum/colon/cecum.
Extract: VH (um), CD (um), VH/CD, mucosal thickness, goblet cell count.

#### 3.5.4 Omics:
- Microbiome: alpha diversity (Shannon/Simpson/Chao1/ACE), beta diversity (PCoA/NMDS), phylum/genus/species abundance. Sample site: cecal content/colonic content/ileal content/feces.
- Metabolome: SCFAs (acetate/propionate/butyrate), bile acids. Units (umol/g, mmol/L).
- Gene/protein expression: target tissue (jejunal mucosa/liver/spleen), target gene (ZO-1/Claudin-1/Occludin/TNF-a/IL-1b/IL-6/IL-10/GLUT2/PEPT1), method (qPCR/Western blot/ELISA).

#### 3.5.5 Serum Biochemistry:
IgA, IgG, IgM, LPS, Haptoglobin, CRP, T-SOD, GSH-Px, MDA, T-AOC, CAT, Glucose, BUN, TP, ALB, TG, TC.

#### 3.5.6 Other:
Diarrhea rate, mortality, organ indices, intestinal permeability (D-lactate, DAO, FITC-dextran).

### For EACH indicator, create one row with:
- standard_name (full name), abbreviation, unit
- indicator_category: macro_phenotype/microbiome/metabolome/molecular
- measured_in: the Tissue_Site name
- measurement_method description
- evidence_text: 1-3 verbatim sentences
- source_location: section/paragraph

### Also extract all Tissue_Site and Method entities separately.

### Source Text (Materials and Methods):
{methods_text}

Output valid JSON matching the schema exactly."""
