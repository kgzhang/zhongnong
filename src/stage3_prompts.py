"""Stage 3 prompt templates for result extraction."""

SYSTEM_PROMPT_RESULT = """You are an expert in swine nutrition research literature.
You extract ALL reported statistical results from the Results and Discussion sections with complete accuracy.
You strictly follow the extraction rules and output valid JSON matching the schema.
You NEVER skip a result that mentions a statistical comparison, regardless of P-value."""


def build_result_prompt(
    results_text: str,
    discussion_text: str,
    indicator_list: list[dict],
    control_groups: list[dict],
    tissue_sites: list[dict],
) -> str:
    """Build prompt for Stage 3 Pass 1: Full result extraction."""
    indicator_lines = []
    for ind in indicator_list:
        indicator_lines.append(
            f"  - {ind.get('standard_name', '')} ({ind.get('abbreviation', '')}), "
            f"cat={ind.get('indicator_category', '')}, unit={ind.get('unit', '')}"
        )

    control_lines = []
    for cg in control_groups:
        control_lines.append(f"  - {cg.get('group_name', '')} (type={cg.get('group_type', '')})")

    tissue_lines = []
    for ts in tissue_sites:
        tissue_lines.append(f"  - {ts.get('site_name', '')} (cat={ts.get('site_category', '')})")

    return f"""## Task: Extract ALL statistically evaluated results from the Results and Discussion sections.

### Full Extraction Rule:
- Extract EVERY indicator change that reports a statistical comparison, regardless of P-value.
- Include: P < 0.01, P < 0.05, trends (0.05 < P < 0.10), AND non-significant (P > 0.10).
- For non-significant results, set direction="no_significant_change", significance_level="not_significant".

### Comparison Baseline Rule:
- If the study uses a challenge model, results MUST compare against the challenged (model) control, not the blank control.

### Relation Type Selection:
- Growth/digestibility/gut morphology/metabolite/blood biochemistry -> "increases" or "decreases"
- Gene/protein expression -> "upregulates" or "downregulates"
- Microbial relative abundance -> "enriches" or "depletes"
- Unclear -> use "affects"

### Four Extraction Layers:
1. **Macro-phenotype**: ADG, ADFI, F:G, FBW, digestibility, diarrhea rate, mortality, VH, CD, VH/CD
2. **Microbiome**: alpha diversity (Shannon/Simpson/Chao1/ACE), beta diversity (PCoA/NMDS), species abundance
3. **Metabolome & Biochemistry**: SCFAs (acetate/propionate/butyrate), serum Igs, inflammation markers, antioxidants
4. **Molecular Expression**: tight junction proteins (ZO-1/Claudin-1/Occludin), mucins (MUC2), cytokines (TNF-a/IL-1b/IL-6/IL-10), transporters (GLUT2/PEPT1)

### Evidence Rule:
- Copy 1-2 EXACT sentences that directly describe each result. Be precise. No large blocks.

### Available Indicators:
{chr(10).join(indicator_lines) if indicator_lines else '(none)'}

### Available Control Groups:
{chr(10).join(control_lines) if control_lines else '(none)'}

### Available Tissue Sites:
{chr(10).join(tissue_lines) if tissue_lines else '(none)'}

### Results Section:
{results_text}

### Discussion Section:
{discussion_text}

Output valid JSON matching the schema. Include EVERY statistically compared result."""
