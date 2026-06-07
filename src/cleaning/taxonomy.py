"""Domain-aware name casing — generic, pattern-driven rules for swine nutrition.

Design
------
Instead of a brittle hand-maintained dictionary mapping every variant to its
canonical form, this module uses a small set of **declarative case rules**
based on semantic categories and name patterns.  Each rule has:

- A **pattern** (regex) or **classifier** (category-based) that tells whether a
  name belongs to a category.
- A **transform** function or target casing, applied when the existing case
  deviates.

The rules are applied in priority order.  Higher-priority rules (preserve
symbols, acronyms) fire first; lower-priority generic rules (Title Case
fallback) fire last.

Additionally, a small **override map** handles ambiguous or idiosyncratic cases
that pattern rules cannot disambiguate (e.g. "ADG" which looks like a gene but
is a production metric abbreviation).

Categories
----------
Each name is first classified by matching against semantic patterns:

====================  ===========================================  ============
Category              Pattern / Classifier                         Target Case
====================  ===========================================  ============
gene_symbol           ALL-CAPS with digits, 3-8 chars               preserve
acronym               ALL-CAPS, 2-6 letters, known abbreviations     preserve
binomial_abbr         ``X. word`` (e.g. C. butyricum)               preserve
microorganism         2-word, second word lowercase in taxonomy      Genus species
organic_acid          ends with "acid", NOT amino acid               lowercase
amino_acid            in known amino acid list                       Title Case
mineral               single-word mineral name (e.g. Zinc)           Title Case
vitamin               "vitamin X" or specific form name              "Vitamin X"
enzyme                ends with "-ase"                               lowercase
cytokine              "interleukin-*", "tumor necrosis factor-*"     lowercase
anatomical_site       single-word organ/tissue name                  Title Case
physiological_marker  multi-word common assay name                   lowercase
====================  ===========================================  ============

Everything else falls through to the generic **Title Case with stop words**
transformation.
"""

from __future__ import annotations

import re
from typing import Callable


# ============================================================================
# 1. PRESERVE-AS-IS patterns — always keep original case
# ============================================================================

_PRESERVE_PATTERNS: list[str] = [
    # Gene / protein symbols: TLR4, SOD1, NRF2, CYP7A1, CPT1A
    r"^[A-Z][A-Z0-9]{1,7}\d*$",
    # Lowercase-prefixed: p38, p65, p53
    r"^[a-z]{1,3}\d{1,3}[A-Za-z]?$",
    # Numbers with units/suffixes: 25(OH)D3, 8-OHdG
    r"^\d+\([A-Za-z]+\)[A-Za-z0-9]*$",
    # Greek-letter prefixed: α-tocopherol, β-glucan (already normalized NFC)
    r"^[α-ωΑ-Ω][\w\-].*$",
]


# ============================================================================
# 2. KNOWN ACRONYM OVERRIDES — short names that look ambiguous
# ============================================================================

# All-caps names < 5 chars that are NOT gene symbols but industry acronyms.
# These map to themselves (preserve uppercase).
_KNOWN_ACRONYMS: set[str] = {
    "ADG", "ADFI", "FCR", "LPS", "BW", "FBW", "NBW", "IUGR",
    "ROS", "MDA", "TNF", "IFN", "IGA", "IGG", "IGM",
    "ALP", "ALT", "AST", "CK", "LDH", "BUN", "TC", "TG",
    "HDL", "LDL", "VLDL", "NEFA", "SOD", "CAT", "GPX",
    "T-AOC", "GSH", "MDA", "PCV", "RBC", "WBC", "HB",
    "CFU", "MIC", "PBS", "PCR", "ELISA", "HPLC", "NMR",
    "SD", "SE", "SEM", "SD", "CI", "OR", "RR", "HR",
    "DM", "CP", "EE", "CF", "NDF", "ADF", "NSP",
    "VFA", "SCFA", "BCFA", "BA", "PUFA", "MUFA", "SFA",
    "DHA", "EPA", "ARA", "CLA",
}


# ============================================================================
# 3. SEMANTIC CATEGORY CLASSIFIERS
# ============================================================================

# --- Amino acids (full names only, not 3-letter codes) ---
_AMINO_ACIDS_LOWER: set[str] = {
    "arginine", "lysine", "methionine", "threonine", "tryptophan",
    "glutamine", "leucine", "valine", "isoleucine", "glycine",
    "alanine", "proline", "serine", "cysteine", "tyrosine",
    "phenylalanine", "histidine", "aspartic acid", "glutamic acid",
    "asparagine", "glutamate", "aspartate", "taurine", "citrulline",
    "ornithine", "homocysteine", "cystine",
}

# --- Minerals / trace elements (single words) ---
# Also includes common mineral compounds like "zinc oxide", "copper sulfate"
_MINERALS_LOWER: set[str] = {
    "selenium", "chromium", "iron", "zinc", "copper", "calcium",
    "phosphorus", "magnesium", "potassium", "sodium", "manganese",
    "iodine", "cobalt", "molybdenum", "sulfur", "boron",
    "silicon", "vanadium", "nickel", "tin",
}

# Mineral compounds (two words) that conventionally stay lowercase
_MINERAL_COMPOUNDS_LOWER: set[str] = {
    "zinc oxide", "copper sulfate", "zinc sulfate", "ferrous sulfate",
    "ferric oxide", "calcium carbonate", "calcium phosphate",
    "magnesium oxide", "magnesium sulfate", "sodium chloride",
    "sodium bicarbonate", "potassium chloride", "potassium iodide",
    "sodium selenite", "zinc chloride", "zinc acetate",
    "copper chloride", "copper oxide", "manganese oxide",
    "manganese sulfate", "chromium picolinate", "zinc methionine",
    "zinc glycinate", "iron oxide", "iron sulfate",
    "nano zinc oxide", "nano-zno",
}

# --- Anatomical sites / organs ---
_ANATOMICAL_SITES_LOWER: set[str] = {
    "liver", "spleen", "kidney", "heart", "lung", "duodenum",
    "jejunum", "ileum", "colon", "cecum", "stomach", "pancreas",
    "hypothalamus", "pituitary", "adrenal", "thyroid", "thymus",
    "serum", "plasma", "blood", "feces", "urine", "saliva",
    "muscle", "adipose", "skin", "brain", "intestine", "rectum",
    "cerebellum", "cortex", "medulla", "ovary", "testis", "uterus",
    "mammary", "lymph", "bone", "cartilage", "tendon", "ligament",
    "aorta", "artery", "vein", "capillary",
}

# --- Common physiological / production metrics (lowercase convention) ---
_PHYSIOLOGICAL_LOWER: set[str] = {
    "body weight", "average daily gain", "average daily feed intake",
    "feed conversion ratio", "diarrhea rate", "diarrhea score",
    "survival rate", "total protein", "crude protein",
}


# ============================================================================
# 4. CATEGORY → TARGET CASE RULES (classifier functions)
# ============================================================================


def _classify_name(name_lower: str, name_original: str) -> str:
    """Classify a name into a semantic category for case normalization.

    Returns one of:
      "preserve"  — skip, keep as-is
      "lowercase" — force all lowercase
      "title"     — Title Case with stop words
      "binomial"  — Genus species (first word Title, second lowercase)
    """
    # --- Preserve checks (no transformation) ---
    if _is_preserved(name_original):
        return "preserve"

    # --- Acronym check ---
    if name_original.upper() in _KNOWN_ACRONYMS and len(name_original) <= 6:
        return "preserve"

    # --- Acid convention: organic/mineral acids → lowercase ---
    if _is_acid(name_lower):
        return "lowercase"

    # --- Enzyme convention: -ase or -zyme suffix → lowercase ---
    if (name_lower.endswith("ase") or name_lower.endswith("zyme")) and len(name_lower) > 5:
        return "lowercase"

    # --- Binomial abbreviation pattern: X. word → preserve ---
    if re.match(r"^[A-Za-z]\. [a-z]", name_original):
        return "preserve"

    # --- Cytokine/immune marker → lowercase ---
    if _is_cytokine(name_lower):
        return "lowercase"

    # --- Physiological marker → lowercase ---
    if name_lower in _PHYSIOLOGICAL_LOWER:
        return "lowercase"

    # --- Mineral compounds → lowercase ---
    if name_lower in _MINERAL_COMPOUNDS_LOWER:
        return "lowercase"

    # --- Amino acids → Title Case ---
    if name_lower in _AMINO_ACIDS_LOWER:
        return "title"

    # --- Minerals → Title Case ---
    if name_lower in _MINERALS_LOWER:
        return "title"

    # --- Anatomical sites → Title Case ---
    if name_lower in _ANATOMICAL_SITES_LOWER:
        return "title"

    # --- Vitamin naming ---
    if _is_vitamin(name_lower):
        return "vitamin"

    # --- Microorganism binomial naming ---
    if _is_microorganism(name_lower):
        return "binomial"

    # --- Generic: multi-word all-lowercase → Title Case ---
    if " " in name_lower and name_original.islower():
        return "title"

    # --- Generic: single word all-lowercase → Title Case (capitalize first) ---
    if " " not in name_lower and name_original.islower() and len(name_original) > 2:
        # But NOT if it is a mineral (already checked) or looks like a verb
        return "title"

    # Everything else stays as-is
    return "preserve"


def _is_preserved(name: str) -> bool:
    """Check if name matches any preserve-as-is pattern."""
    for pat in _PRESERVE_PATTERNS:
        if re.match(pat, name):
            return True
    return False


def _is_acid(name_lower: str) -> bool:
    """Check if name is an organic/mineral acid (not amino acid, not nucleic acid)."""
    if not name_lower.endswith(" acid"):
        return False
    # Exclude amino acids
    if name_lower in _AMINO_ACIDS_LOWER:
        return False
    # Exclude nucleic acids
    if "nucleic" in name_lower:
        return False
    # Exclude fatty acid (handled separately)
    if "fatty" in name_lower:
        return False
    return True


def _is_cytokine(name_lower: str) -> bool:
    """Check if name is a cytokine/immune marker."""
    cytokine_prefixes = (
        "interleukin", "interferon", "immunoglobulin",
        "tumor necrosis factor", "transforming growth factor",
        "colony stimulating factor", "chemokine",
    )
    for prefix in cytokine_prefixes:
        if name_lower.startswith(prefix):
            return True
    return False


def _is_microorganism(name_lower: str) -> bool:
    """Check if name follows binomial nomenclature pattern (Genus species)."""
    # Two words, second word is all-lowercase or contains strain designation
    words = name_lower.split()
    if len(words) != 2:
        # Sometimes 3+ words (e.g. "porcine reproductive and respiratory syndrome virus")
        return False
    # Genus should start with uppercase (if already mixed) or be lowercase (to fix)
    # Species should be lowercase
    # Common genera in swine nutrition
    common_genera = {
        "bacillus", "lactobacillus", "clostridium", "escherichia",
        "salmonella", "staphylococcus", "streptococcus", "enterococcus",
        "bifidobacterium", "saccharomyces", "pediococcus", "prevotella",
        "bacteroides", "faecalibacterium", "campylobacter", "lawsonia",
        "mycobacterium", "mycoplasma", "brucella", "pasteurella",
        "haemophilus", "actinobacillus", "fusobacterium", "ruminococcus",
        "aspergillus", "penicillium", "fusarium", "candida",
        "pseudomonas", "klebsiella", "proteus", "shigella",
        "porcine", "avian", "bovine", "swine", "chicken",
    }
    # Check if first word looks like a genus name (capitalized or in known list)
    first = words[0]
    if first in common_genera or first.rstrip(".") in common_genera:
        return True
    # Check pattern: first word capitalized (or should be), second word all lowercase
    if first[0].isupper() and words[1].islower():
        return True
    # Both lowercase → probably a binomial that needs fixing
    if first.islower() and words[1].islower():
        return True
    return False


def _is_vitamin(name_lower: str) -> bool:
    """Check if name is a vitamin."""
    if name_lower.startswith("vitamin "):
        return True
    # Specific vitamin forms
    vitamin_names = {
        "alpha-tocopherol", "beta-carotene", "retinol", "retinoic acid",
        "ascorbic acid", "cholecalciferol", "ergocalciferol",
        "thiamine", "riboflavin", "niacin", "pyridoxine",
        "cobalamin", "biotin", "folate", "folic acid",
        "menadione", "phylloquinone",
    }
    if name_lower in vitamin_names:
        return True
    return False


# ============================================================================
# 5. TRANSFORM FUNCTIONS
# ============================================================================

_STOP_WORDS: frozenset[str] = frozenset({
    "of", "in", "and", "the", "to", "for", "with", "from",
    "by", "or", "at", "on", "as", "per", "via",
})


def _apply_title_case(text: str) -> str:
    """Title Case each word, preserving stop words in lower case."""
    words = text.split()
    result = []
    for i, w in enumerate(words):
        if i > 0 and w.lower() in _STOP_WORDS:
            result.append(w.lower())
        elif w.isupper() and len(w) <= 3 and i > 0:
            # Small all-caps fragments inside a longer name: keep as-is
            # e.g. "fatty acid" not "Fatty Acid"
            result.append(w)
        else:
            result.append(w[0].upper() + w[1:].lower() if len(w) > 1 else w.upper())
    return " ".join(result)


def _apply_binomial_case(text: str) -> str:
    """Genus species: first word Title Case, second word lowercase."""
    words = text.split()
    if len(words) >= 2:
        genus = words[0][0].upper() + words[0][1:].lower()
        species = words[1].lower()
        rest = words[2:]
        return " ".join([genus, species] + rest)
    return text


def _apply_vitamin_case(text: str) -> str:
    """Vitamin naming convention: 'vitamin X' or specific form name."""
    lower = text.lower()
    # Greek-letter prefix forms
    known = {
        "alpha-tocopherol": "α-tocopherol",
        "alpha-tocopheryl": "α-tocopheryl",
        "beta-carotene": "β-carotene",
        "gamma-tocopherol": "γ-tocopherol",
    }
    if lower in known:
        return known[lower]
    if lower.startswith("vitamin "):
        letter = lower[len("vitamin "):].strip().upper()
        return f"Vitamin {letter}"
    return _apply_title_case(text)


# ============================================================================
# 6. MAIN ENTRY POINT
# ============================================================================


def apply_case_rule(name: str, entity_type: str = "") -> str:
    """Apply domain-aware case normalization to *name*.

    Returns the name with semantically appropriate casing, or the original
    name unchanged if no rule applies.
    """
    if not name or not isinstance(name, str):
        return name

    original = name.strip()
    if not original:
        return original

    name_lower = original.lower()

    # Determine category and apply transformation
    category = _classify_name(name_lower, original)

    if category == "preserve":
        return original

    if category == "lowercase":
        return name_lower

    if category == "title":
        return _apply_title_case(original)

    if category == "binomial":
        return _apply_binomial_case(original)

    if category == "vitamin":
        return _apply_vitamin_case(original)

    return original


# ============================================================================
# 7. SMALL EXPLICIT OVERRIDES — only for truly ambiguous cases
# ============================================================================

# Entries here ONLY for names that the pattern rules get wrong because of
# genuine ambiguity (e.g. "CAT" could be catalase enzyme or CAT gene, or
# "ADG" looks like a gene symbol but is a production metric).
#
# This should stay small — all systematic cases belong in the classifiers above.
_CASE_OVERRIDES: dict[str, str] = {
    # None currently needed — the classifiers handle everything generically.
    # Add entries ONLY when the automated rules produce wrong results.
}

# ---------------------------------------------------------------------------
# Convenience function — mirrors old get_canonical_name() interface
# ---------------------------------------------------------------------------


def get_canonical_name(name: str) -> str:
    """Return the canonical-cased form of *name*."""
    key = name.strip().lower()
    if key in _CASE_OVERRIDES:
        return _CASE_OVERRIDES[key]
    return apply_case_rule(name)


# ============================================================================
# 8. DOMAIN-SPECIFIC NORMALIZATION MAPS (control groups, methods, direction)
# ============================================================================
# These maps handle VALUE normalization (not case).  They map non-standard
# enum values to canonical forms — e.g. "CON" → "Control group",
# "improved" → "increased".  These are inherently domain-specific and must
# be maintained, but they are small and stable.

CONTROL_GROUP_NORMALIZATION: dict[str, str] = {
    "con": "Control group",
    "con.": "Control group",
    "ctr": "Control group",
    "ctrl": "Control group",
    "control": "Control group",
    "control group": "Control group",
    "control diet": "Basal diet",
    "nc": "Negative control",
    "n.c.": "Negative control",
    "pc": "Positive control",
    "p.c.": "Positive control",
    "ab": "Antibiotic group",
    "abx": "Antibiotic group",
    "anti": "Antibiotic group",
    "antibiotic": "Antibiotic group",
    "antibiotic group": "Antibiotic group",
    "pbs": "PBS",
    "saline": "Saline",
}

METHOD_NAME_NORMALIZATION: dict[str, str] = {
    "elisa": "ELISA",
    "elisa assay": "ELISA assay",
    "elisa kit": "ELISA kit",
    "elisa test": "ELISA test",
    "pcr": "PCR",
    "rt-pcr": "RT-PCR",
    "qpcr": "qPCR",
    "rt-qpcr": "RT-qPCR",
    "western blot": "Western blot",
    "western blotting": "Western blotting",
    "sds-page": "SDS-PAGE",
    "nmr": "NMR",
    "gc-ms": "GC-MS",
    "lc-ms": "LC-MS",
    "hplc": "HPLC",
    "icp-oes": "ICP-OES",
    "icp-ms": "ICP-MS",
}

DIRECTION_MAPPING: dict[str, str] = {
    "altered": "no_significant_change",
    "improved": "increased",
    "downregulated": "decreased",
    "changed": "no_significant_change",
    "affected": "no_significant_change",
    "regulated": "no_significant_change",
    "promoted": "increased",
    "reduced": "decreased",
    "alleviated": "decreased",
    "modified": "no_significant_change",
    "enhanced": "increased",
    "elevated": "increased",
    "up-regulated": "increased",
    "down-regulated": "decreased",
    "lowered": "decreased",
}

RELATION_TYPE_MAPPING: dict[str, str] = {
    "enriches": "increases",
    "depletes": "decreases",
    "enhances": "upregulates",
    "suppresses": "downregulates",
    "reduces": "decreases",
    "raises": "increases",
    "lowers": "decreases",
    "boosts": "increases",
    "inhibits": "decreases",
    "stimulates": "upregulates",
    "induces": "upregulates",
    "represses": "downregulates",
    "activates": "upregulates",
    "blocks": "decreases",
    "attenuates": "decreases",
    "modulates": "affects",
    "regulates": "affects",
    "alters": "affects",
    "impairs": "decreases",
    "improves": "increases",
    "ameliorates": "increases",
    "augments": "increases",
    "diminishes": "decreases",
}


def get_canonical_control_group(name: str) -> str:
    """Normalize a control group name to its canonical form."""
    key = name.strip().lower()
    return CONTROL_GROUP_NORMALIZATION.get(key, name)


def get_canonical_direction(direction: str) -> str:
    """Map a non-standard direction value to canonical form."""
    key = direction.strip().lower()
    return DIRECTION_MAPPING.get(key, direction)


def get_canonical_relation_type(rel_type: str) -> str:
    """Map a non-standard relation type to canonical form."""
    key = rel_type.strip().lower()
    return RELATION_TYPE_MAPPING.get(key, rel_type)
