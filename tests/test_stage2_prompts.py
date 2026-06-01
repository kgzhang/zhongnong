from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from stage2_prompts import build_module_c_prompt, build_module_a_prompt, build_module_b_prompt

METHODS_TEXT = """Dietary treatments included a basal diet supplemented with 500 mg/kg thymol (THY, purity >= 99%, Sigma-Aldrich).
A total of 72 weaned barrows (Duroc x Landrace x Yorkshire, 21 d of age, initial BW 7.5 +/- 0.2 kg) were randomly allocated to 4 treatments.
On day 7, piglets were orally challenged with 10 mL of E. coli K88 (1x10^9 CFU/mL)."""

def test_module_c_prompt_contains_key_rules():
    prompt = build_module_c_prompt(METHODS_TEXT)
    assert "Thymol" in prompt or "thymol" in prompt
    assert "Composite_Product" in prompt
    assert "Other" in prompt
    assert "字符串拼接" in prompt or "禁止" in prompt or "composite" in prompt.lower()

def test_module_a_prompt_contains_dose_rules():
    prompt = build_module_a_prompt(METHODS_TEXT)
    assert "mg/kg_feed" in prompt  # dose standardization
    assert "challenge" in prompt.lower()
    assert "72" in prompt  # the methods text is in the prompt

def test_module_b_prompt_contains_indicator_categories():
    prompt = build_module_b_prompt(METHODS_TEXT)
    assert "ADG" in prompt  # 3.5.1 growth performance
    assert "qPCR" in prompt or "ELISA" in prompt or "Western" in prompt  # 3.5.4 methods
    assert "indicator_category" in prompt.lower()

def test_all_prompts_contain_source_text():
    for build_fn in [build_module_c_prompt, build_module_a_prompt, build_module_b_prompt]:
        prompt = build_fn(METHODS_TEXT)
        assert "thymol" in prompt.lower()
