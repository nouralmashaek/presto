"""
Core Arabic normalization used by every other script in this pipeline.
Keep this identical across training, indexing, and inference.
"""
import re

_ALEF_VARIANTS = "أإآا"
# Tatweel + Arabic diacritics (fatha, damma, kasra, shadda, sukun, etc.)
_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u0640]")

def normalize_arabic(text: str) -> str:
    """Unify common orthographic variation and clean leaked JSON syntax."""
    if not text or not isinstance(text, str):
        return ""
    
    # 1. Clean leaked JSON keys and syntax (e.g., 'attribute_discriminator": "' or '{"query": "')
    text = re.sub(r'("[a-zA-Z0-9_]+"|([a-zA-Z0-9_]+))\s*:\s*"?', ' ', text)
    text = re.sub(r'[\{\}"\'\[\]]', ' ', text)
    
    # 2. Strip diacritics / tatweel
    text = _DIACRITICS.sub("", text)
    
    # 3. Normalize letters
    text = re.sub(f"[{_ALEF_VARIANTS}]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    
    # 4. Collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_synonyms(path: str = "synonyms.json") -> dict:
    import json
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def apply_normalization(text: str, synonyms: dict) -> str:
    """Normalize orthography and replace known synonyms."""
    norm_text = normalize_arabic(text)
    for target, variants in synonyms.items():
        for var in variants:
            if var in norm_text:
                norm_text = norm_text.replace(var, target)
    return norm_text
