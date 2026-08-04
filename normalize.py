
import re

_ALEF_VARIANTS = "أإآا"
_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u0640]")

def normalize_arabic(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    
    text = re.sub(r'("[a-zA-Z0-9_]+"|([a-zA-Z0-9_]+))\s*:\s*"?', ' ', text)
    text = re.sub(r'[\{\}"\'\[\]]', ' ', text)
    
    text = _DIACRITICS.sub("", text)
    
    text = re.sub(f"[{_ALEF_VARIANTS}]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    
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
