"""
Core Arabic normalization used by every other script in this pipeline.
Keep this identical across training, indexing, and inference - a mismatch
here silently breaks retrieval quality.
"""
import re

_ALEF_VARIANTS = "أإآا"
# Tatweel + Arabic diacritics (fatha, damma, kasra, shadda, sukun, etc.)
_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u0640]")


def normalize_arabic(text: str) -> str:
    """Unify common orthographic variation:
    - strip diacritics / tatweel
    - collapse alef variants (أ إ آ) -> ا
    - collapse yaa/alef maksura (ى) -> ي
    - collapse taa marbuta (ة) -> ه
    """
    if not text:
        return text
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
    """Normalize orthography, then collapse known brand/dialect synonyms
    down to one canonical token (e.g. بطاطا شيبس / ليز / سناكس -> شيبس)."""
    text = normalize_arabic(text)
    for canonical, variants in synonyms.items():
        canonical_norm = normalize_arabic(canonical)
        for v in variants:
            v_norm = normalize_arabic(v)
            if v_norm and v_norm in text:
                text = text.replace(v_norm, canonical_norm)
    return text
