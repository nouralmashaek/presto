"""
Step 1 - Run this FIRST.

Mines candidate synonym groups: finds cases where different query phrasings
all point to the SAME ground-truth product in train_positives.parquet.
That's a strong signal those phrasings are dialect/spelling synonyms of
each other, even if you don't recognize all of them yourself.

Output: synonym_candidates.json (needs your manual review - some grouped
queries will genuinely be different products with similar text, not real
synonyms) and synonyms_seed.json (a starter dictionary using the examples
you already know). Merge the reviewed candidates into synonyms.json,
which every later script reads.
"""
import json
from collections import defaultdict

import pandas as pd

from normalize import normalize_arabic

DATA_PATH = "data/train_positives.parquet"

# Seed with what you already know about the domain - free accuracy, no
# training required.
SEED_SYNONYMS = {
    "شيبس": ["بطاطا شيبس", "ليز", "شيبس", "سناكس", "تسالي", "مقرمشات"],
}


def mine_synonym_candidates(path: str, min_variants: int = 2) -> dict:
    df = pd.read_parquet(path)
    product_to_queries = defaultdict(set)

    for _, row in df.iterrows():
        doc = row["product_document"]
        name = doc.get("name") if isinstance(doc, dict) else str(doc)
        q_norm = normalize_arabic(row["query"])
        if q_norm:
            product_to_queries[name].add(q_norm)

    # Keep only products where multiple distinct query phrasings were used -
    # those phrasings are candidate synonyms of each other.
    return {
        name: sorted(qs)
        for name, qs in product_to_queries.items()
        if len(qs) >= min_variants
    }


def main():
    candidates = mine_synonym_candidates(DATA_PATH)

    with open("synonym_candidates.json", "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)

    with open("synonyms_seed.json", "w", encoding="utf-8") as f:
        json.dump(SEED_SYNONYMS, f, ensure_ascii=False, indent=2)

    print(f"Found {len(candidates)} products with 2+ distinct query phrasings.")
    print("Next steps:")
    print("  1. Open synonym_candidates.json - skim for real synonym groups")
    print("     (products where the phrasings are just dialect/spelling")
    print("     variants of the same term, not different products).")
    print("  2. Merge good groups + synonyms_seed.json into synonyms.json,")
    print("     format: {\"canonical_term\": [\"variant1\", \"variant2\", ...]}")


if __name__ == "__main__":
    main()
