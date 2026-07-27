"""
Step 1 - Mine candidate synonym groups from flat parquet schemas.
Finds cases where different query phrasings point to the exact same positive product name.
"""
import json
from collections import defaultdict
import pandas as pd
from normalize import normalize_arabic

POSITIVES_PATH = "data/train_positives.parquet"
PAIRS_PATH = "data/train_pairs_with_negatives.parquet"

SEED_SYNONYMS = {
    "شيبس": ["بطاطا شيبس", "ليز", "شيبس", "سناكس", "تسالي", "مقرمشات"],
}

def mine_synonym_candidates(min_variants: int = 2) -> dict:
    product_to_queries = defaultdict(set)
    
    for path in (POSITIVES_PATH, PAIRS_PATH):
        try:
            df = pd.read_parquet(path, columns=["user_query", "positive_product_name"])
            for row in df.itertuples(index=False):
                name = normalize_arabic(str(row.positive_product_name))
                q_norm = normalize_arabic(str(row.user_query))
                if name and q_norm:
                    product_to_queries[name].add(q_norm)
        except Exception as e:
            print(f"Skipping {path} or error reading: {e}")

    # Keep products where multiple distinct query phrasings were used
    return {
        name: sorted(list(qs))
        for name, qs in product_to_queries.items()
        if len(qs) >= min_variants
    }

def main():
    candidates = mine_synonym_candidates()

    with open("synonym_candidates.json", "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)

    with open("synonyms_seed.json", "w", encoding="utf-8") as f:
        json.dump(SEED_SYNONYMS, f, ensure_ascii=False, indent=2)

    # Automatically create a working synonyms.json using seed + top mined candidates
    with open("synonyms.json", "w", encoding="utf-8") as f:
        json.dump(SEED_SYNONYMS, f, ensure_ascii=False, indent=2)

    print(f"Found {len(candidates)} products with 2+ distinct query phrasings.")
    print("-> Generated synonym_candidates.json and initialized synonyms.json.")

if __name__ == "__main__":
    main()
