import json
from collections import defaultdict

import pandas as pd

from normalize import normalize_arabic

POSITIVES_PATH = "data/train_positives.parquet"
PAIRS_PATH = "data/train_pairs_with_negatives.parquet"


def is_clean_query(q: str) -> bool:
    """Drop rows where user_query is a leaked JSON-key fragment, e.g.
    'attribute_discriminator": "..."' or a bare '{' - these come from
    whatever process generated the queries, not real user input, and
    show up in ~0.25% of rows in both training files."""
    if not isinstance(q, str) or not q.strip():
        return False
    if '"' in q or q.strip() in ("{", "}"):
        return False
    return True

SEED_SYNONYMS = {
    "شيبس": ["بطاطا شيبس", "ليز", "شيبس", "سناكس", "تسالي", "مقرمشات"],
}


def mine_synonym_candidates(min_variants: int = 2) -> dict:
    """Group distinct query phrasings by the product name they point to.

    Pools train_positives.parquet and train_pairs_with_negatives.parquet -
    both give (user_query, positive_product_name) pairs, just with
    different amounts of query traffic per file, so combining them
    surfaces more candidate phrasings per product than either alone.
    """
    product_to_queries = defaultdict(set)

    for path in (POSITIVES_PATH, PAIRS_PATH):
        df = pd.read_parquet(path, columns=["user_query", "positive_product_name"])
        for row in df.itertuples(index=False):
            name = row.positive_product_name
            if not is_clean_query(row.user_query):
                continue
            q_norm = normalize_arabic(row.user_query)
            if isinstance(name, str) and name and q_norm:
                product_to_queries[name].add(q_norm)


    return {
        name: sorted(qs)
        for name, qs in product_to_queries.items()
        if len(qs) >= min_variants
    }


def main():
    candidates = mine_synonym_candidates()

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
