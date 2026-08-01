"""
Step 6 - Build a local validation set for tuning DENSE_WEIGHT/BM25_WEIGHT
in 05_local_eval.py, without burning real leaderboard submissions.

WHY THIS EXISTS: train_pairs_with_negatives.parquet gives you
(query, positive_name, negative_name) as free text, not product_ids from
your actual catalog. Only ~8.5% of positive names match the catalog by
exact string (checked earlier in this project) - most refer to the same
*type* of product but a different SKU/wording. Exact matching would throw
away 90%+ of your usable validation signal, so this uses fuzzy string
matching (rapidfuzz) to link each name to its closest catalog product_id,
keeping only matches above a similarity threshold.

CAVEAT (same one 05_local_eval.py already documents): this is an
approximation for A/B comparing your own configs (fusion weights,
normalization changes), not a substitute for the real leaderboard score -
some fuzzy links will be wrong, but that noise applies equally to every
config you compare, so relative comparisons stay meaningful.

Output: local_validation.json - a list of
  {"query": ..., "positive_id": ..., "negative_id": ...}
rows, ready to feed into 05_local_eval.py's evaluate() as ground truth.
"""
import json
import random

import pandas as pd
from rapidfuzz import process, fuzz

SAMPLE_SIZE = 5000       # rows sampled from train_pairs_with_negatives before fuzzy matching
MATCH_THRESHOLD = 85     # rapidfuzz token_sort_ratio score (0-100) - only keep confident links
RANDOM_SEED = 0


def main():
    catalog = pd.read_csv("data/product_catalog.csv")
    cat_names = catalog["product_name_ar"].tolist()
    cat_ids = catalog["product_id"].tolist()
    name_to_id = dict(zip(cat_names, cat_ids))

    df = pd.read_parquet("data/train_pairs_with_negatives.parquet")
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_SEED)

    rows = []
    skipped = 0
    for i, row in enumerate(df.itertuples(index=False), 1):
        pos_match = process.extractOne(row.positive_product_name, cat_names, scorer=fuzz.token_sort_ratio)
        neg_match = process.extractOne(row.negative_product_name, cat_names, scorer=fuzz.token_sort_ratio)

        if not pos_match or pos_match[1] < MATCH_THRESHOLD:
            skipped += 1
            continue
        # negative match is nice-to-have, not required - a validation row
        # is still useful with just a linked positive. Also guard against
        # positive and negative both fuzzy-matching to the SAME catalog
        # product (happens when the two source names are similar) - that
        # would silently overwrite the grade-3 signal with a grade-0 one
        # for that product_id when building ground truth.
        neg_id = None
        if neg_match and neg_match[1] >= MATCH_THRESHOLD:
            candidate_neg_id = name_to_id[neg_match[0]]
            if candidate_neg_id != name_to_id[pos_match[0]]:
                neg_id = candidate_neg_id

        rows.append({
            "query": row.user_query,
            "positive_id": name_to_id[pos_match[0]],
            "negative_id": neg_id,
        })

        if i % 500 == 0:
            print(f"  processed {i}/{len(df)}, kept {len(rows)}, skipped {skipped}")

    with open("local_validation.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Kept {len(rows)}/{len(df)} rows (threshold={MATCH_THRESHOLD}) -> local_validation.json")


if __name__ == "__main__":
    main()
