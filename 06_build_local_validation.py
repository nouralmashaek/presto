
import json
import random

import pandas as pd
from rapidfuzz import process, fuzz

SAMPLE_SIZE = 5000      
MATCH_THRESHOLD = 85
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
