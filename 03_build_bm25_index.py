import pickle

import pandas as pd
from rank_bm25 import BM25Okapi

from normalize import load_synonyms, apply_normalization


def main():
    synonyms = load_synonyms()
    catalog = pd.read_csv("data/product_catalog.csv")

    product_ids = []
    tokenized_corpus = []
    for row in catalog.itertuples():
        text = f"{row.product_name_ar} {row.category_name_ar}"
        text = apply_normalization(text, synonyms)
        product_ids.append(row.product_id)
        tokenized_corpus.append(text.split())

    bm25 = BM25Okapi(tokenized_corpus)

    with open("bm25_index.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "product_ids": product_ids}, f)

    print(f"Indexed {len(product_ids)} products -> bm25_index.pkl")


if __name__ == "__main__":
    main()
