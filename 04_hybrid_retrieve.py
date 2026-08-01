import pickle

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

from normalize import load_synonyms, apply_normalization

MODEL_DIR = "finetuned-arabic-ecom-embed"
DENSE_WEIGHT = 0.7
BM25_WEIGHT = 0.3
TOP_K = 10
ENCODE_BATCH_SIZE = 128  # inference-only, no gradients/optimizer state, so this
                         # can be much larger than the training batch sizes were


def minmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def main():
    synonyms = load_synonyms()

    catalog = pd.read_csv("data/product_catalog.csv")
    product_ids = catalog["product_id"].tolist()
    product_texts = [
        apply_normalization(row.product_name_ar, synonyms)
        for row in catalog.itertuples()
    ]

    with open("bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    bm25 = bm25_data["bm25"]
    if bm25_data["product_ids"] != product_ids:
        raise RuntimeError(
            "Catalog order differs from when the BM25 index was built - "
            "re-run 03_build_bm25_index.py."
        )

    model = SentenceTransformer(MODEL_DIR)
    model.max_seq_length = 64  # match what the model was fine-tuned with

    with torch.no_grad():
        product_embeddings = model.encode(
            product_texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=ENCODE_BATCH_SIZE,
        )

    test_queries = pd.read_csv("data/test_queries.csv")
    rows = []

    for i, row in enumerate(test_queries.itertuples(), 1):
        q_norm = apply_normalization(row.query_text, synonyms)

        with torch.no_grad():
            q_emb = model.encode(q_norm, convert_to_tensor=True)
        dense_scores = util.cos_sim(q_emb, product_embeddings)[0].cpu().numpy()

        bm25_scores = bm25.get_scores(q_norm.split())

        combined = DENSE_WEIGHT * minmax(dense_scores) + BM25_WEIGHT * minmax(bm25_scores)
        top_idx = np.argsort(-combined)[:TOP_K]
        top_ids = [product_ids[i] for i in top_idx]

        rows.append({"query_id": row.query_id, "product_id": " ".join(top_ids)})

        if i % 2000 == 0:
            print(f"  {i}/{len(test_queries)} queries processed")

    submission = pd.DataFrame(rows)
    submission.to_csv("submission.csv", index=False)
    print(f"Wrote submission.csv ({len(submission)} queries)")


if __name__ == "__main__":
    main()
