
import json
import pickle

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

from normalize import load_synonyms, apply_normalization

MODEL_DIR = "finetuned-arabic-ecom-embed"
WEIGHT_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  


def minmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def dcg_at_10(ranked_grades: list) -> float:
    return sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(ranked_grades[:10]))


def ndcg_at_10(ranked_grades: list) -> float:
    ideal = sorted(ranked_grades, reverse=True)
    idcg = dcg_at_10(ideal)
    return dcg_at_10(ranked_grades) / idcg if idcg else 0.0


def main():
    synonyms = load_synonyms()

    catalog = pd.read_csv("data/product_catalog.csv")
    product_ids = catalog["product_id"].tolist()
    product_texts = [apply_normalization(row.product_name_ar, synonyms) for row in catalog.itertuples()]

    with open("bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    bm25 = bm25_data["bm25"]
    if bm25_data["product_ids"] != product_ids:
        raise RuntimeError("Catalog order differs from BM25 index - re-run 03_build_bm25_index.py.")

    with open("local_validation.json", encoding="utf-8") as f:
        val_rows = json.load(f)
    print(f"Loaded {len(val_rows)} local validation queries")

    model = SentenceTransformer(MODEL_DIR)
    model.max_seq_length = 64

  
    id_to_text = dict(zip(product_ids, product_texts))

    with torch.no_grad():
        product_embeddings = model.encode(
            product_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=128
        )
        
    per_query = []
    for i, row in enumerate(val_rows, 1):
        q_norm = apply_normalization(row["query"], synonyms)
        with torch.no_grad():
            q_emb = model.encode(q_norm, convert_to_tensor=True)
        dense_norm = minmax(util.cos_sim(q_emb, product_embeddings)[0].cpu().numpy())
        bm25_norm = minmax(bm25.get_scores(q_norm.split()))

        grades = {row["positive_id"]: 3}
        if row["negative_id"]:
            grades[row["negative_id"]] = 0
        # also grade by TEXT so duplicate-ID products with identical
        # names get the same grade as the ground-truth id
        text_grades = {}
        for pid, grade in grades.items():
            text = id_to_text.get(pid)
            if text:
                text_grades[text] = max(text_grades.get(text, -1), grade)

        per_query.append((dense_norm, bm25_norm, grades, text_grades))
        if i % 500 == 0:
            print(f"  scored {i}/{len(val_rows)} validation queries")

    print()
    print(f"{'DENSE_WEIGHT':>12} {'BM25_WEIGHT':>12} {'nDCG@10':>10}")
    best = (None, None, -1.0)
    diagnostic_grid = [0.0] + WEIGHT_GRID + [1.0]  # 0.0 = pure BM25, 1.0 = pure dense
    for dense_w in diagnostic_grid:
        bm25_w = round(1 - dense_w, 2)
        scores = []
        for dense_norm, bm25_norm, grades, text_grades in per_query:
            combined = dense_w * dense_norm + bm25_w * bm25_norm
            top_idx = np.argsort(-combined)[:10]
            top_ids = [product_ids[i] for i in top_idx]
            ranked_grades = []
            for pid in top_ids:
                if pid in grades:
                    ranked_grades.append(max(grades[pid], 0))
                else:
                    text = id_to_text.get(pid)
                    ranked_grades.append(max(text_grades.get(text, 0), 0))
            scores.append(ndcg_at_10(ranked_grades))
        mean_ndcg = float(np.mean(scores))
        tag = " (pure BM25)" if dense_w == 0.0 else (" (pure dense)" if dense_w == 1.0 else "")
        print(f"{dense_w:>12} {bm25_w:>12} {mean_ndcg:>10.4f}{tag}")
        if mean_ndcg > best[2] and 0.0 < dense_w < 1.0:
            best = (dense_w, bm25_w, mean_ndcg)

    print()
    print(f"Best: DENSE_WEIGHT={best[0]}, BM25_WEIGHT={best[1]} (nDCG@10={best[2]:.4f})")
    print("Set these in 04_hybrid_retrieve.py before generating your real submission.csv")


if __name__ == "__main__":
    main()
