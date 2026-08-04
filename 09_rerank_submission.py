
import pickle
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from normalize import load_synonyms, apply_normalization

BI_ENCODER_DIR = "finetuned-arabic-ecom-embed"
RERANKER_DIR = "finetuned-arabic-ecom-reranker"
CATALOG_PATH = "data/product_catalog.csv"
BM25_INDEX_PATH = "bm25_index.pkl"
TEST_QUERIES_PATH = "data/test_queries.csv"
OUTPUT_SUBMISSION_PATH = "submission.csv"

DENSE_WEIGHT = 0.7
BM25_WEIGHT = 0.3


ALPHA_STAGE1_WEIGHT = 0.5  

TOP_K_INITIAL = 150     
TOP_K_FINAL = 10       
ENCODE_BATCH_SIZE = 128
RERANK_BATCH_SIZE = 64


def minmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def main():
    synonyms = load_synonyms()
    print(f"Loaded {len(synonyms)} synonym groups")

    # 1. Load Product Catalog and Normalize
    catalog = pd.read_csv(CATALOG_PATH)
    product_ids = catalog["product_id"].tolist()
    product_names = catalog["product_name_ar"].tolist()
    product_texts = [
        apply_normalization(name, synonyms)
        for name in product_names
    ]

    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    bm25 = bm25_data["bm25"]
    if bm25_data["product_ids"] != product_ids:
        raise RuntimeError("Catalog order differs from BM25 index. Re-run 03_build_bm25_index.py.")

    print(f"Loading Bi-Encoder from {BI_ENCODER_DIR}...")
    bi_encoder = SentenceTransformer(BI_ENCODER_DIR)
    bi_encoder.max_seq_length = 64

    with torch.no_grad():
        product_embeddings = bi_encoder.encode(
            product_texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=ENCODE_BATCH_SIZE,
        )

    print(f"Loading Cross-Encoder Re-Ranker from {RERANKER_DIR}...")
    reranker = CrossEncoder(
        RERANKER_DIR,
        max_length=128,
        model_kwargs={"torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    )

    test_queries = pd.read_csv(TEST_QUERIES_PATH)
    print(f"Loaded {len(test_queries)} test queries. Processing 2-stage retrieval...")

    rows = []

    for i, row in enumerate(test_queries.itertuples(), 1):
        q_norm = apply_normalization(row.query_text, synonyms)

        with torch.no_grad():
            q_emb = bi_encoder.encode(q_norm, convert_to_tensor=True)
        dense_scores = util.cos_sim(q_emb, product_embeddings)[0].cpu().numpy()
        bm25_scores = bm25.get_scores(q_norm.split())

        stage1_combined = DENSE_WEIGHT * minmax(dense_scores) + BM25_WEIGHT * minmax(bm25_scores)
        
        top_idx = np.argsort(-stage1_combined)[:TOP_K_INITIAL]
        stage1_top_scores = stage1_combined[top_idx]

        candidates = [
            {
                "product_id": product_ids[idx],
                "product_name": product_names[idx],
                "stage1_raw_score": stage1_top_scores[j]
            }
            for j, idx in enumerate(top_idx)
        ]

        pair_inputs = [
            [q_norm, apply_normalization(cand["product_name"], synonyms)]
            for cand in candidates
        ]

        rerank_scores = reranker.predict(pair_inputs, batch_size=RERANK_BATCH_SIZE, show_progress_bar=False)

        stage1_norm = minmax(np.array([c["stage1_raw_score"] for c in candidates]))
        rerank_norm = minmax(np.array(rerank_scores))

        for j, cand in enumerate(candidates):
            cand["final_score"] = (ALPHA_STAGE1_WEIGHT * stage1_norm[j]) + ((1.0 - ALPHA_STAGE1_WEIGHT) * rerank_norm[j])

        ranked_candidates = sorted(candidates, key=lambda x: x["final_score"], reverse=True)

        top_10 = ranked_candidates[:TOP_K_FINAL]
        top_ids = [str(item["product_id"]) for item in top_10]

        rows.append({"query_id": row.query_id, "product_id": " ".join(top_ids)})

        if i % 1000 == 0:
            print(f"  Processed {i}/{len(test_queries)} queries")

    submission = pd.DataFrame(rows)
    submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
    print(f"\nSUCCESS! Blended re-ranked submission saved to {OUTPUT_SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
