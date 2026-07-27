import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
 
from normalize import load_synonyms, apply_normalization
 
MODEL_NAME = "prestoai/qwen3-embedding-0.6b-arabic-ecom"
OUTPUT_DIR = "finetuned-arabic-ecom-embed"
BATCH_SIZE = 64   # drop to 32/16 if you hit OOM on a smaller vast.ai GPU
EPOCHS = 2
 
 
def build_pair_examples(synonyms: dict) -> list:
    """(query, positive_name) pairs from train_positives.parquet.
    No negative_product_name here - MultipleNegativesRankingLoss uses the
    other positives in the batch as in-batch negatives."""
    examples = []
    df = pd.read_parquet(
        "data/train_positives.parquet",
        columns=["user_query", "positive_product_name"],
    )
    for row in df.itertuples(index=False):
        if not isinstance(row.positive_product_name, str) or not row.positive_product_name:
            continue
        q = apply_normalization(row.user_query, synonyms)
        p = apply_normalization(row.positive_product_name, synonyms)
        examples.append(InputExample(texts=[q, p]))
    return examples
 
 
def build_triplet_examples(synonyms: dict) -> list:
    """(query, positive_name, hard_negative_name) triplets from
    train_pairs_with_negatives.parquet - the highest-value training
    signal since the negative is a real near-miss, not a random other
    product."""
    examples = []
    df = pd.read_parquet("data/train_pairs_with_negatives.parquet")
    for row in df.itertuples(index=False):
        if not isinstance(row.negative_product_name, str) or not row.negative_product_name:
            continue
        q = apply_normalization(row.user_query, synonyms)
        pos = apply_normalization(row.positive_product_name, synonyms)
        neg = apply_normalization(row.negative_product_name, synonyms)
        examples.append(InputExample(texts=[q, pos, neg]))
    return examples
 
 
def main():
    synonyms = load_synonyms()
    print(f"Loaded {len(synonyms)} synonym groups")
 
    model = SentenceTransformer(MODEL_NAME)
 
    pair_examples = build_pair_examples(synonyms)
    triplet_examples = build_triplet_examples(synonyms)
    print(f"Built {len(pair_examples)} pair examples (in-batch negatives)")
    print(f"Built {len(triplet_examples)} triplet examples (hard negatives)")
 
    pair_dataloader = DataLoader(pair_examples, shuffle=True, batch_size=BATCH_SIZE)
    triplet_dataloader = DataLoader(triplet_examples, shuffle=True, batch_size=BATCH_SIZE)
 
    # Same loss class works for both 2-text and 3-text InputExamples - it
    # just treats any texts beyond the first two as extra hard negatives
    # when present.
    pair_loss = losses.MultipleNegativesRankingLoss(model)
    triplet_loss = losses.MultipleNegativesRankingLoss(model)
 
    total_steps = len(pair_dataloader) + len(triplet_dataloader)
 
    model.fit(
        train_objectives=[
            (triplet_dataloader, triplet_loss),
            (pair_dataloader, pair_loss),
        ],
        epochs=EPOCHS,
        warmup_steps=int(0.1 * total_steps),
        use_amp=True,  # mixed precision - big throughput/cost win on rented GPUs
        output_path=OUTPUT_DIR,
        checkpoint_path=f"{OUTPUT_DIR}/checkpoints",
        checkpoint_save_steps=1000,
    )
    print(f"Saved fine-tuned model to {OUTPUT_DIR}")
 
 
if __name__ == "__main__":
    main()
 
