"""
Step 2 - Run on your vast.ai GPU instance after Step 1.

Fine-tunes prestoai/qwen3-embedding-0.6b-arabic-ecom further using:
  - train_positives.parquet as (query, positive) pairs (in-batch negatives)
  - train_pairs_with_negatives.parquet as (query, positive, hard_negative)
    triplets - these are the highest-value examples, since hard negatives
    teach the model the exact semantic boundaries it currently gets wrong.

All text is run through the same normalize.apply_normalization() used
everywhere else in the pipeline - train, index, and query time must match.
"""
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

from normalize import load_synonyms, apply_normalization

MODEL_NAME = "prestoai/qwen3-embedding-0.6b-arabic-ecom"
OUTPUT_DIR = "finetuned-arabic-ecom-embed"
BATCH_SIZE = 64   # drop to 32/16 if you hit OOM on a smaller vast.ai GPU
EPOCHS = 2


def product_text(doc: dict, synonyms: dict) -> str:
    parts = [doc.get("name", "")]
    if doc.get("brand"):
        parts.append(doc["brand"])
    if doc.get("categories"):
        parts.extend(doc["categories"])
    if doc.get("description"):
        parts.append(doc["description"])
    raw = " ".join(str(p) for p in parts if p)
    return apply_normalization(raw, synonyms)


def build_examples(synonyms: dict) -> list:
    examples = []

    pos_df = pd.read_parquet("data/train_positives.parquet")
    for _, row in pos_df.iterrows():
        q = apply_normalization(row["query"], synonyms)
        p = product_text(row["product_document"], synonyms)
        examples.append(InputExample(texts=[q, p]))

    trip_df = pd.read_parquet("data/train_pairs_with_negatives.parquet")
    for _, row in trip_df.iterrows():
        q = apply_normalization(row["query"], synonyms)
        pos = product_text(row["product_document"], synonyms)
        neg = product_text(row["hard_negative_document"], synonyms)
        examples.append(InputExample(texts=[q, pos, neg]))

    return examples


def main():
    synonyms = load_synonyms()
    print(f"Loaded {len(synonyms)} synonym groups")

    model = SentenceTransformer(MODEL_NAME)
    examples = build_examples(synonyms)
    print(f"Built {len(examples)} training examples")

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=int(0.1 * len(train_dataloader)),
        use_amp=True,  # mixed precision - big throughput/cost win on rented GPUs
        output_path=OUTPUT_DIR,
        checkpoint_path=f"{OUTPUT_DIR}/checkpoints",
        checkpoint_save_steps=1000,
    )
    print(f"Saved fine-tuned model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
