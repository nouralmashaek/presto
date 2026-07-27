"""
Step 2 - Fine-tune prestoai/qwen3-embedding-0.6b-arabic-ecom on flat parquet data.
"""
import os
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

from normalize import load_synonyms, apply_normalization

MODEL_NAME = "prestoai/qwen3-embedding-0.6b-arabic-ecom"
OUTPUT_DIR = "finetuned-arabic-ecom-embed"
BATCH_SIZE = 32  # Reduce to 16 or 8 if your Vast.ai GPU runs out of VRAM (OOM)
EPOCHS = 1

def build_examples(synonyms: dict):
    pair_examples = []
    triplet_examples = []
    
    # 1. Load Positives (Pairs: Query + Positive)
    if os.path.exists("data/train_positives.parquet"):
        print("Loading train_positives.parquet...")
        df_pos = pd.read_parquet("data/train_positives.parquet")
        for row in df_pos.itertuples(index=False):
            q = apply_normalization(str(row.user_query), synonyms)
            pos = apply_normalization(str(row.positive_product_name), synonyms)
            if q and pos:
                pair_examples.append(InputExample(texts=[q, pos]))
                
    # 2. Load Pairs with Negatives (Triplets: Query + Positive + Hard Negative)
    if os.path.exists("data/train_pairs_with_negatives.parquet"):
        print("Loading train_pairs_with_negatives.parquet...")
        df_neg = pd.read_parquet("data/train_pairs_with_negatives.parquet")
        for row in df_neg.itertuples(index=False):
            q = apply_normalization(str(row.user_query), synonyms)
            pos = apply_normalization(str(row.positive_product_name), synonyms)
            neg = apply_normalization(str(row.negative_product_name), synonyms)
            if q and pos and neg:
                triplet_examples.append(InputExample(texts=[q, pos, neg]))
                
    return pair_examples, triplet_examples

def main():
    synonyms = load_synonyms()
    print(f"Loaded {len(synonyms)} synonym groups.")

    print(f"Downloading/Loading base model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    pair_examples, triplet_examples = build_examples(synonyms)
    print(f"Built {len(pair_examples)} pair examples (in-batch negatives).")
    print(f"Built {len(triplet_examples)} triplet examples (hard negatives).")

    train_objectives = []
    
    if triplet_examples:
        triplet_loader = DataLoader(triplet_examples, shuffle=True, batch_size=BATCH_SIZE)
        triplet_loss = losses.MultipleNegativesRankingLoss(model)
        train_objectives.append((triplet_loader, triplet_loss))
        
    if pair_examples:
        pair_loader = DataLoader(pair_examples, shuffle=True, batch_size=BATCH_SIZE)
        pair_loss = losses.MultipleNegativesRankingLoss(model)
        train_objectives.append((pair_loader, pair_loss))

    total_steps = sum(len(loader) for loader, _ in train_objectives)
    print(f"Starting training for {EPOCHS} epoch(s)... Total steps: {total_steps}")

    model.fit(
        train_objectives=train_objectives,
        epochs=EPOCHS,
        warmup_steps=int(0.1 * total_steps),
        use_amp=True,  # Mixed precision (faster on RTX 3090/4090/A5000 GPUs)
        output_path=OUTPUT_DIR,
        show_progress_bar=True
    )
    print(f"Training complete! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
