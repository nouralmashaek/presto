import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

from normalize import load_synonyms, apply_normalization

MODEL_NAME = "prestoai/qwen3-embedding-0.6b-arabic-ecom"
OUTPUT_DIR = "finetuned-arabic-ecom-embed"
PAIR_BATCH_SIZE = 12     # (query, positive) - 2 texts encoded per example
TRIPLET_BATCH_SIZE = 8   # (query, positive, negative) - 3 texts per example,
                         # kept roughly proportional to PAIR_BATCH_SIZE by texts-
                         # per-batch (16*2=32 vs 10*3=30). Dropped further from
                         # 48/32 - AdamW optimizer states + gradients + fp32
                         # master weights for a 0.6B param model already consume
                         # a large fixed chunk of a 24GB card before any batch rtx 3090 gpu 
                         # activations are even allocated, so there was less
                         # headroom than the raw VRAM total suggested.
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

    
    model = SentenceTransformer(
        MODEL_NAME,
        model_kwargs={"torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    )

  
    model.max_seq_length = 64

    try:
       
        model[0].auto_model.config.use_cache = False
        model[0].auto_model.gradient_checkpointing_enable()
        
        if hasattr(model[0].auto_model, "enable_input_require_grads"):
            model[0].auto_model.enable_input_require_grads()
        print("Gradient checkpointing & use_cache=False enabled successfully")
    except (AttributeError, IndexError) as e:
        print(f"Could not enable gradient checkpointing ({e}) - continuing without it")

    pair_examples = build_pair_examples(synonyms)
    triplet_examples = build_triplet_examples(synonyms)
    print(f"Built {len(pair_examples)} pair examples (in-batch negatives)")
    print(f"Built {len(triplet_examples)} triplet examples (hard negatives)")

    pair_dataloader = DataLoader(
    pair_examples,
    shuffle=True,
    batch_size=PAIR_BATCH_SIZE,
    pin_memory=True,
    )

    triplet_dataloader = DataLoader(
    triplet_examples,
    shuffle=True,
    batch_size=TRIPLET_BATCH_SIZE,
    pin_memory=True,
    )

  
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
    output_path=OUTPUT_DIR,
    checkpoint_save_steps=10000,
    checkpoint_save_total_limit=1,
    )
    print(f"Saved fine-tuned model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
