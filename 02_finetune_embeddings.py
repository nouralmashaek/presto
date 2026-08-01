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
                         # a large fixed chunk of a 24GB card before any batch
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

    # 1. CRITICAL FIX: Load weights in 16-bit to cut static VRAM & AdamW overhead in half!
    model = SentenceTransformer(
        MODEL_NAME,
        model_kwargs={"torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    )

    # Queries/product names are almost always short (p99 is 6 words for
    # queries, 13 words for product names), but a handful of outlier rows
    # run to 100+ words. Since every sequence in a batch gets padded to
    # the longest one in that batch, one such outlier landing in a batch
    # spikes memory far past what a normal batch needs - that's what was
    # causing the OOM partway through training, not a bad batch size.
    # Capping this truncates rare outliers instead of blowing up memory.
    model.max_seq_length = 64

    # Gradient checkpointing trades ~20% slower training for a large cut in
    # activation memory (it recomputes activations during the backward pass
    # instead of keeping them all resident) - worth it given how much of the
    # 24GB is already claimed by optimizer state before batches even start.
    try:
        # 2. CRITICAL FIX: Disable use_cache before checkpointing so attention KV caches don't hoard VRAM!
        model[0].auto_model.config.use_cache = False
        model[0].auto_model.gradient_checkpointing_enable()
        # Companion call required for checkpointing to actually free
        # activations on decoder-style models like this one - without it,
        # checkpointing partially engages (you get further into training
        # before OOM, since some savings apply) but earlier-step activations
        # aren't fully released, so memory creeps up until it runs out. This
        # matches the symptom exactly: crashing further along each time
        # rather than immediately.
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
    output_path=OUTPUT_DIR,
    checkpoint_save_steps=10000,
    checkpoint_save_total_limit=1,
    )
    print(f"Saved fine-tuned model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
