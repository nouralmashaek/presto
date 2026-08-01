"""
Step 8 - Stage 2 Cross-Encoder Re-Ranker Training (Modern v4/v5 API)

Why this script exists:
Stage 1 (Bi-Encoder + BM25) retrieves top candidate products fast, but caps out
around 0.55 NDCG/Recall because compressing strings into static vectors loses 
token-level reasoning. This script trains a Cross-Encoder that reads:
    [CLS] user_query [SEP] product_title [SEP]
allowing full self-attention between query words and product attributes.

Data flow:
Takes train_pairs_with_negatives.parquet and creates two labeled pairs per row:
  1. (normalized_query, normalized_positive_name) -> Label: 1.0 (Relevant)
  2. (normalized_query, normalized_negative_name) -> Label: 0.0 (Hard Negative)

Memory Optimizations for RTX 3090 (24GB):
  - 16-bit BFloat16 weight loading via modern CrossEncoderTrainer
  - max_length capped at 128 tokens
  - use_cache=False + Gradient Checkpointing
"""
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer, CrossEncoderTrainingArguments
from sklearn.model_selection import train_test_split

from normalize import load_synonyms, apply_normalization

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
OUTPUT_DIR = "finetuned-arabic-ecom-reranker"

BATCH_SIZE = 16          # Safe for 24GB VRAM with 560M param Cross-Encoder
EPOCHS = 1               # 1 epoch is usually plenty for re-rankers on large datasets
MAX_LENGTH = 128         # Caps padding memory spikes for short e-commerce titles
LEARNING_RATE = 2e-5     # Standard gentle fine-tuning rate for re-rankers


def build_labeled_dictionaries(synonyms: dict) -> list:
    """Builds explicit positive (1.0) and negative (0.0) training dictionaries
    compatible with HuggingFace datasets."""
    records = []
    df = pd.read_parquet("data/train_pairs_with_negatives.parquet")
    
    print("Normalizing text and generating positive/negative pairs...")
    for row in df.itertuples(index=False):
        if not isinstance(row.positive_product_name, str) or not row.positive_product_name:
            continue
            
        q = apply_normalization(row.user_query, synonyms)
        pos = apply_normalization(row.positive_product_name, synonyms)
        
        # Add the positive match (Label 1.0)
        records.append({"text_a": q, "text_b": pos, "label": 1.0})
        
        # Add the hard negative match if it exists (Label 0.0)
        if isinstance(row.negative_product_name, str) and row.negative_product_name:
            neg = apply_normalization(row.negative_product_name, synonyms)
            records.append({"text_a": q, "text_b": neg, "label": 0.0})
            
    return records


def main():
    synonyms = load_synonyms()
    print(f"Loaded {len(synonyms)} synonym groups")

    # 1. Load Cross-Encoder in 16-bit precision to halve static VRAM consumption
    print(f"Loading Cross-Encoder base model: {MODEL_NAME}...")
    model = CrossEncoder(
        MODEL_NAME,
        num_labels=1,
        max_length=MAX_LENGTH,
        model_kwargs={"torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    )

    # 2. Enable Memory Optimizations on the underlying HuggingFace transformer
    try:
        model.model.config.use_cache = False
        model.model.gradient_checkpointing_enable()
        if hasattr(model.model, "enable_input_require_grads"):
            model.model.enable_input_require_grads()
        print("Gradient checkpointing & use_cache=False enabled successfully")
    except (AttributeError, IndexError) as e:
        print(f"Could not enable gradient checkpointing ({e}) - continuing without it")

    # 3. Build HuggingFace Datasets and split off a small 2% validation set
    all_records = build_labeled_dictionaries(synonyms)
    print(f"Total labeled pairs created: {len(all_records)}")
    
    train_records, val_records = train_test_split(all_records, test_size=0.02, random_state=42)
    print(f"Training pairs: {len(train_records)} | Validation pairs: {len(val_records)}")

    train_dataset = Dataset.from_list(train_records)
    
    # 4. Set up binary classification evaluator (requires legacy InputExample list format)
    from sentence_transformers import InputExample
    val_examples = [InputExample(texts=[r["text_a"], r["text_b"]], label=r["label"]) for r in val_records]
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        val_examples, name="arabic-ecom-val"
    )

    # 5. Define Modern Training Arguments (Bypasses FP16 GradScaler bug via native bf16=True!)
    training_args = CrossEncoderTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.05,
        bf16=True,                      # Native Ampere 16-bit precision (No GradScaler!)
        fp16=False,
        eval_strategy="steps",
        eval_steps=5000,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=1,
        logging_steps=500,
        remove_unused_columns=False,    # Required for custom string pair columns
    )

    # 6. Launch Modern CrossEncoderTrainer
    print("Starting Cross-Encoder training with modern Trainer API...")
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        evaluator=evaluator,
    )
    
    trainer.train()
    model.save(OUTPUT_DIR)
    print(f"Training complete! Saved best re-ranker model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
