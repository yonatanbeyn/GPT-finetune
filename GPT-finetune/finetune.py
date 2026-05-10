"""
finetune.py — Fine-tune GPT-2 on a custom text corpus.

Usage:
    python finetune.py

Requires:
    pip install -r requirements.txt
"""

import os
import torch
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

import config

# ──────────────────────────────────────────────
#  1. Device Detection
# ──────────────────────────────────────────────
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ──────────────────────────────────────────────
#  2. Load & Tokenize Dataset
# ──────────────────────────────────────────────
def load_dataset(tokenizer: GPT2TokenizerFast):
    """Read train.txt, split into paragraphs, tokenize, then split train/eval."""
    with open(config.DATA_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Split on blank lines; drop empty entries
    paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
    print(f"  Loaded {len(paragraphs)} paragraphs from {config.DATA_FILE}")

    dataset = Dataset.from_dict({"text": paragraphs})

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.BLOCK_SIZE,
            # No padding here — DataCollatorForLanguageModeling pads dynamically
            # and correctly masks pad tokens in labels
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    split = tokenized.train_test_split(test_size=config.EVAL_SPLIT, seed=42)
    return split["train"], split["test"]


# ──────────────────────────────────────────────
#  3. Build Training Arguments
# ──────────────────────────────────────────────
def build_training_args(device: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_steps=config.WARMUP_STEPS,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        logging_steps=config.LOGGING_STEPS,
        logging_dir=os.path.join(config.OUTPUT_DIR, "logs"),
        fp16=(device == "cuda"),       # mixed precision on CUDA only
        report_to="none",              # disable wandb / tensorboard by default
    )


# ──────────────────────────────────────────────
#  4. Main
# ──────────────────────────────────────────────
def main():
    device = get_device()
    print(f"\n{'='*50}")
    print(f"  GPT-2 Fine-Tuning")
    print(f"  Model    : {config.MODEL_NAME}")
    print(f"  Device   : {device.upper()}")
    print(f"  Epochs   : {config.NUM_EPOCHS}")
    print(f"  Eff. Batch: {config.BATCH_SIZE * config.GRAD_ACCUM_STEPS}")
    print(f"{'='*50}\n")

    # ── Tokenizer ──────────────────────────────
    print("[1/4] Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token   # GPT-2 has no pad token by default

    # ── Dataset ────────────────────────────────
    print("[2/4] Preparing dataset...")
    train_dataset, eval_dataset = load_dataset(tokenizer)
    print(f"  Train samples : {len(train_dataset)}")
    print(f"  Eval samples  : {len(eval_dataset)}")

    # ── Model ──────────────────────────────────
    print("[3/4] Loading model...")
    model = GPT2LMHeadModel.from_pretrained(config.MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))   # align with padded vocab
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ── Trainer ────────────────────────────────
    print("[4/4] Starting training...\n")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,   # causal LM — predict next token, not masked tokens
    )

    training_args = build_training_args(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # ── Save ───────────────────────────────────
    print(f"\nSaving model to {config.OUTPUT_DIR} ...")
    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)

    print("\nDone. Fine-tuned model saved.")
    print(f"Run  python inference.py  to generate text with the new model.\n")


if __name__ == "__main__":
    main()