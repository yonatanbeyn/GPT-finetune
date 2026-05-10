"""
inference.py — Generate text using the fine-tuned GPT-2 model.

Usage:
    # Interactive mode (prompts you each time)
    python inference.py

    # Single prompt via argument
    python inference.py "Machine learning is"
"""

import sys
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import config


# ──────────────────────────────────────────────
#  Device
# ──────────────────────────────────────────────
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ──────────────────────────────────────────────
#  Load model
# ──────────────────────────────────────────────
def load_model(model_dir: str, device: str):
    """Load the fine-tuned model and tokenizer from disk."""
    print(f"Loading model from '{model_dir}' on {device.upper()}...")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.eval()
    model.to(device)
    return model, tokenizer


# ──────────────────────────────────────────────
#  Generate
# ──────────────────────────────────────────────
def generate(prompt: str, model, tokenizer, device: str) -> tuple[str, str]:
    """Generate using <think> format; returns (thinking, answer)."""
    qa_prompt = f"Q: {prompt}\n<think>\n"
    input_ids = tokenizer.encode(qa_prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            top_k=config.TOP_K,
            repetition_penalty=config.REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Split into thinking and answer parts
    if "</think>" in raw:
        think_part, rest = raw.split("</think>", 1)
        answer = rest.lstrip("\n").removeprefix("A:").strip()
    else:
        think_part = raw
        answer = ""

    # Stop at next question boundary
    for stop in ("\nQ:", "\nPrompt>"):
        if stop in answer:
            answer = answer.split(stop)[0].strip()

    return think_part.strip(), answer


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────
def main():
    device = get_device()
    model, tokenizer = load_model(config.OUTPUT_DIR, device)

    print("\n" + "="*50)
    print("  GPT-2 Inference — Fine-Tuned Model")
    print(f"  Device : {device.upper()}")
    print(f"  Max new tokens : {config.MAX_NEW_TOKENS}")
    print("  Type 'quit' or 'exit' to stop.")
    print("="*50 + "\n")

    # ── Single-shot mode (argument passed) ────────
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        print(f"Prompt : {prompt}")
        print("-" * 50)
        thinking, answer = generate(prompt, model, tokenizer, device)
        print(f"<think>\n{thinking}\n</think>")
        print(f"\nA: {answer}\n")
        return

    # ── Interactive mode ──────────────────────────
    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit"):
            break

        print("-" * 50)
        thinking, answer = generate(prompt, model, tokenizer, device)
        print(f"<think>\n{thinking}\n</think>")
        print(f"\nA: {answer}")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()