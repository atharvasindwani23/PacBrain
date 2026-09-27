"""LoRA fine-tune DeepSeek-R1-Distill-Qwen-1.5B on Pacman expert moves.
Run:  modal run --detach modal_train.py
Saves the merged model to Modal Volume 'pacbrain-models' at /tuned.
"""

import json
import os
import hashlib
import time

import modal

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MAX_LENGTH = 512


def encode_pair(tokenizer, pair, max_length=MAX_LENGTH):
    """Keep every supervised move token; never train an all-masked example."""
    if not isinstance(pair, dict) or any(
            not isinstance(pair.get(key), str) or not pair[key].strip()
            for key in ("prompt", "completion")):
        raise ValueError("Training pairs require nonempty prompt and completion strings")
    prompt = tokenizer(pair["prompt"], add_special_tokens=True)["input_ids"]
    completion = tokenizer(pair["completion"], add_special_tokens=False)["input_ids"]
    if not completion or tokenizer.eos_token_id is None:
        raise ValueError("A completion token and tokenizer EOS token are required")
    completion = completion + [tokenizer.eos_token_id]
    if len(prompt) + len(completion) > max_length:
        raise ValueError("Training example exceeds max length; shorten the prompt or raise the limit")
    return {"input_ids": prompt + completion,
            "labels": [-100] * len(prompt) + completion}

app = modal.App("pacbrain-train")
vol = modal.Volume.from_name("pacbrain-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "transformers==4.44.2", "peft==0.12.0",
                 "accelerate==0.33.0", "datasets==2.21.0", "hf-transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=image, gpu="A100", timeout=3600, volumes={"/vol": vol})
def train(pairs: list, seed: int = 42, revision: str = "main"):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments, set_seed)

    if not pairs:
        raise ValueError("No training pairs supplied")
    set_seed(seed)
    started = time.monotonic()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=revision)
    examples = [encode_pair(tok, pair) for pair in pairs]
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=revision, torch_dtype=torch.bfloat16, device_map="cuda")
    resolved_revision = getattr(model.config, "_commit_hash", None)

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad] * n)
            labels.append(b["labels"] + [-100] * n)
            attn.append([1] * len(b["input_ids"]) + [0] * n)
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}

    args = TrainingArguments(
        output_dir="/tmp/out", num_train_epochs=3,
        per_device_train_batch_size=32, gradient_accumulation_steps=1,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, logging_steps=20, save_strategy="no", report_to=[],
        seed=seed, data_seed=seed)

    Trainer(model=model, args=args, train_dataset=examples,
            data_collator=collate).train()

    merged = model.merge_and_unload()
    merged.save_pretrained("/vol/tuned")
    tok.save_pretrained("/vol/tuned")
    manifest = {
        "base_model": BASE_MODEL, "requested_revision": revision,
        "resolved_revision": resolved_revision, "seed": seed,
        "pairs": len(pairs), "max_length": MAX_LENGTH,
        "dataset_sha256": hashlib.sha256(json.dumps(
            pairs, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
        "elapsed_scope": "tokenizer/model loading, training and merged checkpoint save",
        "training_args": args.to_dict(),
    }
    with open("/vol/tuned/training_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    vol.commit()
    print("saved merged model to volume at /tuned")


@app.local_entrypoint()
def main(data: str = "data/train.jsonl", seed: int = 42, revision: str = "main"):
    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, data)) as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    if not pairs:
        raise ValueError("The training dataset is empty")
    print(f"training on {len(pairs)} pairs")
    train.remote(pairs, seed, revision)
