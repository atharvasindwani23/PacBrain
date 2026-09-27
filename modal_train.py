"""LoRA fine-tune DeepSeek-R1-Distill-Qwen-1.5B on Pacman expert moves.
Run:  modal run --detach modal_train.py
Saves the merged model to Modal Volume 'pacbrain-models' at /tuned.
"""

import json
import os

import modal

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

app = modal.App("pacbrain-train")
vol = modal.Volume.from_name("pacbrain-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "transformers==4.44.2", "peft==0.12.0",
                 "accelerate==0.33.0", "datasets==2.21.0", "hf-transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=image, gpu="A100", timeout=3600, volumes={"/vol": vol})
def train(pairs: list):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def encode(p):
        prompt_ids = tok(p["prompt"], add_special_tokens=True)["input_ids"]
        comp_ids = tok(p["completion"], add_special_tokens=False)["input_ids"]
        comp_ids = comp_ids + [tok.eos_token_id]
        ids = (prompt_ids + comp_ids)[:512]
        labels = ([-100] * len(prompt_ids) + comp_ids)[:512]
        return {"input_ids": ids, "labels": labels}

    examples = [encode(p) for p in pairs]

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id or tok.eos_token_id
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
        bf16=True, logging_steps=20, save_strategy="no", report_to=[])

    Trainer(model=model, args=args, train_dataset=examples,
            data_collator=collate).train()

    merged = model.merge_and_unload()
    merged.save_pretrained("/vol/tuned")
    tok.save_pretrained("/vol/tuned")
    vol.commit()
    print("saved merged model to volume at /tuned")


@app.local_entrypoint()
def main():
    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, "data", "train.jsonl")) as f:
        pairs = [json.loads(line) for line in f]
    print(f"training on {len(pairs)} pairs")
    train.remote(pairs)
