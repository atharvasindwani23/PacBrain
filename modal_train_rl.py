"""GRPO RL fine-tune: DeepSeek-1.5B learns Pacman from environment rewards
only (no expert imitation). For each collected state the model samples moves;
rewards come from a per-action lookup computed by the game engine
(see rewards.py). Illegal outputs get -1.5.

Run:  modal run --detach modal_train_rl.py
Saves the merged model to Modal Volume 'pacbrain-models' at /rl.
"""

import json
import os

import modal

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ILLEGAL_REWARD = -1.5

app = modal.App("pacbrain-train-rl")
vol = modal.Volume.from_name("pacbrain-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.49.0", "trl==0.15.2",
                 "peft==0.14.0", "accelerate==1.3.0", "datasets==3.2.0",
                 "hf-transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

ACTION_WORDS = {"north": "NORTH", "south": "SOUTH", "east": "EAST",
                "west": "WEST", "stop": "STOP"}


def completion_to_action(text: str):
    text = text.strip().lower()
    first = text.split()[0].strip(".,!") if text.split() else ""
    if first in ACTION_WORDS:
        return ACTION_WORDS[first]
    for word, action in ACTION_WORDS.items():
        if word in text:
            return action
    return None


@app.function(image=image, gpu="A100", timeout=7200, volumes={"/vol": vol})
def train(states: list, base_model: str = BASE_MODEL, out_dir: str = "/vol/rl"):
    import datasets
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    def reward_env(prompts=None, completions=None, rewards=None, **kwargs):
        out = []
        for comp, table in zip(completions, rewards):
            action = completion_to_action(comp)
            # datasets pads the rewards struct with None for absent actions
            val = table.get(action) if action else None
            out.append(ILLEGAL_REWARD if val is None else val)
        return out

    ds = datasets.Dataset.from_list(states)

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])

    cfg = GRPOConfig(
        output_dir="/tmp/out",
        model_init_kwargs={"torch_dtype": "bfloat16"},
        num_train_epochs=1,
        per_device_train_batch_size=48,
        gradient_accumulation_steps=1,
        num_generations=8,
        max_prompt_length=448,
        max_completion_length=8,
        temperature=1.0,
        beta=0.04,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=base_model,
        reward_funcs=reward_env,
        args=cfg,
        train_dataset=ds,
        peft_config=lora,
    )
    trainer.train()

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(out_dir)
    with open(f"{out_dir}/log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f)
    vol.commit()
    print(f"saved merged RL model + log history to {out_dir}")


@app.local_entrypoint()
def main(states_file: str = "rl_states.jsonl",
         base_model: str = BASE_MODEL, out_dir: str = "/vol/rl"):
    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, "data", states_file)) as f:
        states = [json.loads(line) for line in f]
    print(f"GRPO on {len(states)} states: {base_model} -> {out_dir}")
    train.remote(states, base_model, out_dir)
