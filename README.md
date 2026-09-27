# PacBrain

We fine-tuned DeepSeek-R1-Distill-Qwen-1.5B to play Pacman — and it went from
wandering into walls to clearing boards.

Built at the YC "Own Your Intelligence" hackathon (Sep 27, 2026).

## How it works

1. **Expert data** — a scripted BFS expert plays Pacman (`edq-pacai` engine);
   every winning game's (board-state, move) pairs become training data
   (`gen_data.py`, ~23k pairs from 400 games).
2. **Fine-tune** — LoRA SFT on Modal (A100, ~8 min), completion-only loss on
   the move word (`modal_train.py`).
3. **Serve** — vLLM OpenAI-compatible endpoints on Modal for the base and
   tuned checkpoints (`modal_serve.py`).
4. **Evaluate** — the same agent harness (`llm_agent.py`, `eval.py`) plays
   fixed-seed games against both endpoints; scores, win rate, illegal-move
   rate, and GIFs land in `results/`.

Every game also writes a structured JSON trace to `traces/` (consumed by our
GBrain/Memorable integration).

## Run it

```sh
python3 -m venv .venv && .venv/bin/pip install edq-pacai modal openai
.venv/bin/python gen_data.py 400          # expert data
.venv/bin/modal run --detach modal_train.py
.venv/bin/modal deploy modal_serve.py
.venv/bin/python eval.py --endpoint <base-url>/v1 --label base
.venv/bin/python eval.py --endpoint <tuned-url>/v1 --label tuned
.venv/bin/python make_results.py
```

## Credits

Game engine: [edq-pacai](https://pypi.org/project/edq-pacai/), a Python 3
rewrite of the UC Berkeley CS188 Pacman projects (educational use,
attribution to UC Berkeley's CS188 course materials — http://ai.berkeley.edu).
Model: DeepSeek-R1-Distill-Qwen-1.5B (MIT).
