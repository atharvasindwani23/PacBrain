# PacBrain

**Teach a small model to play. Give it a memory it can keep.**

Built for the **Own Your Intelligence** hackathon. PacBrain combines a Pac-Man
policy trained with reinforcement learning — no teacher, no example games, only
the game's own score counter (GRPO) — with procedural memory powered by
**Memorable + Gbrain**.

[Recorded results](results/results.md) · [Memory contract](contract.md) · [Architecture](docs/architecture.md) · [Verified status](docs/setup-status.md)

## See the demo

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Open **http://127.0.0.1:8000/viewer.html** for the recorded before/after replays, per-seed results, and score comparison. The viewer uses checked-in artifacts and requires no API keys. For a live game in the browser: `./demo.sh both`.

| Recorded evaluation | Base model | Reward-trained (GRPO) |
| --- | ---: | ---: |
| Invalid response / API-error fallback rate | 82.5% | 0% |
| Mean score | −428.1 | −422.1 |
| Pellets eaten (10 games) | 109 | 116 |

Ten recorded games per model on `classic-small`, seeds 2000–2009. The fallback
metric counts proposals that could not become legal moves, including request
failures; the engine receives a legal fallback. The reward-only run learned the
game's action rules completely (0% invalid) in ~11 GPU-minutes; ghost evasion
is the open frontier. Training checkpoints (reward every 10 steps) are in
`results/training_curve.json`.

## Reward-only RL (no expert anywhere)

1. **Explore** — 200 cheap rollout games produce 3,000 unique board states
   (`collect_states.py`). No scripted expert plays a single move.
2. **Price every move** — for each state, the game engine itself scores every
   legal action: +10 pellet, +200 ghost, +500 board clear, −500 death, −1 per
   time step, plus potential-based shaping toward food; illegal outputs get
   −1.5 (`rewards.py`).
3. **Reinforce (GRPO)** — on a Modal A100 the model samples 8 moves per board;
   moves that out-earn their siblings are reinforced (`modal_train_rl.py`,
   LoRA, ~11 min). Metric history is checkpointed every 10 steps.
4. **Serve** — vLLM OpenAI-compatible endpoints on Modal for base and trained
   checkpoints (`modal_serve.py`).
5. **Rematch** — the identical agent harness (`llm_agent.py`, `eval.py`) plays
   the same 10 fixed-seed games against both endpoints, decoding game events
   (pellets, ghosts eaten, deaths) from the per-move score timeline.

## Run locally

Use Python **3.11 or 3.12** for the complete project. The standalone memory package also supports Python 3.9.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/demo_memory.py
```

Copy `.env.example` to a private `.env` and supply the memory provider credentials when using live extraction or recall. Set `PACBRAIN_API_KEY` if your model endpoint requires authentication. Never commit credentials. The recorded viewer and offline tests work without them.

## How the pieces fit

| Component | Responsibility |
| --- | --- |
| `collect_agent.py`, `collect_states.py`, `rewards.py` | Explore the game and price every legal move from environment rewards |
| `serializer.py` | Shared board prompt and action parsing for training and inference |
| `modal_train_rl.py` | GRPO LoRA reinforcement learning on a Modal GPU (reward-only) |
| `modal_train.py`, `expert.py`, `gen_data.py` | Legacy SFT path (expert imitation), kept for comparison |
| `modal_serve.py` | Serve base and trained models through vLLM |
| `llm_agent.py`, `eval.py` | Run a model policy, guarded memory replay, and per-game evaluation |
| `memory/`, `extract.py`, `recall.py` | Validate observed openings, extract through Memorable, persist and retrieve through Gbrain |
| `viewer.html`, `results/`, `build_story.py` | Present recorded evaluation results, replays, and the story page |

## Train and evaluate

Training and deployment use your Modal account and GPU resources; merging this repository does not launch either job.

```bash
.venv/bin/python -m pip install -r requirements-modal.txt
.venv/bin/python collect_states.py 200 3000        # explore (reward-only path)
.venv/bin/modal run --detach modal_train_rl.py     # reinforce (GRPO)
.venv/bin/modal deploy modal_serve.py              # serve
.venv/bin/python eval.py --endpoint <base-url>/v1 --label base_new
.venv/bin/python eval.py --endpoint <rl-url>/v1 --label rl_new
.venv/bin/python make_results.py
./demo.sh both                                     # live browser demo
```

Generation defaults to `data/generated/` and `traces/generated/`. Evaluation uses separate labels to preserve the supplied benchmark. Existing outputs require an explicit `--overwrite`. The report command reads the `base` and `rl` artifacts. Set the policy endpoint and its private credential according to `eval.py --help` and the deployment configuration.

For memory setup and the full ingestion/replay loop, follow [the memory guide](docs/memory.md) and [the gameplay contract](contract.md). Both local Gbrain and authenticated hosted MCP are supported. The cloud transport requires no laptop tunnel.

## Scope and provenance

The implemented model path uses **DeepSeek + Modal**. River and QM were part of the original plan; their handoff notes remain reference material, and no River training or QM gameplay run is claimed here. A real winning expert run was recorded, extracted into an 18-step opening by Memorable, stored in hosted Gbrain, and recalled from an empty local store. All 18 guarded actions matched the observed trace. This verifies the memory loop; matched model runs are still needed to measure its gameplay benefit.

This integration preserves both the `pacbrain` gameplay history and the original memory history.

## Credits

Game engine: [edq-pacai](https://pypi.org/project/edq-pacai/), a Python rewrite of the [UC Berkeley CS188 Pac-Man projects](http://ai.berkeley.edu). Preserve the engine's educational-use attribution. Model: [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), MIT license. GRPO via TRL.
