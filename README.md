# PacBrain

**Teach a small model to play. Give it a memory it can keep.**

Built for the **Own Your Intelligence** hackathon. PacBrain combines a Pac-Man policy fine-tuned on expert games with procedural memory powered by **Memorable + Gbrain**.

[Recorded results](results/results.md) · [Memory contract](contract.md) · [Architecture](docs/architecture.md) · [Verified status](docs/setup-status.md)

## See the demo

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Open **http://127.0.0.1:8000/viewer.html** for the recorded before/after replays, per-seed results, and score comparison. The viewer uses checked-in artifacts and requires no API keys.

| Recorded evaluation | Base model | Fine-tuned model |
| --- | ---: | ---: |
| Mean score | −428.1 | 387.1 |
| Games won | 0 / 10 | 5 / 10 |
| Invalid response / API-error fallback rate | 82.48% | 0% |

These are ten recorded games per model on `classic-small`, seeds 2000–2009. The fallback metric counts proposals that could not become legal moves, including request failures; the engine receives a legal fallback. These artifacts demonstrate the recorded fine-tuning result, **not a measured memory benefit**. They do not establish performance on new layouts or a large evaluation set.

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
| `expert.py`, `gen_data.py` | Play CPU expert games and collect training examples |
| `serializer.py` | Shared board prompt and action parsing for training and inference |
| `modal_train.py` | LoRA SFT of DeepSeek-R1-Distill-Qwen-1.5B on a Modal GPU |
| `modal_serve.py` | Serve base and tuned models through vLLM |
| `llm_agent.py`, `eval.py` | Run a model policy, guarded memory replay, and per-game evaluation |
| `memory/`, `extract.py`, `recall.py` | Validate observed openings, extract through Memorable, persist and retrieve through Gbrain |
| `viewer.html`, `results/` | Present recorded evaluation results and replays |

The repository includes **400 recorded expert games**, of which **300 were wins**, and **23,439 unique training examples**. The legacy prompt/completion traces remain useful training evidence, but omit per-step scores and cannot be passed directly to the guarded memory extractor. See [trace formats](traces/README.md).

## Train and evaluate

The existing scripts are included for reproducibility. Training and deployment use your Modal account and GPU resources; merging this repository does not launch either job.

```bash
.venv/bin/python -m pip install -r requirements-modal.txt
# The supplied dataset is data/train.jsonl. Generate a new one only if needed:
.venv/bin/python gen_data.py 400
.venv/bin/modal run --detach modal_train.py --data data/generated/train.jsonl
.venv/bin/modal deploy modal_serve.py
.venv/bin/python eval.py --endpoint <base-endpoint>/v1 --label base_new
.venv/bin/python eval.py --endpoint <tuned-endpoint>/v1 --label tuned_new
.venv/bin/python make_results.py
```

Generation defaults to `data/generated/` and `traces/generated/`. Evaluation uses separate labels to preserve the supplied benchmark. Existing outputs require an explicit `--overwrite`. The report command reads the original `base` and `tuned` artifacts. Set the policy endpoint and its private credential according to `eval.py --help` and the deployment configuration.

For memory setup and the full ingestion/replay loop, follow [the memory guide](docs/memory.md) and [the gameplay contract](contract.md). Both local Gbrain and authenticated hosted MCP are supported. The cloud transport requires no laptop tunnel.

## Scope and provenance

The implemented model path uses **DeepSeek + Modal**. River and QM were part of the original plan; their handoff notes remain reference material, and no River training or QM gameplay run is claimed here. A real winning expert run was recorded, extracted into an 18-step opening by Memorable, stored in hosted Gbrain, and recalled from an empty local store. All 18 guarded actions matched the observed trace. This verifies the memory loop; matched model runs are still needed to measure its gameplay benefit.

This integration preserves both the `pacbrain` gameplay history and the original memory history.

## Credits

Game engine: [edq-pacai](https://pypi.org/project/edq-pacai/), a Python rewrite of the [UC Berkeley CS188 Pac-Man projects](http://ai.berkeley.edu). Preserve the engine's educational-use attribution. Model: [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), MIT license. Training examples and evaluation artifacts were supplied in the `pacbrain` branch.
