# Pac-Man traces

There are two formats in this repository. Use canonical traces for procedural memory.

## Original training traces

The checked-in `expert_1000.json` through `expert_1399.json` contain the original 400 games:

```json
{
  "game": "pacman",
  "board": "classic-small",
  "seed": 1000,
  "agent": "expert",
  "score": -88,
  "win": false,
  "steps": [
    {"prompt": "<pre-action serialized board and legal moves>", "completion": " West"}
  ]
}
```

These preserve training prompts, chosen moves, and final outcomes. Their 31,598 steps have no per-action scores or final observation, so they cannot establish observed reward for safe memory extraction. Do not submit the raw legacy object as a Memorable tool trace or infer intermediate scores from the final result.

Audit the original files without changing them:

```sh
.venv/bin/python -m memory.pacbrain traces
```

## Canonical memory traces

New data generation writes separate canonical files with authoritative engine scores and confirmed executed actions:

```sh
.venv/bin/python gen_data.py 1 --start-seed 3000 --workers 1
```

The example produces `traces/generated/expert_3000.json` for training pairs and `traces/generated/canonical/expert_3000.json` for memory. Existing output requires an explicit `--overwrite`. Both the expert and LLM agent also enable canonical logging when `PACBRAIN_TRACE` is set:

```sh
PACBRAIN_TRACE=/tmp/expert-3000.json PACBRAIN_BOARD=classic-small PACBRAIN_SEED=3000 \
  .venv/bin/python -m pacai.pacman --ui null --board classic-small \
  --pacman expert.py:ExpertAgent --seed 3000 --max-turns 600
```

Each canonical file includes:

- `episode_id`, `layout`, `seed`, `agent`, and `synthetic: false`.
- `frames`: the grid and `state.score` immediately before each action; cardinal direction, source, legal moves, and terminal flags.
- `action_ok`: confirmation against PacAI's action history on the following observation. A changed action is recorded as executed and labeled `fallback`.
- `terminal_observation` and `result`: the actual final board, score, clear/death state, confirmed Pac-Man moves, engine turns, model calls, and confirmed procedure steps.

`policy` includes the scripted expert's chosen actions; the `agent` field distinguishes expert from LLM play. `STOP` actions remain visible in traces but stop extraction because memory currently replays only cardinal moves. LLM traces additionally record retrieval provider, procedure step, and fallback reason. The episode-completion callback saves the file atomically, including death states whose board has no `P`.

Extract a new canonical trace through the validated sponsor pipeline:

```sh
.venv/bin/python -m memory extract traces/generated/canonical/expert_3000.json --gbrain
```

This uses configured Memorable and GBrain credentials. For a local check without provider calls, replace `--gbrain` with `--extractor local`. Extraction only keeps an independently verified, positively rewarded safe opening; a game's win or high final score does not make every move replayable. See [the memory contract](../contract.md).

`eval.py` saves canonical traces under `results/<label>_traces/` by default. Set `--trace-dir` to choose another directory. Use a fresh result label for each comparison; `--overwrite` explicitly replaces that label's existing artifacts and removes stale GIFs.
