# Own Your Intelligence — Pac-Man Memory

Person C's memory layer for the hackathon: validate a successful Pac-Man opening, extract its actual action sequence through Memorable, store it in GBrain, and replay it only while the current game still matches.

This repository supplies memory and integration utilities. The playable environment, River policy/training, replay viewer, and QM worker execution must be connected by the other teammates.

## Start in three steps

1. **Run the local checks and agree on the trace contract.** Python 3.9+ is sufficient for the Python code; there are no Python package dependencies.

   ```bash
   python3 -m unittest discover -s tests -v
   python3 scripts/demo_memory.py
   python3 -m memory doctor
   ```

   The demo uses a **synthetic fixture and local storage only**. It shows an opening being recalled and replay stopping when a ghost approaches. Send the gameplay teammate [contract.md](contract.md); they must log pre-action observations and the actions actually executed.

2. **Connect the real memory providers.**

   Create a private `.env` from [.env.example](.env.example), preserving any existing values. Supply `MEMORABLE_API_KEY` from the [Memorable dashboard](https://memorable.sh/dash). For hosted GBrain, set `GBRAIN_MCP_URL` and `GBRAIN_ACCESS_TOKEN` to the approved workspace-memory connection. With both present, the adapter automatically uses authenticated HTTP MCP. A partial hosted configuration reports an error.

   To use local PGLite instead, run:

   ```bash
   bash scripts/setup_gbrain.sh
   export GBRAIN_TRANSPORT=local
   export GBRAIN_BIN="$PWD/.runtime/bin/gbrain"
   export GBRAIN_HOME="$PWD/data/gbrain-home"
   ```

   Local GBrain keyword search needs no API key. The setup downloads a verified workspace-local Bun runtime and pinned GBrain source; it does not modify global shell configuration. `GBRAIN_TRANSPORT=local` explicitly overrides hosted settings. `doctor` reports configuration and selected transport only, not a live connection check.

3. **Ingest a real game and use the recalled procedure.** Replace these example paths/layout with the gameplay runner's actual output:

   ```bash
   python3 extract.py traces/episode.json --gbrain
   python3 -m memory recall --layout mediumClassic --provider gbrain --observation traces/current_observation.json
   ```

   Extraction defaults to the real Memorable API. `--gbrain` indexes the validated envelope; recall defaults to actual GBrain search. Add the per-move cursor loop from [contract.md](contract.md). A failed provider call returns an error, without silently claiming local fallback as sponsor usage.

## What works and what remains

| Component | Current status |
|---|---|
| Opening extraction and guarded replay | Implemented and tested; rejects mismatched positions, changed walls, illegal moves, and nearby active ghosts |
| GBrain | Hosted HTTP MCP write → search → canonical page retrieval verified with a synthetic memory page; isolated local PGLite also verified |
| Memorable | Live extraction API verified with the labeled synthetic trace; returned steps and request receipt stored with the procedure |
| Gameplay benefit | Needs genuine episodes and comparison runs; no score improvement is claimed |
| River | Gameplay/training teammate integration required |
| QM | Handoff instructions provided; no hosted worker execution has been demonstrated |

Both live memory providers have been verified using synthetic data only. Real gameplay benefit still needs a game trace matching the contract and the teammate's environment/policy entry point. A new machine or QM worker also needs its own approved hosted connection values; credentials are not included in this repository.

Useful explicit local-only commands:

```bash
python3 -m memory extract examples/synthetic_trace.json --extractor local
python3 -m memory recall --layout testCorridor --provider local
```

These are development checks, not evidence of Memorable usage. The stored envelope records `extraction` and provider receipts; the CLI reports `gbrain_indexed` and retrieval `provider` separately.

Implementation details: [trace and replay contract](contract.md), [GBrain setup](docs/gbrain.md), [Memorable API](docs/memorable.md), [QM handoff](docs/qm-handoff.md).
