# Memory integration

Memory stores a verified opening with explicit replay conditions. Each recalled action is rechecked against a fresh game observation. Stochastic ghosts can invalidate an otherwise familiar route.

## Configure

Supply private values in `.env`, using [the example](../.env.example):

- `MEMORABLE_API_KEY` for live extraction.
- `GBRAIN_MCP_URL` and `GBRAIN_ACCESS_TOKEN` for hosted Gbrain.
- Or `GBRAIN_TRANSPORT=local` and the local CLI configuration from [Gbrain setup](gbrain.md).

```bash
python3 -m memory doctor
python3 scripts/demo_memory.py
```

Doctor checks configuration; the demo uses a labeled synthetic fixture without sponsor API calls.

## Ingest and recall

Use a **canonical pre-action trace** with observed per-step scores, as specified in [contract.md](../contract.md). The older `traces/expert_*.json` training traces do not contain enough evidence for this operation.

```bash
python3 extract.py path/to/canonical-game.json --gbrain
python3 -m memory recall --layout classic-small --provider gbrain \
  --observation path/to/current-observation.json
```

Extraction validates contiguous observed transitions, retains a positively rewarded opening, asks Memorable to extract the selected actions, and rejects a draft that changes those actions. It stores JSON and Markdown locally before publishing immutable versions to hosted Gbrain.

`gbrain_stored` confirms persistence. `gbrain_search_verified` confirms that the layout query returned the procedure ID; `gbrain_indexed` is a compatibility alias for that search check. A successful write alone does not prove immediate search visibility.

The runner uses a fresh `ProcedureCursor` per episode. It stops on changed topology, position divergence, an illegal move, nearby active ghosts, or an ended episode. Inspect the recorded source and stop reason to distinguish a memory action, a policy action, and a fallback.

## Record an actual game

The CPU expert can produce a canonical trace without a model endpoint:

```bash
PACBRAIN_TRACE=data/expert-3000.json PACBRAIN_BOARD=classic-small PACBRAIN_SEED=3000 \
  .venv/bin/python -m pacai.pacman --ui null --board classic-small \
  --pacman expert.py:ExpertAgent --seed 3000 --max-turns 600
.venv/bin/python extract.py data/expert-3000.json --gbrain
```

For a model evaluation, use a separate label for each condition:

```bash
.venv/bin/python eval.py --endpoint <tuned-endpoint>/v1 --label tuned_no_memory --memory none
.venv/bin/python eval.py --endpoint <tuned-endpoint>/v1 --label tuned_gbrain --memory gbrain
```

Both use seeds 2000–2009 by default. Freeze the memory corpus before either evaluation. Supply `PACBRAIN_API_KEY` if the model endpoint requires authentication. Each run saves canonical traces under `results/<label>_traces/`, plus policy-call, executed-procedure, fallback, and provider-failure counters. Memory is disabled unless explicitly selected; an unavailable provider is recorded and control returns to the model policy.

## Evidence and limits

The real `classic-small` expert episode at seed 3000 scored **973** and cleared the board. Its 77 actions were confirmed against the engine history. Memorable extracted an **18-step** opening with **162 observed reward**; hosted Gbrain stored it and returned it through search to an empty local store. Its first guarded move was `W`, and all 18 actions matched the recorded observations before the cursor completed. The canonical evidence is [checked in](../examples/pacbrain_expert_trace.json). See [verified status](setup-status.md) for the limits of this check.

Memory is currently opening replay, not general strategic reasoning or model training. A reduction in policy calls does not by itself prove better scores. Compare matched runs and report the model, layout, seeds, memory corpus, score, wins, policy attempts, procedure actions, and fallback reasons.

More detail: [hosted transport](gbrain-http.md), [Memorable API](memorable.md).
