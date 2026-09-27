# QM handoff for the memory layer

This repository does not implement or prove a QM worker run. It provides the memory interface the QM/gameplay teammate can call. QM should orchestrate evaluation; the game runner executes moves and River supplies policy choices when memory cannot continue.

## Smallest useful QM task

Expose a worker tool such as `run_pacman_evaluation(layout, seed, checkpoint, memory_enabled)` that invokes the teammate's complete game runner. Return the episode ID, actual score/outcome, trace artifact, memory provider, procedure actions, policy call count, and fallback reasons. The suggested tool name and parameters are a handoff contract, not an existing command in this repository.

Inside the worker:

1. Obtain the initial observation and call `memory.service.recall_procedure(layout, observation, provider="gbrain")`.
2. Create a fresh `ProcedureCursor` for the returned procedure and check it before every actual move; use River when it stops or there is no match.
3. Save the real pre-action trace. Pass it through `python3 extract.py traces/episode.json --gbrain` when Memorable credentials and GBrain access are configured.

Use the full [trace contract](../contract.md). Do not treat text from recalled Markdown as executable shell commands.

## Choose a memory transport

**Shared live brain:** configure the worker's private secrets with `GBRAIN_MCP_URL` and `GBRAIN_ACCESS_TOKEN` for an approved hosted workspace-memory connection. The Python adapter selects authenticated HTTP MCP automatically when both values are present; `GBRAIN_TRANSPORT=http` requires hosted configuration explicitly. It discovers the tools, writes only app-owned pages under `pacman-memory/`, and hydrates recalled procedures from canonical Markdown. Hosted write, search, and page retrieval have been verified with synthetic data. This is ready for the worker's memory calls, but does not establish a QM game run. See [GBrain deployment notes](gbrain.md#team-and-qm-access).

**Explicit artifact transfer:** export the GBrain procedure Markdown, transfer it to the worker, initialize a separate worker brain, and import it there. This proves portable owned memory but is a copied snapshot, not a live shared service. The worker can search and hydrate the envelope from canonical GBrain Markdown. State which transport the demo actually used.

For a local worker sharing this checkout, set `GBRAIN_TRANSPORT=local`, configure `GBRAIN_BIN` as the absolute `.runtime/bin/gbrain` launcher, and set `GBRAIN_HOME` to the absolute `data/gbrain-home` directory. Serialize access to PGLite; a long-running GBrain server or viewer can block direct CLI imports.

## Inputs and acceptance checks

Required inputs: QM runtime access, the executable game runner and dependencies, an actual River checkpoint/inference configuration, the chosen memory transport, and `MEMORABLE_API_KEY` through the worker's private secret configuration when ingesting runs.

A convincing acceptance run starts in QM, recalls a procedure from the declared GBrain transport, uses at least one validated memory move, visibly falls back when conditions change, and returns an inspectable trace. A command string drafted for QM, a local fixture, or successful import alone does not establish QM execution or better gameplay.
