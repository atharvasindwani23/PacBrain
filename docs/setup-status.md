# Verified setup status

Updated September 27, 2026.

## Memory implementation

- Python memory extraction, trace validation, guarded procedure replay, local persistence, and provider adapters are implemented.
- 80 unit tests pass on Python 3.9 and 3.12. The offline demo recalls a stored move and stops replay when an active ghost approaches.
- A **synthetic** two-move trace successfully passed through the real Memorable extraction API and was written to hosted Gbrain over authenticated HTTP MCP.
- Hosted Gbrain search and canonical page retrieval were verified from a fresh local store. Local Gbrain CLI/PGLite also passed import, search, and retrieval.
- Final hosted check selected the Memorable procedure `52c7585bcf24cb59` from an empty local store, validated its receipt, and returned the guarded action `E`.
- Review fixes cover concurrent file publication, ended episodes, malformed flags, useful positive prefixes, receipt/content integrity, bounded draft expansion, and provider-ranked ties. Hosted search uses a bounded candidate pool before namespace filtering.
- Ingestion reports storage and observed search visibility separately. A write/readback alone does not prove discovery by search.
- Hosted writes use immutable pages under `pacman-memory/`. Provider failures are surfaced; local fallback is explicit.
- Credentials are stored only in the ignored local `.env`. Teammates must configure their own private environment; keys are not included in this repository.

These checks prove integration, not gameplay improvement. Genuine game traces, a policy, and paired evaluation runs are still required.

## Runtime setup (paused)

Personal-account sign-in and workspace creation are complete. The inspected workspace has **zero instances and a $0 balance**. The QM setup page asks for a $5 payment; Billing offers card-based promotional credit. No payment, card entry, or automatic top-up was authorized or enabled during setup.

The team subsequently dropped Agent 37 in favor of Modal, then paused Modal setup. No hosted worker has been launched. The Modal SDK was installed in an ignored local environment; sandbox execution and the game runner remain separate follow-up work. The memory module works locally and supports hosted Gbrain independently of that decision.

## Training

River provides both training and inference, so neither Modal nor Baseten is required for the proposed River path. The game worker sends examples to River for LoRA training, saves an inference checkpoint, and uses that checkpoint for future sampling. This needs a River API key and model access, owned by the gameplay/training integration.

Sources: [River guide](https://docs.river.ai/) and [Python API](https://docs.river.ai/python-api/). Baseten also offers [managed training](https://docs.baseten.co/training/getting-started), but is an alternative rather than an additional dependency.

## Next team integration

1. Connect the environment's real pre-action observations and executed actions to [the trace contract](../contract.md).
2. Ingest a genuine rewarded opening, then demonstrate replay and the guarded handoff back to the policy.
3. Evaluate the same layouts and seeds with and without memory; record scores, API calls, latency, and replay stop reasons. Connect that runner to QM after hosted access is available.

See [architecture](architecture.md) and [QM handoff](qm-handoff.md). `system-design.md` is a separate design proposal using DeepSeek; choosing the action model remains a team decision and does not change the memory interface.
