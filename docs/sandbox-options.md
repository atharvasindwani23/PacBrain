# Sandbox choice for the hackathon

Research date: September 27, 2026. Exa returned 40 search results across four workstreams (QM/Agent37, E2B, Daytona, Modal); the conclusions below use official documentation and source code. Setup rankings are project-specific judgments, not measured startup benchmarks. No provider has been provisioned by this research.

## Recommendation

Use the existing **hosted QM computer through Agent37** if it is already available. Give it a ten-minute acceptance gate: run Python, install the complete game runner, call the required external APIs, and export a JSON trace. If that fails, run the game in **E2B** through one narrow QM tool that launches an evaluation and returns its result. Keep local Docker as the development fallback when Docker is already installed.

Do not implement a new QM sandbox backend during the hackathon. A QM tool may call a sandbox service's SDK without replacing QM's own agent computer. Describe that accurately in the demo: QM launches the job; E2B isolates the gameplay process; River chooses policy actions.

## What QM supplies

QM describes itself as a multiplayer agent harness, but it also defines a per-scope durable sandbox interface and an `execute` tool. A provider supplies the actual isolated computer. The distinction is **QM orchestrates; the configured provider executes and isolates**. The official README links Agent37 as a hosted QM option. [QM README](https://github.com/yc-software/qm), [Agent37 hosted QM](https://www.agent37.com/qm).

QM's repository includes provider-specific implementations, including a [Modal implementation](https://github.com/yc-software/qm/blob/main/src/sandbox/modal-sandbox.ts), [Sprites implementation](https://github.com/yc-software/qm/blob/main/src/sandbox/sprites-sandbox.ts), and [Superserve guide](https://github.com/yc-software/qm/blob/main/docs/superserve.md). Available deployment configuration varies by release; verify the exact version before assuming one environment variable enables a provider. Exa returned different revisions of configuration and documentation, so this research does not claim that every listed implementation is available in the hosted hackathon instance.

## Options

| Option | Why it fits | Credentials and file handoff | Main tradeoff for this build |
| --- | --- | --- | --- |
| Agent37 hosted QM | Keeps the sponsor integration and job execution in the same existing environment. | Requires access to the provisioned QM workspace; verify Python, private secret configuration, outbound API access, and trace export there. | Least additional infrastructure **if already provisioned**. The public landing page does not establish the hackathon instance's specific limits. |
| E2B | Python SDK exposes sandbox creation, shell commands, environment variables, internet access, and filesystem reads/writes. | `E2B_API_KEY` on the launcher; narrowly scoped runtime secrets via `envs`; read the trace with the filesystem API before destroying the sandbox. | Best separate job-runner fallback. It is a new account/SDK unless already available, and is not automatically a native QM backend. |
| Modal | Python sandbox API, image dependencies, secret injection, configurable egress, and explicit filesystem transfer. | Modal authentication; `modal.Secret` for provider credentials; `filesystem.copy_to_local` or `read_text` for artifacts. | Strong choice if a teammate already has Modal configured; flexible batch execution. Requires an App and image setup. |
| Daytona | Python sandbox creation, command/session execution, custom images, environment configuration, and file upload/download. | `DAYTONA_API_KEY`; runtime `env_vars` or organization secrets; `sandbox.fs` transfers. | Strong alternative for a persistent development workspace, but no demonstrated setup-speed advantage for this single Python job. |
| Local Docker | Runs the same packaged Python runner with explicit CPU/memory limits and a mounted output directory. | Local Docker installation; ignored env file; output mount contains JSON traces. | Fast local fallback when installed. It is container isolation and still depends on the host; it does not make a laptop-local memory service remotely reachable. |

Sources: [E2B quickstart](https://e2b.dev/docs/quickstart), [E2B Python sandbox API](https://docs.e2b.dev/sdk-reference/python-sdk/v2.29.4/sandbox_sync), [Modal sandboxes](https://modal.com/docs/guide/sandboxes), [Modal files](https://modal.com/docs/guide/sandbox-files), [Modal networking](https://modal.com/docs/guide/sandbox-networking), [Daytona Python SDK](https://www.daytona.io/docs/en/python-sdk/), [Daytona creation/configuration](https://www.daytona.io/docs/en/python-sdk/sync/daytona/), [Daytona files](https://www.daytona.io/docs/en/file-system-operations/), [Docker resource limits](https://docs.docker.com/engine/containers/resource_constraints/).

## What belongs inside the sandbox

- The complete headless game runner, dependencies, current episode state, and procedure cursor.
- The River policy client, with credentials passed privately to that run.
- A trace writer that records actual observations/actions, memory hits, fallbacks, score, and policy-call count.
- Either a worker-local GBrain process with its persistent data stored explicitly, or an authenticated client to a shared HTTPS GBrain service.

The frontend, durable experiment results, and long-term memory should outlive any individual game sandbox. Export trace artifacts before ending the job. Keep River training/checkpoints in River, and keep procedural memory in the chosen GBrain store. The memory pipeline uses Memorable to extract a procedure from a real trace, then stores the returned draft and game validation metadata in GBrain. See [memory handoff](qm-handoff.md).

## Critical networking detail

A hosted sandbox cannot call the laptop's `localhost`. The current Python GBrain adapter invokes a local CLI; a hosted worker therefore needs that CLI/database installed next to it, explicit Markdown artifact transfer, or a new HTTP MCP transport to an authenticated GBrain server. A public hostname by itself does not solve this. Do not call a copied Markdown snapshot a live shared-memory connection. See [GBrain deployment notes](gbrain.md#team-and-qm-access).

## Minimal runtime contract

Proposed tool: `run_pacman_evaluation(layout, seed, checkpoint, memory_enabled)`.

1. Launch the selected sandbox with a pinned game-runner revision and a bounded timeout.
2. Run the episode. Recall a procedure, validate it before every action, and ask River when the procedure no longer applies.
3. Return `episode_id`, score/outcome, trace artifact, provider used, memory-action count, policy-call count, and fallback reasons. Export artifacts and terminate the job.

One CPU-only runner is sufficient architecturally when River serves policy inference remotely; actual game latency and memory requirements still need measurement. No desktop, VNC, GPU, public inbound game server, or additional sandbox nested inside an already adequate QM computer is required for this headless design.

## Credits and claims

The earlier sponsor lecture was reported to offer Agent37 access, but this research did not independently verify that specific hackathon promotion. The public Agent37 cloud site describes its own normal onboarding and billing; those terms are not evidence of the hackathon entitlement. Verify the available balance in the user's existing account and avoid a paid top-up without explicit authorization. [Agent37 cloud](https://www.agent37.com/cloud), [billing documentation](https://www.agent37.com/docs/agents-api/billing).
