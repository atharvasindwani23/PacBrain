# Verified setup status

Updated September 27, 2026.

## Integrated application

The `pacbrain` gameplay branch and memory implementation are merged with both histories preserved. The application contains CPU expert data collection, shared board serialization, DeepSeek LoRA training and vLLM serving definitions for Modal, model evaluation, guarded Memorable/Gbrain memory, and a static results dashboard.

The dashboard was checked on desktop and a 390px mobile viewport. Both replay seeds, the alternate result variant, still-frame controls, the score chart, and the source-data table work without API keys.

## Real memory evidence

- A CPU expert run on `classic-small`, seed **3000**, cleared the board with score **973** in 229 engine turns.
- The [canonical trace](../examples/pacbrain_expert_trace.json) contains **77 pre-action frames**, each confirmed against the engine action history, plus the final observation.
- Memorable extracted the independently validated **18-step opening**, with **162 observed reward**, into procedure `8d2532d94a9e87fb`.
- Authenticated hosted Gbrain reported one successful import. Search visibility and canonical page retrieval were verified.
- A fresh temporary local store retrieved that same Memorable-backed procedure from Gbrain and returned guarded action **W**. Replaying against the recorded observations matched all **18 actions**, then stopped with reason `complete`.
- Separate CPU runs verified a death ending (seed 1000) and a turn-limit ending (seed 777). The terminal serializer now handles cleared boards with no active agent.

This proves real-game recording, provider extraction, durable storage, fresh-store retrieval, and guarded trace replay. It does **not** prove a memory-related win-rate improvement or 18 saved model calls in a new live episode. That requires matched model evaluations with and without memory.

## Recorded model benchmark

The supplied `pacbrain` artifacts report ten games per model on seeds 2000–2009: base mean **−428.1**, zero wins; tuned mean **387.1**, five wins. The reported invalid-response/API-failure fallback rate changes from **82.48%** to **0%**.

The source branch supplied 400 expert games (300 wins), 23,439 unique training examples, and four GIFs. These original training traces omit per-step scores and final observations; they cannot be used as verified memory traces. New collection writes both training pairs and canonical traces.

The old benchmark does not record the exact model revision, elapsed training time, or complete sampling metadata. Its results are preserved as supplied. The updated training path records a manifest and rejects overlong examples instead of silently discarding supervised move tokens.

## Verification and runtime

**99 offline tests pass on Python 3.12.** CI runs the same suite on Python 3.11 and 3.12. Unit and integration tests cover extraction, provider transports, replay guards, concurrent persistence, complete game recording, policy-call bypass during replay, and evaluation counters. Live sponsor calls were limited to the memory check above; no new GPU training or model-serving deployment was started during integration.

Credentials remain in the ignored local `.env`; each teammate must configure their own runtime. Both hosted Gbrain HTTP MCP and local Gbrain CLI/PGLite are supported. Provider failures are surfaced and local fallback is explicit.

QM and Agent 37 are out of scope at the team's request. Modal training/serving code is included from the gameplay branch; additional sandbox provisioning remains paused. River is a historical proposal, not the implemented model path.

## Next experiment

Use a fixed memory corpus and the same model endpoint, settings, layout, and evaluation seeds for separate `--memory none` and `--memory gbrain` runs. Report scores, wins, policy calls, confirmed procedure actions, failed games, and replay stop reasons. The [memory guide](memory.md) contains the commands.
