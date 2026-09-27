# Memorable integration

This adapter calls the real extraction service. It does not present local move
copying as a successful Memorable extraction. Live extraction needs a workspace
API key from [the Memorable dashboard](https://memorable.sh/dash): Account →
New key for an agent (or Environments → Connect). Set `MEMORABLE_API_KEY` in
your shell or your private `.env`; never commit the value. The service retains
the task line and extracted steps for its dashboard. Only game-action metadata
is sent by this adapter, never board contents or the complete trace.

## Exact integration

```python
from memory.memorable import MemorableClient, game_trace_to_memorable, moves_from_draft

wire_trace = game_trace_to_memorable(game_trace, max_steps=40)
extraction = MemorableClient.from_env().extract_trace(wire_trace)
moves = moves_from_draft(extraction["draft"])
# Persist extraction["draft"] + request_id in the team's GBrain procedure
# envelope, together with preconditions taken from the original game trace.
```

Frames are pre-action records for actions actually executed by the environment.
Use `frame.action_ok` only when the environment recorded that action's outcome.
It describes successful tool execution, not whether the game was won. Without
it the adapter omits the outcome instead of inferring success from score.
Extraction may refuse traces without enough useful evidence. The adapter
serializes each actual action as the generic harness's `pacman.step` tool, with
`input.command = "pacman.step --action E"`, and never executes those strings
through a shell. The parser accepts only those exact four action commands from
the returned draft and expands Memorable's `repeat_count`.

The service deterministically parses recorded steps; it does not discover an
optimal strategy, prove safety, or produce a calibrated probability of success.
Our game code must check layout, position, legality and ghosts before replaying
every move. Empirical success/failure counts belong in our procedure envelope.

## Storage and recall

The public API is extraction plus query embedding. There is no documented
public `/recall` endpoint. Calling `/v1/extract` returns a draft that the caller
must store. This project stores that real draft alongside the game's validation
metadata in GBrain and retrieves it from GBrain. Do not claim the local fallback
used Memorable when the API key was absent or extraction failed.

The official CLI offers an alternative complete storage/recall loop:

```sh
npm install -g memorable-cli@0.5.30
memorable login
memorable init gbrain
memorable enable
memorable ingest game-tool-trace.json
memorable recall --single "Pac-Man mediumClassic opening"
memorable show procedures/THE-RETURNED-SLUG
```

The package requires Node 20+. The GBrain backend additionally needs GBrain and
Bun. `login` is a human authorization flow; `login --paste` can accept an
already-created dashboard key via stdin. `enable` opts into writes and extraction.
Do not run `setup` merely to initialize a game: it also edits `AGENTS.md`.
Automatic coding-session hooks are unnecessary for explicit game ingestion.

CLI recall uses exact/lexical matching before an embedding fallback. Its printed
similarity is a retrieval score, not a game's success confidence. The CLI can
return chained plans; `--single` forces individual retrieval. `list --json` is
documented, while `recall --json` is not. The CLI supports `MEMORABLE_API_URL`
and `MEMORABLE_API_KEY` together. `MEMORABLE_HOME` is a parent directory: the
0.5.30 package appends `.memorable` to it. A sandboxed CLI installation can use
`MEMORABLE_NO_KEYCHAIN=1` to avoid a global macOS keychain key.

## Failure behavior

Missing keys, network failures, malformed drafts and unknown commands raise
`MemorableError`; rejected traces raise `MemorableRefused`. HTTP 200 can still
contain `refused: "allowance_exhausted"`: do not persist its draft or retry.
No retry loop is built into this adapter. The API permits at most 2,000 calls
and 8 MB per request. It documents 300 requests/minute and 5,000 requests/day.
The adapter sends `skip_embedding: true` because GBrain owns retrieval.
It also sends an explicit application User-Agent: a live check found that
Cloudflare rejects Python urllib's default User-Agent with HTTP 403 / code 1010.

Sources verified September 27, 2026:

- [Official API contract](https://www.memorable.sh/docs/api)
- [Custom harness integration](https://www.memorable.sh/docs/integrate)
- [CLI reference](https://www.memorable.sh/docs/cli)
- [GBrain integration](https://www.memorable.sh/docs/gbrain)
- [QM integration](https://www.memorable.sh/docs/qm)
- npm package `memorable-cli@0.5.30`, its bundled README and CLI inspected directly.
