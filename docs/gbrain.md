# GBrain memory integration

This integration uses the real [garrytan/gbrain](https://github.com/garrytan/gbrain) CLI. The project stores game procedures as Markdown, imports them into GBrain, and uses GBrain's ranked search results to retrieve them. API credentials are unnecessary for this keyless keyword mode. A local fallback must be labeled separately from a successful GBrain read.

## Setup

```bash
bash scripts/setup_gbrain.sh
export GBRAIN_BIN="$PWD/.runtime/bin/gbrain"
export GBRAIN_HOME="$PWD/data/gbrain-home"
```

The installer stays inside the checkout. It verifies the Bun 1.3.11 download checksum, installs GBrain commit `e78f1c38b947b053f3a46881340f74f316be855a` (version `0.59.0.0`), disables dependency lifecycle scripts, and initializes an isolated PGLite database. It does not register an MCP server or edit shell profiles. `GBRAIN_HOME` is the **parent** of `.gbrain`.

The initialization command is:

```bash
"$GBRAIN_BIN" init --pglite --no-embedding --non-interactive --db-only
```

`--db-only` avoids claiming this application Git repository as GBrain's own content repository. Our application retains its Markdown procedures separately. GBrain is not the unrelated npm package named `gbrain`; use the GitHub source installation above. See the official [installation guidance](https://github.com/garrytan/gbrain/blob/master/README.md#install).

## Python interface and observed responses

```python
from pathlib import Path
from memory.gbrain import GBrainClient

brain = GBrainClient(
    Path("data/gbrain-home"),
    binary=str(Path(".runtime/bin/gbrain").resolve()),
)
summary = brain.import_directory(Path("data/memory"))
rows = brain.search("pacman corridor ghost escape", limit=5)
if rows:
    page = brain.get_page(rows[0]["slug"])
    markdown = page["content"]
```

`available()` checks binary/config presence, not database health. Calls raise `GBrainError` on missing setup, command failure, timeout, or an unexpected response; they do not report fabricated sponsor success. Errors omit raw subprocess output to avoid leaking credentials. The adapter clears inherited database/source selectors and runs from its isolated home.

Live verification on 2026-09-27 observed:

```json
{"status":"success","imported":1,"skipped":0,"errors":0,"chunks":2,"source_id":"default"}
```

Search returns an **array**, with these useful fields:

```json
[{"slug":"procedures/8cf750eaafe10001","chunk_text":"...fenced JSON...","score":1,"source_id":"default","evidence":"keyword_exact","id":"gbrain-page:v1:..."}]
```

Use `slug` to retrieve a page. Its `id` is an opaque source-qualified identifier, not a file path. `get_page()` returns canonical Markdown in `content`, plus `compiled_truth`, `frontmatter`, and `source_path`. Parse and validate the fenced procedure JSON as application data before choosing an action. [Page contract](https://github.com/garrytan/gbrain/blob/master/src/core/ops/pages.ts), [search contract](https://github.com/garrytan/gbrain/blob/master/src/core/ops/search.ts).

Imports run with `--no-embed --include-gitignored --fresh --json`: generated runtime files may be ignored by Git, and previously imported files must be revisited after an update. No model API call is needed. [Import implementation](https://github.com/garrytan/gbrain/blob/master/src/commands/import.ts).

## Team and QM access

PGLite is a local, single-process database. Serialize local CLI calls. A running `gbrain serve` or Memorable viewer may hold its lock; do not delete a live lock. Other machines cannot reach a laptop's `localhost`.

For a hosted QM worker, run an authenticated GBrain HTTP server behind HTTPS, or deploy that server next to the worker. Create the scoped token **before** starting a PGLite server:

```bash
"$GBRAIN_BIN" auth create qm-pacman --scopes read,write
"$GBRAIN_BIN" serve --http --port 3131 --public-url https://YOUR_BRAIN_HOST
```

The MCP endpoint is `https://YOUR_BRAIN_HOST/mcp`. Supply the generated token through the worker's secret configuration. These commands need a real HTTPS tunnel or host; the setup script does not publish a server. Local CLI imports must stop while the server holds the same PGLite database, or be redesigned to use MCP writes. [Deployment and token guidance](https://github.com/garrytan/gbrain/blob/master/docs/mcp/DEPLOY.md).

To move indexed Markdown explicitly:

```bash
"$GBRAIN_BIN" export --dir "$PWD/data/gbrain-export"
```

Markdown export is not a full database backup. [Export interface](https://github.com/garrytan/gbrain/blob/master/src/commands/export.ts).

## Validation

`python3 -m unittest discover -s tests -p test_gbrain.py -v` covers subprocess isolation, real response shapes, partial failure, timeout, and secret-safe diagnostics. A real import → keyword search → canonical-page retrieval round trip also passed against the pinned installation. This verifies persistence and retrieval, not improved gameplay; the game evaluator must measure that separately.
