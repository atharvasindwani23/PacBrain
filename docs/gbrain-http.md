# Hosted GBrain transport

`memory.gbrain_http.GBrainHTTPClient` connects directly to an authenticated HTTPS MCP endpoint. A cloud worker needs outbound HTTPS and these secrets, with no laptop process or tunnel:

```text
GBRAIN_MCP_URL=https://gbrain.io/mcp
GBRAIN_ACCESS_TOKEN=your_actual_unmasked_token
```

Store the actual token in the worker's private secret configuration or this checkout's ignored `.env`. On GBrain, grant the client access to the intended workspace's Memory application at **Full** level. [Hosted setup and permissions](https://gbrain.io/docs/workspace/memory-anywhere).

```python
import os
from pathlib import Path
from memory.gbrain_http import GBrainHTTPClient
from memory.service import load_env

load_env()
brain = GBrainHTTPClient(os.environ["GBRAIN_MCP_URL"], os.environ["GBRAIN_ACCESS_TOKEN"])
brain.import_directory(Path("data/memory"))
rows = brain.search("Pac-Man testCorridor")
if rows:
    page = brain.get_page(rows[0]["slug"])
```

The class has the same `available`, `import_directory`, `search`, and `get_page` methods as the local CLI adapter. `available()` checks configuration only. `discover_tools()` performs authenticated initialization and returns tool names/input schemas. Provider failures raise `GBrainError`; they are never silently converted into local-memory success.

## Observed hosted behavior

On September 27, 2026, the approved GBrain connection exposed `search`, `get_page` with `include_content`, and `put_page` with `slug` and `content`. Its `put_page` schema did **not** expose revision guards or idempotency identifiers. The adapter therefore publishes immutable, content-versioned pages under `pacman-memory/<procedure-id>-v-<payload-hash>`. An identical import is skipped after checking the stored fingerprint. A changed payload receives a new slug; no existing page is overwritten. Unrelated workspace pages are never hydrated.

A real remote test wrote one explicitly synthetic Pac-Man procedure (`44d62747020b6464`), found it through hosted search, and read its canonical Markdown back. A second import reported `imported: 0, skipped: 1`. A fresh HTTP client with an empty local store hydrated that remote procedure and returned the first guarded action, `E`. This verifies remote persistence/retrieval and network access. It is not a real gameplay or QM worker run, and does not establish improved scores.

The host wraps missing-page failures as `structuredContent.what = memory_refused`, with the engine's JSON error inside `why`; the adapter understands that wrapper without exposing raw diagnostic bodies. Successful search returns ranked rows including `slug`, `chunk_text`, and `score`; `get_page` returns canonical `content` plus frontmatter.

## Transport boundaries

The client implements JSON-RPC initialization, tool discovery, JSON and SSE response parsing, negotiated protocol headers, and MCP session headers using Python 3.9's standard library. It accepts HTTPS only and refuses redirects, so bearer credentials cannot be forwarded to a different endpoint. Responses are bounded to 8 MB. A session-expiry error requires a fresh client; writes are verified by readback before import reports success. [MCP Streamable HTTP specification](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports).

Run the transport checks with `python3 -m unittest tests.test_gbrain_http -v`. Hosted credentials are not needed for those tests.
