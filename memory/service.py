"""Explicit orchestration: Memorable extracts, GBrain indexes and retrieves."""

import json
import os
import re
from pathlib import Path

from .core import MemoryStore, extract_procedure, validate_procedure
from .gbrain import GBrainClient, GBrainError
from .gbrain_http import GBrainHTTPClient
from .memorable import MemorableClient, MemorableError, game_trace_to_memorable, moves_from_draft


def load_env(path=".env"):
    """Minimal literal KEY=value loader; never executes shell interpolation."""
    file = Path(path)
    if not file.is_file():
        return
    for line in file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key.strip()):
            raise ValueError("Environment file must contain literal KEY=value lines")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            os.environ.setdefault(key.strip(), value)


def gbrain_client():
    """Prefer the authenticated hosted connection when fully configured."""
    transport = (os.getenv("GBRAIN_TRANSPORT") or "auto").strip().lower()
    if transport not in ("auto", "local", "http"):
        raise GBrainError("GBRAIN_TRANSPORT must be auto, local, or http.")
    if transport != "local":
        url = (os.getenv("GBRAIN_MCP_URL") or "").strip()
        token = os.getenv("GBRAIN_ACCESS_TOKEN") or ""
        if bool(url) != bool(token) or (transport == "http" and not url):
            raise GBrainError("Hosted GBrain requires both GBRAIN_MCP_URL and GBRAIN_ACCESS_TOKEN. "
                              "Set both, or choose GBRAIN_TRANSPORT=local explicitly.")
        if url and token:
            return GBrainHTTPClient(url, token)
    launcher = Path(".runtime/bin/gbrain")
    return GBrainClient(Path(os.getenv("GBRAIN_HOME") or "data/gbrain-home"),
                        binary=os.getenv("GBRAIN_BIN") or (str(launcher.resolve()) if launcher.exists() else "gbrain"))


def ingest(trace, store=None, extractor="memorable", memorable=None, gbrain=None, max_steps=40):
    store = store or MemoryStore()
    procedure = extract_procedure(trace, max_steps=max_steps)
    if extractor == "memorable":
        # Only send the contiguous, independently validated opening, not a
        # death sequence or later unrelated successful moves.
        selected = dict(trace, frames=trace["frames"][:len(procedure["steps"])])
        wire = game_trace_to_memorable(selected, max_steps=max_steps)
        response = (memorable or MemorableClient.from_env()).extract_trace(wire)
        moves = moves_from_draft(response["draft"])
        expected = [step["action"] for step in procedure["steps"]]
        if moves != expected:
            raise MemorableError("Memorable draft differs from the observed safe opening; refusing unverified replay")
        procedure["extraction"] = "memorable"
        procedure["providers"]["memorable"] = response
    elif extractor != "local":
        raise ValueError("extractor must be memorable or local")
    # Save before indexing, so a temporary GBrain outage cannot lose the actual
    # extraction. Index errors propagate; they are never reported as success.
    path = store.save(procedure)
    indexed = gbrain.import_directory(store.root) if gbrain else None
    return {"procedure_id": procedure["id"], "path": str(path), "steps": len(procedure["steps"]),
            "extraction": procedure["extraction"], "gbrain_indexed": indexed is not None,
            "gbrain_import": indexed}


def recall_procedure(layout, observation=None, store=None, provider="gbrain", gbrain=None):
    store = store or MemoryStore()
    if provider == "local":
        procedure = store.recall(layout, observation)
        return {"provider": "local", "procedure": procedure, "matches": []}
    if provider != "gbrain":
        raise ValueError("provider must be gbrain or local")
    client = gbrain or gbrain_client()
    rows = client.search("Pac-Man " + layout, limit=20)
    # Only procedures explicitly returned by real GBrain retrieval are eligible.
    # No silent local fallback if search finds nothing or fails.
    ids = set()
    # A different harness may have only the brain, not this checkout's JSON.
    # Hydrate canonical pages, never execute text from recalled memory.
    fetched_slugs = set()
    for row in rows:
        slug = row.get("slug")
        if not isinstance(slug, str) or slug in fetched_slugs:
            continue
        matching = set(re.findall(r"(?<![a-f0-9])[a-f0-9]{16}(?![a-f0-9])", slug))
        if not matching:
            continue
        fetched_slugs.add(slug)
        page = client.get_page(slug)
        content = page.get("content", "")
        if not isinstance(content, str):
            continue
        for block in re.findall(r"```json\s*\n(.*?)\n```", content, flags=re.DOTALL):
            try:
                recovered = validate_procedure(json.loads(block))
            except (ValueError, TypeError):
                continue
            if recovered["id"] in matching and recovered["layout"] == layout and recovered["id"] not in ids:
                # Immutable hosted versions can share one procedure id. Keep
                # the first validated version in search rank order, including
                # its provider receipt; later versions must not overwrite it.
                store.save(recovered)
                ids.add(recovered["id"])
    procedure = store.recall(layout, observation, allowed_ids=ids)
    return {"provider": "gbrain", "procedure": procedure, "matches": rows}
