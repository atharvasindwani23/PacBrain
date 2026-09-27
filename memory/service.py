"""Explicit orchestration: Memorable extracts, GBrain indexes and retrieves."""

import json
import os
import re
from pathlib import Path

from .core import MemoryStore, extract_procedure, validate_procedure
from .gbrain import GBrainClient, GBrainError
from .gbrain_http import GBrainHTTPClient
from .memorable import (MemorableClient, MemorableError, game_trace_to_memorable,
                        moves_from_draft, validate_extraction)


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
        response = validate_extraction((memorable or MemorableClient.from_env()).extract_trace(wire))
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
    validate_recalled_procedure(procedure)
    path = store.save(procedure)
    imported = gbrain.import_directory(store.root) if gbrain else None
    search_verified = False
    if gbrain:
        rows = gbrain.search("Pac-Man " + procedure["layout"], limit=20)
        search_verified = any(procedure["id"] in re.findall(r"(?<![a-f0-9])[a-f0-9]{16}(?![a-f0-9])",
                                                             row.get("slug", ""))
                              for row in rows if isinstance(row.get("slug"), str))
    return {"procedure_id": procedure["id"], "path": str(path), "steps": len(procedure["steps"]),
            "extraction": procedure["extraction"], "gbrain_stored": imported is not None,
            "gbrain_search_verified": search_verified,
            # Kept for existing callers; it now means observed search visibility.
            "gbrain_indexed": search_verified, "gbrain_import": imported}


def validate_recalled_procedure(procedure):
    """Require a claimed Memorable extraction to retain matching evidence."""
    validate_procedure(procedure)
    method = procedure.get("extraction")
    if method == "local_validated_trace":
        if "memorable" in procedure["providers"]:
            raise ValueError("Local procedure has an inconsistent Memorable provider receipt")
        return procedure
    if method != "memorable":
        raise ValueError("Procedure is missing a supported extraction method")
    if "memorable" not in procedure["providers"]:
        raise ValueError("Memorable procedure is missing its provider receipt")
    receipt = procedure["providers"]["memorable"]
    try:
        response = validate_extraction(receipt)
        if not isinstance(response["request_id"], str) or not response["request_id"].strip():
            raise ValueError("Memorable receipt is missing its request_id")
        draft = response["draft"]
        if "session_id" in draft and draft["session_id"] != procedure["source_episode"]:
            raise ValueError("Memorable receipt belongs to another source episode")
        if moves_from_draft(draft) != [step["action"] for step in procedure["steps"]]:
            raise ValueError("Memorable receipt does not match the stored procedure actions")
    except MemorableError as error:
        raise ValueError("Memorable procedure has a malformed or refused provider receipt") from error
    return procedure


def recall_procedure(layout, observation=None, store=None, provider="gbrain", gbrain=None):
    store = store or MemoryStore()
    if provider == "local":
        procedure = _recall_validated(store, layout, observation)
        return {"provider": "local", "procedure": procedure, "matches": []}
    if provider != "gbrain":
        raise ValueError("provider must be gbrain or local")
    client = gbrain or gbrain_client()
    rows = client.search("Pac-Man " + layout, limit=20)
    # Only procedures explicitly returned by real GBrain retrieval are eligible.
    # No silent local fallback if search finds nothing or fails.
    selected = {}
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
                recovered = validate_recalled_procedure(json.loads(block))
            except (ValueError, TypeError):
                continue
            if recovered["id"] in matching and recovered["layout"] == layout:
                previous = selected.get(recovered["id"])
                # Prefer verified provider evidence over a local extraction of
                # the same opening; preserve search rank within each method.
                if previous is None or (previous["extraction"] == "local_validated_trace"
                                        and recovered["extraction"] == "memorable"):
                    selected[recovered["id"]] = recovered
    # Publish only the selected versions, without temporarily downgrading an
    # existing cached provider receipt while walking ranked search results.
    for recovered in selected.values():
        store.save(recovered)
    procedure = _recall_validated(store, layout, observation, set(selected), list(selected))
    return {"provider": "gbrain", "procedure": procedure, "matches": rows}


def _recall_validated(store, layout, observation, allowed_ids=None, rank_order=None):
    """Apply the same provenance contract to cached and newly hydrated data."""
    eligible = set(allowed_ids) if allowed_ids is not None else {p["id"] for p in store.procedures()}
    while eligible:
        procedure = store.recall(layout, observation, allowed_ids=eligible, rank_order=rank_order)
        if procedure is None:
            return None
        try:
            return validate_recalled_procedure(procedure)
        except ValueError:
            eligible.discard(procedure["id"])
    return None
