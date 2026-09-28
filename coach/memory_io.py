"""Coach pages in GBrain: scout notes and reward/eval designs.

Pages are body-only Markdown (the format both GBrain clients import) with one
machine-readable JSON block, so the designer reads back exactly what the scout
wrote. They share the app's pacman-memory namespace with Memorable procedures.
"""

import json
import re
from pathlib import Path

from memory.service import gbrain_client

BLOCK = re.compile(r"```json coach\n(.*?)\n```", re.S)


def _block(payload):
    return "```json coach\n" + json.dumps(payload, indent=1, sort_keys=True) + "\n```\n"


def note_page(game, episode, facts, notes, procedure=None):
    result = episode["result"]
    lines = [
        f"# Coach scout notes: {game} episode {episode['seed']}",
        "",
        f"Policy `{episode['policy']}` · score {result['score']} · turns {result['turns']} · "
        f"{'won' if result['won'] else 'caught' if result['died'] and game == 'pacman' else 'crashed' if result['died'] else 'survived'}",
        "",
        "## Notes",
    ]
    for i, note in enumerate(notes):
        evidence = ", ".join(f"{key}={note['values'][key]}" for key in note["evidence"])
        lines.append(f"- **{note['kind']}** ({episode['seed']}-{i}): {note['text']} _Evidence: {evidence}_")
    if procedure:
        lines += ["", "## Memorable opening", f"`{json.dumps(procedure, sort_keys=True)}`"]
    payload = {"kind": "scout-notes", "game": game, "seed": episode["seed"], "digest": facts,
               "notes": notes, "procedure": procedure}
    return "\n".join(lines) + "\n\n" + _block(payload)


def design_page(design):
    reward = design["reward"]
    lines = [
        f"# Coach reward design: {design['game']}",
        "",
        f"Designed from {design['notes_used']} scout notes across {design['episodes']} episodes.",
        "",
        "## Reward",
        f"reward = clip(sum(weight × primitive) / {reward['scale']}, ±{reward['clip']}); "
        f"illegal or unparseable output = {reward['illegal_output']}",
        "",
    ]
    lines += [f"- `{t['name']}` × {t['weight']}: {t['why']}" for t in reward["terms"]]
    lines += ["", "## Evals"]
    lines += [f"- `{e['metric']}` {'≥' if e['direction'] == 'max' else '≤'} {e['target']}: {e['why']}"
              for e in design["evals"]]
    lines += ["", "## Rationale", design["rationale"] or "(none given)"]
    return "\n".join(lines) + "\n\n" + _block(dict(design, kind="design"))


def write_pages(root, pages):
    root = Path(root)
    written = []
    for relative, text in pages.items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9/_-]*\.md", relative):
            raise ValueError(f"Memory page paths must be lowercase slugs ending in .md: {relative!r}")
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(str(path))
    return written


def publish(root):
    """Import the coach's Markdown pages into GBrain (hosted MCP or local CLI)."""
    return gbrain_client().import_directory(Path(root))


def parse_page(content):
    match = BLOCK.search(content)
    if not match:
        raise ValueError("Memory page has no coach JSON block")
    return json.loads(match.group(1))


def recall_notes(game, limit=100):
    """Scout notes for a game, read back from GBrain."""
    client = gbrain_client()
    rows = client.search(f"coach scout notes {game}", limit=limit)
    pages = []
    for row in rows:
        slug = row.get("slug")
        if not isinstance(slug, str) or f"notes/{game}/" not in slug:
            continue
        payload = parse_page(client.get_page(slug)["content"])
        if payload.get("kind") == "scout-notes" and payload.get("game") == game:
            pages.append(payload)
    if not pages:
        raise LookupError(f"GBrain has no coach scout notes for {game}; run `python -m coach scout --game {game}` first")
    return pages


def load_local_notes(root, game):
    """Scout notes from the local Markdown directory, for runs without GBrain."""
    paths = sorted((Path(root) / "notes" / game).glob("*.md"))
    if not paths:
        raise LookupError(f"No local scout notes under {Path(root) / 'notes' / game}")
    return [parse_page(path.read_text(encoding="utf-8")) for path in paths]
