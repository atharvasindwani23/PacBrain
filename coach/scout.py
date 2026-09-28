"""Scout: play a game many times and turn each episode into grounded notes.

Each episode is reduced to a digest of facts the engine actually observed.
The coach model writes 1-6 notes about it, and every note must cite digest
fields as evidence; notes that cite anything else are rejected. Notes become
GBrain pages. For Pac-Man, the scout's canonical traces can also go through
Memorable to extract replayable openings (see contract.md).
"""

import json
from collections import Counter
from pathlib import Path

from memory.service import gbrain_client, ingest

from .games import load_game
from .memory_io import note_page, publish, write_pages

NOTE_KINDS = ("failure", "success", "pattern")
MAX_NOTE_CHARS = 280

SYSTEM = """You are the scout for a model that is learning to play a game.
You get the game rules and a digest of one episode the model played. Write 1-6 short notes on
what the episode shows: what ended the game or wasted moves, what earned points, and repeated habits.
Every note must cite the digest fields that support it in "evidence", using the exact field names.
Only state what the digest shows. Do not suggest fixes here.
Reply with JSON only: {"notes": [{"kind": "failure|success|pattern", "text": "...", "evidence": ["field", ...]}]}"""


def digest(game, episode):
    """Facts about one episode, straight from the engine log."""
    steps = episode["steps"]
    actions = [s["action"] for s in steps]
    illegal = sum(s["illegal_output"] for s in steps)
    facts = {
        "seed": episode["seed"],
        "policy": episode["policy"],
        "score": episode["result"]["score"],
        "won": episode["result"]["won"],
        "died": episode["result"]["died"],
        "turns": episode["result"]["turns"],
        "illegal_output_rate": round(illegal / len(steps), 3) if steps else 0.0,
        "action_counts": dict(Counter(actions)),
        "last_actions": actions[-8:],
    }
    facts.update(game.digest_extras(episode))
    return facts


def validate_notes(reply, facts):
    if not isinstance(reply, dict) or not isinstance(reply.get("notes"), list):
        raise ValueError("Scout reply must be a JSON object with a notes list")
    notes = reply["notes"]
    if not 1 <= len(notes) <= 6:
        raise ValueError(f"Scout must write 1-6 notes per episode, got {len(notes)}")
    clean = []
    for note in notes:
        if not isinstance(note, dict):
            raise ValueError("Each note must be an object")
        kind, text, evidence = note.get("kind"), note.get("text"), note.get("evidence")
        if kind not in NOTE_KINDS:
            raise ValueError(f"Note kind must be one of {NOTE_KINDS}, got {kind!r}")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_NOTE_CHARS:
            raise ValueError(f"Note text must be 1-{MAX_NOTE_CHARS} characters")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("Every note must cite at least one digest field")
        unknown = [key for key in evidence if key not in facts]
        if unknown:
            raise ValueError(f"Note cites evidence that is not in the episode digest: {unknown}")
        clean.append({"kind": kind, "text": text.strip(), "evidence": evidence,
                      "values": {key: facts[key] for key in evidence}})
    return clean


def extract_opening(trace_path):
    """Send a Pac-Man trace through Memorable; a trace with no safe rewarded opening is recorded as rejected."""
    trace = json.loads(Path(trace_path).read_text())
    try:
        receipt = ingest(trace, extractor="memorable", gbrain=gbrain_client())
    except ValueError as error:
        return {"status": "rejected", "reason": str(error)}
    return {"status": "stored", "procedure_id": receipt["procedure_id"], "steps": receipt["steps"],
            "gbrain_search_verified": receipt["gbrain_search_verified"]}


def scout(game_name, episodes, seed0, llm, endpoint=None, out_dir="data/coach",
          memory="gbrain", memorable=False, max_turns=None):
    game = load_game(game_name)
    if memorable and game_name != "pacman":
        raise ValueError("Memorable opening extraction is only defined for Pac-Man traces")
    out_dir = Path(out_dir)
    runs = game.play_episodes(list(range(seed0, seed0 + episodes)), endpoint=endpoint, label="scout",
                              max_turns=max_turns, trace_dir=out_dir / "traces" / game_name)
    pages, summary = {}, []
    for episode in runs:
        facts = digest(game, episode)
        request = json.dumps({"game": game.DESCRIPTION, "digest": facts})
        notes = validate_notes(llm.complete(SYSTEM, request), facts)
        procedure = extract_opening(episode["trace_path"]) if memorable and episode["trace_path"] else None
        pages[f"notes/{game_name}/episode-{episode['seed']}.md"] = note_page(game_name, episode, facts, notes, procedure)
        summary.append({"seed": episode["seed"], "score": facts["score"], "notes": len(notes),
                        "procedure": procedure["status"] if procedure else None})
    root = out_dir / "memory"
    written = write_pages(root, pages)
    receipt = publish(root) if memory == "gbrain" else None
    return {"game": game_name, "episodes": len(runs), "pages": written, "gbrain": receipt,
            "metrics": game.episode_metrics(runs), "per_episode": summary}
