"""Designer: read the scout's notes back from memory and design the training signal.

The coach model may only combine the game's declared reward primitives and
eval metrics. Its reply is validated before anything is written, so a design
can never introduce code, unknown terms, or non-finite weights.
"""

import json
import re
from pathlib import Path

from .evals import validate_eval_suite
from .games import load_game
from .memory_io import design_page, write_pages
from .rewards import validate_reward_spec

MAX_NOTES_IN_PROMPT = 150

SYSTEM = """You design the training signal for a model that learns a game with reinforcement learning (GRPO).
You get the game rules, a fixed menu of reward primitives and eval metrics, aggregate stats, and the
scout's notes from many episodes, each with an id like "5003-1".

Reward: a weighted sum of primitives, divided by scale and clipped to [-clip, clip], plus a fixed
negative reward for model output that is not a legal move. Weight the failures the notes show most
often. Keep the engine's own score signal unless the notes give a reason not to. Unparseable output
should score worse than an ordinary move but not worse than the game's worst outcome.

Evals: pick 2-6 metrics with targets that would show the trained model fixed what the notes describe.

In every "why", cite the note ids you relied on.
Reply with JSON only:
{"reward": {"terms": [{"name": "<primitive>", "weight": <number>, "why": "..."}],
            "scale": <number>, "clip": <number>, "illegal_output": <number>},
 "evals": [{"metric": "<metric>", "target": <number>, "why": "..."}],
 "rationale": "<2-4 sentences>"}"""


def _flatten(pages):
    notes = []
    for page in pages:
        for i, note in enumerate(page["notes"]):
            notes.append({"id": f"{page['seed']}-{i}", "kind": note["kind"], "text": note["text"],
                          "evidence": note["values"]})
    return notes


def _stats(pages):
    digests = [page["digest"] for page in pages]
    n = len(digests)
    return {
        "episodes": n,
        "mean_score": sum(d["score"] for d in digests) / n,
        "died_rate": sum(bool(d["died"]) for d in digests) / n,
        "mean_turns": sum(d["turns"] for d in digests) / n,
        "mean_illegal_output_rate": sum(d["illegal_output_rate"] for d in digests) / n,
        "failure_notes": sum(note["kind"] == "failure" for page in pages for note in page["notes"]),
    }


def design(game_name, pages, llm):
    if not pages:
        raise ValueError("The designer needs scout notes; none were supplied")
    game = load_game(game_name)
    notes = _flatten(pages)
    request = {"game": game.DESCRIPTION, "actions": list(game.ACTIONS),
               "reward_primitives": game.REWARD_PRIMITIVES, "eval_metrics": game.EVAL_METRICS,
               "stats": _stats(pages), "notes": notes[:MAX_NOTES_IN_PROMPT]}
    reply = llm.complete(SYSTEM, json.dumps(request))
    if not isinstance(reply, dict):
        raise ValueError("Designer reply must be a JSON object")
    reward = validate_reward_spec(reply.get("reward"), game)
    evals = validate_eval_suite(reply.get("evals"), game)
    known = {note["id"] for note in notes}
    reasons = " ".join(item["why"] for item in reward["terms"] + evals)
    cited = sorted(set(re.findall(r"\b\d+-\d+\b", reasons)) & known)
    if not cited:
        raise ValueError("The design does not cite any scout note; refusing an ungrounded reward")
    return {"game": game_name, "reward": reward, "evals": evals,
            "rationale": str(reply.get("rationale", ""))[:2000],
            "episodes": len(pages), "notes_used": min(len(notes), MAX_NOTES_IN_PROMPT),
            "cited_notes": cited, "coach_model": getattr(llm, "model", None)}


def write_design(result, out_dir, memory_root=None):
    """Write reward_spec.json and eval_suite.json for training, plus a design page for memory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = {"game": result["game"], "version": 1, **result["reward"],
            "source": f"coach design from {result['episodes']} scouted episodes",
            "cited_notes": result["cited_notes"]}
    suite = {"game": result["game"], "version": 1, "evals": result["evals"]}
    (out_dir / "reward_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    (out_dir / "eval_suite.json").write_text(json.dumps(suite, indent=2) + "\n")
    page = design_page(result)
    (out_dir / "design.md").write_text(page)
    if memory_root:
        write_pages(memory_root, {f"designs/{result['game']}/latest.md": page})
    return {"reward_spec": str(out_dir / "reward_spec.json"), "eval_suite": str(out_dir / "eval_suite.json"),
            "design": str(out_dir / "design.md")}
