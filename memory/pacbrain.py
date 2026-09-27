"""PacAI observations, complete episode traces, and optional guarded recall.

No PacAI dependency is imported until a live state is serialized. Legacy
prompt/completion traces can be audited, but missing scores are never inferred.
"""

import argparse
import json
import math
import os
import tempfile
import uuid
from pathlib import Path

from .core import MemoryStore, ProcedureCursor
from .gbrain import GBrainError
from .service import load_env, recall_procedure

CARDINAL = {"NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W"}


def action_code(action):
    name = str(action).strip().upper()
    return CARDINAL.get(name, name)


def parse_prompt(prompt):
    """Recover the exact grid and cardinal legal moves from serializer output."""
    if not isinstance(prompt, str):
        raise ValueError("prompt must be text")
    header, separator, remainder = prompt.partition("\n")
    grid, legal_separator, tail = remainder.rpartition("\nLegal moves: ")
    if not separator or not legal_separator or not header.startswith("You are Pacman."):
        raise ValueError("not a PacBrain serialized observation")
    rows = grid.splitlines()
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("serialized grid must be rectangular")
    if any(set(row) - set("% .oPGg") for row in rows):
        raise ValueError("serialized grid contains unsupported glyphs")
    names = [word.strip().upper() for word in tail.split("\n", 1)[0].split(",") if word.strip()]
    if any(name not in CARDINAL and name != "STOP" for name in names):
        raise ValueError("serialized legal moves contain an unknown action")
    return {"grid": grid, "legal_moves": [CARDINAL[name] for name in names if name in CARDINAL]}


def observation_from_state(state):
    """Use PacAI 2.1's authoritative score, terminal state, and board serializer."""
    from serializer import serialize

    observation = parse_prompt(serialize(state))
    score = state.score
    if type(score) not in (int, float) or not math.isfinite(score):
        raise ValueError("PacAI state score must be finite")
    terminal = bool(state.game_over)
    observation.update(score=score, terminal=terminal, cleared=state.food_count() == 0,
                       died=terminal and state.get_agent_position(0) is None)
    return observation


class EpisodeRecorder:
    """Record pre-action state, then confirm execution from PacAI action history."""

    def __init__(self, path, layout, seed=None, episode_id=None, agent="llm"):
        self.path = Path(path)
        self.trace = {"episode_id": episode_id or "pacai-" + str(seed) + "-" + uuid.uuid4().hex[:12],
                      "layout": layout, "seed": seed, "agent": agent, "synthetic": False,
                      "observation_timing": "pre_action", "frames": [],
                      "provenance": {"engine": "edq-pacai", "score_source": "state.score",
                                     "action_source": "state.get_agent_actions(0)"}}
        self._pending_count = None

    def _confirm_previous(self, state):
        if self._pending_count is None:
            return
        previous = self.trace["frames"][-1]
        actions = state.get_agent_actions(0)
        if len(actions) <= self._pending_count:
            previous["action_ok"] = False
        else:
            actual = action_code(actions[self._pending_count])
            previous["action_ok"] = actual == previous["action"]
            if not previous["action_ok"]:
                previous["requested_action"] = previous["action"]
                previous["source"] = "fallback"
            previous["action"] = actual
        self._pending_count = None

    def record(self, state, action, source="policy", decision=None):
        self._confirm_previous(state)
        frame = observation_from_state(state)
        frame.update(action=action_code(action), source=source)
        if decision:
            frame["decision"] = dict(decision)
        self.trace["frames"].append(frame)
        self._pending_count = len(state.get_agent_actions(0))

    def finish(self, state):
        self._confirm_previous(state)
        terminal = observation_from_state(state)
        self.trace["terminal_observation"] = terminal
        frames = self.trace["frames"]
        self.trace["result"] = {"score": terminal["score"], "cleared": terminal["cleared"],
                                "died": terminal["died"], "terminal": terminal["terminal"],
                                "turns": sum(f.get("action_ok") is True for f in frames),
                                "engine_turns": state.turn_count,
                                "llm_calls": sum(f.get("decision", {}).get("policy_called") is True for f in frames),
                                "procedure_steps": sum(f["source"] == "procedure" and f.get("action_ok") is True for f in frames)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        staged = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(self.path.parent),
                                             prefix="." + self.path.name + ".", suffix=".tmp", delete=False) as stream:
                staged = Path(stream.name)
                json.dump(self.trace, stream, indent=2)
                stream.write("\n")
            os.replace(str(staged), str(self.path))
        finally:
            if staged is not None and staged.exists():
                staged.unlink()
        return self.trace


def recorder_from_env(agent="llm"):
    path = os.getenv("PACBRAIN_TRACE")
    if not path:
        return None
    seed = os.getenv("PACBRAIN_SEED")
    return EpisodeRecorder(path, os.getenv("PACBRAIN_BOARD", "classic-small"),
                           seed=int(seed) if seed is not None else None, agent=agent)


class MemoryPolicy:
    """Recall once per episode; inspect the actual state before every replay move."""

    def __init__(self, layout, provider="local", memory_dir="data/memory"):
        if provider not in ("local", "gbrain"):
            raise ValueError("memory provider must be local or gbrain")
        self.layout, self.provider = layout, provider
        self.store = MemoryStore(memory_dir)
        self.cursor, self._queried = None, False
        self.reason, self.error = "ready", None

    def choose(self, state):
        observation = observation_from_state(state)
        if observation["terminal"] or observation["cleared"] or observation["died"]:
            self.reason = "episode_finished"
            return None
        if not self._queried:
            self._queried = True
            try:
                hit = recall_procedure(self.layout, observation, store=self.store, provider=self.provider)
                if hit["procedure"]:
                    self.cursor = ProcedureCursor(hit["procedure"])
                else:
                    self.reason = "no_matching_procedure"
            except (GBrainError, OSError, ValueError) as error:
                self.reason, self.error = "memory_unavailable", type(error).__name__
        if not self.cursor:
            return None
        code = self.cursor.next_action(observation, self.layout)
        self.reason = self.cursor.reason
        if code is None:
            return None
        legal = {action_code(action): action for action in state.get_legal_actions()}
        if code not in legal:
            self.cursor.stopped = True
            self.cursor.reason = self.reason = "illegal_action"
            return None
        return legal[code]

    def decision(self):
        return {"memory_provider": self.provider, "fallback_reason": self.reason,
                "procedure_id": self.cursor.procedure["id"] if self.cursor else None,
                "procedure_step": self.cursor.index if self.cursor else None,
                "memory_error": self.error}


def memory_from_env():
    provider = os.getenv("PACBRAIN_MEMORY", "none").lower()
    if provider == "none":
        return None
    load_env()
    return MemoryPolicy(os.getenv("PACBRAIN_BOARD", "classic-small"), provider,
                        os.getenv("PACBRAIN_MEMORY_DIR", "data/memory"))


def audit_legacy_trace(trace):
    """Report conversion blockers without inventing intermediate reward values."""
    if not isinstance(trace, dict):
        raise ValueError("legacy trace must be an object")
    steps = trace.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("legacy trace must contain nonempty steps")
    for step in steps:
        parse_prompt(step.get("prompt"))
        action = str(step.get("completion", "")).strip().upper()
        if action not in CARDINAL and action != "STOP":
            raise ValueError("legacy completion is not one executed move")
    missing = sum("score" not in step for step in steps)
    return {"board": trace.get("board"), "seed": trace.get("seed"), "steps": len(steps),
            "missing_step_scores": missing, "replayable": False,
            "blocker": "Legacy traces lack authoritative per-action scores and a final observation. Record a new game with PACBRAIN_TRACE."}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit original PacBrain traces; never infer missing observed scores.")
    parser.add_argument("path", help="One trace JSON or directory of trace JSON files")
    args = parser.parse_args(argv)
    path = Path(args.path)
    paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
    reports = []
    for source in paths:
        try:
            report = audit_legacy_trace(json.loads(source.read_text()))
            reports.append(dict(path=str(source), **report))
        except (OSError, ValueError, TypeError) as error:
            reports.append({"path": str(source), "replayable": False, "error": str(error)})
    print(json.dumps({"files": len(paths), "replayable": sum(r["replayable"] for r in reports),
                      "missing_score_files": sum(r.get("missing_step_scores", 0) > 0 for r in reports),
                      "reports": reports}, indent=2))
    return 0 if reports else 2


if __name__ == "__main__":
    raise SystemExit(main())
