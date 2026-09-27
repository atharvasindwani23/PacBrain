"""Provider-independent memory, using pre-action game observations.

Grid rows are top-to-bottom. Explicit pacman/ghost positions must consistently
use the game's own coordinate convention; distance checks use grid cells.
"""

import hashlib
import json
import math
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

ACTIONS = {"N", "S", "E", "W"}
DELTAS = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_danger_distance(value):
    if not _finite_number(value) or value < 2:
        raise ValueError("danger_distance must be finite and at least 2")


def validate_procedure(procedure):
    """Validate a replay envelope before storage, ranking, or acting on it."""
    if not isinstance(procedure, dict) or type(procedure.get("version")) is not int or procedure["version"] != 1:
        raise ValueError("unsupported procedure version")
    for field in ("id", "topology_hash"):
        if not isinstance(procedure.get(field), str) or not re.fullmatch(r"[a-f0-9]{16}", procedure[field]):
            raise ValueError("invalid procedure " + field)
    for field in ("layout", "source_episode"):
        if not isinstance(procedure.get(field), str) or not procedure[field].strip():
            raise ValueError("invalid procedure " + field)
    _validate_danger_distance(procedure.get("danger_distance"))
    if not isinstance(procedure.get("providers"), dict) or not isinstance(procedure.get("source_result"), dict):
        raise ValueError("invalid procedure metadata")
    steps = procedure.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("procedure requires steps")
    previous = None
    reward = 0
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("action"), str) or step["action"] not in ACTIONS:
            raise ValueError("invalid procedure action")
        for field in ("position", "next_position"):
            point = step.get(field)
            if not isinstance(point, list) or len(point) != 2 or any(type(c) is not int or c < 0 for c in point):
                raise ValueError("invalid procedure " + field)
        if previous is not None and step["position"] != previous:
            raise ValueError("procedure steps are not contiguous")
        dx, dy = DELTAS[step["action"]]
        if step["next_position"] != [step["position"][0] + dx, step["position"][1] + dy]:
            raise ValueError("procedure move does not match its observed transition")
        delta = step.get("score_delta")
        if not _finite_number(delta) or delta < -1:
            raise ValueError("invalid procedure score_delta")
        reward += delta
        previous = step["next_position"]
    observed = procedure.get("observed_reward")
    if not _finite_number(observed) or observed <= 0 or not math.isclose(observed, reward):
        raise ValueError("procedure requires a consistent positive observed_reward")
    return procedure


def grid_rows(frame):
    if not isinstance(frame, dict):
        raise ValueError("observation must be an object")
    grid = frame.get("grid")
    if isinstance(grid, str):
        rows = grid.splitlines()
    elif isinstance(grid, list) and all(isinstance(row, str) for row in grid):
        rows = grid
    else:
        raise ValueError("grid must be an ASCII string or list of rows")
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("grid must be nonempty and rectangular")
    if sum(row.count("P") for row in rows) != 1:
        raise ValueError("grid must contain exactly one P")
    if any(set(row) - set("% .oPGg") for row in rows):
        raise ValueError("grid must use the documented % . o P G g glyphs")
    return rows


def topology_hash(frame):
    rows = grid_rows(frame)
    walls = "\n".join("".join("%" if c == "%" else " " for c in row) for row in rows)
    return hashlib.sha256(walls.encode()).hexdigest()[:16]


def position(frame):
    rows = grid_rows(frame)
    return next((x, y) for y, row in enumerate(rows) for x, c in enumerate(row) if c == "P")


def nearest_danger(frame):
    """Maze-path distance to a non-scared ghost; None means unreachable/absent."""
    rows = grid_rows(frame)
    start = position(frame)
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (x, y), distance = queue.popleft()
        if rows[y][x] == "G":
            return distance
        for nx, ny in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
            if (0 <= ny < len(rows) and 0 <= nx < len(rows[0])
                    and rows[ny][nx] != "%" and (nx, ny) not in seen):
                seen.add((nx, ny))
                queue.append(((nx, ny), distance + 1))
    return None


def legal_actions(frame):
    rows = grid_rows(frame)
    x, y = position(frame)
    actions = []
    for action, (dx, dy) in DELTAS.items():
        nx, ny = x + dx, y + dy
        if 0 <= ny < len(rows) and 0 <= nx < len(rows[0]) and rows[ny][nx] != "%":
            actions.append(action)
    supplied = frame.get("legal_moves")
    if supplied is not None and (not isinstance(supplied, (list, tuple, set)) or
                                 not all(isinstance(a, str) and a in ACTIONS for a in supplied)):
        raise ValueError("legal_moves must contain N/S/E/W actions")
    return [a for a in actions if supplied is None or a in supplied]


def extract_procedure(trace, max_steps=40, danger_distance=2):
    """Extract a contiguous, observed, safe opening; never promise it will win."""
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be positive")
    _validate_danger_distance(danger_distance)
    if not isinstance(trace, dict) or any(not isinstance(trace.get(k), str) or not trace[k].strip()
                                          for k in ("episode_id", "layout")):
        raise ValueError("trace needs episode_id and layout")
    if trace.get("observation_timing", "pre_action") != "pre_action":
        raise ValueError("memory requires pre_action frames; convert post-action traces first")
    if not isinstance(trace.get("result", {}), dict):
        raise ValueError("trace result must be an object")
    frames = trace.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("trace contains no frames")
    topology = topology_hash(frames[0])
    steps = []
    # A next observation is needed to confirm each executed move. The final
    # action is excluded unless a terminal observation is supplied explicitly.
    observations = frames + ([trace["terminal_observation"]] if trace.get("terminal_observation") else [])
    for index, frame in enumerate(frames[:max_steps]):
        if index + 1 >= len(observations):
            break
        following = observations[index + 1]
        if not isinstance(frame, dict) or not isinstance(following, dict):
            break
        action = frame.get("action")
        if not isinstance(action, str):
            break
        if not isinstance(frame.get("source"), str) or frame["source"] not in {"policy", "procedure"} or action not in ACTIONS:
            break
        if frame.get("action_ok") is False or frame.get("died") is True or following.get("died") is True:
            break
        if index == len(frames) - 1 and trace.get("result", {}).get("died") is True:
            break
        # A terminal renderer can replace P with a ghost. It cannot establish
        # another transition, but it does not invalidate the verified prefix.
        try:
            if topology_hash(frame) != topology or topology_hash(following) != topology:
                break
            if action not in legal_actions(frame):
                break
            danger = nearest_danger(frame)
            x, y = position(frame)
            next_position = position(following)
        except (ValueError, TypeError, KeyError):
            break
        if danger is not None and danger <= danger_distance:
            break
        dx, dy = DELTAS[action]
        if next_position != (x + dx, y + dy):
            break
        before_score, after_score = frame.get("score"), following.get("score")
        if not _finite_number(before_score) or not _finite_number(after_score):
            break
        reward = after_score - before_score
        if not math.isfinite(reward) or reward < -1:
            break
        # A verified move through an empty cell costs one point in Pac-Man.
        steps.append({"action": action, "position": [x, y], "next_position": list(next_position),
                      "score_delta": reward})
    reward = sum(step["score_delta"] for step in steps)
    while steps and reward <= 0 and steps[-1]["score_delta"] <= 0:
        reward -= steps.pop()["score_delta"]
    if not steps or reward <= 0:
        raise ValueError("no positively rewarded, safe contiguous opening in this trace")
    identity = {"episode": trace["episode_id"], "layout": trace["layout"], "steps": steps}
    pid = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    procedure = {"version": 1, "id": pid, "layout": trace["layout"], "topology_hash": topology,
            "source_episode": trace["episode_id"], "source_seed": trace.get("seed"),
            "synthetic": trace.get("synthetic") is True,
            "source_result": trace.get("result", {}), "steps": steps,
            "observed_reward": sum(s["score_delta"] for s in steps),
            "danger_distance": danger_distance, "extraction": "local_validated_trace",
            "providers": {}}
    return validate_procedure(procedure)


class ProcedureCursor:
    """One episode's replay state. Stop permanently on mismatch or danger."""

    def __init__(self, procedure):
        self.procedure = procedure
        self.index = 0
        self.stopped = False
        self.reason = "ready"

    def next_action(self, observation, layout=None):
        if self.stopped:
            return None
        reason = None
        try:
            validate_procedure(self.procedure)
            steps = self.procedure["steps"]
            if self.index >= len(steps):
                reason = "complete"
            elif layout is not None and layout != self.procedure["layout"]:
                reason = "layout_changed"
            elif topology_hash(observation) != self.procedure["topology_hash"]:
                reason = "topology_changed"
            elif list(position(observation)) != steps[self.index]["position"]:
                reason = "position_diverged"
            elif steps[self.index]["action"] not in legal_actions(observation):
                reason = "illegal_action"
            else:
                danger = nearest_danger(observation)
                if danger is not None and danger <= self.procedure["danger_distance"]:
                    reason = "ghost_nearby"
        except (ValueError, KeyError, TypeError, IndexError):
            reason = "invalid_observation_or_procedure"
        if reason:
            self.stopped = True
            self.reason = reason
            return None
        action = self.procedure["steps"][self.index]["action"]
        self.index += 1
        self.reason = "recalled_step"
        return action


class MemoryStore:
    """Local source of truth mirrored into provider stores explicitly."""

    def __init__(self, root="data/memory"):
        self.root = Path(root)

    def save(self, procedure):
        validate_procedure(procedure)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / (procedure["id"] + ".json")
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(procedure, indent=2) + "\n")
        temporary.replace(path)
        markdown = self.root / (procedure["id"] + ".md")
        markdown.write_text("# Pac-Man procedure: " + procedure["layout"] + "\n\n"
                            + "Source episode: " + str(procedure["source_episode"]) + "\n\n"
                            + "Observed opening; recheck position, legality and ghost proximity before every step.\n\n"
                            + "```json\n" + json.dumps(procedure, indent=2) + "\n```\n")
        return path

    def procedures(self):
        if not self.root.exists():
            return []
        procedures = []
        for path in sorted(self.root.glob("*.json")):
            try:
                value = json.loads(path.read_text())
                procedures.append(validate_procedure(value))
            except (ValueError, OSError, TypeError):
                continue
        return procedures

    def recall(self, layout, observation=None, allowed_ids=None):
        candidates = [p for p in self.procedures() if p["layout"] == layout
                      and (allowed_ids is None or p["id"] in allowed_ids)]
        candidates.sort(key=lambda p: (p["observed_reward"], len(p["steps"])), reverse=True)
        for procedure in candidates:
            if observation is None or ProcedureCursor(procedure).next_action(observation, layout) is not None:
                return procedure
        return None
