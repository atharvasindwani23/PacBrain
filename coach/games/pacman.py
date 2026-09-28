"""Pac-Man (UC Berkeley CS188 rules via edq-pacai) as a coach game.

Gameplay runs in the pacai engine as a subprocess, the same way eval.py and
collect_states.py run it. coach/scout_agent.py logs every decision so the
scout can write notes grounded in what actually happened.
"""

import concurrent.futures
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pacai.pacman.board

from rewards import bfs_dist

NAME = "pacman"
BOARD = "classic-small"
ACTIONS = ("NORTH", "SOUTH", "EAST", "WEST", "STOP")
ROOT = Path(__file__).resolve().parents[2]
DESCRIPTION = (
    "Pac-Man on the classic-small maze (UC Berkeley CS188 rules). Every move costs 1 point, "
    "a pellet is +10, eating a scared ghost is +200, clearing the board is +500 and being "
    "caught by an active ghost is -500 and ends the game. The model sees an ASCII grid "
    "(% wall, . pellet, o power pellet, G ghost, g scared ghost, P Pac-Man) and must answer "
    "with exactly one legal move word: north, south, east, west or stop."
)

REWARD_PRIMITIVES = {
    "score_delta": {"description": "Engine score change for the move: -1 per move, +10 pellet, "
                                   "+200 scared ghost, +500 board clear, -500 caught.",
                    "range": [-501, 509]},
    "food_progress": {"description": "Maze steps closer to the nearest pellet (+1 closer, -1 farther).",
                      "range": [-1, 1]},
    "ghost_adjacent": {"description": "1 if the move ends within 1 maze step of an active ghost.",
                       "range": [0, 1]},
    "ghost_near": {"description": "1 if the move ends exactly 2 maze steps from an active ghost.",
                   "range": [0, 1]},
    "capsule_progress": {"description": "Maze steps closer to the nearest power pellet.",
                         "range": [-1, 1]},
    "scared_ghost_progress": {"description": "Maze steps closer to the nearest scared ghost.",
                              "range": [-1, 1]},
    "stop": {"description": "1 if the move is STOP (standing still).", "range": [0, 1]},
}

EVAL_METRICS = {
    "avg_score": {"description": "Mean final game score.", "direction": "max"},
    "win_rate": {"description": "Share of games where the board was cleared.", "direction": "max"},
    "illegal_move_rate": {"description": "Share of model outputs that were not a legal move word.",
                          "direction": "min"},
    "pellets_per_game": {"description": "Mean pellets eaten per game.", "direction": "max"},
    "death_rate": {"description": "Share of games that ended with Pac-Man caught.", "direction": "min"},
    "avg_survival_moves": {"description": "Mean moves made per game.", "direction": "max"},
    "ghosts_eaten_per_game": {"description": "Mean scared ghosts eaten per game.", "direction": "max"},
}

OPPOSITE = {"NORTH": "SOUTH", "SOUTH": "NORTH", "EAST": "WEST", "WEST": "EAST"}


def legal_actions(state):
    return list(state.get_legal_actions())


def _progress(board_before, pos_before, targets_before, board_after, pos_after, targets_after):
    before = bfs_dist(board_before, pos_before, targets_before)
    after = bfs_dist(board_after, pos_after, targets_after)
    if before is None or after is None:
        return 0.0
    return float(before - after)


def primitive_values(state, action):
    """Raw value of every reward primitive for taking `action` in `state`."""
    pac = state.get_agent_position(0)
    succ = state.generate_successor(action)
    npos = pac.apply_action(action)
    capsule = pacai.pacman.board.MARKER_CAPSULE
    ghost_dist = bfs_dist(state.board, npos, list(state.get_nonscared_ghost_positions().values()))
    return {
        "score_delta": float(succ.score - state.score),
        "food_progress": _progress(state.board, pac, set(state.get_food()),
                                   succ.board, npos, set(succ.get_food())),
        "ghost_adjacent": 1.0 if ghost_dist is not None and ghost_dist <= 1 else 0.0,
        "ghost_near": 1.0 if ghost_dist == 2 else 0.0,
        "capsule_progress": _progress(state.board, pac, set(state.board.get_marker_positions(capsule)),
                                      succ.board, npos, set(succ.board.get_marker_positions(capsule))),
        "scared_ghost_progress": _progress(state.board, pac, set(state.get_scared_ghost_positions().values()),
                                           succ.board, npos, set(succ.get_scared_ghost_positions().values())),
        "stop": 1.0 if str(action).upper() == "STOP" else 0.0,
    }


def step_events(delta, cleared):
    """Decode one move's score change into game events (same rules as eval.decode_events)."""
    events = {"pellets": 0, "ghosts_eaten": 0, "caught": False, "cleared": False}
    if delta <= -400:
        events["caught"] = True
        delta += 500
    if cleared and delta >= 400:
        events["cleared"] = True
        delta -= 500
    if delta >= 190:
        events["ghosts_eaten"] = (delta + 1) // 200
        delta = (delta + 1) % 200 - 1
    if delta > 0:
        events["pellets"] = (delta + 1) // 10
    return events


def _play_one(job):
    seed, endpoint, label, max_turns, trace_dir = job
    step_log = Path(trace_dir) / f"{label}_steps_{seed}.jsonl"
    trace_path = Path(trace_dir) / f"{label}_seed{seed}.json"
    for stale in (step_log, trace_path):
        stale.unlink(missing_ok=True)
    env = dict(os.environ, COACH_STEP_LOG=str(step_log), PACBRAIN_TRACE=str(trace_path),
               PACBRAIN_BOARD=BOARD, PACBRAIN_SEED=str(seed))
    env.pop("PACBRAIN_ENDPOINT", None)
    if endpoint:
        env["PACBRAIN_ENDPOINT"] = endpoint
    cmd = [sys.executable, "-m", "pacai.pacman", "--ui", "null", "--board", BOARD,
           "--pacman", "coach/scout_agent.py:ScoutAgent", "--seed", str(seed),
           "--max-turns", str(max_turns)]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT, timeout=1800)
    text = out.stdout + out.stderr
    if out.returncode != 0 or re.search(r"Agent \d+ crashed", text):
        raise RuntimeError(f"Pac-Man scout game {seed} failed:\n{text[-2000:]}")
    match = re.search(r"Scores:\s+(-?\d+)", text)
    if not match:
        raise RuntimeError(f"Pac-Man scout game {seed} finished without printing a score")
    score = int(match.group(1))
    won = bool(re.search(r"Record:\s+Win", text))
    rows = [json.loads(line) for line in step_log.read_text().splitlines() if line.strip()]
    steps = []
    for i, row in enumerate(rows):
        last = i + 1 == len(rows)
        after = score if last else rows[i + 1]["score"]
        steps.append(dict(row, events=step_events(int(after - row["score"]), cleared=won and last)))
    return {
        "game": NAME, "seed": seed, "policy": label if endpoint else "random", "steps": steps,
        "result": {"score": score, "won": won, "died": any(s["events"]["caught"] for s in steps),
                   "turns": len(steps)},
        "trace_path": str(trace_path) if trace_path.is_file() else None,
    }


def play_episodes(seeds, endpoint=None, label="scout", max_turns=None, trace_dir="data/coach/traces/pacman"):
    """Play one game per seed. Without an endpoint the scout plays randomly."""
    trace_dir = Path(trace_dir).resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, endpoint, label, max_turns or 300, str(trace_dir)) for seed in seeds]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(_play_one, jobs))


def digest_extras(episode):
    steps = episode["steps"]
    actions = [s["action"] for s in steps]
    recent_ghost = [s["ghost_dist"] for s in steps[-5:] if s.get("ghost_dist") is not None]
    return {
        "pellets_eaten": sum(s["events"]["pellets"] for s in steps),
        "ghosts_eaten": sum(s["events"]["ghosts_eaten"] for s in steps),
        "caught_at_turn": next((s["t"] for s in steps if s["events"]["caught"]), None),
        "closest_ghost_last_5_moves": min(recent_ghost) if recent_ghost else None,
        "stop_moves": actions.count("STOP"),
        "reversals": sum(1 for a, b in zip(actions, actions[1:]) if OPPOSITE.get(a) == b),
    }


def episode_metrics(episodes):
    games = len(episodes)
    if not games:
        raise ValueError("No episodes to score")
    moves = sum(len(e["steps"]) for e in episodes)
    return {
        "avg_score": sum(e["result"]["score"] for e in episodes) / games,
        "win_rate": sum(e["result"]["won"] for e in episodes) / games,
        "illegal_move_rate": sum(s["illegal_output"] for e in episodes for s in e["steps"]) / max(moves, 1),
        "pellets_per_game": sum(s["events"]["pellets"] for e in episodes for s in e["steps"]) / games,
        "death_rate": sum(e["result"]["died"] for e in episodes) / games,
        "avg_survival_moves": moves / games,
        "ghosts_eaten_per_game": sum(s["events"]["ghosts_eaten"] for e in episodes for s in e["steps"]) / games,
    }


def metrics_from_summary(summary):
    """Eval metrics from an eval.py results file (results/<label>.json)."""
    games = summary["games"]
    return {
        "avg_score": summary["avg_score"],
        "win_rate": summary["win_rate"],
        "illegal_move_rate": summary["illegal_move_rate"],
        "pellets_per_game": summary["pellets"] / games,
        "death_rate": summary["deaths"] / games,
        "avg_survival_moves": summary["avg_survival_moves"],
        "ghosts_eaten_per_game": summary["ghosts_eaten"] / games,
    }
