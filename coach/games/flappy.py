"""Flappy Bird as a coach game: a small deterministic clone that runs in process.

The world is 100 x 100 abstract units with y increasing downward; one step is
one frame. It exists so the same scout -> notes -> reward design loop can be
pointed at a second game without touching the coach itself.
"""

import random
import re
from dataclasses import dataclass

from coach.llm import CompletionPolicy

NAME = "flappy"
ACTIONS = ("FLAP", "WAIT")
DESCRIPTION = (
    "Flappy Bird. The bird falls under gravity; FLAP gives it an upward kick and WAIT does "
    "nothing. Pipes scroll left, each with a vertical gap. Passing a pipe scores 1. Touching a "
    "pipe, the floor or the ceiling ends the game. The model sees an ASCII screen "
    "(B bird, | pipe) and must answer with exactly one move word: flap or wait."
)

HEIGHT = 100.0
WIDTH = 100.0
BIRD_X = 20.0
GRAVITY = 0.55
FLAP_VELOCITY = -4.8
MAX_FALL = 6.0
PIPE_SPEED = 1.6
PIPE_WIDTH = 9.0
PIPE_SPACING = 42.0
GAP = 30.0
EDGE = 8.0
MAX_FRAMES = 1500
SCREEN_COLS, SCREEN_ROWS = 24, 12
RANDOM_FLAP_PROBABILITY = 0.12

REWARD_PRIMITIVES = {
    "alive": {"description": "1 if the bird is still flying after the move.", "range": [0, 1]},
    "pipe_passed": {"description": "Pipes cleared by this move (the engine's score change).",
                    "range": [0, 1]},
    "crash": {"description": "1 if this move ends the game.", "range": [0, 1]},
    "gap_alignment": {"description": "Minus the bird's distance from the next gap's centre, "
                                     "as a fraction of the screen height (0 is perfectly centred).",
                      "range": [-1, 0]},
    "edge_danger": {"description": "1 if the bird ends within 8 units of the floor or ceiling.",
                    "range": [0, 1]},
}

EVAL_METRICS = {
    "avg_pipes": {"description": "Mean pipes passed per game.", "direction": "max"},
    "crash_rate": {"description": "Share of games that ended in a crash rather than the frame limit.",
                   "direction": "min"},
    "avg_survival_frames": {"description": "Mean frames survived per game.", "direction": "max"},
    "illegal_move_rate": {"description": "Share of model outputs that were not flap or wait.",
                          "direction": "min"},
}

PROMPT_HEADER = ("You are the bird in Flappy Bird. Screen: B you, | pipe. Fly through the gaps "
                 "without touching a pipe, the floor or the ceiling. Reply with exactly one move word.\n")


@dataclass(frozen=True)
class State:
    y: float
    vy: float
    pipes: tuple  # ((x, gap_centre, scored), ...)
    frame: int
    passed: int
    alive: bool
    seed: int
    next_index: int


def _gap_centre(seed, index):
    return random.Random(seed * 1_000_003 + index).uniform(25.0, 75.0)


def reset(seed):
    pipes = tuple((WIDTH + i * PIPE_SPACING, _gap_centre(seed, i), False) for i in range(3))
    return State(HEIGHT / 2, 0.0, pipes, 0, 0, True, seed, 3)


def _hits_pipe(x, gap_y, y):
    return x <= BIRD_X <= x + PIPE_WIDTH and abs(y - gap_y) > GAP / 2


def step(state, action):
    if not state.alive:
        raise ValueError("Cannot step a finished Flappy Bird game")
    if action not in ACTIONS:
        raise ValueError(f"Unknown Flappy Bird action {action!r}")
    vy = FLAP_VELOCITY if action == "FLAP" else min(MAX_FALL, state.vy + GRAVITY)
    y = state.y + vy
    pipes, passed, next_index = [], state.passed, state.next_index
    for x, gap_y, scored in state.pipes:
        x -= PIPE_SPEED
        if not scored and x + PIPE_WIDTH < BIRD_X:
            scored = True
            passed += 1
        if x + PIPE_WIDTH > 0:
            pipes.append((x, gap_y, scored))
    while len(pipes) < 3:
        last_x = pipes[-1][0] if pipes else WIDTH
        pipes.append((last_x + PIPE_SPACING, _gap_centre(state.seed, next_index), False))
        next_index += 1
    alive = 0.0 < y < HEIGHT and not any(_hits_pipe(x, gap_y, y) for x, gap_y, _ in pipes)
    return State(y, vy, tuple(pipes), state.frame + 1, passed, alive, state.seed, next_index)


def next_gap(state):
    """Centre of the first gap the bird has not yet flown past."""
    return next(gap_y for x, gap_y, _ in state.pipes if x + PIPE_WIDTH >= BIRD_X)


def observe(state):
    grid = [[" "] * SCREEN_COLS for _ in range(SCREEN_ROWS)]
    for x, gap_y, _ in state.pipes:
        col = int(x / WIDTH * SCREEN_COLS)
        if 0 <= col < SCREEN_COLS:
            for row in range(SCREEN_ROWS):
                centre = (row + 0.5) / SCREEN_ROWS * HEIGHT
                if abs(centre - gap_y) > GAP / 2:
                    grid[row][col] = "|"
    bird_row = min(SCREEN_ROWS - 1, max(0, int(state.y / HEIGHT * SCREEN_ROWS)))
    grid[bird_row][int(BIRD_X / WIDTH * SCREEN_COLS)] = "B"
    screen = "\n".join("".join(row) for row in grid)
    return PROMPT_HEADER + screen + "\nLegal moves: flap, wait\nMove:"


def parse_action(text):
    if not isinstance(text, str):
        return None
    match = re.fullmatch(r"\s*(flap|wait)[.!]?\s*", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def legal_actions(state):
    return list(ACTIONS)


def primitive_values(state, action):
    succ = step(state, action)
    return {
        "alive": 1.0 if succ.alive else 0.0,
        "pipe_passed": float(succ.passed - state.passed),
        "crash": 0.0 if succ.alive else 1.0,
        "gap_alignment": -abs(succ.y - next_gap(succ)) / HEIGHT,
        "edge_danger": 1.0 if succ.y < EDGE or succ.y > HEIGHT - EDGE else 0.0,
    }


def _choose(state, policy, rng):
    """Return (action, source, illegal_output) for one decision."""
    if policy is None:
        return ("FLAP" if rng.random() < RANDOM_FLAP_PROBABILITY else "WAIT"), "random", False
    action = parse_action(policy(observe(state)))
    if action is None:
        return ("FLAP" if rng.random() < RANDOM_FLAP_PROBABILITY else "WAIT"), "fallback", True
    return action, "policy", False


def play_episodes(seeds, endpoint=None, label="scout", max_turns=None, trace_dir=None, policy=None):
    """Play one game per seed with a model endpoint, a callable policy, or the random scout."""
    if endpoint and policy:
        raise ValueError("Pass either a model endpoint or a policy callable, not both")
    policy = policy or (CompletionPolicy(endpoint) if endpoint else None)
    episodes = []
    for seed in seeds:
        rng = random.Random(seed)
        state = reset(seed)
        steps = []
        while state.alive and state.frame < (max_turns or MAX_FRAMES):
            action, source, illegal = _choose(state, policy, rng)
            succ = step(state, action)
            steps.append({"t": state.frame, "action": action, "source": source,
                          "illegal_output": illegal, "score": state.passed,
                          "y": round(state.y, 1), "gap_offset": round(state.y - next_gap(state), 1),
                          "events": {"pipe_passed": succ.passed - state.passed, "crashed": not succ.alive}})
            state = succ
        episodes.append({
            "game": NAME, "seed": seed, "policy": label if policy else "random", "steps": steps,
            "result": {"score": state.passed, "won": False, "died": not state.alive, "turns": state.frame,
                       "final_y": round(state.y, 1)},
            "trace_path": None,
        })
    return episodes


def digest_extras(episode):
    steps = episode["steps"]
    final_y = episode["result"]["final_y"]
    if not episode["result"]["died"]:
        crash = "none"
    elif final_y <= 0:
        crash = "ceiling"
    elif final_y >= HEIGHT:
        crash = "floor"
    else:
        crash = "pipe_above_gap" if steps and steps[-1]["gap_offset"] < 0 else "pipe_below_gap"
    offsets = [abs(s["gap_offset"]) for s in steps]
    return {
        "pipes_passed": episode["result"]["score"],
        "crash_type": crash,
        "flap_rate": round(sum(s["action"] == "FLAP" for s in steps) / len(steps), 3) if steps else 0.0,
        "mean_distance_from_gap_centre": round(sum(offsets) / len(offsets), 1) if offsets else None,
    }


def episode_metrics(episodes):
    games = len(episodes)
    if not games:
        raise ValueError("No episodes to score")
    decisions = sum(len(e["steps"]) for e in episodes)
    return {
        "avg_pipes": sum(e["result"]["score"] for e in episodes) / games,
        "crash_rate": sum(e["result"]["died"] for e in episodes) / games,
        "avg_survival_frames": sum(e["result"]["turns"] for e in episodes) / games,
        "illegal_move_rate": sum(s["illegal_output"] for e in episodes for s in e["steps"]) / max(decisions, 1),
    }


def training_states(seeds, reward_fn, max_states=3000):
    """GRPO rows ({prompt, rewards}) from states the random scout visits."""
    seen, rows = set(), []
    for seed in seeds:
        rng = random.Random(seed)
        state = reset(seed)
        while state.alive and state.frame < MAX_FRAMES and len(rows) < max_states:
            prompt = observe(state)
            if prompt not in seen:
                seen.add(prompt)
                rows.append({"prompt": prompt, "rewards": reward_fn(state)})
            action, _, _ = _choose(state, None, rng)
            state = step(state, action)
        if len(rows) >= max_states:
            break
    return rows
