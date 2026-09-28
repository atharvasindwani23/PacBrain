"""Reward specs: a weighted sum of a game's reward primitives, scaled and clipped.

compile_reward turns a spec into the per-action reward table the GRPO trainer
consumes ({action: reward} for every legal action in a state), the same shape
rewards.action_rewards produces. coach/specs/pacman_handwritten.json expresses
that hand-written function as a spec, for comparison with designed ones.
"""

import json
import math
import os
from pathlib import Path

from .games import load_game

MAX_TERMS = 10
MAX_ABS_WEIGHT = 1000.0


def _number(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if not low <= value <= high:
        raise ValueError(f"{label} must be between {low} and {high}, got {value}")
    return float(value)


def validate_reward_spec(reward, game):
    if not isinstance(reward, dict):
        raise ValueError("Reward spec must be an object")
    terms = reward.get("terms")
    if not isinstance(terms, list) or not 1 <= len(terms) <= MAX_TERMS:
        raise ValueError(f"Reward spec needs 1-{MAX_TERMS} terms")
    clean, seen = [], set()
    for term in terms:
        name = term.get("name") if isinstance(term, dict) else None
        if name not in game.REWARD_PRIMITIVES:
            raise ValueError(f"Unknown reward primitive {name!r} for {game.NAME}; "
                             f"allowed: {', '.join(game.REWARD_PRIMITIVES)}")
        if name in seen:
            raise ValueError(f"Reward primitive {name!r} appears twice")
        seen.add(name)
        weight = _number(term.get("weight"), f"weight of {name}", -MAX_ABS_WEIGHT, MAX_ABS_WEIGHT)
        clean.append({"name": name, "weight": weight, "why": str(term.get("why", ""))[:500]})
    illegal = _number(reward.get("illegal_output"), "illegal_output", -100.0, 0.0)
    if illegal == 0.0:
        raise ValueError("illegal_output must be negative, so unparseable output is never free")
    return {"terms": clean,
            "scale": _number(reward.get("scale"), "scale", 0.1, 1000.0),
            "clip": _number(reward.get("clip"), "clip", 0.1, 100.0),
            "illegal_output": illegal}


def compile_reward(spec, game):
    reward = validate_reward_spec(spec, game)

    def action_rewards(state):
        table = {}
        for action in game.legal_actions(state):
            values = game.primitive_values(state, action)
            raw = sum(term["weight"] * values[term["name"]] for term in reward["terms"])
            table[str(action)] = max(-reward["clip"], min(reward["clip"], raw / reward["scale"]))
        return table

    return action_rewards


def load_spec(path, game_name):
    spec = json.loads(Path(path).read_text())
    if spec.get("game") != game_name:
        raise ValueError(f"{path} is a reward spec for {spec.get('game')!r}, not {game_name!r}")
    return spec


def spec_rewards_from_env(game_name):
    """Compiled reward from COACH_REWARD_SPEC, or None when that variable is unset."""
    path = os.getenv("COACH_REWARD_SPEC")
    if not path:
        return None
    return compile_reward(load_spec(path, game_name), load_game(game_name))
