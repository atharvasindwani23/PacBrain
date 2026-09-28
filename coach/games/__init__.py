"""Game registry. The coach knows nothing about game rules beyond what a game
module declares: its description, actions, reward primitives and eval metrics,
plus functions to play episodes and evaluate candidate actions."""

import importlib

GAMES = {"pacman": "coach.games.pacman", "flappy": "coach.games.flappy"}
REQUIRED = ("NAME", "DESCRIPTION", "ACTIONS", "REWARD_PRIMITIVES", "EVAL_METRICS",
            "legal_actions", "primitive_values", "play_episodes", "digest_extras",
            "episode_metrics")


def load_game(name):
    if name not in GAMES:
        raise ValueError(f"Unknown game {name!r}; choose one of: {', '.join(sorted(GAMES))}")
    module = importlib.import_module(GAMES[name])
    missing = [attr for attr in REQUIRED if not hasattr(module, attr)]
    if missing:
        raise TypeError(f"{GAMES[name]} does not implement the game contract: missing {', '.join(missing)}")
    return module
