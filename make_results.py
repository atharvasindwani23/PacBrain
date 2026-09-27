"""Recompute results/results.md from saved per-game results (base vs rl);
does not run inference."""

import json
import os
import math

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(label):
    path = os.path.join(ROOT, "results", f"{label}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def gifs(label):
    d = os.path.join(ROOT, "results")
    return sorted(f for f in os.listdir(d)
                  if f.startswith(f"{label}_seed") and f.endswith(".gif"))


def summarize(result):
    games = result.get("per_game")
    if not isinstance(games, list) or not games:
        raise ValueError("A result needs nonempty per_game observations")
    seeds = set()
    for game in games:
        if not isinstance(game.get("score"), (int, float)) or not math.isfinite(game["score"]):
            raise ValueError("Each game needs a finite observed score")
        if not isinstance(game.get("win"), bool):
            raise ValueError("Each game needs an observed win boolean")
        if game.get("seed") in seeds:
            raise ValueError("Duplicate evaluation seed")
        seeds.add(game.get("seed"))
        if any(type(game.get(key)) is not int or game[key] < 0 for key in ("moves", "illegal")):
            raise ValueError("Each game needs nonnegative moves and illegal counts")
        if game["illegal"] > game["moves"]:
            raise ValueError("Invalid response count exceeds moves")
    moves = sum(game["moves"] for game in games)
    summary = {
        "games": len(games), "avg_score": sum(game["score"] for game in games) / len(games),
        "win_rate": sum(game["win"] for game in games) / len(games),
        "illegal_move_rate": sum(game["illegal"] for game in games) / moves if moves else 0.0,
    }
    for key, value in summary.items():
        if key in result and not math.isclose(result[key], value, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f"Saved {key} disagrees with per-game observations")
    return summary


def main():
    base, rl = load("base"), load("rl")
    rows = []
    for r in (base, rl):
        if r:
            measured = summarize(r)
            rows.append(
                f"| {r['label']} | {measured['games']} | {measured['avg_score']:.1f} "
                f"| {measured['win_rate']:.0%} | {measured['illegal_move_rate']:.1%} "
                f"| {r.get('pellets', '?')} | {r.get('ghosts_eaten', '?')} "
                f"| {r.get('deaths', '?')} |")

    lines = [
        "# PacBrain: DeepSeek learns Pacman from rewards alone",
        "",
        "Same 1.5B DeepSeek model before and after GRPO reinforcement",
        "learning — no expert imitation, only the game's own score as reward.",
        "Identical prompts, seeds, and board. Metrics are recomputed from",
        "checked-in per-game observations; this report does not rerun inference.",
        "",
        "| model | games | avg score | win rate | invalid-response rate "
        "| pellets | ghosts eaten | deaths |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        "The `illegal` counter includes unparsable or unavailable model "
        "responses and API exceptions before a legal fallback action.",
        "",
        "Training checkpoints (reward per step): `results/training_curve.json`.",
        "",
    ]
    for label, title in (("base", "Before (base model)"),
                         ("rl", "After (reward-trained, GRPO)")):
        gs = gifs(label)
        if gs:
            lines.append(f"## {title}")
            lines.extend(f"![{g}]({g})" for g in gs)
            lines.append("")

    with open(os.path.join(ROOT, "results", "results.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
