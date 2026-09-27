"""Recompute a report from saved per-game results; does not run inference."""

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
    results = [load(label) for label in ("base", "tuned", "tuned_t04")]
    rows = []
    for r in results:
        if r:
            measured = summarize(r)
            rows.append(
                f"| {r['label']} | {measured['games']} | {measured['avg_score']:.1f} "
                f"| {measured['win_rate']:.0%} | {measured['illegal_move_rate']:.1%} |")

    lines = [
        "# PacBrain: recorded evaluation results",
        "",
        "Metrics are recomputed from checked-in per-game observations. Generating "
        "this report does not rerun inference.",
        "",
        "| saved run | games | avg score | win rate | invalid-response / error rate |",
        "|---|---|---|---|---|",
        *rows,
        "",
        "The original `illegal` counter includes unparsable or unavailable model "
        "responses and API exceptions before a legal fallback action. It is not "
        "a count of illegal actions executed by the engine.",
        "",
        "The three legacy artifacts share seeds 2000–2009. They do not record "
        "a checkpoint hash, decoding configuration, package versions or training "
        "duration. These saved outcomes do not establish a controlled before/after "
        "comparison, an eight-minute training time, or a benefit from memory. "
        "The tuned_t04 filename alone does not verify temperature.",
        "",
    ]
    for label, title in (("base", "Before (base model)"),
                         ("tuned", "Tuned run"), ("tuned_t04", "Additional tuned run")):
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
