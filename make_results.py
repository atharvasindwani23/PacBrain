"""Build results/results.md from results/base.json and results/tuned.json."""

import json
import os

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


def main():
    base, tuned = load("base"), load("tuned")
    rows = []
    for r in (base, tuned):
        if r:
            rows.append(
                f"| {r['label']} | {r['games']} | {r['avg_score']:.0f} "
                f"| {r['win_rate']:.0%} | {r['illegal_move_rate']:.1%} |")

    lines = [
        "# PacBrain: fine-tuned DeepSeek plays Pacman",
        "",
        "Same 1.5B DeepSeek model, before and after an 8-minute LoRA fine-tune",
        "on 23k expert moves. Identical prompts, seeds, and board.",
        "",
        "| model | games | avg score | win rate | illegal moves |",
        "|---|---|---|---|---|",
        *rows,
        "",
    ]
    for label, title in (("base", "Before (base model)"),
                         ("tuned", "After (fine-tuned)")):
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
