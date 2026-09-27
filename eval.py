"""Evaluate an LLM endpoint playing Pacman.
Usage: python eval.py --endpoint URL --label base [--games 10] [--gifs 2]
Writes results/<label>.json and results/<label>_seed<k>.gif
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(ROOT, ".venv", "bin", "python")
BOARD = "classic-small"
SEEDS = list(range(2000, 2100))  # fixed eval seeds, disjoint from training


def run_game(args):
    seed, endpoint, label, gif = args
    stats_path = os.path.join(ROOT, "results", f"{label}_stats_{seed}.jsonl")
    if os.path.exists(stats_path):
        os.remove(stats_path)
    cmd = [PY, "-m", "pacai.pacman", "--ui", "null", "--board", BOARD,
           "--pacman", "llm_agent.py:LLMAgent", "--seed", str(seed),
           "--max-turns", "600"]
    if gif:
        cmd += ["--animation-path",
                os.path.join(ROOT, "results", f"{label}_seed{seed}.gif")]
    env = dict(os.environ, PACBRAIN_ENDPOINT=endpoint, PACBRAIN_STATS=stats_path)
    out = subprocess.run(cmd, capture_output=True, text=True, env=env,
                         cwd=ROOT, timeout=1800)
    text = out.stdout + out.stderr
    score = int(m.group(1)) if (m := re.search(r"Scores:\s+(-?\d+)", text)) else None
    win = bool(re.search(r"Record:\s+Win", text))
    moves, illegal = 0, 0
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            for line in f:
                moves += 1
                illegal += json.loads(line)["illegal"]
        os.remove(stats_path)
    return {"seed": seed, "score": score, "win": win,
            "moves": moves, "illegal": illegal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--gifs", type=int, default=2)
    a = ap.parse_args()

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    jobs = [(seed, a.endpoint, a.label, i < a.gifs)
            for i, seed in enumerate(SEEDS[:a.games])]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        games = list(pool.map(run_game, jobs))

    scores = [g["score"] for g in games if g["score"] is not None]
    moves = sum(g["moves"] for g in games)
    illegal = sum(g["illegal"] for g in games)
    summary = {
        "label": a.label, "games": len(games),
        "avg_score": sum(scores) / max(len(scores), 1),
        "win_rate": sum(g["win"] for g in games) / max(len(games), 1),
        "illegal_move_rate": illegal / max(moves, 1),
        "per_game": games,
    }
    with open(os.path.join(ROOT, "results", f"{a.label}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_game"},
                     indent=2))


if __name__ == "__main__":
    main()
