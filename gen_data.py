"""Run N expert games, collect (prompt, completion) pairs into data/train.jsonl
and per-game traces into traces/. Usage: python gen_data.py [n_games]
"""

import concurrent.futures
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(ROOT, ".venv", "bin", "python")
BOARD = "classic-small"


def run_game(seed):
    log_path = os.path.join(ROOT, "data", f"game_{seed}.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)
    env = dict(os.environ, PACBRAIN_LOG=log_path)
    out = subprocess.run(
        [PY, "-m", "pacai.pacman", "--ui", "null", "--board", BOARD,
         "--pacman", "expert.py:ExpertAgent", "--seed", str(seed),
         "--max-turns", "600"],
        capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    text = out.stdout + out.stderr
    score = int(m.group(1)) if (m := re.search(r"Scores:\s+(-?\d+)", text)) else None
    win = bool(re.search(r"Record:\s+Win", text))
    pairs = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            pairs = [json.loads(line) for line in f]
        os.remove(log_path)
    trace = {"game": "pacman", "board": BOARD, "seed": seed, "agent": "expert",
             "score": score, "win": win, "steps": pairs}
    with open(os.path.join(ROOT, "traces", f"expert_{seed}.json"), "w") as f:
        json.dump(trace, f)
    return seed, score, win, pairs


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "traces"), exist_ok=True)

    all_pairs, wins, scores = [], 0, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for seed, score, win, pairs in pool.map(run_game, range(1000, 1000 + n_games)):
            if not win:
                continue  # imitate winning games only
            all_pairs.extend(pairs)
            wins += win
            if score is not None:
                scores.append(score)

    seen, unique = set(), []
    for p in all_pairs:
        if p["prompt"] not in seen:
            seen.add(p["prompt"])
            unique.append(p)

    with open(os.path.join(ROOT, "data", "train.jsonl"), "w") as f:
        for p in unique:
            f.write(json.dumps(p) + "\n")

    print(f"games={n_games} wins={wins} avg_score={sum(scores)/max(len(scores),1):.0f}")
    print(f"pairs={len(all_pairs)} unique={len(unique)} -> data/train.jsonl")


if __name__ == "__main__":
    main()
