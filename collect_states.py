"""Collect RL training states: run rollout games with CollectAgent, dedupe
prompts, cap the set, write data/rl_states.jsonl.
Usage: python collect_states.py [n_games] [max_states] [out_name] [seed0]
"""

import concurrent.futures
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(ROOT, ".venv", "bin", "python")
BOARD = "classic-small"


def run_game(seed):
    log_path = os.path.join(ROOT, "data", f"rl_game_{seed}.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)
    env = dict(os.environ, PACBRAIN_RL_LOG=log_path)
    subprocess.run(
        [PY, "-m", "pacai.pacman", "--ui", "null", "--board", BOARD,
         "--pacman", "collect_agent.py:CollectAgent", "--seed", str(seed),
         "--max-turns", "300"],
        capture_output=True, text=True, env=env, cwd=ROOT, timeout=600)
    rows = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            rows = [json.loads(line) for line in f]
        os.remove(log_path)
    return rows


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    max_states = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    out_name = sys.argv[3] if len(sys.argv) > 3 else "rl_states.jsonl"
    seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)

    seen, states = set(), []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for rows in pool.map(run_game, range(seed0, seed0 + n_games)):
            for r in rows:
                if r["prompt"] not in seen:
                    seen.add(r["prompt"])
                    states.append(r)

    states = states[:max_states]
    out = os.path.join(ROOT, "data", out_name)
    with open(out, "w") as f:
        for s in states:
            f.write(json.dumps(s) + "\n")
    print(f"games={n_games} unique_states={len(states)} -> {out}")


if __name__ == "__main__":
    main()
