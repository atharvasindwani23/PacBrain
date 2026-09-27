"""Evaluate an endpoint with optional guarded procedural memory.

Example: python eval.py --endpoint URL --label trial --memory none
Memory-enabled runs require a separate label for a fair matched-seed comparison.
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
BOARD = "classic-small"
SEEDS = list(range(2000, 2100))


def decode_events(timeline, final_score, win):
    """Decode game events from the per-move score timeline.
    Per pacman move: -1 time step, +10 per pellet, +200 per ghost eaten,
    +500 board clear, -500 caught."""
    out = {"pellets": 0, "ghosts_eaten": 0, "died": 0}
    if not timeline:
        return out
    diffs = [b - a for a, b in zip(timeline, timeline[1:])]
    if final_score is not None:
        diffs.append(final_score - timeline[-1])
    for d in diffs:
        if d <= -400:
            out["died"] = 1
            d += 500
        if win and d >= 400:
            d -= 500
        if d >= 190:
            out["ghosts_eaten"] += (d + 1) // 200
            d = (d + 1) % 200 - 1
        if d > 0:
            out["pellets"] += (d + 1) // 10
    return out


def run_game(args):
    seed, endpoint, label, gif, provider, memory_dir, board, trace_dir = args
    stats_path = os.path.join(ROOT, "results", f"{label}_stats_{seed}.jsonl")
    trace_path = os.path.join(trace_dir, f"{label}_seed{seed}.json")
    for stale in (stats_path, trace_path):
        if os.path.exists(stale):
            os.remove(stale)
    cmd = [PY, "-m", "pacai.pacman", "--ui", "null", "--board", board,
           "--pacman", "llm_agent.py:LLMAgent", "--seed", str(seed), "--max-turns", "600"]
    gif_path = Path(ROOT, "results", f"{label}_seed{seed}.gif")
    if gif:
        cmd += ["--animation-path", str(gif_path)]
    env = dict(os.environ, PACBRAIN_ENDPOINT=endpoint, PACBRAIN_STATS=stats_path,
               PACBRAIN_MEMORY=provider, PACBRAIN_MEMORY_DIR=memory_dir,
               PACBRAIN_TRACE=trace_path, PACBRAIN_BOARD=board, PACBRAIN_SEED=str(seed))
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT, timeout=1800)
    except subprocess.TimeoutExpired:
        return {"seed": seed, "score": None, "win": False, "moves": 0, "illegal": 0,
                "policy_calls": 0, "procedure_steps": 0, "fallback_steps": 0,
                "memory_failures": 0, "error": "game_timeout", "trace": None, "gif": None}
    text = out.stdout + out.stderr
    score = int(match.group(1)) if (match := re.search(r"Scores:\s+(-?\d+)", text)) else None
    win = bool(re.search(r"Record:\s+Win", text))
    rows = []
    if os.path.exists(stats_path):
        with open(stats_path) as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        os.remove(stats_path)
    recorded = None
    if os.path.exists(trace_path):
        with open(trace_path) as stream:
            recorded = json.load(stream)
    procedure_steps = (recorded["result"]["procedure_steps"] if recorded else 0)
    gif_ready = gif and gif_path.is_file() and gif_path.stat().st_size > 0
    timeline = [row["score"] for row in rows if "score" in row]
    events = decode_events(timeline, score, win)
    return {"seed": seed, "score": score, "win": win,
            "moves": len(rows), "illegal": sum(row["illegal"] for row in rows),
            **events,
            "policy_calls": sum(row.get("policy_called", True) for row in rows),
            "procedure_steps": procedure_steps,
            "fallback_steps": sum(row.get("source") == "fallback" for row in rows),
            "memory_failures": sum(row.get("fallback_reason") == "memory_unavailable" for row in rows),
            "error": ("game_process_failed" if out.returncode else
                      "agent_crashed" if re.search(r"Agent \d+ crashed", text) else
                      "missing_game_result" if score is None else
                      "missing_trace" if recorded is None else
                      "missing_gif" if gif and not gif_ready else None),
            "trace": trace_path if recorded else None,
            "gif": str(gif_path) if gif_ready else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--gifs", type=int, default=2)
    parser.add_argument("--board", default=BOARD)
    parser.add_argument("--memory", choices=["none", "local", "gbrain"], default="none")
    parser.add_argument("--memory-dir", default="data/memory")
    parser.add_argument("--trace-dir", help="Defaults to results/<label>_traces")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace this label's existing result, GIFs, stats, and traces")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("--label must contain only letters, numbers, underscores, or hyphens")
    if not 1 <= args.games <= len(SEEDS) or args.gifs < 0:
        parser.error("--games must be 1–100 and --gifs must be nonnegative")
    # Use explicit paths in subprocesses; caller's working directory may differ.
    memory_dir = str(Path(args.memory_dir).resolve())
    trace_dir = str(Path(args.trace_dir or os.path.join(ROOT, "results", args.label + "_traces")).resolve())
    results_dir = Path(ROOT, "results")
    summary_path = results_dir / (args.label + ".json")
    artifacts = {summary_path}
    artifacts.update(results_dir.glob(args.label + "_seed*.gif"))
    artifacts.update(results_dir.glob(args.label + "_stats_*.jsonl"))
    artifacts.update(Path(trace_dir).glob(args.label + "_seed*.json"))
    existing = sorted(path for path in artifacts if path.exists() or path.is_symlink())
    if existing and not args.overwrite:
        parser.error("Output already exists for this label; select a new label or explicitly pass --overwrite")
    if any(path.is_dir() for path in existing):
        parser.error("An output artifact is a directory; select a different label or trace directory")
    # Explicit overwrite clears old GIFs even when the new run requests fewer
    # animations, so failed or shorter reruns cannot display stale evidence.
    for path in existing:
        path.unlink()
    Path(trace_dir).mkdir(parents=True, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    jobs = [(seed, args.endpoint, args.label, i < args.gifs, args.memory, memory_dir, args.board, trace_dir)
            for i, seed in enumerate(SEEDS[:args.games])]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        games = list(pool.map(run_game, jobs))
    completed = [game for game in games if game["error"] is None]
    scores = [game["score"] for game in completed if game["score"] is not None]
    moves = sum(game["moves"] for game in games)
    summary = {"label": args.label, "memory": args.memory, "board": args.board, "games": len(games),
               "completed_games": len(completed),
               "avg_score": sum(scores) / len(scores) if scores else None,
               "win_rate": sum(game["win"] for game in completed) / len(completed) if completed else None,
               "illegal_move_rate": sum(game["illegal"] for game in games) / max(moves, 1),
               "pellets": sum(game.get("pellets", 0) for game in games),
               "ghosts_eaten": sum(game.get("ghosts_eaten", 0) for game in games),
               "deaths": sum(game.get("died", 0) for game in games),
               "avg_survival_moves": moves / max(len(games), 1),
               "policy_calls": sum(game["policy_calls"] for game in games),
               "procedure_steps": sum(game["procedure_steps"] for game in games),
               "fallback_steps": sum(game["fallback_steps"] for game in games),
               "memory_failures": sum(game["memory_failures"] for game in games),
               "failed_games": sum(game["error"] is not None for game in games),
               "per_game": games}
    with open(os.path.join(ROOT, "results", args.label + ".json"), "w") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != "per_game"}, indent=2))
    return 1 if summary["failed_games"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
