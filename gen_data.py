"""Collect winning expert pairs and separate canonical traces with observed scores.

Defaults write under data/generated and traces/generated, preserving the original
dataset. Example: python gen_data.py 10 --workers 2
"""

import argparse
import concurrent.futures
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def parse_summary(text):
    score = re.search(r"Scores:\s+(-?\d+(?:\.\d+)?)", text)
    outcome = re.search(r"Record:\s+(Win|Loss)\b", text)
    # PacAI may exit successfully even if an agent callback crashed.
    if not score or not outcome or re.search(r"Agent \d+ crashed", text):
        raise RuntimeError("PacAI did not produce a successful, complete game summary")
    return float(score.group(1)), outcome.group(1) == "Win"


def run_game(seed, *, board="classic-small", trace_dir=None, max_turns=600):
    trace_dir = Path(trace_dir or ROOT / "traces/generated").resolve()
    canonical = trace_dir / "canonical" / f"expert_{seed}.json"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pacbrain-expert-") as scratch:
        log_path = Path(scratch) / "pairs.jsonl"
        env = dict(os.environ, PACBRAIN_LOG=str(log_path),
                   PACBRAIN_TRACE=str(canonical), PACBRAIN_BOARD=board,
                   PACBRAIN_SEED=str(seed))
        output = subprocess.run(
            [sys.executable, "-m", "pacai.pacman", "--ui", "null", "--board", board,
             "--pacman", "expert.py:ExpertAgent", "--seed", str(seed),
             "--max-turns", str(max_turns)],
            capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
        if output.returncode:
            raise RuntimeError(f"PacAI failed for seed {seed} (exit {output.returncode})")
        score, win = parse_summary(output.stdout + output.stderr)
        if not log_path.is_file() or not canonical.is_file():
            raise RuntimeError(f"Missing expert log or canonical trace for seed {seed}")
        pairs = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        recorded = json.loads(canonical.read_text())
        if recorded.get("result", {}).get("score") != score or len(recorded.get("frames", [])) != len(pairs):
            raise RuntimeError(f"Incomplete or inconsistent expert trace for seed {seed}")
        if any(frame.get("action_ok") is not True for frame in recorded["frames"]):
            raise RuntimeError(f"An expert action was not confirmed by the engine for seed {seed}")
    trace = {"game": "pacman", "board": board, "seed": seed, "agent": "expert",
             "score": score, "win": win, "steps": pairs,
             "canonical_trace": str(canonical.relative_to(trace_dir))}
    (trace_dir / f"expert_{seed}.json").write_text(json.dumps(trace) + "\n")
    return seed, score, win, pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_games", nargs="?", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--board", default="classic-small")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=600,
                        help="Engine turns, including ghost turns")
    parser.add_argument("--output", type=Path, default=ROOT / "data/generated/train.jsonl")
    parser.add_argument("--trace-dir", type=Path, default=ROOT / "traces/generated")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if min(args.n_games, args.workers, args.max_turns) <= 0:
        parser.error("n_games, workers and max-turns must be positive")
    seeds = list(range(args.start_seed, args.start_seed + args.n_games))
    targets = [args.output, args.output.with_suffix(".manifest.json")]
    targets += [args.trace_dir / f"expert_{seed}.json" for seed in seeds]
    targets += [args.trace_dir / "canonical" / f"expert_{seed}.json" for seed in seeds]
    if not args.overwrite and any(path.exists() for path in targets):
        parser.error("Output already exists; select new paths or explicitly pass --overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    worker = functools.partial(run_game, board=args.board, trace_dir=args.trace_dir,
                               max_turns=args.max_turns)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        games = list(pool.map(worker, seeds))
    all_pairs = [pair for _, _, win, pairs in games if win for pair in pairs]
    seen, unique = set(), []
    for pair in all_pairs:
        if pair["prompt"] not in seen:
            seen.add(pair["prompt"])
            unique.append(pair)
    payload = "".join(json.dumps(pair) + "\n" for pair in unique)
    args.output.write_text(payload)
    manifest = {
        "board": args.board, "seeds": seeds, "max_engine_turns": args.max_turns,
        "games": len(games), "wins": sum(win for _, _, win, _ in games),
        "avg_score_all_games": sum(score for _, score, _, _ in games) / len(games),
        "selection": "winning games only; first completion per identical prompt",
        "winning_pairs": len(all_pairs), "unique_pairs": len(unique),
        "dataset_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "edq_pacai_version": importlib.metadata.version("edq-pacai"),
        "python_version": sys.version.split()[0],
        "canonical_trace_directory": str((args.trace_dir / "canonical").resolve()),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"games={len(games)} wins={manifest['wins']} avg_score={manifest['avg_score_all_games']:.1f}")
    print(f"winning_pairs={len(all_pairs)} unique={len(unique)} -> {args.output}")
    if not unique:
        print("No winning examples were generated; this dataset is not ready for training.")


if __name__ == "__main__":
    main()
